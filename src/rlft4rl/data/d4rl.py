import os
import logging
import tyro
from dataclasses import dataclass
from typing import Dict, Any
import json

from rlft4rl.utils import setup_logger, set_seed_everywhere
from rlft4rl.prompts import ENV_DESC, INSTRUCTIONS

import minari


@dataclass
class Args:
    """Arguments for Minari dataset exploration."""

    dataset_id: str = "mujoco/halfcheetah/expert-v0"  # Minari dataset ID
    env_name: str = "HalfCheetah-v4"  # Environment name
    seed: int = 42  # Random seed
    n_episodes: int = 5  # Number of sampling iterations
    verbose: bool = True  # Print detailed information
    log_level: str = "INFO"  # Logging level
    system_prompt: bool = False  # Use system prompt
    chat_template: bool = False  # Use chat template


def obs_to_prompt(obs):
    # Convert obs list to a comma-separated string without brackets
    obs_string = ",".join([f"{val:.5f}" for val in obs])
    prompt = f"""<observation>{obs_string}</observation>"""
    return prompt


def action_to_prompt(action):
    # Convert obs list to a comma-separated string without brackets
    action_string = ",".join([f"{val:.5f}" for val in action])
    prompt = f"""<action>{action_string}</action>"""
    return prompt


def explore_minari_dataset(args: Args, logger: logging.Logger) -> Dict[str, Any]:
    """Explore a Minari dataset by sampling episodes."""
    # Create a filename for the training examples
    filename = (
        "data/sft/"
        + args.dataset_id.replace("/", "_")
        + f"_{args.n_episodes}_sp{int(args.system_prompt)}_ct{int(args.chat_template)}."
        + "json"
    )

    # make system prompt
    desc = ENV_DESC[args.env_name]

    system_prompt = f"""
    You are the controller for a {args.env_name} robot in a physics simulation.

    Environment Description:
    """
    system_prompt += desc
    system_prompt += INSTRUCTIONS

    dataset = minari.load_dataset(args.dataset_id, download=True)
    dataset.set_seed(seed=args.seed)

    episodes = dataset.sample_episodes(n_episodes=args.n_episodes)

    for i, ep in enumerate(episodes):
        n_steps = ep.observations.shape[0]
        for t in range(n_steps - 1):
            obs = ep.observations[t]
            act = ep.actions[t]
            # rew = ep.rewards[t]
            # term = ep.terminations[t]
            # trunc = ep.truncations[t]

            obs_promp = obs_to_prompt(obs)
            act_promp = action_to_prompt(act)

            if args.chat_template:
                # Construct a training example similar to the one in the user's example
                messages = []
                if args.system_prompt:
                    messages.append({"content": system_prompt, "role": "system"})
                messages.append({"content": obs_promp, "role": "user"})
                messages.append({"content": act_promp, "role": "assistant"})
                training_example = {"messages": messages}
            else:
                training_example = {"observation": obs_promp, "action": act_promp}

            # Write the training example to a JSON file
            with open(filename, "a") as f:
                json.dump(training_example, f)  # indent=2
                f.write("\n")  # Add newline after each example

            # If verbose, print the training example
            if i == 0 and t == 0:
                logger.info(f"Training example: {training_example}")
    return


def main(args: Args):
    set_seed_everywhere(seed=args.seed)

    # Setup logger
    logger, _, _ = setup_logger(
        logger_name="MINARI",
        log_level=args.log_level,
        log_dir="logs",
        env_id="data",
        exp_name=args.dataset_id.replace("/", "_"),
        create_ts_writer=False,
    )

    # Create a directory to store the training examples
    os.makedirs("data/sft", exist_ok=True)

    # Run exploration
    explore_minari_dataset(args, logger)

    # Log results
    logger.info(f"\n-> Converted dataset {args.dataset_id} successfully!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
