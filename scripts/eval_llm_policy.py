import os
import time
import numpy as np
import tyro
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from rlft4rl.utils import setup_logger, make_env, set_seed_everywhere
from rlft4rl.policies.llm_policy import LLMPolicy


@dataclass
class Args:
    """Arguments for LLM policy evaluation."""

    env_id: str = "CartPole-v1"  # Environment ID
    seed: int = 42  # Random seed
    episodes: int = 10  # Number of evaluation episodes
    capture_video: bool = False  # Whether to capture video
    verbose: bool = True  # Print detailed information during evaluation
    log_level: str = "INFO"  # Logging level
    model: str = "llama-3.3-70b-instruct"  # LLM model
    temperature: float = 0.3  # Temperature for sampling
    prediction_horizon: int = 200  # Prediction horizon
    logit_bias: float = 0.0  # Logit bias
    good_tokens: Optional[str] = None  # Good tokens for LLM
    api_url: str = "http://10.227.91.60:4000/v1"  # Playground For Europe, or localhost
    api_key: str = "sk-1234"  # API key
    freq_log_action: int = 250  # Frequency to log action
    n_examples: int = 3  # Number of examples for LLM policy system prompt
    system_prompt: bool = True  # Use system prompt
    use_vllm: bool = False
    tol_repeat_gen: int = 10  # Tolerance for repeated generations
    device: str = "cuda:0"
    discretized: bool = False  # Whether to use discretized actions


def evaluate_llm_policy(
    args: Args, logger: logging.Logger, ts_writer: SummaryWriter
) -> Dict[str, Any]:
    """Evaluate a LLM as policy in the given environment."""

    # Create directory for videos if needed
    if args.capture_video:
        os.makedirs("videos", exist_ok=True)

    # Create environment
    model_name = args.model.split("/final_model")[0].split("/")[0].split("--")[-1]
    run_name = f"{args.env_id}/{model_name}/{int(time.time())}__seed-{args.seed}"
    env_fn = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    env = env_fn()

    # Get environment info
    obs_space = env.observation_space
    act_space = env.action_space

    if args.verbose:
        logger.info(f"Environment: {args.env_id}")
        logger.info(f"Observation space: {obs_space}")
        logger.info(f"Action space: {act_space}")

    # Collect metrics across episodes
    episode_returns = []
    episode_lengths = []

    # Generate n examples for the LLM policy system prompt
    examples = {}
    for i in range(1, args.n_examples + 1):
        examples[i] = {"obs": env.reset()[0], "action": env.action_space.sample()}

    # instantiate llm_policy
    llm_policy = LLMPolicy(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature,
        prediction_horizon=args.prediction_horizon,
        logit_bias=args.logit_bias,
        good_tokens=[] if not args.good_tokens else args.good_tokens.split(","),
        env_name=args.env_id,
        n_action_dim=act_space.n if hasattr(act_space, "n") else act_space.shape[0],
        examples=examples if args.n_examples else None,
        system_prompt=args.system_prompt,
        use_vllm=args.use_vllm,
        tol_repeat_gen=args.tol_repeat_gen,
        device=args.device,
        discretized=args.discretized,
        discrete_actions=hasattr(act_space, "n"),
    )

    # Log the system prompt from LLM policy
    if args.verbose:
        logger.info("LLM Policy System Prompt:")
        logger.info(llm_policy.system_prompt)

    # Run evaluation episodes
    for ep in tqdm(range(args.episodes), desc="-> Evaluating LLM policy"):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False

        # Initialize a progress bar for this episode
        step_count = 0
        pbar = tqdm(desc="(-> Timesteps", leave=False)
        while not done:
            # LLM action
            action = llm_policy.act(obs, logger=logger)
            if step_count % args.freq_log_action == 0:
                logger.info(
                    f"[Ep {ep + 1}, Step {step_count}] obs: {[f'{e:.4f}' for e in obs]}"
                )
                if action.shape:
                    logger.info(
                        f"[Ep {ep + 1}, Step {step_count}] "
                        f"action: {[f'{e:.4f}' for e in action]}"
                    )
                else:
                    logger.info(f"[Ep {ep + 1}, Step {step_count}] action: {action}")

            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update step counter and progress bar
            step_count += 1
            pbar.update(1)

            # log action values
            if action.shape:
                for i in range(len(action)):
                    ts_writer.add_scalar(f"action/ep_{ep + 1}/dim_{i}", action[i], step_count)
            else:
                ts_writer.add_scalar(f"action/ep_{ep + 1}/value", action, step_count)

        # Close the progress bar for this episode
        pbar.close()

        if args.verbose:
            logger.info(
                f"[Ep {ep + 1}] "
                f"return={env.return_queue[-1]}, length={env.length_queue[-1]}"
            )

        # Record episode metrics
        episode_returns.append(env.return_queue[-1])
        episode_lengths.append(env.length_queue[-1])

        ts_writer.add_scalar("Return/Value", episode_returns[-1], ep + 1)
        ts_writer.add_scalar("Return/Length", episode_lengths[-1], ep + 1)

    # Calculate statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # Compile results
    results = {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "std_length": std_length,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }

    env.close()

    llm_policy.log(logger=logger)

    return results


def main(args: Args):
    # Set random seed
    set_seed_everywhere(seed=args.seed)

    # Setup logger
    model_name = args.model.split("/")[1]
    logger, _, ts_writer = setup_logger(
        logger_name="LLMPolicy",
        log_level=args.log_level,
        log_dir="logs",
        env_id=args.env_id,
        exp_name=f"{model_name}",
    )

    # Run evaluation
    results = evaluate_llm_policy(args, logger, ts_writer)

    # Log results
    logger.info(
        f"\n-> Results for {args.env_id} with llm policy "
        f"(model={args.model.split('/')[0]}) (seed={args.seed}):"
    )
    logger.info(f"Episodes: {args.episodes}")
    logger.info(
        f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
    )
    logger.info(
        f"Mean episode length: "
        f"{results['mean_length']:.2f} ± {results['std_length']:.2f}"
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
