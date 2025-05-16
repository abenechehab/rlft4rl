import os
import time
import numpy as np
import tyro
import logging
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm

from rlft4rl.utils import setup_logger, make_env, set_seed_everywhere


@dataclass
class Args:
    """Arguments for random policy evaluation."""

    env_id: str = "CartPole-v1"  # Environment ID
    seed: int = 42  # Random seed
    episodes: int = 10  # Number of evaluation episodes
    capture_video: bool = False  # Whether to capture video
    verbose: bool = True  # Print detailed information during evaluation
    log_level: str = "INFO"  # Logging level


def evaluate_random_policy(args: Args, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate a random policy in the given environment."""

    # Create directory for videos if needed
    if args.capture_video:
        os.makedirs("videos", exist_ok=True)

    # Create environment
    run_name = f"{args.env_id}/random-policy__{args.seed}__{int(time.time())}"
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

    # Run evaluation episodes
    for ep in tqdm(range(args.episodes), desc="-> Evaluating random policy"):
        _, _ = env.reset(seed=args.seed + ep)
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()
            # action = np.zeros_like(action)  # Zero action for random policy

            # Step environment
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if args.verbose:
            logger.debug(
                f"Episode {ep + 1}: "
                f"return={env.return_queue[-1]}, length={env.length_queue[-1]}"
            )

        # Record episode metrics
        episode_returns.append(env.return_queue[-1])
        episode_lengths.append(env.length_queue[-1])

    logger.info(f"all episodes return: {episode_returns}")

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
    return results


def main(args: Args):
    set_seed_everywhere(seed=args.seed)

    # Setup logger
    logger, _, _ = setup_logger(
        logger_name="RP",
        log_level=args.log_level,
        log_dir="logs",
        env_id=args.env_id,
        exp_name="random_policy",
    )

    # Run evaluation
    results = evaluate_random_policy(args, logger)

    # Log results
    logger.info(
        f"\n-> Results for {args.env_id} with random policy (seed={args.seed}):"
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
