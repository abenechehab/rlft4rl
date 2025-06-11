import os
import time
import numpy as np
import tyro
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from tqdm import tqdm

import torch

from rlft4rl.utils import setup_logger, make_env, set_seed_everywhere


@dataclass
class Args:
    """Arguments for policy evaluation."""

    env_id: str = "CartPole-v1"  # Environment ID
    seed: int = 42  # Random seed
    episodes: int = 10  # Number of evaluation episodes
    capture_video: bool = False  # Whether to capture video
    verbose: bool = True  # Print detailed information during evaluation
    log_level: str = "INFO"  # Logging level
    mode: str = "random"  # "constant", "mlp"
    mlp_policy_path: str = "models/mlp-policy/halfcheetah_medium/BC_test.pth"  # Path to MLP policy (if using MLP mode)


def evaluate_policy(args: Args, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate a policy in the given environment."""

    # Create directory for videos if needed
    if args.capture_video:
        os.makedirs("videos", exist_ok=True)

    # Create environment
    run_name = f"{args.env_id}/{args.mode}-policy__{args.seed}__{int(time.time())}"
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

    if args.mode == "mlp" and args.mlp_policy_path:
        # Load MLP policy if specified
        from rlft4rl.policies.mlp_policy import MLPPolicy, PolicyTrainer

        policy = MLPPolicy(
            obs_dim=obs_space.shape[0],
            action_dim=act_space.shape[0],
            hidden_dims=(256, 256),
            activation="leaky_relu",
            output_activation="linear",
            dropout=0.1,
        )
        policy_trainer = PolicyTrainer(policy=policy)
        policy_trainer.load_model(path=args.mlp_policy_path)
        policy = policy_trainer.policy
        policy.eval()
        device = policy_trainer.device
        logger.info(f"Loaded MLP policy from {args.mlp_policy_path}")
    elif args.mode == "constant":

        def policy(observation: Optional[np.array] = None):
            action = env.action_space.sample()
            return np.zeros_like(action)
    else:

        def policy(observation: Optional[np.array] = None):
            return env.action_space.sample()

    # Run evaluation episodes
    for ep in tqdm(range(args.episodes), desc="-> Evaluating policy"):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False

        while not done:
            # action
            if args.mode == "mlp" and args.mlp_policy_path:
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy(obs)
            if type(action) is torch.Tensor:
                action = action.detach().cpu().numpy().flatten()

            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)
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
        logger_name="Peval",
        log_level=args.log_level,
        log_dir="logs",
        env_id=args.env_id,
        exp_name=f"{args.mode}-policy",
    )

    # Run evaluation
    results = evaluate_policy(args, logger)

    # Log results
    logger.info(
        f"\n-> Results for {args.env_id} with {args.mode} policy (seed={args.seed}):"
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
