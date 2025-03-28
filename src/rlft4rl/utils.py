import os
import logging
from datetime import datetime

import shimmy  # noqa: F401 (needed for dm-control envs registration)
import gymnasium as gym

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    logger_name,
    exp_name,
    log_dir,
    env_id,
    log_level: str = "INFO",
    create_ts_writer: bool = True,
) -> tuple:
    # Clear existing handlers
    root = logging.getLogger(logger_name)
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, env_id, f"{exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, f"{timestamp}.log")

    # Set format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Set up root logger
    root.setLevel(getattr(logging, log_level))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Set up TensorBoard logger
    writer = None
    if create_ts_writer:
        try:
            tb_log_dir = os.path.join(run_dir, "tensorboard", timestamp)
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_log_dir)
        except ImportError:
            root.warning(
                "Failed to import TensorBoard. "
                "No TensorBoard logging will be performed."
            )

    return root, run_dir, writer


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            if "dm_control" in env_id:
                env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            if "dm_control" in env_id:
                env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
