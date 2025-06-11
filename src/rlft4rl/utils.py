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


def trl_generate_completions(
    model, inputs, processing_class, generation_config, device
):
    # prompts_text = [
    #     processing_class.apply_chat_template(
    #         example["prompt"], tokenize=False, add_generation_prompt=True
    #     ).split("<|im_start|>assistant\n<action>")[0]
    #     + "<|im_start|>assistant\n<action>"
    #     for example in inputs
    # ]

    prompts_text = [example["prompt"] for example in inputs]

    prompt_inputs = processing_class(
        prompts_text,
        return_tensors="pt",
        # padding=True,
        # padding_side="left",
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        prompt_completion_ids = model.generate(
            **prompt_inputs, generation_config=generation_config
        )

    prompt_length = prompt_inputs["input_ids"].size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]
    completions = processing_class.batch_decode(
        completion_ids, skip_special_tokens=True
    )

    return completions, prompts_text


def scale_observation_to_tokens(obs_values, obs_ranges):
    """Scale observation values to 0-999 range for single-token encoding"""
    scaled = []
    for i, val in enumerate(obs_values):
        min_val, max_val = obs_ranges[i]
        # Clamp and scale to 0-999
        clamped = max(min_val, min(max_val, val))
        scaled_val = int(((clamped - min_val) / (max_val - min_val)) * 999)
        scaled.append(str(scaled_val))
    return scaled
