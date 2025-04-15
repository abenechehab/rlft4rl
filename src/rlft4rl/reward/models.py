import os
from typing import Optional
import time
from dataclasses import dataclass
import tyro

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import minari

from rlft4rl.utils import setup_logger, set_seed_everywhere


@dataclass
class Args:
    dataset_name: str = "mujoco/halfcheetah/expert-v0"
    hidden_dim: int = 256
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "models/reward_models/halfcheetah_expert"
    normalize: bool = True
    dropout_rate: float = 0.1
    use_data_parallel: bool = True  # Add a flag to control DataParallel usage
    version: Optional[str] = None
    seed: int = 42


########################
# Setup logging
########################
logger, _, ts_writer = setup_logger(
    logger_name="reward_model",
    log_dir="logs",
    env_id="reward_model_training",
    exp_name="rwm",
    create_ts_writer=True,
)
#############################


class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Register buffers for normalization stats
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_std", torch.tensor(1.0))

        # Enhanced architecture with BatchNorm and Dropout
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def normalize_state(self, state):
        if torch.all(self.state_std == 1) and torch.all(self.state_mean == 0):
            raise RuntimeError("State normalization statistics have not been updated.")
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def normalize_action(self, action):
        if torch.all(self.action_std == 1) and torch.all(self.action_mean == 0):
            raise RuntimeError("Action normalization statistics have not been updated.")
        return (action - self.action_mean) / (self.action_std + 1e-8)

    def denormalize_reward(self, reward):
        if self.reward_std == 1 and self.reward_mean == 0:
            raise RuntimeError("Reward normalization statistics have not been updated.")
        return reward * self.reward_std + self.reward_mean

    def forward(self, state, action, normalize_inputs=True, denormalize_output=True):
        if normalize_inputs:
            state = self.normalize_state(state)
            action = self.normalize_action(action)

        x = torch.cat([state, action], dim=-1)
        pred_reward = self.net(x).squeeze(-1)

        if denormalize_output:
            pred_reward = self.denormalize_reward(pred_reward)

        return pred_reward

    def update_normalization_stats(self, states, actions, rewards=None):
        """Update the normalization statistics using the provided data."""
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0) + 1e-8

        self.action_mean = actions.mean(dim=0)
        self.action_std = actions.std(dim=0) + 1e-8

        if rewards is not None:
            self.reward_mean = rewards.mean()
            self.reward_std = rewards.std() + 1e-8


class MinariRLDataset(Dataset):
    def __init__(self, dataset_name, seed: int = 42):
        dataset = minari.load_dataset(dataset_name)
        dataset.set_seed(seed=seed)

        self.states = []
        self.actions = []
        self.rewards = []
        for episode in dataset:
            self.states.append(episode.observations[:-1])
            self.actions.append(episode.actions)
            self.rewards.append(episode.rewards)
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]


def train_reward_model(
    dataset,
    hidden_dim=256,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=100,
    patience=10,
    val_split=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_dir="models/reward_models/test",
    normalize=True,
    dropout_rate=0.1,
    use_data_parallel=False,  # Add use_data_parallel parameter
    version=None,
):
    os.makedirs(log_dir, exist_ok=True)

    # dataset
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # Split dataset
    val_size = int(round(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    assert train_size + val_size == len(dataset), (
        "Train and val sizes do not sum to dataset length"
    )
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, optimizer, scheduler
    reward_model = RewardModel(state_dim, action_dim, hidden_dim, dropout_rate).to(
        device
    )

    # Compute normalization statistics using the entire dataset
    if normalize:
        logger.info("Computing normalization statistics...")
        reward_model.update_normalization_stats(
            dataset.states.to(device),
            dataset.actions.to(device),
            dataset.rewards.to(device),
        )
        logger.info("Normalization statistics updated.")

    if use_data_parallel and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        reward_model = nn.DataParallel(reward_model)

    optimizer = optim.Adam(reward_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        reward_model.train()
        train_losses = []

        for states, actions, rewards in train_loader:
            states, actions, rewards = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
            )
            optimizer.zero_grad()

            # During training, we normalize inputs but work with normalized rewards
            pred_rewards = reward_model(
                states,
                actions,
                normalize_inputs=normalize,
                denormalize_output=normalize,
            )

            loss = criterion(pred_rewards, rewards)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        reward_model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions, rewards in val_loader:
                states, actions, rewards = (
                    states.to(device),
                    actions.to(device),
                    rewards.to(device),
                )

                # Same approach for validation as training
                pred_rewards = reward_model(
                    states,
                    actions,
                    normalize_inputs=normalize,
                    denormalize_output=normalize,
                )

                loss = criterion(pred_rewards, rewards)

                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # TensorBoard logging
        ts_writer.add_scalar("Loss/Train", train_loss, epoch)
        ts_writer.add_scalar("Loss/Val", val_loss, epoch)
        ts_writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, epoch_time={epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(reward_model.state_dict(), log_dir + f"/rw{version}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    ts_writer.close()
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {log_dir}/rw{version}.pt")
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s")


def main(args: Args):
    set_seed_everywhere(args.seed)

    dataset = MinariRLDataset(args.dataset_name, seed=args.seed)

    train_reward_model(
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        val_split=args.val_split,
        device=args.device,
        log_dir=args.log_dir,
        normalize=args.normalize,
        dropout_rate=args.dropout_rate,
        use_data_parallel=args.use_data_parallel,  # Pass the flag to train_reward_model
        version=args.version,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
