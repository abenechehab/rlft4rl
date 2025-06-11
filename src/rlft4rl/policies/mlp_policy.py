import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import tyro
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import minari

from rlft4rl.utils import setup_logger, set_seed_everywhere


@dataclass
class Args:
    dataset_name: str = "mujoco/halfcheetah/medium-v0"
    hidden_dim: int = 256
    batch_size: int = 128
    activation: str = "leaky_relu"
    output_activation: str = "linear"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    scheduler_patience: int = 10
    scheduler_ratio: float = 0.5
    early_stopping_patience: int = 20
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "models/mlp-policy/halfcheetah_medium"
    # normalize: bool = True
    dropout_rate: float = 0.1
    use_data_parallel: bool = True  # Add a flag to control DataParallel usage
    version: Optional[str] = None
    seed: int = 42


########################
# Setup logging
########################
logger, _, ts_writer = setup_logger(
    logger_name="mlp_policy",
    log_dir="logs",
    env_id="mlp_policy_training",
    exp_name="mlp_pi",
    create_ts_writer=True,
)
#############################


def create_policy_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
) -> TensorDataset:
    """Create a DataLoader for policy training."""
    obs_tensor = torch.FloatTensor(observations)
    action_tensor = torch.FloatTensor(actions)

    dataset = TensorDataset(obs_tensor, action_tensor)

    return dataset


class MLPPolicy(nn.Module):
    """MLP Policy network that maps observations to actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        output_activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Activation functions
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build network layers
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activations[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        if output_activation in activations:
            layers.append(activations[output_activation])

        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.network(observations)

    def get_action(
        self, observation: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Get action from observation."""
        with torch.no_grad():
            action = self.forward(observation)
            if not deterministic:
                # Add noise for exploration
                noise = torch.randn_like(action) * 0.1
                action = action + noise
        return action


class PolicyTrainer:
    """Trainer for MLP Policy using behavioral cloning."""

    def __init__(
        self,
        policy: MLPPolicy,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-6,
        use_data_parallel=False,  # Add use_data_parallel parameter
    ):
        self.policy = policy
        self.device = self._setup_device(device)
        self.policy.to(self.device)
        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.policy = nn.DataParallel(self.policy)

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.training_history = []

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, (observations, actions) in enumerate(dataloader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)

            # Forward pass
            predicted_actions = self.policy(observations)
            loss = self.criterion(predicted_actions, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # if batch_idx % 100 == 0:
            #     logger.info(
            #         f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
            #         f"Loss: {loss.item():.6f}"
            #     )

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the policy."""
        self.policy.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for observations, actions in dataloader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)

                predicted_actions = self.policy(observations)
                loss = self.criterion(predicted_actions, actions)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.patience_counter} epochs without improvement"
                )
                return True
        return False

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the policy."""
        logger.info(f"Starting policy training for {epochs} epochs")

        best_model_state = None

        for epoch in tqdm(range(epochs), desc="Epochs", total=epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # Validation
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)

                # Check early stopping
                if self._check_early_stopping(val_metrics["val_loss"]):
                    break

                # Save best model state
                if val_metrics["val_loss"] == self.best_val_loss:
                    best_model_state = self.policy.state_dict().copy()

            # Learning rate scheduling
            if self.scheduler is not None and val_dataloader is not None:
                self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            if self.scheduler is not None:
                epoch_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

            # TensorBoard logging
            if ts_writer is not None:
                ts_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                if val_metrics:
                    ts_writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
                if self.scheduler is not None:
                    ts_writer.add_scalar(
                        "train/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch,
                    )

            self.training_history.append(epoch_metrics)

            logger.info(
                f"Epoch {epoch}: "
                f"Loss: {train_metrics['loss']:.6f}"
                + (f", Val Loss: {val_metrics['val_loss']:.6f}" if val_metrics else "")
                + (
                    f", LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    if self.scheduler
                    else ""
                )
            )

        # Load best model if available
        if best_model_state is not None:
            self.policy.load_state_dict(best_model_state)
            logger.info("Loaded best model state")

        # Save the best model at the end of training
        self.save_model(save_path)

        return {"training_history": self.training_history}

    def save_model(self, path: str):
        """Save the trained policy."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "training_history": self.training_history,
                "best_val_loss": self.best_val_loss,
                "device": str(self.device),
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.device = torch.device(checkpoint.get("device", "cpu"))
        logger.info(f"Model loaded from {path}")


# Example usage
def main(args: Args):
    set_seed_everywhere(args.seed)

    dataset = minari.load_dataset(args.dataset_name)
    dataset.set_seed(seed=args.seed)

    observations = []
    actions = []
    for episode in dataset:
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    # Create data loaders
    dataset = create_policy_dataset(observations, actions)
    # Split dataset
    val_size = int(round(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    assert train_size + val_size == len(dataset), (
        "Train and val sizes do not sum to dataset length"
    )
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    save_path = f"{args.log_dir}/BC_{args.version}.pth"
    os.makedirs(args.log_dir, exist_ok=True)

    # Create policy
    policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
        activation=args.activation,
        output_activation=args.output_activation,
        dropout=args.dropout_rate,
    )

    # Create trainer
    trainer = PolicyTrainer(
        policy,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_ratio,
        early_stopping_patience=args.early_stopping_patience,
        use_data_parallel=args.use_data_parallel,
    )

    # Train policy
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        save_path=save_path,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
