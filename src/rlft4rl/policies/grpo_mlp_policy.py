import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import tyro
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import minari

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from rlft4rl.utils import setup_logger, set_seed_everywhere

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    dataset_name: str = "mujoco/halfcheetah/medium-v0"
    hidden_dim: int = 256
    batch_size: int = 128
    activation: str = "leaky_relu"
    output_activation: str = "id"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    scheduler_patience: int = 10
    scheduler_ratio: float = 0.5
    early_stopping_patience: int = 20
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "models/mlp-policy/grpo_compatible"
    # normalize: bool = True
    dropout_rate: float = 0.0
    use_data_parallel: bool = True  # Add a flag to control DataParallel usage
    version: Optional[str] = None
    seed: int = 42
    log_prob: bool = True  # Use log probability loss instead of MSE loss
    eval_deterministic: bool = True  # Use deterministic policy for evaluation
    use_grpo: bool = False  # Use GRPO loss instead of MSE loss
    num_generations: int = 10  # Number of generations for GRPO
    clip_coef: float = 0.2  # Clipping coefficient for GRPO


def create_policy_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
) -> TensorDataset:
    """Create a DataLoader for policy training."""
    obs_tensor = torch.FloatTensor(observations)
    action_tensor = torch.FloatTensor(actions)

    dataset = TensorDataset(obs_tensor, action_tensor)

    return dataset


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPConfig(PretrainedConfig):
    model_type = "halfcheetah-mlp"

    def __init__(
        self,
        input_dim=17,
        output_dim=6,
        hidden_sizes=[64, 64],
        dropout=0.0,
        activation="tanh",
        output_activation="id",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = activation
        self.output_activation = output_activation


class MLPPolicy(PreTrainedModel):
    config_class = MLPConfig  # enable AutoModel support
    base_model_prefix = "halfcheetah-mlp"
    # Activation functions
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "id": nn.Identity(),
    }

    def __init__(self, config: MLPConfig):
        if (config.activation not in self.activations) or (
            config.output_activation not in self.activations
        ):
            raise ValueError(f"Unsupported activation: {config.activation}")
        super().__init__(config)
        layers = []
        dims = [config.input_dim] + config.hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            layers.append(self.activations[config.activation])
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)

        self.mean = nn.Sequential(
            layer_init(nn.Linear(dims[-1], config.output_dim), std=0.01),
            self.activations[config.output_activation],
        )
        self.log_std = nn.Sequential(
            layer_init(nn.Linear(dims[-1], config.output_dim)),
            self.activations[config.output_activation],
        )
        # self.log_std = nn.Sequential(
        #     nn.Linear(dims[-1], config.output_dim),
        #     self.activations[config.output_activation],
        # )
        # For stochastic policy: a trainable log_std parameter per action
        # self.log_std = nn.Parameter(torch.zeros(1, config.output_dim))

    def forward(self, state, num_logits_to_keep=None):
        # state shape: (batch_size, 17)
        if state.shape[-1] == self.config.output_dim:
            return state
        elif state.shape[-1] == self.config.input_dim:
            state = state.to(self.net[0].weight.device)
            common = self.net(state)  # (batch_size, 6)
            mean = self.mean(common)
            log_std = self.log_std(common)
            std = torch.exp(log_std)
            return mean, std
        else:
            raise ValueError(
                f"Expected input dimension {self.config.input_dim}, but got {state.shape[-1]}"
            )

    def generate(self, inputs=None, num_return_sequences=1, **kwargs):
        """
        Generate action samples given a batch of inputs (states).

        Args:
            inputs (torch.Tensor): input tensor of shape [batch_size, obs_dim]
            num_return_sequences (int): number of action samples per input

        Returns:
            torch.Tensor: tensor of shape [batch_size * num_return_sequences, action_dim]
        """

        mean, std = self.forward(kwargs["input_ids"].float())
        dist = torch.distributions.Normal(mean, std)

        actions = []
        for _ in range(num_return_sequences):
            sampled = dist.rsample()
            actions.append(sampled)

        # Stack along new dimension and reshape to interleave samples per batch element
        # Shape: [num_return_sequences, batch_size, action_dim] → [batch_size, num_return_sequences, action_dim]
        stacked_actions = torch.stack(actions, dim=0).transpose(0, 1)
        # Reshape to [batch_size * num_return_sequences, action_dim]
        all_actions = stacked_actions.reshape(-1, self.config.output_dim)

        # shape: [num_return_sequences, batch_size, action_dim] → [batch_size * num_return_sequences, action_dim]
        # all_actions = torch.cat(actions, dim=0)
        return all_actions

    def get_action(
        self, observation: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """
        Get action for a given state.

        Args:
            state (torch.Tensor): input tensor of shape [batch_size, obs_dim]

        Returns:
            torch.Tensor: action tensor of shape [batch_size, action_dim]
        """
        mean, std = self.forward(observation)
        if deterministic:
            # Return mean action for deterministic policy
            return mean
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return action


# Register the configuration
AutoConfig.register("halfcheetah-mlp", MLPConfig)
# Register the model
AutoModel.register(MLPConfig, MLPPolicy)


def create_mlp_model(
    obs_dim,
    action_dim,
    hidden_sizes=[64, 64],
    dropout=0.0,
    activation="tanh",
    output_activation="id",
    **kwargs,
):
    config = MLPConfig(
        input_dim=obs_dim,
        output_dim=action_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        output_activation=output_activation,
        **kwargs,
    )
    return MLPPolicy(config)


def load_mlp_model(model_path):
    config = MLPConfig.from_pretrained(model_path)
    return MLPPolicy.from_pretrained(model_path, config=config)


@dataclass
class PolicyTrainerArguments:
    """Arguments for PolicyTrainer configuration."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    use_data_parallel: bool = False
    log_prob: bool = False  # Use log probability loss instead of MSE loss
    deterministic: bool = True  # Use deterministic policy for evaluation
    use_grpo: bool = False  # Use GRPO loss instead of MSE loss
    num_generations: int = 10  # Number of generations for GRPO
    clip_coef: float = 0.2  # Clipping coefficient for GRPO


class PolicyTrainer:
    """Trainer for MLP Policy using behavioral cloning."""

    def __init__(
        self,
        policy: MLPPolicy,
        args: PolicyTrainerArguments,
        logger: Optional[logging.Logger] = None,
        ts_writer: Optional["SummaryWriter"] = None,
    ):
        self.policy = policy
        self.args = args

        self.device = self._setup_device(args.device)
        self.policy.to(self.device)
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.policy = nn.DataParallel(self.policy)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True,
            threshold=args.early_stopping_min_delta,
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.training_history = []

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.ts_writer = ts_writer

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

        for _, (observations, actions) in enumerate(dataloader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)

            # Forward pass and loss
            if self.args.use_grpo:
                loss = self.compute_grpo_loss(observations, actions)
            else:
                loss = self.compute_supervised_loss(observations, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def compute_supervised_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised loss for the policy."""
        if self.args.log_prob:
            mean, std = self.policy.forward(observations)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions)
            loss = -log_prob.mean()
        else:
            predicted_actions = self.policy.get_action(observations)
            loss = self.criterion(predicted_actions, actions)
        return loss

    def compute_grpo_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute GRPO loss for the policy."""
        mean, std = self.policy.forward(observations)
        dist = torch.distributions.Normal(mean, std)
        generated_actions = dist.sample(sample_shape=(self.args.num_generations,))
        log_probs = dist.log_prob(generated_actions).mean(dim=-1)
        ratio = torch.exp(log_probs)
        stacked_actions = torch.stack(
            [actions for _ in range(self.args.num_generations)], dim=0
        )
        rewards = (
            -nn.MSELoss(reduction="none")(generated_actions, stacked_actions)
            .mean(dim=-1)
            .detach()
        )
        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        advantages /= rewards.std(dim=0, keepdim=True) + 1e-4
        loss = -advantages * torch.clamp(
            ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
        )
        return loss.mean()

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the policy."""
        self.policy.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for observations, actions in dataloader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)

                predicted_actions = self.policy.get_action(
                    observations, deterministic=self.args.deterministic
                )
                loss = self.criterion(predicted_actions, actions)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        if val_loss < self.best_val_loss - self.args.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.args.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {self.patience_counter} epochs without improvement"
                )
                return True
        return False

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Train the policy."""
        self.logger.info(f"Starting policy training for {epochs} epochs")

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
            if self.ts_writer is not None:
                self.ts_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                # ts_writer.add_scalar(
                #     "action/std", torch.norm(torch.exp(self.policy.log_std), p=2), epoch
                # )
                if val_metrics:
                    self.ts_writer.add_scalar(
                        "val/loss", val_metrics["val_loss"], epoch
                    )
                if self.scheduler is not None:
                    self.ts_writer.add_scalar(
                        "train/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch,
                    )

            self.training_history.append(epoch_metrics)

            self.logger.info(
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
            self.logger.info("Loaded best model state")

        # Save the best model at the end of training
        self.save_model(save_path)

        return {"training_history": self.training_history}

    def save_model(self, path: Path):
        """Save the trained policy."""
        print(f"{self.policy}\n===========\n")
        # for k, v in self.policy.named_parameters():
        #     print(f"k - {k}\nv - {v}\n---\n")
        self.policy.save_pretrained(path)
        torch.save(
            {
                # "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "training_history": self.training_history,
                "best_val_loss": self.best_val_loss,
                "device": str(self.device),
            },
            path / "all_state_dict.pth",
        )
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained policy."""
        config = MLPConfig.from_pretrained(path)
        self.policy = MLPPolicy.from_pretrained(path, config=config)

        checkpoint = torch.load(path / "all_state_dict.pth", map_location=self.device)
        # self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.device = torch.device(checkpoint.get("device", "cpu"))
        self.logger.info(f"Model loaded from {path}")


def main(args: Args):
    set_seed_everywhere(args.seed)

    ########################
    # Setup logging
    ########################
    logger, _, ts_writer = setup_logger(
        logger_name="grpo-compatible-mlp_policy",
        log_dir="logs",
        env_id="mlp_policy_training",
        exp_name=f"grpo-mlp_pi/{args.version}",
        create_ts_writer=True,
    )
    #############################

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

    os.makedirs(args.log_dir, exist_ok=True)

    # Create policy
    policy = create_mlp_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
        dropout=args.dropout_rate,
        activation=args.activation,
        output_activation=args.output_activation,
    )

    # Create trainer
    trainer_args = PolicyTrainerArguments(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_ratio,
        early_stopping_patience=args.early_stopping_patience,
        use_data_parallel=args.use_data_parallel,
        log_prob=args.log_prob,
        deterministic=args.eval_deterministic,
        use_grpo=args.use_grpo,
        num_generations=args.num_generations,
        clip_coef=args.clip_coef,
    )
    trainer = PolicyTrainer(
        policy=policy, args=trainer_args, logger=logger, ts_writer=ts_writer
    )

    # Train policy
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        save_path=Path(args.log_dir) / args.version,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
