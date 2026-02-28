from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    feature_dim: int = 12
    hidden_dim: int = 32
    n_actions: int = 4
    dataset_size: int = 512
    train_split: float = 0.8
    learning_rate: float = 3e-3
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    batch_size: int = 64
    train_steps: int = 150
    seed: int = 7
