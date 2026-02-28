from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from PPO.config import PPOConfig
from PPO.data import ComplexDataset, make_complex_dataset, train_test_split
from PPO.model import MLPActorCritic


@dataclass(frozen=True)
class TrainResult:
    train_accuracy: float
    test_accuracy: float


def _batches(size: int, batch_size: int, rng: np.random.Generator) -> Generator[np.ndarray, None, None]:
    indices = np.arange(size)
    rng.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    encoded = np.zeros((len(actions), n_actions), dtype=float)
    encoded[np.arange(len(actions)), actions] = 1.0
    return encoded


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def _accuracy(model: MLPActorCritic, dataset: ComplexDataset) -> float:
    _, probs, _ = model.forward(dataset.features)
    predictions = np.argmax(probs, axis=1)
    return float(np.mean(predictions == dataset.labels))


def train_ppo(config: PPOConfig = PPOConfig()) -> TrainResult:
    rng = np.random.default_rng(config.seed)
    dataset = make_complex_dataset(config.dataset_size, config.feature_dim, config.seed)
    train_set, test_set = train_test_split(dataset, config.train_split)
    model = MLPActorCritic(config.feature_dim, config.hidden_dim, config.n_actions, config.seed)

    for _ in range(config.train_steps):
        for batch_idx in _batches(len(train_set.features), config.batch_size, rng):
            states = train_set.features[batch_idx]
            labels = train_set.labels[batch_idx]

            hidden, old_probs, old_values = model.forward(states)
            actions = np.array([rng.choice(config.n_actions, p=p) for p in old_probs], dtype=int)
            rewards = np.where(actions == labels, 1.0, -1.0)
            returns = _discounted_returns(rewards, config.gamma)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            selected_old = old_probs[np.arange(len(actions)), actions]
            action_1h = _one_hot(actions, config.n_actions)

            for _ in range(config.update_epochs):
                hidden, probs, values = model.forward(states)
                selected_new = probs[np.arange(len(actions)), actions]
                ratio = selected_new / (selected_old + 1e-8)
                clipped = np.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                policy_weight = np.minimum(ratio * advantages, clipped * advantages).reshape(-1, 1)
                policy_grad_logits = (probs - action_1h) * policy_weight / len(actions)

                value_error = (values - returns).reshape(-1, 1)
                value_grad = 2.0 * value_error / len(actions)

                d_hidden = policy_grad_logits @ model.w_policy.T + value_grad @ model.w_value.T
                d_hidden *= (1.0 - hidden**2)

                grads = {
                    "w1": states.T @ d_hidden,
                    "b1": np.sum(d_hidden, axis=0),
                    "w_policy": hidden.T @ policy_grad_logits,
                    "b_policy": np.sum(policy_grad_logits, axis=0),
                    "w_value": hidden.T @ value_grad,
                    "b_value": np.sum(value_grad, axis=0),
                }
                model.apply_gradients(grads, config.learning_rate)

    return TrainResult(train_accuracy=_accuracy(model, train_set), test_accuracy=_accuracy(model, test_set))
