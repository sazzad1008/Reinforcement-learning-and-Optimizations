from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class MLPActorCritic:
    def __init__(self, feature_dim: int, hidden_dim: int, n_actions: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.1, size=(feature_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w_policy = rng.normal(0.0, 0.1, size=(hidden_dim, n_actions))
        self.b_policy = np.zeros(n_actions, dtype=float)
        self.w_value = rng.normal(0.0, 0.1, size=(hidden_dim, 1))
        self.b_value = np.zeros(1, dtype=float)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w_policy + self.b_policy
        probs = _softmax(logits)
        values = (hidden @ self.w_value + self.b_value).reshape(-1)
        return hidden, probs, values

    def apply_gradients(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.w1 -= lr * grads["w1"]
        self.b1 -= lr * grads["b1"]
        self.w_policy -= lr * grads["w_policy"]
        self.b_policy -= lr * grads["b_policy"]
        self.w_value -= lr * grads["w_value"]
        self.b_value -= lr * grads["b_value"]
