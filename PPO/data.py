from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComplexDataset:
    features: np.ndarray
    labels: np.ndarray


def _label_rule(features: np.ndarray) -> np.ndarray:
    logits = np.zeros((features.shape[0], 4), dtype=float)
    logits[:, 0] = 1.6 * features[:, 0] + 0.9 * np.sin(features[:, 3]) - 0.5 * features[:, 9]
    logits[:, 1] = -0.8 * features[:, 1] + 1.2 * features[:, 4] + np.cos(features[:, 5])
    logits[:, 2] = 1.0 * features[:, 2] * features[:, 6] - 0.7 * features[:, 7]
    logits[:, 3] = 0.6 * features[:, 8] ** 2 - 0.4 * features[:, 10] + 0.3 * features[:, 11]
    return np.argmax(logits, axis=1)


def make_complex_dataset(size: int, feature_dim: int, seed: int) -> ComplexDataset:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(size, feature_dim)).astype(np.float64)
    features[:, 6] = np.tanh(features[:, 6])
    features[:, 8] = np.clip(features[:, 8], -2.5, 2.5)
    labels = _label_rule(features)
    return ComplexDataset(features=features, labels=labels)


def train_test_split(dataset: ComplexDataset, split_ratio: float) -> tuple[ComplexDataset, ComplexDataset]:
    split = int(split_ratio * len(dataset.features))
    return (
        ComplexDataset(dataset.features[:split], dataset.labels[:split]),
        ComplexDataset(dataset.features[split:], dataset.labels[split:]),
    )
