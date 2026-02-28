"""Reference implementations for classic and modern reinforcement learning algorithms.

This module contains compact, runnable NumPy implementations of:
- Dynamic Programming (policy evaluation/improvement, policy iteration, value iteration)
- REINFORCE (Monte Carlo policy gradient)
- PPO (clipped surrogate objective)
- TRPO (KL-constrained trust-region update)

The implementations target small discrete Markov Decision Processes (MDPs) and are
intended for learning, experimentation, and extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class TabularMDP:
    """Finite MDP with known transition dynamics.

    Attributes:
        transition_probabilities: Array with shape (S, A, S).
        rewards: Array with shape (S, A, S).
        gamma: Discount factor in [0, 1).
    """

    transition_probabilities: np.ndarray
    rewards: np.ndarray
    gamma: float = 0.99

    @property
    def n_states(self) -> int:
        return int(self.transition_probabilities.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition_probabilities.shape[1])


def policy_evaluation(mdp: TabularMDP, policy: np.ndarray, theta: float = 1e-8) -> np.ndarray:
    """Evaluate V^pi for a deterministic policy with iterative updates."""
    values = np.zeros(mdp.n_states, dtype=float)
    while True:
        delta = 0.0
        for state in range(mdp.n_states):
            action = int(policy[state])
            p = mdp.transition_probabilities[state, action]
            r = mdp.rewards[state, action]
            updated_value = np.sum(p * (r + mdp.gamma * values))
            delta = max(delta, abs(updated_value - values[state]))
            values[state] = updated_value
        if delta < theta:
            break
    return values


def policy_improvement(mdp: TabularMDP, values: np.ndarray) -> np.ndarray:
    """Return greedy deterministic policy with respect to value function."""
    q_values = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)
    for action in range(mdp.n_actions):
        q_values[:, action] = np.sum(
            mdp.transition_probabilities[:, action, :] * (mdp.rewards[:, action, :] + mdp.gamma * values), axis=1
        )
    return np.argmax(q_values, axis=1)


def policy_iteration(mdp: TabularMDP, theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal policy and value function using policy iteration."""
    policy = np.zeros(mdp.n_states, dtype=int)
    while True:
        values = policy_evaluation(mdp, policy, theta=theta)
        updated_policy = policy_improvement(mdp, values)
        if np.array_equal(updated_policy, policy):
            return policy, values
        policy = updated_policy


def value_iteration(mdp: TabularMDP, theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal policy and value function using value iteration."""
    values = np.zeros(mdp.n_states, dtype=float)
    while True:
        delta = 0.0
        for state in range(mdp.n_states):
            q_state = []
            for action in range(mdp.n_actions):
                p = mdp.transition_probabilities[state, action]
                r = mdp.rewards[state, action]
                q_state.append(np.sum(p * (r + mdp.gamma * values)))
            best_value = max(q_state)
            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
        if delta < theta:
            break
    return policy_improvement(mdp, values), values


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _collect_episode(
    mdp: TabularMDP,
    policy_logits: np.ndarray,
    start_state: int,
    max_steps: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, float]]:
    episode = []
    state = start_state
    for _ in range(max_steps):
        action_probabilities = _softmax(policy_logits[state])
        action = int(rng.choice(mdp.n_actions, p=action_probabilities))
        next_state = int(rng.choice(mdp.n_states, p=mdp.transition_probabilities[state, action]))
        reward = float(mdp.rewards[state, action, next_state])
        episode.append((state, action, reward))
        state = next_state
    return episode


def _discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def reinforce(
    mdp: TabularMDP,
    episodes: int = 300,
    learning_rate: float = 0.05,
    max_steps: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Train a tabular softmax policy using REINFORCE."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)

    for _ in range(episodes):
        trajectory = _collect_episode(mdp, logits, start_state=0, max_steps=max_steps, rng=rng)
        rewards = [step[2] for step in trajectory]
        returns = _discounted_returns(rewards, mdp.gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for t, (state, action, _) in enumerate(trajectory):
            probs = _softmax(logits[state])
            grad_log = -probs
            grad_log[action] += 1.0
            logits[state] += learning_rate * returns[t] * grad_log

    return np.argmax(logits, axis=1)


def ppo(
    mdp: TabularMDP,
    iterations: int = 80,
    episodes_per_iter: int = 8,
    learning_rate: float = 0.05,
    clip_epsilon: float = 0.2,
    max_steps: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Train tabular softmax policy with PPO clipped objective."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)

    for _ in range(iterations):
        updates = np.zeros_like(logits)
        for _ in range(episodes_per_iter):
            trajectory = _collect_episode(mdp, logits, start_state=0, max_steps=max_steps, rng=rng)
            rewards = [step[2] for step in trajectory]
            returns = _discounted_returns(rewards, mdp.gamma)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

            for t, (state, action, _) in enumerate(trajectory):
                old_probs = _softmax(logits[state])
                old_action_prob = old_probs[action]

                candidate_logits = logits.copy()
                grad_log = -old_probs
                grad_log[action] += 1.0
                candidate_logits[state] += learning_rate * advantages[t] * grad_log
                new_probs = _softmax(candidate_logits[state])
                new_action_prob = new_probs[action]

                ratio = new_action_prob / (old_action_prob + 1e-8)
                unclipped = ratio * advantages[t]
                clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages[t]
                chosen_objective = (
                    min(unclipped, clipped) if advantages[t] >= 0 else max(unclipped, clipped)
                )
                updates[state] += chosen_objective * grad_log

        logits += (learning_rate / episodes_per_iter) * updates

    return np.argmax(logits, axis=1)


def trpo(
    mdp: TabularMDP,
    iterations: int = 80,
    episodes_per_iter: int = 8,
    learning_rate: float = 0.05,
    max_steps: int = 30,
    max_kl: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Train tabular softmax policy with a KL-constrained trust region update."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)

    for _ in range(iterations):
        grad = np.zeros_like(logits)

        for _ in range(episodes_per_iter):
            trajectory = _collect_episode(mdp, logits, start_state=0, max_steps=max_steps, rng=rng)
            rewards = [step[2] for step in trajectory]
            returns = _discounted_returns(rewards, mdp.gamma)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

            for t, (state, action, _) in enumerate(trajectory):
                old_probs = _softmax(logits[state])
                grad_log = -old_probs
                grad_log[action] += 1.0
                proposed = logits[state] + learning_rate * advantages[t] * grad_log
                new_probs = _softmax(proposed)
                state_kl = np.sum(old_probs * (np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)))
                scaling = min(1.0, np.sqrt(max_kl / (state_kl + 1e-8)))
                grad[state] += scaling * advantages[t] * grad_log

        logits += learning_rate * grad / max(episodes_per_iter, 1)

    return np.argmax(logits, axis=1)


def build_sample_mdp(gamma: float = 0.95) -> TabularMDP:
    """Create a tiny 3-state MDP for demonstrations."""
    transition_probabilities = np.array(
        [
            [[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]],
            [[0.0, 0.2, 0.8], [0.0, 0.7, 0.3]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    rewards = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    return TabularMDP(transition_probabilities=transition_probabilities, rewards=rewards, gamma=gamma)


def run_all_algorithms() -> dict:
    """Run all included algorithms and return their resulting policies."""
    mdp = build_sample_mdp()
    pi_policy, pi_values = policy_iteration(mdp)
    vi_policy, vi_values = value_iteration(mdp)
    return {
        "policy_iteration": {"policy": pi_policy.tolist(), "values": pi_values.round(4).tolist()},
        "value_iteration": {"policy": vi_policy.tolist(), "values": vi_values.round(4).tolist()},
        "reinforce": {"policy": reinforce(mdp).tolist()},
        "ppo": {"policy": ppo(mdp).tolist()},
        "trpo": {"policy": trpo(mdp).tolist()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_all_algorithms(), indent=2))
