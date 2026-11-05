"""Q-learning agent with a state x action Q-table."""
from __future__ import annotations

import numpy as np

from environment import ACTION_NAMES


class QLearningAgent:
    def __init__(self, height: int, width: int, alpha: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.5, seed: int = 42):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("alpha must be in (0,1], gamma and epsilon in [0,1]")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q_table = np.zeros((height, width, len(ACTION_NAMES)), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, epsilon: float | None = None) -> int:
        explore = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < explore:
            return int(self.rng.integers(self.q_table.shape[-1]))
        values = self.q_table[state]
        # Random tie breaking avoids a persistent directional bias at initialization.
        best = np.flatnonzero(np.isclose(values, values.max()))
        return int(self.rng.choice(best))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> float:
        old = self.q_table[state + (action,)]
        future = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * future
        self.q_table[state + (action,)] = old + self.alpha * (target - old)
        return float(self.q_table[state + (action,)])

    def save(self, path):
        np.savez_compressed(path, q_table=self.q_table,
                            alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            table = data["q_table"]
            if table.shape != self.q_table.shape:
                raise ValueError(f"Q-table shape {table.shape} does not match {self.q_table.shape}")
            self.q_table[:] = table
            self.alpha = float(data["alpha"])
            self.gamma = float(data["gamma"])
            self.epsilon = float(data["epsilon"])
