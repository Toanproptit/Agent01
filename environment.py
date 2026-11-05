"""Tabular maze environment used by the Q-learning demo."""
from __future__ import annotations

import numpy as np


DEFAULT_MAZE = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
], dtype=np.int8)

START = (0, 0)
GOAL = (9, 9)
ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("Lên", "Xuống", "Trái", "Phải")


class MazeEnvironment:
    """Grid maze. Invalid moves terminate with -10; goal terminates with +50."""

    def __init__(self, maze: np.ndarray | None = None, start=START, goal=GOAL,
                 goal_reward: float = 50, invalid_reward: float = -10,
                 step_reward: float = -1):
        self.maze = np.array(DEFAULT_MAZE if maze is None else maze, dtype=np.int8, copy=True)
        if self.maze.ndim != 2 or not np.all(np.isin(self.maze, (0, 1))):
            raise ValueError("maze must be a 2D array containing only 0 and 1")
        self.start, self.goal = tuple(start), tuple(goal)
        self.goal_reward, self.invalid_reward, self.step_reward = goal_reward, invalid_reward, step_reward
        self.height, self.width = self.maze.shape
        if not self.is_valid(self.start) or not self.is_valid(self.goal):
            raise ValueError("start and goal must be valid open cells")
        self.reset()

    def is_valid(self, state) -> bool:
        r, c = state
        return 0 <= r < self.height and 0 <= c < self.width and self.maze[r, c] == 0

    def reset(self):
        self.state = self.start
        self.terminated = self.state == self.goal
        return self.state

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("episode is already terminal; call reset() before step()")
        if not 0 <= int(action) < len(ACTION_DELTAS):
            raise ValueError("action must be 0 (up), 1 (down), 2 (left), or 3 (right)")
        dr, dc = ACTION_DELTAS[int(action)]
        candidate = (self.state[0] + dr, self.state[1] + dc)
        if not self.is_valid(candidate):
            self.terminated = True
            return self.state, self.invalid_reward, True, {"invalid": True, "reached_goal": False}
        self.state = candidate
        reached_goal = candidate == self.goal
        self.terminated = reached_goal
        reward = self.goal_reward if reached_goal else self.step_reward
        return self.state, reward, self.terminated, {"invalid": False, "reached_goal": reached_goal}
