"""Training loop and default hyperparameters."""
from dataclasses import dataclass

import numpy as np

from agent import QLearningAgent
from environment import MazeEnvironment


@dataclass
class TrainingResult:
    rewards: list[float]
    successes: list[bool]
    final_epsilon: float


def train_agent(env: MazeEnvironment, agent: QLearningAgent, episodes: int = 5000,
                epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                max_steps: int = 200) -> TrainingResult:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    rewards, successes = [], []
    for _ in range(episodes):
        state = env.reset()
        total, reached_goal = 0.0, False
        for _step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
            if done:
                reached_goal = bool(info["reached_goal"])
                break
        rewards.append(total)
        successes.append(reached_goal)
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
    return TrainingResult(rewards, successes, agent.epsilon)


def moving_average(values, window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    return np.convolve(arr, np.ones(window) / window, mode="valid")
