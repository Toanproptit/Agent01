import numpy as np
import pytest

from agent import QLearningAgent
from environment import MazeEnvironment
from evaluation import evaluate_agent
from training import train_agent


def test_q_update_uses_bellman_target():
    agent = QLearningAgent(2, 2, alpha=.1, gamma=.9, seed=0)
    agent.q_table[1, 1, :] = [1, 2, 3, 4]
    value = agent.update((0, 0), 3, -1, (1, 1), False)
    assert value == pytest.approx(.1 * (-1 + .9 * 4))


def test_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(2, 2, alpha=.1, gamma=.9, seed=0)
    agent.q_table[1, 1, :] = 100
    assert agent.update((0, 0), 0, 50, (1, 1), True) == pytest.approx(5)


def test_rewards_invalid_boundary_and_goal():
    env = MazeEnvironment(maze=np.zeros((2, 2), dtype=int), start=(0, 0), goal=(1, 1))
    assert env.step(2)[1:] == (-10, True, {"invalid": True, "reached_goal": False})
    env.reset()
    state, reward, done, info = env.step(1)
    assert (state, reward, done, info["reached_goal"]) == ((1, 0), -1, False, False)
    env.step(3)
    assert env.state == (1, 1) and env.terminated
    with pytest.raises(RuntimeError):
        env.step(0)


def test_training_and_greedy_path_reach_goal_on_default_maze():
    env = MazeEnvironment()
    agent = QLearningAgent(10, 10, alpha=.1, gamma=.9, epsilon=.5, seed=42)
    history = train_agent(env, agent, episodes=5000, epsilon_decay=.995,
                          epsilon_min=.01, max_steps=200)
    result = evaluate_agent(env, agent)
    assert len(history.rewards) == 5000
    assert result["reached_goal"]
    assert result["path"][0] == env.start and result["path"][-1] == env.goal
    assert all(env.is_valid(pos) for pos in result["path"])


def test_q_table_round_trip(tmp_path):
    first = QLearningAgent(2, 3, seed=1)
    first.q_table[1, 2, 3] = 4.5
    path = tmp_path / "q.npz"
    first.save(path)
    second = QLearningAgent(2, 3, seed=2)
    second.load(path)
    np.testing.assert_array_equal(first.q_table, second.q_table)
