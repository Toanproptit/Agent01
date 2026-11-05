"""Greedy, no-exploration evaluation helpers."""
from agent import QLearningAgent
from environment import MazeEnvironment


def evaluate_agent(env: MazeEnvironment, agent: QLearningAgent,
                   max_steps: int = 200) -> dict:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    state = env.reset()
    path = [state]
    total_reward = 0.0
    reached_goal = state == env.goal
    for _ in range(max_steps):
        if reached_goal:
            break
        action = agent.choose_action(state, epsilon=0.0)
        state, reward, done, info = env.step(action)
        total_reward += reward
        # Invalid attempts don't move, so only append actual movements.
        if path[-1] != state:
            path.append(state)
        if done:
            reached_goal = bool(info["reached_goal"])
            break
    return {"path": path, "reward": total_reward, "steps": len(path) - 1,
            "reached_goal": reached_goal}
