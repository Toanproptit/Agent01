"""Matplotlib views for maze routes and training history."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_maze(env, path=None, agent_position=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env.maze, cmap=ListedColormap(["#f2f6f4", "#263238"]), vmin=0, vmax=1)
    if path:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color="#258c70", linewidth=3, marker="o", markersize=3, label="Đường đi")
    ax.scatter(env.start[1], env.start[0], s=180, marker="s", color="#4c9aff", edgecolor="white", label="Bắt đầu", zorder=4)
    ax.scatter(env.goal[1], env.goal[0], s=230, marker="*", color="#ffb020", edgecolor="black", label="Đích", zorder=4)
    if agent_position is not None:
        ax.scatter(agent_position[1], agent_position[0], s=100, marker="o", color="#e84a5f", edgecolor="white", label="Agent", zorder=5)
    ax.set_xticks(np.arange(env.width)); ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True); ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="both", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_training_rewards(rewards, window=100):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(1, len(rewards) + 1)
    ax.plot(x, rewards, color="#a8b0b8", alpha=.45, linewidth=.8, label="Reward mỗi episode")
    if len(rewards):
        n = max(1, min(window, len(rewards)))
        avg = np.convolve(rewards, np.ones(n) / n, mode="valid")
        ax.plot(np.arange(n, len(rewards) + 1), avg, color="#187c63", linewidth=2, label=f"Trung bình trượt ({n})")
    ax.set(xlabel="Episode", ylabel="Tổng reward", title="Dữ liệu huấn luyện")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    return fig
