"""Streamlit demo for Q-learning maze navigation. Run: streamlit run app.py"""
from pathlib import Path

import streamlit as st

from agent import QLearningAgent
from environment import MazeEnvironment
from evaluation import evaluate_agent
from training import train_agent
from visualization import plot_maze, plot_training_rewards


st.set_page_config(page_title="Q-Learning mê cung", page_icon="🧭", layout="wide")
st.title("🧭 Agent học tìm đường trong mê cung")
st.caption("Q-Learning dạng bảng • cấu hình mê cung theo ví dụ GeeksforGeeks")

with st.sidebar:
    st.header("Hyperparameter")
    episodes = st.number_input("Số episodes", min_value=1, max_value=100_000, value=5000, step=500)
    alpha = st.slider("Alpha (learning rate)", .01, 1.0, .1, .01)
    gamma = st.slider("Gamma (discount)", 0.0, 1.0, .9, .01)
    epsilon = st.slider("Epsilon ban đầu", 0.0, 1.0, .5, .01)
    decay = st.slider("Epsilon decay mỗi episode", .90, 1.0, .995, .001, format="%.3f")
    epsilon_min = st.slider("Epsilon tối thiểu", 0.0, .2, .01, .005)
    max_steps = st.number_input("Giới hạn bước / episode", 1, 2000, 200, 10)
    seed = st.number_input("Random seed", 0, 2**31 - 1, 42)
    model_file = st.text_input("Tệp Q-Table", "q_table.npz")

if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.training = None
    st.session_state.evaluation = None
env = MazeEnvironment()

left, right = st.columns([1, 1.15])
with left:
    st.subheader("Mê cung")
    st.pyplot(plot_maze(env), use_container_width=True)
    st.markdown("`0` = đường đi · `1` = vật cản · Start `(0, 0)` · Goal `(9, 9)`")
with right:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎓 Train Agent", type="primary", use_container_width=True):
            agent = QLearningAgent(env.height, env.width, alpha, gamma, epsilon, int(seed))
            with st.spinner(f"Đang huấn luyện {episodes:,} episodes..."):
                result = train_agent(env, agent, int(episodes), decay, epsilon_min, int(max_steps))
            st.session_state.agent, st.session_state.training = agent, result
            st.session_state.evaluation = None
            st.success(f"Huấn luyện xong. Epsilon cuối: {result.final_epsilon:.4f}")
    with c2:
        if st.button("▶ Run Agent", use_container_width=True):
            if st.session_state.agent is None:
                st.warning("Hãy Train Agent hoặc tải Q-Table trước.")
            else:
                st.session_state.evaluation = evaluate_agent(env, st.session_state.agent, int(max_steps))
    with c3:
        if st.button("💾 Lưu Q-Table", use_container_width=True):
            if st.session_state.agent is None:
                st.warning("Chưa có Q-Table để lưu.")
            else:
                st.session_state.agent.save(model_file)
                st.success(f"Đã lưu: {model_file}")
    upload = st.file_uploader("Tải Q-Table (.npz)", type=["npz"])
    if upload is not None:
        try:
            temp_path = Path("uploaded_q_table.npz")
            temp_path.write_bytes(upload.getvalue())
            agent = QLearningAgent(env.height, env.width, alpha, gamma, epsilon, int(seed))
            agent.load(temp_path)
            st.session_state.agent = agent
            st.success("Đã tải Q-Table. Bấm Run Agent để đánh giá.")
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Không thể tải Q-Table: {exc}")

    result = st.session_state.evaluation
    if result is not None:
        st.subheader("Kết quả đánh giá (greedy, epsilon = 0)")
        a, b, c = st.columns(3)
        a.metric("Đến đích", "Có ✅" if result["reached_goal"] else "Chưa ❌")
        b.metric("Số bước", result["steps"])
        c.metric("Reward đánh giá", f"{result['reward']:.0f}")
        playback_step = st.slider("Quan sát vị trí Agent trên đường đi", 0,
                                  max(0, len(result["path"]) - 1),
                                  max(0, len(result["path"]) - 1),
                                  key="playback_step")
        st.pyplot(plot_maze(env, result["path"], result["path"][playback_step]),
                  use_container_width=True)
        st.caption("Đây là một lượt đánh giá riêng, không được cộng vào dữ liệu huấn luyện.")

if st.session_state.training is not None:
    history = st.session_state.training
    st.subheader("Kết quả huấn luyện")
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", f"{len(history.rewards):,}")
    c2.metric("Tỷ lệ thành công toàn bộ", f"{sum(history.successes) / len(history.successes):.1%}")
    last = history.successes[-100:]
    c3.metric("Tỷ lệ thành công 100 episode cuối", f"{sum(last) / len(last):.1%}")
    st.pyplot(plot_training_rewards(history.rewards), use_container_width=True)
    st.caption("Reward/episode và trung bình trượt là dữ liệu huấn luyện. Chúng không bao gồm lượt đánh giá ở trên.")
