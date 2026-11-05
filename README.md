# Q-Learning tìm đường trong mê cung

Demo Streamlit mô phỏng agent học tìm đường trên mê cung 10×10 bằng Q-Learning. Mê cung, start, goal, phần thưởng và công thức cập nhật bám theo ví dụ trong [bài GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/).

## Cấu trúc

```text
assign01/
├── agent.py             # Q-table, epsilon-greedy, cập nhật và lưu/tải
├── app.py               # Giao diện Streamlit
├── environment.py       # Mê cung và reward/terminal states
├── evaluation.py        # Chạy policy greedy, epsilon = 0
├── training.py          # Vòng huấn luyện và trung bình trượt
├── visualization.py     # Vẽ mê cung và biểu đồ reward
├── requirements.txt
└── tests/test_rl.py
```

## Cài đặt và chạy local

Python 3.10+ được khuyến nghị.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Mở URL local được Streamlit in ra (thường là `http://localhost:8501`). Nhấn **Train Agent**, rồi **Run Agent** để xem lượt đánh giá. Nút **Lưu Q-Table** ghi file `.npz`; có thể tải file đó lên lần chạy kế tiếp.

## Huấn luyện dòng lệnh và kiểm thử

```bash
python -m pytest -q
```

Thiết lập mặc định: 5.000 episodes, α=0.1, γ=0.9, ε=0.5, decay=0.995, ε tối thiểu=0.01, tối đa 200 bước mỗi episode, seed=42. Mỗi episode kết thúc khi chạm đích, thử bước vào tường/ra ngoài biên, hoặc chạm giới hạn bước. Bước không hợp lệ nhận -10 và kết thúc episode; bước hợp lệ nhận -1; đến đích nhận +50.

## Q-Learning

Q-table có kích thước `10 × 10 × 4`, với bốn hành động lên, xuống, trái, phải. Sau mỗi hành động agent cập nhật:

```text
Q(s,a) ← Q(s,a) + α [r + γ maxₐ′ Q(s′,a′) − Q(s,a)]
```

Nếu chuyển trạng thái kết thúc, giá trị tương lai bằng 0 nên không bootstrap qua terminal state. Epsilon-greedy chọn hành động ngẫu nhiên với xác suất ε, nếu không thì chọn hành động có Q lớn nhất; ε giảm sau từng episode. Đánh giá gọi cùng policy với ε=0 và không thay đổi Q-table. Biểu đồ và tỷ lệ thành công theo episode là dữ liệu huấn luyện; số bước/reward ở phần đánh giá thuộc một lượt chạy riêng.
