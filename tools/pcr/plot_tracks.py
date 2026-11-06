
import numpy as np
import matplotlib.pyplot as plt

# NumPy 데이터 불러오기
data = np.load("recordings/left_right_numpy.npz")
targets = data["targets"]

# 열 구조: time, frame_id, target_id, x, y, status
time = targets[:, 0]
target_id = targets[:, 2].astype(int)
x = targets[:, 3]

# ID별로 색을 다르게 표시
plt.figure(figsize=(11, 5))

for tid in np.unique(target_id):
    mask = target_id == tid
    plt.scatter(time[mask], x[mask], s=10, label=f"ID {tid}")

plt.xlabel("Time (seconds)")
plt.ylabel("X Position (m)")
plt.title("Left-Right Movement")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("recordings/left_right_track.png", dpi=150)
plt.show()
