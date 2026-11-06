
from pathlib import Path
from collections import Counter
import csv
import numpy as np

root = Path(__file__).resolve().parents[2]

data = np.load(root / "recordings" / "left_right_numpy.npz")
points = data["points"]
targets = data["targets"]

# 프레임별·타겟별 포인트 수 계산
counts = Counter(
    (int(p[1]), int(p[7])) for p in points
)

rows = []

for t in targets:
    frame_id = int(t[1])
    target_id = int(t[2])
    n_points = counts[(frame_id, target_id)]

    rows.append({
        "time_s": round(float(t[0]), 4),
        "frame_id": frame_id,
        "target_id": target_id,
        "x": round(float(t[3]), 4),
        "y": round(float(t[4]), 4),
        "n_points": n_points,
        "quality": "low_points" if n_points < 30 else "ok"
    })

rows.sort(key=lambda r: r["time_s"])

# CSV 저장
output = root / "recordings" / "left_right_tracks_raw.csv"

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("총 트랙 기록:", len(rows))
print("추적 ID:", sorted(set(r["target_id"] for r in rows)))
print("포인트 부족 기록:",
      sum(r["quality"] == "low_points" for r in rows))
print("CSV 저장 완료:", output)
