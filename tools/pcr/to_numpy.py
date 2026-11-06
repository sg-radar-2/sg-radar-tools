
from pathlib import Path
import numpy as np
from pcr_loader import load

# 1. PCR 파일 불러오기
root = Path(__file__).resolve().parents[2]
pcr_path = root / "recordings" / "20261104_left_right_walk_run.pcr"
recording = load(pcr_path)

# 2. 배열에 넣을 데이터 준비
point_rows = []
target_rows = []

for session in recording.sessions:
    elapsed_us = 0

    for frame in session.frames:
        elapsed_us += frame.delta_us
        time_s = elapsed_us / 1_000_000

        # 포인트 클라우드
        for p in frame.points:
            point_rows.append([
                time_s, frame.frame_count,
                p.x, p.y, p.z,
                p.doppler, p.power, p.target_id
            ])

        # 레이더가 추적한 사람 위치
        for t in frame.targets:
            target_rows.append([
                time_s, frame.frame_count,
                t.target_id, t.x, t.y, t.status
            ])

# 3. NumPy 배열 변환
points = np.array(point_rows, dtype=np.float64).reshape(-1, 8)
targets = np.array(target_rows, dtype=np.float64).reshape(-1, 6)

# 4. 저장
output = root / "recordings" / "left_right_numpy.npz"
np.savez_compressed(output, points=points, targets=targets)

print("포인트 배열 크기:", points.shape)
print("타겟 배열 크기:", targets.shape)
print("첫 번째 포인트:", points[0] if len(points) else "없음")
print("저장 완료:", output)
