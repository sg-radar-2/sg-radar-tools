"""문 앞 상황 가짜 데이터 생성기

녹화가 생기기 전에 파이프라인(특징 계산 → 결정 트리 → 상황 판정 → 카메라 촬영)을 끝까지 시험하려고
8가지 상황의 궤적을 만들고, 레이더가 실제로 주는 형태(tracks_raw.csv)로 바꿔 저장합니다.
가짜 데이터로 학습한 결과는 코드 확인용입니다. 실제 성능은 녹화로 다시 확인해야 합니다.

좌표: 레이더 = 문 위치 = (0, 0), 레이더 정면이 +y. 단위 m, 초.

레이더 흉내
  - 위치: 방사형 격자(거리 7.5cm, 각도 약 1.04도)로 끊어서 찍고, 흔들림(표준편차 약 7cm)을 더함
  - 도플러: 문 방향 속도의 일부 + 잡음 (몸 전체 평균이라 실제 속도보다 작게 나옴)
  - doppler_std: 걷거나 손을 움직이면 커짐
  - 포인트 수: 가끔 30개 미만으로 떨어지는 구간 (트래킹 품질 저하)
  - 시야각 90도 밖, 0.15m 안쪽, 6m 밖은 안 보임

실행:
    python tools/behavior/scenario_sim.py                      # demo 시나리오 CSV 만들기
    python tools/behavior/scenario_sim.py --out recordings/sim_demo_tracks_raw.csv --seed 3
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

FPS = 18.0
DT = 1.0 / FPS

# 상황 이름 (한국어 표시명)
PASS = "통과"
RETURN = "귀가"
GO_OUT = "외출"
VISIT = "방문"
DELIVERY = "배달"
DWELL = "장시간 체류"
LOITER = "배회"
REPEAT = "반복 접근"
UNDECIDED = "판단 중"

SITUATIONS = [PASS, RETURN, GO_OUT, VISIT, DELIVERY, DWELL, LOITER, REPEAT]
ALIASES = {"배달/택배": DELIVERY, "택배": DELIVERY, "배달·택배": DELIVERY}   # 정답 표에서 같은 뜻으로 받는 이름
IDLE_FAR = "멀리 서 있기"   # 학습용 '해당 없음' 예시 (정답은 끝까지 '판단 중')
ABNORMAL = {DWELL, LOITER, REPEAT}

# 움직임 이름
M_APPROACH = "다가옴"
M_LEAVE = "멀어짐"
M_PASS = "지나감"
M_STILL = "정지"

HFOV_HALF = math.radians(45)
RANGE_RES = 0.075
ANGLE_RES = math.radians(1.04)


# =========================================================================== 궤적 만들기

@dataclass
class Frame:
    t: float
    x: float
    y: float
    hand: bool = False      # 손을 움직이는 중 (노크, 비밀번호 입력)
    crouch: bool = False    # 숙이거나 앉음 (택배 내려놓기)
    situation: str = UNDECIDED


@dataclass
class Path_:
    """걷기·멈추기를 이어 붙여 프레임 목록을 만든다."""
    x: float
    y: float
    t: float = 0.0
    frames: list = field(default_factory=list)

    def _emit(self, **kw):
        self.frames.append(Frame(self.t, self.x, self.y, **kw))
        self.t += DT

    def walk_to(self, x, y, speed, **kw):
        d = math.hypot(x - self.x, y - self.y)
        n = max(1, int(d / speed * FPS))
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            a = i / n
            # 걸음마다 좌우로 살짝 흔들림
            sway = 0.03 * math.sin(self.t * 2 * math.pi * 1.8)
            nx, ny = -(y - y0) / (d + 1e-9), (x - x0) / (d + 1e-9)
            self.x = x0 + (x - x0) * a + nx * sway
            self.y = y0 + (y - y0) * a + ny * sway
            self._emit(**kw)
        self.x, self.y = x, y

    def stay(self, seconds, **kw):
        cx, cy = self.x, self.y
        for _ in range(max(1, int(seconds * FPS))):
            # 서 있어도 몸이 조금씩 흔들림
            self.x = cx + 0.02 * math.sin(self.t * 1.3)
            self.y = cy + 0.02 * math.cos(self.t * 0.9)
            self._emit(**kw)
        self.x, self.y = cx, cy

    def mark(self, from_t, label):
        """from_t 이후 프레임의 정답 상황을 label로 정한다."""
        for f in self.frames:
            if f.t >= from_t - 1e-9:
                f.situation = label


def _far_point(rng, r=(3.4, 4.6)):
    ang = rng.uniform(-0.6, 0.6)
    d = rng.uniform(*r)
    return d * math.sin(ang), d * math.cos(ang)


def _side_exit(rng, y):
    side = rng.choice([-1, 1])
    return side * (y + 1.0), y


def episode(kind: str, rng: random.Random) -> list[Frame]:
    """상황 하나의 궤적. 프레임마다 정답 상황(판단 가능해진 뒤부터)이 붙는다."""
    def walk():
        v = rng.uniform(0.8, 1.4)
        return v * rng.uniform(1.6, 2.2) if rng.random() < 0.25 else v   # 4번에 1번은 뜀

    if kind == PASS:
        y = rng.uniform(0.7, 3.0)
        side = rng.choice([-1, 1])
        p = Path_(side * (y + 0.8), y + rng.uniform(-0.2, 0.2))
        p.walk_to(-side * (y + 0.8), y + rng.uniform(-0.2, 0.2), walk() * rng.choice([1, 1, 1.8]))
        mid = next(f.t for f in p.frames if f.x * side <= 0)
        p.mark(mid, PASS)

    elif kind == RETURN:
        p = Path_(*_far_point(rng))
        p.walk_to(rng.uniform(-0.12, 0.12), rng.uniform(0.32, 0.45), walk())
        p.stay(rng.uniform(1.5, 5.0), hand=True)        # 도어락 비밀번호 입력
        # 문 안으로 들어가며 사라짐 → 마지막 순간에만 판단 가능
        p.mark(p.t + 1, RETURN)

    elif kind == GO_OUT:
        p = Path_(rng.uniform(-0.12, 0.12), rng.uniform(0.32, 0.45))
        p.stay(rng.uniform(0.3, 1.5))
        r0 = math.hypot(p.x, p.y)
        if rng.random() < 0.7:
            p.walk_to(*_far_point(rng), walk())
        else:
            p.walk_to(rng.uniform(-0.2, 0.2), rng.uniform(1.0, 1.4), walk())
            p.walk_to(*_side_exit(rng, p.y), walk())
        t_dec = next((f.t for f in p.frames if math.hypot(f.x, f.y) >= r0 + 1.0), p.t)
        p.mark(t_dec, GO_OUT)

    elif kind == VISIT:
        # 문에 다가와 벨을 누르고 기다리다가, 집 안으로 들어감 (문에서 사라짐)
        p = Path_(*_far_point(rng))
        p.walk_to(rng.uniform(-0.3, 0.3), rng.uniform(0.6, 1.0), walk())
        t_stop = p.t
        p.stay(rng.uniform(0.5, 1.5))
        p.stay(rng.uniform(1.0, 2.0), hand=True)        # 벨·노크
        p.stay(rng.uniform(3.0, 28.0))                  # 문 열어주기를 기다림
        p.walk_to(rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.4), 0.6)
        p.mark(t_stop + 3, VISIT)

    elif kind == DELIVERY:
        # 문에 다가와 물건을 내려놓고 사진을 찍은 뒤 다시 나감
        p = Path_(*_far_point(rng))
        p.walk_to(rng.uniform(-0.3, 0.3), rng.uniform(0.6, 1.0), walk())
        t_stop = p.t
        if rng.random() < 0.7:
            p.stay(rng.uniform(1.0, 2.0), crouch=True)  # 내려놓기
        p.stay(rng.uniform(2.0, 6.0), hand=rng.random() < 0.5)   # 사진 찍기
        if rng.random() < 0.6:
            p.walk_to(*_far_point(rng), walk())
        else:
            p.walk_to(*_side_exit(rng, p.y + 0.3), walk())
        p.mark(t_stop + 3, DELIVERY)

    elif kind == DWELL:
        p = Path_(*_far_point(rng))
        p.walk_to(rng.uniform(-0.5, 0.5), rng.uniform(0.6, 1.4), walk())
        t_stop = p.t
        stay = rng.uniform(20, 40)
        p.stay(stay)
        p.walk_to(*_far_point(rng), walk())
        p.mark(t_stop + 15, DWELL)

    elif kind == LOITER:
        p = Path_(*_side_exit(rng, rng.uniform(1.5, 2.5)))
        t_start = p.t
        end = rng.uniform(35, 60)
        while p.t < end:
            ang = rng.uniform(-0.65, 0.65)
            d = rng.uniform(0.9, 3.0)
            p.walk_to(d * math.sin(ang), d * math.cos(ang), rng.uniform(0.5, 1.0))
            if rng.random() < 0.4:
                p.stay(rng.uniform(0.5, 3.0))
        p.walk_to(*_side_exit(rng, p.y), walk())
        p.mark(t_start + 20, LOITER)

    elif kind == REPEAT:
        # 외곽에서 나타났다가 (레이더 밖으로) 사라지기를 반복. 두 번째로 나타날 때부터 반복 접근
        p = Path_(*_far_point(rng, (6.3, 7.0)))
        t_dec = None
        for k in range(rng.choice([3, 3, 4])):
            p.walk_to(rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.6), walk())
            p.stay(rng.uniform(0.0, 1.5))
            p.walk_to(*_far_point(rng, (6.3, 7.0)), walk())
            if k == 0:
                t_dec = p.t
            p.stay(rng.uniform(1.0, 4.0))
        p.mark(t_dec, REPEAT)

    elif kind == IDLE_FAR:
        p = Path_(*_far_point(rng, (2.5, 4.5)))
        end = rng.uniform(10, 40)
        while p.t < end:
            p.stay(rng.uniform(3, 10))
            if rng.random() < 0.3:
                ang = rng.uniform(-0.5, 0.5)
                d = rng.uniform(2.5, 4.5)
                p.walk_to(d * math.sin(ang), d * math.cos(ang), rng.uniform(0.4, 0.8))

    else:
        raise ValueError(kind)
    return p.frames


# =========================================================================== 레이더 흉내

def true_motion(frames: list[Frame], i: int) -> str:
    a, b = frames[max(0, i - 4)], frames[min(len(frames) - 1, i + 4)]
    dt = b.t - a.t
    if dt <= 0:
        return M_STILL
    vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
    speed = math.hypot(vx, vy)
    if speed < 0.2:
        return M_STILL
    r = math.hypot(frames[i].x, frames[i].y) + 1e-9
    vr = (vx * frames[i].x + vy * frames[i].y) / r
    if abs(vr) / speed >= 0.7:
        return M_APPROACH if vr < 0 else M_LEAVE
    return M_PASS


def visible(x: float, y: float) -> bool:
    r = math.hypot(x, y)
    return 0.15 <= r <= 6.0 and y > 0 and abs(math.atan2(x, y)) <= HFOV_HALF


JITTER_SIGMA = 0.14   # 위치가 천천히 출렁이는 크기 (m). 실제 서 있는 녹화의 속도 분포에 맞춘 값
JITTER_TAU = 0.35     # 출렁임이 이어지는 시간 (초)


def sense(frames: list[Frame], rng: random.Random, target_id: int, t_offset: float = 0.0,
          id_switch: bool = False, final: str = "", jitter: float = JITTER_SIGMA) -> list[dict]:
    """궤적 → 레이더가 주는 프레임별 타겟 행 (tracks_raw.csv 형식 + 정답 열)."""
    rows = []
    ja = math.exp(-DT / JITTER_TAU)
    jb = math.sqrt(1 - ja * ja)
    ex = ey = 0.0
    low_until = -1.0
    tid = target_id
    switch_t = frames[len(frames) // 2].t if id_switch and len(frames) > 200 else None
    gap_until = -1.0
    last_vis = None
    for i, f in enumerate(frames):
        if not visible(f.x, f.y):
            continue
        if last_vis is not None and f.t - last_vis > 0.5:   # 레이더 밖에 나갔다 다시 들어오면 새 ID
            tid += 1000
        last_vis = f.t
        if switch_t is not None and f.t >= switch_t:
            gap_until, switch_t, tid = f.t + rng.uniform(0.3, 1.0), None, target_id + 100
        if f.t < gap_until:
            continue
        # 방사형 격자 + 흔들림
        r = math.hypot(f.x, f.y)
        ang = math.atan2(f.x, f.y)
        r = round((r + rng.gauss(0, 0.05)) / RANGE_RES) * RANGE_RES
        ang = round((ang + rng.gauss(0, 0.05 / max(r, 0.3))) / ANGLE_RES) * ANGLE_RES
        # 몸 중심이 천천히 출렁임 (시간 상관 있는 잡음) + 프레임마다 작은 잡음
        sig = jitter * (1.3 if f.hand else 1.0)
        ex = ja * ex + jb * rng.gauss(0, sig)
        ey = ja * ey + jb * rng.gauss(0, sig)
        x = r * math.sin(ang) + ex + rng.gauss(0, 0.03)
        y = r * math.cos(ang) + ey + rng.gauss(0, 0.03)

        # 문 방향 속도 → 도플러 (몸 평균이라 작아짐)
        a, b = frames[max(0, i - 2)], frames[min(len(frames) - 1, i + 2)]
        dt = max(b.t - a.t, 1e-6)
        vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
        rr = math.hypot(f.x, f.y) + 1e-9
        vr = (vx * f.x + vy * f.y) / rr
        speed = math.hypot(vx, vy)
        dop = 0.55 * vr + rng.gauss(0, 0.08) + (rng.gauss(0, 0.12) if f.hand else 0.0)
        dstd = 0.10 + 0.25 * min(speed, 1.5) / 1.5 + (0.22 if f.hand else 0.0) + abs(rng.gauss(0, 0.04))
        dabs = abs(dop) + 0.6 * dstd

        # 포인트 수 (가끔 품질 저하)
        if f.t > low_until and rng.random() < 0.004:
            low_until = f.t + rng.uniform(0.3, 1.2)
        n_points = rng.randint(3, 25) if f.t < low_until else rng.randint(80, 600)

        rows.append({
            "time_s": round(f.t + t_offset, 4), "target_id": tid,
            "x": round(x, 4), "y": round(y, 4), "status": 1 if f.crouch else 0,
            "n_points": n_points, "doppler_mean": round(dop, 4),
            "doppler_abs_mean": round(dabs, 4), "doppler_std": round(dstd, 4),
            "truth_motion": true_motion(frames, i), "truth_situation": f.situation,
            "truth_final": final,
        })
    return rows


# =========================================================================== 데이터셋

CSV_COLUMNS = ["session", "time_s", "frame_id", "target_id", "x", "y", "status", "n_points",
               "doppler_mean", "doppler_abs_mean", "doppler_std", "quality",
               "truth_motion", "truth_situation", "truth_final"]


def make_session(name: str, tracks: list[list[dict]]) -> list[dict]:
    """여러 사람의 행을 한 세션으로 합치고 프레임 번호를 붙인다."""
    rows = [dict(r, session=name) for tr in tracks for r in tr]
    rows.sort(key=lambda r: (r["time_s"], r["target_id"]))
    frame_of = {}
    for r in rows:
        r["frame_id"] = frame_of.setdefault(round(r["time_s"] * FPS), len(frame_of))
        r["quality"] = "low_points" if r["n_points"] < 30 else "ok"
    return rows


def training_episodes(n_per_kind: int = 30, seed: int = 0) -> list[tuple[str, list[dict]]]:
    """학습용: 상황마다 n개씩, 사람 한 명씩."""
    rng = random.Random(seed)
    out = []
    for kind in SITUATIONS + [IDLE_FAR]:
        for k in range(n_per_kind):
            fr = episode(kind, rng)
            final = UNDECIDED if kind == IDLE_FAR else kind
            rows = sense(fr, rng, target_id=1, id_switch=False, final=final)
            if len(rows) > 20:
                # 멀리 서 있는 사람은 녹화 끝까지 그대로 있음. 나머지는 사라진 뒤 녹화가 조금 더 이어짐
                out.append((f"{kind} #{k + 1}", rows if kind == IDLE_FAR else rows + [end_marker(rows)]))
    return out


def end_marker(rows: list[dict], after_s: float = 2.0) -> dict:
    """사람이 사라진 뒤에도 녹화가 조금 더 이어진 것처럼 하는 빈 행 (포인트 0개라 트랙으로는 안 씀)"""
    return {"time_s": rows[-1]["time_s"] + after_s, "target_id": -1, "x": 0.0, "y": 0.0, "status": 0,
            "n_points": 0, "doppler_mean": 0.0, "doppler_abs_mean": 0.0, "doppler_std": 0.0}


def demo_sessions(seed: int = 7) -> list[dict]:
    """화면 확인용: 상황마다 한 세션 + 여러 사람이 섞인 세션."""
    rng = random.Random(seed)
    rows = []
    for kind in SITUATIONS:
        fr = episode(kind, rng)
        rows += make_session(kind, [sense(fr, rng, target_id=rng.randint(1, 9), final=kind)])
    rows += make_session("멀리 서 있기 (해당 없음)",
                         [sense(episode(IDLE_FAR, rng), rng, target_id=5, final=UNDECIDED)])
    # 여러 명: 두 사람이 지나가는 사이 한 명이 방문
    a = sense(episode(PASS, rng), rng, target_id=2, t_offset=0.0, final=PASS)
    b = sense(episode(PASS, rng), rng, target_id=3, t_offset=1.5, final=PASS)
    c = sense(episode(VISIT, rng), rng, target_id=4, t_offset=3.0, final=VISIT)
    rows += make_session("여러 명 (통과 2 + 방문 1)", [a, b, c])
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="문 앞 상황 가짜 데이터(CSV) 만들기")
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", type=Path, default=here.parents[1] / "recordings" / "sim_demo_tracks_raw.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = demo_sessions(args.seed)
    write_csv(rows, args.out)
    print(f"저장: {args.out}  ({len(rows)}행, 세션 {len({r['session'] for r in rows})}개)")


if __name__ == "__main__":
    main()
