"""문 앞 상황 판정 - 특징 계산, 결정 트리 학습, 판정, 카메라 촬영 이벤트

세 개의 결정 트리를 씁니다.
  1) 움직임 트리 (프레임마다): 속도, 문 쪽 속도, 문 방향 비율, 도플러 → 다가옴 / 멀어짐 / 지나감 / 정지
  2) 상황 트리 (0.5초마다, 지금까지의 흐름): 머문 시간, 나타난 위치, 접근 횟수, 움직임 등
     → 통과 / 귀가 / 외출 / 방문 / 배달 / 장시간 체류 / 배회 / 반복 접근 / 도어락 조작 시도 / 판단 중
  3) 최종 판정 트리 (사람이 사라지거나 녹화가 끝난 순간): 상황 트리와 같은 특징으로, '판단 중' 없이
     어떤 상황이었는지만 고름 (진행 중 판정과 최종 판정을 한 트리로 같이 하면 최종 샘플이 너무 적어서 따로 둠)
  고른 상황이 상황 정의 조건(RULES)을 어기면 다음으로 확률 높은 상황을 고릅니다.
상황 트리가 '이상' 상황을 CAMERA_PROB 이상 확률로 CAMERA_HOLD_S 동안 유지하면 카메라 촬영 이벤트를 냅니다.

지금은 scenario_sim.py의 가짜 데이터로 학습합니다. 녹화에 정답을 붙이면 같은 방식으로 다시 학습하면 됩니다.
필요: numpy, scikit-learn
"""

from __future__ import annotations

import csv
import math
import pickle
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scenario_sim import (ABNORMAL, M_APPROACH, M_LEAVE, M_PASS, M_STILL,  # noqa: F401
                          MOTIONS, SITUATIONS, UNDECIDED, training_episodes)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "situation_trees.pkl"
MODEL_VERSION = 3          # 2: 트랙 잇기(멈춘 사람) + 특징 27개, 방문·배달 분리
                           # 3: 트랙 시작 직후(속도를 아직 못 구하는 구간)를 정지로 세던 버그 수정,
                           #    '문에서 나옴'을 처음 움직임 방향으로 판단, 특징 5개 추가(32개), 상황 정의 조건(RULES) 적용
M_UNKNOWN = "판단 전"      # 트랙 시작 직후 속도를 아직 구할 수 없는 구간 (정지·움직임 어느 쪽으로도 안 셈)

# --- 전처리·판정 상수
MIN_POINTS = 30          # 포인트가 이보다 적은 프레임은 버림
SMOOTH_S = 0.3           # 위치 이동평균
MOTION_WIN_S = 1.0       # 속도 계산 구간
CONFIRM_S = 0.5          # 움직임 판정이 이 시간 유지돼야 확정 (깜빡임 방지)
DOOR_R = 0.6             # 문 바로 앞
NEAR_R = 2.0             # 근처
STILL_NEAR_R = 1.5       # '머문 시간'을 세는 범위
LONG_STAY_S = 10.0       # 머문 시간 '오래' 기준 (표시용)
APPEAR_DOOR_R = 0.8      # 이 안에서 처음 잡히면 '문에서 나옴'
APPROACH_MIN_DR = 0.4    # 접근/이탈 1회로 세는 거리 변화
SNAP_STEP_S = 0.5        # 상황 트리 학습용 스냅샷 간격
CAMERA_PROB = 0.6        # 이상 상황 확률이 이 이상이고
CAMERA_HOLD_S = 1.0      # 이 시간 이상 유지되면 촬영
WATCH_R = 5.0            # 한 번도 이 거리 안에 들어오지 않은 트랙(멀리 있는 벽·물체 등)은 판정하지 않음

# --- 트랙 잇기 (레이더는 멈춘 사람을 놓침)
#   측정 영역 안에서 (외곽·문이 아닌 곳에서) 갑자기 사라짐  = 그 자리에 멈춰 있는 사람
#   측정 영역 안에서 (외곽·문이 아닌 곳에서) 갑자기 나타남  = 멈춰 있던 사람이 다시 움직임
#   사라진 자리와 나타난 자리가 가까우면 같은 사람으로 이어 붙이고, 그 사이는 '안 보이는 정지'로 채움
#   (다시 안 나타나면 녹화 끝까지 멈춰 있다고 봄. 단, 사라지기 직전에 거의 멈춰 있었을 때만)
#   외곽이 아닌 곳에서 끊겼다가 몇 초 안에 가까이서 다시 잡히면(문 앞 포함) 레이더가 다시 잡은 것으로 보고 이어 붙임
#   한 사람이 동시에 여러 ID로 잡히는 경우(시간이 겹치면서 계속 붙어 다님)도 한 사람으로 합침 (3개 이상도)
#   외곽으로 나간 뒤 REENTRY_MAX_GAP_S 안에 나간 자리 가까이(REENTRY_DIST)에서 다시 나타나면 같은 사람이 다시 들어온 것
#   (반복 접근: 나타났다 사라지기를 반복하는 한 사람). 나가 있던 동안은 프레임을 채우지 않음(정지로 세지 않음)
EDGE_MARGIN = 0.5        # 측정 영역 경계(좌우 끝, 먼 쪽 끝, 시야각 끝)에서 이 거리 안이면 '외곽'
DOOR_ENTER_R = 1.0       # 문(레이더)에서 이 거리 안에서 사라지면 '문으로 들어감' (나타나는 쪽은 아래 DOOR_EXIT_* 참고)
                         # 실제 귀가·방문 녹화는 레이더 0.36~0.73m에서 사라짐 (사람 몸 중심이 레이더에 닿기 전에 놓침).
                         # 0.5로 하면 귀가·방문 22개 중 16개가 '영역 안에서 사라짐'이 됨
STITCH_DIST = 1.0        # 사라진 자리와 나타난 자리가 이 거리 안이면 같은 사람
STITCH_MAX_GAP_S = 900.0 # 15분 넘게 안 보이면 다른 사람으로 봄
DOOR_RELINK_S = 300.0      # 외곽이 아닌 곳(문·영역 안)에서 끊겼다가 이 시간 안에 가까이서 다시 잡히면 같은 사람
DUP_DIST = 0.6           # 시간이 겹치는 두 트랙이 이 거리 안(중앙값)에 붙어 다니면 같은 사람이 여러 ID로 잡힌 것
DUP_MIN_OVERLAP_S = 1.0  # 그렇게 보려면 최소 이만큼 겹쳐야 함
HIDDEN_STEP_S = 0.2      # 안 보이는 정지 구간을 이 간격의 가짜 프레임으로 채움
REENTRY_MAX_GAP_S = 30.0 # 외곽으로 나갔다가 이 시간 안에
REENTRY_DIST = 1.2       # 나간 자리에서 이 거리 안에서 다시 나타나면 같은 사람
ENTRY_WINDOW_S = 60.0    # 최근 이 시간 동안 외곽에서 들어온 횟수 (반복 접근)
STOP_SPEED = 0.35        # 사라지기 직전 1초 동안 이보다 느렸으면 '멈춰서 놓친 것' (빠르면 그냥 트래킹 실패)
AWAY_R = 1.2             # 문에서 이 거리 밖에서 멈춰 있으면 '문에서 떨어져 정지' (엘리베이터 대기 등)
SIDE_CHANGE_DX = 1.0     # 처음 잡힌 곳과 x 부호가 반대이고 이만큼 옆으로 갔으면 '문 반대편으로 건너감' (통과)
OUT_MAX_APPROACH = 0.4   # 외출: 처음 잡힌 곳보다 이 이상 문에 가까워지면 외출이 아님
# '문에서 나옴' 판단. 레이더는 사람이 문에서 나와 1~2걸음 걸은 뒤에야 트랙을 만들어서(실제 녹화: 1.1~1.7m)
# 거리만으로는 문에서 나온 사람을 못 알아봄 → 처음 잡힌 뒤 짧은 시간 동안 레이더에서 곧장 멀어지면 문에서 나온 것
DOOR_EXIT_R = 1.75       # 처음 잡힌 곳이 이 거리 안이고
DOOR_EXIT_WIN_S = 0.6    # 처음 이 시간 동안
DOOR_EXIT_MIN_MOVE = 0.25  # 이만큼 이상 움직였는데
DOOR_EXIT_RADIAL = 0.6   # 움직임 중 레이더에서 곧장 멀어지는 비율이 이 이상이면 '문에서 나옴'

SIM_REGION = dict(xmin=-3.5, xmax=3.5, ymin=0.0, ymax=4.0, fov_deg=90.0, yaw_deg=0.0)  # 가짜 데이터용

MOTION_FEATURES = ["speed", "vr", "radial_ratio", "doppler_win"]
SITUATION_FEATURES = [
    # 시간·거리
    "elapsed", "appear_r", "cur_r", "min_r", "rise_from_min",
    # 정지
    "cur_still_s", "max_still_s", "door_still_s", "still_total_s", "still_away_s", "hidden_s",
    "near_door_s", "hand_still",
    # 움직임
    "moving_s", "n_approach", "n_leave", "path_len", "straightness", "lateral_frac", "spread",
    # 나타난 곳·사라진 곳
    "appear_edge_d", "cur_edge_d", "ended", "ended_at_door", "ended_at_edge",
    # 자세·주변
    "sit_frac", "edge_entries",
    # 방향 (처음 잡힌 곳 기준)
    "lateral_disp", "side_change", "net_dr", "approach_depth", "appear_door",
]
FEATURE_KO = {
    "speed": "속력(m/s)", "vr": "문 쪽 속도(m/s, -는 다가옴)", "radial_ratio": "문 방향 비율",
    "doppler_win": "도플러 평균",
    "elapsed": "잡힌 지 지난 시간(초)", "appear_r": "처음 잡힌 거리(m)", "cur_r": "현재 거리(m)",
    "min_r": "가장 가까웠던 거리(m)", "rise_from_min": "가장 가까웠던 뒤 멀어진 거리(m)",
    "cur_still_s": "지금 연속 정지 시간(초)", "max_still_s": "최장 연속 정지 시간(초)",
    "door_still_s": "문 바로 앞 정지 시간(초)", "hand_still": "정지 중 움직임(도플러 퍼짐)",
    "n_approach": "접근 횟수", "n_leave": "이탈 횟수", "path_len": "움직인 경로(m)",
    "straightness": "직진도", "lateral_frac": "옆으로 이동한 비율", "sit_frac": "앉음·숙임 비율",
    "ended": "사라짐(1=사라짐)",
    "still_total_s": "총 정지 시간(초, 안 보이는 정지 포함)", "still_away_s": "문에서 떨어져 정지한 시간(초)",
    "hidden_s": "안 보이는 정지 시간(초)", "near_door_s": "문 가까이(1.2m) 있던 시간(초)",
    "moving_s": "움직인 시간(초)", "spread": "머문 범위(m, 위치 퍼짐)",
    "appear_edge_d": "처음 잡힌 곳~외곽 거리(m, 작으면 외곽에서 들어옴)",
    "cur_edge_d": "현재 위치~외곽 거리(m)", "ended_at_door": "문으로 사라짐(1=예)",
    "ended_at_edge": "외곽으로 사라짐(1=예)", "edge_entries": f"최근 {int(ENTRY_WINDOW_S)}초 사이 나타난 횟수",
    "lateral_disp": "처음 잡힌 곳에서 옆(x)으로 간 거리(m)",
    "side_change": "문 반대편으로 건너감(1=예)",
    "net_dr": "처음 잡힌 곳보다 문에서 멀어진 거리(m, -는 가까워짐)",
    "approach_depth": "처음 잡힌 곳보다 문에 가까이 간 정도(m)",
    "appear_door": "문에서 나옴(1=예)",
}
OLD_FEATURES_V1 = ["elapsed", "appear_r", "cur_r", "min_r", "rise_from_min", "cur_still_s",
                   "max_still_s", "door_still_s", "hand_still", "n_approach", "n_leave",
                   "path_len", "straightness", "lateral_frac", "sit_frac", "ended"]
STATUS_KO = {0: "서 있음", 1: "앉음·숙임", 2: "누움", 4: "서 있음(걷는 중)"}


# =========================================================================== 데이터

@dataclass
class Track:
    session: str
    target_id: int
    t_abs: np.ndarray
    x: np.ndarray
    y: np.ndarray
    status: np.ndarray
    dop: np.ndarray
    dstd: np.ndarray
    truth_motion: list | None = None
    truth_situation: list | None = None
    truth_final: str | None = None
    hidden: np.ndarray | None = None         # True = 레이더가 놓쳤지만 멈춰 있다고 본 프레임
    target_ids: list = field(default_factory=list)   # 이어 붙인 레이더 ID들
    entry_times: list | None = None          # 이 사람이 (문이 아닌 곳에서) 나타난 시각들. 외곽으로 나갔다 다시 들어오면 늘어남
    session_end: float = math.inf            # 녹화(세션) 끝 시각
    # 계산 결과
    t: np.ndarray | None = None
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    r: np.ndarray | None = None
    mfeat: np.ndarray | None = None          # (N, 4) 움직임 트리 입력
    motion: np.ndarray | None = None         # 확정된 움직임
    sfeat: np.ndarray | None = None          # (N, F) 상황 트리 입력
    final_feat: np.ndarray | None = None
    sit_pred: np.ndarray | None = None
    sit_prob: np.ndarray | None = None
    final_pred: str = ""
    final_prob: float = 0.0
    cameras: list = field(default_factory=list)

    @property
    def t0(self) -> float:
        return float(self.t_abs[0])

    @property
    def t1(self) -> float:
        return float(self.t_abs[-1])


def _f(v, default=math.nan):
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def open_csv_text(path: str | Path):
    """UTF-8(BOM 있어도 됨)로 읽고, 안 되면 엑셀이 저장하는 CP949(한국어 Windows 기본)로 읽는다."""
    import io
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    return io.StringIO(text, newline="")


def read_csv_rows(path: str | Path) -> list[dict]:
    with open_csv_text(path) as f:
        rows = list(csv.DictReader(f))
    if rows and not {"time_s", "target_id", "x", "y"} <= set(rows[0]):
        raise ValueError("트랙 CSV가 아닙니다 (time_s, target_id, x, y 열 필요)")
    return rows


# =========================================================================== 측정 영역

PCR_HEADER = struct.Struct("<4s4sQ16fQQ32sQQ")
SPEC_FIELDS = ["hfov_deg", "vfov_deg", "sensor_width", "sensor_height", "pos_x", "pos_y", "pos_z",
               "yaw_deg", "pitch_deg", "range_min_x", "range_min_y", "range_min_z",
               "range_max_x", "range_max_y", "range_max_z", "range"]


def pcr_spec(path: str | Path) -> dict | None:
    """.pcr 헤더에 저장된 센서 설정 (Radar Studio의 Sensor Spec). 헤더만 읽어서 빠름."""
    try:
        with open(path, "rb") as f:
            h = PCR_HEADER.unpack(f.read(PCR_HEADER.size))
    except (OSError, struct.error):
        return None
    if h[0] != b"PCR1":
        return None
    return {k: float(v) for k, v in zip(SPEC_FIELDS, h[3:19])}


@dataclass
class Region:
    """측정 영역: 박스(x, y 범위) 안이면서 시야각 안. 레이더(=문) 기준 좌표, m / 도."""
    xmin: float = -2.0
    xmax: float = 2.0
    ymin: float = 0.0
    ymax: float = 3.0
    fov_deg: float = 135.0
    yaw_deg: float = 0.0

    @classmethod
    def from_spec(cls, spec: dict) -> "Region":
        return cls(spec["range_min_x"], spec["range_max_x"], spec["range_min_y"], spec["range_max_y"],
                   spec["hfov_deg"], spec["yaw_deg"])

    def contains(self, x, y):
        ang = np.degrees(np.arctan2(x, y)) - self.yaw_deg
        return ((x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)
                & (np.abs(ang) <= self.fov_deg / 2))

    def edge_dist(self, x, y):
        """외곽(좌우 끝, 먼 쪽 끝, 시야각 끝)까지의 거리. 문이 있는 벽 쪽(y 최소)은 외곽으로 보지 않음."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        ang = np.radians(np.abs(np.degrees(np.arctan2(x, y)) - self.yaw_deg))
        fov_d = r * np.sin(np.clip(np.radians(self.fov_deg / 2) - ang, 0, None))
        fov_d = np.where(r < DOOR_ENTER_R, np.inf, fov_d)   # 레이더(문) 바로 앞은 시야각이 좁아도 외곽이 아님
        return np.minimum.reduce([x - self.xmin, self.xmax - x, self.ymax - y, fov_d])

    def text(self) -> str:
        t = f"x {self.xmin:g}~{self.xmax:g}m, y {self.ymin:g}~{self.ymax:g}m, 시야각 {self.fov_deg:g}°"
        return t + (f", 방향 {self.yaw_deg:g}°" if self.yaw_deg else "")


def clip_to_region(tr: "Track", region: Region, min_frames: int = 10) -> bool:
    """트랙에서 측정 영역 안의 프레임만 남긴다. 남은 프레임이 너무 적으면 False (트랙은 그대로)."""
    m = region.contains(tr.x, tr.y)
    if m.all():
        return True
    if m.sum() < min_frames:
        return False
    # 같은 레이더 ID인데 영역 밖에 나갔다가 다시 들어온 시각 (반복 접근에서 레이더가 영역 밖까지 계속 잡는 경우)
    idx = np.where(m)[0]
    jumps = np.where(np.diff(idx) > 1)[0]
    tr.reentries = [float(tr.t_abs[idx[j + 1]]) for j in jumps
                    if tr.t_abs[idx[j + 1]] - tr.t_abs[idx[j]] >= 0.3]
    for name in ("t_abs", "x", "y", "status", "dop", "dstd", "hidden"):
        setattr(tr, name, getattr(tr, name)[m])
    if tr.truth_motion is not None:
        tr.truth_motion = [v for v, k in zip(tr.truth_motion, m) if k]
    if tr.truth_situation is not None:
        tr.truth_situation = [v for v, k in zip(tr.truth_situation, m) if k]
    return True


def rows_from_pcr(path: str | Path) -> list[dict]:
    """.pcr을 바로 읽어서 트랙 CSV와 같은 행 목록으로 만든다 (tools/pcr/to_numpy_gui.pyw 사용)."""
    import importlib.machinery
    import importlib.util
    import sys as _sys
    src = HERE.parent / "pcr" / "to_numpy_gui.pyw"
    if "to_numpy_gui" in _sys.modules:
        tn = _sys.modules["to_numpy_gui"]
    else:
        loader = importlib.machinery.SourceFileLoader("to_numpy_gui", str(src))
        spec = importlib.util.spec_from_loader("to_numpy_gui", loader)
        tn = importlib.util.module_from_spec(spec)
        _sys.modules["to_numpy_gui"] = tn
        loader.exec_module(tn)
    a = tn.pcr_to_arrays(path, with_points=False)
    names = {int(i): str(n) for i, n in zip(a["session_ids"], a["session_names"])}
    tg, dop = a["targets"], a["target_doppler"]
    return [{
        "session": names.get(int(sid), str(sid)), "time_s": t[0], "frame_id": int(t[1]),
        "target_id": int(t[2]), "x": t[3], "y": t[4], "status": int(t[5]), "n_points": int(n),
        "doppler_mean": d[0], "doppler_std": d[2],
    } for t, sid, n, d in zip(tg, a["target_session"], a["target_n_points"], dop)]


def load_rows(path: str | Path) -> list[dict]:
    """트랙 CSV 또는 .pcr"""
    return rows_from_pcr(path) if str(path).lower().endswith(".pcr") else read_csv_rows(path)


def load_models(path: str | Path) -> "Models":
    with open(path, "rb") as f:
        m = pickle.load(f)
    if not hasattr(m, "situation"):
        raise ValueError("결정 트리 모델 파일이 아닙니다")
    check_model(m)
    return m


def check_model(m) -> None:
    """특징 목록이 지금 코드와 같은지 확인 (예전 특징으로 학습한 모델은 다시 학습해야 함)"""
    feats = getattr(m, "feature_names", OLD_FEATURES_V1)
    if list(feats) == SITUATION_FEATURES and getattr(m, "version", 0) < MODEL_VERSION:
        raise ValueError("이 모델은 특징 계산이 고쳐지기 전 코드로 학습되었습니다 "
                         "(트랙 시작 구간을 정지로 세던 버그). 정답 표로 다시 학습해주세요.")
    if list(feats) != SITUATION_FEATURES:
        raise ValueError(f"이 모델은 예전 특징({len(feats)}개)으로 학습되어 지금 코드(특징 {len(SITUATION_FEATURES)}개)와 "
                         "맞지 않습니다. 정답 표로 다시 학습해주세요.")


def tracks_from_rows(rows: list[dict]) -> list[Track]:
    """(세션, target_id)마다 하나의 트랙. 포인트가 적은 행은 버린다."""
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        npts = _f(r.get("n_points"), 999)
        if npts < MIN_POINTS:
            continue
        key = (r.get("session") or "Session #0", int(float(r["target_id"])))
        groups.setdefault(key, []).append(r)
    ends: dict = {}
    for r in rows:
        k = r.get("session") or "Session #0"
        ends[k] = max(ends.get(k, -math.inf), _f(r["time_s"], -math.inf))
    tracks = []
    for (session, tid), rs in groups.items():
        rs.sort(key=lambda r: _f(r["time_s"]))
        if len(rs) < 10:
            continue
        has_truth = "truth_motion" in rs[0] and rs[0].get("truth_motion")
        tracks.append(Track(
            session=session, target_id=tid,
            t_abs=np.array([_f(r["time_s"]) for r in rs]),
            x=np.array([_f(r["x"]) for r in rs]), y=np.array([_f(r["y"]) for r in rs]),
            status=np.array([int(_f(r.get("status"), 0)) for r in rs]),
            dop=np.array([_f(r.get("doppler_mean")) for r in rs]),
            dstd=np.array([_f(r.get("doppler_std")) for r in rs]),
            truth_motion=[r["truth_motion"] for r in rs] if has_truth else None,
            truth_situation=[r.get("truth_situation", "") for r in rs] if has_truth else None,
            truth_final=rs[0].get("truth_final") or None,
            hidden=np.zeros(len(rs), dtype=bool), target_ids=[tid], session_end=ends.get(session, math.inf),
        ))
    return tracks


# =========================================================================== 특징 계산

def _trailing_mean(t: np.ndarray, v: np.ndarray, win: float) -> np.ndarray:
    """최근 win초 평균 (NaN은 빼고). 앞쪽 데이터만 쓰므로 실시간에서도 같은 값이 나옴."""
    ok = ~np.isnan(v)
    i0 = np.searchsorted(t, t - win, side="left")
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(1, len(t) + 1)
    n = cn[idx] - cn[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[idx] - cs[i0]) / np.maximum(n, 1), np.nan)


def _trailing_slope(t: np.ndarray, v: np.ndarray, win: float) -> tuple[np.ndarray, np.ndarray]:
    """최근 win초 최소제곱 기울기와 유효 여부."""
    i0 = np.searchsorted(t, t - win, side="left")
    idx = np.arange(1, len(t) + 1)
    pre = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    St, Stt, Sv, Stv = pre(t), pre(t * t), pre(v), pre(t * v)
    n = (idx - i0).astype(float)
    st, stt, sv, stv = St[idx] - St[i0], Stt[idx] - Stt[i0], Sv[idx] - Sv[i0], Stv[idx] - Stv[i0]
    den = n * stt - st * st
    span = t - t[i0]
    ok = (n >= 4) & (den > 1e-9) & (span >= 0.6 * win)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(ok, (n * stv - st * sv) / np.where(ok, den, 1.0), 0.0)
    return slope, ok


def motion_features(tr: Track) -> None:
    t = tr.t_abs - tr.t_abs[0]
    tr.t = t
    tr.xs = _trailing_mean(t, tr.x, SMOOTH_S)
    tr.ys = _trailing_mean(t, tr.y, SMOOTH_S)
    tr.r = np.hypot(tr.xs, tr.ys)
    vx, ok = _trailing_slope(t, tr.xs, MOTION_WIN_S)
    vy, _ = _trailing_slope(t, tr.ys, MOTION_WIN_S)
    vr, _ = _trailing_slope(t, tr.r, MOTION_WIN_S)
    speed = np.hypot(vx, vy)
    ratio = np.where(speed > 1e-6, np.abs(vr) / np.maximum(speed, 1e-6), 0.0)
    dw = np.nan_to_num(_trailing_mean(t, tr.dop, MOTION_WIN_S), nan=0.0)
    tr.mfeat = np.stack([speed, vr, ratio, dw], axis=1)
    tr._dstd_win = np.nan_to_num(_trailing_mean(t, tr.dstd, MOTION_WIN_S), nan=0.2)
    tr._motion_ok = ok


def confirm_motion(t: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """새 판정이 CONFIRM_S 이상 유지될 때만 바꾼다 (깜빡임 방지).
    속도를 처음 구할 수 있게 된 순간의 판정은 바로 쓴다 (그 전은 '판단 전')."""
    out = np.empty(len(raw), dtype=object)
    cur, cand, cand_t = None, None, 0.0
    for i, m in enumerate(raw):
        if cur is None:
            if m != M_UNKNOWN:
                cur = m
            out[i] = cur if cur is not None else M_UNKNOWN
            continue
        if m == M_UNKNOWN:
            out[i] = cur
            continue
        if m == cur:
            cand = None
        elif m != cand:
            cand, cand_t = m, t[i]
        elif t[i] - cand_t >= CONFIRM_S:
            cur, cand = m, None
        out[i] = cur
    return out


def situation_features(tr: Track, region: "Region | None" = None, entry_times=None) -> None:
    """프레임마다 '지금까지의 흐름' 특징. 마지막에 사라진 시점의 특징(ended=1 등)도 만든다.
    region이 있으면 외곽까지 거리를 계산하고, entry_times(같은 세션에서 외곽으로 들어온 시각들)로
    반복 접근 특징을 만든다."""
    t, r, xs, ys, m = tr.t, tr.r, tr.xs, tr.ys, tr.motion
    hid = tr.hidden if tr.hidden is not None else np.zeros(len(t), dtype=bool)
    n = len(t)
    edge = region.edge_dist(xs, ys) if region is not None else np.full(n, 9.0)
    if entry_times is not None and len(entry_times):
        et = np.sort(np.asarray(entry_times))
        entries = (np.searchsorted(et, tr.t_abs, side="right")
                   - np.searchsorted(et, tr.t_abs - ENTRY_WINDOW_S, side="right")).astype(float)
    else:
        entries = np.zeros(n)
    F = np.zeros((n, len(SITUATION_FEATURES)))
    min_r, cur_still, max_still, door_still = math.inf, 0.0, 0.0, 0.0
    still_total = still_away = hidden_s = near_door = moving = 0.0
    hand_sum, hand_n, n_app, n_lv = 0.0, 0, 0, 0
    seg, seg_r0, seg_done = None, 0.0, False
    path, last_t, lx, ly = 0.0, -math.inf, 0.0, 0.0
    lat_n = sit_n = 0
    sx = sy = sxx = syy = 0.0
    r0, e0 = r[0], edge[0]
    hx = float(np.mean(xs[:5]))                       # 처음 잡힌 곳 (x)
    from_door = float(_head_kind(tr, region) == "door")
    for i in range(n):
        dt = min(t[i] - t[i - 1], 0.5) if i else 0.0
        mi = m[i]
        min_r = min(min_r, r[i])
        still = mi == M_STILL
        if still:
            still_total += dt
            if r[i] >= AWAY_R:
                still_away += dt
        elif mi != M_UNKNOWN:
            moving += dt
        if hid[i]:
            hidden_s += dt
        if r[i] < AWAY_R:
            near_door += dt
        if still and r[i] < STILL_NEAR_R:
            cur_still += dt
            if not hid[i]:
                hand_sum += tr._dstd_win[i]
                hand_n += 1
        else:
            cur_still = 0.0
        max_still = max(max_still, cur_still)
        if still and r[i] < DOOR_R:
            door_still += dt
        if mi != seg:
            seg, seg_r0, seg_done = mi, r[i], False
        if not seg_done:
            if mi == M_APPROACH and seg_r0 - r[i] >= APPROACH_MIN_DR:
                n_app, seg_done = n_app + 1, True
            elif mi == M_LEAVE and r[i] - seg_r0 >= APPROACH_MIN_DR:
                n_lv, seg_done = n_lv + 1, True
        if t[i] - last_t >= 1.0:
            if last_t > -math.inf:
                path += math.hypot(xs[i] - lx, ys[i] - ly)
            last_t, lx, ly = t[i], xs[i], ys[i]
        net = math.hypot(xs[i] - xs[0], ys[i] - ys[0])
        lat_n += mi == M_PASS
        sit_n += tr.status[i] in (1, 2)
        sx += xs[i]; sy += ys[i]; sxx += xs[i] ** 2; syy += ys[i] ** 2  # noqa: E702
        k = i + 1
        spread = math.sqrt(max(sxx / k - (sx / k) ** 2, 0) + max(syy / k - (sy / k) ** 2, 0))
        F[i] = [t[i], r0, r[i], min_r, r[i] - min_r,
                cur_still, max_still, door_still, still_total, still_away, hidden_s, near_door,
                hand_sum / hand_n if hand_n else 0.0,
                moving, n_app, n_lv, path, net / path if path > 0.5 else 1.0, lat_n / k, spread,
                e0, edge[i], 0.0, 0.0, 0.0,
                sit_n / k, entries[i],
                abs(xs[i] - hx), float(hx * xs[i] < 0 and abs(xs[i] - hx) >= SIDE_CHANGE_DX),
                r[i] - r0, r0 - min_r, from_door]
    tr.sfeat = F
    fin = F[-1].copy()
    # 녹화가 끝나기 전에 사라졌을 때만 '사라짐' (끝까지 보이거나 멈춘 채 남아 있으면 아님)
    gone = bool(tr.t_abs[-1] < tr.session_end - 0.5)
    ix = SITUATION_FEATURES.index
    fin[ix("ended")] = float(gone)
    end_kind = _end_kind(*_tail(tr), region)          # 문이 외곽보다 우선
    fin[ix("ended_at_door")] = float(gone and end_kind == "door")
    fin[ix("ended_at_edge")] = float(gone and end_kind == "edge")
    tr.final_feat = fin


# =========================================================================== 트랙 잇기

def _end_kind(x, y, region: "Region | None") -> str:
    """사라지거나 나타난 위치의 종류: door(문) / edge(외곽) / inside(영역 안)"""
    if math.hypot(x, y) < DOOR_ENTER_R:
        return "door"
    if region is not None and float(region.edge_dist(x, y)) < EDGE_MARGIN:
        return "edge"
    return "inside"


def _tail(tr: Track, k: int = 5):
    return float(np.mean(tr.x[-k:])), float(np.mean(tr.y[-k:]))


def _head(tr: Track, k: int = 5):
    return float(np.mean(tr.x[:k])), float(np.mean(tr.y[:k]))


def _head_kind(tr: Track, region: "Region | None") -> str:
    """처음 나타난 곳의 종류: door(문에서 나옴) / edge(외곽에서 들어옴) / inside(영역 안에서 나타남).
    문 바로 앞(DOOR_ENTER_R 안)에서 처음 잡혔거나, 조금 떨어진 곳(DOOR_EXIT_R 안)에서 잡혔어도
    처음 DOOR_EXIT_WIN_S초 동안 레이더에서 곧장 멀어지면 문에서 나온 것으로 본다."""
    hx, hy = _head(tr)
    kind = _end_kind(hx, hy, region)
    if kind == "door":
        return kind
    r0 = math.hypot(float(tr.x[0]), float(tr.y[0]))     # 처음 잡힌 순간 (빨리 멀어지므로 평균 말고 첫 프레임)
    if r0 < DOOR_EXIT_R:
        t = tr.t_abs - tr.t_abs[0]
        vis = ~tr.hidden if tr.hidden is not None else np.ones(len(t), dtype=bool)
        k = np.where(vis & (t <= DOOR_EXIT_WIN_S))[0]
        if len(k) >= 3:
            x, y = tr.x[k], tr.y[k]
            move = math.hypot(x[-1] - x[0], y[-1] - y[0])
            dr = math.hypot(x[-1], y[-1]) - math.hypot(x[0], y[0])
            if move >= DOOR_EXIT_MIN_MOVE and dr / move >= DOOR_EXIT_RADIAL:
                return "door"
    return kind


def _was_stopping(tr: Track, win: float = 1.0) -> bool:
    """사라지기 직전 win초 동안 거의 안 움직였는지 (레이더는 멈춘 사람을 놓침)"""
    t = tr.t_abs
    i0 = int(np.searchsorted(t, t[-1] - win))
    if len(t) - i0 < 4 or t[-1] - t[i0] < 0.5:
        return True
    k = max(1, (len(t) - i0) // 3)
    x0, y0 = np.mean(tr.x[i0:i0 + k]), np.mean(tr.y[i0:i0 + k])
    x1, y1 = np.mean(tr.x[-k:]), np.mean(tr.y[-k:])
    return math.hypot(x1 - x0, y1 - y0) / max(t[-1] - t[i0], 1e-6) < STOP_SPEED


def _hidden_block(tr: Track, t0: float, t1: float, x: float, y: float):
    """t0~t1 사이를 '안 보이는 정지' 프레임으로 채울 배열들"""
    ts = np.arange(t0 + HIDDEN_STEP_S, t1, HIDDEN_STEP_S)
    n = len(ts)
    return dict(t_abs=ts, x=np.full(n, x), y=np.full(n, y), status=np.full(n, tr.status[-1]),
                dop=np.zeros(n), dstd=np.full(n, np.nan), hidden=np.ones(n, dtype=bool))


def _concat(a: Track, block: dict | None, b: Track | None) -> None:
    """a 뒤에 block(안 보이는 구간)과 b를 이어 붙인다 (a를 직접 바꿈)."""
    has_block = bool(block) and len(block["t_abs"]) > 0
    parts = [block] if has_block else []
    keep = None
    if b is not None:
        keep = b.t_abs > a.t_abs[-1]
        parts.append(dict(t_abs=b.t_abs[keep], x=b.x[keep], y=b.y[keep], status=b.status[keep],
                          dop=b.dop[keep], dstd=b.dstd[keep], hidden=b.hidden[keep]))
    for name in ("t_abs", "x", "y", "status", "dop", "dstd", "hidden"):
        setattr(a, name, np.concatenate([getattr(a, name)] + [p[name] for p in parts]))
    for lab in ("truth_motion", "truth_situation"):
        la = getattr(a, lab)
        if la is None:
            continue
        add = []
        if has_block:
            add += [M_STILL if lab == "truth_motion" else la[-1]] * len(block["t_abs"])
        if b is not None and getattr(b, lab) is not None:
            add += [v for v, k in zip(getattr(b, lab), keep) if k]
        setattr(a, lab, la + add)
    if b is not None:
        a.target_ids = a.target_ids + [i for i in b.target_ids if i not in a.target_ids]


def _link_sequential(trs: list[Track], region: "Region | None") -> list[Track]:
    """끊겼다가 다시 잡힌 같은 사람 잇기 (시간 순서대로, 몇 조각이든)"""
    people: list[Track] = []
    for tr in sorted(trs, key=lambda t: t.t_abs[0]):
        if tr.entry_times is None:
            tr.entry_times = ([] if _head_kind(tr, region) == "door" else [float(tr.t_abs[0])]) \
                + list(getattr(tr, "reentries", []))
        hx, hy = _head(tr)
        # 잇기는 위치로만 판단 (멈췄던 사람이 문 반대쪽으로 다시 걸어가는 것도 '문에서 나옴'처럼 보일 수 있음)
        head_kind = _end_kind(hx, hy, region)
        best, best_d, best_re = None, math.inf, False
        for p in people:
            tx, ty = _tail(p)
            tail_kind = _end_kind(tx, ty, region)
            gap = tr.t_abs[0] - p.t_abs[-1]
            if gap < -0.5:
                continue
            d = math.hypot(hx - tx, hy - ty)
            if tail_kind == "edge":
                # 외곽으로 나갔던 사람이 그 근처로 다시 들어옴 (문에서 나온 사람은 아님)
                ok = head_kind != "door" and gap <= REENTRY_MAX_GAP_S and d <= REENTRY_DIST
                re = True
            elif head_kind == "edge":
                ok, re = False, False                  # 외곽에서 새로 들어온 사람
            elif head_kind == "inside" and tail_kind == "inside":
                ok, re = gap <= STITCH_MAX_GAP_S and d <= STITCH_DIST, False   # 영역 안에서 멈췄다가 다시 움직임
            else:
                ok, re = gap <= DOOR_RELINK_S and d <= STITCH_DIST, False      # 문 앞 등에서 잠깐 놓쳤다가 다시 잡음
            if ok and d < best_d:
                best, best_d, best_re = p, d, re
        if best is None:
            people.append(tr)
            continue
        tx, ty = _tail(best)
        gap = tr.t_abs[0] - best.t_abs[-1]
        if best_re:
            # 외곽 밖에 나가 있던 동안은 비워 둠 (정지로 세지 않음). 다시 들어온 시각을 기록
            best.entry_times = (best.entry_times or []) + [float(tr.t_abs[0])]
            _concat(best, None, tr)
            continue
        block = _hidden_block(best, best.t_abs[-1], tr.t_abs[0], tx, ty) if gap > 2 * HIDDEN_STEP_S else None
        _concat(best, block, tr)
    return people


def _overlap_dist(a: Track, b: Track) -> float | None:
    """두 트랙이 겹치는 동안의 거리 중앙값 (충분히 안 겹치면 None)"""
    lo, hi = max(a.t_abs[0], b.t_abs[0]), min(a.t_abs[-1], b.t_abs[-1])
    if hi - lo < DUP_MIN_OVERLAP_S:
        return None
    ts = np.arange(lo, hi, 0.2)
    d = np.hypot(np.interp(ts, a.t_abs, a.x) - np.interp(ts, b.t_abs, b.x),
                 np.interp(ts, a.t_abs, a.y) - np.interp(ts, b.t_abs, b.y))
    return float(np.median(d))


def _near(t_query: np.ndarray, t_ref: np.ndarray, tol: float = 0.3) -> np.ndarray:
    """t_query의 각 시각 근처(tol초 안)에 t_ref 시각이 있는지"""
    if len(t_ref) == 0:
        return np.zeros(len(t_query), dtype=bool)
    i = np.clip(np.searchsorted(t_ref, t_query), 1, len(t_ref) - 1) if len(t_ref) > 1 else np.zeros(len(t_query), int)
    d = np.minimum(np.abs(t_ref[i] - t_query), np.abs(t_ref[np.maximum(i - 1, 0)] - t_query))
    return d <= tol


def _merge_dup(a: Track, b: Track) -> None:
    """같은 사람이 동시에 두 ID로 잡힌 경우 a에 b를 합친다. 겹치는 시간은 a의 보이는 프레임을 쓰고,
    a가 놓친(안 보이는) 구간에 b가 보이면 b를 쓴다."""
    a_vis = a.t_abs[~a.hidden]
    keep_b = ~_near(b.t_abs, a_vis)
    b_vis = b.t_abs[keep_b & ~b.hidden]
    keep_a = ~(a.hidden & _near(a.t_abs, b_vis))
    names = ("t_abs", "x", "y", "status", "dop", "dstd", "hidden")
    merged = {n: np.concatenate([getattr(a, n)[keep_a], getattr(b, n)[keep_b]]) for n in names}
    order = np.argsort(merged["t_abs"], kind="stable")
    for n in names:
        setattr(a, n, merged[n][order])
    for lab in ("truth_motion", "truth_situation"):
        la, lb = getattr(a, lab), getattr(b, lab)
        if la is not None and lb is not None:
            comb = [v for v, k in zip(la, keep_a) if k] + [v for v, k in zip(lb, keep_b) if k]
            setattr(a, lab, [comb[i] for i in order])
    a.target_ids = a.target_ids + [i for i in b.target_ids if i not in a.target_ids]


def _merge_duplicates(people: list[Track]) -> tuple[list[Track], bool]:
    changed = False
    i = 0
    while i < len(people):
        j = i + 1
        while j < len(people):
            d = _overlap_dist(people[i], people[j])
            if d is not None and d <= DUP_DIST:
                a, b = people[i], people[j]
                if len(b.t_abs) > len(a.t_abs):        # 더 오래 잡힌 쪽을 기준으로
                    a, b = b, a
                    people[i] = a
                _merge_dup(a, b)
                del people[j]
                changed = True
            else:
                j += 1
        i += 1
    return people, changed


def stitch(tracks: list[Track], region: "Region | None") -> list[Track]:
    """같은 세션의 트랙들에서 같은 사람을 하나로 정리한다.
    1) 끊겼다가 다시 잡힌 조각 잇기  2) 동시에 여러 ID로 잡힌 것 합치기  (바뀌지 않을 때까지 반복)
    3) 영역 안에서 멈춘 채 사라지고 다시 안 나타난 사람은 녹화 끝까지 그 자리에 멈춰 있는 것으로 채움"""
    people = [copy_track(t) for t in tracks]
    for _ in range(5):
        people = _link_sequential(people, region)
        people, changed = _merge_duplicates(people)
        if not changed:
            break
    people.sort(key=lambda t: t.t_abs[0])
    # 영역 안에서 멈춘 채 사라지고 다시 안 나타난 사람: 녹화 끝까지 그 자리에 멈춰 있음
    for p in people:
        tx, ty = _tail(p)
        end = min(p.session_end, p.t_abs[-1] + STITCH_MAX_GAP_S)
        if (_end_kind(tx, ty, region) == "inside" and _was_stopping(p)
                and end - p.t_abs[-1] > 2 * HIDDEN_STEP_S):
            _concat(p, _hidden_block(p, p.t_abs[-1], end, tx, ty), None)
    return people


def copy_track(tr: Track) -> Track:
    """원본 배열을 건드리지 않도록 복사 (이어 붙이기는 복사본에 함)"""
    import copy as _copy
    c = _copy.copy(tr)
    for name in ("t_abs", "x", "y", "status", "dop", "dstd", "hidden"):
        setattr(c, name, np.array(getattr(tr, name)))
    c.target_ids = list(tr.target_ids or [tr.target_id])
    c.entry_times = None if tr.entry_times is None else list(tr.entry_times)
    if tr.truth_motion is not None:
        c.truth_motion = list(tr.truth_motion)
    if tr.truth_situation is not None:
        c.truth_situation = list(tr.truth_situation)
    return c


def assemble(tracks: list[Track], region: "Region | None", motion_tree) -> tuple[list[Track], list[Track]]:
    """판정·학습 공통 전처리: 영역으로 자르기 → 세션마다 같은 사람 잇기 → 특징 계산.
    (영역 안 사람들, 영역 밖 트랙들)을 돌려준다. 원본 트랙은 바꾸지 않는다."""
    inside, outside = [], []
    by_session: "OrderedDict[str, list]" = OrderedDict()
    for tr in tracks:
        c = copy_track(tr)
        if region is not None and not clip_to_region(c, region):
            outside.append(c)
        else:
            by_session.setdefault(c.session, []).append(c)
    for session, trs in by_session.items():
        people = stitch(trs, region)
        # 문이 아닌 곳에서 나타난 횟수 (새 사람 + 외곽으로 나갔다 다시 들어온 사람). 영역 안에서 멈췄다
        # 다시 움직인 경우는 이미 이어 붙였으므로 세지 않음. 레이더는 걷기 시작하고 조금 지나서 잡기 때문에
        # 외곽보다 안쪽에서 처음 잡히는 경우가 많아서 외곽만 세지 않음
        entries = sorted(t for p in people for t in (p.entry_times or []))
        for p in people:
            _prepare(p, motion_tree, region, entries)
        inside += people
    for c in outside:
        _prepare(c, motion_tree, region, None)
    return inside, outside


# =========================================================================== 학습

@dataclass
class Models:
    motion: object
    situation: object
    report: str
    version: int = MODEL_VERSION
    final: object = None      # 최종 판정 트리 (없으면 상황 트리로 최종 판정)


def _prepare(tr: Track, motion_tree=None, region=None, entry_times=None) -> None:
    motion_features(tr)
    if motion_tree is not None:
        raw = motion_tree.predict(tr.mfeat).astype(object)
        raw = np.where(tr._motion_ok, raw, M_UNKNOWN)     # 속도를 아직 못 구한 구간은 판단 전
        if tr.hidden is not None:
            raw = np.where(tr.hidden, M_STILL, raw)      # 안 보이는 정지 구간은 정지
        tr.motion = confirm_motion(tr.t, raw)
        situation_features(tr, region, entry_times)


def snap_index(tr: Track) -> np.ndarray:
    """0.5초 간격 스냅샷의 프레임 번호"""
    idx, next_t = [], 0.0
    for i in range(len(tr.t)):
        if tr.t[i] >= next_t:
            idx.append(i)
            next_t = tr.t[i] + SNAP_STEP_S
    return np.array(idx, dtype=int)


def snap_truth(tr: Track, i: int, rules: bool = True) -> str:
    """i번째 프레임의 학습용 정답 (정의 조건을 아직 못 채웠으면 판단 중)"""
    lab = tr.truth_situation[i]
    if rules and lab != UNDECIDED and rule_violations(lab, tr.sfeat[i], False):
        return UNDECIDED
    return lab


def final_ok(tr: Track) -> list[str]:
    """최종 정답이 어기는 정의 조건 (비어 있으면 괜찮음)"""
    return rule_violations(tr.truth_final, tr.final_feat, True)


def _snapshots(tr: Track, rules: bool = True, final: bool = True):
    """상황 트리 학습용: 0.5초마다 하나 + 사라진 시점 (3배 가중).
    rules=True면 정의 조건(RULES)을 아직 못 채운 순간의 정답은 '판단 중'으로 바꾼다
    (예: 반복 접근인데 아직 한 번밖에 안 나타남). 최종 정답이 조건에 안 맞으면 최종 샘플은 넣지 않는다."""
    idx = snap_index(tr)
    X = [tr.sfeat[i] for i in idx]
    y = [snap_truth(tr, i, rules) for i in idx]
    if final and not (rules and final_ok(tr)):
        for _ in range(3):
            X.append(tr.final_feat)
            y.append(tr.truth_final)
    return X, y


FINAL_REP = 5            # 최종 판정 트리 학습에서 사라진 시점 샘플의 가중치(반복 횟수)


def final_samples(tr: Track, rep: int = FINAL_REP):
    """최종 판정 트리 학습용: 정답이 정해진 뒤의 0.5초 스냅샷 + 사라진 시점(rep배). '판단 중' 스냅샷은 안 씀."""
    X, y = [], []
    for i in snap_index(tr):
        lab = snap_truth(tr, i)
        if lab != UNDECIDED:
            X.append(tr.sfeat[i])
            y.append(lab)
    if not final_ok(tr):
        X += [tr.final_feat] * rep
        y += [tr.truth_final] * rep
    return X, y


def fit_final(people: list[Track], depth: int, min_leaf: int):
    from sklearn.tree import DecisionTreeClassifier
    X, y = [], []
    for p in people:
        a, b = final_samples(p)
        X += a
        y += b
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                  class_weight="balanced", random_state=0)
    tree.fit(np.array(X), y)
    return tree


def _fit(episodes: list[list[Track]]):
    """episodes: 세션(에피소드)마다 트랙 목록"""
    from sklearn.tree import DecisionTreeClassifier
    tracks = [tr for ep in episodes for tr in ep]
    mX, my = [], []
    for tr in tracks:
        tr = copy_track(tr)
        _prepare(tr)
        sel = np.where(tr._motion_ok)[0][::2]
        mX.append(tr.mfeat[sel])
        my += [tr.truth_motion[i] for i in sel]
    motion_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40,
                                         class_weight="balanced", random_state=0)
    motion_tree.fit(np.vstack(mX), my)
    sX, sy = [], []
    region = Region(**SIM_REGION)
    people = []
    for ep in episodes:
        for p in assemble(ep, region, motion_tree)[0]:
            p.truth_final = ep[0].truth_final
            X, y = _snapshots(p)
            sX += X
            sy += y
            people.append(p)
    sit_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10,
                                      class_weight="balanced", random_state=0)
    sit_tree.fit(np.array(sX), sy)
    return motion_tree, sit_tree, fit_final(people, 6, 10)


def train(n_per_kind: int = 30, seed: int = 0, on_status=None) -> Models:
    """가짜 데이터로 학습. 에피소드의 1/4은 검증용으로 빼서 정확도를 재고, 최종 모델은 전체로 다시 학습."""
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import export_text
    say = on_status or (lambda s: None)
    say("가짜 시나리오 만드는 중...")
    eps = training_episodes(n_per_kind, seed)
    episodes = []
    for name, rows in eps:
        trs = tracks_from_rows([dict(r, session=name) for r in rows])
        if trs:
            episodes.append(trs)
    train_trs = [ep for k, ep in enumerate(episodes) if k % 4 != 0]
    test_trs = [ep for k, ep in enumerate(episodes) if k % 4 == 0]

    say("결정 트리 학습 중 (검증용)...")
    mt, st, ft = _fit(train_trs)
    # 검증
    m_true, m_pred, s_true, s_pred, f_true, f_pred = [], [], [], [], [], []
    region = Region(**SIM_REGION)
    vm = Models(mt, st, "")
    vm.final = ft
    vm.feature_names = list(SITUATION_FEATURES)
    for ep in test_trs:
        people = assemble(ep, region, mt)[0]
        for tr in people:
            tr.truth_final = ep[0].truth_final
            tr.out_of_range = False
        predict(people, vm)                    # 정의 조건까지 적용한 판정으로 검증
        for tr in people:
            ok = tr._motion_ok & ~tr.hidden
            m_true += [tr.truth_motion[i] for i in np.where(ok)[0]]
            m_pred += list(mt.predict(tr.mfeat[ok]))
            idx = snap_index(tr)
            s_true += [snap_truth(tr, i) for i in idx]
            s_pred += [str(tr.sit_pred[i]) for i in idx]
            if final_ok(tr):                   # 정답이 정의에 안 맞는 경우(반복 접근의 첫 등장 등)는 최종 검증에서 뺌
                continue
            f_true.append(tr.truth_final)
            f_pred.append(tr.final_pred)
    acc = lambda a, b: np.mean(np.array(a) == np.array(b)) if a else 0.0  # noqa: E731

    say("결정 트리 학습 중 (전체)...")
    mt, st, ft = _fit(episodes)

    labels = SITUATIONS + [UNDECIDED]
    cm = confusion_matrix(f_true, f_pred, labels=labels)
    lines = [
        "※ 가짜 데이터(scenario_sim.py)로 학습·검증한 결과입니다. 실제 성능은 녹화로 다시 확인해야 합니다.",
        f"학습 에피소드 {len(train_trs)}개 / 검증 에피소드 {len(test_trs)}개 (상황마다 {n_per_kind}개 생성, "
        f"'멀리 서 있기'는 해당 없음 예시로 정답 '판단 중')",
        "",
        f"[검증] 움직임 트리 프레임 정확도: {acc(m_true, m_pred):.1%}",
        f"[검증] 상황 트리 0.5초 스냅샷 정확도 ('판단 중' 포함): {acc(s_true, s_pred):.1%}",
        f"[검증] 상황 트리 최종 판정 정확도 (사라진 시점): {acc(f_true, f_pred):.1%}",
        "",
        "최종 판정 혼동 행렬 (행: 정답, 열: 예측)",
        "        " + " ".join(f"{s[:4]:>5}" for s in labels),
    ]
    for s, row in zip(labels, cm):
        lines.append(f"{s[:6]:<8}" + " ".join(f"{v:>5d}" for v in row))
    ko = lambda names: [FEATURE_KO[n] for n in names]  # noqa: E731
    lines += ["", "=" * 60, "움직임 트리 (프레임마다)", "=" * 60,
              export_text(mt, feature_names=ko(MOTION_FEATURES), decimals=2),
              "=" * 60, "상황 트리 (0.5초마다, 지금까지의 흐름)", "=" * 60,
              export_text(st, feature_names=ko(SITUATION_FEATURES), decimals=2),
              "=" * 60, "최종 판정 트리 (사라진 순간)", "=" * 60,
              export_text(ft, feature_names=ko(SITUATION_FEATURES), decimals=2)]
    m = Models(mt, st, "\n".join(lines))
    m.final = ft
    m.feature_names = list(SITUATION_FEATURES)
    m.region = dict(SIM_REGION)
    m.watch_r = math.inf
    return m


def load_or_train(force: bool = False, on_status=None) -> Models:
    if not force and MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                m = pickle.load(f)
            if getattr(m, "version", 0) == MODEL_VERSION and \
                    list(getattr(m, "feature_names", [])) == SITUATION_FEATURES:
                return m
        except Exception:
            pass
    m = train(on_status=on_status)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(m, f)
    return m


# =========================================================================== 판정

def model_region(models) -> "Region | None":
    r = getattr(models, "region", None)
    return Region(**r) if r else None


def analyze(tracks: list[Track], models: Models) -> list[Track]:
    """트랙들(원본은 그대로)을 영역으로 자르고, 같은 사람을 이어 붙인 뒤 판정한다.
    돌려주는 목록: 영역 안 사람들 + 영역 밖 트랙들(판정 안 함)."""
    check_model(models)
    region = model_region(models)            # 학습할 때 쓴 측정 영역을 그대로 사용
    watch_r = getattr(models, "watch_r", WATCH_R)
    inside, outside = assemble(tracks, region, models.motion)
    for tr in outside:
        tr.out_of_range = True
    for tr in inside:
        tr.out_of_range = bool(tr.r.min() > watch_r)
    return predict(inside + outside, models)


# 상황 정의로 정해진 조건. 결정 트리 확률이 높아도 이 조건을 어기는 상황은 고르지 않는다.
# 학습할 때도 이 조건을 어기는 순간은 정답을 '판단 중'으로 바꿔서 학습한다.
#   final=True  : 사라진 순간(최종 판정)에만 확인하는 조건
#   final=False : 진행 중에도 항상 확인하는 조건
def _at_door_end(f):
    """문으로 사라짐, 또는 녹화가 끝날 때까지 문 앞에 있음"""
    return f["ended_at_door"] >= 0.5 or (f["ended"] < 0.5 and f["cur_r"] < DOOR_ENTER_R)


def _at_edge_end(f):
    """외곽으로 나감, 또는 녹화가 끝날 때 외곽에 있음"""
    return f["ended_at_edge"] >= 0.5 or (f["ended"] < 0.5 and f["cur_edge_d"] < EDGE_MARGIN)


RULES = [
    # (상황, 최종에만?, 조건 설명, 조건 함수(특징 dict) -> bool)
    ("귀가", True, "문으로 사라져야 함", _at_door_end),
    ("방문", True, "문으로 사라져야 함 (집 안으로 들어감)", _at_door_end),
    ("배달", True, "외곽(구역 바깥)으로 나가야 함", _at_edge_end),
    ("외출", False, "문에서 나온 사람이어야 함", lambda f: f["appear_door"] >= 0.6),
    ("외출", False, f"처음 잡힌 곳보다 문에 {OUT_MAX_APPROACH}m 이상 다가가면 외출 아님",
     lambda f: f["approach_depth"] < OUT_MAX_APPROACH),
    ("외출", True, "문으로 사라지면 외출 아님", lambda f: f["ended_at_door"] < 0.5),
    ("통과", True, "문으로 사라지면 통과 아님", lambda f: f["ended_at_door"] < 0.5),
    ("반복 접근", False, f"최근 {int(ENTRY_WINDOW_S)}초 안에 2번 이상 나타나야 함", lambda f: f["edge_entries"] >= 2),
]


def rule_violations(sit: str, feat: np.ndarray, final: bool) -> list[str]:
    f = dict(zip(SITUATION_FEATURES, feat))
    return [desc for name, only_final, desc, ok in RULES
            if name == sit and (final or not only_final) and not ok(f)]


def rules_text() -> str:
    """정의 조건 목록 (보고서·화면 표시용)"""
    return "\n".join(f"  {name:<6} {'(사라질 때)' if only_final else '(항상)    '} {desc}"
                     for name, only_final, desc, _ in RULES)


def _pick(proba: np.ndarray, classes, feat: np.ndarray, final: bool):
    """조건을 지키는 상황 중 확률이 가장 높은 것 (없으면 판단 중).
    사람이 실제로 사라진 최종 판정에서는 '판단 중'보다 조건을 지키는 실제 상황을 고른다."""
    order = np.argsort(-proba)
    gone = final and feat[SITUATION_FEATURES.index("ended")] >= 0.5
    for k in order:
        if proba[k] <= 0:
            break
        if gone and classes[k] == UNDECIDED:
            continue
        if not rule_violations(str(classes[k]), feat, final):
            return classes[k], float(proba[k])
    return UNDECIDED, 0.0


def _sustained(tr: Track, hold_s: float = CAMERA_HOLD_S, min_prob: float = CAMERA_PROB) -> dict:
    """진행 중 판정에서 같은 상황이 min_prob 이상으로 hold_s 이상 이어진 구간들.
    {상황: {"prob": 그 구간들 중 최고 확률, "t": 처음 hold_s를 채운 시각(절대), "dur": 가장 긴 구간 길이}}"""
    out: dict = {}
    run_s, run_kind, run_max = 0.0, None, 0.0
    for i in range(len(tr.t)):
        kind = str(tr.sit_pred[i])
        dt = min(tr.t[i] - tr.t[i - 1], 0.5) if i else 0.0
        if kind != UNDECIDED and tr.sit_prob[i] >= min_prob:
            if kind == run_kind:
                run_s += dt
                run_max = max(run_max, float(tr.sit_prob[i]))
            else:
                run_s, run_kind, run_max = 0.0, kind, float(tr.sit_prob[i])
            if run_s >= hold_s:
                b = out.setdefault(kind, {"prob": 0.0, "t": float(tr.t_abs[i]), "dur": 0.0})
                b["prob"] = max(b["prob"], run_max)
                b["dur"] = max(b["dur"], run_s)
        else:
            run_s, run_kind, run_max = 0.0, None, 0.0
    return out


NORMAL_FROM_PROGRESS = True   # 정상 행동도 진행 중 1초 넘게 이어진 것을 후보로 (끝날 때 정의 조건은 지켜야 함)

# 우선순위 예외: 한 사람에게 왼쪽 행동이 후보로 있으면 오른쪽 행동들은 후보에서 뺀다.
#   외출 > 장시간 체류: 집에서 나와 엘리베이터를 기다리느라 오래 서 있을 수 있음
#   (외출 후보는 '문에서 나옴' 등 외출 정의 조건을 지킨 경우에만 생김)
PRIORITY_OVERRIDES = {
    "외출": {"장시간 체류"},
}


def apply_overrides(cands: dict) -> tuple[dict, list[str]]:
    """PRIORITY_OVERRIDES를 적용한 후보와 뺀 내용 설명"""
    out, notes = dict(cands), []
    for win, losers in PRIORITY_OVERRIDES.items():
        if win in out:
            for lose in losers:
                if lose in out:
                    del out[lose]
                    notes.append(f"{win}이면 {lose}은 뺌")
    return out, notes


def priority_pick(cands: dict) -> tuple[str, float]:
    """{상황: 확률 또는 (확률, 동점 기준...)}에서 우선순위대로 하나 고른다.
    이상 행동이 1순위, 정상 행동이 2순위, 같은 순위 안에서는 확률이 높은 것
    (확률이 같으면 튜플의 다음 값들: 끝날 때 판정과 같은지, 이어진 시간).
    한 사람의 판정에는 먼저 apply_overrides로 예외(외출 > 장시간 체류)를 적용해서 넘긴다."""
    if not cands:
        return UNDECIDED, 0.0
    key = {k: (v if isinstance(v, tuple) else (v,)) for k, v in cands.items()}
    key = {k: (round(v[0], 2),) + tuple(v[1:]) for k, v in key.items()}
    abn = [k for k in key if k in ABNORMAL]
    pool = abn or [k for k in key if k != UNDECIDED] or list(key)
    best = max(pool, key=lambda k: key[k])
    return best, float(key[best][0])


def predict(tracks: list[Track], models: Models) -> list[Track]:
    """특징까지 계산된 트랙들에 상황 트리 판정과 카메라 촬영을 붙인다.
    결정 트리 확률 중 RULES를 어기는 상황은 빼고 고른다.

    최종 판정 (한 사람에게 여러 행동이 나올 수 있음):
      후보 = 진행 중 CAMERA_PROB 이상으로 CAMERA_HOLD_S 넘게 이어진 이상 행동들 (최고 확률)
           + 사라진(녹화가 끝난) 순간 최종 판정 트리의 판정
      우선순위: 이상 행동이 하나라도 있으면 그중 확률이 가장 높은 것, 없으면 끝날 때 판정 (정상 행동)
      정상 행동(귀가·방문·배달 등)은 어디로 사라졌는지가 있어야 알 수 있어서 끝날 때 판정으로 고른다."""
    for tr in tracks:
        if tr.out_of_range:     # 감시 범위 밖: 판정·촬영 안 함
            tr.sit_pred = np.array([UNDECIDED] * len(tr.t), dtype=object)
            tr.sit_prob = np.ones(len(tr.t))
            tr.final_pred, tr.final_prob, tr.cameras = UNDECIDED, 1.0, []
            tr.end_pred, tr.end_prob, tr.behaviors, tr.rule_note = UNDECIDED, 1.0, [], ""
            continue
        proba = models.situation.predict_proba(tr.sfeat)
        classes = models.situation.classes_
        picks = [_pick(p, classes, f, False) for p, f in zip(proba, tr.sfeat)]
        tr.sit_pred = np.array([c for c, _ in picks], dtype=object)
        tr.sit_prob = np.array([q for _, q in picks])
        ftree = getattr(models, "final", None) or models.situation
        fp = ftree.predict_proba(tr.final_feat[None])[0]
        tr.final_pred, tr.final_prob = _pick(fp, ftree.classes_, tr.final_feat, True)
        raw_best = ftree.classes_[fp.argmax()]
        if raw_best == tr.final_pred:
            tr.rule_note = ""
        elif raw_best == UNDECIDED:
            tr.rule_note = (f"트리는 판단 중 {fp.max():.0%}였지만 사람이 사라졌으므로 "
                            f"다음으로 확률 높은 {tr.final_pred} {tr.final_prob:.0%}")
        else:
            why = "; ".join(rule_violations(str(raw_best), tr.final_feat, True))
            path = describe_path(tr, model_region(models))
            tr.rule_note = (f"결정 트리 1순위는 {raw_best} {fp.max():.0%}였지만 {raw_best}의 조건 '{why}'에 안 맞음 ("
                            f"{path}) → 조건을 지키는 다음 순위 {tr.final_pred} {tr.final_prob:.0%}")
        # 카메라 촬영: 이상 상황이 CAMERA_HOLD_S 동안 유지되면 1회 (상황마다)
        tr.cameras = []
        run_s, run_kind, fired = 0.0, None, set()
        for i in range(len(tr.t)):
            kind = tr.sit_pred[i]
            dt = min(tr.t[i] - tr.t[i - 1], 0.5) if i else 0.0
            if kind in ABNORMAL and tr.sit_prob[i] >= CAMERA_PROB:
                run_s = run_s + dt if kind == run_kind else 0.0
                run_kind = kind
                if run_s >= CAMERA_HOLD_S and kind not in fired:
                    fired.add(kind)
                    tr.cameras.append({"t": float(tr.t_abs[i]), "situation": kind,
                                       "prob": float(tr.sit_prob[i])})
            else:
                run_s, run_kind = 0.0, None
        # 사라질 때 최종 판정이 이상 상황이면 그때 촬영 (짧게 나타났다 사라지는 반복 접근 등)
        if (tr.final_pred in ABNORMAL and tr.final_prob >= CAMERA_PROB
                and tr.final_feat[SITUATION_FEATURES.index("ended")] >= 0.5
                and tr.final_pred not in fired):
            tr.cameras.append({"t": float(tr.t_abs[-1]), "situation": tr.final_pred,
                               "prob": float(tr.final_prob), "final": True})

        # ---- 최종 판정: 이상 행동 1순위, 정상 행동 2순위
        tr.end_pred, tr.end_prob = tr.final_pred, tr.final_prob       # 끝날 때 판정 (최종 판정 트리)
        sus = _sustained(tr)
        # 후보: (확률, 끝날 때 판정과 같은지, 이어진 시간). 정상 행동은 끝날 때 정의 조건을 지키는 것만
        cands = {}
        for k, v in sus.items():
            if k in ABNORMAL or (NORMAL_FROM_PROGRESS and not rule_violations(k, tr.final_feat, True)):
                cands[k] = (v["prob"], k == tr.end_pred, v["dur"])
        if tr.end_pred != UNDECIDED and (tr.end_pred not in ABNORMAL or tr.end_prob >= CAMERA_PROB
                                         or not cands):
            old = cands.get(tr.end_pred, (0.0, True, 0.0))
            cands[tr.end_pred] = (max(old[0], tr.end_prob), True, old[2])
        cands, override_notes = apply_overrides(cands)
        best, prob = priority_pick(cands)
        if best != UNDECIDED:
            tr.final_pred, tr.final_prob = best, prob
        # 이 사람에게서 나온 행동들 (화면 표시용): 진행 중 1초 넘게 이어진 것 + 끝날 때 판정
        tr.behaviors = []
        for k, v in sorted(sus.items(), key=lambda kv: (kv[0] not in ABNORMAL, -kv[1]["prob"])):
            tr.behaviors.append({"situation": k, "prob": v["prob"], "t": v["t"], "dur": v["dur"],
                                 "source": "진행 중", "abnormal": k in ABNORMAL})
        if tr.end_pred != UNDECIDED:
            tr.behaviors.append({"situation": tr.end_pred, "prob": tr.end_prob, "t": float(tr.t_abs[-1]),
                                 "dur": 0.0, "source": "끝날 때", "abnormal": tr.end_pred in ABNORMAL})
        if tr.final_pred != tr.end_pred:
            b = sus.get(tr.final_pred)
            when = f" ({b['t'] - tr.t_abs[0]:.1f}초부터 {b['dur']:.1f}초 넘게)" if b else ""
            why = ("이상 행동 우선" if tr.final_pred in ABNORMAL and tr.end_pred not in ABNORMAL
                   else "확률이 더 높음")
            note = (f"{why}: 진행 중 {tr.final_pred} {tr.final_prob:.0%}{when} "
                    f"→ 끝날 때 판정 {tr.end_pred} {tr.end_prob:.0%} 대신")
            tr.rule_note = note + (f" / 끝날 때 판정: {tr.rule_note}" if tr.rule_note else "")
        if override_notes:
            tr.rule_note = ("우선순위 예외: " + ", ".join(override_notes)
                            + (f" / {tr.rule_note}" if tr.rule_note else ""))
    return tracks


def recording_verdict(tracks: list[Track]) -> tuple[str, float, "Track | None"]:
    """녹화(세션) 하나의 판정: 영역 안 사람들의 최종 판정 중 이상 행동 1순위, 정상 행동 2순위,
    같은 순위면 확률이 높은 것. (판정, 확률, 그 사람)"""
    people = [t for t in tracks if not getattr(t, "out_of_range", False)
              and getattr(t, "final_pred", UNDECIDED) != UNDECIDED]
    if not people:
        return UNDECIDED, 0.0, None
    cands: dict = {}
    who: dict = {}
    for t in people:
        v = (t.final_prob, float(t.t[-1]))             # 확률이 같으면 오래 잡힌 사람
        if t.final_pred not in cands or v > cands[t.final_pred]:
            cands[t.final_pred], who[t.final_pred] = v, t
    best, prob = priority_pick(cands)
    return best, prob, who.get(best)


def describe_path(tr: Track, region: "Region | None") -> str:
    """'외곽에서 들어옴(1.8m) → 문으로 사라짐(0.5m)' 같은 한 줄 설명"""
    hx, hy = _head(tr)
    tx, ty = _tail(tr)
    a = {"door": "문에서 나옴", "edge": "외곽에서 들어옴", "inside": "영역 안에서 나타남"}[_head_kind(tr, region)]
    gone = bool(tr.t_abs[-1] < tr.session_end - 0.5)
    b = ({"door": "문으로 사라짐", "edge": "외곽으로 나감", "inside": "영역 안에서 사라짐"}[_end_kind(tx, ty, region)]
         if gone else "녹화 끝까지 있음")
    return f"{a}({math.hypot(hx, hy):.1f}m) → {b}({math.hypot(tx, ty):.1f}m)"


def track_name(tr: Track) -> str:
    ids = tr.target_ids or [tr.target_id]
    return "T" + "+".join(str(i) for i in ids)


def explain(models: Models, x: np.ndarray, which: str = "situation") -> list[str]:
    """결정 트리가 이 판정을 내린 근거 (지나온 질문들)를 특징별로 묶어서 돌려준다."""
    tree = {"situation": models.situation, "motion": models.motion,
            "final": getattr(models, "final", None) or models.situation}[which]
    names = MOTION_FEATURES if which == "motion" else SITUATION_FEATURES
    t = tree.tree_
    node, bounds = 0, {}
    while t.children_left[node] != -1:
        f, thr = t.feature[node], t.threshold[node]
        lo, hi = bounds.get(f, (-math.inf, math.inf))
        if x[f] <= thr:
            node, hi = t.children_left[node], min(hi, thr)
        else:
            node, lo = t.children_right[node], max(lo, thr)
        bounds[f] = (lo, hi)
    out = []
    for f, (lo, hi) in bounds.items():
        name = FEATURE_KO[names[f]]
        if lo > -math.inf and hi < math.inf:
            out.append(f"{lo:.2f} < {name} ≤ {hi:.2f}   (지금 {x[f]:.2f})")
        elif lo > -math.inf:
            out.append(f"{name} > {lo:.2f}   (지금 {x[f]:.2f})")
        else:
            out.append(f"{name} ≤ {hi:.2f}   (지금 {x[f]:.2f})")
    return out


def observations(tr: Track, i: int, n_people: int, ended: bool = False,
                 region: "Region | None" = None) -> list[tuple[str, str]]:
    """화면에 보여줄 관찰값."""
    r = tr.r[i]
    zone = "문 바로 앞" if r < DOOR_R else "근처" if r < NEAR_R else "멀리"
    f = tr.sfeat[i]
    ix = SITUATION_FEATURES.index
    stay = f[ix("cur_still_s")]
    stay_txt = (f"오래 ({stay:.1f}초)" if stay >= LONG_STAY_S else
                f"잠깐 ({stay:.1f}초)" if stay > 0 else "-")
    if tr.hidden is not None and tr.hidden[i]:
        stay_txt += "  · 레이더가 놓침 → 그 자리에 멈춰 있다고 봄"
    kind_txt = {"door": "문에서 나옴", "edge": "외곽에서 들어옴", "inside": "영역 안에서 나타남"}
    where = kind_txt[_head_kind(tr, region)]
    if ended:
        end = _end_kind(*_tail(tr), region)
        where += " → " + {"door": "문으로 사라짐", "edge": "외곽으로 나감",
                          "inside": "영역 안에서 사라짐"}[end]
    return [
        ("거리", f"{zone} ({r:.2f} m)"),
        ("움직임 (트리)", str(tr.motion[i])),
        ("머문 시간", stay_txt),
        ("나타난/사라진 위치", where),
        ("이어 붙인 레이더 ID", ", ".join(str(t) for t in (tr.target_ids or [tr.target_id]))),
        ("자세", STATUS_KO.get(int(tr.status[i]), str(tr.status[i]))),
        ("인원", f"{n_people}명"),
        ("최근 나타난 횟수", f"{int(f[ix('edge_entries')])}회 ({int(ENTRY_WINDOW_S)}초 안, 문 제외)"),
    ]
