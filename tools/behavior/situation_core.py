"""문 앞 상황 판정 - 특징 계산, 결정 트리 학습, 판정, 카메라 촬영 이벤트

두 개의 결정 트리를 씁니다.
  1) 움직임 트리 (프레임마다): 속도, 문 쪽 속도, 문 방향 비율, 도플러 → 다가옴 / 멀어짐 / 지나감 / 정지
  2) 상황 트리 (0.5초마다, 지금까지의 흐름): 머문 시간, 나타난 위치, 접근 횟수, 손 움직임 등
     → 통과 / 귀가 / 외출 / 방문·배달 / 장시간 체류 / 배회 / 반복 접근 / 도어락 조작 시도 / 판단 중
상황 트리가 '이상' 상황을 CAMERA_PROB 이상 확률로 CAMERA_HOLD_S 동안 유지하면 카메라 촬영 이벤트를 냅니다.

지금은 scenario_sim.py의 가짜 데이터로 학습합니다. 녹화에 정답을 붙이면 같은 방식으로 다시 학습하면 됩니다.
필요: numpy, scikit-learn
"""

from __future__ import annotations

import csv
import math
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scenario_sim import (ABNORMAL, M_APPROACH, M_LEAVE, M_PASS, M_STILL,  # noqa: F401
                          MOTIONS, SITUATIONS, UNDECIDED, training_episodes)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "situation_trees.pkl"
MODEL_VERSION = 1

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

MOTION_FEATURES = ["speed", "vr", "radial_ratio", "doppler_win"]
SITUATION_FEATURES = ["elapsed", "appear_r", "cur_r", "min_r", "rise_from_min", "cur_still_s",
                      "max_still_s", "door_still_s", "hand_still", "n_approach", "n_leave",
                      "path_len", "straightness", "lateral_frac", "sit_frac", "ended"]
FEATURE_KO = {
    "speed": "속력(m/s)", "vr": "문 쪽 속도(m/s, -는 다가옴)", "radial_ratio": "문 방향 비율",
    "doppler_win": "도플러 평균",
    "elapsed": "잡힌 지 지난 시간(초)", "appear_r": "처음 잡힌 거리(m)", "cur_r": "현재 거리(m)",
    "min_r": "가장 가까웠던 거리(m)", "rise_from_min": "가장 가까웠던 뒤 멀어진 거리(m)",
    "cur_still_s": "지금 연속 정지 시간(초)", "max_still_s": "최장 연속 정지 시간(초)",
    "door_still_s": "문 바로 앞 정지 시간(초)", "hand_still": "정지 중 손 움직임(도플러 퍼짐)",
    "n_approach": "접근 횟수", "n_leave": "이탈 횟수", "path_len": "움직인 경로(m)",
    "straightness": "직진도", "lateral_frac": "옆으로 이동한 비율", "sit_frac": "앉음·숙임 비율",
    "ended": "사라짐(1=사라짐)",
}
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


def read_csv_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"time_s", "target_id", "x", "y"} <= set(rows[0]):
        raise ValueError("트랙 CSV가 아닙니다 (time_s, target_id, x, y 열 필요)")
    return rows


def tracks_from_rows(rows: list[dict]) -> list[Track]:
    """(세션, target_id)마다 하나의 트랙. 포인트가 적은 행은 버린다."""
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        npts = _f(r.get("n_points"), 999)
        if npts < MIN_POINTS:
            continue
        key = (r.get("session") or "Session #0", int(float(r["target_id"])))
        groups.setdefault(key, []).append(r)
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
    """새 판정이 CONFIRM_S 이상 유지될 때만 바꾼다 (깜빡임 방지)."""
    out = np.empty(len(raw), dtype=object)
    cur, cand, cand_t = M_STILL, None, 0.0
    for i, m in enumerate(raw):
        if m == cur:
            cand = None
        elif m != cand:
            cand, cand_t = m, t[i]
        elif t[i] - cand_t >= CONFIRM_S:
            cur, cand = m, None
        out[i] = cur
    return out


def situation_features(tr: Track) -> None:
    """프레임마다 '지금까지의 흐름' 특징. 마지막에 사라진 시점의 특징(ended=1)도 만든다."""
    t, r, xs, ys, m = tr.t, tr.r, tr.xs, tr.ys, tr.motion
    n = len(t)
    F = np.zeros((n, len(SITUATION_FEATURES)))
    min_r, cur_still, max_still, door_still = math.inf, 0.0, 0.0, 0.0
    hand_sum, hand_n, n_app, n_lv = 0.0, 0, 0, 0
    seg, seg_r0, seg_done = None, 0.0, False
    path, last_t, lx, ly = 0.0, -math.inf, 0.0, 0.0
    lat_n = sit_n = 0
    r0 = r[0]
    for i in range(n):
        dt = min(t[i] - t[i - 1], 0.5) if i else 0.0
        mi = m[i]
        min_r = min(min_r, r[i])
        if mi == M_STILL and r[i] < STILL_NEAR_R:
            cur_still += dt
            hand_sum += tr._dstd_win[i]
            hand_n += 1
        else:
            cur_still = 0.0
        max_still = max(max_still, cur_still)
        if mi == M_STILL and r[i] < DOOR_R:
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
        F[i] = [t[i], r0, r[i], min_r, r[i] - min_r, cur_still, max_still, door_still,
                hand_sum / hand_n if hand_n else 0.0, n_app, n_lv, path,
                net / path if path > 0.5 else 1.0, lat_n / (i + 1), sit_n / (i + 1), 0.0]
    tr.sfeat = F
    tr.final_feat = F[-1].copy()
    tr.final_feat[-1] = 1.0


# =========================================================================== 학습

@dataclass
class Models:
    motion: object
    situation: object
    report: str
    version: int = MODEL_VERSION


def _prepare(tr: Track, motion_tree=None) -> None:
    motion_features(tr)
    if motion_tree is not None:
        raw = motion_tree.predict(tr.mfeat)
        raw = np.where(tr._motion_ok, raw, M_STILL)
        tr.motion = confirm_motion(tr.t, raw)
        situation_features(tr)


def _snapshots(tr: Track):
    """상황 트리 학습용: 0.5초마다 하나 + 사라진 시점 (3배 가중)."""
    X, y = [], []
    next_t = 0.0
    for i in range(len(tr.t)):
        if tr.t[i] >= next_t:
            X.append(tr.sfeat[i])
            y.append(tr.truth_situation[i])
            next_t = tr.t[i] + SNAP_STEP_S
    for _ in range(3):
        X.append(tr.final_feat)
        y.append(tr.truth_final)
    return X, y


def _fit(tracks: list[Track]):
    from sklearn.tree import DecisionTreeClassifier
    mX, my = [], []
    for tr in tracks:
        _prepare(tr)
        sel = np.where(tr._motion_ok)[0][::2]
        mX.append(tr.mfeat[sel])
        my += [tr.truth_motion[i] for i in sel]
    motion_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40,
                                         class_weight="balanced", random_state=0)
    motion_tree.fit(np.vstack(mX), my)
    sX, sy = [], []
    for tr in tracks:
        _prepare(tr, motion_tree)
        X, y = _snapshots(tr)
        sX += X
        sy += y
    sit_tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=15,
                                      class_weight="balanced", random_state=0)
    sit_tree.fit(np.array(sX), sy)
    return motion_tree, sit_tree


def train(n_per_kind: int = 30, seed: int = 0, on_status=None) -> Models:
    """가짜 데이터로 학습. 에피소드의 1/4은 검증용으로 빼서 정확도를 재고, 최종 모델은 전체로 다시 학습."""
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import export_text
    say = on_status or (lambda s: None)
    say("가짜 시나리오 만드는 중...")
    eps = training_episodes(n_per_kind, seed)
    tracks = []
    for name, rows in eps:
        trs = tracks_from_rows([dict(r, session=name) for r in rows])
        tracks += trs
    train_trs = [tr for k, tr in enumerate(tracks) if k % 4 != 0]
    test_trs = [tr for k, tr in enumerate(tracks) if k % 4 == 0]

    say("결정 트리 학습 중 (검증용)...")
    mt, st = _fit(train_trs)
    # 검증
    m_true, m_pred, s_true, s_pred, f_true, f_pred = [], [], [], [], [], []
    for tr in test_trs:
        _prepare(tr, mt)
        ok = tr._motion_ok
        m_true += [tr.truth_motion[i] for i in np.where(ok)[0]]
        m_pred += list(mt.predict(tr.mfeat[ok]))
        X, y = _snapshots(tr)
        s_true += y[:-3]
        s_pred += list(st.predict(np.array(X[:-3])))
        f_true.append(tr.truth_final)
        f_pred.append(st.predict(tr.final_feat[None])[0])
    acc = lambda a, b: np.mean(np.array(a) == np.array(b)) if a else 0.0  # noqa: E731

    say("결정 트리 학습 중 (전체)...")
    mt, st = _fit(tracks)

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
              export_text(st, feature_names=ko(SITUATION_FEATURES), decimals=2)]
    return Models(mt, st, "\n".join(lines))


def load_or_train(force: bool = False, on_status=None) -> Models:
    if not force and MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                m = pickle.load(f)
            if getattr(m, "version", 0) == MODEL_VERSION:
                return m
        except Exception:
            pass
    m = train(on_status=on_status)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(m, f)
    return m


# =========================================================================== 판정

def analyze(tracks: list[Track], models: Models) -> list[Track]:
    for tr in tracks:
        _prepare(tr, models.motion)
        proba = models.situation.predict_proba(tr.sfeat)
        classes = models.situation.classes_
        k = proba.argmax(axis=1)
        tr.sit_pred = classes[k]
        tr.sit_prob = proba[np.arange(len(k)), k]
        fp = models.situation.predict_proba(tr.final_feat[None])[0]
        tr.final_pred, tr.final_prob = classes[fp.argmax()], float(fp.max())
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
    return tracks


def explain(models: Models, x: np.ndarray, which: str = "situation") -> list[str]:
    """결정 트리가 이 판정을 내린 근거 (지나온 질문들)를 특징별로 묶어서 돌려준다."""
    tree = models.situation if which == "situation" else models.motion
    names = SITUATION_FEATURES if which == "situation" else MOTION_FEATURES
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


def observations(tr: Track, i: int, n_people: int, ended: bool = False) -> list[tuple[str, str]]:
    """화면에 보여줄 6가지 관찰값."""
    r = tr.r[i]
    zone = "문 바로 앞" if r < DOOR_R else "근처" if r < NEAR_R else "멀리"
    stay = tr.sfeat[i][SITUATION_FEATURES.index("cur_still_s")]
    stay_txt = (f"오래 ({stay:.1f}초)" if stay >= LONG_STAY_S else
                f"잠깐 ({stay:.1f}초)" if stay > 0 else "-")
    appear = "문에서 나옴" if tr.r[0] < APPEAR_DOOR_R else "바깥에서 옴"
    if ended:
        appear += " → " + ("문으로 사라짐" if r < APPEAR_DOOR_R else "바깥으로 사라짐")
    return [
        ("거리", f"{zone} ({r:.2f} m)"),
        ("움직임 (트리)", str(tr.motion[i])),
        ("머문 시간", stay_txt),
        ("나타난/사라진 위치", appear),
        ("자세", STATUS_KO.get(int(tr.status[i]), str(tr.status[i]))),
        ("인원", f"{n_people}명"),
    ]
