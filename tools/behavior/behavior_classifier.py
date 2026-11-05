"""행동 분류기 (초안) - 접근(APPROACH) / 이탈(LEAVE) / 체류(DWELL) / 배회(LOITER)

규칙 기반 + 스트리밍 구조입니다. 레이더 프레임을 하나씩 넣으면 사람(트랙)마다 상태를 갱신하고,
행동이 시작되거나 끝나는 순간에만 이벤트(dict, JSON으로 바로 직렬화 가능)를 돌려줍니다.
CSV 검증(classify_csv.py)과 실시간 동작(라즈베리파이/서버)에서 같은 코드를 씁니다.

    from behavior_classifier import BehaviorClassifier, Config, TargetObs

    clf = BehaviorClassifier(Config(), session="Session #0")
    for t, frame_targets in frames:                     # t: 초, 세션 시작부터
        obs = [TargetObs(target_id=..., x=..., y=..., n_points=..., doppler=...) for ...]
        for event in clf.update(t, obs):
            send_to_server(event)                       # 예: FastAPI로 POST
    for event in clf.flush():                           # 녹화/세션이 끝날 때 진행 중인 행동 닫기
        send_to_server(event)

좌표: 레이더 좌표계(x 좌우, y 정면 방향, 단위 m). 문 위치는 Config.door_x/door_y로 정하고,
"거리 r"은 문까지의 거리입니다. 기본값은 레이더가 문에 붙어 있다고 보고 (0, 0)입니다.

판정 흐름
  1) 입력 거르기: 포인트 수가 min_points 미만인 관측은 위치를 믿기 어려워 버립니다.
  2) 트랙 관리: 레이더 target_id를 트랙에 연결하고, 잠깐 끊겼다 가까운 곳에서 새 ID가 나오면 이어 붙입니다.
  3) 짧은 판정(motion): 최근 motion_window_s 동안의 위치로 속도를 구해서
     approaching / receding / still / moving 중 하나. confirm_s 이상 유지돼야 확정합니다.
     - 문 쪽 속도(거리 변화율)가 충분히 크고, 움직임 대부분이 문 방향(radial ratio)일 때만
       approaching/receding. 문 앞을 옆으로 지나가는 움직임을 접근으로 잘못 보지 않기 위함.
     - 도플러가 같은 방향을 가리키면 더 약한 속도로도 인정합니다.
  4) 행동 판정:
     APPROACH  approaching이 이어지는 동안 거리가 approach_min_dr_m 이상 줄었을 때 시작
     LEAVE     receding이 이어지는 동안 거리가 approach_min_dr_m 이상 늘었을 때 시작
     DWELL     문 앞 구역(zone_radius_m) 안에서 dwell_check_s 동안 거의 안 움직인 상태가
               dwell_min_s 이상 이어질 때
     LOITER    최근 loiter_window_s 동안 감시 범위(watch_radius_m) 안에 머물면서 계속 움직였고
               (움직인 비율, 경로 길이) 곧게 가지 않았거나(직진도) 접근·이탈을 반복했을 때.
               배회 중에는 접근/이탈 이벤트를 따로 내지 않습니다 (suppress_moves_during_loiter).
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

APPROACH = "APPROACH"
LEAVE = "LEAVE"
DWELL = "DWELL"
LOITER = "LOITER"
BEHAVIORS = (APPROACH, LEAVE, DWELL, LOITER)

APPROACHING = "approaching"
RECEDING = "receding"
STILL = "still"
MOVING = "moving"
UNKNOWN = "unknown"

NAN = float("nan")


# =========================================================================== 설정

@dataclass
class Config:
    # --- 좌표: 문 위치 (레이더 기준, m)
    door_x: float = 0.0
    door_y: float = 0.0

    # --- 입력 품질
    min_points: int = 30            # 포인트 수가 이보다 적은 관측은 버림
    smooth_s: float = 0.3           # 위치 흔들림을 줄이는 이동평균 길이 (초)

    # --- 트랙 관리
    lost_timeout_s: float = 2.0     # 이 시간 동안 관측이 없으면 트랙 종료
    stitch_gap_s: float = 4.0       # 끊긴 지 이 시간 안에 나타난 새 ID는 이어 붙이기 후보
    stitch_dist_m: float = 1.5      # 마지막 위치와 이 거리 안이면 같은 사람으로 봄

    # --- 짧은 판정 (motion)
    motion_window_s: float = 1.0    # 이 시간 동안의 위치로 속도 계산 (최소제곱 기울기)
    still_speed: float = 0.3        # 속력이 이보다 작으면 still (서 있어도 위치가 흔들려서 0.2 정도 나옴)
    approach_speed: float = 0.3     # 문 쪽 속도가 이보다 크면 approaching / receding
    approach_speed_weak: float = 0.15  # 도플러가 같은 방향이면 이 정도 속도로도 인정
    doppler_thresh: float = 0.2     # 도플러 평균이 이보다 크면 방향 근거로 사용 (음수=다가옴)
    min_radial_ratio: float = 0.85  # 움직임 중 문 방향 성분 비율 (작으면 옆으로 지나가는 중)
    confirm_s: float = 0.5          # 새 판정이 이 시간 이상 유지돼야 확정
    approach_min_dr_m: float = 0.4  # 접근/이탈 중 거리가 이만큼 변해야 이벤트 시작

    # --- 체류
    zone_radius_m: float = 1.5      # 문 앞 구역 반경
    dwell_check_s: float = 3.0      # 이 시간 동안의 순이동으로 '머물러 있음'을 판단
    dwell_max_disp_m: float = 0.5   # 그 순이동이 이보다 작으면 머무는 중
    dwell_min_s: float = 10.0       # 머무는 상태가 이 시간 이상이면 DWELL
    dwell_break_s: float = 2.0      # 잠깐 움직여도 이 시간 안에 다시 멈추면 체류가 이어진 것으로 봄

    # --- 배회
    watch_radius_m: float = 3.5     # 감시 범위 반경
    loiter_window_s: float = 30.0   # 최근 이 시간을 보고 판단
    loiter_min_presence: float = 0.8  # 그중 이 비율 이상을 감시 범위 안에 있어야 함
    loiter_min_moving: float = 0.5  # 그중 이 비율 이상 움직이고 있어야 함 (서 있는 사람 제외)
    loiter_min_path_m: float = 10.0  # 그동안 움직인 경로 길이 최소값 (서 있으면 흔들림만으로 4~6m)
    loiter_max_straightness: float = 0.35  # 직진도(순이동/경로)가 이보다 작으면 맴도는 것
    loiter_min_reversals: int = 2   # 또는 접근↔이탈 방향 전환이 이 횟수 이상
    loiter_end_s: float = 5.0       # 조건이 이 시간 이상 깨져야 배회 종료
    path_step_s: float = 1.0        # 경로 길이는 이 간격으로 뽑은 위치로 계산 (흔들림 영향 줄이기)
    suppress_moves_during_loiter: bool = True  # 배회 중에는 접근/이탈 이벤트를 내지 않음

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"알 수 없는 설정 이름: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


# =========================================================================== 입력/상태

@dataclass
class TargetObs:
    """한 프레임에서 레이더가 준 사람 한 명의 관측값."""
    target_id: int
    x: float
    y: float
    n_points: int | None = None     # 모르면 None (품질 거르기 생략)
    doppler: float = NAN            # 그 사람 포인트들의 도플러 평균, 모르면 NaN


@dataclass
class _Sample:
    t: float
    x: float
    y: float
    r: float
    doppler: float
    motion: str = UNKNOWN           # 이 시점의 짧은 판정 (확정 전 값)


@dataclass
class Track:
    id: int
    target_ids: list[int]
    first_t: float
    last_t: float
    raw: deque = field(default_factory=deque)       # 평활화 전 관측 (t, x, y, doppler)
    hist: deque = field(default_factory=deque)      # 평활화된 샘플 (_Sample)
    motion: str = UNKNOWN
    raw_motion: str = UNKNOWN
    cand: str | None = None
    cand_since: float = 0.0
    cand_r: float = NAN
    move_t0: float = 0.0              # 현재 approaching/receding이 시작된 시각
    move_r0: float = NAN              # 그때 거리
    motion_log: deque = field(default_factory=deque)  # (t, motion) 확정된 판정 변화
    active: dict = field(default_factory=dict)        # 행동 -> 시작 정보
    dwell_since: float | None = None
    dwell_break_since: float | None = None
    loiter_off_since: float | None = None
    stats: dict = field(default_factory=dict)         # 마지막 계산값 (디버그/라벨용)

    @property
    def pos(self) -> _Sample | None:
        return self.hist[-1] if self.hist else None


# =========================================================================== 분류기

class BehaviorClassifier:
    def __init__(self, config: Config | None = None, session: str = ""):
        self.cfg = config or Config()
        self.session = session
        self.tracks: dict[int, Track] = {}          # 살아 있는 트랙 (트랙 id -> Track)
        self.id_map: dict[int, int] = {}            # 레이더 target_id -> 트랙 id
        self._next_track = 0
        self._event_seq = 0
        self.t = 0.0

    # ------------------------------------------------------------------ 공개 API
    def update(self, t: float, targets: Iterable[TargetObs]) -> list[dict]:
        """프레임 하나를 처리하고, 이번 프레임에서 생긴 이벤트 목록을 돌려준다."""
        self.t = t
        events: list[dict] = []
        seen: set[int] = set()
        good = [ob for ob in targets
                if ob.n_points is None or ob.n_points >= self.cfg.min_points]
        current_ids = {ob.target_id for ob in good}
        # 이미 아는 ID를 먼저 처리해야 새 ID가 살아 있는 트랙에 잘못 붙지 않음
        good.sort(key=lambda ob: ob.target_id not in self.id_map)
        for ob in good:
            tr = self._assign(ob, t, seen, current_ids)
            seen.add(tr.id)
            self._add_obs(tr, t, ob)

        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tid in seen:
                events += self._evaluate(tr, t)
            elif t - tr.last_t > self.cfg.lost_timeout_s:
                events += self._close(tr, tr.last_t, reason="lost")
        return events

    def flush(self) -> list[dict]:
        """진행 중인 행동을 모두 닫는다 (녹화 끝, 세션 끝, 프로그램 종료 때)."""
        events: list[dict] = []
        for tr in list(self.tracks.values()):
            events += self._close(tr, tr.last_t, reason="end_of_data")
        return events

    def snapshot(self) -> dict[int, dict]:
        """레이더 target_id별 현재 상태 (라벨 CSV, 화면 표시용)."""
        out = {}
        for target_id, tid in self.id_map.items():
            tr = self.tracks.get(tid)
            if tr is None or tr.pos is None:
                continue
            p = tr.pos
            out[target_id] = {
                "track": tr.id, "x_s": p.x, "y_s": p.y, "r": p.r, "motion": tr.motion,
                "raw_motion": tr.raw_motion, "behaviors": sorted(tr.active), **tr.stats,
            }
        return out

    # ------------------------------------------------------------------ 트랙 연결
    def _dist_to_door(self, x: float, y: float) -> float:
        return math.hypot(x - self.cfg.door_x, y - self.cfg.door_y)

    def _assign(self, ob: TargetObs, t: float, taken: set[int], current_ids: set[int]) -> Track:
        tid = self.id_map.get(ob.target_id)
        if tid is not None and tid in self.tracks and tid not in taken:
            return self.tracks[tid]
        # 이어 붙이기: 최근에 끊긴 트랙 중 가장 가까운 것
        best, best_d = None, self.cfg.stitch_dist_m
        for tr in self.tracks.values():
            if tr.id in taken or tr.pos is None:
                continue
            gap = t - tr.last_t
            if 0 < gap <= self.cfg.stitch_gap_s and tr.target_ids[-1] not in current_ids:
                d = math.hypot(ob.x - tr.pos.x, ob.y - tr.pos.y)
                if d <= best_d:
                    best, best_d = tr, d
        if best is not None:
            best.target_ids.append(ob.target_id)
            self.id_map[ob.target_id] = best.id
            return best
        tr = Track(id=self._next_track, target_ids=[ob.target_id], first_t=t, last_t=t)
        self._next_track += 1
        self.tracks[tr.id] = tr
        self.id_map[ob.target_id] = tr.id
        return tr

    def _add_obs(self, tr: Track, t: float, ob: TargetObs) -> None:
        c = self.cfg
        tr.last_t = t
        tr.raw.append((t, ob.x, ob.y, ob.doppler))
        while tr.raw and tr.raw[0][0] < t - c.smooth_s:
            tr.raw.popleft()
        n = len(tr.raw)
        x = sum(o[1] for o in tr.raw) / n
        y = sum(o[2] for o in tr.raw) / n
        dops = [o[3] for o in tr.raw if not math.isnan(o[3])]
        dop = sum(dops) / len(dops) if dops else NAN
        tr.hist.append(_Sample(t, x, y, self._dist_to_door(x, y), dop))
        keep = max(c.loiter_window_s, c.motion_window_s) + 1.0
        while tr.hist and tr.hist[0].t < t - keep:
            tr.hist.popleft()
        while tr.motion_log and tr.motion_log[0][0] < t - c.loiter_window_s:
            tr.motion_log.popleft()

    # ------------------------------------------------------------------ 판정
    def _window(self, tr: Track, seconds: float) -> list[_Sample]:
        t0 = tr.hist[-1].t - seconds
        return [s for s in tr.hist if s.t >= t0]

    @staticmethod
    def _slope(ts: list[float], vs: list[float]) -> float:
        n = len(ts)
        mt, mv = sum(ts) / n, sum(vs) / n
        den = sum((t - mt) ** 2 for t in ts)
        return sum((t - mt) * (v - mv) for t, v in zip(ts, vs)) / den if den > 0 else 0.0

    def _classify_motion(self, tr: Track) -> str:
        c = self.cfg
        win = self._window(tr, c.motion_window_s)
        if len(win) < 4 or win[-1].t - win[0].t < 0.6 * c.motion_window_s:
            return UNKNOWN
        ts = [s.t for s in win]
        vx = self._slope(ts, [s.x for s in win])
        vy = self._slope(ts, [s.y for s in win])
        vr = self._slope(ts, [s.r for s in win])      # 음수 = 문으로 다가옴
        speed = math.hypot(vx, vy)
        ratio = abs(vr) / speed if speed > 1e-6 else 0.0
        dops = [s.doppler for s in win if not math.isnan(s.doppler)]
        dop = sum(dops) / len(dops) if dops else NAN
        tr.stats.update(speed=speed, vr=vr, radial_ratio=ratio, doppler_win=dop)
        if speed < c.still_speed:
            return STILL
        if ratio >= c.min_radial_ratio:
            if vr <= -c.approach_speed or (vr <= -c.approach_speed_weak and dop <= -c.doppler_thresh):
                return APPROACHING
            if vr >= c.approach_speed or (vr >= c.approach_speed_weak and dop >= c.doppler_thresh):
                return RECEDING
        return MOVING

    def _evaluate(self, tr: Track, t: float) -> list[dict]:
        c = self.cfg
        events: list[dict] = []

        p = tr.pos
        # --- 짧은 판정 + 확정(깜빡임 방지)
        raw = self._classify_motion(tr)
        tr.raw_motion = p.motion = raw
        if raw == tr.motion:
            tr.cand = None
        elif tr.cand != raw:
            tr.cand, tr.cand_since, tr.cand_r = raw, t, p.r
        elif t - tr.cand_since >= c.confirm_s:
            prev, tr.motion, tr.cand = tr.motion, raw, None
            tr.motion_log.append((t, raw))
            tr.move_t0, tr.move_r0 = tr.cand_since, tr.cand_r
            if prev == APPROACHING:
                events += self._end(tr, APPROACH, t, "motion_changed")
            if prev == RECEDING:
                events += self._end(tr, LEAVE, t, "motion_changed")

        # --- 접근 / 이탈: 같은 방향으로 거리가 충분히 변했을 때 시작
        quiet = c.suppress_moves_during_loiter and LOITER in tr.active
        if tr.motion == APPROACHING and APPROACH not in tr.active and not quiet:
            if tr.move_r0 - p.r >= c.approach_min_dr_m:
                events += self._start(tr, APPROACH, t, start_t=tr.move_t0, start_r=tr.move_r0)
        if tr.motion == RECEDING and LEAVE not in tr.active and not quiet:
            if p.r - tr.move_r0 >= c.approach_min_dr_m:
                events += self._start(tr, LEAVE, t, start_t=tr.move_t0, start_r=tr.move_r0)

        # --- 체류
        dw = self._window(tr, c.dwell_check_s)
        settled = (dw[-1].t - dw[0].t >= 0.8 * c.dwell_check_s
                   and math.hypot(dw[-1].x - dw[0].x, dw[-1].y - dw[0].y) < c.dwell_max_disp_m)
        if p.r <= c.zone_radius_m and settled:
            if tr.dwell_since is None:
                tr.dwell_since = dw[0].t   # 확인 구간 시작부터 이미 머물러 있었음
            tr.dwell_break_since = None
        elif tr.dwell_since is not None:
            if tr.dwell_break_since is None:
                tr.dwell_break_since = t
            elif t - tr.dwell_break_since > c.dwell_break_s:
                tr.dwell_since = None
                tr.dwell_break_since = None
                events += self._end(tr, DWELL, t, "moved_or_left_zone")
        if (tr.dwell_since is not None and DWELL not in tr.active
                and t - tr.dwell_since >= c.dwell_min_s):
            events += self._start(tr, DWELL, t, start_t=tr.dwell_since)

        # --- 배회
        loiter = self._loiter_condition(tr, t)
        if loiter:
            tr.loiter_off_since = None
            if LOITER not in tr.active:
                if c.suppress_moves_during_loiter:
                    events += self._end(tr, APPROACH, t, "loiter_started")
                    events += self._end(tr, LEAVE, t, "loiter_started")
                events += self._start(tr, LOITER, t, start_t=max(tr.first_t, t - c.loiter_window_s))
        elif LOITER in tr.active:
            if tr.loiter_off_since is None:
                tr.loiter_off_since = t
            elif t - tr.loiter_off_since >= c.loiter_end_s:
                tr.loiter_off_since = None
                events += self._end(tr, LOITER, t, "condition_cleared")
        return events

    def _loiter_condition(self, tr: Track, t: float) -> bool:
        c = self.cfg
        if t - tr.first_t < c.loiter_window_s or DWELL in tr.active:
            return False
        win = self._window(tr, c.loiter_window_s)
        if len(win) < 5:
            return False
        inside = sum(s.r <= c.watch_radius_m for s in win) / len(win)
        moving = sum(s.motion in (MOVING, APPROACHING, RECEDING) for s in win) / len(win)
        # 경로 길이: path_step_s 간격으로 뽑은 위치로 계산
        pts, next_t = [], win[0].t
        for s in win:
            if s.t >= next_t:
                pts.append(s)
                next_t = s.t + c.path_step_s
        path = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:]))
        disp = math.hypot(win[-1].x - win[0].x, win[-1].y - win[0].y)
        straight = disp / path if path > 1e-6 else 1.0
        dirs = [m for _, m in tr.motion_log if m in (APPROACHING, RECEDING)]
        reversals = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
        tr.stats.update(presence=inside, moving=moving, path=path, straightness=straight,
                        reversals=reversals)
        return (inside >= c.loiter_min_presence and moving >= c.loiter_min_moving
                and path >= c.loiter_min_path_m
                and (straight <= c.loiter_max_straightness or reversals >= c.loiter_min_reversals))

    # ------------------------------------------------------------------ 이벤트
    def _event(self, tr: Track, kind: str, phase: str, t: float, **extra) -> dict:
        p = tr.pos
        ev = {
            "event_id": f"{self.session or 'S'}-T{tr.id}-{kind}-{tr.active.get(kind, {}).get('seq', self._event_seq)}",
            "type": kind,
            "phase": phase,
            "session": self.session,
            "track_id": tr.id,
            "target_ids": list(tr.target_ids),
            "time_s": round(t, 3),
            "x": round(p.x, 3) if p else None,
            "y": round(p.y, 3) if p else None,
            "distance_m": round(p.r, 3) if p else None,
        }
        ev.update(extra)
        return ev

    def _start(self, tr: Track, kind: str, t: float, start_t: float,
               start_r: float | None = None) -> list[dict]:
        if kind in tr.active:
            return []
        self._event_seq += 1
        p = tr.pos
        r0 = start_r if start_r is not None else (p.r if p else NAN)
        tr.active[kind] = {"seq": self._event_seq, "start_t": start_t, "start_r": r0}
        return [self._event(tr, kind, "start", t, start_s=round(start_t, 3),
                            start_distance_m=round(r0, 3))]

    def _end(self, tr: Track, kind: str, t: float, reason: str) -> list[dict]:
        info = tr.active.get(kind)
        if info is None:
            return []
        p = tr.pos
        ev = self._event(tr, kind, "end", t, start_s=round(info["start_t"], 3), end_s=round(t, 3),
                         duration_s=round(t - info["start_t"], 3),
                         start_distance_m=round(info["start_r"], 3),
                         end_distance_m=round(p.r, 3) if p else None, reason=reason)
        del tr.active[kind]
        return [ev]

    def _close(self, tr: Track, t: float, reason: str) -> list[dict]:
        events = []
        for kind in list(tr.active):
            events += self._end(tr, kind, t, reason)
        self.tracks.pop(tr.id, None)
        for target_id in [k for k, v in self.id_map.items() if v == tr.id]:
            del self.id_map[target_id]
        return events
