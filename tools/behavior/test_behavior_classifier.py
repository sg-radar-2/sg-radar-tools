"""행동 분류기 테스트 - 인공 궤적으로 기본 동작을 확인한다.

실행:
    python tools/behavior/test_behavior_classifier.py     # pytest 없이
    pytest tools/behavior                                 # pytest가 있으면

기준값을 바꾼 뒤 이 테스트가 깨지면, 의도한 변화인지 한 번 확인해보세요.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_classifier import (APPROACH, DWELL, LEAVE, LOITER,  # noqa: E402
                                 BehaviorClassifier, Config, TargetObs)

FPS = 18.0
NOISE = 0.05  # 레이더 위치 흔들림 흉내 (m)


def run(path_fn, duration, target_id=1, seed=0, cfg=None, switch_at=None, new_id=2, gap=0.0):
    """path_fn(t) -> (x, y) 궤적을 프레임으로 만들어 분류기에 넣고 이벤트를 모은다."""
    rnd = random.Random(seed)
    clf = BehaviorClassifier(cfg or Config(), session="test")
    events = []
    n = int(duration * FPS)
    for i in range(n):
        t = i / FPS
        tid = target_id
        if switch_at is not None and t >= switch_at:
            if t < switch_at + gap:
                events += clf.update(t, [])      # 트래킹 손실 구간
                continue
            tid = new_id
        x, y = path_fn(t)
        ob = TargetObs(tid, x + rnd.gauss(0, NOISE), y + rnd.gauss(0, NOISE), n_points=200)
        events += clf.update(t, [ob])
    events += clf.flush()
    return events, clf


def starts(events, kind=None):
    return [e for e in events if e["phase"] == "start" and (kind is None or e["type"] == kind)]


def test_approach_then_leave():
    # 4m에서 1m까지 다가왔다가(3초), 잠깐 서 있다가, 다시 4m로 멀어짐
    def path(t):
        if t < 3:
            return 0.1, 4 - t
        if t < 5:
            return 0.1, 1.0
        return 0.1, min(4.0, 1 + (t - 5))
    ev, _ = run(path, 9)
    kinds = [e["type"] for e in starts(ev)]
    assert kinds == [APPROACH, LEAVE], kinds


def test_lateral_pass_is_not_approach():
    # 문 앞 1.5m를 옆으로 지나감
    ev, _ = run(lambda t: (-2.0 + 0.8 * t, 1.5), 5)
    assert not starts(ev, APPROACH) and not starts(ev, LEAVE), starts(ev)


def test_dwell():
    # 1m 앞에 15초 서 있음
    ev, _ = run(lambda t: (0.2, 1.0), 15)
    d = starts(ev, DWELL)
    assert len(d) == 1 and 9.5 <= d[0]["time_s"] <= 12.0, d


def test_standing_far_is_not_dwell():
    # 3m 앞(문 앞 구역 밖)에 서 있음
    ev, _ = run(lambda t: (0.0, 3.0), 15)
    assert not starts(ev), starts(ev)


def test_loiter_circle():
    # (0, 2) 주변 반지름 1m 원을 40초 동안 돎
    ev, _ = run(lambda t: (math.cos(t * 0.6), 2 + math.sin(t * 0.6)), 40)
    lo = starts(ev, LOITER)
    assert len(lo) == 1 and lo[0]["time_s"] >= 30, lo


def test_stitch_id_switch():
    # 앞뒤로 걷다가 1초 끊기고 레이더 ID가 바뀜 → 같은 트랙으로 이어져야 함
    def path(t):
        return 0.0, 2.0 + 0.8 * math.sin(t * 0.8)
    ev, clf = run(path, 12, switch_at=6.0, gap=1.0)
    track_ids = {e["track_id"] for e in ev}
    assert track_ids == {0}, track_ids
    assert any(e["target_ids"] == [1, 2] for e in ev), [e["target_ids"] for e in ev]


def test_low_points_filtered():
    clf = BehaviorClassifier(Config())
    for i in range(100):
        clf.update(i / FPS, [TargetObs(1, 0.0, 1.0, n_points=5)])
    assert not clf.tracks


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  통과  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  실패  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    sys.exit(1 if failed else 0)
