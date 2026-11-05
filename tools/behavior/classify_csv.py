"""트랙 CSV에 행동 분류기를 돌려서 이벤트(JSON)와 프레임별 라벨(CSV)을 만든다.

입력: tools/pcr/to_numpy_gui.pyw가 만든 <이름>_tracks_raw.csv
      (session, time_s, frame_id, target_id, x, y, status, n_points, doppler_mean, ...)
      예전 check_tracks.py CSV(세션·도플러 열 없음)도 읽힙니다.

실행:
    python tools/behavior/classify_csv.py recordings/20261104_back_forth_walk_run_tracks_raw.csv
    python tools/behavior/classify_csv.py recordings/*_tracks_raw.csv --out-dir results
    python tools/behavior/classify_csv.py a.csv --config my_config.json
    python tools/behavior/classify_csv.py --dump-config config.json   # 기본 설정 파일 만들기

출력 (CSV 옆, 또는 --out-dir):
    <이름>_events.json   행동 시작/끝 이벤트 목록 (서버로 보낼 형태 그대로)
    <이름>_labels.csv    입력 행마다 평활화 위치, 거리, motion, 진행 중인 행동
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_classifier import BehaviorClassifier, Config, TargetObs  # noqa: E402

LABEL_COLUMNS = ["session", "time_s", "frame_id", "target_id", "track", "x", "y", "x_s", "y_s",
                 "distance_m", "doppler_mean", "motion", "raw_motion", "behaviors",
                 "speed", "vr", "radial_ratio", "doppler_win", "moving", "path", "straightness",
                 "reversals"]


def _float(v, default=math.nan) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:          # 엑셀로 저장한 CSV (CP949)
        text = raw.decode("cp949")
    import io
    with io.StringIO(text, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return rows
    need = {"time_s", "target_id", "x", "y"}
    missing = need - set(rows[0])
    if missing:
        raise ValueError(f"{path.name}: 필요한 열이 없습니다: {sorted(missing)}")
    for r in rows:
        r.setdefault("session", "Session #0")
        r["_t"] = _float(r["time_s"])
        r["_id"] = int(float(r["target_id"]))
        r["_frame"] = r.get("frame_id", r["time_s"])
    return rows


def classify_rows(rows: list[dict], cfg: Config) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    labels: list[dict] = []
    # 세션마다 분류기를 새로 만든다 (세션마다 시간이 0초부터 다시 시작)
    for session, srows in groupby(rows, key=lambda r: r["session"]):
        srows = sorted(srows, key=lambda r: (r["_t"], r["_id"]))
        clf = BehaviorClassifier(cfg, session=session)
        for (t, _frame), frows in groupby(srows, key=lambda r: (r["_t"], r["_frame"])):
            frows = list(frows)
            obs = [TargetObs(target_id=r["_id"], x=_float(r["x"]), y=_float(r["y"]),
                             n_points=int(float(r["n_points"])) if r.get("n_points") else None,
                             doppler=_float(r.get("doppler_mean")))
                   for r in frows]
            events += clf.update(t, obs)
            snap = clf.snapshot()
            for r in frows:
                s = snap.get(r["_id"], {})
                labels.append({
                    "session": session, "time_s": r["time_s"], "frame_id": r.get("frame_id", ""),
                    "target_id": r["_id"], "track": s.get("track", ""),
                    "x": r["x"], "y": r["y"],
                    "x_s": _r(s.get("x_s")), "y_s": _r(s.get("y_s")), "distance_m": _r(s.get("r")),
                    "doppler_mean": r.get("doppler_mean", ""),
                    "motion": s.get("motion", "filtered"), "raw_motion": s.get("raw_motion", ""),
                    "behaviors": "|".join(s.get("behaviors", [])),
                    "speed": _r(s.get("speed")), "vr": _r(s.get("vr")),
                    "radial_ratio": _r(s.get("radial_ratio")), "doppler_win": _r(s.get("doppler_win")),
                    "moving": _r(s.get("moving")), "path": _r(s.get("path")),
                    "straightness": _r(s.get("straightness")), "reversals": s.get("reversals", ""),
                })
        events += clf.flush()
    return events, labels


def _r(v, nd=3):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, nd)


def summarize(events: list[dict]) -> str:
    """시작/끝 이벤트를 짝지어 한 줄씩 보여준다."""
    ends = {e["event_id"]: e for e in events if e["phase"] == "end"}
    lines = []
    for e in events:
        if e["phase"] != "start":
            continue
        end = ends.get(e["event_id"])
        span = (f"{e['start_s']:7.1f}s ~ {end['end_s']:7.1f}s ({end['duration_s']:5.1f}초)"
                if end else f"{e['start_s']:7.1f}s ~ (진행 중)")
        dist = (f"거리 {end['start_distance_m']:.2f} → {end['end_distance_m']:.2f} m"
                if end and end.get("end_distance_m") is not None else f"거리 {e['distance_m']:.2f} m")
        lines.append(f"  {e['session']:<12} 트랙 {e['track_id']:<3} {e['type']:<8} {span}  {dist}"
                     f"  (레이더 ID {e['target_ids']})")
    return "\n".join(lines) if lines else "  (이벤트 없음)"


def run(path: Path, cfg: Config, out_dir: Path | None) -> None:
    rows = read_rows(path)
    events, labels = classify_rows(rows, cfg)
    stem = path.stem[:-len("_tracks_raw")] if path.stem.endswith("_tracks_raw") else path.stem
    out = out_dir or path.parent
    out.mkdir(parents=True, exist_ok=True)
    ev_path = out / f"{stem}_events.json"
    lb_path = out / f"{stem}_labels.csv"
    ev_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(lb_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(labels)
    print(f"▶ {path.name}  ({len(rows)}행)")
    print(summarize(events))
    print(f"  저장: {ev_path.name}, {lb_path.name}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="트랙 CSV → 행동 이벤트(JSON) + 라벨(CSV)")
    ap.add_argument("csv", nargs="*", type=Path, help="<이름>_tracks_raw.csv 파일")
    ap.add_argument("--config", type=Path, help="설정 JSON (없으면 기본값)")
    ap.add_argument("--out-dir", type=Path, help="결과 폴더 (없으면 CSV와 같은 폴더)")
    ap.add_argument("--dump-config", type=Path, metavar="PATH", help="기본 설정을 JSON으로 저장하고 끝내기")
    args = ap.parse_args()

    if args.dump_config:
        Config().save(args.dump_config)
        print(f"기본 설정 저장: {args.dump_config}")
        return
    if not args.csv:
        ap.error("CSV 파일을 하나 이상 넣어주세요.")
    cfg = Config.load(args.config) if args.config else Config()
    for p in args.csv:
        run(p, cfg, args.out_dir)


if __name__ == "__main__":
    main()
