"""녹화(.pcr)의 레이더 타겟마다 높이(z)·움직임·도플러를 뽑아서 사람과 사물을 가를 기준을 찾는다.

실행:
    python tools/behavior/target_z_report.py recordings/labels_1111.csv
        정답 표에 target_id가 적힌 녹화만 봄. 적힌 ID = 사람, 나머지 ID = 사물.
        측정 영역 안에서 10프레임 이상 잡힌 타겟만 (판정에 들어가는 타겟과 같은 기준).
        타겟마다 한 줄 + 값마다 사람/사물 분포 + 한 기준으로 가장 잘 가르는 값.
    python tools/behavior/target_z_report.py a.pcr b.pcr ...
        정답 없이 타겟마다 한 줄만.

z는 pcr에 저장된 타겟 박스(그 프레임에 그 타겟으로 잡힌 포인트들의 z 범위)의 중앙값입니다.
센서 설정(pos_z, pitch)에 따라 기준이 달라질 수 있어서 같은 설치에서 찍은 녹화끼리 비교해야 합니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import situation_core as C  # noqa: E402
import train_situation as T  # noqa: E402

FEATS = [  # (이름, 설명)
    ("dur", "잡힌 시간(s)"),
    ("reach", "처음 위치에서 가장 멀리 간 거리(m)"),
    ("path", "경로 길이(m)"),
    ("speed", "평균 속력(m/s, 경로/시간)"),
    ("minz", "z 바닥 중앙값(m)"),
    ("maxz", "z 꼭대기 중앙값(m)"),
    ("height", "키 = maxz-minz 중앙값(m)"),
    ("npts", "포인트 수 중앙값"),
    ("dop_std", "도플러 표준편차 중앙값"),
    ("dop_abs", "도플러 절댓값 평균 중앙값"),
]


def load_arrays(path: Path) -> dict:
    """situation_core.rows_from_pcr와 같은 방식으로 to_numpy_gui를 불러옴 (박스(z)까지 쓰려고 배열을 직접 받음)"""
    import importlib.machinery
    import importlib.util
    src = HERE.parent / "pcr" / "to_numpy_gui.pyw"
    if "to_numpy_gui" in sys.modules:
        tn = sys.modules["to_numpy_gui"]
    else:
        loader = importlib.machinery.SourceFileLoader("to_numpy_gui", str(src))
        spec = importlib.util.spec_from_loader("to_numpy_gui", loader)
        tn = importlib.util.module_from_spec(spec)
        sys.modules["to_numpy_gui"] = tn
        loader.exec_module(tn)
    return tn.pcr_to_arrays(path, with_points=False)


def target_stats(path: Path, session: str = "") -> list[dict]:
    """타겟(세션, 레이더 ID)마다 값 dict. 측정 영역 안 프레임만 씀."""
    a = load_arrays(path)
    spec = C.pcr_spec(path)
    region = C.Region.from_spec(spec) if spec else None
    tg, box = a["targets"], a["target_box"]
    npt, dop, sess = a["target_n_points"], a["target_doppler"], a["target_session"]
    names = {int(i): str(n) for i, n in zip(a["session_ids"], a["session_names"])}
    out = []
    for sid in np.unique(sess):
        sname = names.get(int(sid), str(sid))
        if session and sname != session:
            continue
        for tid in np.unique(tg[sess == sid, 2]).astype(int):
            m = np.where((sess == sid) & (tg[:, 2] == tid))[0]
            m = m[npt[m] >= C.MIN_POINTS]
            if region is not None:
                m = m[region.contains(tg[m, 3], tg[m, 4])]
            if len(m) < 10:              # clip_to_region과 같은 기준 (이보다 적으면 판정에 안 들어감)
                continue
            t, x, y = tg[m, 0], tg[m, 3], tg[m, 4]
            dur = float(t[-1] - t[0])
            path_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
            minz, maxz = float(np.nanmedian(box[m, 4])), float(np.nanmedian(box[m, 5]))
            out.append({
                "file": path.name, "session": sname, "id": int(tid), "frames": len(m),
                "x0": float(x[0]), "y0": float(y[0]),
                "dur": dur, "reach": float(np.max(np.hypot(x - x[0], y - y[0]))), "path": path_len,
                "speed": path_len / dur if dur > 0 else 0.0,
                "minz": minz, "maxz": maxz, "height": maxz - minz,
                "npts": float(np.median(npt[m])),
                "dop_std": float(np.nanmedian(dop[m, 2])), "dop_abs": float(np.nanmedian(dop[m, 1])),
            })
    return out


def print_rows(rows: list[dict]) -> None:
    print(f"{'파일':<10} {'ID':>3} {'구분':<4} {'프레임':>5} " + " ".join(f"{k:>7}" for k, _ in FEATS)
          + f" {'첫위치':>12}")
    for r in rows:
        kind = r.get("kind", "")
        first = "({:.1f},{:.1f})".format(r["x0"], r["y0"])
        print(f"{r['file'][:10]:<10} {r['id']:>3} {kind:<4} {r['frames']:>5} "
              + " ".join(f"{r[k]:>7.2f}" for k, _ in FEATS) + f" {first:>12}")


def best_split(people: np.ndarray, objects: np.ndarray) -> tuple[float, str, float, int, int]:
    """한 기준(값 >= 기준이면 사람, 또는 <= 이면 사람)으로 가장 잘 가르는 값.
    돌려줌: (기준, 방향, 균형 정확도, 사람 중 맞은 수, 사물 중 맞은 수)"""
    vals = np.unique(np.concatenate([people, objects]))
    cuts = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
    best = (float("nan"), ">=", 0.0, 0, 0)
    for c in cuts:
        for d in (">=", "<="):
            hp = int(np.sum(people >= c) if d == ">=" else np.sum(people <= c))
            ho = int(np.sum(objects < c) if d == ">=" else np.sum(objects > c))
            acc = (hp / len(people) + ho / len(objects)) / 2
            if acc > best[2]:
                best = (float(c), d, acc, hp, ho)
    return best


def from_labels(labels_path: Path) -> None:
    labels = T.read_labels(labels_path)
    base = labels_path.resolve().parent
    rows, seen = [], set()
    for lb in labels:
        ids = {int(v) for v in (lb.get("target_id") or "").split()}
        if not ids:
            continue                     # ID가 안 적힌 녹화는 누가 사람인지 몰라서 뺌
        path = (base / lb["file"].strip()).resolve()
        key = (str(path), (lb.get("session") or "").strip())
        if key in seen:                  # 같은 파일이 여러 줄이면 ID를 합쳐야 해서 아래에서 처리
            continue
        seen.add(key)
        all_ids = set()
        for lb2 in labels:
            if (base / lb2["file"].strip()).resolve() == path and (lb2.get("session") or "").strip() == key[1]:
                all_ids |= {int(v) for v in (lb2.get("target_id") or "").split()}
        if not path.is_file() or path.suffix.lower() != ".pcr":
            print(f"  (건너뜀: {lb['file']} - pcr 파일이 아니거나 없음)")
            continue
        for r in target_stats(path, key[1]):
            r["kind"] = "사람" if r["id"] in all_ids else "사물"
            rows.append(r)
    print(f"===== {labels_path.name}: 타겟 {len(rows)}개 (사람 {sum(r['kind'] == '사람' for r in rows)}, "
          f"사물 {sum(r['kind'] == '사물' for r in rows)}) =====")
    print_rows(sorted(rows, key=lambda r: (r["kind"] != "사물", r["file"], r["id"])))
    people = [r for r in rows if r["kind"] == "사람"]
    objects = [r for r in rows if r["kind"] == "사물"]
    if not people or not objects:
        print("\n사람이나 사물이 한쪽뿐이라 기준을 찾을 수 없습니다.")
        return
    print("\n===== 값마다 분포 (10% / 중앙값 / 90%) 와 한 기준으로 가르기 =====")
    print(f"{'값':<9} {'설명':<28} {'사람 10/50/90%':>22} {'사물 10/50/90%':>22}   가장 잘 가르는 기준")
    res = []
    for k, desc in FEATS:
        p = np.array([r[k] for r in people], dtype=float)
        o = np.array([r[k] for r in objects], dtype=float)
        p, o = p[np.isfinite(p)], o[np.isfinite(o)]
        if not len(p) or not len(o):
            continue
        q = lambda v: "/".join(f"{x:.2f}" for x in np.percentile(v, [10, 50, 90]))  # noqa: E731
        c, d, acc, hp, ho = best_split(p, o)
        res.append((acc, k))
        print(f"{k:<9} {desc:<28} {q(p):>22} {q(o):>22}   {k} {d} {c:.2f} 이면 사람: "
              f"사람 {hp}/{len(p)}, 사물 {ho}/{len(o)} 맞음 (균형 정확도 {acc:.0%})")
    res.sort(reverse=True)
    print("\n가장 잘 가르는 값: " + ", ".join(f"{k} {a:.0%}" for a, k in res[:4]))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        return
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".csv"):
        from_labels(Path(sys.argv[1]))
        return
    for p in sys.argv[1:]:
        print(f"\n===== {Path(p).name} =====")
        print_rows(target_stats(Path(p)))


if __name__ == "__main__":
    main()
