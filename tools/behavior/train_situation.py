"""실제 녹화로 상황 결정 트리 학습하기

정답 표(labels.csv)에 적힌 녹화로 상황 트리를 학습하고, split이 test인 녹화로 판정해본 뒤
모델을 저장합니다. 저장한 모델은 situation_viewer.pyw의 '모델 열기...'로 불러와서 재생해볼 수 있습니다.

실행 (sg-radar-tools 폴더에서):
    python tools\\behavior\\train_situation.py recordings\\labels.csv
    python tools\\behavior\\train_situation.py recordings\\labels.csv --with-sim     # 가짜 데이터도 섞어서

정답 표 (CSV, UTF-8 또는 엑셀로 저장한 CSV). 파일 경로는 정답 표가 있는 폴더 기준입니다.
    file,situation,split,decide_s,target_id,session
    split/1-1.pcr,통과,train,2,,
    split/2-3.pcr,배회,test,10,,

    file        .pcr 또는 <이름>_tracks_raw.csv
    situation   통과 / 귀가 / 외출 / 방문/배달 / 장시간 체류 / 배회 / 반복 접근 / 도어락 조작 시도 / 판단 중
    split       train(학습) 또는 test(판정만 해보기)
    decide_s    트랙이 잡힌 뒤 몇 초부터 그 상황이라고 말할 수 있는지. 그 전은 '판단 중'으로 학습
                (비우면 처음부터 그 상황으로 학습)
    target_id   그 상황을 한 사람의 레이더 ID (여러 개면 공백으로 구분). 비우면 거리 조건에 맞는 트랙 전부
    session     파일 안의 세션 이름. 비우면 모든 세션

트랙 고르기: target_id를 비우면 가장 가까울 때 거리가 --max-r(기본 5m) 이하인 트랙만 씁니다.
멀리서 계속 잡히는 벽·물체 같은 트랙을 학습에서 빼기 위해서입니다.
움직임 트리는 가짜 데이터로 학습한 것을 그대로 쓰고, 상황 트리만 새로 학습합니다.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import situation_core as C  # noqa: E402
import scenario_sim as S  # noqa: E402

DEFAULT_OUT = C.HERE / "models" / "situation_trees_real.pkl"


def read_labels(path: Path) -> list[dict]:
    with C.open_csv_text(path) as f:      # UTF-8, 엑셀 저장(CP949) 둘 다 읽음
        rows = [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]
    valid = set(S.SITUATIONS) | {S.UNDECIDED}
    for r in rows:
        r["situation"] = r["situation"].strip()
        if r["situation"] not in valid:
            raise SystemExit(f"알 수 없는 상황 이름: {r['situation']!r}\n쓸 수 있는 이름: {', '.join(sorted(valid))}")
        r["split"] = (r.get("split") or "train").strip().lower()
        if r["split"] not in ("train", "test"):
            raise SystemExit(f"split은 train 또는 test여야 합니다: {r['split']!r}")
    return rows


def labeled_tracks(label: dict, base: Path, max_r: float, motion_tree) -> tuple[list, list]:
    """정답 표 한 줄 → (정답을 붙인 트랙들, 뺀 트랙 설명들)"""
    path = (base / label["file"].strip()).resolve()
    if not path.is_file():
        raise SystemExit(f"파일이 없습니다: {path}")
    rows = C.load_rows(path)
    if (label.get("session") or "").strip():
        rows = [r for r in rows if r.get("session") == label["session"].strip()]
    tracks = C.tracks_from_rows(rows)
    ids = {int(x) for x in (label.get("target_id") or "").split()}
    decide = float(label["decide_s"]) if (label.get("decide_s") or "").strip() else 0.0
    kept, skipped = [], []
    for tr in tracks:
        C._prepare(tr, motion_tree)
        if ids and tr.target_id not in ids:
            skipped.append(f"ID{tr.target_id}(정답 표에 없는 ID)")
            continue
        if not ids and tr.r.min() > max_r:
            skipped.append(f"ID{tr.target_id}(가장 가까울 때 {tr.r.min():.1f}m)")
            continue
        tr.truth_final = label["situation"]
        tr.truth_situation = [label["situation"] if t >= decide else S.UNDECIDED for t in tr.t]
        tr.source = path.name
        kept.append(tr)
    return kept, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="실제 녹화로 상황 결정 트리 학습")
    ap.add_argument("labels", type=Path, help="정답 표 CSV")
    ap.add_argument("--with-sim", action="store_true", help="가짜 데이터(scenario_sim)도 학습에 섞기")
    ap.add_argument("--max-r", type=float, default=5.0, help="이 거리 안까지 들어온 트랙만 사용 (m)")
    ap.add_argument("--depth", type=int, default=4, help="트리 최대 깊이 (데이터가 적으면 작게)")
    ap.add_argument("--min-leaf", type=int, default=8, help="잎 하나에 필요한 최소 샘플 수")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="모델 저장 경로")
    args = ap.parse_args()

    from sklearn.tree import DecisionTreeClassifier, export_text

    base_models = C.load_or_train(on_status=print)      # 움직임 트리 (가짜 데이터로 학습된 것)
    motion_tree = base_models.motion

    labels = read_labels(args.labels)
    base = args.labels.resolve().parent
    train_trs, test_trs = [], []
    print("정답 표 읽는 중...")
    for lb in labels:
        trs, skipped = labeled_tracks(lb, base, args.max_r, motion_tree)
        (train_trs if lb["split"] == "train" else test_trs).extend(trs)
        desc = ", ".join(f"ID{t.target_id}({t.t[-1]:.1f}초)" for t in trs) or "없음"
        print(f"  [{lb['split']:<5}] {lb['file']:<22} {lb['situation']:<8} 사용: {desc}"
              + (f"  /  뺌: {', '.join(skipped)}" if skipped else ""))
    if not train_trs:
        raise SystemExit("학습할 트랙이 없습니다.")

    X, y = [], []
    for tr in train_trs:
        xs, ys = C._snapshots(tr)
        X += xs
        y += ys
    n_real = len(X)
    if args.with_sim:
        print("가짜 데이터 섞는 중...")
        for name, rows in S.training_episodes(30, 0):
            for tr in C.tracks_from_rows([dict(r, session=name) for r in rows]):
                C._prepare(tr, motion_tree)
                xs, ys = C._snapshots(tr)
                X += xs
                y += ys

    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=args.min_leaf,
                                  class_weight="balanced", random_state=0)
    tree.fit(np.array(X), y)
    models = C.Models(motion_tree, tree, "")

    # ------------------------------------------------------------ 결과
    lines = [f"※ 실제 녹화로 학습한 상황 트리 ({args.labels.name}"
             + (", 가짜 데이터 포함" if args.with_sim else "") + ")",
             f"학습 트랙 {len(train_trs)}개, 학습 샘플 {len(X)}개 (실제 {n_real}개)",
             "학습 샘플 정답 분포: " + ", ".join(f"{k} {v}" for k, v in Counter(y[:n_real]).items()), ""]

    def evaluate(trs, title):
        if not trs:
            return
        C.analyze(trs, models)
        lines.append(f"[{title}]")
        snap_ok = snap_n = 0
        by_file: dict[str, list] = {}
        for tr in trs:
            by_file.setdefault(tr.source, []).append(tr)
            xs, ys = C._snapshots(tr)
            pred = tree.predict(np.array(xs[:-3]))
            snap_ok += int(np.sum(pred == np.array(ys[:-3])))
            snap_n += len(pred)
        for fname, ftrs in by_file.items():
            truth = ftrs[0].truth_final
            lines.append(f"  {fname}  (정답: {truth})")
            for tr in ftrs:
                mark = "O" if tr.final_pred == truth else "X"
                cams = ", ".join(f"{c['t'] - tr.t0:.1f}초 {c['situation']}" for c in tr.cameras) or "없음"
                share = Counter(str(p) for p in tr.sit_pred).most_common(2)
                lines.append(f"    ID{tr.target_id:<3} {tr.t[-1]:5.1f}초  최종 {tr.final_pred} {tr.final_prob:.0%} [{mark}]"
                             f"  / 진행 중 판정 {', '.join(f'{k} {v / len(tr.t):.0%}' for k, v in share)}"
                             f"  / 촬영 {cams}")
            should = truth in S.ABNORMAL
            fired = any(tr.cameras for tr in ftrs)
            lines.append(f"    → 파일 단위 촬영: {'함' if fired else '안 함'} "
                         f"({'맞음' if fired == should else '틀림'}: {'이상이라 촬영해야 함' if should else '정상이라 촬영하면 안 됨'})")
        lines.append(f"  0.5초 스냅샷 정확도 ('판단 중' 포함): {snap_ok}/{snap_n} = {snap_ok / max(snap_n, 1):.0%}")
        lines.append("")

    evaluate(train_trs, "학습 데이터에 다시 돌려본 결과 (참고용, 당연히 잘 나옴)")
    evaluate(test_trs, "테스트 데이터 결과")
    ko = [C.FEATURE_KO[n] for n in C.SITUATION_FEATURES]
    lines += ["=" * 60, "상황 트리", "=" * 60, export_text(tree, feature_names=ko, decimals=2)]
    models.report = "\n".join(lines)
    print("\n" + models.report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(models, f)
    print(f"모델 저장: {args.out}\n→ situation_viewer.pyw에서 '모델 열기...'로 불러오세요.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
