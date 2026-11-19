"""실제 녹화로 상황 결정 트리 학습하기

정답 표(labels.csv)에 적힌 녹화로 상황 트리를 학습하고, split이 test인 녹화로 판정해본 뒤
모델을 저장합니다. 저장한 모델은 situation_viewer.pyw의 '모델 열기...'로 불러와서 재생해볼 수 있습니다.

실행 (sg-radar-tools 폴더에서):
    python tools\\behavior\\train_situation.py recordings\\labels.csv
    python tools\\behavior\\train_situation.py recordings\\labels.csv --with-sim     # 가짜 데이터도 섞어서
    python tools\\behavior\\train_situation.py recordings\\labels.csv --cv 5 --out tools\\behavior\\models\\situation_trees_1111.pkl
        → 5겹 교차 검증: situation_trees_1111_1.pkl ~ _5.pkl을 만들고 결과를 한 번에 출력
    (화면으로 하려면 train_situation_gui.pyw 더블클릭)

정답 표 (CSV, UTF-8 또는 엑셀로 저장한 CSV). 파일 경로는 정답 표가 있는 폴더 기준입니다.
    file,situation,split,decide_s,target_id,session
    split/1-1.pcr,통과,train,2,,
    split/2-3.pcr,배회,test,10,,

    file        .pcr 또는 <이름>_tracks_raw.csv
    situation   통과 / 귀가 / 외출 / 방문 / 배달(=배달/택배) / 장시간 체류 / 배회 / 반복 접근 / 판단 중
    split       train(학습) 또는 test(판정만 해보기)
    decide_s    트랙이 잡힌 뒤 몇 초부터 그 상황이라고 말할 수 있는지. 그 전은 '판단 중'으로 학습
                (비우면 처음부터 그 상황으로 학습)
    target_id   그 상황을 한 사람의 레이더 ID. 여러 개면 공백으로 구분 (예: 3 7). 비우면 측정 영역 안의 트랙 전부
                한 파일에 서로 다른 상황의 사람이 있으면 같은 파일을 두 줄로 쓰고 줄마다 ID와 상황을 다르게
    session     파일 안의 세션 이름. 비우면 모든 세션

측정 영역: 기본으로 녹화 파일(.pcr) 헤더에 저장된 Radar Studio의 측정 설정(x·y 범위, 시야각)을 읽어서
그 안의 데이터만 씁니다. 영역 밖에서 잡힌 벽·물체 트랙은 빠지고, 이 영역은 모델에 같이 저장되어
뷰어도 같은 영역으로 판정합니다. --region으로 직접 지정하거나 끌 수 있습니다.
움직임 트리는 가짜 데이터로 학습한 것을 그대로 쓰고, 상황 트리만 새로 학습합니다.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import situation_core as C  # noqa: E402
import scenario_sim as S  # noqa: E402

DEFAULT_OUT = C.HERE / "models" / "situation_trees_real.pkl"


class TrainError(RuntimeError):
    """정답 표나 녹화에 문제가 있어서 학습을 진행할 수 없을 때"""


def read_labels(path: Path) -> list[dict]:
    with C.open_csv_text(path) as f:      # UTF-8, 엑셀 저장(CP949) 둘 다 읽음
        rows = [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]
    if not rows:
        raise TrainError("정답 표에 줄이 없습니다 (file 열이 비어 있음)")
    valid = set(S.SITUATIONS) | {S.UNDECIDED}
    for k, r in enumerate(rows, start=2):
        r["situation"] = (r.get("situation") or "").strip()
        r["situation"] = S.ALIASES.get(r["situation"], r["situation"])
        if r["situation"] not in valid:
            raise TrainError(f"{k}번째 줄: 알 수 없는 상황 이름 {r['situation']!r}\n"
                             f"쓸 수 있는 이름: {', '.join(sorted(valid))}")
        r["split"] = (r.get("split") or "train").strip().lower()
        if r["split"] not in ("train", "test"):
            raise TrainError(f"{k}번째 줄: split은 train 또는 test여야 합니다 ({r['split']!r})")
        d = (r.get("decide_s") or "").strip()
        try:
            r["_decide"] = float(d) if d else 0.0
        except ValueError:
            raise TrainError(f"{k}번째 줄: decide_s가 숫자가 아닙니다 ({d!r})") from None
    return rows


def file_region(path: Path, region):
    """region이 "auto"면 녹화 파일 헤더의 측정 설정, Region이면 그대로, None이면 영역 안 씀"""
    if region == "auto":
        spec = C.pcr_spec(path) if path.suffix.lower() == ".pcr" else None
        return C.Region.from_spec(spec) if spec else None
    return region


def labeled_tracks(label: dict, base: Path, region, max_r, motion_tree):
    """정답 표 한 줄 → (정답을 붙인 트랙들, 뺀 트랙 설명들, 쓴 영역)"""
    path = (base / label["file"].strip()).resolve()
    if not path.is_file():
        raise TrainError(f"파일이 없습니다: {path}")
    reg = file_region(path, region)
    rows = C.load_rows(path)
    if (label.get("session") or "").strip():
        rows = [r for r in rows if r.get("session") == label["session"].strip()]
    raw = C.tracks_from_rows(rows)
    # 판정할 때와 똑같이: 영역으로 자르기 → 멈췄다 다시 움직인 같은 사람 잇기 → 특징 계산
    people, outside = C.assemble(raw, reg, motion_tree)
    ids = {int(x) for x in (label.get("target_id") or "").split()}
    decide = label["_decide"]
    kept = []
    skipped = [f"{C.track_name(t)}(측정 영역 밖)" for t in outside]
    for tr in people:
        if ids and not (set(tr.target_ids) & ids):
            skipped.append(f"{C.track_name(tr)}(정답 표에 없는 ID)")
            continue
        if not ids and max_r and tr.r.min() > max_r:
            skipped.append(f"{C.track_name(tr)}(가장 가까울 때 {tr.r.min():.1f}m)")
            continue
        tr.out_of_range = False
        tr.truth_final = label["situation"]
        tr.truth_situation = [label["situation"] if t >= decide else S.UNDECIDED for t in tr.t]
        tr.source = label["file"].strip()
        tr.split = label["split"]
        kept.append(tr)
    return kept, skipped, reg


@dataclass
class TrainResult:
    models: object
    report: str
    out_path: Path
    rows: list           # 트랙별 결과 (GUI 표용)
    summary: dict        # {"train": {...}, "test": {...}}
    test_files: list     # test 녹화 경로 (뷰어로 열기용)
    warnings: list = None  # 정답이 상황 정의 조건에 안 맞는 트랙 설명
    base: Path = None      # 정답 표 폴더 (파일 경로 기준)


MARK_TEXT = {"O": "맞음", "△": "비슷함", "X": "틀림"}   # △: 구분이 어려운 쌍(C.SIMILAR_PAIRS), 오류로 안 셈


def evaluate(trs: list, models) -> tuple[list, dict, list]:
    """트랙들을 판정해서 (표 행, 요약, 글 줄)을 돌려준다."""
    C.predict(trs, models)          # 학습 때 이미 잇기·특징 계산을 마친 트랙에 판정만
    rows, lines = [], []
    snap_ok = snap_n = 0
    by_file: dict[str, list] = {}
    for tr in trs:
        by_file.setdefault(tr.source, []).append(tr)
        idx = C.snap_index(tr)              # 진행 중 판정(정의 조건 적용 후)을 0.5초마다 정답과 비교
        truth = [C.snap_truth(tr, i) for i in idx]
        snap_ok += sum(str(tr.sit_pred[i]) == t for i, t in zip(idx, truth))
        snap_n += len(idx)
    trk_ok = file_ok = rec_ok = trk_near = rec_near = 0
    for fname, ftrs in by_file.items():
        truths = list(dict.fromkeys(C.final_truth(tr) or tr.truth_final for tr in ftrs))
        # 녹화 정답: 이상 행동이 있으면 그것(1순위), 없으면 정상 행동
        expect = [t for t in truths if t in S.ABNORMAL] or truths
        should = any(t in S.ABNORMAL for t in truths)
        fired = any(tr.cameras for tr in ftrs)
        cam_good = fired == should
        file_ok += cam_good
        verdict, vprob, _ = C.recording_verdict(ftrs)
        vlabels = C.recording_labels(ftrs)           # 이상 행동 동점이면 여러 개 (모두 판정)
        verdict = " + ".join(vlabels)
        rec_mark = min((C.match_labels(vlabels, e) for e in expect), key="O△X".index)
        rec_good = rec_mark != "X"          # △(구분이 어려운 쌍)는 오류로 세지 않음
        rec_ok += rec_good
        rec_near += rec_mark == "△"
        lines.append(f"  {fname}  (정답: {', '.join(truths)})  → 녹화 판정 {verdict} {vprob:.0%} "
                     f"[{rec_mark}]")
        for k, tr in enumerate(ftrs):
            truth = C.final_truth(tr) or tr.truth_final    # 정의 조건에 안 맞으면 가장 비슷한 상황
            mark = C.match_labels(C.final_labels(tr), truth)   # 동점 판정 중 하나라도 맞으면 맞음
            good = mark != "X"
            trk_ok += good
            trk_near += mark == "△"
            cams = ", ".join(f"{c['t'] - tr.t0:.1f}초 {c['situation']}" for c in tr.cameras) or "-"
            share = Counter(str(p) for p in tr.sit_pred).most_common(2)
            share_txt = ", ".join(f"{s_} {v / len(tr.t):.0%}" for s_, v in share)
            why = getattr(tr, "rule_note", "") or " / ".join(C.explain(models, tr.final_feat, "final")[:3])
            beh = ", ".join(f"{b['situation']} {b['prob']:.0%}" + ("(끝)" if b["source"] == "끝날 때" else "")
                            for b in getattr(tr, "behaviors", []))
            rows.append({
                "why": why, "end": f"{tr.end_pred} {tr.end_prob:.0%}", "behaviors": beh,
                "file_verdict": ("" if k else f"{verdict} {vprob:.0%} ({MARK_TEXT[rec_mark]})"),
                "file_ok": rec_good, "file_mark": rec_mark, "verdict": verdict, "rec_truth": "/".join(expect),
                "first": k == 0,
                "split": tr.split, "file": fname, "truth": truth, "track": C.track_name(tr),
                # 표시용: 원래 정답, 정의 조건에 안 맞아 바꿨으면 '원래 → 바꾼 것'
                "truth_label": tr.truth_final if truth == tr.truth_final else f"{tr.truth_final} → {truth}",
                "dur": f"{tr.t[-1]:.1f}초", "pred": C.final_text(tr), "pred_main": tr.final_pred,
                "prob": f"{tr.final_prob:.0%}",
                "ok": mark, "progress": share_txt, "camera": cams,
                "file_cam": ("" if k else (f"{'촬영함' if fired else '촬영 안 함'} "
                                           f"({'맞음' if cam_good else '틀림'})")),
            })
            lines.append(f"    {C.track_name(tr):<8} {tr.t[-1]:5.1f}초  최종 {C.final_text(tr)} {tr.final_prob:.0%} "
                         f"[{mark}]  / 나온 행동 {beh or '-'}  / 촬영 {cams}")
            if mark != "O":
                lines.append(f"             근거: {why}")
        lines.append(f"    → 파일 단위 촬영: {'함' if fired else '안 함'} "
                     f"({'맞음' if cam_good else '틀림'}: "
                     f"{'이상이라 촬영해야 함' if should else '정상이라 촬영하면 안 됨'})")
    summary = {"tracks": (trk_ok, len(trs)), "files": (file_ok, len(by_file)),
               "snapshots": (snap_ok, snap_n), "recordings": (rec_ok, len(by_file)),
               "near": (trk_near, rec_near)}     # 맞음에 포함된 △(구분이 어려운 쌍) 수: (트랙, 녹화)
    lines.append(f"  녹화 판정 {rec_ok}/{len(by_file)} (△ {rec_near}), 트랙 최종 판정 {trk_ok}/{len(trs)} (△ {trk_near}), "
                 f"파일 촬영 판단 {file_ok}/{len(by_file)}, "
                 f"0.5초 스냅샷 정확도 {snap_ok}/{snap_n} = {snap_ok / max(snap_n, 1):.0%}")
    return rows, summary, lines


def train(labels_path: Path, out_path: Path = DEFAULT_OUT, region="auto", max_r: float | None = None,
          depth: int = 7, min_leaf: int = 5, with_sim: bool = False, log=print, progress=None,
          final_depth: int = 5, labels: list | None = None, cache: dict | None = None,
          eval_train: bool = True) -> TrainResult:
    """정답 표로 상황 트리를 학습·평가하고 저장한다.
    region: "auto"(녹화 파일의 측정 설정), C.Region(직접 지정), None(영역 안 씀)
    max_r : 이 거리 안까지 들어온 트랙만 사용 (None이면 안 씀)
    log(글), progress(0~1)는 화면 표시용."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    prog = progress or (lambda v: None)
    labels_path = Path(labels_path)
    out_path = Path(out_path)

    log("움직임 트리 준비 중 (처음이면 가짜 데이터로 학습해서 10초 정도 걸림)...")
    motion_tree = C.load_or_train(on_status=log).motion
    prog(0.05)

    if labels is None:
        labels = read_labels(labels_path)
    base = labels_path.resolve().parent
    train_trs, test_trs, test_files = [], [], []
    log(f"정답 표 {labels_path.name}: {len(labels)}줄")
    used_regions = []
    warnings = []           # 정답이 상황 정의 조건에 안 맞는 트랙 (최종 샘플을 학습에서 뺌)
    n_first_repeat = 0      # 반복 접근 녹화에서 처음 나타난 사람 (아직 반복이 아니라서 빼는 게 정상)
    for i, lb in enumerate(labels):
        key = (lb["file"].strip(), lb.get("target_id") or "", lb.get("session") or "", lb["situation"])
        if cache is not None and key in cache:          # 교차 검증: 같은 녹화를 여러 번 읽지 않음
            trs, skipped, reg = cache[key]
        else:
            trs, skipped, reg = labeled_tracks(lb, base, region, max_r, motion_tree)
            if cache is not None:
                cache[key] = (trs, skipped, reg)
        for t in trs:
            t.split = lb["split"]
            t.region_used = reg
        if reg is not None and reg not in used_regions:
            used_regions.append(reg)
        (train_trs if lb["split"] == "train" else test_trs).extend(trs)
        if lb["split"] == "test":
            test_files.append(str((base / lb["file"].strip()).resolve()))
        desc = ", ".join(f"{C.track_name(t)}({t.t[-1]:.1f}초)" for t in trs) or "없음"
        log(f"  [{lb['split']:<5}] {lb['file']:<12} {lb['situation']:<8} 사용: {desc}"
            + (f"  /  뺌: {', '.join(skipped)}" if skipped else ""))
        prog(0.05 + 0.75 * (i + 1) / len(labels))
    if not train_trs:
        raise TrainError("학습할 트랙이 없습니다. split이 train인 줄이 있는지, 거리 기준이 너무 작지 않은지 확인하세요.")

    sim_people = []
    if with_sim:
        log("가짜 데이터 섞는 중...")
        for name, rows in S.training_episodes(30, 0):
            trs = C.tracks_from_rows([dict(r, session=name) for r in rows])
            for p in C.assemble(trs, C.Region(**C.SIM_REGION), motion_tree)[0]:   # 에피소드 단위로 잇기
                p.truth_final = trs[0].truth_final
                sim_people.append(p)

    # 정답이 정의 조건에 안 맞는 순간 → 조건을 지키는 정답만으로 학습한 임시 트리에서 가장 비슷한 상황으로 바꿈
    log("정의 조건에 안 맞는 정답을 가장 비슷한 상황으로 바꾸는 중...")
    rt = C.relabel_tree(train_trs + sim_people, depth, min_leaf)
    C.assign_fallback(train_trs + test_trs + sim_people, rt)
    for t in train_trs + test_trs:
        bad = C.final_ok(t)
        if bad and t.truth_final == "반복 접근" and all("번 이상 나타나야" in b for b in bad):
            n_first_repeat += 1
        elif bad:
            alt = t.final_fallback or "못 정함 (사라진 시점 샘플 뺌)"
            warnings.append(f"  {t.source} {C.track_name(t)} 정답 {t.truth_final} → {alt}: {'; '.join(bad)}"
                            f"  ({C.describe_path(t, t.region_used)})")

    X, y = [], []
    for tr in train_trs:
        xs, ys = C._snapshots(tr)
        X += xs
        y += ys
    n_real = len(X)
    for p in sim_people:
        xs, ys = C._snapshots(p)
        X += xs
        y += ys

    log(f"상황 트리 학습 중... (샘플 {len(X)}개)")
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                  class_weight="balanced", random_state=0)
    tree.fit(np.array(X), y)
    log(f"최종 판정 트리 학습 중... (깊이 {final_depth})")
    final_tree = C.fit_final(train_trs + sim_people, final_depth, min_leaf)
    models = C.Models(motion_tree, tree, "")
    models.final = final_tree
    models.feature_names = list(C.SITUATION_FEATURES)
    # 판정할 때도 같은 영역·거리 기준으로 트랙을 고르도록 모델에 같이 저장
    from dataclasses import asdict
    models.region = asdict(used_regions[0]) if used_regions else None
    models.watch_r = max_r if max_r else float("inf")
    if len(used_regions) > 1:
        log("※ 녹화마다 측정 영역이 달라서, 모델에는 첫 번째 영역을 저장합니다: "
            + " / ".join(r.text() for r in used_regions))
    prog(0.85)

    log("판정해보는 중...")
    lines = [f"※ 실제 녹화로 학습한 상황 트리 ({labels_path.name}" + (", 가짜 데이터 포함" if with_sim else "") + ")",
             f"측정 영역: {used_regions[0].text() if used_regions else '안 씀'}"
             + (" (녹화 파일 설정)" if region == "auto" and used_regions else "")
             + (f", 최대 거리 {max_r}m" if max_r else ""),
             f"상황 트리 깊이 {depth}, 최종 판정 트리 깊이 {final_depth}, 잎 최소 샘플 {min_leaf}",
             "상황 정의 조건 (트리가 골라도 이 조건을 어기면 다음으로 확률 높은 상황을 고름, 학습 정답에도 적용):",
             C.rules_text(),
             C.places_text(),
             f"학습 트랙 {len(train_trs)}개, 학습 샘플 {len(X)}개 (실제 {n_real}개)",
             "학습 샘플 정답 분포: " + ", ".join(f"{k} {v}" for k, v in Counter(y[:n_real]).items()), ""]
    if n_first_repeat:
        lines += [f"반복 접근 녹화에서 {C.REPEAT_MIN_ENTRIES}번째 전까지 나타난 사람 {n_first_repeat}명은 아직 반복이 아니라서 "
                  "'반복 접근' 대신 가장 비슷한 상황으로 학습했습니다 (정상).", ""]
    if warnings:
        lines += ["[정답이 상황 정의와 안 맞는 트랙] 조건을 어기는 순간은 '→' 뒤의 가장 비슷한 상황으로 학습·평가했습니다. "
                  "정답이나 target_id를 확인해보세요."] + warnings + [""]
        for w in warnings:
            log("※ 정의와 안 맞음:" + w)
    all_rows, summary = [], {}
    for name, trs, title in (("test", test_trs, "테스트 데이터 결과"),
                             ("train", train_trs, "학습 데이터에 다시 돌려본 결과 (참고용, 당연히 잘 나옴)")):
        if not trs or (name == "train" and not eval_train):
            continue
        r_, s_, l_ = evaluate(trs, models)
        all_rows += r_
        summary[name] = s_
        lines += [f"[{title}]"] + l_ + [""]
    ko = [C.FEATURE_KO[n] for n in C.SITUATION_FEATURES]
    lines += ["=" * 60, "최종 판정 트리 (사람이 사라진 순간 / 녹화가 끝난 순간)", "=" * 60,
              export_text(final_tree, feature_names=ko, decimals=2),
              "=" * 60, "상황 트리 (진행 중 0.5초마다, 카메라 촬영 판단)", "=" * 60,
              export_text(tree, feature_names=ko, decimals=2)]
    models.report = "\n".join(lines)
    prog(0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(models, f)
    log(f"모델 저장: {out_path}")
    prog(1.0)
    return TrainResult(models, models.report, out_path, all_rows, summary, test_files,
                       [w.strip() for w in warnings], base)


# =========================================================================== 교차 검증

@dataclass
class CVResult:
    report: str
    out_path: Path          # 첫 번째 모델 (화면 표시용)
    rows: list              # 모든 폴드의 test 트랙 결과 (split = "1번" ...), model = 그 트랙을 판정한 모델
    summary: dict           # {"1번": {...}, ..., "전체": {...}}
    test_files: list        # 첫 번째 폴드의 test 녹화
    warnings: list
    base: Path
    folds: list             # [(모델 경로, test 녹화 경로들)]
    cv: bool = True


def make_folds(labels: list, k: int, seed: int = 0) -> list[int]:
    """녹화 파일 단위로 k개 폴드에 나눈다 (같은 녹화는 한 폴드에만). 상황마다 골고루 섞이도록
    상황별로 파일을 섞은 뒤 돌아가며 나눠 줌. 돌려주는 값: 정답 표 줄마다 폴드 번호(0~k-1)."""
    import random
    rng = random.Random(seed)
    by_sit: dict[str, list[str]] = {}
    for lb in labels:
        f = lb["file"].strip()
        lst = by_sit.setdefault(lb["situation"], [])
        if f not in lst and not any(f in v for v in by_sit.values() if v is not lst):
            lst.append(f)
    fold_of: dict[str, int] = {}
    nxt = 0
    for sit in sorted(by_sit, key=lambda s: len(by_sit[s])):      # 적은 상황부터 (폴드마다 하나씩 가도록)
        files = by_sit[sit][:]
        rng.shuffle(files)
        for f in files:
            fold_of[f] = nxt % k
            nxt += 1
    return [fold_of[lb["file"].strip()] for lb in labels]


def cross_validate(labels_path: Path, out_path: Path, k: int = 5, seed: int = 0, region="auto",
                   max_r: float | None = None, depth: int = 7, min_leaf: int = 5, with_sim: bool = False,
                   final_depth: int = 5, log=print, progress=None) -> CVResult:
    """k겹 교차 검증. 정답 표의 split 열은 무시하고, 녹화 파일 단위로 k묶음으로 나눠서
    한 묶음씩 test로 두고 나머지로 학습한다. 모델은 <out 이름>_1.pkl ~ _k.pkl로 저장."""
    from sklearn.metrics import confusion_matrix
    prog = progress or (lambda v: None)
    labels_path, out_path = Path(labels_path), Path(out_path)
    labels = read_labels(labels_path)
    n_files = len({lb["file"].strip() for lb in labels})
    if k < 2 or k > n_files:
        raise TrainError(f"폴드 수는 2 이상, 녹화 수({n_files}) 이하여야 합니다.")
    fold = make_folds(labels, k, seed)
    base = labels_path.resolve().parent
    stem = out_path.stem
    cache: dict = {}
    rows, summary, folds, merged = [], {}, [], {}
    lines = [f"※ {k}겹 교차 검증 ({labels_path.name}, 녹화 {n_files}개, 정답 표의 split 열은 쓰지 않음)",
             "녹화 파일 단위로 나눔 (같은 녹화가 학습과 test에 같이 들어가지 않음). 상황마다 골고루 섞음 "
             f"(섞는 순서 seed={seed}, 같은 정답 표면 매번 같은 폴드)",
             f"상황 트리 깊이 {depth}, 최종 판정 트리 깊이 {final_depth}, 잎 최소 샘플 {min_leaf}"
             + (", 가짜 데이터 포함" if with_sim else ""), ""]
    for f in range(k):
        name = f"{f + 1}번"
        lab_f = [dict(lb, split="test" if fold[j] == f else "train") for j, lb in enumerate(labels)]
        path = out_path.with_name(f"{stem}_{f + 1}{out_path.suffix or '.pkl'}")
        test_names = sorted({lb["file"].strip() for lb in lab_f if lb["split"] == "test"},
                            key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)])
        log(f"===== 폴드 {f + 1}/{k}: test {', '.join(test_names)} =====")
        res = train(labels_path, path, region, max_r, depth, min_leaf, with_sim,
                    log=lambda s: log("  " + s) if s.startswith("모델 저장") or s.startswith("※") else None,
                    progress=lambda v, f=f: prog((f + v) / k), final_depth=final_depth,
                    labels=lab_f, cache=cache, eval_train=False)
        for r in res.rows:
            r["split"] = name
            r["model"] = str(path)
            rows.append(r)
        summary[name] = res.summary.get("test")
        folds.append((path, res.test_files))
        for w in res.warnings or []:
            # "파일 트랙 정답 X → 바꾼 상황: 조건  (경로)" 를 트랙마다 한 줄로 합침 (폴드마다 바꾼 상황이 다를 수 있음)
            head, rest = w.split(" → ", 1)
            alt, tail = rest.split(": ", 1)
            merged.setdefault((head, tail), [])
            if alt not in merged[(head, tail)]:
                merged[(head, tail)].append(alt)
        (ro, rn), (to, tn), (fo, fn), (so, sn) = (res.summary["test"][key] for key in
                                                  ("recordings", "tracks", "files", "snapshots"))
        log(f"  → 폴드 {f + 1}: 녹화 판정 {ro}/{rn}, 트랙 최종 판정 {to}/{tn}, 파일 촬영 판단 {fo}/{fn}, "
            f"0.5초 판정 {so / max(sn, 1):.0%}  ({path.name})")
    warnings = [f"{head} → {' / '.join(alts)}{' (폴드마다 다름)' if len(alts) > 1 else ''}: {tail}"
                for (head, tail), alts in merged.items()]
    tot = {key: tuple(sum(summary[n][key][i] for n in summary) for i in range(2))
           for key in ("recordings", "tracks", "files", "snapshots")}
    summary["전체"] = tot
    acc = lambda p: f"{p[0]}/{p[1]} = {p[0] / max(p[1], 1):.0%}"  # noqa: E731
    lines += ["[폴드별 결과 (각 폴드의 test 녹화만)]"]
    for f in range(k):
        s = summary[f"{f + 1}번"]
        lines.append(f"  {f + 1}번  녹화 {acc(s['recordings']):<14} 트랙 {acc(s['tracks']):<14} "
                     f"파일 촬영 {acc(s['files']):<14} 0.5초 판정 {acc(s['snapshots']):<16} {folds[f][0].name}")
    lines.append(f"  전체 녹화 {acc(tot['recordings'])}, 트랙 {acc(tot['tracks'])}, 파일 촬영 {acc(tot['files'])}, "
                 f"0.5초 판정 {acc(tot['snapshots'])}")
    nt = sum(summary[f"{f + 1}번"].get("near", (0, 0))[0] for f in range(k))
    nr = sum(summary[f"{f + 1}번"].get("near", (0, 0))[1] for f in range(k))
    lines.append(f"  (맞음에 포함된 △ 구분이 어려운 쌍({', '.join('/'.join(sorted(p)) for p in C.SIMILAR_PAIRS)}): "
                 f"트랙 {nt}개, 녹화 {nr}개)")
    for key, title in (("recordings", "녹화"), ("tracks", "트랙")):
        a = [summary[f"{f + 1}번"][key][0] / max(summary[f"{f + 1}번"][key][1], 1) for f in range(k)]
        lines.append(f"  폴드별 {title} 정확도 평균 {np.mean(a):.0%} ± {np.std(a):.0%} (표준편차)")
    lines += ["", "최종 판정 = 이상 행동 1순위(진행 중 60% 이상으로 1초 넘게 이어진 것 + 끝날 때 판정), "
                  "없으면 끝날 때 판정(정상 행동). 같은 순위면 확률이 높은 것.",
              "녹화 판정 = 그 녹화의 사람들 최종 판정 중 같은 우선순위로 하나."]
    # 상황별·혼동 행렬 (모든 녹화가 한 번씩 test가 됨)
    sits = [s for s in S.SITUATIONS + [S.UNDECIDED] if any(r["truth"] == s or r["pred_main"] == s for r in rows)]
    lines += ["", "[사람(트랙)별 상황 정답률 (모든 녹화가 한 번씩 test)]"]
    for s in sits:
        rs = [r for r in rows if r["truth"] == s]
        if rs:
            ok = sum(r["ok"] != "X" for r in rs)
            wrong = Counter(r["pred"] for r in rs if r["ok"] == "X")
            near = Counter(r["pred"] for r in rs if r["ok"] == "△")
            lines.append(f"  {s:<8} {ok}/{len(rs)}"
                         + (f"   △: {', '.join(f'{p} {n}' for p, n in near.items())}" if near else "")
                         + (f"   틀린 것: {', '.join(f'{p} {n}' for p, n in wrong.items())}" if wrong else ""))
    # 동점으로 여러 개 판정한 트랙은 정답과 맞는 것(없으면 대표 판정)으로 셈
    cm_pred = [r["truth"] if r["ok"] == "O" else r["pred_main"] for r in rows]
    cm = confusion_matrix([r["truth"] for r in rows], cm_pred, labels=sits)
    lines += ["", "[최종 판정 혼동 행렬] 행: 정답, 열: 판정", "          " + " ".join(f"{s[:4]:>6}" for s in sits)]
    for s, row in zip(sits, cm):
        lines.append(f"  {s[:6]:<8}" + " ".join(f"{v:>6d}" for v in row))
    recs = [r for r in rows if r.get("first")]
    lines += ["", f"[녹화 판정] {sum(r['file_ok'] for r in recs)}/{len(recs)}"]
    for s in sits:
        rs = [r for r in recs if r["rec_truth"] == s]
        if rs:
            ok = sum(r["file_ok"] for r in rs)
            wrong = Counter(r["verdict"] for r in rs if not r["file_ok"])
            lines.append(f"  {s:<8} {ok}/{len(rs)}" + (f"   틀린 것: {', '.join(f'{p} {n}' for p, n in wrong.items())}"
                                                      if wrong else ""))
    bad = [r for r in rows if r["ok"] == "X"]
    if bad:
        lines += ["", f"[틀린 트랙 {len(bad)}개]"]
        for r in bad:
            lines.append(f"  {r['split']} {r['file']:<10} {r['track']:<10} 정답 {r['truth_label']} / 판정 {r['pred']} {r['prob']}"
                         f"   나온 행동: {r.get('behaviors') or '-'}   근거: {r.get('why', '')}")
    if warnings:
        lines += ["", "[정답이 상황 정의와 안 맞는 트랙] ('→' 뒤의 가장 비슷한 상황으로 학습)"] + ["  " + w for w in warnings]
    lines += ["", "각 폴드의 트리는 모델 파일마다 저장되어 있습니다 (뷰어에서 모델 열기 → 결정 트리 탭)."]
    report = "\n".join(lines)
    prog(1.0)
    return CVResult(report, folds[0][0], rows, summary, folds[0][1], warnings, base, folds)


def parse_region(text: str):
    t = (text or "auto").strip().lower()
    if t == "auto":
        return "auto"
    if t in ("none", "off", ""):
        return None
    try:
        v = [float(x) for x in t.split(",")]
        return C.Region(*v)
    except (TypeError, ValueError):
        raise SystemExit("--region은 auto, none 또는 xmin,xmax,ymin,ymax[,시야각[,방향]] 형식이어야 합니다")


def main() -> None:
    ap = argparse.ArgumentParser(description="실제 녹화로 상황 결정 트리 학습")
    ap.add_argument("labels", type=Path, help="정답 표 CSV")
    ap.add_argument("--with-sim", action="store_true", help="가짜 데이터(scenario_sim)도 학습에 섞기")
    ap.add_argument("--region", default="auto",
                    help="측정 영역. auto(기본, 녹화 파일에 저장된 설정) / none / xmin,xmax,ymin,ymax[,시야각[,방향]]"
                         " 예: -2,2,0,3,135")
    ap.add_argument("--max-r", type=float, default=None,
                    help="추가로 이 거리 안까지 들어온 트랙만 사용 (m). 기본은 안 씀")
    ap.add_argument("--depth", type=int, default=7, help="상황 트리(진행 중) 최대 깊이 (데이터가 적으면 작게)")
    ap.add_argument("--final-depth", type=int, default=5, help="최종 판정 트리 최대 깊이")
    ap.add_argument("--min-leaf", type=int, default=5, help="잎 하나에 필요한 최소 샘플 수")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="모델 저장 경로")
    ap.add_argument("--cv", type=int, default=0,
                    help="k겹 교차 검증 (예: 5). split 열은 무시하고 <out 이름>_1 ~ _k 모델을 만듦")
    ap.add_argument("--seed", type=int, default=0, help="교차 검증 폴드를 섞는 순서")
    args = ap.parse_args()
    if args.cv:
        try:
            res = cross_validate(args.labels, args.out, args.cv, args.seed, parse_region(args.region), args.max_r,
                                 args.depth, args.min_leaf, args.with_sim, args.final_depth)
        except TrainError as exc:
            raise SystemExit(str(exc))
        print("\n" + res.report)
        return
    try:
        res = train(args.labels, args.out, parse_region(args.region), args.max_r, args.depth,
                    args.min_leaf, args.with_sim, final_depth=args.final_depth)
    except TrainError as exc:
        raise SystemExit(str(exc))
    print("\n" + res.report)
    print("→ situation_viewer.pyw에서 '모델 열기...'로 불러오세요.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
