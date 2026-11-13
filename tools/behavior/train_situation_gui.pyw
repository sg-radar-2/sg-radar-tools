"""상황 결정 트리 학습 (GUI)

train_situation.py를 화면으로 하는 버전입니다.
  1) 정답 표 CSV를 고르면 내용과 녹화 파일이 있는지 바로 보여줍니다 (.pcr은 CSV와 같은 폴더에 있다고 봄)
  2) 모델 이름과 설정을 정하고 '학습 시작'
  3) 진행 기록, test 녹화 판정 결과(틀린 것은 근거까지), 정의와 안 맞는 정답, 학습된 트리를 탭으로 확인
  4) '뷰어에서 보기'로 방금 만든 모델과 test 녹화 전부를 situation_viewer.pyw에서 재생
     (결과 표에서 줄을 더블클릭하면 그 녹화만 바로 열림)
  교차 검증: '교차 검증'을 켜면 split 열은 무시하고 녹화를 k묶음으로 나눠 k번 학습합니다.
     모델은 <모델 이름>_1.pkl ~ _k.pkl로 저장되고, 모든 녹화의 test 결과를 결과 탭에서 한 번에 봅니다.

실행: train_situation_gui.pyw 더블클릭
필요: numpy, scikit-learn, (Python 3.14 미만이면) zstandard
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

RESULT_COLS = ("split", "file", "file_verdict", "truth", "track", "dur", "pred", "prob", "ok", "why",
               "behaviors", "end", "camera", "file_cam")
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _fatal(msg: str) -> None:
    try:
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("실행 오류", msg)
        r.destroy()
    except Exception:
        pass
    raise SystemExit(msg)


try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
except ImportError as exc:
    _fatal(f"필요한 패키지가 없습니다: {exc.name}\n\n명령 프롬프트에서 아래를 실행하세요.\n"
           "py -m pip install numpy scikit-learn zstandard")

import scenario_sim as S  # noqa: E402
import situation_core as C  # noqa: E402
import train_situation as T  # noqa: E402

UI_FONT = ("Malgun Gothic", 10) if sys.platform.startswith("win") else ("Noto Sans CJK KR", 10)
MODEL_DIR = C.HERE / "models"
RECORDINGS = C.HERE.parents[1] / "recordings"
BAD_NAME = re.compile(r'[\\/:*?"<>|]')


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class TrainApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.result: T.TrainResult | None = None
        self._build()
        default = next((p for p in (RECORDINGS / "labels_1111.csv", RECORDINGS / "labels.csv") if p.is_file()), None)
        if default:
            self.csv_var.set(str(default))
            self.load_labels()

    # ------------------------------------------------------------------ 화면
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="정답 표 CSV").grid(row=0, column=0, sticky="w")
        self.csv_var = tk.StringVar()
        e = ttk.Entry(top, textvariable=self.csv_var)
        e.grid(row=0, column=1, sticky="ew", padx=6)
        e.bind("<Return>", lambda _e: self.load_labels())
        ttk.Button(top, text="찾기...", command=self.pick_csv).grid(row=0, column=2)
        ttk.Button(top, text="다시 읽기", command=self.load_labels).grid(row=0, column=3, padx=(4, 0))

        ttk.Label(top, text="모델 이름").grid(row=1, column=0, sticky="w", pady=(6, 0))
        name_row = ttk.Frame(top)
        name_row.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.name_var = tk.StringVar(value="situation_trees_real")
        ttk.Entry(name_row, textvariable=self.name_var, width=36).pack(side="left")
        ttk.Label(name_row, text=".pkl   →  tools\\behavior\\models 폴더에 저장", foreground="#666").pack(side="left", padx=6)
        ttk.Button(top, text="모델 폴더 열기", command=self.open_models).grid(row=1, column=2, columnspan=2,
                                                                        sticky="ew", pady=(6, 0))

        reg = ttk.LabelFrame(self, text="측정 영역 (이 안의 데이터만 학습, 모델에 저장되어 뷰어도 같은 영역으로 판정)",
                             padding=(8, 4))
        reg.pack(fill="x", pady=(8, 0))
        self.region_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(reg, text="녹화 파일 설정 따르기 (Radar Studio에서 정한 범위·시야각)",
                        variable=self.region_auto, command=self._region_state).grid(row=0, column=0, columnspan=12,
                                                                                    sticky="w")
        self.region_vars = {k: tk.DoubleVar(value=v) for k, v in
                            (("xmin", -2.0), ("xmax", 2.0), ("ymin", 0.0), ("ymax", 3.0), ("fov", 135.0), ("yaw", 0.0))}
        self.region_spins = []
        for col, (k, label) in enumerate((("xmin", "x 최소"), ("xmax", "x 최대"), ("ymin", "y 최소"), ("ymax", "y 최대"),
                                          ("fov", "시야각 (°)"), ("yaw", "방향 (°)"))):
            ttk.Label(reg, text=label).grid(row=1, column=col * 2, sticky="e", padx=(0 if col == 0 else 10, 3))
            sp = ttk.Spinbox(reg, from_=-360 if k == "yaw" else (-20 if k != "fov" else 10),
                             to=360 if k in ("fov", "yaw") else 20, increment=5 if k in ("fov", "yaw") else 0.5,
                             textvariable=self.region_vars[k], width=7)
            sp.grid(row=1, column=col * 2 + 1, sticky="w")
            self.region_spins.append(sp)
        self.region_note = tk.StringVar()
        ttk.Label(reg, textvariable=self.region_note, foreground="#777",
                  font=(UI_FONT[0], 8)).grid(row=2, column=0, columnspan=12, sticky="w", pady=(2, 0))
        self._region_state()

        opt = ttk.LabelFrame(self, text="설정", padding=(8, 4))
        opt.pack(fill="x", pady=8)
        self.depth = tk.IntVar(value=7)
        self.final_depth = tk.IntVar(value=5)
        self.min_leaf = tk.IntVar(value=5)
        self.with_sim = tk.BooleanVar(value=False)
        items = [
            ("상황 트리 깊이", self.depth, 1, 14, 1, "진행 중(0.5초마다) 판정·카메라 촬영용. 질문을 몇 단계까지 할지"),
            ("최종 판정 트리 깊이", self.final_depth, 1, 14, 1, "사람이 사라진 순간 어떤 상황이었는지 고르는 트리"),
            ("잎 최소 샘플", self.min_leaf, 1, 200, 1, "판정 하나를 내리는 데 필요한 최소 샘플 수. 크면 단순해짐"),
        ]
        for col, (label, var, lo, hi, step, tip) in enumerate(items):
            f = ttk.Frame(opt)
            f.grid(row=0, column=col, sticky="nw", padx=(0, 18))
            ttk.Label(f, text=label).pack(anchor="w")
            ttk.Spinbox(f, from_=lo, to=hi, increment=step, textvariable=var, width=8).pack(anchor="w")
            ttk.Label(f, text=tip, foreground="#777", wraplength=230, font=(UI_FONT[0], 8)).pack(anchor="w")
        f = ttk.Frame(opt)
        f.grid(row=0, column=len(items), sticky="nw", padx=(0, 18))
        ttk.Checkbutton(f, text="가짜 데이터도 섞기", variable=self.with_sim).pack(anchor="w", pady=(16, 0))
        ttk.Label(f, text="녹화가 적을 때 보조로. 실제와 길이가 안 맞으면 오히려 나빠질 수 있음",
                  foreground="#777", wraplength=230, font=(UI_FONT[0], 8)).pack(anchor="w")
        self.cv_on = tk.BooleanVar(value=False)
        self.cv_k = tk.IntVar(value=5)
        f = ttk.Frame(opt)
        f.grid(row=0, column=len(items) + 1, sticky="nw")
        cvr = ttk.Frame(f)
        cvr.pack(anchor="w", pady=(14, 0))
        ttk.Checkbutton(cvr, text="교차 검증", variable=self.cv_on, command=self._cv_state).pack(side="left")
        self.cv_spin = ttk.Spinbox(cvr, from_=2, to=20, increment=1, textvariable=self.cv_k, width=4,
                                   command=self._cv_state)
        self.cv_spin.pack(side="left", padx=(6, 2))
        ttk.Label(cvr, text="겹").pack(side="left")
        self.cv_hint = tk.StringVar()
        ttk.Label(f, textvariable=self.cv_hint, foreground="#777", wraplength=260,
                  font=(UI_FONT[0], 8)).pack(anchor="w")
        self.name_var.trace_add("write", lambda *_: self._cv_state())
        self._cv_state()

        run = ttk.Frame(self)
        run.pack(fill="x")
        self.run_btn = ttk.Button(run, text="학습 시작", command=self.start)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(run, maximum=1.0)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status = tk.StringVar(value="정답 표를 고르고 '학습 시작'을 누르세요.")
        ttk.Label(run, textvariable=self.status, width=40).pack(side="left")
        self.view_btn = ttk.Button(run, text="뷰어에서 보기", command=self.open_viewer, state="disabled")
        self.view_btn.pack(side="left", padx=(6, 0))

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, pady=(8, 0))

        # 정답 표 탭
        lt = ttk.Frame(self.nb, padding=6)
        self.nb.add(lt, text="정답 표")
        self.label_summary = tk.StringVar()
        ttk.Label(lt, textvariable=self.label_summary, justify="left").pack(anchor="w", pady=(0, 4))
        cols = ("file", "situation", "split", "decide_s", "target_id", "state")
        heads = ("파일", "상황", "구분", "판단 시점(초)", "레이더 ID", "파일 상태")
        self.label_tree = self._table(lt, cols, heads, (140, 140, 70, 110, 90, 140))
        self.label_tree.tag_configure("missing", foreground="#c0161b")
        self.label_tree.tag_configure("test", background="#eef5ff")

        # 진행 기록 탭
        lg = ttk.Frame(self.nb)
        self.nb.add(lg, text="진행 기록")
        self.log = self._text(lg, ("Consolas", 10), wrap="none")

        # 결과 탭
        rs = ttk.Frame(self.nb, padding=6)
        self.nb.add(rs, text="결과")
        bar = ttk.Frame(rs)
        bar.pack(fill="x")
        self.result_summary = tk.StringVar(value="아직 학습하지 않았습니다.")
        ttk.Label(bar, textvariable=self.result_summary, font=(UI_FONT[0], 10, "bold"),
                  justify="left").pack(side="left", anchor="w")
        self.split_var = tk.StringVar(value="test")
        self.split_combo = ttk.Combobox(bar, textvariable=self.split_var, values=["test", "train", "전체"],
                                        state="readonly", width=7)
        self.split_combo.pack(side="right")
        ttk.Label(bar, text="보기").pack(side="right", padx=4)
        self.split_var.trace_add("write", lambda *_: self.fill_results())
        self.only_bad = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="틀린 것만 (노란 △ 포함)", variable=self.only_bad,
                        command=self.fill_results).pack(side="right", padx=8)
        cols = RESULT_COLS
        heads = ("구분", "파일", "녹화 판정", "정답", "사람(트랙)", "길이", "최종 판정", "확률", "판정",
                 "근거 (틀린 것)", "나온 행동 (이상 먼저)", "끝날 때 판정", "촬영", "파일 촬영 판단")
        self.res_tree = self._table(rs, cols, heads,
                                    (45, 65, 130, 85, 85, 50, 90, 45, 40, 300, 190, 95, 140, 120), pady=(6, 0))
        self.res_tree.tag_configure("good", foreground="#1b7f3b")
        self.res_tree.tag_configure("bad", foreground="#c0161b")
        # △: 구분이 어려운 쌍(방문/귀가, 배회/장시간 체류)끼리 바뀐 것. 오류로 세지 않음
        self.res_tree.tag_configure("near", foreground="#7a5c00", background="#fff3b0")
        self.res_tree.bind("<Double-1>", self.open_row)
        ttk.Label(rs, foreground="#666", justify="left", text=(
            "최종 판정: 이상 행동이 1순위(진행 중 60% 이상으로 1초 넘게 이어진 이상 행동, 끝날 때 판정 포함), "
            "없으면 끝날 때 판정(정상 행동). 같은 순위면 확률이 높은 것. 판정 = 최종 판정이 정답과 같은지 (O/X). "
            "노란 △ = 레이더로 구분이 어려운 쌍(방문/귀가, 배회/장시간 체류)끼리 바뀜, 오류로 세지 않음\n"
            "녹화 판정: 그 녹화의 사람들 최종 판정 중 같은 우선순위로 하나 (녹화 첫 줄에 표시)\n"
            "근거: 정의 조건 때문에 판정이 바뀌었으면 그 내용, 아니면 최종 판정 트리가 지나온 질문 (앞 3개)\n"
            "파일 촬영 판단: 이상 상황 녹화면 촬영했어야 맞고, 정상 상황이면 촬영하지 않아야 맞음\n"
            "줄을 더블클릭하면 그 녹화를 뷰어에서 엽니다.")
        ).pack(anchor="w", pady=(6, 0))

        # 확인 필요 탭: 정의 조건과 정답이 안 맞는 트랙
        ck = ttk.Frame(self.nb)
        self.nb.add(ck, text="확인 필요")
        self.check_text = self._text(ck, (UI_FONT[0], 10))
        self.check_text.insert("1.0", self._rules_intro())

        # 결정 트리 탭
        tt = ttk.Frame(self.nb)
        self.nb.add(tt, text="결정 트리")
        self.tree_text = self._text(tt, ("Consolas", 10), wrap="none")

    def _table(self, master, cols, heads, widths, pady=(0, 0)):
        box = ttk.Frame(master)
        box.pack(fill="both", expand=True, pady=pady)
        tv = ttk.Treeview(box, columns=cols, show="headings")
        for c, h, w in zip(cols, heads, widths):
            tv.heading(c, text=h)
            tv.column(c, width=w, anchor="w" if c in ("progress", "camera", "file_cam", "state", "why",
                                                       "behaviors", "file_verdict") else "center")
        ys = ttk.Scrollbar(box, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        tv.pack(fill="both", expand=True)
        return tv

    @staticmethod
    def _rules_intro() -> str:
        return ("상황 정의 조건 (결정 트리가 골라도 이 조건을 어기면 다음으로 확률 높은 상황을 고름.\n"
                "학습할 때도 이 조건을 어기는 순간은 정답을 조건을 지키는 가장 비슷한 상황으로 바꿔서 학습함)\n\n"
                + C.rules_text() + "\n\n" + C.places_text() + "\n"
                "(위 값은 situation_core.py의 설정값입니다)\n")

    def _text(self, master, font, wrap="word"):
        tx = tk.Text(master, wrap=wrap, font=font, padx=8, pady=6)
        ys = ttk.Scrollbar(master, orient="vertical", command=tx.yview)
        xs = ttk.Scrollbar(master, orient="horizontal", command=tx.xview)
        tx.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        tx.pack(fill="both", expand=True)
        return tx

    # ------------------------------------------------------------------ 정답 표
    def pick_csv(self):
        start = Path(self.csv_var.get()).parent if self.csv_var.get() else RECORDINGS
        p = filedialog.askopenfilename(title="정답 표 CSV 선택", initialdir=start if start.is_dir() else RECORDINGS,
                                       filetypes=[("CSV", "*.csv"), ("모든 파일", "*.*")])
        if p:
            self.csv_var.set(p)
            self.load_labels()

    def load_labels(self) -> bool:
        self.label_tree.delete(*self.label_tree.get_children())
        path = Path(self.csv_var.get().strip())
        if not path.is_file():
            self.label_summary.set("정답 표 파일을 찾을 수 없습니다.")
            return False
        try:
            labels = T.read_labels(path)
        except T.TrainError as exc:
            self.label_summary.set(f"정답 표 오류: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            self.label_summary.set(f"정답 표를 읽지 못했습니다: {exc}")
            return False
        base = path.parent
        missing = 0
        counts: dict[str, list[int]] = {}
        for lb in labels:
            f = base / lb["file"].strip()
            ok = f.is_file()
            missing += not ok
            c = counts.setdefault(lb["situation"], [0, 0])
            c[lb["split"] == "test"] += 1
            tags = ("missing",) if not ok else (("test",) if lb["split"] == "test" else ())
            self.label_tree.insert("", "end", tags=tags, values=(
                lb["file"], lb["situation"], lb["split"], lb.get("decide_s", ""), lb.get("target_id", ""),
                "있음" if ok else "없음 (같은 폴더에 없음)"))
        self._file_spec = None
        for lb in labels:
            f = base / lb["file"].strip()
            if f.suffix.lower() == ".pcr" and f.is_file():
                self._file_spec = C.pcr_spec(f)
                break
        self._region_state()
        parts = [f"{s} {c[0]}/{c[1]}" for s, c in counts.items()]
        self.label_summary.set(
            f"{len(labels)}줄  ·  상황별 train/test: {',  '.join(parts)}"
            + (f"\n※ 파일이 없는 줄 {missing}개 (빨간색). 학습하려면 CSV와 같은 폴더에 .pcr이 있어야 합니다." if missing else ""))
        return missing == 0

    def _region_state(self):
        """'녹화 파일 설정 따르기'면 입력칸을 막고 첫 녹화 파일의 설정을 보여줌"""
        auto = self.region_auto.get()
        for sp in self.region_spins:
            sp.configure(state="disabled" if auto else "normal")
        spec = getattr(self, "_file_spec", None)
        if auto and spec:
            r = C.Region.from_spec(spec)
            for k, v in (("xmin", r.xmin), ("xmax", r.xmax), ("ymin", r.ymin), ("ymax", r.ymax),
                         ("fov", r.fov_deg), ("yaw", r.yaw_deg)):
                self.region_vars[k].set(round(v, 3))
            self.region_note.set("녹화 파일마다 헤더에 저장된 설정을 씁니다 (위 값은 첫 번째 녹화 파일 기준).")
        elif auto:
            self.region_note.set("녹화 파일마다 헤더에 저장된 설정을 씁니다. (CSV 파일은 영역 없이 전부 사용)")
        else:
            self.region_note.set("모든 녹화에 위 값을 씁니다. 레이더 기준 좌표, x는 좌우·y는 정면 거리(m), 시야각은 전체 각도.")

    # ------------------------------------------------------------------ 학습
    def _out_path(self) -> Path | None:
        name = self.name_var.get().strip()
        if name.lower().endswith(".pkl"):
            name = name[:-4]
        if not name or BAD_NAME.search(name):
            messagebox.showerror("모델 이름", '모델 이름이 비었거나 쓸 수 없는 글자(\\ / : * ? " < > |)가 있습니다.')
            return None
        if name == C.MODEL_PATH.stem:
            messagebox.showerror("모델 이름", f"'{name}'은 가짜 데이터 기본 모델 이름이라 쓸 수 없습니다.")
            return None
        return MODEL_DIR / f"{name}.pkl"

    def _cv_state(self):
        on = self.cv_on.get()
        self.cv_spin.configure(state="normal" if on else "disabled")
        try:
            k = int(self.cv_k.get())
        except (tk.TclError, ValueError):
            k = 5
        name = self.name_var.get().strip() or "모델"
        name = name[:-4] if name.endswith(".pkl") else name
        self.cv_hint.set(f"split 열은 무시하고 녹화를 {k}묶음으로 나눠 {k}번 학습. "
                         f"{name}_1 ~ {name}_{k}.pkl로 저장" if on else
                         "끄면 정답 표의 split(train/test)대로 한 번 학습")

    def start(self):
        if self.busy:
            return
        if not self.load_labels():
            messagebox.showerror("정답 표", "정답 표에 문제가 있습니다. '정답 표' 탭을 확인하세요.")
            self.nb.select(0)
            return
        out = self._out_path()
        if out is None:
            return
        cv_k = 0
        if self.cv_on.get():
            try:
                cv_k = int(self.cv_k.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("교차 검증", "폴드 수가 숫자가 아닙니다.")
                return
            outs = [out.with_name(f"{out.stem}_{i + 1}.pkl") for i in range(cv_k)]
            exist = [p.name for p in outs if p.exists()]
            if exist and not messagebox.askyesno("모델 덮어쓰기", f"{', '.join(exist)}이(가) 이미 있습니다. 덮어쓸까요?"):
                return
        elif out.exists() and not messagebox.askyesno("모델 덮어쓰기", f"{out.name}이 이미 있습니다. 덮어쓸까요?"):
            return
        try:
            if self.region_auto.get():
                region = "auto"
            else:
                v = {k: float(var.get()) for k, var in self.region_vars.items()}
                if v["xmin"] >= v["xmax"] or v["ymin"] >= v["ymax"]:
                    messagebox.showerror("측정 영역", "최소값이 최대값보다 작아야 합니다.")
                    return
                region = C.Region(v["xmin"], v["xmax"], v["ymin"], v["ymax"], v["fov"], v["yaw"])
            params = dict(region=region, depth=int(self.depth.get()), final_depth=int(self.final_depth.get()),
                          min_leaf=int(self.min_leaf.get()), with_sim=bool(self.with_sim.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("설정", "설정 값이 숫자가 아닙니다.")
            return
        labels = Path(self.csv_var.get().strip())
        self.busy = True
        self.result = None
        self.run_btn.configure(state="disabled")
        self.view_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        self.nb.select(1)
        self.status.set("학습 중...")

        def work():
            try:
                if cv_k:
                    res = T.cross_validate(labels, out, cv_k, log=lambda s: self.q.put(("log", s)),
                                           progress=lambda v: self.q.put(("progress", v)), **params)
                else:
                    res = T.train(labels, out, log=lambda s: self.q.put(("log", s)),
                                  progress=lambda v: self.q.put(("progress", v)), **params)
                self.q.put(("done", res))
            except T.TrainError as exc:
                self.q.put(("error", str(exc)))
            except Exception:  # noqa: BLE001
                self.q.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, self.poll)

    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                elif kind == "progress":
                    self.progress["value"] = val
                elif kind == "error":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self.status.set("실패")
                    self.log.insert("end", "\n[실패]\n" + val + "\n")
                    self.log.see("end")
                    messagebox.showerror("학습 실패", val[-1500:])
                    return
                elif kind == "done":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self.result = val
                    cv = getattr(val, "cv", False)
                    if cv:
                        self.status.set(f"완료 → {val.folds[0][0].stem} ~ {val.folds[-1][0].name}")
                        names = ["전체"] + [f"{i + 1}번" for i in range(len(val.folds))]
                    else:
                        self.status.set(f"완료 → {val.out_path.name}")
                        names = ["test", "train", "전체"]
                    self.split_combo.configure(values=names)
                    if self.split_var.get() not in names:
                        self.split_var.set(names[0])
                    self.view_btn.configure(state="normal")
                    self.tree_text.delete("1.0", "end")
                    self.tree_text.insert("1.0", val.report)
                    self.check_text.delete("1.0", "end")
                    warns = val.warnings or []
                    self.check_text.insert("1.0", (
                        (f"정답이 상황 정의와 안 맞는 트랙 {len(warns)}개 (조건을 어기는 순간은 '→' 뒤의 가장 비슷한 상황으로 학습).\n"
                         "정답(situation)이나 target_id가 맞는지, 녹화가 그 상황대로 찍혔는지 확인해보세요.\n\n"
                         + "\n".join(warns) if warns else "정답이 상황 정의와 안 맞는 트랙이 없습니다.")
                        + "\n\n" + "-" * 70 + "\n" + self._rules_intro()))
                    self.nb.tab(3, text=f"확인 필요 ({len(warns)})" if warns else "확인 필요")
                    self.fill_results()
                    self.nb.select(2)
                    return
        except queue.Empty:
            pass
        self.after(50, self.poll)

    def fill_results(self):
        self.res_tree.delete(*self.res_tree.get_children())
        res = self.result
        if res is None:
            return
        want = self.split_var.get()
        self._row_files = {}
        for r in res.rows:
            if want != "전체" and r["split"] != want:
                continue
            if self.only_bad.get() and r["ok"] == "O":
                continue
            vals = [r[c] for c in RESULT_COLS]
            if r["ok"] == "O":
                vals[RESULT_COLS.index("why")] = ""
            tag = {"O": "good", "△": "near"}.get(r["ok"], "bad")
            iid = self.res_tree.insert("", "end", values=vals, tags=(tag,))
            self._row_files[iid] = (r["file"], r.get("model"))
        txt = []
        if getattr(res, "cv", False):
            k = len(res.folds)
            accs = []
            for i in range(k):
                (ro, rn) = res.summary[f"{i + 1}번"]["recordings"]
                (to, tn) = res.summary[f"{i + 1}번"]["tracks"]
                accs.append(f"{i + 1}번 녹화 {ro}/{rn}·사람 {to}/{tn}")
            (ro, rn), (to, tn), (fo, fn), (so, sn) = (res.summary["전체"][key] for key in
                                                      ("recordings", "tracks", "files", "snapshots"))
            txt.append(f"[{k}겹 교차 검증, 모든 녹화가 한 번씩 test] 녹화 판정 {ro}/{rn} = {ro / max(rn, 1):.0%}  ·  "
                       f"사람별 최종 판정 {to}/{tn} = {to / max(tn, 1):.0%}  ·  "
                       f"파일 촬영 판단 {fo}/{fn}  ·  0.5초 판정 {so / max(sn, 1):.0%}")
            txt.append("폴드별: " + ",  ".join(accs) + "   (상황별 정답률·혼동 행렬은 '결정 트리' 탭)")
        for name, title in (("test", "test"), ("train", "train(참고)")):
            s = res.summary.get(name)
            if s:
                (to, tn), (fo, fn), (so, sn) = s["tracks"], s["files"], s["snapshots"]
                ro, rn = s.get("recordings", (0, 0))
                txt.append(f"[{title}] 녹화 판정 {ro}/{rn}  ·  사람별 최종 판정 {to}/{tn}  ·  "
                           f"파일 촬영 판단 {fo}/{fn}  ·  0.5초 판정 {so / max(sn, 1):.0%}")
        if "test" not in res.summary and not getattr(res, "cv", False):
            txt.append("※ split이 test인 녹화가 없어서 진짜 성능은 확인할 수 없습니다.")
        if res.warnings:
            txt.append(f"※ 정답이 상황 정의와 안 맞는 트랙 {len(res.warnings)}개 → '확인 필요' 탭")
        self.result_summary.set("\n".join(txt))

    # ------------------------------------------------------------------ 기타
    def open_models(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        open_path(MODEL_DIR)

    def open_row(self, event):
        iid = self.res_tree.identify_row(event.y)
        item = getattr(self, "_row_files", {}).get(iid)
        if item and self.result:
            f, model = item
            base = self.result.base or Path(self.csv_var.get()).parent
            self.open_viewer([str((Path(base) / f).resolve())], model)   # 교차 검증이면 그 녹화를 판정한 폴드의 모델

    def open_viewer(self, files=None, model=None):
        if not self.result:
            return
        if files is None and getattr(self.result, "cv", False):
            # 교차 검증: 선택한 줄의 폴드(없으면 1번)의 test 녹화 전부 + 그 폴드 모델
            sel = self.res_tree.selection()
            item = getattr(self, "_row_files", {}).get(sel[0]) if sel else None
            k = next((i for i, (p, _) in enumerate(self.result.folds) if item and str(p) == item[1]), 0)
            model, files = str(self.result.folds[k][0]), self.result.folds[k][1]
        viewer = HERE / "situation_viewer.pyw"
        exe = Path(sys.executable)
        pyw = exe.with_name("pythonw.exe")
        cmd = [str(pyw if pyw.exists() else exe), str(viewer)]
        cmd += files or self.result.test_files   # test 녹화를 전부 열어서 ◀ ▶로 넘겨보기
        cmd += ["--model", str(model or self.result.out_path)]
        subprocess.Popen(cmd, cwd=str(HERE))


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title("상황 결정 트리 학습")
    root.geometry("1250x820")
    try:
        ttk.Style(root).theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass

    def report(exc_type, exc, tb):
        messagebox.showerror("오류", "".join(traceback.format_exception(exc_type, exc, tb))[-2000:])
    root.report_callback_exception = report
    TrainApp(root).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
