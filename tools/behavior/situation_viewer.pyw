"""문 앞 상황 판정 뷰어 (결정 트리)

트랙 CSV(실제 녹화 또는 가짜 시나리오)를 재생하면서
  - 위에서 본 궤적과 문 앞 구역
  - 6가지 관찰값 (거리, 움직임, 머문 시간, 나타난/사라진 위치, 자세, 인원)
  - 결정 트리의 상황 판정과 그 근거 (트리가 지나온 질문들)
  - 이상 상황일 때 카메라 촬영 이벤트
를 보여줍니다. '시나리오 결과' 탭에서는 정답과 판정을 한눈에 비교하고, '결정 트리' 탭에서는 트리 전체와
검증 정확도를 볼 수 있습니다.

실행: situation_viewer.pyw 더블클릭 (처음 한 번은 결정 트리 학습에 10초 정도 걸림)
      '파일 열기...'에서 .pcr/CSV를 여러 개 고르거나 '폴더 열기...'로 폴더의 .pcr을 한꺼번에 열 수 있습니다.
      보고 있는 파일을 먼저 읽고, 나머지는 뒤에서 다음 파일부터 차례로 읽어 둡니다 (◀ ▶ 또는 PageUp/PageDown).
      명령줄: python tools/behavior/situation_viewer.pyw a.pcr b.pcr --model models/x.pkl
필요: numpy, matplotlib, scikit-learn  (pip install numpy matplotlib scikit-learn)
"""

from __future__ import annotations

import copy
import queue
import re
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    import numpy as np
    import matplotlib
    import sklearn  # noqa: F401
except ImportError as exc:
    _fatal(f"필요한 패키지가 없습니다: {exc.name}\n\n명령 프롬프트에서 아래를 실행하세요.\n"
           "pip install numpy matplotlib scikit-learn")

matplotlib.use("TkAgg")
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import scenario_sim as S  # noqa: E402
import situation_core as C  # noqa: E402

for _name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

UI_FONT = ("Malgun Gothic", 10) if sys.platform.startswith("win") else ("Noto Sans CJK KR", 10)
COLORS = {
    S.PASS: "#6baed6", S.RETURN: "#41ab5d", S.GO_OUT: "#a1d99b", S.VISIT: "#fdae6b", S.DELIVERY: "#fdd0a2",
    S.DWELL: "#fb6a4a", S.LOITER: "#de2d26", S.REPEAT: "#a50f15", S.DOORLOCK: "#67000d",
    S.UNDECIDED: "#d0d0d0",
}
TRACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
TRAIL_S = 5.0
DEFAULT_VIEW = {"xmin": -3.2, "xmax": 3.2, "ymin": 0.0, "ymax": 5.2, "fov": 90.0, "yaw": 0.0}   # 파일 설정이 없을 때
VIEW_MARGIN = 0.6           # 표시 영역 바깥으로 이만큼 더 보여줌 (영역 밖 트랙도 보이게)
SHOW_AFTER_END_S = 3.0
TICK_MS = 60


class Viewer(ttk.Frame):
    def __init__(self, master, initial=None, model_path: str | None = None):
        super().__init__(master)
        self.models: C.Models | None = None
        # 연 파일들: {"name", "path", "rows"(가짜 시나리오만), "raw"(판정 전 트랙), "tracks"(판정 결과),
        #             "gen"(어느 모델로 판정했는지), "busy", "error"}
        self.sources: list[dict] = []
        self.cur = -1
        self.gen = 0                      # 모델이 바뀔 때마다 1씩 증가 → 다시 판정
        self._cv = threading.Condition()
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self.session_tracks: list[C.Track] = []
        if isinstance(initial, str):
            initial = [initial]
        self.t_lo = self.t_hi = 0.0
        self.cur_t = 0.0
        self.playing = False
        self.demo_seed = 7
        self._slider_lock = False
        self._panel_cache = ""
        self._build()
        self._model_path = model_path     # 시작할 때 열 모델 (train_situation_gui에서 넘겨줌)
        self.after(100, lambda: self._load_models(initial))

    # ------------------------------------------------------------------ 화면 구성
    def _build(self):
        bar = ttk.Frame(self, padding=(8, 8, 8, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="파일 열기...", command=self.pick_files).pack(side="left")
        ttk.Button(bar, text="폴더 열기...", command=self.pick_folder).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="가짜 시나리오", command=self.load_demo).pack(side="left", padx=4)
        ttk.Button(bar, text="모델 열기...", command=self.pick_model).pack(side="left")
        ttk.Button(bar, text="가짜 데이터로 다시 학습",
                   command=lambda: self._load_models(None, force=True)).pack(side="left", padx=4)
        self.path_var = tk.StringVar(value="결정 트리를 준비하는 중...")
        ttk.Label(bar, textvariable=self.path_var, foreground="#555").pack(side="left", padx=10)
        self.model_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.model_var, foreground="#1f4e79").pack(side="right")

        fbar = ttk.Frame(self, padding=(8, 2, 8, 2))
        fbar.pack(fill="x")
        ttk.Label(fbar, text="파일").pack(side="left")
        ttk.Button(fbar, text="◀", width=3, command=lambda: self.step_file(-1)).pack(side="left", padx=(4, 0))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(fbar, textvariable=self.file_var, state="readonly", width=34)
        self.file_combo.pack(side="left", padx=2)
        self.file_combo.bind("<<ComboboxSelected>>", lambda e: self.select_source(self.file_combo.current()))
        ttk.Button(fbar, text="▶", width=3, command=lambda: self.step_file(1)).pack(side="left")
        self.load_var = tk.StringVar()
        ttk.Label(fbar, textvariable=self.load_var, foreground="#555").pack(side="left", padx=10)

        vbar = ttk.Frame(self, padding=(8, 2, 8, 2))
        vbar.pack(fill="x")
        ttk.Label(vbar, text="표시 영역").pack(side="left")
        self.view_vars = {k: tk.DoubleVar(value=v) for k, v in DEFAULT_VIEW.items()}
        for k, label, step in (("xmin", "x", 0.5), ("xmax", "~", 0.5), ("ymin", "  y", 0.5), ("ymax", "~", 0.5),
                               ("fov", "  시야각°", 5), ("yaw", "  방향°", 5)):
            ttk.Label(vbar, text=label).pack(side="left", padx=(2, 1))
            sp = ttk.Spinbox(vbar, from_=-360, to=360, increment=step, width=5, textvariable=self.view_vars[k])
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self.apply_view())
        ttk.Button(vbar, text="적용", width=5, command=self.apply_view).pack(side="left", padx=(6, 0))
        self.follow_file = tk.BooleanVar(value=True)
        ttk.Checkbutton(vbar, text="녹화 파일 설정 따르기", variable=self.follow_file,
                        command=self._redraw_view).pack(side="left", padx=8)
        self.region_var = tk.StringVar()
        ttk.Label(vbar, textvariable=self.region_var, foreground="#1f4e79").pack(side="right")

        ctl = ttk.Frame(self, padding=(8, 2, 8, 6))
        ctl.pack(fill="x")
        ttk.Label(ctl, text="세션").pack(side="left")
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(ctl, textvariable=self.session_var, state="readonly", width=28)
        self.session_combo.pack(side="left", padx=(4, 10))
        self.session_combo.bind("<<ComboboxSelected>>", lambda e: self.select_session())
        self.play_btn = ttk.Button(ctl, text="▶ 재생", width=8, command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Label(ctl, text="속도").pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1x")
        ttk.Combobox(ctl, textvariable=self.speed_var, values=["0.5x", "1x", "2x", "4x", "8x"],
                     state="readonly", width=5).pack(side="left")
        self.time_scale = ttk.Scale(ctl, from_=0, to=1, command=self._on_slider)
        self.time_scale.pack(side="left", fill="x", expand=True, padx=10)
        self.time_var = tk.StringVar(value="0.0 / 0.0 초")
        ttk.Label(ctl, textvariable=self.time_var, width=16).pack(side="left")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # --- 재생 탭
        play = ttk.Frame(self.nb)
        self.nb.add(play, text="재생")
        pw = ttk.PanedWindow(play, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw, width=420)
        pw.add(left, weight=3)
        pw.add(right, weight=2)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.28)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_tl = self.fig.add_subplot(gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Label(right, text="지금 판정", font=(UI_FONT[0], 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self.panel = tk.Text(right, wrap="word", font=UI_FONT, padx=8, pady=6, height=24,
                             relief="flat", background="#fafafa")
        self.panel.pack(fill="both", expand=True, padx=6)
        self.panel.configure(tabs=("165",))
        self.panel.tag_configure("h", font=(UI_FONT[0], 11, "bold"))
        self.panel.tag_configure("k", foreground="#666")
        self.panel.tag_configure("normal", foreground="#1b7f3b", font=(UI_FONT[0], 11, "bold"))
        self.panel.tag_configure("abnormal", foreground="#c0161b", font=(UI_FONT[0], 11, "bold"))
        self.panel.tag_configure("wait", foreground="#777", font=(UI_FONT[0], 11, "bold"))
        self.panel.tag_configure("why", foreground="#444", font=(UI_FONT[0], 9))
        ttk.Label(right, text="카메라 촬영 기록", font=(UI_FONT[0], 11, "bold")).pack(anchor="w", padx=6, pady=(8, 2))
        self.cam_list = tk.Listbox(right, height=7, font=UI_FONT, activestyle="none")
        self.cam_list.pack(fill="x", padx=6, pady=(0, 6))
        self.cam_list.bind("<<ListboxSelect>>", self._jump_to_camera)

        # --- 시나리오 결과 탭
        res = ttk.Frame(self.nb, padding=8)
        self.nb.add(res, text="시나리오 결과")
        self.summary_var = tk.StringVar()
        ttk.Label(res, textvariable=self.summary_var, font=(UI_FONT[0], 11, "bold"),
                  justify="left").pack(anchor="w", pady=(0, 6))
        cols = ("session", "verdict", "track", "truth", "pred", "prob", "ok", "behaviors", "camera", "cam_ok")
        heads = ("파일 · 세션", "녹화 판정", "사람(트랙)", "정답 (최종)", "판정 (최종)", "확률", "판정",
                 "나온 행동 (이상 먼저)", "카메라 촬영", "촬영")
        self.result = ttk.Treeview(res, columns=cols, show="headings", height=18)
        for c, h, w in zip(cols, heads, (200, 130, 90, 110, 110, 55, 50, 230, 200, 55)):
            self.result.heading(c, text=h)
            self.result.column(c, width=w, anchor="w" if c in ("session", "camera", "behaviors", "verdict")
                               else "center")
        self.result.pack(fill="both", expand=True)
        self.result.bind("<Double-1>", self._open_from_result)
        ttk.Label(res, foreground="#666", justify="left", text=(
            "최종 판정: 이상 행동 1순위(진행 중 60% 이상으로 1초 넘게 이어진 것), 없으면 정상 행동 중 확률이 "
            "가장 높은 것 (끝날 때 정의 조건을 지키는 것만)\n"
            "녹화 판정: 그 세션 사람들의 최종 판정 중 같은 우선순위로 하나 (세션 첫 줄에 표시)\n"
            "판정: 최종 판정이 정답과 같은지\n"
            "촬영: 이상 상황이면 촬영했는지, 정상 상황이면 촬영하지 않았는지 (정답이 있는 데이터만)\n"
            "행을 더블클릭하면 그 세션을 재생합니다. 여러 파일을 열면 읽은 파일부터 차례로 채워집니다.")
                  ).pack(anchor="w", pady=(6, 0))

        # --- 결정 트리 탭
        tree_tab = ttk.Frame(self.nb)
        self.nb.add(tree_tab, text="결정 트리")
        self.tree_text = tk.Text(tree_tab, wrap="none", font=("Consolas", 10), padx=8, pady=6)
        ys = ttk.Scrollbar(tree_tab, orient="vertical", command=self.tree_text.yview)
        self.tree_text.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.tree_text.pack(fill="both", expand=True)

        top = self.winfo_toplevel()
        top.bind("<space>", lambda e: self.toggle_play())
        top.bind("<Prior>", lambda e: self.step_file(-1))     # PageUp
        top.bind("<Next>", lambda e: self.step_file(1))       # PageDown

    # ------------------------------------------------------------------ 모델 / 데이터
    def _load_models(self, initial, force=False):
        self.path_var.set("결정 트리 학습 중... (처음 한 번 10초 정도)" if force or not C.MODEL_PATH.exists()
                          else "결정 트리 불러오는 중...")
        box = {}

        def work():
            try:
                box["m"] = C.load_or_train(force=force)
            except Exception:
                box["err"] = traceback.format_exc()

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def wait():
            if th.is_alive():
                self.after(100, wait)
                return
            if "err" in box:
                messagebox.showerror("학습 실패", box["err"][-1500:])
                return
            self._set_models(box["m"], "가짜 데이터 모델")
            if self._model_path:
                try:
                    self._set_models(C.load_models(self._model_path), Path(self._model_path).name)
                except Exception as exc:
                    messagebox.showerror("모델 열기 실패", str(exc))
                self._model_path = None
            if initial:
                self.open_files(initial)
            elif not self.sources:
                self.load_demo(first=True)

        wait()

    def _set_models(self, models, name):
        with self._cv:
            self.models = models
            self.gen += 1                 # 열려 있는 파일은 새 모델로 다시 판정
            self._cv.notify_all()
        self.model_var.set(f"모델: {name}")
        reg = C.model_region(models)
        self.region_var.set(f"판정 영역(모델): {reg.text()}" if reg else "판정 영역(모델): 제한 없음")
        self.tree_text.delete("1.0", "end")
        self.tree_text.insert("1.0", self.models.report)
        if self.sources:
            self.select_source(self.cur)

    def pick_model(self):
        p = filedialog.askopenfilename(title="결정 트리 모델 열기", initialdir=C.MODEL_PATH.parent,
                                       filetypes=[("모델", "*.pkl"), ("모든 파일", "*.*")])
        if not p:
            return
        try:
            self._set_models(C.load_models(p), Path(p).name)
        except Exception as exc:
            messagebox.showerror("모델 열기 실패", str(exc))

    # ------------------------------------------------------------------ 파일 여러 개
    @staticmethod
    def _natural_key(p):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]

    def pick_files(self):
        start = C.HERE.parents[1] / "recordings"
        ps = filedialog.askopenfilenames(title="PCR 또는 트랙 CSV 열기 (여러 개 선택 가능)",
                                         initialdir=start if start.is_dir() else C.HERE,
                                         filetypes=[("PCR / 트랙 CSV", "*.pcr *.csv"), ("모든 파일", "*.*")])
        if ps:
            self.open_files(sorted(ps, key=self._natural_key))

    def pick_folder(self):
        start = C.HERE.parents[1] / "recordings"
        d = filedialog.askdirectory(title="녹화 폴더 열기 (.pcr 전부)", initialdir=start if start.is_dir() else C.HERE)
        if not d:
            return
        ps = sorted(Path(d).glob("*.pcr"), key=self._natural_key)
        if not ps:
            ps = sorted(Path(d).glob("*_tracks_raw.csv"), key=self._natural_key)
        if not ps:
            messagebox.showinfo("폴더 열기", "이 폴더에 .pcr 파일이 없습니다.")
            return
        self.open_files([str(p) for p in ps])

    def open_files(self, paths):
        self._set_sources([{"name": Path(p).name, "path": str(p)} for p in paths])

    def load_demo(self, first=False):
        if not first:
            self.demo_seed += 1
        rows = [{k: str(v) for k, v in r.items()} for r in S.demo_sessions(self.demo_seed)]
        self._set_sources([{"name": f"가짜 시나리오 (seed {self.demo_seed})", "path": None, "rows": rows}])

    def _set_sources(self, items):
        self._stop()
        with self._cv:
            base = dict(rows=None, raw=None, tracks=None, gen=-1, busy=False, error=None)
            self.sources = [{**base, **it} for it in items]   # it의 값(rows 등)이 기본값을 덮어씀
            self.cur = 0
            self._cv.notify_all()
        self.file_combo["values"] = [s_["name"] for s_ in self.sources]
        self._ensure_worker()
        self._fill_results()
        self.select_source(0)

    def step_file(self, d):
        if self.sources:
            self.select_source(max(0, min(len(self.sources) - 1, self.cur + d)))

    def select_source(self, i):
        if not self.sources or i < 0:
            return
        self._stop()
        with self._cv:
            self.cur = i
            self._cv.notify_all()         # 이 파일을 먼저 읽도록
        src = self.sources[i]
        if "spec" not in src:          # 녹화 파일 헤더의 측정 설정 (헤더만 읽어서 빠름)
            src["spec"] = C.pcr_spec(src["path"]) if src.get("path", "") and str(src["path"]).lower().endswith(".pcr") else None
        if self.follow_file.get():
            self._fill_view_from_file()
        self.file_combo.current(i)
        self.path_var.set(f"{i + 1}/{len(self.sources)}  {src['name']}")
        if src["error"]:
            self._show_message(f"읽지 못했습니다: {src['error']}")
        elif src["tracks"] is None or src["gen"] != self.gen:
            self._show_message("불러오는 중...")
        else:
            self._show_source()

    def _fill_view_from_file(self):
        spec = self.sources[self.cur].get("spec") if self.sources else None
        if spec:
            r = C.Region.from_spec(spec)
            vals = {"xmin": r.xmin, "xmax": r.xmax, "ymin": r.ymin, "ymax": r.ymax, "fov": r.fov_deg, "yaw": r.yaw_deg}
        else:
            vals = DEFAULT_VIEW
        for k, v in vals.items():
            self.view_vars[k].set(round(v, 3))

    def _view(self) -> "C.Region":
        try:
            v = {k: float(var.get()) for k, var in self.view_vars.items()}
        except (tk.TclError, ValueError):
            v = dict(DEFAULT_VIEW)
        if v["xmin"] >= v["xmax"] or v["ymin"] >= v["ymax"]:
            v = dict(DEFAULT_VIEW)
        return C.Region(v["xmin"], v["xmax"], v["ymin"], v["ymax"], v["fov"], v["yaw"])

    def apply_view(self):
        """직접 고친 표시 영역 적용 (파일 설정 따르기는 꺼짐)"""
        self.follow_file.set(False)
        self._redraw_view()

    def _redraw_view(self):
        if self.follow_file.get() and self.sources:
            self._fill_view_from_file()
        if self.session_tracks:
            keep = self.cur_t
            self._setup_axes()
            self.cur_t = keep
            self._render()

    def _show_source(self):
        tracks = self.sources[self.cur]["tracks"]
        if not tracks:
            self._show_message("포인트 30개 이상인 트랙이 없습니다.")
            return
        sessions = list(dict.fromkeys(tr.session for tr in tracks))
        self.session_combo["values"] = sessions
        if self.session_var.get() not in sessions:
            self.session_var.set(sessions[0])
        self.select_session()

    def _show_message(self, msg):
        self.session_tracks = []
        self.session_combo["values"] = []
        self.session_var.set("")
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=13, color="#555",
                     transform=self.ax.transAxes)
        self.ax_tl.clear()
        self.ax_tl.set_axis_off()
        self.canvas.draw_idle()
        self.cam_list.delete(0, "end")
        self._panel_cache = ""
        self.panel.configure(state="normal")
        self.panel.delete("1.0", "end")
        self.panel.configure(state="disabled")
        self.time_var.set("")

    def _stop(self):
        if self.playing:
            self.toggle_play()

    # ------------------------------------------------------------------ 뒤에서 읽기
    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
        self.after(100, self._poll)

    def _next_job(self):
        """지금 보는 파일 → 그다음 파일들 순서로, 아직 이 모델로 판정하지 않은 파일"""
        n = len(self.sources)
        if not n or self.models is None:
            return None
        start = max(self.cur, 0)
        for k in range(n):
            i = (start + k) % n
            s_ = self.sources[i]
            if not s_["busy"] and s_["error"] is None and s_["gen"] != self.gen:
                return i
        return None

    def _worker_loop(self):
        while True:
            with self._cv:
                i = self._next_job()
                while i is None:
                    self._cv.wait()
                    i = self._next_job()
                src = self.sources[i]
                src["busy"] = True
                gen, models, owner = self.gen, self.models, self.sources
            try:
                raw = src["raw"]
                if raw is None:                       # 처음 한 번만 파일을 읽음
                    rows = src["rows"] if src["rows"] is not None else C.load_rows(src["path"])
                    raw = C.tracks_from_rows(rows)
                # 판정은 복사본에 (모델을 바꿔도 원본 트랙은 그대로 두고 다시 판정)
                tracks = C.analyze([copy.copy(tr) for tr in raw], models)
                self._q.put((owner, i, gen, raw, tracks, None))
            except Exception as exc:  # noqa: BLE001
                self._q.put((owner, i, gen, None, None, f"{type(exc).__name__}: {exc}"))

    def _poll(self):
        changed = False
        try:
            while True:
                owner, i, gen, raw, tracks, err = self._q.get_nowait()
                with self._cv:
                    if owner is not self.sources:     # 그사이 다른 파일들을 열었음
                        continue
                    src = self.sources[i]
                    src["busy"] = False
                    if err:
                        src["error"] = err
                    else:
                        src["raw"] = raw
                        if gen == self.gen:
                            src["tracks"], src["gen"] = tracks, gen
                    self._cv.notify_all()
                changed = True
                if i == self.cur:
                    self.select_source(i)
        except queue.Empty:
            pass
        done = sum(1 for s_ in self.sources if (s_["gen"] == self.gen) or s_["error"])
        total = len(self.sources)
        if total > 1 or done < total:
            self.load_var.set(f"판정 완료 {done}/{total}" + ("" if done == total else "  (뒤에서 읽는 중)"))
        else:
            self.load_var.set("")
        if changed:
            self._fill_results()
        self.after(150 if done < total else 500, self._poll)

    # ------------------------------------------------------------------ 세션 선택
    def select_session(self):
        name = self.session_var.get()
        src_tracks = self.sources[self.cur]["tracks"] if self.sources else None
        self.session_tracks = [tr for tr in (src_tracks or []) if tr.session == name]
        if not self.session_tracks:
            return
        self._panel_cache = ""
        self.t_lo = min(tr.t0 for tr in self.session_tracks)
        self.t_hi = max(tr.t1 for tr in self.session_tracks) + SHOW_AFTER_END_S
        self.cur_t = self.t_lo
        self.time_scale.configure(from_=self.t_lo, to=self.t_hi)
        self.cam_list.delete(0, "end")
        self._cams = sorted(((c["t"], tr, c) for tr in self.session_tracks for c in tr.cameras),
                            key=lambda e: e[0])
        for t, tr, c in self._cams:
            self.cam_list.insert("end", f"[촬영] {t - self.t_lo:6.1f}초  {C.track_name(tr)}  "
                                        f"{c['situation']} ({c['prob']:.0%})")
        if not self._cams:
            self.cam_list.insert("end", "(이 세션에서는 촬영 없음)")
        self._setup_axes()
        self._render()

    def _setup_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_axis_on()
        v = self._view()
        ax.set_xlim(v.xmin - VIEW_MARGIN, v.xmax + VIEW_MARGIN)
        ax.set_ylim(min(v.ymin, 0.0) - 0.4, v.ymax + VIEW_MARGIN)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("위에서 본 모습 (레이더 = 문)", fontsize=11)
        ax.set_xlabel("x (m) 좌우")
        ax.set_ylabel("y (m) 앞")
        ax.grid(True, alpha=0.3)
        # 측정 영역 (박스) + 시야각 (레이더에서 뻗는 점선)
        ax.add_patch(Rectangle((v.xmin, v.ymin), v.xmax - v.xmin, v.ymax - v.ymin, fill=False,
                               ls="--", lw=1.2, ec="#1f77b4", alpha=0.8))
        ax.text(v.xmin + 0.05, v.ymax - 0.05, "측정 영역", va="top", fontsize=8, color="#1f77b4")
        L = 3 * max(abs(v.xmin), abs(v.xmax), v.ymax) + 3
        for sgn in (-1, 1):
            a = np.radians(v.yaw_deg + sgn * v.fov_deg / 2)
            ax.plot([0, L * np.sin(a)], [0, L * np.cos(a)], color="#bbb", lw=0.8, ls="--")
        ax.add_patch(Circle((0, 0), C.NEAR_R, fill=False, ls=":", color="#888"))
        ax.add_patch(Circle((0, 0), C.DOOR_R, color="#fdd0a2", alpha=0.6))
        ax.text(0, C.DOOR_R + 0.05, "문 바로 앞", ha="center", fontsize=8, color="#a63603")
        ax.text(C.NEAR_R * 0.72, C.NEAR_R * 0.72, "근처", fontsize=8, color="#666", clip_on=True)
        ax.add_patch(Rectangle((-0.5, -0.35), 1.0, 0.2, color="#555"))
        ax.plot([0], [0], marker="^", ms=11, color="black")
        ax.text(0.15, -0.3, "문 / 레이더", fontsize=8)
        self.artists = {}
        for k, tr in enumerate(self.session_tracks):
            col = TRACK_COLORS[k % len(TRACK_COLORS)]
            trail, = ax.plot([], [], color=col, lw=1.5, alpha=0.7)
            dot = ax.scatter([], [], s=140, color=col, edgecolor="black", zorder=5)
            label = ax.text(0, 0, "", fontsize=9, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))
            self.artists[id(tr)] = (trail, dot, label)
        self.flash = ax.text(0.5, 0.97, "", transform=ax.transAxes, ha="center", va="top",
                             fontsize=14, fontweight="bold", color="white",
                             bbox=dict(boxstyle="round,pad=0.4", fc="#c0161b", ec="none"))
        self.flash.set_visible(False)

        # 타임라인
        tl = self.ax_tl
        tl.clear()
        tl.set_axis_on()
        rows, yt = [], []
        y = 0
        for k, tr in enumerate(self.session_tracks[:5]):
            t = tr.t_abs - self.t_lo
            self._bands(tl, t, list(tr.sit_pred), y, 0.6)
            # 사라진 뒤: 최종 판정 (빗금)
            tl.add_patch(Rectangle((t[-1], y), SHOW_AFTER_END_S, 0.6, hatch="///", edgecolor="white",
                                   facecolor=COLORS.get(tr.final_pred, "#ccc"), lw=0))
            yt.append((y + 0.3, f"{C.track_name(tr)} 판정"))
            y += 0.8
            if tr.truth_situation:
                self._bands(tl, t, tr.truth_situation, y, 0.35)
                if tr.truth_final:
                    tl.add_patch(Rectangle((t[-1], y), SHOW_AFTER_END_S, 0.35, hatch="///", edgecolor="white",
                                           facecolor=COLORS.get(tr.truth_final, "#ccc"), lw=0))
                yt.append((y + 0.17, f"{C.track_name(tr)} 정답"))
                y += 0.6
            for c in tr.cameras:
                tl.plot([c["t"] - self.t_lo], [y - 0.05], marker="v", color="#c0161b", ms=9)
            y += 0.3
        tl.set_ylim(-0.1, max(y, 1))
        tl.set_xlim(0, self.t_hi - self.t_lo)
        tl.set_yticks([p for p, _ in yt], [s for _, s in yt], fontsize=8)
        tl.set_xlabel("시간 (초)   ▼ = 카메라 촬영,  빗금 = 사라진 뒤 최종 판정", fontsize=9)
        tl.invert_yaxis()
        tl.grid(True, axis="x", alpha=0.3)
        used = {s for tr in self.session_tracks
                for s in list(tr.sit_pred) + (tr.truth_situation or []) + [tr.final_pred, tr.truth_final]}
        handles = [Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in COLORS if s in used]
        tl.legend(handles, [s for s in COLORS if s in used], fontsize=7, ncol=min(5, len(handles)),
                  loc="upper center", bbox_to_anchor=(0.5, -0.45), frameon=False)
        self.now_line = tl.axvline(0, color="black", lw=1.2)
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.17)

    @staticmethod
    def _bands(ax, t, labels, y, h):
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                end = t[i] if i < len(t) else t[-1]
                ax.add_patch(Rectangle((t[start], y), max(end - t[start], 0.05), h,
                                       color=COLORS.get(str(labels[start]), "#ccc"), lw=0))
                start = i

    # ------------------------------------------------------------------ 재생
    def toggle_play(self):
        if not self.session_tracks:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ 정지" if self.playing else "▶ 재생")
        if self.playing:
            if self.cur_t >= self.t_hi - 0.05:
                self.cur_t = self.t_lo
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        speed = float(self.speed_var.get().rstrip("x"))
        self.cur_t += TICK_MS / 1000 * speed
        if self.cur_t >= self.t_hi:
            self.cur_t = self.t_hi
            self.toggle_play()
        self._render()
        if self.playing:
            self.after(TICK_MS, self._tick)

    def _on_slider(self, v):
        if self._slider_lock:
            return
        self.cur_t = float(v)
        self._render()

    def _jump_to_camera(self, _e):
        sel = self.cam_list.curselection()
        if sel and sel[0] < len(self._cams):
            self.cur_t = self._cams[sel[0]][0]
            self._render()

    def _render(self):
        T = self.cur_t
        self._slider_lock = True
        self.time_scale.set(T)
        self._slider_lock = False
        self.time_var.set(f"{T - self.t_lo:6.1f} / {self.t_hi - self.t_lo:.1f} 초")
        self.now_line.set_xdata([T - self.t_lo])

        active = [tr for tr in self.session_tracks
                  if tr.t0 <= T <= tr.t1 and not getattr(tr, "out_of_range", False)]   # 영역 안 사람만
        n_people = len(active)
        panel = []
        for tr in self.session_tracks:
            trail, dot, label = self.artists[id(tr)]
            if T < tr.t0 or T > tr.t1 + SHOW_AFTER_END_S:
                trail.set_data([], [])
                dot.set_offsets(np.empty((0, 2)))
                label.set_text("")
                continue
            ended = T > tr.t1
            i = int(np.searchsorted(tr.t_abs, T, side="right") - 1)
            i = max(0, min(i, len(tr.t_abs) - 1))
            j = int(np.searchsorted(tr.t_abs, tr.t_abs[i] - TRAIL_S))
            trail.set_data(tr.xs[j:i + 1], tr.ys[j:i + 1])
            dot.set_offsets([[tr.xs[i], tr.ys[i]]])
            hidden = tr.hidden is not None and bool(tr.hidden[i])
            dot.set_alpha(0.35 if ended or hidden else 1.0)      # 흐림 = 사라짐 또는 레이더가 놓친 정지
            sit = tr.final_pred if ended else str(tr.sit_pred[i])
            prob = tr.final_prob if ended else float(tr.sit_prob[i])
            label.set_text(f"{C.track_name(tr)} {sit}")
            label.set_position((tr.xs[i] + 0.15, tr.ys[i] + 0.15))
            label.get_bbox_patch().set_facecolor(COLORS.get(sit, "white"))
            label.set_color("white" if sit in (S.LOITER, S.REPEAT, S.DOORLOCK) else "black")
            feat = tr.final_feat if ended else tr.sfeat[i]
            panel.append((tr, i, ended, sit, prob, feat, max(n_people, 1)))

        recent = [c for t, _, c in self._cams if 0 <= T - t <= 1.5]
        self.flash.set_visible(bool(recent))
        if recent:
            self.flash.set_text(f"● 카메라 촬영 - {recent[-1]['situation']}")
        self.canvas.draw_idle()
        self._write_panel(panel)

    def _out_text(self):
        if C.model_region(self.models) is not None:
            return "측정 영역 밖"
        return f"감시 범위 {getattr(self.models, 'watch_r', C.WATCH_R):.1f}m 밖"

    def _write_panel(self, panel):
        panel = sorted(panel, key=lambda p_: bool(getattr(p_[0], "out_of_range", False)))  # 영역 안 트랙 먼저
        parts = []
        for tr, i, ended, sit, prob, feat, n_people in panel:
            obs = C.observations(tr, i, n_people, ended=ended, region=C.model_region(self.models))
            if getattr(tr, "out_of_range", False):
                why = []
            elif ended:      # 사라진 뒤: 최종 판정 트리 + 정의 조건으로 바뀐 내용
                why = C.explain(self.models, feat, "final")
                if getattr(tr, "rule_note", ""):
                    why = ["[정의 조건] " + tr.rule_note] + why
            else:
                why = C.explain(self.models, feat)
                raw = self.models.situation.predict(feat[None])[0]
                if str(raw) != str(sit):
                    bad = C.rule_violations(str(raw), feat, False)
                    why = [f"[정의 조건] 트리는 {raw}였지만 조건({'; '.join(bad)})을 아직 못 채워서 {sit}"] + why
            parts.append((tr, ended, sit, prob, obs, why))
        key = repr([(p[0].target_id, p[1], p[2], round(p[3], 2), p[4], p[5]) for p in parts])
        if key == self._panel_cache:
            return
        self._panel_cache = key
        tx = self.panel
        tx.configure(state="normal")
        tx.delete("1.0", "end")
        if not parts:
            tx.insert("end", "지금 잡힌 사람이 없습니다.", "k")
        for tr, ended, sit, prob, obs, why in parts:
            head = (f"트랙 {C.track_name(tr)}" + ("  (사라짐 - 최종 판정)" if ended else "")
                    + (f"  ({self._out_text()} - 판정 안 함)" if getattr(tr, "out_of_range", False) else "")
                    + "\n")
            tx.insert("end", head, "h")
            for k, v in obs:
                tx.insert("end", f"  {k}\t", "k")
                tx.insert("end", f"{v}\n")
            if ended and not getattr(tr, "out_of_range", False):
                beh = getattr(tr, "behaviors", [])
                if beh:
                    tx.insert("end", "  나온 행동\t", "k")
                    tx.insert("end", ", ".join(
                        f"{b['situation']} {b['prob']:.0%}" + ("(이상)" if b["abnormal"] else "")
                        + ("(끝날 때)" if b["source"] == "끝날 때" else "") for b in beh) + "\n")
                if getattr(tr, "end_pred", sit) != sit:
                    tx.insert("end", "  끝날 때 판정\t", "k")
                    tx.insert("end", f"{tr.end_pred} {tr.end_prob:.0%}\n")
            tag = "abnormal" if sit in S.ABNORMAL else "wait" if sit == S.UNDECIDED else "normal"
            kind = "이상" if tag == "abnormal" else "판단 중" if tag == "wait" else "정상"
            tx.insert("end", "  최종 판정 (이상 우선)\t" if ended else "  상황 (결정 트리)\t", "k")
            tx.insert("end", f"{sit}  {prob:.0%}  [{kind}]\n", tag)
            tx.insert("end", "  근거 (트리가 지나온 질문)\n", "k")
            for w in why:
                tx.insert("end", f"   · {w}\n", "why")
            tx.insert("end", "\n")
        tx.configure(state="disabled")

    # ------------------------------------------------------------------ 결과 탭
    def _fill_results(self):
        self.result.delete(*self.result.get_children())
        self._result_map = {}
        n = ok = cam_n = cam_ok = 0
        all_tracks = []
        for si, src in enumerate(self.sources):
            if src["tracks"] and src["gen"] == self.gen:
                multi = len({tr.session for tr in src["tracks"]}) > 1
                for tr in src["tracks"]:
                    all_tracks.append((si, f"{src['name']} · {tr.session}" if multi else src["name"], tr))
        verdicts, shown = {}, set()
        for si, label, tr in all_tracks:
            verdicts.setdefault((si, tr.session), []).append(tr)
        verdicts = {k: C.recording_verdict(v) for k, v in verdicts.items()}
        for si, label, tr in all_tracks:
            truth = tr.truth_final or "-"
            vkey = (si, tr.session)
            if vkey in shown:
                vtxt = ""
            else:
                shown.add(vkey)
                v, vp, _ = verdicts[vkey]
                vtxt = f"{v} {vp:.0%}" + (" (이상)" if v in S.ABNORMAL else "")
            beh = ", ".join(f"{b['situation']} {b['prob']:.0%}" + ("(끝)" if b["source"] == "끝날 때" else "")
                            for b in getattr(tr, "behaviors", []))
            cams = ", ".join(f"{c['t'] - tr.t0:.1f}초 {c['situation']}" for c in tr.cameras) or "-"
            if tr.truth_final:
                n += 1
                good = tr.final_pred == tr.truth_final
                ok += good
                should = tr.truth_final in S.ABNORMAL
                cam_good = bool(tr.cameras) == should
                cam_n += 1
                cam_ok += cam_good
                mark, cmark = ("O" if good else "X"), ("O" if cam_good else "X")
            else:
                mark = cmark = "-"
            iid = self.result.insert("", "end", values=(label, vtxt, C.track_name(tr), truth, tr.final_pred,
                                                        f"{tr.final_prob:.0%}", mark, beh or "-", cams, cmark))
            self._result_map[iid] = (si, tr.session)
        if n:
            self.summary_var.set(f"최종 판정 {ok}/{n} 맞음   ·   촬영 판단 {cam_ok}/{cam_n} 맞음 "
                                 f"(이상이면 촬영, 정상이면 촬영 안 함)\n※ 가짜 시나리오는 결정 트리 학습에 쓴 것과 "
                                 f"같은 생성기로 만든 데이터라 실제보다 잘 나옵니다.")
        else:
            trs = [t for _, _, t in all_tracks]
            self.summary_var.set("정답이 없는 데이터(실제 녹화)라서 판정 결과만 보여줍니다. "
                                 f"트랙 {len(trs)}개, 촬영 {sum(len(t.cameras) for t in trs)}회")

    def _open_from_result(self, _e):
        sel = self.result.selection()
        if not sel:
            return
        si, session = self._result_map.get(sel[0], (None, None))
        if si is None:
            return
        self.session_var.set(session)
        self.select_source(si)
        self.nb.select(0)


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title("문 앞 상황 판정 (결정 트리)")
    root.geometry("1400x880")
    try:
        ttk.Style(root).theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass

    def report(exc_type, exc, tb):
        messagebox.showerror("오류", "".join(traceback.format_exception(exc_type, exc, tb))[-2000:])
    root.report_callback_exception = report
    args = sys.argv[1:]
    model = None
    if "--model" in args:                   # situation_viewer.pyw [파일 ...] --model 모델.pkl
        i = args.index("--model")
        model = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    Viewer(root, args or None, model_path=model).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
