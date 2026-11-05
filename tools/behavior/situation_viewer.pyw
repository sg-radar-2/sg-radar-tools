"""문 앞 상황 판정 뷰어 (결정 트리)

트랙 CSV(실제 녹화 또는 가짜 시나리오)를 재생하면서
  - 위에서 본 궤적과 문 앞 구역
  - 6가지 관찰값 (거리, 움직임, 머문 시간, 나타난/사라진 위치, 자세, 인원)
  - 결정 트리의 상황 판정과 그 근거 (트리가 지나온 질문들)
  - 이상 상황일 때 카메라 촬영 이벤트
를 보여줍니다. '시나리오 결과' 탭에서는 정답과 판정을 한눈에 비교하고, '결정 트리' 탭에서는 트리 전체와
검증 정확도를 볼 수 있습니다.

실행: situation_viewer.pyw 더블클릭 (처음 한 번은 결정 트리 학습에 10초 정도 걸림)
      CSV를 아이콘 위로 끌어다 놓거나: python tools/behavior/situation_viewer.pyw recordings/a_tracks_raw.csv
필요: numpy, matplotlib, scikit-learn  (pip install numpy matplotlib scikit-learn)
"""

from __future__ import annotations

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
    S.PASS: "#6baed6", S.RETURN: "#41ab5d", S.GO_OUT: "#a1d99b", S.VISIT: "#fdae6b",
    S.DWELL: "#fb6a4a", S.LOITER: "#de2d26", S.REPEAT: "#a50f15", S.DOORLOCK: "#67000d",
    S.UNDECIDED: "#d0d0d0",
}
TRACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
TRAIL_S = 5.0
SHOW_AFTER_END_S = 3.0
TICK_MS = 60


class Viewer(ttk.Frame):
    def __init__(self, master, initial: str | None = None):
        super().__init__(master)
        self.models: C.Models | None = None
        self.tracks: list[C.Track] = []
        self.session_tracks: list[C.Track] = []
        self.t_lo = self.t_hi = 0.0
        self.cur_t = 0.0
        self.playing = False
        self.demo_seed = 7
        self._slider_lock = False
        self._panel_cache = ""
        self._build()
        self.after(100, lambda: self._load_models(initial))

    # ------------------------------------------------------------------ 화면 구성
    def _build(self):
        bar = ttk.Frame(self, padding=(8, 8, 8, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="CSV 열기...", command=self.pick_csv).pack(side="left")
        ttk.Button(bar, text="가짜 시나리오 새로 만들기", command=self.load_demo).pack(side="left", padx=4)
        ttk.Button(bar, text="결정 트리 다시 학습", command=lambda: self._load_models(None, force=True)).pack(side="left")
        self.path_var = tk.StringVar(value="결정 트리를 준비하는 중...")
        ttk.Label(bar, textvariable=self.path_var, foreground="#555").pack(side="left", padx=10)

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
        cols = ("session", "track", "truth", "pred", "prob", "ok", "camera", "cam_ok")
        heads = ("세션", "트랙", "정답 (최종)", "판정 (최종)", "확률", "판정", "카메라 촬영", "촬영")
        self.result = ttk.Treeview(res, columns=cols, show="headings", height=18)
        for c, h, w in zip(cols, heads, (220, 70, 130, 130, 60, 60, 260, 70)):
            self.result.heading(c, text=h)
            self.result.column(c, width=w, anchor="w" if c in ("session", "camera") else "center")
        self.result.pack(fill="both", expand=True)
        self.result.bind("<Double-1>", self._open_from_result)
        ttk.Label(res, foreground="#666", justify="left", text=(
            "판정: 사라진 시점에 상황 트리가 내린 최종 판정이 정답과 같은지\n"
            "촬영: 이상 상황이면 촬영했는지, 정상 상황이면 촬영하지 않았는지 (정답이 있는 데이터만)\n"
            "행을 더블클릭하면 그 세션을 재생합니다.")).pack(anchor="w", pady=(6, 0))

        # --- 결정 트리 탭
        tree_tab = ttk.Frame(self.nb)
        self.nb.add(tree_tab, text="결정 트리")
        self.tree_text = tk.Text(tree_tab, wrap="none", font=("Consolas", 10), padx=8, pady=6)
        ys = ttk.Scrollbar(tree_tab, orient="vertical", command=self.tree_text.yview)
        self.tree_text.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.tree_text.pack(fill="both", expand=True)

        self.winfo_toplevel().bind("<space>", lambda e: self.toggle_play())

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
            self.models = box["m"]
            self.tree_text.delete("1.0", "end")
            self.tree_text.insert("1.0", self.models.report)
            if initial:
                self.open_csv(initial)
            elif self.tracks:
                self._analyze_and_show(self.path_var.get(), self._rows)
            else:
                self.load_demo(first=True)

        wait()

    def pick_csv(self):
        start = C.HERE.parents[1] / "recordings"
        p = filedialog.askopenfilename(title="트랙 CSV 열기", initialdir=start if start.is_dir() else C.HERE,
                                       filetypes=[("트랙 CSV", "*_tracks_raw.csv *.csv"), ("모든 파일", "*.*")])
        if p:
            self.open_csv(p)

    def open_csv(self, path):
        try:
            rows = C.read_csv_rows(path)
        except Exception as exc:
            messagebox.showerror("열기 실패", str(exc))
            return
        if rows and "doppler_std" not in rows[0]:
            messagebox.showinfo("참고", "이 CSV에는 doppler_std(손 움직임) 열이 없어서 기본값으로 계산합니다.\n"
                                      "tools/pcr/to_numpy_gui.pyw로 다시 변환하면 정확해집니다.")
        self._analyze_and_show(str(path), rows)

    def load_demo(self, first=False):
        if not first:
            self.demo_seed += 1
        rows = [{k: str(v) for k, v in r.items()} for r in S.demo_sessions(self.demo_seed)]
        self._analyze_and_show(f"가짜 시나리오 (seed {self.demo_seed}) - 정답이 있는 데이터", rows)

    def _analyze_and_show(self, label, rows):
        if self.models is None:
            return
        self._rows = rows
        self.tracks = C.analyze(C.tracks_from_rows(rows), self.models)
        if not self.tracks:
            messagebox.showinfo("데이터 없음", "포인트 30개 이상인 트랙이 없습니다.")
            return
        self.path_var.set(label)
        sessions = list(dict.fromkeys(tr.session for tr in self.tracks))
        self.session_combo["values"] = sessions
        self.session_var.set(sessions[0])
        self._fill_results()
        self.select_session()

    # ------------------------------------------------------------------ 세션 선택
    def select_session(self):
        name = self.session_var.get()
        self.session_tracks = [tr for tr in self.tracks if tr.session == name]
        self.t_lo = min(tr.t0 for tr in self.session_tracks)
        self.t_hi = max(tr.t1 for tr in self.session_tracks) + SHOW_AFTER_END_S
        self.cur_t = self.t_lo
        self.time_scale.configure(from_=self.t_lo, to=self.t_hi)
        self.cam_list.delete(0, "end")
        self._cams = sorted(((c["t"], tr, c) for tr in self.session_tracks for c in tr.cameras),
                            key=lambda e: e[0])
        for t, tr, c in self._cams:
            self.cam_list.insert("end", f"[촬영] {t - self.t_lo:6.1f}초  T{tr.target_id}  "
                                        f"{c['situation']} ({c['prob']:.0%})")
        if not self._cams:
            self.cam_list.insert("end", "(이 세션에서는 촬영 없음)")
        self._setup_axes()
        self._render()

    def _setup_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-0.4, 5.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("위에서 본 모습 (레이더 = 문)", fontsize=11)
        ax.set_xlabel("x (m) 좌우")
        ax.set_ylabel("y (m) 앞")
        ax.grid(True, alpha=0.3)
        for s in (-1, 1):
            ax.plot([0, s * 5.6], [0, 5.6], color="#bbb", lw=0.8, ls="--")
        ax.add_patch(Circle((0, 0), C.NEAR_R, fill=False, ls=":", color="#888"))
        ax.add_patch(Circle((0, 0), C.DOOR_R, color="#fdd0a2", alpha=0.6))
        ax.text(0, C.DOOR_R + 0.05, "문 바로 앞", ha="center", fontsize=8, color="#a63603")
        ax.text(C.NEAR_R * 0.72, C.NEAR_R * 0.72, "근처", fontsize=8, color="#666")
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
        rows, yt = [], []
        y = 0
        for k, tr in enumerate(self.session_tracks[:5]):
            t = tr.t_abs - self.t_lo
            self._bands(tl, t, list(tr.sit_pred), y, 0.6)
            # 사라진 뒤: 최종 판정 (빗금)
            tl.add_patch(Rectangle((t[-1], y), SHOW_AFTER_END_S, 0.6, hatch="///", edgecolor="white",
                                   facecolor=COLORS.get(tr.final_pred, "#ccc"), lw=0))
            yt.append((y + 0.3, f"T{tr.target_id} 판정"))
            y += 0.8
            if tr.truth_situation:
                self._bands(tl, t, tr.truth_situation, y, 0.35)
                if tr.truth_final:
                    tl.add_patch(Rectangle((t[-1], y), SHOW_AFTER_END_S, 0.35, hatch="///", edgecolor="white",
                                           facecolor=COLORS.get(tr.truth_final, "#ccc"), lw=0))
                yt.append((y + 0.17, f"T{tr.target_id} 정답"))
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

        active = [tr for tr in self.session_tracks if tr.t0 <= T <= tr.t1]
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
            dot.set_alpha(0.35 if ended else 1.0)
            sit = tr.final_pred if ended else str(tr.sit_pred[i])
            prob = tr.final_prob if ended else float(tr.sit_prob[i])
            label.set_text(f"T{tr.target_id} {sit}")
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

    def _write_panel(self, panel):
        parts = []
        for tr, i, ended, sit, prob, feat, n_people in panel:
            obs = C.observations(tr, i, n_people, ended=ended)
            why = C.explain(self.models, feat)
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
            head = f"트랙 T{tr.target_id}" + ("  (사라짐 - 최종 판정)" if ended else "") + "\n"
            tx.insert("end", head, "h")
            for k, v in obs:
                tx.insert("end", f"  {k}\t", "k")
                tx.insert("end", f"{v}\n")
            tag = "abnormal" if sit in S.ABNORMAL else "wait" if sit == S.UNDECIDED else "normal"
            kind = "이상" if tag == "abnormal" else "판단 중" if tag == "wait" else "정상"
            tx.insert("end", "  상황 (결정 트리)\t", "k")
            tx.insert("end", f"{sit}  {prob:.0%}  [{kind}]\n", tag)
            tx.insert("end", "  근거 (트리가 지나온 질문)\n", "k")
            for w in why:
                tx.insert("end", f"   · {w}\n", "why")
            tx.insert("end", "\n")
        tx.configure(state="disabled")

    # ------------------------------------------------------------------ 결과 탭
    def _fill_results(self):
        self.result.delete(*self.result.get_children())
        n = ok = cam_n = cam_ok = 0
        for tr in self.tracks:
            truth = tr.truth_final or "-"
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
            self.result.insert("", "end", values=(tr.session, f"T{tr.target_id}", truth, tr.final_pred,
                                                  f"{tr.final_prob:.0%}", mark, cams, cmark))
        if n:
            self.summary_var.set(f"최종 판정 {ok}/{n} 맞음   ·   촬영 판단 {cam_ok}/{cam_n} 맞음 "
                                 f"(이상이면 촬영, 정상이면 촬영 안 함)\n※ 가짜 시나리오는 결정 트리 학습에 쓴 것과 "
                                 f"같은 생성기로 만든 데이터라 실제보다 잘 나옵니다.")
        else:
            self.summary_var.set("정답이 없는 데이터(실제 녹화)라서 판정 결과만 보여줍니다. "
                                 f"트랙 {len(self.tracks)}개, 촬영 {sum(len(t.cameras) for t in self.tracks)}회")

    def _open_from_result(self, _e):
        sel = self.result.selection()
        if not sel:
            return
        session = self.result.item(sel[0], "values")[0]
        self.session_var.set(session)
        self.select_session()
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
    Viewer(root, sys.argv[1] if len(sys.argv) > 1 else None).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
