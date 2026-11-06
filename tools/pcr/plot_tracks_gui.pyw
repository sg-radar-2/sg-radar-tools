"""트랙 그래프 보기 (GUI)

plot_tracks.py를 GUI로 옮긴 버전입니다. to_numpy_gui.pyw가 만든 npz(또는 .pcr 파일 자체)를 열면
시간에 따른 사람(ID)별 위치를 그려줍니다. 원래 plot_tracks.py가 그리던 "시간-x(좌우)" 그래프에
더해 y(앞뒤), r(거리), status(자세)와 위에서 본 궤적(x-y)도 볼 수 있습니다.

실행:
    plot_tracks_gui.pyw를 더블클릭 (콘솔 창 없이 열림)
    또는 npz / pcr 파일을 plot_tracks_gui.pyw 아이콘 위로 끌어다 놓기
    명령줄: python tools/pcr/plot_tracks_gui.pyw recordings/a_numpy.npz

예전 to_numpy.py로 만든 npz(points/targets만 있는 파일)도 열립니다. 그 경우 세션은 하나로 봅니다.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _sibling(name: str):
    """같은 폴더의 to_numpy_gui / plot_tracks_gui 모듈 불러오기 (.pyw도 지원)."""
    import importlib
    import importlib.machinery
    import importlib.util
    try:
        return importlib.import_module(name)
    except ImportError:
        path = HERE / f"{name}.pyw"
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        loader.exec_module(mod)
        return mod


def _fatal(msg: str) -> None:
    """pythonw로 실행하면 콘솔이 없으니 오류를 창으로 보여준다."""
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _r = _tk.Tk()
        _r.withdraw()
        _mb.showerror("실행 오류", msg)
        _r.destroy()
    except Exception:
        pass
    raise SystemExit(msg)


try:
    import numpy as np
except ImportError:
    _fatal("numpy가 설치되어 있지 않습니다.\n명령 프롬프트에서 'pip install numpy'를 실행하세요.")
try:
    import matplotlib
except ImportError:
    _fatal("matplotlib가 설치되어 있지 않습니다.\n명령 프롬프트에서 'pip install matplotlib'를 실행하세요.")

matplotlib.use("TkAgg")
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  # noqa: E402
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure  # noqa: E402

for _name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = HERE.parents[1]
RECORDINGS = REPO_ROOT / "recordings"

STATUS_NAMES = {0: "Standing", 1: "Sitting", 2: "Lying", 4: "Walking"}
AXES = {  # 표시 이름 -> (설명, 단위)
    "x": ("x (좌우)", "X 위치 (m)"),
    "y": ("y (앞뒤)", "Y 위치 (m)"),
    "r": ("r (레이더까지 거리)", "거리 (m)"),
    "status": ("status (자세)", "status"),
    "doppler": ("doppler (도플러 평균)", "도플러 평균"),
}
ALL_SESSIONS = "전체 (이어서 보기)"
SESSION_GAP_S = 1.0


# =========================================================================== 데이터

def load_data(path: str | Path) -> dict:
    """npz 또는 pcr을 읽어 그래프에 필요한 배열을 돌려준다."""
    path = Path(path)
    if path.suffix.lower() == ".pcr":
        d = _sibling("to_numpy_gui").pcr_to_arrays(path, with_points=False)
    else:
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
    if "targets" not in d:
        raise ValueError("이 npz에는 targets 배열이 없습니다.")
    tg = np.asarray(d["targets"], dtype=np.float64).reshape(-1, 6)
    d["targets"] = tg
    if "target_session" not in d:
        d["target_session"] = np.zeros(len(tg), dtype=np.int32)
    if "session_ids" not in d:
        d["session_ids"] = np.unique(d["target_session"]).astype(np.int32)
        if len(d["session_ids"]) == 0:
            d["session_ids"] = np.array([0], dtype=np.int32)
    if "session_names" not in d:
        d["session_names"] = np.array([f"Session #{i}" for i in d["session_ids"]])
    d["path"] = path
    return d


def default_png(path: Path) -> Path:
    stem = path.stem[:-6] if path.stem.endswith("_numpy") else path.stem
    return path.with_name(f"{stem}_track.png")


# =========================================================================== GUI

class PlotFrame(ttk.Frame):
    def __init__(self, master, path: str | Path | None = None):
        super().__init__(master)
        self.data: dict | None = None
        self.all_ids: list[int] = []

        # ------------------------------------------------ 위쪽: 파일
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(fill="x")
        ttk.Button(top, text="열기...", command=self.pick_file).pack(side="left")
        self.path_var = tk.StringVar(value="npz 또는 pcr 파일을 여세요.")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(top, text="PNG 저장...", command=self.save_png).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        # ------------------------------------------------ 왼쪽: 설정
        side = ttk.Frame(body, padding=(8, 0, 4, 8), width=240)
        side.pack(side="left", fill="y")

        ttk.Label(side, text="세션").pack(anchor="w")
        self.session_var = tk.StringVar(value=ALL_SESSIONS)
        self.session_combo = ttk.Combobox(side, textvariable=self.session_var, state="readonly", width=26)
        self.session_combo.pack(fill="x", pady=(0, 8))
        self.session_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_ids())

        ttk.Label(side, text="세로축").pack(anchor="w")
        self.axis_var = tk.StringVar(value="x")
        for key, (label, _) in AXES.items():
            ttk.Radiobutton(side, text=label, value=key, variable=self.axis_var,
                            command=self.draw).pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        self.top_view_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="위에서 본 궤적(x-y) 같이 보기", variable=self.top_view_var,
                        command=self.draw).pack(anchor="w")
        self.line_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="점을 선으로 잇기", variable=self.line_var,
                        command=self.draw).pack(anchor="w")

        size_row = ttk.Frame(side)
        size_row.pack(fill="x", pady=(6, 0))
        ttk.Label(size_row, text="점 크기").pack(side="left")
        self.size_var = tk.DoubleVar(value=10)
        ttk.Scale(size_row, from_=2, to=40, variable=self.size_var,
                  command=lambda v: self._draw_later()).pack(side="left", fill="x", expand=True, padx=4)

        min_row = ttk.Frame(side)
        min_row.pack(fill="x", pady=(6, 0))
        ttk.Label(min_row, text="숨길 ID: 프레임 수 <").pack(side="left")
        self.min_frames_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(min_row, from_=0, to=10000, increment=10, width=6,
                           textvariable=self.min_frames_var, command=self.refresh_ids)
        spin.pack(side="left", padx=4)
        spin.bind("<Return>", lambda e: self.refresh_ids())
        ttk.Label(side, text="잠깐 잡혔다 사라진 ID를 숨길 때 사용", foreground="#777").pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        id_head = ttk.Frame(side)
        id_head.pack(fill="x")
        ttk.Label(id_head, text="ID (프레임 수)").pack(side="left")
        ttk.Button(id_head, text="해제", width=5, command=lambda: self._select_ids(False)).pack(side="right")
        ttk.Button(id_head, text="전체", width=5, command=lambda: self._select_ids(True)).pack(side="right", padx=2)
        self.id_list = tk.Listbox(side, selectmode="multiple", height=12, exportselection=False,
                                  activestyle="none")
        self.id_list.pack(fill="both", expand=True, pady=(4, 0))
        self.id_list.bind("<<ListboxSelect>>", lambda e: self.draw())
        self.list_ids: list[int] = []

        self.info_var = tk.StringVar()
        ttk.Label(side, textvariable=self.info_var, foreground="#555", wraplength=230,
                  justify="left").pack(anchor="w", pady=(8, 0))

        # ------------------------------------------------ 오른쪽: 그래프
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._after_id = None
        if path:
            self.after(100, lambda: self.open(path))

    # ------------------------------------------------------------------ 파일
    def pick_file(self) -> None:
        start = RECORDINGS if RECORDINGS.is_dir() else REPO_ROOT
        p = filedialog.askopenfilename(
            title="npz 또는 pcr 열기", initialdir=start,
            filetypes=[("NumPy / PCR", "*.npz *.pcr"), ("NumPy", "*.npz"), ("PCR", "*.pcr"),
                       ("모든 파일", "*.*")])
        if p:
            self.open(p)

    def open(self, path: str | Path) -> None:
        path = Path(path)
        self.path_var.set(f"불러오는 중... {path}")
        result: dict = {}

        def work():
            try:
                result["data"] = load_data(path)
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def wait():
            if th.is_alive():
                self.after(50, wait)
            elif "error" in result:
                self.path_var.set(str(path))
                messagebox.showerror("열기 실패", result["error"])
            else:
                self._loaded(result["data"])

        wait()

    def _loaded(self, d: dict) -> None:
        self.data = d
        self.path_var.set(str(d["path"]))
        names = [str(n) for n in d["session_names"]]
        self.session_combo["values"] = [ALL_SESSIONS] + names
        self.session_var.set(ALL_SESSIONS if len(names) > 1 else names[0])
        self.all_ids = sorted(set(d["targets"][:, 2].astype(int).tolist()))
        self.refresh_ids(select_all=True)

    # ------------------------------------------------------------------ 선택
    def _session_mask(self) -> np.ndarray:
        d = self.data
        name = self.session_var.get()
        if name == ALL_SESSIONS:
            return np.ones(len(d["targets"]), dtype=bool)
        idx = [str(n) for n in d["session_names"]].index(name)
        return d["target_session"] == d["session_ids"][idx]

    def refresh_ids(self, select_all: bool = False) -> None:
        if self.data is None:
            return
        prev = {self.list_ids[i] for i in self.id_list.curselection()}
        m = self._session_mask()
        ids, counts = np.unique(self.data["targets"][m, 2].astype(int), return_counts=True)
        try:
            min_frames = int(self.min_frames_var.get())
        except (tk.TclError, ValueError):
            min_frames = 0
        keep = counts >= min_frames
        self.list_ids = ids[keep].tolist()
        self.id_list.delete(0, "end")
        for i, c in zip(ids[keep], counts[keep]):
            self.id_list.insert("end", f"ID {i}  ({c})")
        for k, i in enumerate(self.list_ids):
            if select_all or not prev or i in prev:
                self.id_list.selection_set(k)
        hidden = int((~keep).sum())
        self.info_var.set(f"타겟 {int(m.sum())}행, ID {len(ids)}개" + (f" (숨김 {hidden})" if hidden else ""))
        self.draw()

    def _select_ids(self, on: bool) -> None:
        if on:
            self.id_list.selection_set(0, "end")
        else:
            self.id_list.selection_clear(0, "end")
        self.draw()

    # ------------------------------------------------------------------ 그리기
    def _draw_later(self) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(120, self.draw)

    def _times(self, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[float, str]]]:
        """세션을 이어서 볼 때는 세션마다 시간을 뒤로 밀어서 겹치지 않게 한다."""
        d = self.data
        t = d["targets"][:, 0].copy()
        bounds: list[tuple[float, str]] = []
        if self.session_var.get() == ALL_SESSIONS and len(d["session_ids"]) > 1:
            offset = 0.0
            for sid, name in zip(d["session_ids"], d["session_names"]):
                sm = d["target_session"] == sid
                bounds.append((offset, str(name)))
                t[sm] += offset
                end = d["targets"][sm, 0].max() if sm.any() else 0.0
                if "session_length_s" in d:
                    end = max(end, float(d["session_length_s"][list(d["session_ids"]).index(sid)]))
                offset += end + SESSION_GAP_S
        return t[mask], bounds

    def draw(self) -> None:
        self._after_id = None
        self.fig.clear()
        if self.data is None:
            self.canvas.draw_idle()
            return
        d = self.data
        tg = d["targets"]
        m = self._session_mask()
        t_all, bounds = self._times(m)
        sub = tg[m]
        sel = [self.list_ids[i] for i in self.id_list.curselection()]
        axis = self.axis_var.get()
        size = float(self.size_var.get())
        cmap = matplotlib.colormaps["tab10"]

        if self.top_view_var.get():
            gs = self.fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
            ax = self.fig.add_subplot(gs[0, 0])
            ax_top = self.fig.add_subplot(gs[0, 1])
        else:
            ax = self.fig.add_subplot(1, 1, 1)
            ax_top = None

        tid = sub[:, 2].astype(int)
        if axis == "doppler" and "target_doppler" not in d:
            ax.set_title("이 파일에는 도플러 값이 없습니다. to_numpy_gui로 다시 변환하세요.")
            self.canvas.draw_idle()
            return
        values = {"x": sub[:, 3], "y": sub[:, 4], "r": np.hypot(sub[:, 3], sub[:, 4]),
                  "status": sub[:, 5],
                  "doppler": d["target_doppler"][m, 0] if "target_doppler" in d else None}[axis]
        for i in sel:
            mk = tid == i
            if not mk.any():
                continue
            color = cmap(self.all_ids.index(i) % 10)
            order = np.argsort(t_all[mk])
            tt, vv = t_all[mk][order], values[mk][order]
            ax.scatter(tt, vv, s=size, color=color, label=f"ID {i}")
            if self.line_var.get():
                ax.plot(tt, vv, color=color, lw=0.8, alpha=0.6)
            if ax_top is not None:
                xs, ys = sub[mk, 3][order], sub[mk, 4][order]
                ax_top.scatter(xs, ys, s=size * 0.6, color=color)
                if self.line_var.get():
                    ax_top.plot(xs, ys, color=color, lw=0.8, alpha=0.6)

        for x0, name in bounds:
            ax.axvline(x0, color="#999", lw=0.8, ls="--")
            ax.text(x0, 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                    va="bottom", ha="left", fontsize=8, color="#555")

        label, ylabel = AXES[axis]
        ax.set_xlabel("시간 (초)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"시간에 따른 {label}", pad=14 if bounds else 6)
        if axis == "doppler":
            ax.axhline(0, color="#666", lw=0.8)
        if axis == "status":
            ax.set_yticks(list(STATUS_NAMES), [f"{k} {v}" for k, v in STATUS_NAMES.items()])
        ax.grid(True, alpha=0.4)
        if sel:
            ax.legend(loc="best", fontsize=8, markerscale=1.5)

        if ax_top is not None:
            ax_top.scatter([0], [0], marker="^", s=80, color="black", zorder=5, label="레이더")
            spec = dict(zip(d.get("sensor_spec_fields", []), d.get("sensor_spec", [])))
            if "range_min_x" in spec:
                ax_top.set_xlim(spec["range_min_x"], spec["range_max_x"])
                ax_top.set_ylim(min(0.0, spec["range_min_y"]) - 0.2, spec["range_max_y"])
            ax_top.set_aspect("equal", adjustable="box")
            ax_top.set_xlabel("x (m) 좌우")
            ax_top.set_ylabel("y (m) 앞뒤")
            ax_top.set_title("위에서 본 궤적")
            ax_top.grid(True, alpha=0.4)
            ax_top.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ 저장
    def save_png(self) -> None:
        if self.data is None:
            messagebox.showinfo("PNG 저장", "먼저 파일을 여세요.")
            return
        suggest = default_png(self.data["path"])
        p = filedialog.asksaveasfilename(title="PNG 저장", initialdir=suggest.parent,
                                         initialfile=suggest.name, defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.fig.savefig(p, dpi=150)
            self.path_var.set(f"{self.data['path']}   →  저장: {p}")


def main() -> None:
    tn = _sibling("to_numpy_gui")
    tn.set_dpi_aware()
    root = tk.Tk()
    tn.setup_tk(root)
    root.title("트랙 그래프")
    root.geometry("1250x760")
    PlotFrame(root, sys.argv[1] if len(sys.argv) > 1 else None).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
