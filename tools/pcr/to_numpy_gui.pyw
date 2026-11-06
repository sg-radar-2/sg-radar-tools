"""PCR -> NumPy(.npz) 변환기 (GUI)

to_numpy.py를 GUI로 옮긴 버전입니다. .pcr 파일을 골라 변환하면 같은 폴더에
`<파일이름>_numpy.npz`가 생기고, 바로 plot_tracks_gui.pyw 그래프 창으로 열어볼 수 있습니다.

실행:
    to_numpy_gui.pyw를 더블클릭 (콘솔 창 없이 열림)
    또는 .pcr 파일을 to_numpy_gui.pyw 아이콘 위로 끌어다 놓기 (여러 개 가능)
    명령줄: python tools/pcr/to_numpy_gui.pyw a.pcr b.pcr

npz 안의 배열 (to_numpy.py와 points / targets 열 순서가 같아서 기존 스크립트도 그대로 읽힘):
    points          (N, 8) float32  time_s, frame_id, x, y, z, doppler, power, target_id
    targets         (M, 6) float64  time_s, frame_id, target_id, x, y, status
    point_session   (N,)   int32    각 포인트의 세션 id
    target_session  (M,)   int32    각 타겟 행의 세션 id
    target_box      (M, 6) float32  minx, maxx, miny, maxy, minz, maxz (빈 박스는 NaN)
    target_n_points (M,)   int32    그 프레임에서 그 사람에게 잡힌 포인트 수
    target_doppler  (M, 3) float32  그 사람 포인트들의 도플러 평균, 절댓값 평균, 표준편차 (포인트 없으면 NaN)
    session_ids / session_names / session_tags / session_length_s
    sensor_spec (16,) + sensor_spec_fields, *_columns (열 이름), source (원본 파일 이름)

CSV (선택, check_tracks.py와 같은 형식에 session·status·도플러 열 추가):
    <파일이름>_tracks_raw.csv  session, time_s, frame_id, target_id, x, y, status, n_points,
                               doppler_mean, doppler_abs_mean, doppler_std, quality
    quality는 n_points가 기준(기본 30) 미만이면 low_points, 아니면 ok

time_s는 to_numpy.py와 같이 세션마다 0초부터 다시 셉니다. 세션을 나눠 보려면 *_session을 쓰세요.

필요: numpy, matplotlib(그래프 창), 그리고 Python 3.14+ 또는 `pip install zstandard`
"""

from __future__ import annotations

import csv
import io
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

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

from pcr_loader import (FILE_MAGIC, FILE_VERSION, HEADER_STRUCT,  # noqa: E402
                        PAYLOAD_MAGIC, PcrError, SensorSpec, parse_header)

REPO_ROOT = HERE.parents[1]
RECORDINGS = REPO_ROOT / "recordings"

POINT_COLUMNS = ["time_s", "frame_id", "x", "y", "z", "doppler", "power", "target_id"]
TARGET_COLUMNS = ["time_s", "frame_id", "target_id", "x", "y", "status"]
BOX_COLUMNS = ["minx", "maxx", "miny", "maxy", "minz", "maxz"]
SPEC_FIELDS = list(SensorSpec.__dataclass_fields__)

_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                         ("doppler", "<f4"), ("power", "<f4"), ("target_id", "<i4")])
_TARGET_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("status", "<u4"), ("target_id", "<u4"),
                          ("minx", "<f4"), ("maxx", "<f4"), ("miny", "<f4"),
                          ("maxy", "<f4"), ("minz", "<f4"), ("maxz", "<f4")])


# =========================================================================== 변환

class _Stream:
    """zstd 페이로드를 조금씩 풀면서 읽는 리더 (큰 파일도 메모리에 한 번에 올리지 않음)."""

    def __init__(self, path: Path, offset: int, total: int):
        f = open(path, "rb")
        f.seek(offset)
        try:
            from compression import zstd  # Python 3.14+
            self._raw = zstd.ZstdFile(f, mode="rb")
        except ImportError:
            try:
                import zstandard
            except ImportError as exc:
                f.close()
                raise PcrError("zstd 모듈이 없습니다. Python 3.14+를 쓰거나 "
                               "'pip install zstandard'를 실행하세요.") from exc
            self._raw = io.BufferedReader(
                zstandard.ZstdDecompressor().stream_reader(f, closefd=True), 1 << 20)
        self.total = total
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n == 0:
            return b""
        parts, left = [], n
        while left:
            chunk = self._raw.read(left)
            if not chunk:
                raise PcrError("페이로드가 예상보다 일찍 끝났습니다 (파일 손상 가능)")
            parts.append(chunk)
            left -= len(chunk)
        self.pos += n
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:
            pass


def pcr_to_arrays(path: str | Path, with_points: bool = True,
                  on_progress: Callable[[float], None] | None = None) -> dict:
    """.pcr 하나를 읽어 npz에 들어갈 배열 dict로 돌려준다."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_STRUCT.size)
    header = parse_header(raw)
    if header.magic != FILE_MAGIC:
        raise PcrError(f"PCR 파일이 아닙니다 (magic={header.magic!r})")
    if header.version != FILE_VERSION:
        raise PcrError(f"지원하지 않는 PCR 버전입니다: {header.version!r}")

    s = _Stream(path, HEADER_STRUCT.size, header.uncompressed_size)
    pts_blocks, psess_blocks = [], []
    tgt_blocks, box_blocks, tsess_blocks, npts_blocks, dop_blocks = [], [], [], [], []
    sess_ids, sess_names, sess_tags, sess_len = [], [], [], []
    try:
        if s.u64() != PAYLOAD_MAGIC:
            raise PcrError("페이로드 magic 값이 올바르지 않습니다")
        for _ in range(header.session_count):
            sid = s.u32()
            s.u64()                      # timestamp
            length_us = s.u64()
            s.u64()                      # bytes
            name, _desc, tag = s.string(), s.string(), s.string()
            nframes = s.u64()
            sess_ids.append(sid)
            sess_names.append(name)
            sess_tags.append(tag)
            sess_len.append(length_us / 1e6)

            elapsed_us = 0
            for i in range(nframes):
                s.u32()                  # packet size
                frame_id = s.u32()
                elapsed_us += s.u64()
                t = elapsed_us / 1_000_000
                npts = s.u64()
                pbuf = s.read(npts * _POINT_DTYPE.itemsize)
                ntg = s.u64()
                tbuf = s.read(ntg * _TARGET_DTYPE.itemsize)
                s.u64()                  # frame bytes

                p = np.frombuffer(pbuf, dtype=_POINT_DTYPE)
                if with_points and npts:
                    block = np.empty((npts, 8), dtype=np.float32)
                    block[:, 0] = t
                    block[:, 1] = frame_id
                    for j, c in enumerate(("x", "y", "z", "doppler", "power", "target_id"), 2):
                        block[:, j] = p[c]
                    pts_blocks.append(block)
                    psess_blocks.append(np.full(npts, sid, dtype=np.int32))
                if ntg:
                    g = np.frombuffer(tbuf, dtype=_TARGET_DTYPE)
                    block = np.empty((ntg, 6), dtype=np.float64)
                    block[:, 0] = t
                    block[:, 1] = frame_id
                    block[:, 2] = g["target_id"]
                    block[:, 3] = g["x"]
                    block[:, 4] = g["y"]
                    block[:, 5] = g["status"]
                    tgt_blocks.append(block)
                    box_blocks.append(np.stack([g[c] for c in BOX_COLUMNS], axis=1))
                    tsess_blocks.append(np.full(ntg, sid, dtype=np.int32))
                    tid = g["target_id"].astype(np.int64)
                    member = (p["target_id"].astype(np.int64)[:, None] == tid[None, :])  # (포인트, 타겟)
                    cnt = member.sum(axis=0)
                    npts_blocks.append(cnt.astype(np.int32))
                    dop = p["doppler"].astype(np.float64)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = (dop @ member) / cnt
                        abs_mean = (np.abs(dop) @ member) / cnt
                        var = (dop * dop) @ member / cnt - mean * mean
                    dop_blocks.append(np.stack([mean, abs_mean, np.sqrt(np.clip(var, 0, None))],
                                               axis=1).astype(np.float32))
                if on_progress and (i & 31) == 0 and s.total:
                    on_progress(min(1.0, s.pos / s.total))

            for _ in range(s.u64()):     # bookmarks
                s.string()
                s.u64()
    finally:
        s.close()
    if on_progress:
        on_progress(1.0)

    def cat(blocks, shape, dtype):
        return np.concatenate(blocks) if blocks else np.empty(shape, dtype=dtype)

    box = cat(box_blocks, (0, 6), np.float32)
    with np.errstate(invalid="ignore"):
        box[np.abs(box) >= 1e30] = np.nan   # 비어 있는 박스(FLT_MAX)는 NaN으로
    spec = header.sensor_spec
    return {
        "points": cat(pts_blocks, (0, 8), np.float32),
        "targets": cat(tgt_blocks, (0, 6), np.float64),
        "point_session": cat(psess_blocks, (0,), np.int32),
        "target_session": cat(tsess_blocks, (0,), np.int32),
        "target_box": box,
        "target_n_points": cat(npts_blocks, (0,), np.int32),
        "target_doppler": cat(dop_blocks, (0, 3), np.float32),
        "target_doppler_columns": np.array(["doppler_mean", "doppler_abs_mean", "doppler_std"], dtype=str),
        "session_ids": np.array(sess_ids, dtype=np.int32),
        "session_names": np.array(sess_names, dtype=str),
        "session_tags": np.array(sess_tags, dtype=str),
        "session_length_s": np.array(sess_len, dtype=np.float64),
        "sensor_spec": np.array([getattr(spec, k) for k in SPEC_FIELDS], dtype=np.float64),
        "sensor_spec_fields": np.array(SPEC_FIELDS, dtype=str),
        "point_columns": np.array(POINT_COLUMNS, dtype=str),
        "target_columns": np.array(TARGET_COLUMNS, dtype=str),
        "target_box_columns": np.array(BOX_COLUMNS, dtype=str),
        "source": np.array(path.name),
    }


def default_output(pcr: Path, out_dir: Path | None = None) -> Path:
    return (out_dir or pcr.parent) / f"{pcr.stem}_numpy.npz"


CSV_COLUMNS = ["session", "time_s", "frame_id", "target_id", "x", "y", "status", "n_points",
               "doppler_mean", "doppler_abs_mean", "doppler_std", "quality"]


def write_tracks_csv(arrays: dict, out: Path, min_points: int = 30) -> int:
    """check_tracks.py와 같은 트랙 CSV를 쓴다. 포인트 부족(low_points) 행 수를 돌려준다."""
    tg, sess, npt = arrays["targets"], arrays["target_session"], arrays["target_n_points"]
    dop = arrays["target_doppler"]

    def fmt(v):  # 포인트가 없어 도플러를 못 구한 칸은 비워둠
        return "" if np.isnan(v) else round(float(v), 4)
    names = {int(i): str(n) for i, n in zip(arrays["session_ids"], arrays["session_names"])}
    order = np.lexsort((tg[:, 2], tg[:, 0], sess))       # 세션 → 시간 → ID 순
    low = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for k in order:
            t = tg[k]
            n = int(npt[k])
            q = "low_points" if n < min_points else "ok"
            low += q == "low_points"
            w.writerow([names.get(int(sess[k]), int(sess[k])), round(float(t[0]), 4), int(t[1]), int(t[2]),
                        round(float(t[3]), 4), round(float(t[4]), 4), int(t[5]), n,
                        fmt(dop[k, 0]), fmt(dop[k, 1]), fmt(dop[k, 2]), q])
    return low


def csv_output(npz: Path) -> Path:
    stem = npz.stem[:-6] if npz.stem.endswith("_numpy") else npz.stem
    return npz.with_name(f"{stem}_tracks_raw.csv")


def convert(pcr: str | Path, out_dir: str | Path | None = None, with_points: bool = True,
            on_progress: Callable[[float], None] | None = None,
            with_csv: bool = True, min_points: int = 30) -> tuple[Path, dict, Path | None, int]:
    pcr = Path(pcr)
    arrays = pcr_to_arrays(pcr, with_points, on_progress)
    out = default_output(pcr, Path(out_dir) if out_dir else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    csv_path, low = None, 0
    if with_csv:
        csv_path = csv_output(out)
        try:
            low = write_tracks_csv(arrays, csv_path, min_points)
        except PermissionError as exc:
            raise PcrError(f"CSV를 쓸 수 없습니다. 엑셀 등에서 열려 있으면 닫고 다시 실행하세요.\n{csv_path}") from exc
    return out, arrays, csv_path, low


def summarize(arrays: dict) -> str:
    pts, tgs = arrays["points"], arrays["targets"]
    lines = [f"  포인트 배열 크기: {pts.shape}", f"  타겟 배열 크기:   {tgs.shape}"]
    for sid, name, length in zip(arrays["session_ids"], arrays["session_names"],
                                 arrays["session_length_s"]):
        m = arrays["target_session"] == sid
        ids, counts = np.unique(tgs[m, 2].astype(int), return_counts=True)
        id_text = ", ".join(f"ID {i}({c})" for i, c in zip(ids, counts)) or "없음"
        lines.append(f"  [{name}] {length:.1f}초, 추적 ID(프레임 수): {id_text}")
    if len(pts):
        lines.append("  첫 번째 포인트: " + np.array2string(pts[0], precision=3, suppress_small=True, max_line_width=200))
    return "\n".join(lines)


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# =========================================================================== GUI

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402


def set_dpi_aware() -> None:
    """Windows에서 글자가 흐릿하게 보이지 않도록 (Tk 창을 만들기 전에 호출)."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def setup_tk(root: tk.Misc) -> None:
    def report(exc_type, exc, tb):  # 버튼 처리 중 생긴 오류도 창으로
        import traceback
        messagebox.showerror("오류", "".join(traceback.format_exception(exc_type, exc, tb))[-2000:])
    root.report_callback_exception = report
    try:
        ttk.Style(root).theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass


class ConvertFrame(ttk.Frame):
    def __init__(self, master, files: list[str] | None = None):
        super().__init__(master, padding=10)
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.results: list[Path] = []

        # 입력 파일 목록
        box = ttk.LabelFrame(self, text="PCR 파일", padding=6)
        box.pack(fill="both", expand=False)
        self.listbox = tk.Listbox(box, height=6, selectmode="extended", activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(box)
        btns.pack(side="left", fill="y", padx=(6, 0))
        ttk.Button(btns, text="파일 추가...", command=self.add_files).pack(fill="x")
        ttk.Button(btns, text="선택 빼기", command=self.remove_selected).pack(fill="x", pady=4)
        ttk.Button(btns, text="모두 비우기", command=lambda: self.listbox.delete(0, "end")).pack(fill="x")

        # 출력 옵션
        opt = ttk.Frame(self)
        opt.pack(fill="x", pady=(8, 0))
        ttk.Label(opt, text="출력 폴더").grid(row=0, column=0, sticky="w")
        self.out_var = tk.StringVar()
        ttk.Entry(opt, textvariable=self.out_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(opt, text="선택...", command=self.pick_out).grid(row=0, column=2)
        opt.columnconfigure(1, weight=1)
        ttk.Label(opt, text="비워두면 .pcr과 같은 폴더에 <이름>_numpy.npz로 저장합니다.",
                  foreground="#666").grid(row=1, column=1, sticky="w", padx=6)
        self.points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="포인트 클라우드(points)도 저장  (끄면 타겟만 저장해서 훨씬 작고 빠름)",
                        variable=self.points_var).grid(row=2, column=1, sticky="w", padx=6, pady=(4, 0))
        csv_row = ttk.Frame(opt)
        csv_row.grid(row=3, column=1, sticky="w", padx=6, pady=(2, 0))
        self.csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(csv_row, text="트랙 CSV도 저장 (<이름>_tracks_raw.csv, 도플러 포함)",
                        variable=self.csv_var).pack(side="left")
        ttk.Label(csv_row, text="   포인트 부족 기준 <").pack(side="left")
        self.min_points_var = tk.IntVar(value=30)
        ttk.Spinbox(csv_row, from_=0, to=10000, increment=5, width=6,
                    textvariable=self.min_points_var).pack(side="left", padx=4)

        # 실행
        run = ttk.Frame(self)
        run.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(run, text="변환 실행", command=self.start)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(run, maximum=1.0)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status = tk.StringVar(value="파일을 추가하고 '변환 실행'을 누르세요.")
        ttk.Label(run, textvariable=self.status, width=34).pack(side="left")

        # 결과
        res = ttk.LabelFrame(self, text="결과", padding=6)
        res.pack(fill="both", expand=True)
        top = ttk.Frame(res)
        top.pack(fill="x")
        self.result_var = tk.StringVar()
        self.result_combo = ttk.Combobox(top, textvariable=self.result_var, state="readonly")
        self.result_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="그래프 보기", command=self.show_plot).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="CSV 열기", command=self.open_csv).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="폴더 열기", command=self.open_folder).pack(side="left", padx=(6, 0))
        self.log = tk.Text(res, height=14, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.log.configure(state="disabled")

        for f in files or []:
            self._add(f)

    # ------------------------------------------------------------------ 입력
    def _add(self, path: str) -> None:
        if path and path not in self.listbox.get(0, "end"):
            self.listbox.insert("end", path)

    def add_files(self) -> None:
        start = RECORDINGS if RECORDINGS.is_dir() else REPO_ROOT
        for p in filedialog.askopenfilenames(title="PCR 파일 선택", initialdir=start,
                                             filetypes=[("PCR 녹화", "*.pcr"), ("모든 파일", "*.*")]):
            self._add(p)

    def remove_selected(self) -> None:
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    def pick_out(self) -> None:
        p = filedialog.askdirectory(title="출력 폴더 선택")
        if p:
            self.out_var.set(p)

    def write(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ 실행
    def start(self) -> None:
        if self.busy:
            return
        files = list(self.listbox.get(0, "end"))
        if not files:
            messagebox.showinfo("변환", "먼저 .pcr 파일을 추가하세요.")
            return
        out_dir = self.out_var.get().strip() or None
        with_points = self.points_var.get()
        with_csv = self.csv_var.get()
        try:
            min_points = int(self.min_points_var.get())
        except (tk.TclError, ValueError):
            min_points = 30
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.progress["value"] = 0

        def work():
            for k, f in enumerate(files):
                self.q.put(("file", k, len(files), f))
                t0 = time.perf_counter()
                try:
                    out, arrays, csv_path, low = convert(
                        f, out_dir, with_points,
                        lambda v, k=k: self.q.put(("progress", (k + v) / len(files))),
                        with_csv, min_points)
                    summary = summarize(arrays)
                    if csv_path:
                        summary += (f"\n  CSV 저장: {csv_path}"
                                    f"\n  포인트 부족(<{min_points}) 행: {low} / {len(arrays['targets'])}")
                    self.q.put(("ok", out, summary, time.perf_counter() - t0))
                except Exception as exc:  # noqa: BLE001
                    self.q.put(("error", f, f"{type(exc).__name__}: {exc}"))
            self.q.put(("done",))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, self.poll)

    def poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self.progress["value"] = msg[1]
                elif kind == "file":
                    _, k, n, f = msg
                    self.status.set(f"변환 중 ({k + 1}/{n}) {Path(f).name}")
                    self.write(f"▶ {f}")
                elif kind == "ok":
                    _, out, summary, sec = msg
                    self.write(summary)
                    self.write(f"  저장 완료: {out}  ({sec:.1f}초)\n")
                    if out not in self.results:
                        self.results.append(out)
                    self.result_combo["values"] = [str(p) for p in self.results]
                    self.result_var.set(str(out))
                elif kind == "error":
                    self.write(f"  ✖ 실패: {msg[2]}\n")
                elif kind == "done":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self.progress["value"] = 1.0
                    self.status.set("완료")
                    return
        except queue.Empty:
            pass
        self.after(50, self.poll)

    # ------------------------------------------------------------------ 결과
    def show_plot(self) -> None:
        path = self.result_var.get()
        if not path:
            messagebox.showinfo("그래프", "먼저 변환을 실행하세요.")
            return
        try:
            PlotFrame = _sibling("plot_tracks_gui").PlotFrame
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("그래프", f"plot_tracks_gui.pyw를 불러오지 못했습니다.\n{exc}")
            return
        win = tk.Toplevel(self)
        win.title(f"트랙 그래프 - {Path(path).name}")
        win.geometry("1250x760")
        PlotFrame(win, path).pack(fill="both", expand=True)

    def open_csv(self) -> None:
        path = self.result_var.get()
        if not path:
            messagebox.showinfo("CSV", "먼저 변환을 실행하세요.")
            return
        c = csv_output(Path(path))
        if c.exists():
            open_path(c)
        else:
            messagebox.showinfo("CSV", f"CSV가 없습니다. '트랙 CSV도 저장'을 켜고 다시 변환하세요.\n{c}")

    def open_folder(self) -> None:
        path = self.result_var.get()
        folder = Path(path).parent if path else (Path(self.out_var.get()) if self.out_var.get() else RECORDINGS)
        if folder.is_dir():
            open_path(folder)


def main() -> None:
    set_dpi_aware()
    root = tk.Tk()
    setup_tk(root)
    root.title("PCR → NumPy 변환")
    root.geometry("900x640")
    ConvertFrame(root, sys.argv[1:]).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
