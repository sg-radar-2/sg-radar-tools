"""누적으로 저장된 .pcr에서 '새로 찍힌 부분'만 남기기

Radar Studio에서 세션을 새로 만들지 않고 녹화를 이어 하면서 중간중간 저장하면
    1.pcr = A
    2.pcr = A + B
    3.pcr = A + B + C
처럼 앞 내용이 계속 들어갑니다. 이 스크립트는 파일들을 저장된 순서대로 보면서, 앞 파일들에 이미 있던
프레임을 빼고 새 프레임만 남긴 .pcr을 만듭니다 (원본은 그대로 두고 split 폴더에 새로 저장).
    split/1.pcr = A,  split/2.pcr = B,  split/3.pcr = C

- 같은 프레임인지는 프레임의 바이트 전체(센서 프레임 번호, 시간 간격, 모든 포인트·타겟)로 비교합니다.
- 순서는 파일 헤더의 저장 시각 순입니다 (파일 이름 순서와 달라도 됨).
- 중간에 지우고 다시 찍은 파일(앞 내용이 없는 파일)도 그대로 맞게 처리됩니다.
- 맨 앞에 혼자 떨어져 있는 프레임(예: 세션을 만들 때 들어간 프레임 1개 뒤로 수십 초가 비는 경우)은
  기본으로 뺍니다. 남기려면 --keep-orphans.

실행 (sg-radar-tools 폴더에서):
    python tools\\pcr\\split_cumulative.py recordings\\1-*.pcr recordings\\2-*.pcr
    python tools\\pcr\\split_cumulative.py recordings\\a.pcr recordings\\b.pcr --out recordings\\fixed
    python tools\\pcr\\split_cumulative.py recordings\\1-*.pcr --dry-run      # 저장 없이 결과만 보기

필요: Python 3.14+ 또는 pip install zstandard
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

HEADER = struct.Struct("<4s4sQ16fQQ32sQQ")
PAYLOAD_MAGIC = 0x0807060504030201
POINT_SIZE, TARGET_SIZE = 24, 40
ORPHAN_GAP_FRAMES = 40      # 프레임 번호가 이만큼(약 2초) 이상 건너뛰면 '떨어져 있다'고 봄
ORPHAN_MAX_FRAMES = 3       # 맨 앞에서 이 개수 이하로 떨어져 있으면 빼기 대상


# --------------------------------------------------------------------------- zstd

def _decompress_stream(f):
    try:
        from compression import zstd  # Python 3.14+
        return zstd.ZstdFile(f, mode="rb")
    except ImportError:
        import zstandard
        return io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 20)


def _compress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
        return zstd.compress(data, level=3)
    except ImportError:
        import zstandard
        return zstandard.ZstdCompressor(level=3).compress(data)


# --------------------------------------------------------------------------- 읽기

@dataclass
class Frame:
    blob: bytes          # 프레임 전체 바이트 (그대로 다시 씀)
    frame_count: int
    delta_us: int
    nbytes: int          # 프레임 끝의 bytes 값 (세션 bytes 합계용)

    @property
    def key(self) -> bytes:
        return hashlib.blake2b(self.blob, digest_size=16).digest()


@dataclass
class Session:
    id: int
    timestamp: int
    name: bytes
    description: bytes
    tag: bytes
    frames: list = field(default_factory=list)
    bookmarks: list = field(default_factory=list)   # (description bytes, frame index)


@dataclass
class Recording:
    path: Path
    header: tuple
    sessions: list

    @property
    def saved_at(self) -> int:
        return self.header[2]


class _Reader:
    def __init__(self, raw):
        self.raw = raw

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.raw.read(n - len(out))
            if not chunk:
                raise ValueError("파일이 중간에 끝났습니다 (손상 가능)")
            out += chunk
        return bytes(out)

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> bytes:
        return self.read(self.u64())


def read_pcr(path: Path) -> Recording:
    with open(path, "rb") as f:
        raw_header = f.read(HEADER.size)
        h = HEADER.unpack(raw_header)
        if h[0] != b"PCR1" or h[1] != b"1000":
            raise ValueError(f"{path.name}: PCR 1.0 파일이 아닙니다")
        session_count = h[19]
        r = _Reader(_decompress_stream(f))
        if r.u64() != PAYLOAD_MAGIC:
            raise ValueError(f"{path.name}: 페이로드가 올바르지 않습니다")
        sessions = []
        for _ in range(session_count):
            sid = r.u32()
            ts = r.u64()
            r.u64()   # lengthUs (다시 계산)
            r.u64()   # bytes    (다시 계산)
            s = Session(sid, ts, r.string(), r.string(), r.string())
            for _ in range(r.u64()):
                head = r.read(16)
                packet, fcount, delta = struct.unpack("<IIQ", head)
                npts_b = r.read(8)
                pts = r.read(struct.unpack("<Q", npts_b)[0] * POINT_SIZE)
                ntg_b = r.read(8)
                tgs = r.read(struct.unpack("<Q", ntg_b)[0] * TARGET_SIZE)
                tail = r.read(8)
                s.frames.append(Frame(head + npts_b + pts + ntg_b + tgs + tail, fcount, delta,
                                      struct.unpack("<Q", tail)[0]))
            for _ in range(r.u64()):
                s.bookmarks.append((r.string(), r.u64()))
            sessions.append(s)
    return Recording(path, h, sessions)


# --------------------------------------------------------------------------- 쓰기

def _str(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def write_pcr(path: Path, header: tuple, sessions: list) -> None:
    body = bytearray(struct.pack("<Q", PAYLOAD_MAGIC))
    total = 0
    for s in sessions:
        length_us = sum(fr.delta_us for fr in s.frames)
        nbytes = sum(fr.nbytes for fr in s.frames)
        body += struct.pack("<IQQQ", s.id, s.timestamp, length_us, nbytes)
        body += _str(s.name) + _str(s.description) + _str(s.tag)
        body += struct.pack("<Q", len(s.frames))
        for fr in s.frames:
            body += fr.blob
        body += struct.pack("<Q", len(s.bookmarks))
        for desc, idx in s.bookmarks:
            body += _str(desc) + struct.pack("<Q", idx)
        total += len(s.frames)
    payload = bytes(body)
    comp = _compress(payload)
    h = list(header)
    h[19], h[20] = len(sessions), total              # sessionCount, totalFrameCount
    h[21] = bytes(32)                                 # reserved
    h[22], h[23] = len(payload), len(comp)            # uncompressed, compressed size
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(HEADER.pack(*h))
        f.write(comp)


# --------------------------------------------------------------------------- 분리

def drop_leading_orphans(frames: list) -> tuple[list, int]:
    """맨 앞에 몇 프레임만 따로 떨어져 있으면 뺀다."""
    for i in range(1, min(ORPHAN_MAX_FRAMES, len(frames) - 1) + 1):
        if frames[i].frame_count - frames[i - 1].frame_count >= ORPHAN_GAP_FRAMES:
            return frames[i:], i
    return frames, 0


def split(paths: list[Path], out_dir: Path, keep_orphans: bool, dry_run: bool) -> None:
    recs = [read_pcr(p) for p in paths]
    recs.sort(key=lambda r: r.saved_at)
    seen: set[bytes] = set()
    print(f"저장 시각 순서: {', '.join(r.path.name for r in recs)}\n")
    for rec in recs:
        out_sessions = []
        line = []
        for s in rec.sessions:
            keep_idx = [i for i, fr in enumerate(s.frames) if fr.key not in seen]
            kept = [s.frames[i] for i in keep_idx]
            dropped_orphans = 0
            if not keep_orphans and kept:
                kept, dropped_orphans = drop_leading_orphans(kept)
                keep_idx = keep_idx[dropped_orphans:]
            # 북마크는 남은 프레임 기준으로 번호를 다시 매김
            remap = {old: new for new, old in enumerate(keep_idx)}
            bms = [(d, remap[i]) for d, i in s.bookmarks if i in remap]
            if kept:
                out_sessions.append(Session(s.id, s.timestamp, s.name, s.description, s.tag, kept, bms))
            sec = sum(fr.delta_us for fr in kept) / 1e6
            msg = (f"{s.name.decode('utf-8', 'replace')}: {len(s.frames)}프레임 중 "
                   f"{len(s.frames) - len(keep_idx) - dropped_orphans}개는 앞 파일과 중복, "
                   f"{len(kept)}개({sec:.1f}초) 남김")
            if dropped_orphans:
                msg += f" (맨 앞에 떨어진 프레임 {dropped_orphans}개 제외)"
            line.append(msg)
            for fr in s.frames:
                seen.add(fr.key)
        print(f"▶ {rec.path.name}")
        for m in line:
            print("   " + m)
        if not out_sessions:
            print("   → 새 프레임이 없어서 저장하지 않음\n")
            continue
        out = out_dir / rec.path.name
        if out.resolve() == rec.path.resolve():
            raise SystemExit("출력 폴더가 원본 폴더와 같습니다. --out으로 다른 폴더를 지정하세요.")
        if dry_run:
            print(f"   → (미리보기) {out}\n")
        else:
            write_pcr(out, rec.header, out_sessions)
            check = read_pcr(out)   # 다시 읽어서 확인
            n = sum(len(s.frames) for s in check.sessions)
            print(f"   → 저장: {out}  (다시 읽어서 {n}프레임 확인)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="누적 저장된 .pcr에서 새로 찍힌 부분만 남기기")
    ap.add_argument("files", nargs="+", help=".pcr 파일들 (와일드카드 가능: recordings\\1-*.pcr)")
    ap.add_argument("--out", type=Path, help="결과 폴더 (기본: 첫 파일 폴더 안의 split)")
    ap.add_argument("--keep-orphans", action="store_true", help="맨 앞에 떨어진 프레임도 남기기")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.files:      # Windows 명령 프롬프트는 *를 펼쳐주지 않아서 직접 펼침
        found = sorted(glob.glob(pat)) or [pat]
        paths += [Path(p) for p in found if Path(p) not in paths]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"파일을 찾을 수 없습니다: {', '.join(map(str, missing))}")
    out_dir = args.out or paths[0].parent / "split"
    split(paths, out_dir, args.keep_orphans, args.dry_run)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
