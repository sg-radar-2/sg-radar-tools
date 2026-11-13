"""Radar Studio 실시간 수신 (UDP)

Radar Studio(radar-studio/src/device/live_relay.cpp)가 센서에서 받은 프레임마다 타겟 정보를
UDP 127.0.0.1:29190으로 보냅니다. 이 모듈은 그걸 받아서 situation_core가 읽는 행(rows) 형식으로 모아 둡니다.
(rows_from_pcr와 같은 키: session, time_s, frame_id, target_id, x, y, status, n_points, doppler_mean, doppler_std)

패킷 (little-endian, 프레임 하나 = 데이터그램 하나):
    char[4] "SGLV", uint32 version, uint32 frameCount, uint64 deltaUs, SensorSpec(float 16개),
    uint32 targetCount, target[targetCount] = Target(40바이트) + uint32 pointCount + float 도플러 평균/절댓값 평균/표준편차
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque

import situation_core as C

DEFAULT_PORT = 29190
SESSION = "실시간"
HEAD = struct.Struct("<4sIIQ16fI")          # 88바이트
TARGET = struct.Struct("<ffII6fIfff")        # Target 40바이트 + 포인트 수 + 도플러 3개 = 56바이트
MAX_DELTA_S = 1.0                            # 프레임 간격이 이보다 크면(연결이 끊겼다 다시 됨 등) 이만큼만 진행


class LiveReceiver:
    """UDP로 프레임을 받아 최근 window_s초의 행을 모아 둔다 (뒤 스레드)."""

    def __init__(self, port: int = DEFAULT_PORT, window_s: float = 180.0):
        self.port = port
        self.window_s = window_s
        self._rows: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.t = 0.0                 # 받은 프레임들의 누적 시간 (초)
        self.version = 0             # 프레임을 받을 때마다 1씩 증가
        self.frames = 0
        self.spec: dict | None = None
        self.last_rx = 0.0           # 마지막으로 받은 시각 (time.monotonic)
        self.error: str | None = None

    def start(self) -> None:
        """포트를 연다. 이미 쓰고 있으면 OSError."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", self.port))
        s.settimeout(0.3)
        self._sock = s
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def snapshot(self) -> tuple[list[dict], int]:
        with self._lock:
            return list(self._rows), self.version

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:       # 닫힘 또는 (Windows) 이전 전송 오류
                if self._stop.is_set():
                    return
                if getattr(exc, "winerror", None) == 10054:
                    continue
                self.error = str(exc)
                return
            try:
                self._parse(data)
            except (struct.error, ValueError):
                continue                 # 깨진 패킷은 버림

    def _parse(self, data: bytes) -> None:
        if len(data) < HEAD.size or data[:4] != b"SGLV":
            return
        h = HEAD.unpack_from(data, 0)
        _, ver, frame_id, delta_us = h[:4]
        spec = dict(zip(C.SPEC_FIELDS, (float(v) for v in h[4:20])))
        n = h[20]
        dt = min(delta_us / 1e6, MAX_DELTA_S)
        rows = []
        with self._lock:
            self.t += dt
            t = self.t
        off = HEAD.size
        for _ in range(n):
            (x, y, status, tid, _x0, _x1, _y0, _y1, _z0, _z1,
             npts, dmean, dabs, dstd) = TARGET.unpack_from(data, off)
            off += TARGET.size
            rows.append({"session": SESSION, "time_s": t, "frame_id": int(frame_id), "target_id": int(tid),
                         "x": float(x), "y": float(y), "status": int(status), "n_points": int(npts),
                         "doppler_mean": dmean, "doppler_abs_mean": dabs, "doppler_std": dstd})
        with self._lock:
            self._rows.extend(rows)
            while self._rows and self._rows[0]["time_s"] < t - self.window_s:
                self._rows.popleft()
            self.spec = spec
            self.frames += 1
            self.version += 1
            self.last_rx = time.monotonic()
