
import json
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

SAVE_DIR = Path.home() / "rad2_captures"
SAVE_DIR.mkdir(exist_ok=True)


class CameraHandler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/capture":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")

            event = json.loads(self.rfile.read(size))
            mode = event.get("capture", "photo")

            if mode not in ("photo", "video"):
                raise ValueError("capture must be photo or video")

            duration = max(1, min(int(event.get("duration_s", 5)), 10))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_id = uuid4().hex[:6]
            extension = "jpg" if mode == "photo" else "mp4"
            path = SAVE_DIR / f"{timestamp}_{unique_id}.{extension}"

            if mode == "photo":
                cmd = [
                    "rpicam-still", "-n", "-t", "1000",
                    "-o", str(path)
                ]
            else:
                cmd = [
                    "rpicam-vid", "-n",
                    "-t", str(duration * 1000),
                    "--codec", "libav",
                    "--libav-format", "mp4",
                    "-o", str(path)
                ]

            subprocess.run(cmd, check=True, timeout=duration + 40)

            self.send_json(200, {
                "ok": True,
                "event_id": event.get("event_id"),
                "capture": mode,
                "file": str(path)
            })

        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"ok": False, "error": str(exc)})
        except (subprocess.SubprocessError, OSError) as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


server = HTTPServer(("0.0.0.0", 8001), CameraHandler)
print("Camera Server listening on port 8001", flush=True)
server.serve_forever()