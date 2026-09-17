#!/usr/bin/env python3
"""Browser lab for the askable arm.

    source scripts/vulkan_env.sh
    uv run python scripts/ask_arm_server.py

Open the printed URL. The live view is the simulated Franka.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_robotics.ask_session import AskArmSession
from jev_robotics.common import jsonable

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "ask_arm" / "index.html"
DEMO = ROOT / "web" / "ask_arm" / "demo.html"


def make_handler(session: AskArmSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(jsonable(payload)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode())

        def _html(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path)
            if route.path in {"/", "/demo", "/demo.html"}:
                self._html(DEMO)
            elif route.path in {"/lab", "/index.html"}:
                self._html(PAGE)
            elif route.path == "/api/meta":
                self._json(session.meta())
            elif route.path == "/api/state":
                self._json(session.public_state())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path)
            try:
                if route.path == "/api/boot":
                    self._json(session.boot_demo())
                elif route.path == "/api/say":
                    body = self._read_json()
                    self._json(session.say(str(body.get("goal") or "")), 202)
                elif route.path == "/api/reset":
                    body = self._read_json()
                    state = session.reset(
                        env_id=str(body.get("env_id") or "PickCube-v1"),
                        seed=int(body.get("seed") or 0),
                        goal=str(body.get("goal") or "").strip(),
                        task_name=body.get("task_name") or None,
                        max_decisions=body.get("max_decisions"),
                    )
                    self._json(state)
                elif route.path == "/api/step":
                    self._json(session.step_once())
                elif route.path == "/api/run":
                    session.start_run()
                    self._json({"ok": True}, 202)
                elif route.path == "/api/stop":
                    session.stop()
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._json({"error": str(error)}, 400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()
    session = AskArmSession(camera=not args.no_camera)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"Askable arm: http://{args.host}:{args.port}", flush=True)
    print(f"Lab UI:      http://{args.host}:{args.port}/lab", flush=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="ask-arm-http")
    thread.start()
    try:
        session.serve_sim()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop_event.set()
        server.shutdown()
        session.close()


if __name__ == "__main__":
    main()
