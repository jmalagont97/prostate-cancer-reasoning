"""HTTP adapter for the Grand Challenge container contract.

Ollama is started by entrypoint.sh. This process stays alive and runs the
existing filesystem-based inference pipeline once for each POST /invoke.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger(__name__)
_RUN_LOCK = threading.Lock()
READY_FILE = Path(os.environ.get("CHIMERA_READY_FILE", "/tmp/chimera-gemma.ready"))


def is_ready() -> bool:
    """Return whether the entrypoint finished loading and warming up Ollama."""
    return READY_FILE.is_file()


def run_inference(config_path: str) -> None:
    """Import the top-level production entrypoint only when an invocation runs."""
    from inference import run_inference as pipeline

    pipeline(config_path)


class SubmissionHandler(BaseHTTPRequestHandler):
    server_version = "CHIMERA-Submission/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            if is_ready():
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(503, {"status": "loading"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/invoke":
            self._send_json(404, {"error": "not found"})
            return

        if not is_ready():
            self._send_json(503, {"error": "model is still loading"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)  # GC sends the data through mounted paths.

        if not _RUN_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "an invocation is already running"})
            return
        try:
            run_inference(os.environ.get("CHIMERA_CONFIG", "/opt/app/configs/config.yaml"))
        except Exception as exc:  # return a useful GC error; preserve traceback in logs
            log.exception("Invocation failed")
            self._send_json(500, {"error": str(exc)})
            return
        finally:
            _RUN_LOCK.release()
        self._send_json(201, {"status": "success"})

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    port = int(os.environ.get("PORT", "4743"))
    server = ThreadingHTTPServer(("0.0.0.0", port), SubmissionHandler)
    log.info("Grand Challenge API listening on 0.0.0.0:%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
