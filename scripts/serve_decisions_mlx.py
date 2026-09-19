#!/usr/bin/env python3
"""Serve NanoJev's native non-generative decision API on Apple Silicon MLX."""
from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from mlx_decisions import MLXDecisionPredictor
from predict_toy_decisions import reject_nonfinite, unique_object, validate_request


def server_class(engine: MLXDecisionPredictor, web_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, mime="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, code, data):
            self.send(code, json.dumps(data, ensure_ascii=False, allow_nan=False).encode())

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/health":
                self.send_json(200, {"ready": True, "backend": "mlx", "model_loaded_once": True,
                                     "provider_calls": 0})
                return
            relative = unquote(route).lstrip("/") or "index.html"
            target = (web_root / relative).resolve()
            if not target.is_relative_to(web_root) or not target.is_file():
                self.send_json(404, {"error": "File not found"})
                return
            self.send(200, target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")

        def do_POST(self):
            if urlparse(self.path).path != "/api/evaluate":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Request must contain 1..2000000 bytes")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Cross-origin requests are disabled")
                payload = json.loads(self.rfile.read(length), object_pairs_hook=unique_object,
                                     parse_constant=reject_nonfinite)
                states = validate_request(payload)
                questions = [q for state in states for q in state["questions"].values()]
                paths = sum(1 if q["type"] == "boolean" else len(q["criteria"]) for q in questions)
                if len(states) > 32 or len(questions) > 96 or paths > 256:
                    raise ValueError("MLX demo limit:32 states,96 questions,256 candidate paths per request")
                before = time.perf_counter()
                result = engine.predict(payload)
                result["execution"]["server_evaluation_seconds"] = time.perf_counter() - before
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                self.send_json(500, {"error": "MLX model inference failed; inspect the server process."})
                raise

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Converted MLX Qwen3 backbone directory")
    parser.add_argument("--decision-head", required=True, help="NanoJev best.safetensors checkpoint")
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    engine = MLXDecisionPredictor(args.model_dir, args.decision_head, max_length=args.max_length)
    root = Path(args.web_root).resolve()
    if not (root / "index.html").is_file():
        raise ValueError("web-root must contain index.html")
    server = HTTPServer((args.host, args.port), server_class(engine, root))
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "ready": True,
                      "backend": "mlx", "provider_calls": 0}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
