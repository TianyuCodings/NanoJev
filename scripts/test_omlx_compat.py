#!/usr/bin/env python3
"""Smoke-test discovery of a converted NanoJev backbone in native oMLX.

This verifies the integration boundary that oMLX owns: model discovery and
ordinary text generation. NanoJev's structured decision endpoint remains the
MLX service in ``serve_decisions_mlx.py`` because oMLX's OpenAI-compatible text
API does not expose final hidden states or a custom decision-head hook.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, api_key: str, payload: dict | None = None, timeout: float = 2.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method="GET" if data is None else "POST")
    request.add_header("Authorization", f"Bearer {api_key}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Converted MLX backbone directory")
    parser.add_argument("--omlx-cli", default="omlx", help="oMLX CLI executable")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    cli = shutil.which(args.omlx_cli) or args.omlx_cli
    api_key = "nanojev-compat-test"
    port = free_port()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nanojev-omlx-compat-") as temp_dir:
        root = Path(temp_dir)
        model_root = root / "models"
        model_root.mkdir()
        (model_root / "nanojev-backbone").symlink_to(model_dir, target_is_directory=True)
        base_path = root / "omlx-base"
        command = [
            cli, "serve", "--model-dir", str(model_root),
            "--host", "127.0.0.1", "--port", str(port),
            "--no-cache", "--memory-guard", "off", "--no-hf-cache",
            "--api-key", api_key, "--base-path", str(base_path),
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + args.timeout
            models = None
            last_error = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"oMLX exited before readiness (code {process.returncode})")
                try:
                    models = request_json(f"{base_url}/v1/models", api_key)
                    break
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                    time.sleep(0.25)
            if models is None:
                raise RuntimeError(f"timed out waiting for oMLX: {last_error}")
            model_ids = [item.get("id") for item in models.get("data", [])]
            if "nanojev-backbone" not in model_ids:
                raise AssertionError(f"oMLX did not discover nanojev-backbone: {model_ids}")
            response = request_json(
                f"{base_url}/v1/chat/completions", api_key,
                {
                    "model": "nanojev-backbone",
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "max_tokens": 2,
                    "temperature": 0,
                },
                timeout=args.timeout,
            )
            if not response.get("choices"):
                raise AssertionError(f"oMLX returned no choices: {response}")
            print(json.dumps({
                "status": "passed",
                "model": "nanojev-backbone",
                "discovered_models": model_ids,
                "chat_choices": len(response["choices"]),
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "native oMLX model discovery and ordinary text-generation smoke test; structured NanoJev decisions use /api/evaluate",
            }, ensure_ascii=False, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
