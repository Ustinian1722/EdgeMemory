"""Persistent llama.cpp server process adapter; inference stays in upstream llama.cpp."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import math


class LlamaCppServer:
    def __init__(self, *, executable: Path, model_path: Path, work_root: Path,
                 model_id: str, port: int = 8088, context: int = 2048,
                 threads: int = 4, max_tokens: int = 192,
                 startup_timeout_s: int = 120, request_timeout_s: int = 180,
                 mode: str = "chat"):
        if not 1024 <= port <= 65535 or context < 256 or threads < 1 or max_tokens < 1:
            raise ValueError("invalid llama.cpp server settings")
        self.executable = executable.resolve()
        self.model_path = model_path.resolve()
        self.work_root = work_root
        self._model_id = model_id
        self.port = port
        self.context = context
        self.threads = threads
        self.max_tokens = max_tokens
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        if mode not in {"chat", "embedding"}:
            raise ValueError("llama.cpp server mode must be chat or embedding")
        self.mode = mode
        self.process: subprocess.Popen | None = None
        self.log_stream = None
        self.start_count = 0
        self.log_path = work_root / "logs" / "llama-server.log"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_id(self) -> str:
        return f"llama.cpp-cpu-{self._model_id}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def start(self) -> None:
        if self.process and self.process.poll() is None and self._healthy():
            return
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise FileNotFoundError("configured ARM64 llama-server is missing or not executable")
        if not self.model_path.is_file():
            raise FileNotFoundError("configured Qwen3-4B model is missing")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = self.log_path.open("ab", buffering=0)
        argv = [str(self.executable), "--model", str(self.model_path),
                "--ctx-size", str(self.context), "--threads", str(self.threads),
                "--host", "127.0.0.1", "--port", str(self.port)]
        if self.mode == "embedding":
            argv.extend(("--embedding", "--pooling", "mean"))
        self.process = subprocess.Popen(argv, cwd=str(self.work_root),
                                        stdin=subprocess.DEVNULL,
                                        stdout=self.log_stream, stderr=self.log_stream,
                                        close_fds=True)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("llama.cpp server exited during model initialization; inspect device log")
            if self._healthy():
                self.start_count += 1
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama.cpp server did not become healthy before the startup deadline")

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> tuple[str, dict[str, int | float]]:
        if self.mode != "chat":
            raise RuntimeError("embedding server does not accept chat completions")
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("persistent llama.cpp server is not running")
        token_limit = self.max_tokens if max_tokens is None else min(self.max_tokens, int(max_tokens))
        if token_limit < 1:
            raise ValueError("LLM request token limit must be positive")
        body = json.dumps({"model": self.model_id,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": token_limit,
                           "temperature": 0, "stream": False},
                          ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
        started = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError("llama.cpp completion request failed") from error
        elapsed_us = (time.perf_counter_ns() - started) // 1000
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise ValueError("llama.cpp chat response has no text content")
        usage = payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        metrics: dict[str, int | float] = {"request_elapsed_us": int(elapsed_us)}
        if isinstance(completion_tokens, int) and completion_tokens >= 0:
            metrics["output_tokens"] = completion_tokens
            if elapsed_us > 0:
                metrics["decode_tokens_per_second_estimated_e2e"] = (
                    completion_tokens * 1_000_000.0 / elapsed_us)
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
            metrics["input_tokens"] = prompt_tokens
        return content, metrics

    def embed(self, text: str) -> list[float]:
        if self.mode != "embedding":
            raise RuntimeError("chat server is not configured for embeddings")
        if not text.strip():
            raise ValueError("cannot embed empty text")
        body = json.dumps({"model": self.model_id, "input": text},
                          ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/v1/embeddings", body,
            {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError("llama.cpp embedding request failed") from error
        try:
            vector = [float(value) for value in payload["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("llama.cpp server returned an invalid embedding") from error
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("llama.cpp server returned a non-finite embedding")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            raise ValueError("llama.cpp server returned a zero-norm embedding")
        return [value / norm for value in vector]

    def snapshot(self) -> dict:
        return {"backend_id": self.backend_id, "model_id": self.model_id,
                "process_state": "ready" if self.process and self.process.poll() is None
                and self._healthy() else "stopped",
                "process_pid": self.process.pid if self.process and self.process.poll() is None else None,
                "model_file_bytes": self.model_path.stat().st_size if self.model_path.is_file() else None,
                "model_load_count": self.start_count,
                "execution_family": "CPU; accelerator offload is not configured"}

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self.log_stream:
            self.log_stream.close()
            self.log_stream = None
