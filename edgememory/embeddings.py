from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("embedding has zero norm")
    return [value / norm for value in values]


def _json_objects(stdout: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _parse_embedding(stdout: str) -> list[float]:
    for value in _json_objects(stdout):
        data = value.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    return _normalize(item["embedding"])
        if isinstance(value.get("embedding"), list):
            return _normalize(value["embedding"])
    raise ValueError("embedding backend returned no parseable vector")


class LlamaCppEmbeddingProvider:
    """Adapter for the official llama.cpp embedding executable; no kernels are reimplemented."""

    def __init__(self, executable: str, model_path: str, *, threads: int = 2,
                 timeout_s: int = 60, extra_args: Sequence[str] = (),
                 model_id: str | None = None):
        self.executable = executable
        self.model_path = model_path
        self.threads = threads
        self.timeout_s = timeout_s
        self.extra_args = tuple(extra_args)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id or f"llama.cpp:{self.model_path}"

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty text")
        command = [self.executable, "-m", self.model_path, "-p", text,
                   "--pooling", "mean", "--embd-normalize", "2",
                   "--embd-output-format", "json", "-t", str(self.threads),
                   *self.extra_args]
        try:
            result = subprocess.run(command, check=False, capture_output=True,
                                    text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("llama.cpp embedding command failed or timed out") from error
        if result.returncode != 0:
            raise RuntimeError(f"llama.cpp embedding exited with status {result.returncode}")
        return _parse_embedding(result.stdout)


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, base_url: str, model: str, *, api_key_env: str | None = None,
                 timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"model": self._model, "input": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(f"{self.base_url}/embeddings", body, headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError("embedding service request failed") from error
        try:
            return _normalize(payload["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("embedding service returned an invalid response") from error


class DeterministicTestEmbedder:
    """Simulation-only deterministic vectors for unit tests; never use for evaluation."""

    def __init__(self, dimensions: int = 64):
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"SIMULATION_ONLY:hash-vector-{self.dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\u3400-\u9fff]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        if not any(vector):
            vector[0] = 1.0
        return _normalize(vector)
