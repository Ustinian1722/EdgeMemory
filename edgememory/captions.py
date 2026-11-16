from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

from .types import CaptionResult


CAPTION_PROMPT = (
    "Describe only directly visible objects, actions, and scene changes in one concise sentence. "
    "Do not identify people or infer identity, age, emotion, intent, or sensitive traits. "
    "Do not invent details that are not visible."
)


class Captioner(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def simulation_only(self) -> bool: ...

    def describe(self, image_path: Path, prompt: str = CAPTION_PROMPT) -> CaptionResult: ...


class MockCaptioner:
    """Explicit simulation fixture. Its outputs must never be reported as model results."""

    def __init__(self, captions: Sequence[str] = ("SIMULATED fixture caption",)):
        if not captions or any(not value.strip() for value in captions):
            raise ValueError("mock captions must be non-empty")
        self.captions = tuple(captions)
        self._index = 0

    @property
    def backend_id(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "SIMULATION_ONLY:fixture-captioner"

    @property
    def simulation_only(self) -> bool:
        return True

    def describe(self, image_path: Path, prompt: str = CAPTION_PROMPT) -> CaptionResult:
        caption = self.captions[self._index % len(self.captions)]
        self._index += 1
        return CaptionResult(caption, self.backend_id, self.model_id, True)


class OpenAICompatibleVlmCaptioner:
    """OpenAI-compatible VLM chat endpoint, including GenieX serve where enabled."""

    def __init__(self, base_url: str, model: str, *, api_key_env: str | None = None,
                 timeout_s: int = 90, max_tokens: int = 96):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    @property
    def backend_id(self) -> str:
        return "openai-compatible-vlm"

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def simulation_only(self) -> bool:
        return False

    def describe(self, image_path: Path, prompt: str = CAPTION_PROMPT) -> CaptionResult:
        image_path = image_path.resolve(strict=True)
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}"}},
            ]}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            json.dumps(body).encode("utf-8"), headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"VLM service returned HTTP {error.code}") from error
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError("VLM service request failed") from error
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("VLM service returned an invalid chat response") from error
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("VLM service returned an empty caption")
        return CaptionResult(content.strip(), self.backend_id, self.model_id, False)
