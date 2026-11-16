from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .types import SearchHit


class AnswerProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def answer(self, question: str, evidence: Sequence[SearchHit]) -> dict: ...


class AnswerGenerationError(RuntimeError):
    pass


class OpenAICompatibleAnswerProvider:
    """Generate citation-bearing claims from retrieved event records."""

    def __init__(self, base_url: str, model: str, *, api_key_env: str | None = None,
                 timeout_s: int = 90, max_tokens: int = 384):
        if not base_url.strip() or not model.strip():
            raise ValueError("answer URL and model are required")
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model

    def answer(self, question: str, evidence: Sequence[SearchHit]) -> dict:
        allowed = {hit.event_id for hit in evidence if not hit.simulation_only}
        if not allowed:
            return {
                "status": "insufficient_evidence",
                "claims": [],
                "missing_information": ["No non-simulation video evidence matched this question."],
                "model_id": self.model_id,
            }
        records = [{
            "event_id": hit.event_id,
            "video": hit.media_title,
            "video_offset_ms": hit.timestamp_ms,
            "caption": hit.caption,
        } for hit in evidence if hit.event_id in allowed]
        instructions = (
            "Answer using only the supplied video event records. Do not add objects, actions, "
            "times, identities, or causal claims that the records do not support. Return one JSON "
            "object with keys claims and missing_information. claims is an array of objects with "
            "kind (observed or inference), text, and event_ids. Every claim must cite one or more "
            "event_ids copied exactly from the evidence. Put unknown or unsupported details in "
            "missing_information. If evidence is insufficient, return no claims. Do not identify "
            "people or infer sensitive personal traits."
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps({
                    "question": question,
                    "evidence": records,
                }, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise AnswerGenerationError(f"answer backend returned HTTP {error.code}") from error
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise AnswerGenerationError("answer backend request failed") from error
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AnswerGenerationError("answer backend returned an invalid chat response") from error
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        if not isinstance(content, str):
            raise AnswerGenerationError("answer backend returned non-text content")
        try:
            generated = json.loads(content)
        except json.JSONDecodeError as error:
            raise AnswerGenerationError("answer backend did not return the required JSON structure") from error
        claims = generated.get("claims")
        missing = generated.get("missing_information", [])
        if not isinstance(claims, list) or not isinstance(missing, list):
            raise AnswerGenerationError("answer backend returned malformed claim fields")
        checked_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise AnswerGenerationError("answer backend returned a malformed claim")
            kind = claim.get("kind")
            text = claim.get("text")
            citations = claim.get("event_ids")
            if (kind not in {"observed", "inference"} or not isinstance(text, str)
                    or not text.strip() or not isinstance(citations, list) or not citations
                    or any(not isinstance(item, str) or item not in allowed for item in citations)):
                raise AnswerGenerationError("answer backend returned an uncited or invalid claim")
            checked_claims.append({
                "kind": kind,
                "text": text.strip(),
                "event_ids": list(dict.fromkeys(citations)),
            })
        checked_missing = [item.strip() for item in missing
                           if isinstance(item, str) and item.strip()]
        return {
            "status": "answered" if checked_claims else "insufficient_evidence",
            "claims": checked_claims,
            "missing_information": checked_missing,
            "model_id": self.model_id,
        }


class GenieXVlmAnswerProvider:
    """Answer from retrieved events using the existing persistent on-device VLM worker."""

    def __init__(self, captioner, *, max_evidence: int = 4):
        if not callable(getattr(captioner, "describe", None)):
            raise ValueError("an on-device VLM captioner is required")
        if not 1 <= max_evidence <= 12:
            raise ValueError("max_evidence must be in 1..12")
        self.captioner = captioner
        self.max_evidence = max_evidence

    @property
    def model_id(self) -> str:
        return str(self.captioner.model_id)

    def answer(self, question: str, evidence: Sequence[SearchHit]) -> dict:
        if len(question.encode("utf-8")) > 1536:
            raise AnswerGenerationError("question is too long for the on-device answer request")
        workspace = Path(__file__).resolve().parents[1]
        selected: list[SearchHit] = []
        for hit in evidence:
            if hit.simulation_only:
                continue
            try:
                frame = Path(hit.frame_path).expanduser().resolve(strict=True)
                frame.relative_to(workspace)
            except (OSError, ValueError):
                continue
            if frame.is_file():
                selected.append(hit)
            if len(selected) == self.max_evidence:
                break
        allowed = {hit.event_id for hit in selected}
        if not selected:
            return {
                "status": "insufficient_evidence",
                "claims": [],
                "missing_information": ["No non-simulation video frame evidence is available."],
                "model_id": self.model_id,
            }

        records = [{
            "event_id": hit.event_id,
            "video": hit.media_title[:160],
            "video_offset_ms": hit.timestamp_ms,
            "caption": hit.caption[:240],
        } for hit in selected]
        prompt = (
            "You answer questions about indexed videos. Use only the supplied event records and the attached "
            "representative video frame. Treat event records as retrieved observations, not permission to "
            "invent unseen events, identities, causes, or exact times. Return exactly one JSON object with "
            "keys claims and missing_information. Each claim must contain kind (observed or inference), text, "
            "and event_ids copied exactly from the evidence. Every claim must cite one or more event IDs. "
            "If the evidence is insufficient, return an empty claims array and explain what is missing. "
            "Do not identify people or infer sensitive personal traits. Do not use Markdown or add text outside JSON.\n"
            + json.dumps({"question": question, "evidence": records}, ensure_ascii=False)
        )
        try:
            result = self.captioner.describe(Path(selected[0].frame_path), prompt)
        except Exception as error:
            raise AnswerGenerationError(
                f"on-device answer backend failed ({type(error).__name__})") from None

        generated = _decode_json_object(result.text)
        claims = generated.get("claims")
        missing = generated.get("missing_information", [])
        # Small on-device models sometimes serialize a single item as an object
        # or a missing-information list as a string. Normalize those shapes
        # without relaxing the requirement for exact evidence citations.
        if claims is None:
            claims = []
        elif isinstance(claims, dict):
            claims = [claims]
        if isinstance(missing, str):
            missing = [missing]
        elif missing is None:
            missing = []
        if not isinstance(claims, list) or not isinstance(missing, list):
            raise AnswerGenerationError("on-device answer backend returned malformed claim fields")
        checked_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise AnswerGenerationError("on-device answer backend returned a malformed claim")
            kind = claim.get("kind")
            text = claim.get("text")
            citations = claim.get("event_ids")
            if isinstance(citations, str):
                citations = [citations]
            if (kind not in {"observed", "inference"} or not isinstance(text, str)
                    or not text.strip() or not isinstance(citations, list) or not citations
                    or any(not isinstance(item, str) or item not in allowed for item in citations)):
                raise AnswerGenerationError("on-device answer backend returned an uncited or invalid claim")
            checked_claims.append({
                "kind": kind,
                "text": text.strip(),
                "event_ids": list(dict.fromkeys(citations)),
            })
        checked_missing = [item.strip() for item in missing
                           if isinstance(item, str) and item.strip()]
        return {
            "status": "answered" if checked_claims else "insufficient_evidence",
            "claims": checked_claims,
            "missing_information": checked_missing,
            "model_id": self.model_id,
            "backend_id": result.backend_id,
            "inference_metrics": result.inference_metrics,
        }


def _decode_json_object(text: str) -> dict:
    """Find a valid JSON response even when a model prepends a short reasoning token block."""
    if not isinstance(text, str):
        raise AnswerGenerationError("answer backend returned non-text content")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "claims" in value:
            return value
    raise AnswerGenerationError("answer backend did not return the required JSON structure")
