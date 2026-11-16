from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .answer import AnswerGenerationError, AnswerProvider
from .embeddings import EmbeddingProvider
from .jobs import IngestJobManager, IngestQueueFull
from .query import search_events, summary_events
from .store import EventStore


_STATIC_ROOT = Path(__file__).resolve().parent / "static"
_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
_UPLOAD_RESERVE_BYTES = 1024 * 1024 * 1024
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _byte_range(value: str, size: int) -> tuple[int, int] | None:
    value = value.strip()
    if not value.startswith("bytes=") or "," in value or size < 1:
        return None
    spec = value.removeprefix("bytes=").strip()
    if "-" not in spec:
        return None
    first, last = spec.split("-", 1)
    try:
        if first:
            if not first.isascii() or not first.isdigit():
                return None
            if last and (not last.isascii() or not last.isdigit()):
                return None
            start = int(first)
            end = int(last) if last else size - 1
        else:
            if not last.isascii() or not last.isdigit():
                return None
            suffix = int(last)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= size:
        return None
    return start, min(end, size - 1)


def make_handler(store: EventStore, artifact_root: Path,
                 embedder: EmbeddingProvider | None = None, *,
                 video_root: Path | None = None,
                 job_manager: IngestJobManager | None = None,
                 captioner=None,
                 answer_provider: AnswerProvider | None = None):
    root = artifact_root.resolve()
    allowed_video_root = video_root.resolve() if video_root is not None else None
    verified_videos: set[tuple[str, str, int, int]] = set()
    verification_lock = threading.Lock()

    def video_path(media_id: str, *, verify_hash: bool = False) -> tuple[Path | None, int]:
        if allowed_video_root is None:
            return None, 404
        media = store.media_item(media_id)
        if media is None:
            return None, 404
        try:
            source = Path(media["source_path"]).resolve()
        except (OSError, RuntimeError):
            return None, 404
        try:
            source.relative_to(allowed_video_root)
        except ValueError:
            return None, 403
        if not source.is_file():
            return None, 404
        if verify_hash:
            expected_hash = media["source_sha256"]
            if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                    or any(char not in "0123456789abcdefABCDEF" for char in expected_hash)):
                return None, 409
            try:
                initial_stat = source.stat()
                cache_key = (str(source), expected_hash.lower(),
                             initial_stat.st_size, initial_stat.st_mtime_ns)
                with verification_lock:
                    if cache_key not in verified_videos:
                        digest = hashlib.sha256()
                        with source.open("rb") as video_stream:
                            for chunk in iter(lambda: video_stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                        final_stat = source.stat()
                        if (final_stat.st_size != initial_stat.st_size
                                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                                or digest.hexdigest() != expected_hash.lower()):
                            return None, 409
                        verified_videos.add(cache_key)
            except OSError:
                return None, 409
        return source, 200

    def hit_payload(hit) -> dict:
        return {
            "event_id": hit.event_id, "media_id": hit.media_id,
            "media_title": hit.media_title, "start_ms": hit.start_ms,
            "end_ms": hit.end_ms, "timestamp_ms": hit.timestamp_ms,
            "caption": hit.caption, "backend_id": hit.backend_id,
            "model_id": hit.model_id, "frame_sha256": hit.frame_sha256,
            "similarity": hit.similarity, "simulation_only": hit.simulation_only,
            "inference_metrics": hit.inference_metrics,
            "video_available": video_path(hit.media_id)[0] is not None,
        }

    def public_event(event: dict) -> dict:
        # Never expose local source or artifact paths through the HTTP API.
        return {key: event[key] for key in (
            "event_id", "media_id", "media_title", "frame_index", "timestamp_ms",
            "start_ms", "end_ms", "frame_sha256", "sample_reason", "scene_delta",
            "caption", "caption_status", "error_code", "backend_id", "model_id",
            "simulation_only", "inference_metrics")}

    class Handler(BaseHTTPRequestHandler):
        server_version = "EdgeMemory/0.1"

        def log_message(self, format, *args):
            # Avoid request-body, source-path, or user-query logging.
            return

        def _write_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self, maximum: int = 16_384) -> dict:
            raw_length = self.headers.get("Content-Length", "0")
            if not raw_length.isascii() or not raw_length.isdigit():
                raise ValueError("invalid Content-Length")
            length = int(raw_length)
            if length < 1 or length > maximum:
                raise ValueError(f"request body size must be in 1..{maximum} bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = (_STATIC_ROOT / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/assets/app.css", "/assets/app.js"}:
                asset = _STATIC_ROOT / Path(parsed.path).name
                if not asset.is_file():
                    return self._write_json({"error": "asset not found"}, 404)
                body = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0]
                                 or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/health":
                return self._write_json({"status": "ok", "service": "EdgeMemory"})
            if parsed.path == "/api/status":
                return self._write_json({
                    "status": "ok", "service": "EdgeMemory",
                    "counts": store.dashboard_counts(),
                    "caption_backend_configured": captioner is not None,
                    "answer_backend_configured": answer_provider is not None,
                })
            if parsed.path == "/api/jobs":
                jobs = job_manager.list() if job_manager else []
                return self._write_json({"jobs": jobs})
            if parsed.path.startswith("/api/jobs/"):
                job_id = unquote(parsed.path.removeprefix("/api/jobs/"))
                job = job_manager.get(job_id) if job_manager else None
                if job is None:
                    return self._write_json({"error": "job not found"}, 404)
                return self._write_json({"job": job})
            if parsed.path == "/api/media":
                return self._write_json({"media": store.media()})
            if parsed.path == "/api/timeline":
                params = parse_qs(parsed.query)
                media_ids = params.get("media_id", [])
                if not media_ids or not media_ids[0]:
                    return self._write_json({"error": "media_id is required"}, 400)
                try:
                    events = store.timeline(unquote(media_ids[0]))
                except ValueError as error:
                    return self._write_json({"error": str(error)}, 400)
                return self._write_json({"events": [public_event(event) for event in events]})
            if parsed.path == "/api/runtime":
                observations = store.runtime_observations(limit=20)
                queue = job_manager.queue_state() if job_manager else {
                    "workers_configured": 0, "pending_limit": 0,
                    "active_futures": 0, "queued_jobs": 0, "running_jobs": 0,
                    "available_slots": 0, "accepting_jobs": False,
                }
                worker_metadata = getattr(captioner, "worker_metadata", None) or {}
                scheduler_metadata = getattr(captioner, "scheduler_metadata", None)
                latest = observations[0] if observations else None
                latest_metrics = latest["inference_metrics"] if latest else {}
                recent = [{
                    "media_title": item["media_title"], "stage": "captioning",
                    "backend_id": item["backend_id"], "model_id": item["model_id"],
                    "caption_status": item["caption_status"],
                    **item["inference_metrics"],
                } for item in observations]
                reported_devices = worker_metadata.get("runtime_devices", [])
                if not isinstance(reported_devices, list):
                    reported_devices = []
                devices = []
                for device in reported_devices:
                    if isinstance(device, str) and device:
                        devices.append({"device_id": device})
                    elif isinstance(device, dict) and isinstance(device.get("device_id"), str):
                        device_record = {"device_id": device["device_id"]}
                        for name in ("backend_id", "runtime_version", "plugin_version",
                                     "estimated_resident_bytes"):
                            value = device.get(name)
                            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                                device_record[name] = value
                        devices.append(device_record)
                reported_device_id = worker_metadata.get("device_id")
                if isinstance(reported_device_id, str) and reported_device_id and not devices:
                    devices.append({"device_id": reported_device_id})
                latest_inference = dict(latest_metrics)
                if latest:
                    latest_inference.update({
                        "backend_id": latest["backend_id"],
                        "model_id": latest["model_id"],
                    })
                return self._write_json({
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "queue": queue,
                    "worker": {
                        "estimated_resident_bytes": worker_metadata.get("estimated_resident_bytes"),
                        "worker_count_ready": worker_metadata.get("worker_count_ready"),
                    },
                    "scheduler": scheduler_metadata,
                    "caption_model": getattr(captioner, "model_id", "") if captioner else "",
                    "caption_backend": getattr(captioner, "backend_id", "") if captioner else "",
                    "embedder_model": getattr(embedder, "model_id", "") if embedder else "",
                    "answer_model": getattr(answer_provider, "model_id", "") if answer_provider else "",
                    "runtime_devices": devices,
                    "latest_inference": latest_inference,
                    "recent_inferences": recent,
                })
            if parsed.path.startswith("/api/frame/"):
                event_id = parsed.path.removeprefix("/api/frame/")
                event = store.event(event_id)
                if event is None:
                    return self._write_json({"error": "not found"}, 404)
                frame = Path(event["frame_path"]).resolve()
                try:
                    frame.relative_to(root)
                except ValueError:
                    return self._write_json({"error": "evidence path outside configured root"}, 403)
                if not frame.is_file():
                    return self._write_json({"error": "evidence file missing"}, 404)
                body = frame.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(frame.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=60")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/api/video/"):
                media_id = parsed.path.removeprefix("/api/video/")
                source, status = video_path(media_id, verify_hash=True)
                if source is None:
                    message = "video changed since indexing" if status == 409 else "video unavailable"
                    return self._write_json({"error": message}, status)
                try:
                    stream = source.open("rb")
                except OSError:
                    return self._write_json({"error": "video unavailable"}, 404)
                with stream:
                    size = os.fstat(stream.fileno()).st_size
                    requested_range = self.headers.get("Range")
                    byte_range = _byte_range(requested_range, size) if requested_range else None
                    if requested_range and byte_range is None:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if byte_range is None:
                        start, end = 0, size - 1
                        response_status = 200
                    else:
                        start, end = byte_range
                        response_status = 206
                    length = max(0, end - start + 1)
                    stream.seek(start)
                    self.send_response(response_status)
                    self.send_header("Content-Type", mimetypes.guess_type(source.name)[0]
                                     or "application/octet-stream")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    if response_status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Cache-Control", "private, no-store")
                    self.end_headers()
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self._write_json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            answer_started_ns = time.perf_counter_ns() if path == "/api/answer" else None
            if path == "/api/videos":
                if job_manager is None or allowed_video_root is None:
                    return self._write_json({"error": "video ingestion is not configured"}, 503)
                host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
                origin = self.headers.get("Origin")
                origin_host = urlparse(origin).hostname.lower() if origin else None
                if host not in {"localhost", "127.0.0.1", "::1"} or (
                        origin_host is not None and origin_host not in {"localhost", "127.0.0.1", "::1"}):
                    return self._write_json({"error": "upload is allowed only from the local UI"}, 403)
                temporary = final = None
                try:
                    raw_length = self.headers.get("Content-Length", "0")
                    if not raw_length.isascii() or not raw_length.isdigit():
                        raise ValueError("invalid Content-Length")
                    length = int(raw_length)
                    if length < 1 or length > _MAX_UPLOAD_BYTES:
                        raise ValueError("video size must be between 1 byte and 4 GiB")
                    name = unquote(self.headers.get("X-File-Name", ""), errors="strict")
                    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
                    suffix = Path(name).suffix.lower()
                    if not name or suffix not in _VIDEO_EXTENSIONS:
                        raise ValueError("supported video extensions: mp4, mov, mkv, webm, avi")
                    title = "".join(char for char in name if ord(char) >= 32 and ord(char) != 127)[:180]
                    if not title:
                        raise ValueError("video filename is empty")
                    allowed_video_root.mkdir(parents=True, exist_ok=True)
                    free_bytes = shutil.disk_usage(allowed_video_root).free
                    if free_bytes < length + _UPLOAD_RESERVE_BYTES:
                        return self._write_json({"error": "insufficient D: workspace space for this upload"}, 507)
                    token = uuid.uuid4().hex
                    temporary = allowed_video_root / f"upload-{token}.part"
                    final = allowed_video_root / f"video-{token}{suffix}"
                    remaining = length
                    with temporary.open("xb") as output:
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise OSError("upload ended before Content-Length bytes arrived")
                            output.write(chunk)
                            remaining -= len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if temporary.stat().st_size != length:
                        raise OSError("uploaded file size does not match Content-Length")
                    temporary.replace(final)
                    job = job_manager.submit(final, title)
                except IngestQueueFull as error:
                    if temporary and temporary.exists():
                        temporary.unlink()
                    if final and final.exists():
                        final.unlink()
                    return self._write_json({"error": str(error)}, 429)
                except (ValueError, OSError, UnicodeError) as error:
                    if temporary and temporary.exists():
                        temporary.unlink()
                    if final and final.exists():
                        final.unlink()
                    return self._write_json({"error": str(error)}, 400)
                return self._write_json({"job": job}, 202)

            if path not in {"/api/query", "/api/answer"}:
                return self._write_json({"error": "not found"}, 404)
            try:
                payload = self._read_json()
                query = payload.get("query", "")
                if not isinstance(query, str) or len(query) > 1024:
                    raise ValueError("query must be a string of at most 1024 characters")
                question = payload.get("question", query)
                if not isinstance(question, str) or len(question) > 1024:
                    raise ValueError("question must be a string of at most 1024 characters")
                media_id = payload.get("media_id")
                if media_id is not None and (not isinstance(media_id, str) or len(media_id) > 128):
                    raise ValueError("media_id must be a short string")
                limit = int(payload.get("limit", 20))
                if not 1 <= limit <= 50:
                    raise ValueError("limit must be in 1..50")
                start_ms = payload.get("start_ms")
                end_ms = payload.get("end_ms")
                if start_ms is not None:
                    start_ms = int(start_ms)
                if end_ms is not None:
                    end_ms = int(end_ms)
                summary = payload.get("summary", False)
                if not isinstance(summary, bool):
                    raise ValueError("summary must be a boolean")
                if summary:
                    if path != "/api/answer":
                        raise ValueError("summary is only available through /api/answer")
                    if not media_id:
                        raise ValueError("summary requires a selected video")
                    hits, total_summary_events = summary_events(store, media_id, limit=32)
                    evidence_scope = {
                        "kind": "time_spaced_video_summary",
                        "sampled_event_count": len(hits),
                        "available_event_count": total_summary_events,
                    }
                else:
                    hits = search_events(store, query, embedder=embedder,
                                         start_ms=start_ms, end_ms=end_ms,
                                         media_id=media_id, limit=limit)
                    evidence_scope = {"kind": "query_results", "sampled_event_count": len(hits)}
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                return self._write_json({"error": str(error)}, 400)
            results = [hit_payload(hit) for hit in hits]
            if path == "/api/query":
                return self._write_json({"results": results})
            if answer_provider is None:
                return self._write_json({
                    "status": "not_configured", "claims": [],
                    "missing_information": ["尚未配置回答模型；当前只返回检索到的事件证据。"],
                    "evidence": results,
                    "evidence_scope": evidence_scope,
                })
            try:
                answer_question = (
                    f"按视频时间顺序概括以下按时间均匀抽样的 {len(hits)} 条事件记录。"
                    f"该视频有 {total_summary_events} 条非模拟已完成事件记录；不得推断未抽样时间段发生的内容，"
                    "只陈述证据支持的信息，并说明时间覆盖限制或无法确定的信息。"
                    if summary else question
                )
                answer = answer_provider.answer(answer_question, hits)
            except AnswerGenerationError as error:
                return self._write_json({
                    "status": "backend_error", "claims": [],
                    "missing_information": ["回答模型暂时不可用；事件证据仍已保留。"],
                    "failure_reason": str(error),
                    "evidence": results,
                    "evidence_scope": evidence_scope,
                }, 502)
            if answer_started_ns is not None:
                answer["request_elapsed_us"] = (time.perf_counter_ns() - answer_started_ns) // 1000
            return self._write_json({**answer, "evidence": results,
                                     "evidence_scope": evidence_scope})

    return Handler


def serve(store: EventStore, artifact_root: Path, *, embedder: EmbeddingProvider | None = None,
          video_root: Path | None = None,
          job_manager: IngestJobManager | None = None,
          captioner=None,
          answer_provider: AnswerProvider | None = None,
          host: str = "127.0.0.1", port: int = 8765) -> None:
    if (video_root is not None or job_manager is not None) and host not in ("127.0.0.1", "localhost"):
        raise ValueError("video uploads and playback may only bind to the local loopback interface")
    server = ThreadingHTTPServer(
        (host, port), make_handler(
            store, artifact_root, embedder, video_root=video_root,
            job_manager=job_manager, captioner=captioner,
            answer_provider=answer_provider))
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
