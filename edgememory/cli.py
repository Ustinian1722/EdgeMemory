from __future__ import annotations

import argparse
from dataclasses import replace
import getpass
import json
import os
import sys
import threading
from pathlib import Path

from .answer import GenieXVlmAnswerProvider, OpenAICompatibleAnswerProvider
from .captions import MockCaptioner, OpenAICompatibleVlmCaptioner
from .embeddings import (DeterministicTestEmbedder, LlamaCppEmbeddingProvider,
                         OpenAICompatibleEmbeddingProvider)
from .jobs import IngestJobManager
from .pipeline import EdgeMemoryPipeline
from .qdc_geniex import QdcGenieXVlmCaptioner, load_qdc_geniex_profile
from .query import search_events
from .scheduling import schedule_qdc_captioner
from .store import EventStore
from .types import SamplingPolicy
from .web import serve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / ".local" / "edgememory" / "index.sqlite3"
DEFAULT_ARTIFACTS = ROOT / ".local" / "edgememory" / "media"
DEFAULT_UPLOADS = ROOT / ".local" / "edgememory" / "uploads"


def _workspace_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("EdgeMemory data and outputs must remain inside the D: project workspace") from error
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgememory")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create or verify a local event index")
    init.add_argument("--db", type=Path, default=DEFAULT_DB)

    ingest = sub.add_parser("ingest", help="sample a video and index evidence-backed captions")
    ingest.add_argument("video", type=Path)
    ingest.add_argument("--db", type=Path, default=DEFAULT_DB)
    ingest.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ingest.add_argument("--title")
    ingest.add_argument("--backend", choices=("openai-compatible", "qdc-geniex", "mock"), required=True)
    ingest.add_argument("--base-url", default="http://127.0.0.1:18181/v1")
    ingest.add_argument("--model", default="")
    ingest.add_argument("--api-key-env")
    ingest.add_argument("--qdc-profile", type=Path,
                        default=ROOT / ".local" / "edgememory" / "qdc-profile.json")
    ingest.add_argument("--qdc-password-env",
                        help="read the QDC device password from this environment variable")
    ingest.add_argument("--embedding-backend", choices=("none", "openai-compatible", "llama.cpp", "simulation"), default="none")
    ingest.add_argument("--embedding-url", default="http://127.0.0.1:18182/v1")
    ingest.add_argument("--embedding-model", default="")
    ingest.add_argument("--embedding-executable", default="")
    ingest.add_argument("--embedding-model-path", default="")
    ingest.add_argument("--min-interval-s", type=float, default=1.5)
    ingest.add_argument("--max-interval-s", type=float, default=12.0)
    ingest.add_argument("--scene-threshold", type=float, default=0.16)
    ingest.add_argument("--max-samples", type=int, default=500)

    query = sub.add_parser("query", help="search captions and return frame/time evidence")
    query.add_argument("text")
    query.add_argument("--db", type=Path, default=DEFAULT_DB)
    query.add_argument("--embedding-backend", choices=("none", "openai-compatible", "llama.cpp", "simulation"), default="none")
    query.add_argument("--embedding-url", default="http://127.0.0.1:18182/v1")
    query.add_argument("--embedding-model", default="")
    query.add_argument("--embedding-executable", default="")
    query.add_argument("--embedding-model-path", default="")
    query.add_argument("--start-ms", type=int)
    query.add_argument("--end-ms", type=int)
    query.add_argument("--limit", type=int, default=20)

    web = sub.add_parser("serve", help="start the local evidence search UI")
    web.add_argument("--db", type=Path, default=DEFAULT_DB)
    web.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    web.add_argument("--video-root", type=Path, default=DEFAULT_UPLOADS,
                     help="D-drive workspace directory for uploaded videos and allowlisted playback")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--embedding-backend", choices=("none", "openai-compatible", "llama.cpp", "simulation"), default="none")
    web.add_argument("--embedding-url", default="http://127.0.0.1:18182/v1")
    web.add_argument("--embedding-model", default="")
    web.add_argument("--embedding-executable", default="")
    web.add_argument("--embedding-model-path", default="")
    web.add_argument("--caption-backend", choices=("none", "openai-compatible", "qdc-geniex", "mock"), default="none")
    web.add_argument("--caption-url", default="http://127.0.0.1:18181/v1")
    web.add_argument("--caption-model", default="")
    web.add_argument("--caption-api-key-env")
    web.add_argument("--qdc-profile", type=Path,
                     default=ROOT / ".local" / "edgememory" / "qdc-profile.json")
    web.add_argument("--qdc-password-env")
    web.add_argument("--answer-backend", choices=("none", "openai-compatible", "qdc-geniex"), default="none")
    web.add_argument("--answer-url", default="http://127.0.0.1:18182/v1")
    web.add_argument("--answer-model", default="")
    web.add_argument("--answer-api-key-env")
    web.add_argument("--workers", type=int, default=2)
    web.add_argument("--pending-limit", type=int, default=6)
    return parser


def _make_embedder(args):
    mode = getattr(args, "embedding_backend", "none")
    if mode == "none":
        return None
    if mode == "simulation":
        return DeterministicTestEmbedder()
    if mode == "openai-compatible":
        if not args.embedding_model:
            raise ValueError("--embedding-model is required for openai-compatible embeddings")
        return OpenAICompatibleEmbeddingProvider(args.embedding_url, args.embedding_model)
    if mode == "llama.cpp":
        if not args.embedding_executable or not args.embedding_model_path:
            raise ValueError("llama.cpp embedding requires executable and model paths")
        return LlamaCppEmbeddingProvider(args.embedding_executable, args.embedding_model_path)
    raise ValueError("unknown embedding backend")


class _RuntimePasswordProvider:
    """Keep a session password in process memory only, shared by worker connections."""

    def __init__(self, environment_name: str | None):
        self.environment_name = environment_name
        self._cached = ""
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            if self.environment_name:
                password = os.environ.get(self.environment_name, "")
                if not password:
                    raise ValueError("the named QDC password environment variable is empty")
                return password
            if not self._cached:
                if not sys.stdin.isatty():
                    raise ValueError("interactive stdin or --qdc-password-env is required")
                self._cached = getpass.getpass("QDC device password: ")
            return self._cached

    def clear(self) -> None:
        with self._lock:
            self._cached = ""


def _make_qdc_captioner(args):
    profile = load_qdc_geniex_profile(_workspace_path(args.qdc_profile))
    password_provider = _RuntimePasswordProvider(args.qdc_password_env)
    backends = []
    try:
        for worker in profile.worker_resources:
            worker_profile = replace(profile, device_id=worker.device_id)
            backends.append(QdcGenieXVlmCaptioner(
                worker_profile, password_provider=password_provider))
        captioner = schedule_qdc_captioner(backends, profile)
        captioner.password_provider = password_provider
        return captioner
    except Exception:
        for backend in backends:
            backend.close()
        password_provider.clear()
        raise


def _make_captioner(args, prefix: str = "caption"):
    mode = getattr(args, f"{prefix}_backend", "none")
    if mode == "none":
        return None
    if mode == "mock":
        return MockCaptioner()
    if mode == "openai-compatible":
        model = getattr(args, f"{prefix}_model", "")
        if not model:
            raise ValueError(f"--{prefix}-model is required for openai-compatible captions")
        return OpenAICompatibleVlmCaptioner(
            getattr(args, f"{prefix}_url"), model,
            api_key_env=getattr(args, f"{prefix}_api_key_env", None))
    if mode == "qdc-geniex":
        return _make_qdc_captioner(args)
    raise ValueError(f"unknown {prefix} caption backend")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            db_path = _workspace_path(args.db)
            store = EventStore(db_path)
            try:
                print(json.dumps({"status": "ready", "database": str(db_path),
                                  "media_count": len(store.media())}, ensure_ascii=False))
            finally:
                store.close()
            return 0

        if args.command == "ingest":
            if args.backend == "mock":
                captioner = MockCaptioner()
            elif args.backend == "qdc-geniex":
                captioner = _make_qdc_captioner(args)
            else:
                if not args.model:
                    raise ValueError("--model is required for a real VLM backend")
                captioner = OpenAICompatibleVlmCaptioner(
                    args.base_url, args.model, api_key_env=args.api_key_env)
            embedder = _make_embedder(args)
            db_path = _workspace_path(args.db)
            artifacts = _workspace_path(args.artifacts)
            store = EventStore(db_path)
            try:
                summary = EdgeMemoryPipeline(store, captioner, embedder).ingest_video(
                    args.video, artifacts,
                    policy=SamplingPolicy(
                        min_interval_s=args.min_interval_s,
                        max_interval_s=args.max_interval_s,
                        scene_change_threshold=args.scene_threshold,
                        max_samples=args.max_samples),
                    title=args.title)
                print(json.dumps({
                    "media_id": summary.media_id,
                    "source_sha256": summary.source_sha256,
                    "sampled_frames": summary.sampled_frames,
                    "indexed_events": summary.indexed_events,
                    "failed_events": summary.failed_events,
                    "simulation_only": summary.simulation_only,
                    "run_dir": str(summary.run_dir),
                    "frame_evidence_root": str(summary.artifact_dir),
                }, ensure_ascii=False, indent=2))
            finally:
                store.close()
                close_captioner = getattr(captioner, "close", None)
                if callable(close_captioner):
                    close_captioner()
                clear_password = getattr(getattr(captioner, "password_provider", None), "clear", None)
                if callable(clear_password):
                    clear_password()
            return 0

        embedder = _make_embedder(args)
        db_path = _workspace_path(args.db)
        store = EventStore(db_path)
        if args.command == "query":
            try:
                hits = search_events(store, args.text, embedder=embedder,
                                    start_ms=args.start_ms, end_ms=args.end_ms,
                                    limit=args.limit)
                print(json.dumps([hit.__dict__ for hit in hits], ensure_ascii=False, indent=2))
            finally:
                store.close()
            return 0
        if args.command == "serve":
            print(f"EdgeMemory UI listening on http://{args.host}:{args.port}")
            video_root = _workspace_path(args.video_root) if args.video_root else None
            artifacts = _workspace_path(args.artifacts)
            captioner = _make_captioner(args)
            answer_captioner = None
            job_manager = None
            try:
                answer_provider = None
                if args.answer_backend == "openai-compatible":
                    if not args.answer_model:
                        raise ValueError("--answer-model is required for openai-compatible answers")
                    answer_provider = OpenAICompatibleAnswerProvider(
                        args.answer_url, args.answer_model,
                        api_key_env=args.answer_api_key_env)
                elif args.answer_backend == "qdc-geniex":
                    if args.caption_backend == "qdc-geniex":
                        answer_captioner = captioner
                    else:
                        answer_captioner = _make_qdc_captioner(args)
                    answer_provider = GenieXVlmAnswerProvider(answer_captioner)
                if captioner is not None:
                    if video_root is None:
                        raise ValueError("--video-root is required when a caption backend is configured")
                    job_manager = IngestJobManager(
                        store, EdgeMemoryPipeline(store, captioner, embedder),
                        artifacts, video_root, workers=args.workers,
                        pending_limit=args.pending_limit)
                serve(store, artifacts, embedder=embedder,
                      video_root=video_root, job_manager=job_manager,
                      captioner=captioner, answer_provider=answer_provider,
                      host=args.host, port=args.port)
            finally:
                if job_manager is not None:
                    job_manager.close(wait=True)
                if answer_captioner is not None and answer_captioner is not captioner:
                    close_answer_captioner = getattr(answer_captioner, "close", None)
                    if callable(close_answer_captioner):
                        close_answer_captioner()
                    clear_answer_password = getattr(
                        getattr(answer_captioner, "password_provider", None), "clear", None)
                    if callable(clear_answer_password):
                        clear_answer_password()
                close_captioner = getattr(captioner, "close", None)
                if callable(close_captioner):
                    close_captioner()
                clear_password = getattr(getattr(captioner, "password_provider", None), "clear", None)
                if callable(clear_password):
                    clear_password()
                store.close()
            return 0
        raise ValueError("unknown command")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"edgememory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
