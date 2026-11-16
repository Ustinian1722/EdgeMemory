from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

from .captions import CAPTION_PROMPT, Captioner
from .embeddings import EmbeddingProvider
from .sampling import extract_adaptive_frames
from .store import EventStore
from .types import IngestSummary, SamplingPolicy


class EdgeMemoryPipeline:
    def __init__(self, store: EventStore, captioner: Captioner,
                 embedder: EmbeddingProvider | None = None):
        self.store = store
        self.captioner = captioner
        self.embedder = embedder
        caption_parallelism = getattr(captioner, "max_concurrent_requests", 1)
        if (isinstance(caption_parallelism, bool) or not isinstance(caption_parallelism, int)
                or not 1 <= caption_parallelism <= 64):
            raise ValueError("caption backend concurrency must be an integer in 1..64")
        self.caption_parallelism = caption_parallelism

    def ingest_video(self, video_path: Path, artifact_dir: Path, *,
                     policy: SamplingPolicy | None = None,
                     title: str | None = None,
                     run_dir: Path | None = None,
                     progress_callback: Callable[[str, int, int], None] | None = None
                     ) -> IngestSummary:
        policy = policy or SamplingPolicy()
        artifact_dir = artifact_dir.expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        def sampling_progress(completed: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback("sampling", completed, total)

        if progress_callback is not None:
            progress_callback("sampling", 0, 0)
        remote_extractor = getattr(self.captioner, "extract_video_frames", None)
        if callable(remote_extractor):
            metadata, samples = remote_extractor(
                video_path, artifact_dir, policy, progress_callback=sampling_progress)
        else:
            metadata, samples = extract_adaptive_frames(
                video_path, artifact_dir, policy, progress_callback=sampling_progress)
        if not samples:
            raise RuntimeError("video produced no evidence frames")
        if progress_callback is not None:
            progress_callback("captioning", 0, len(samples))
        if run_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = Path(__file__).resolve().parents[1] / "runs" / "edgememory" / f"ingest-{stamp}-{metadata.media_id[:8]}"
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.store.register_media(metadata, title or metadata.source_path.name)
        completed = 0
        failed = 0
        processed = 0
        frame_results: list[dict | None] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.caption_parallelism,
                                thread_name_prefix="edgememory-caption") as workers:
            try:
                describe_remote = getattr(self.captioner, "describe_remote", None)
                futures = {
                    workers.submit(
                        describe_remote, sample.remote_path, sample.frame_path, CAPTION_PROMPT
                    ) if sample.remote_path and callable(describe_remote) else workers.submit(
                        self.captioner.describe, sample.frame_path, CAPTION_PROMPT
                    ): (sample_number, sample)
                    for sample_number, sample in enumerate(samples)
                }
                for future in as_completed(futures):
                    sample_number, sample = futures[future]
                    event_id = hashlib.sha256((
                        f"{sample.media_id}:{sample.frame_index}:{sample.frame_sha256}:"
                        f"{self.captioner.model_id}"
                    ).encode("utf-8")).hexdigest()[:32]
                    try:
                        result = future.result()
                        caption = result.text.strip()
                        if not caption:
                            raise ValueError("caption backend returned empty text")
                        vector = self.embedder.embed(caption) if self.embedder else None
                        self.store.upsert_event(
                            event_id=event_id, media_id=sample.media_id,
                            frame_index=sample.frame_index, timestamp_ms=sample.timestamp_ms,
                            start_ms=sample.timestamp_ms, end_ms=sample.timestamp_ms,
                            frame_path=sample.frame_path, frame_sha256=sample.frame_sha256,
                            sample_reason=sample.sample_reason, scene_delta=sample.scene_delta,
                            caption=caption, caption_status="completed", backend_id=result.backend_id,
                            model_id=result.model_id, embedding=vector,
                            embedding_model_id=self.embedder.model_id if vector is not None else "",
                            simulation_only=result.simulation_only,
                            inference_metrics=result.inference_metrics,
                        )
                        completed += 1
                        frame_results[sample_number] = {
                            "frame_index": sample.frame_index,
                            "event_id": event_id,
                            "status": "completed",
                            "caption": caption,
                            "backend_id": result.backend_id,
                            "model_id": result.model_id,
                            "simulation_only": result.simulation_only,
                            "inference_metrics": result.inference_metrics,
                        }
                    except Exception as error:
                        self.store.upsert_event(
                            event_id=event_id, media_id=sample.media_id,
                            frame_index=sample.frame_index, timestamp_ms=sample.timestamp_ms,
                            start_ms=sample.timestamp_ms, end_ms=sample.timestamp_ms,
                            frame_path=sample.frame_path, frame_sha256=sample.frame_sha256,
                            sample_reason=sample.sample_reason, scene_delta=sample.scene_delta,
                            caption_status="failed", error_code=type(error).__name__,
                            backend_id=self.captioner.backend_id, model_id=self.captioner.model_id,
                            simulation_only=self.captioner.simulation_only,
                        )
                        failed += 1
                        frame_results[sample_number] = {
                            "frame_index": sample.frame_index,
                            "event_id": event_id,
                            "status": "failed",
                            "error_code": type(error).__name__,
                        }
                    processed += 1
                    if progress_callback is not None:
                        progress_callback("captioning", processed, len(samples))
            finally:
                cleanup_remote = getattr(self.captioner, "cleanup_remote_frames", None)
                if callable(cleanup_remote):
                    cleanup_remote(samples)
        self.store.set_event_bounds(metadata.media_id, metadata.duration_ms)
        summary = IngestSummary(
            media_id=metadata.media_id,
            source_sha256=metadata.source_sha256,
            sampled_frames=len(samples),
            indexed_events=completed,
            failed_events=failed,
            simulation_only=self.captioner.simulation_only,
            artifact_dir=artifact_dir,
            run_dir=run_dir,
            frame_samples=samples,
        )
        record = {
            "schema_version": 1,
            "media_id": summary.media_id,
            "source_sha256": summary.source_sha256,
            "caption_prompt_sha256": hashlib.sha256(CAPTION_PROMPT.encode("utf-8")).hexdigest(),
            "source_frame_count": metadata.source_frame_count,
            "duration_ms": metadata.duration_ms,
            "fps": metadata.fps,
            "dimensions": [metadata.width, metadata.height],
            "sampling_policy": policy.__dict__,
            "sampled_frames": summary.sampled_frames,
            "indexed_events": summary.indexed_events,
            "failed_events": summary.failed_events,
            "simulation_only": summary.simulation_only,
            "caption_parallelism": self.caption_parallelism,
            "caption_backend": self.captioner.backend_id,
            "caption_model": self.captioner.model_id,
            "caption_configuration": getattr(self.captioner, "run_configuration", None),
            "embedding_model": self.embedder.model_id if self.embedder else None,
            "caption_worker": getattr(self.captioner, "worker_metadata", None),
            "video_ingestion": getattr(self.captioner, "video_ingestion_metadata", None),
            "frame_results": [item for item in frame_results if item is not None],
            "frame_evidence": [{
                "frame_index": frame.frame_index,
                "timestamp_ms": frame.timestamp_ms,
                "frame_sha256": frame.frame_sha256,
                "sample_reason": frame.sample_reason,
                "scene_delta": frame.scene_delta,
            } for frame in samples],
        }
        record["artifact_root"] = str(artifact_dir)
        (run_dir / "ingest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from tools.run_manifest import create as create_run_manifest
        create_run_manifest(run_dir)
        if progress_callback is not None:
            progress_callback("completed", len(samples), len(samples))
        return summary
