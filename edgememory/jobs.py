from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import EdgeMemoryPipeline
from .store import EventStore
from .types import SamplingPolicy


class IngestQueueFull(RuntimeError):
    pass


class IngestJobManager:
    """Bounded background ingestion with durable progress records in SQLite."""

    def __init__(self, store: EventStore, pipeline: EdgeMemoryPipeline,
                 artifact_root: Path, video_root: Path, *,
                 policy: SamplingPolicy | None = None, workers: int = 2,
                 pending_limit: int = 8):
        if workers < 1 or pending_limit < 0:
            raise ValueError("workers must be positive and pending_limit non-negative")
        self.store = store
        self.pipeline = pipeline
        self.artifact_root = artifact_root.expanduser().resolve()
        self.video_root = video_root.expanduser().resolve()
        self.policy = policy or SamplingPolicy()
        self.workers = workers
        self.pending_limit = pending_limit
        self._slots = threading.BoundedSemaphore(workers + pending_limit)
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="edgememory-ingest")
        # A Future may finish before add_done_callback(); Python then runs the
        # callback inline, so the callback must be able to re-enter this lock.
        self._lock = threading.RLock()
        self._futures: dict[str, Future] = {}
        self._closed = False
        self._restore_interrupted_jobs()

    def _restore_interrupted_jobs(self) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for job in self.store.recoverable_ingest_jobs(limit=1000):
            try:
                source = Path(job["source_path"]).expanduser().resolve(strict=True)
                source.relative_to(self.video_root)
                if not source.is_file():
                    raise ValueError("source_not_file")
            except (OSError, RuntimeError, ValueError):
                self.store.update_ingest_job(
                    job["job_id"], status="failed", stage="interrupted",
                    error_code="source_unavailable_after_restart", finished_at_utc=now)
                continue
            if not self._slots.acquire(blocking=False):
                self.store.update_ingest_job(
                    job["job_id"], status="failed", stage="interrupted",
                    error_code="restart_queue_capacity_exceeded", finished_at_utc=now)
                continue
            try:
                self.store.update_ingest_job(
                    job["job_id"], status="queued", stage="restoring",
                    progress=0.0, completed_units=0, total_units=0,
                    error_code="", started_at_utc="", finished_at_utc="")
                self._schedule(job["job_id"], source, job["title"])
            except BaseException:
                self._slots.release()
                self.store.update_ingest_job(
                    job["job_id"], status="failed", stage="interrupted",
                    error_code="recovery_submit_failed", finished_at_utc=now)

    def _schedule(self, job_id: str, source_path: Path, title: str) -> Future:
        future = self._executor.submit(self._run, job_id, source_path, title)
        self._futures[job_id] = future
        future.add_done_callback(lambda _future, key=job_id: self._discard(key))
        return future

    def submit(self, source_path: Path, title: str) -> dict:
        resolved = source_path.expanduser().resolve(strict=True)
        try:
            resolved.relative_to(self.video_root)
        except ValueError as error:
            raise ValueError("uploaded video is outside the configured media root") from error
        if not resolved.is_file():
            raise ValueError("uploaded video is not a regular file")
        with self._lock:
            if self._closed:
                raise RuntimeError("ingest service is shutting down")
            if not self._slots.acquire(blocking=False):
                raise IngestQueueFull("ingest queue is full; wait for a job to finish")
            job_id = uuid.uuid4().hex
            try:
                job = self.store.create_ingest_job(job_id, title, resolved)
                self._schedule(job_id, resolved, title)
                return job
            except BaseException:
                self._slots.release()
                raise

    def _discard(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _progress(self, job_id: str, stage: str, completed: int, total: int) -> None:
        if stage == "sampling":
            fraction = completed / total if total > 0 else 0.0
            percent = min(45.0, 45.0 * fraction)
        elif stage == "captioning":
            fraction = completed / total if total > 0 else 0.0
            percent = 45.0 + min(50.0, 50.0 * fraction)
        elif stage == "completed":
            percent = 100.0
        else:
            percent = 0.0
        self.store.update_ingest_job(
            job_id, stage=stage, progress=percent,
            completed_units=max(0, int(completed)), total_units=max(0, int(total)))

    def _run(self, job_id: str, source_path: Path, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.update_ingest_job(
            job_id, status="running", stage="sampling", progress=0.0,
            completed_units=0, total_units=0,
            started_at_utc=now)
        try:
            result = self.pipeline.ingest_video(
                source_path,
                self.artifact_root,
                policy=self.policy,
                title=title,
                run_dir=self.artifact_root / "runs" / job_id,
                progress_callback=lambda stage, completed, total: self._progress(
                    job_id, stage, completed, total),
            )
            finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
            status = ("partial" if result.failed_events and result.indexed_events else
                      "failed" if result.failed_events else "completed")
            self.store.update_ingest_job(
                job_id, media_id=result.media_id, status=status,
                stage="indexed" if status != "failed" else "captioning_failed",
                progress=100.0, completed_units=result.sampled_frames,
                total_units=result.sampled_frames,
                error_code="all_events_failed" if status == "failed" else "",
                finished_at_utc=finished)
        except Exception as error:
            finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.store.update_ingest_job(
                job_id, status="failed", stage="failed", progress=100.0,
                error_code=type(error).__name__, finished_at_utc=finished)
        finally:
            self._slots.release()

    def get(self, job_id: str) -> dict | None:
        return self.store.ingest_job(job_id)

    def list(self, limit: int = 50) -> list[dict]:
        return self.store.ingest_jobs(limit)

    def queue_state(self) -> dict[str, int | bool]:
        with self._lock:
            active = len(self._futures)
            closed = self._closed
        jobs = self.store.ingest_jobs(limit=200)
        queued = sum(job["status"] == "queued" for job in jobs)
        running = sum(job["status"] == "running" for job in jobs)
        return {
            "workers_configured": self.workers,
            "pending_limit": self.pending_limit,
            "active_futures": active,
            "queued_jobs": queued,
            "running_jobs": running,
            "available_slots": max(0, self.workers + self.pending_limit - active),
            "accepting_jobs": not closed,
        }

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
