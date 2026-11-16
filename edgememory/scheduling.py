from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from .types import CaptionResult


ROOT = Path(__file__).resolve().parents[1]


class _ResourceSpec(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char_p),
        ("backend_id", ctypes.c_char_p),
        ("interference_domain_id", ctypes.c_char_p),
        ("family", ctypes.c_int),
        ("max_concurrency", ctypes.c_uint32),
        ("memory_capacity_bytes", ctypes.c_uint64),
        ("physical_resource_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("physical_resource_count", ctypes.c_size_t),
        ("exclusive_group_id", ctypes.c_char_p),
    ]


class _ServiceEstimate(ctypes.Structure):
    _fields_ = [
        ("backend_id", ctypes.c_char_p),
        ("service_us", ctypes.c_uint64),
    ]


class _WorkloadSpec(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char_p),
        ("model_id", ctypes.c_char_p),
        ("kind", ctypes.c_int),
        ("priority", ctypes.c_int32),
        ("has_deadline", ctypes.c_int),
        ("absolute_deadline_us", ctypes.c_uint64),
        ("estimated_service_us", ctypes.c_uint64),
        ("model_memory_bytes", ctypes.c_uint64),
        ("requested_output_tokens", ctypes.c_uint64),
        ("input_units", ctypes.c_uint64),
        ("allowed_backend_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("allowed_backend_count", ctypes.c_size_t),
        ("service_estimates_by_backend", ctypes.POINTER(_ServiceEstimate)),
        ("service_estimate_count", ctypes.c_size_t),
    ]


class _DispatchDecision(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_char * 128),
        ("resource_id", ctypes.c_char * 128),
        ("reason", ctypes.c_char * 192),
        ("dispatched", ctypes.c_int),
    ]


class SchedulerUnavailable(RuntimeError):
    pass


_TRANSIENT_DISPATCH_BLOCKS = {
    "concurrency-capacity-exhausted",
    "exclusive-contention-group-busy",
    "verified-physical-resource-overlap",
}


def _library_candidates() -> list[Path]:
    configured = os.environ.get("HEXAGONFLOW_SCHEDULER_LIB", "").strip()
    if configured:
        return [Path(configured).expanduser()]
    if sys.platform == "win32":
        return [
            ROOT / "build" / "windows-msvc" / "RelWithDebInfo" / "hexagonflow_scheduler_c.dll",
            ROOT / "build" / "windows-msvc" / "Release" / "hexagonflow_scheduler_c.dll",
            ROOT / "build" / "scheduler" / "Release" / "hexagonflow_scheduler_c.dll",
            ROOT / "build" / "scheduler" / "RelWithDebInfo" / "hexagonflow_scheduler_c.dll",
            ROOT / "build" / "scheduler" / "hexagonflow_scheduler_c.dll",
        ]
    if sys.platform == "darwin":
        return [ROOT / "build" / "scheduler" / "libhexagonflow_scheduler_c.dylib"]
    return [
        ROOT / "build" / "scheduler" / "libhexagonflow_scheduler_c.so",
        ROOT / "build" / "wsl-gcc" / "libhexagonflow_scheduler_c.so",
    ]


def _load_library() -> ctypes.CDLL:
    candidates = _library_candidates()
    library_path = next((path for path in candidates if path.is_file()), None)
    if library_path is None:
        expected = str(candidates[0]) if candidates else ""
        raise SchedulerUnavailable(
            "the native HexagonFlow scheduler library is not built; "
            f"build target hexagonflow_scheduler_c (expected {expected})"
        )
    library = ctypes.CDLL(str(library_path))
    library.hf_scheduler_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(_ResourceSpec), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    library.hf_scheduler_create.restype = ctypes.c_void_p
    library.hf_scheduler_destroy.argtypes = [ctypes.c_void_p]
    library.hf_scheduler_destroy.restype = None
    library.hf_scheduler_submit.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_WorkloadSpec),
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    library.hf_scheduler_submit.restype = ctypes.c_int
    library.hf_scheduler_dispatch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_DispatchDecision), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    library.hf_scheduler_dispatch.restype = ctypes.c_ssize_t
    library.hf_scheduler_complete.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.hf_scheduler_complete.restype = ctypes.c_int
    library.hf_scheduler_fail.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    library.hf_scheduler_fail.restype = ctypes.c_int
    library.hf_scheduler_cancel_queued.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.hf_scheduler_cancel_queued.restype = ctypes.c_int
    library.hf_scheduler_load_interference_tsv.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    library.hf_scheduler_load_interference_tsv.restype = ctypes.c_int
    return library


class NativeCaptionScheduler:
    """Thread-safe Python boundary around the project's C++ task scheduler."""

    _POLICIES = {
        "fifo": 0,
        "priority": 1,
        "edf": 2,
        "predicted_latency": 3,
        "interference_aware_latency": 4,
    }

    def __init__(self, resource_id: str | None = None, backend_id: str | None = None, *,
                 resources: list[dict] | None = None, policy: str = "fifo",
                 service_estimates_by_backend_us: dict[str, int] | None = None,
                 interference_profile_path: Path | None = None,
                 interference_profile_sha256: str = "",
                 wait_timeout_s: float = 600.0):
        if resources is None:
            if not resource_id or not backend_id:
                raise ValueError("scheduler resource and backend IDs are required")
            resources = [{"resource_id": resource_id, "backend_id": backend_id}]
        if not 1 <= len(resources) <= 64:
            raise ValueError("scheduler requires 1..64 explicitly configured resources")
        if policy not in self._POLICIES:
            raise ValueError("unsupported native scheduler policy")
        if not 1.0 <= wait_timeout_s <= 3600.0:
            raise ValueError("scheduler wait timeout must be in 1..3600 seconds")
        self.policy = policy
        self.wait_timeout_s = wait_timeout_s
        self.resources: list[dict[str, str | int]] = []
        seen_resources: set[str] = set()
        seen_backends: set[str] = set()
        for item in resources:
            if not isinstance(item, dict):
                raise ValueError("scheduler resources must be objects")
            rid, bid = item.get("resource_id"), item.get("backend_id")
            if not isinstance(rid, str) or not rid.strip() or len(rid.encode("utf-8")) >= 128:
                raise ValueError("scheduler resource IDs must be non-empty and shorter than 128 bytes")
            if not isinstance(bid, str) or not bid.strip() or len(bid.encode("utf-8")) >= 128:
                raise ValueError("scheduler backend IDs must be non-empty and shorter than 128 bytes")
            if rid in seen_resources or bid in seen_backends:
                raise ValueError("scheduler resource and backend IDs must be unique")
            seen_resources.add(rid)
            seen_backends.add(bid)
            concurrency = item.get("max_concurrency", 1)
            memory_capacity = item.get("memory_capacity_bytes", 0)
            family = item.get("family", 1)
            domain = item.get("interference_domain_id", "")
            physical_ids = item.get("physical_resource_ids", ())
            exclusive_group = item.get("exclusive_group_id", "")
            if isinstance(concurrency, bool) or not isinstance(concurrency, int) or not 1 <= concurrency <= 256:
                raise ValueError("resource max_concurrency must be in 1..256")
            if isinstance(memory_capacity, bool) or not isinstance(memory_capacity, int) or memory_capacity < 0:
                raise ValueError("resource memory capacity must be a non-negative integer")
            if isinstance(family, bool) or family not in range(6):
                raise ValueError("resource family is invalid")
            if not isinstance(domain, str) or "\x00" in domain or len(domain.encode("utf-8")) >= 128:
                raise ValueError("resource interference domain ID is invalid")
            if (not isinstance(physical_ids, (list, tuple)) or len(physical_ids) > 64 or
                    any(not isinstance(value, str) or not value.strip() or "\x00" in value or
                        len(value.encode("utf-8")) >= 128 for value in physical_ids) or
                    len(set(physical_ids)) != len(physical_ids)):
                raise ValueError("resource physical IDs must be unique non-empty strings")
            if (not isinstance(exclusive_group, str) or "\x00" in exclusive_group or
                    len(exclusive_group.encode("utf-8")) >= 128):
                raise ValueError("resource exclusive group ID is invalid")
            self.resources.append({
                "resource_id": rid, "backend_id": bid, "max_concurrency": concurrency,
                "memory_capacity_bytes": memory_capacity, "family": family,
                "interference_domain_id": domain,
                "physical_resource_ids": tuple(physical_ids),
                "exclusive_group_id": exclusive_group,
            })
        self.resource_id = self.resources[0]["resource_id"] if len(self.resources) == 1 else ""
        self.backend_id = self.resources[0]["backend_id"] if len(self.resources) == 1 else ""
        self.backend_ids = tuple(str(item["backend_id"]) for item in self.resources)
        self.service_estimates_by_backend_us = dict(service_estimates_by_backend_us or {})
        for key, value in self.service_estimates_by_backend_us.items():
            if key not in self.backend_ids or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("service estimates must be positive and match configured backend IDs")
        if policy == "interference_aware_latency" and any(
                backend not in self.service_estimates_by_backend_us for backend in self.backend_ids):
            raise ValueError("interference-aware policy requires a service estimate for every resource")
        self.interference_profile_path: Path | None = None
        self.interference_profile_sha256 = ""
        if policy == "interference_aware_latency":
            if interference_profile_path is None or len(interference_profile_sha256) != 64:
                raise ValueError("interference-aware policy requires a TSV path and expected SHA256")
            profile_path = interference_profile_path.expanduser().resolve(strict=True)
            try:
                profile_path.relative_to(ROOT.resolve())
            except ValueError as error:
                raise ValueError("interference profiles must stay under D:\\A-IQ-9075") from error
            digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            if digest.casefold() != interference_profile_sha256.casefold():
                raise ValueError("interference profile SHA256 does not match the local profile")
            self.interference_profile_path = profile_path
            self.interference_profile_sha256 = digest
        elif interference_profile_path is not None:
            raise ValueError("an interference TSV is only used with the interference-aware policy")
        self._library = _load_library()
        self._condition = threading.Condition(threading.RLock())
        self._assigned: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._decision_reasons: dict[str, str] = {}
        self._decision_history: list[dict[str, str | int | bool]] = []
        self._thread_decisions: dict[int, dict[str, str | int | bool]] = {}
        self._closed = False
        physical_arrays = [
            (ctypes.c_char_p * len(item["physical_resource_ids"]))(*[
                value.encode("utf-8") for value in item["physical_resource_ids"]
            ]) if item["physical_resource_ids"] else None
            for item in self.resources
        ]
        resource_array = (_ResourceSpec * len(self.resources))(*[
            _ResourceSpec(
                str(item["resource_id"]).encode("utf-8"),
                str(item["backend_id"]).encode("utf-8"),
                str(item["interference_domain_id"]).encode("utf-8"),
                int(item["family"]), int(item["max_concurrency"]),
                int(item["memory_capacity_bytes"]),
                physical_arrays[index], len(item["physical_resource_ids"]),
                str(item["exclusive_group_id"]).encode("utf-8"))
            for index, item in enumerate(self.resources)
        ])
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.hf_scheduler_create(
            self._POLICIES[policy], resource_array, len(resource_array), error, len(error))
        if not self._handle:
            raise RuntimeError(self._decode_error(error) or "native scheduler initialization failed")
        if self.interference_profile_path is not None:
            loaded = self._library.hf_scheduler_load_interference_tsv(
                self._handle, str(self.interference_profile_path).encode("utf-8"), error, len(error))
            if loaded <= 0:
                self._library.hf_scheduler_destroy(self._handle)
                self._handle = None
                raise RuntimeError(self._decode_error(error) or "interference profile could not be loaded")

    @staticmethod
    def _decode_error(buffer) -> str:
        return bytes(buffer).split(b"\0", 1)[0].decode("utf-8", errors="replace")

    def observe_service_time(self, backend_id: str, observed_us: int,
                             *, alpha: float = 0.2) -> int:
        if backend_id not in self.backend_ids:
            raise ValueError("observation backend is not registered with this scheduler")
        if isinstance(observed_us, bool) or not isinstance(observed_us, int) or observed_us <= 0:
            raise ValueError("observed service time must be positive microseconds")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        previous = self.service_estimates_by_backend_us.get(backend_id)
        updated = observed_us if previous is None else int(
            (1.0 - alpha) * previous + alpha * observed_us)
        self.service_estimates_by_backend_us[backend_id] = max(1, updated)
        return self.service_estimates_by_backend_us[backend_id]

    def _dispatch_locked(self) -> None:
        decisions = (_DispatchDecision * 256)()
        error = ctypes.create_string_buffer(512)
        count = self._library.hf_scheduler_dispatch(
            self._handle, decisions, len(decisions), error, len(error))
        if count < 0:
            raise RuntimeError(self._decode_error(error) or "native scheduler dispatch failed")
        for decision in decisions[:count]:
            task_id = bytes(decision.task_id).split(b"\0", 1)[0].decode("utf-8")
            resource_id = bytes(decision.resource_id).split(b"\0", 1)[0].decode("utf-8")
            reason = bytes(decision.reason).split(b"\0", 1)[0].decode("utf-8")
            record = {
                "task_id": task_id, "resource_id": resource_id, "reason": reason,
                "dispatched": bool(decision.dispatched), "recorded_ns": time.time_ns(),
            }
            if decision.dispatched and reason in _TRANSIENT_DISPATCH_BLOCKS:
                record["candidate_blocker_reason"] = reason
                record["reason"] = "dispatched-after-alternative-blocked"
            self._decision_history.append(record)
            if len(self._decision_history) > 256:
                del self._decision_history[:-256]
            if decision.dispatched:
                self._assigned[task_id] = resource_id
                self._decision_reasons[task_id] = str(record["reason"])
            elif reason not in _TRANSIENT_DISPATCH_BLOCKS:
                self._errors[task_id] = reason or "scheduler-rejected-task"
        self._condition.notify_all()

    @contextmanager
    def reserve(self, model_id: str, *, allowed_backend_ids: tuple[str, ...] | None = None,
                priority: int = 0, absolute_deadline_us: int | None = None,
                input_units: int = 0, output_units: int = 0,
                model_memory_bytes: int = 0, workload_kind: int = 4) -> Iterator[tuple[str, int]]:
        if not model_id:
            raise ValueError("scheduled caption workload requires a model ID")
        if absolute_deadline_us is not None and (
                isinstance(absolute_deadline_us, bool) or not isinstance(absolute_deadline_us, int)
                or not 0 < absolute_deadline_us < 2**64):
            raise ValueError("deadline must be a positive uint64 timestamp in microseconds")
        allowed = allowed_backend_ids or self.backend_ids
        if not allowed or any(item not in self.backend_ids for item in allowed):
            raise ValueError("workload backend allowlist must match configured scheduler resources")
        if isinstance(priority, bool) or not isinstance(priority, int) or not -(2**31) <= priority < 2**31:
            raise ValueError("workload priority is outside the C++ int32 range")
        for name, value in (("input_units", input_units), ("output_units", output_units),
                            ("model_memory_bytes", model_memory_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(workload_kind, bool) or workload_kind not in range(5):
            raise ValueError("workload_kind must be a C++ WorkloadKind value in 0..4")
        task_id = uuid.uuid4().hex
        started_ns = time.monotonic_ns()
        encoded_id = task_id.encode("ascii")
        backend_ids = (ctypes.c_char_p * len(allowed))(*[item.encode("utf-8") for item in allowed])
        service_estimates = (_ServiceEstimate * len(self.service_estimates_by_backend_us))(*[
            _ServiceEstimate(key.encode("utf-8"), value)
            for key, value in self.service_estimates_by_backend_us.items()
        ])
        workload = _WorkloadSpec(
            encoded_id, model_id.encode("utf-8"), workload_kind, priority,
            int(absolute_deadline_us is not None), absolute_deadline_us or 0,
            1, model_memory_bytes, output_units, input_units,
            backend_ids, len(backend_ids), service_estimates, len(service_estimates))
        error = ctypes.create_string_buffer(512)
        with self._condition:
            if self._closed:
                raise RuntimeError("native scheduler is closed")
            accepted = self._library.hf_scheduler_submit(
                self._handle, ctypes.byref(workload), error, len(error))
            if not accepted:
                raise RuntimeError(self._decode_error(error) or "native scheduler rejected task")
            self._dispatch_locked()
            deadline = time.monotonic() + self.wait_timeout_s
            while task_id not in self._assigned:
                if self._closed:
                    self._library.hf_scheduler_cancel_queued(self._handle, encoded_id)
                    raise RuntimeError("native scheduler closed while task was queued")
                failure = self._errors.pop(task_id, None)
                if failure:
                    raise RuntimeError(f"native scheduler could not place task: {failure}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._library.hf_scheduler_cancel_queued(self._handle, encoded_id)
                    self._dispatch_locked()
                    raise TimeoutError("native scheduler admission timed out")
                self._condition.wait(remaining)
            resource_id = self._assigned.pop(task_id)
            decision = {
                "task_id": task_id, "resource_id": resource_id,
                "reason": self._decision_reasons.pop(task_id, ""),
                "dispatched": True, "recorded_ns": time.time_ns(),
            }
            self._thread_decisions[threading.get_ident()] = decision
        queue_wait_us = (time.monotonic_ns() - started_ns) // 1000
        try:
            yield resource_id, queue_wait_us
        except BaseException as error:
            with self._condition:
                reason = type(error).__name__.encode("ascii", errors="replace")
                if not self._library.hf_scheduler_fail(self._handle, encoded_id, reason):
                    raise RuntimeError("native scheduler failed to release a failed task") from error
                self._dispatch_locked()
            raise
        else:
            with self._condition:
                if not self._library.hf_scheduler_complete(self._handle, encoded_id):
                    raise RuntimeError("native scheduler failed to complete a task")
                self._dispatch_locked()

    def decision_for_current_thread(self) -> dict[str, str | int | bool] | None:
        """Return the most recent concrete C++ dispatch decision for this caller."""
        with self._condition:
            return self._thread_decisions.pop(threading.get_ident(), None)

    @property
    def decision_history(self) -> list[dict[str, str | int | bool]]:
        with self._condition:
            return [dict(item) for item in self._decision_history]

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            self._library.hf_scheduler_destroy(self._handle)
            self._handle = None


class ScheduledCaptioner:
    """Schedule each caption against one of the explicitly configured workers."""

    def __init__(self, backends: list[object], scheduler: NativeCaptionScheduler):
        if not backends:
            raise ValueError("at least one caption worker is required")
        model_ids = {backend.model_id for backend in backends}
        if len(model_ids) != 1:
            raise ValueError("all scheduled caption workers must use the same model ID")
        self.backends = tuple(backends)
        self.scheduler = scheduler
        self._backend_by_resource = {
            backend.profile.device_id: backend for backend in self.backends
        }
        self._video_sampler_backend = None
        if set(self._backend_by_resource) != {str(item["resource_id"]) for item in scheduler.resources}:
            raise ValueError("caption workers do not match the configured scheduler resources")

    @property
    def backend_id(self) -> str:
        if len(self.backends) == 1:
            return self.backends[0].backend_id
        return "hexagonflow-cpp-scheduled-geniex-pool"

    @property
    def model_id(self) -> str:
        return self.backends[0].model_id

    @property
    def max_concurrent_requests(self) -> int:
        """Expose only the concurrency explicitly admitted by scheduler resources."""
        return sum(int(item["max_concurrency"]) for item in self.scheduler.resources)

    @property
    def simulation_only(self) -> bool:
        return all(backend.simulation_only for backend in self.backends)

    @property
    def run_configuration(self) -> dict:
        result = {
            "scheduler": "hexagonflow-cpp",
            "scheduler_policy": self.scheduler.policy,
            "scheduler_resources": self.scheduler.resources,
            "service_estimates_by_backend_us": self.scheduler.service_estimates_by_backend_us,
        }
        if len(self.backends) == 1:
            source = getattr(self.backends[0], "run_configuration", None)
            if isinstance(source, dict):
                result["worker"] = dict(source)
        return result

    @property
    def worker_metadata(self):
        ready = []
        for backend in self.backends:
            metadata = getattr(backend, "worker_metadata", None)
            if metadata is not None:
                ready.append({
                    "device_id": metadata.get("device_id", backend.profile.device_id),
                    "backend_id": backend.backend_id,
                    "runtime_version": metadata.get("runtime_version", ""),
                    "plugin_version": metadata.get("plugin_version", ""),
                    "estimated_resident_bytes": metadata.get("estimated_resident_bytes"),
                })
        if not ready:
            return None
        return {
            "runtime_devices": ready,
            "scheduler": "hexagonflow-cpp",
            "scheduler_policy": self.scheduler.policy,
            "worker_count_ready": len(ready),
        }

    @property
    def video_ingestion_metadata(self):
        metadata = getattr(self._video_sampler_backend, "video_ingestion_metadata", None)
        return dict(metadata) if isinstance(metadata, dict) else None

    @property
    def scheduler_metadata(self) -> dict:
        return {
            "engine": "hexagonflow-cpp",
            "policy": self.scheduler.policy,
            "resources": self.scheduler.resources,
            "memory_budget_status": "unknown_until_runtime-accounting-is-integrated",
            "interference_profile_sha256": self.scheduler.interference_profile_sha256 or None,
        }

    def describe(self, image_path: Path, prompt: str):
        with self.scheduler.reserve(self.model_id) as (resource_id, queue_wait_us):
            backend = self._backend_by_resource.get(resource_id)
            if backend is None:
                raise RuntimeError("native scheduler selected an unconfigured caption resource")
            started_ns = time.monotonic_ns()
            result: CaptionResult = backend.describe(image_path, prompt)
            active_us = (time.monotonic_ns() - started_ns) // 1000
            metrics = dict(result.inference_metrics)
            metrics["scheduler_queue_wait_us"] = queue_wait_us
            metrics["scheduler_resource_hold_us"] = active_us
            return replace(result, inference_metrics=metrics)

    def extract_video_frames(self, video_path: Path, artifact_dir: Path, policy, *,
                             progress_callback=None):
        sampler = next((backend for backend in self.backends
                        if callable(getattr(backend, "extract_video_frames", None))), None)
        if sampler is None:
            raise RuntimeError("no configured caption worker supports device-side video decoding")
        self._video_sampler_backend = sampler
        return sampler.extract_video_frames(
            video_path, artifact_dir, policy, progress_callback=progress_callback)

    def describe_remote(self, remote_path: str, image_path: Path, prompt: str):
        with self.scheduler.reserve(self.model_id) as (resource_id, queue_wait_us):
            backend = self._backend_by_resource.get(resource_id)
            if backend is None:
                raise RuntimeError("native scheduler selected an unconfigured caption resource")
            describe_remote = getattr(backend, "describe_remote", None)
            started_ns = time.monotonic_ns()
            result: CaptionResult = (
                describe_remote(remote_path, image_path, prompt) if callable(describe_remote)
                else backend.describe(image_path, prompt))
            metrics = dict(result.inference_metrics)
            metrics["scheduler_queue_wait_us"] = queue_wait_us
            metrics["scheduler_resource_hold_us"] = (time.monotonic_ns() - started_ns) // 1000
            return replace(result, inference_metrics=metrics)

    def cleanup_remote_frames(self, samples) -> None:
        if self._video_sampler_backend is None:
            return
        cleanup = getattr(self._video_sampler_backend, "cleanup_remote_frames", None)
        if callable(cleanup):
            cleanup(samples)

    def close(self) -> None:
        try:
            for backend in self.backends:
                close_backend = getattr(backend, "close", None)
                if callable(close_backend):
                    close_backend()
        finally:
            try:
                self.scheduler.close()
            finally:
                clear_password = getattr(getattr(self, "password_provider", None), "clear", None)
                if callable(clear_password):
                    clear_password()


def schedule_qdc_captioner(backends: list[object], profile) -> ScheduledCaptioner:
    """Create scheduler resources from the local profile, without inferring topology."""
    if not backends:
        raise ValueError("QDC GenieX requires at least one configured backend")
    workers = tuple(getattr(profile, "worker_resources", ()))
    if not workers:
        raise ValueError("QDC profile has no explicit worker resource configuration")
    backend_by_device = {backend.profile.device_id: backend for backend in backends}
    resources = []
    estimates: dict[str, int] = {}
    for worker in workers:
        backend = backend_by_device.get(worker.device_id)
        if backend is None:
            raise ValueError("QDC worker resource has no matching backend instance")
        resources.append({
            "resource_id": worker.device_id,
            "backend_id": backend.backend_id,
            "family": 1,
            "max_concurrency": worker.max_concurrency,
            "memory_capacity_bytes": 0,
            "interference_domain_id": worker.interference_domain_id,
        })
        if worker.service_estimate_us is not None:
            estimates[backend.backend_id] = worker.service_estimate_us
    interference_path = None
    if profile.interference_profile_path:
        candidate = Path(profile.interference_profile_path)
        interference_path = candidate if candidate.is_absolute() else ROOT / candidate
    scheduler = NativeCaptionScheduler(
        resources=resources, policy=profile.scheduling_policy,
        service_estimates_by_backend_us=estimates,
        interference_profile_path=interference_path,
        interference_profile_sha256=profile.interference_profile_sha256,
    )
    return ScheduledCaptioner(backends, scheduler)
