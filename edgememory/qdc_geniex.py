from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .captions import CAPTION_PROMPT
from .types import CaptionResult


ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_PROFILE_KEYS = {
    "password", "api_key", "session_id", "private_key", "key_content",
    "secret", "credential", "token", "tunnel_command", "raw_tunnel_command",
}


@dataclass(frozen=True)
class QdcWorkerResource:
    device_id: str
    max_concurrency: int = 1
    interference_domain_id: str = ""
    service_estimate_us: int | None = None


@dataclass(frozen=True)
class QdcGenieXVlmProfile:
    host: str
    port: int
    username: str
    device_id: str
    model_id: str
    remote_service_binary: str
    remote_runtime_root: str
    remote_model_path: str
    remote_mmproj_path: str
    context_length: int = 2048
    threads: int = 4
    max_tokens: int = 96
    connect_timeout_s: int = 15
    request_timeout_s: int = 180
    remote_work_root: str = "/tmp"
    worker_resources: tuple[QdcWorkerResource, ...] = ()
    scheduling_policy: str = "fifo"
    interference_profile_path: str = ""
    interference_profile_sha256: str = ""
    hexagon_devices: str = ""
    vision_device_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "QdcGenieXVlmProfile":
        def contains_forbidden(item: object) -> bool:
            if isinstance(item, Mapping):
                return any(
                    str(key).casefold() in _FORBIDDEN_PROFILE_KEYS or contains_forbidden(nested)
                    for key, nested in item.items()
                )
            if isinstance(item, (list, tuple)):
                return any(contains_forbidden(nested) for nested in item)
            return False

        if contains_forbidden(value):
            raise ValueError("QDC profile must not contain credentials or raw tunnel commands")
        if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
            raise ValueError("unsupported QDC GenieX profile schema")

        allowed_fields = {
            "schema_version", "host", "port", "username", "device_id", "model_id",
            "remote_service_binary", "remote_runtime_root", "remote_model_path",
            "remote_mmproj_path", "context_length", "threads", "max_tokens",
            "connect_timeout_s", "request_timeout_s", "remote_work_root",
            "worker_resources", "scheduling_policy", "interference_profile_path",
            "interference_profile_sha256", "hexagon_devices", "vision_device_id",
        }
        if set(value) - allowed_fields:
            raise ValueError("QDC profile contains unsupported fields")
        string_fields = (
            "host", "username", "device_id", "model_id", "remote_service_binary",
            "remote_runtime_root", "remote_model_path", "remote_mmproj_path",
        )
        strings: dict[str, str] = {}
        for name in string_fields:
            item = value.get(name)
            if not isinstance(item, str) or not item.strip() or "\x00" in item:
                raise ValueError(f"QDC profile requires {name}")
            strings[name] = item.strip()
        if any(any(c.isspace() for c in strings[name]) for name in ("host", "username")):
            raise ValueError("QDC host and username must not contain whitespace")
        for name in ("remote_service_binary", "remote_runtime_root", "remote_model_path", "remote_mmproj_path"):
            if not strings[name].startswith("/"):
                raise ValueError(f"QDC profile requires an absolute {name}")

        work_root = value.get("remote_work_root", "/tmp")
        if (not isinstance(work_root, str) or "\x00" in work_root or
                work_root not in ("/tmp", "/root/hexagonflow")):
            raise ValueError(
                "remote_work_root must be /tmp or the supported /root/hexagonflow disk-backed root")

        numeric_bounds = {
            "port": (1, 65535, None),
            "context_length": (1, 131072, 2048),
            "threads": (1, 256, 4),
            "max_tokens": (1, 4096, 96),
            "connect_timeout_s": (1, 60, 15),
            "request_timeout_s": (1, 600, 180),
        }
        numbers: dict[str, int] = {}
        for name, (lower, upper, default) in numeric_bounds.items():
            item = value.get(name, default)
            if isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper:
                raise ValueError(f"QDC profile has invalid {name}")
            numbers[name] = item
        resource_rows = value.get("worker_resources")
        parsed_resources: list[QdcWorkerResource] = []
        if resource_rows is None:
            parsed_resources = [QdcWorkerResource(strings["device_id"])]
        else:
            if not isinstance(resource_rows, list) or not 1 <= len(resource_rows) <= 4:
                raise ValueError("worker_resources must contain 1..4 configured resources")
            seen_device_ids: set[str] = set()
            for row in resource_rows:
                if not isinstance(row, dict) or set(row) - {
                    "device_id", "max_concurrency", "interference_domain_id", "service_estimate_us"
                }:
                    raise ValueError("worker resource contains unsupported fields")
                device_id = row.get("device_id")
                if (not isinstance(device_id, str) or not device_id.strip() or
                        "\x00" in device_id or len(device_id.encode("utf-8")) >= 128):
                    raise ValueError("worker resource requires a short device_id")
                device_id = device_id.strip()
                if device_id in seen_device_ids:
                    raise ValueError("worker resource device IDs must be unique")
                seen_device_ids.add(device_id)
                concurrency = row.get("max_concurrency", 1)
                if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency != 1:
                    raise ValueError("each current GenieX service worker must use max_concurrency=1")
                domain = row.get("interference_domain_id", "")
                if (not isinstance(domain, str) or "\x00" in domain or
                        len(domain.encode("utf-8")) >= 128):
                    raise ValueError("worker resource interference domain is invalid")
                estimate = row.get("service_estimate_us")
                if estimate is not None and (
                        isinstance(estimate, bool) or not isinstance(estimate, int) or
                        not 0 < estimate < 2**64):
                    raise ValueError("service_estimate_us must be a positive integer")
                parsed_resources.append(QdcWorkerResource(
                    device_id=device_id, max_concurrency=1,
                    interference_domain_id=domain.strip(), service_estimate_us=estimate))
            if strings["device_id"] not in seen_device_ids:
                raise ValueError("worker_resources must include the primary device_id")

        policy = value.get("scheduling_policy", "fifo")
        allowed_policies = {"fifo", "priority", "edf", "predicted_latency", "interference_aware_latency"}
        if not isinstance(policy, str) or policy not in allowed_policies:
            raise ValueError("unsupported scheduling_policy")
        profile_path = value.get("interference_profile_path", "")
        profile_sha = value.get("interference_profile_sha256", "")
        if not isinstance(profile_path, str) or "\x00" in profile_path:
            raise ValueError("interference_profile_path must be a string")
        if not isinstance(profile_sha, str) or (profile_sha and not all(
                character in "0123456789abcdefABCDEF" for character in profile_sha)):
            raise ValueError("interference_profile_sha256 must be hexadecimal")
        hexagon_devices = value.get("hexagon_devices", "")
        vision_device_id = value.get("vision_device_id", "")
        if (not isinstance(hexagon_devices, str) or
                (hexagon_devices and not re.fullmatch(r"[A-Za-z0-9,:\[\]-]{1,255}", hexagon_devices))):
            raise ValueError("hexagon_devices has an invalid selector list")
        if (not isinstance(vision_device_id, str) or
                (vision_device_id and not re.fullmatch(r"[A-Za-z0-9,:\[\]-]{1,127}", vision_device_id))):
            raise ValueError("vision_device_id has an invalid selector")
        if policy == "interference_aware_latency":
            if not profile_path or len(profile_sha) != 64:
                raise ValueError("interference-aware scheduling requires a profile path and SHA256")
            if any(row.service_estimate_us is None for row in parsed_resources):
                raise ValueError("interference-aware scheduling requires service estimates for every resource")
        elif profile_path or profile_sha:
            raise ValueError("interference profile fields require interference_aware_latency")
        if policy == "predicted_latency" and any(row.service_estimate_us is None for row in parsed_resources):
            raise ValueError("predicted_latency requires service estimates for every resource")
        return cls(**strings, **numbers, remote_work_root=work_root,
                   worker_resources=tuple(parsed_resources), scheduling_policy=policy,
                   interference_profile_path=profile_path,
                   interference_profile_sha256=profile_sha,
                   hexagon_devices=hexagon_devices,
                   vision_device_id=vision_device_id)


def load_qdc_geniex_profile(path: Path) -> QdcGenieXVlmProfile:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to((ROOT / ".local").resolve())
    except ValueError as error:
        raise ValueError("QDC profiles must remain under the ignored .local directory") from error
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("QDC GenieX profile is unreadable or malformed") from error
    if not isinstance(raw, dict):
        raise ValueError("QDC GenieX profile must be a JSON object")
    return QdcGenieXVlmProfile.from_mapping(raw)


def _hex(value: str) -> str:
    return value.encode("utf-8").hex()


def _unhex(value: str, *, max_bytes: int) -> str:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("GenieX service returned invalid hex text") from error
    if len(decoded) > max_bytes or b"\x00" in decoded:
        raise ValueError("GenieX service returned an invalid text field")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("GenieX service returned invalid UTF-8 text") from error


def parse_service_ready(line: str) -> dict[str, str | int]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 7 or fields[0] != "READY" or fields[1] != "1":
        raise RuntimeError("GenieX worker returned an invalid readiness response")
    try:
        return {
            "model_id": _unhex(fields[2], max_bytes=256),
            "device_id": _unhex(fields[3], max_bytes=256),
            "runtime_version": _unhex(fields[4], max_bytes=256),
            "plugin_version": _unhex(fields[5], max_bytes=256),
            "estimated_resident_bytes": int(fields[6]),
        }
    except (ValueError, OverflowError) as error:
        raise RuntimeError("GenieX worker returned malformed readiness metadata") from error


def parse_service_result(line: str, request_id: str) -> tuple[str, dict[str, int | float]]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) >= 3 and fields[0] == "ERROR":
        reason = _unhex(fields[2], max_bytes=1024) if fields[1] == request_id else "worker-protocol-error"
        raise RuntimeError(f"GenieX remote request failed: {reason}")
    if len(fields) != 15 or fields[0] != "RES" or fields[1] != request_id:
        raise RuntimeError("GenieX worker returned a mismatched response")
    if fields[2] != "1" or fields[3] != "0":
        message = _unhex(fields[14], max_bytes=1024)
        raise RuntimeError(f"GenieX remote request failed: {message or 'backend-error'}")
    try:
        metrics: dict[str, int | float] = {
            "output_tokens": int(fields[4]),
            "queue_wait_us": int(fields[5]),
            "worker_elapsed_us": int(fields[6]),
            "ttft_us": int(fields[7]),
            "media_us": int(fields[8]),
            "prefill_us": int(fields[9]),
            "decode_us": int(fields[10]),
            "prefill_tokens_per_second": float(fields[11]),
            "decode_tokens_per_second": float(fields[12]),
        }
    except (ValueError, OverflowError) as error:
        raise RuntimeError("GenieX worker returned malformed timing fields") from error
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise RuntimeError("GenieX worker returned non-finite timing fields")
    return _unhex(fields[13], max_bytes=65536), metrics


class QdcGenieXVlmCaptioner:
    """Use a resident external GenieX C SDK worker over the current QDC SSH tunnel."""

    def __init__(self, profile: QdcGenieXVlmProfile, *,
                 password_provider: Callable[[], str],
                 log_dir: Path | None = None,
                 ssh_client_factory: Callable[[], object] | None = None):
        if not callable(password_provider):
            raise ValueError("a runtime password provider is required")
        self.profile = profile
        self.password_provider = password_provider
        self.log_dir = (log_dir or (ROOT / ".local" / "edgememory" / "qdc-logs")).resolve()
        try:
            self.log_dir.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError("QDC logs must remain on D: inside the project workspace") from error
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._ssh_client_factory = ssh_client_factory
        self._client = None
        self._sftp = None
        self._channel = None
        self._stdin = None
        self._stdout = None
        self._stderr = None
        self._stdout_buffer = bytearray()
        self._remote_work_dir = ""
        self._request_lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._ready: dict[str, str | int] | None = None
        self._video_ingestion_metadata: dict[str, object] | None = None
        self._remote_sample_roots: set[str] = set()
        self._stderr_path = self.log_dir / f"geniex-{uuid.uuid4().hex}.stderr.log"

    @property
    def backend_id(self) -> str:
        return f"geniex-c-sdk-{self.profile.device_id}"

    @property
    def run_configuration(self) -> dict[str, str | int]:
        """Return non-secret inference settings for the local experiment record."""
        return {
            "context_length": self.profile.context_length,
            "threads": self.profile.threads,
            "max_tokens": self.profile.max_tokens,
            "connect_timeout_s": self.profile.connect_timeout_s,
            "request_timeout_s": self.profile.request_timeout_s,
            "hexagon_devices": self.profile.hexagon_devices,
            "vision_device_id": self.profile.vision_device_id,
        }

    @property
    def model_id(self) -> str:
        return self.profile.model_id

    @property
    def simulation_only(self) -> bool:
        return False

    @property
    def worker_metadata(self) -> dict[str, str | int] | None:
        return dict(self._ready) if self._ready is not None else None

    @property
    def video_ingestion_metadata(self) -> dict[str, object] | None:
        return dict(self._video_ingestion_metadata) if self._video_ingestion_metadata else None

    def _new_client(self):
        if self._ssh_client_factory is not None:
            return self._ssh_client_factory()
        try:
            import paramiko
        except ImportError as error:
            raise RuntimeError("the QDC caption backend requires the existing Paramiko package") from error
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    def _remote_command(self) -> str:
        profile = self.profile
        library = f"{profile.remote_runtime_root}/lib/libgeniex.so"
        plugin_dir = f"{profile.remote_runtime_root}/lib"
        plugin_lib = f"{profile.remote_runtime_root}/lib/llama_cpp"
        environment = {
            "LD_LIBRARY_PATH": f"{plugin_dir}:{plugin_lib}:/usr/lib:/usr/lib/rfsa/adsp",
            "ADSP_LIBRARY_PATH": f"{plugin_lib}:/usr/lib/rfsa/adsp",
            "GENIEX_PLUGIN_PATH": plugin_dir,
        }
        if profile.hexagon_devices:
            environment["GGML_HEXAGON_DEVICES"] = profile.hexagon_devices
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
        arguments = [
            profile.remote_service_binary,
            "--model", profile.remote_model_path,
            "--mmproj", profile.remote_mmproj_path,
            "--model-id", profile.model_id,
            "--device-id", profile.device_id,
            "--library", library,
            "--plugin-directory", plugin_dir,
            "--context", str(profile.context_length),
            "--threads", str(profile.threads),
            "--max-tokens", str(profile.max_tokens),
        ]
        if profile.vision_device_id:
            arguments.extend(("--vit-device-id", profile.vision_device_id))
        return "ulimit -c 0; exec env " + prefix + " " + " ".join(
            shlex.quote(argument) for argument in arguments
        )

    def _drain_stderr(self) -> None:
        if self._channel is None:
            return
        if not self._channel.recv_stderr_ready():
            return
        block = self._channel.recv_stderr(65536)
        if block:
            with self._stderr_path.open("ab") as stream:
                stream.write(block)

    def _readline(self, timeout_s: float) -> str:
        if self._channel is None:
            raise RuntimeError("GenieX worker is not connected")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._drain_stderr()
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[:newline + 1]
                try:
                    return line.decode("utf-8").rstrip("\r")
                except UnicodeDecodeError as error:
                    raise RuntimeError("GenieX worker output is not UTF-8") from error
            if self._channel.recv_ready():
                block = self._channel.recv(65536)
                if not block:
                    break
                self._stdout_buffer.extend(block)
                continue
            if self._channel.exit_status_ready():
                self._drain_stderr()
                break
            time.sleep(0.01)
        raise TimeoutError("GenieX worker response timed out or the SSH channel closed")

    def _ensure_connected(self) -> None:
        if self._closed:
            raise RuntimeError("GenieX captioner is closed")
        if self._failed:
            raise RuntimeError("GenieX captioner transport failed; create a new worker before retrying")
        if self._channel is not None:
            if self._channel.exit_status_ready():
                raise RuntimeError("GenieX worker process exited")
            return
        client = self._new_client()
        password = self.password_provider()
        if not isinstance(password, str) or not password:
            raise RuntimeError("the runtime password provider returned no password")
        sftp = None
        remote_dir = ""
        try:
            client.connect(
                hostname=self.profile.host,
                port=self.profile.port,
                username=self.profile.username,
                password=password,
                look_for_keys=False,
                allow_agent=False,
                timeout=self.profile.connect_timeout_s,
                banner_timeout=self.profile.connect_timeout_s,
                auth_timeout=self.profile.connect_timeout_s,
            )
            password = ""
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise RuntimeError("QDC SSH transport is not active")
            transport.set_keepalive(20)
            sftp = client.open_sftp()
            remote_dir = f"{self.profile.remote_work_root}/hexagonflow-edgememory-{uuid.uuid4().hex}"
            sftp.mkdir(remote_dir, mode=0o700)
            stdin, stdout, stderr = client.exec_command(self._remote_command(), get_pty=False)
            self._client, self._sftp = client, sftp
            # Paramiko's channel file wrappers own the channel's input/output sides.
            # Keep them alive for the whole service lifetime; dropping stdin here
            # can close the remote process input immediately after READY.
            self._stdin, self._stdout, self._stderr = stdin, stdout, stderr
            self._channel = stdout.channel
            self._channel.settimeout(self.profile.request_timeout_s)
            ready = self._readline(self.profile.request_timeout_s)
            self._ready = parse_service_ready(ready)
            if self._ready["device_id"] != self.profile.device_id or self._ready["model_id"] != self.profile.model_id:
                raise RuntimeError("GenieX worker metadata does not match the local profile")
            self._remote_work_dir = remote_dir
        except Exception as error:
            password = ""
            if sftp is not None:
                if remote_dir:
                    try:
                        sftp.rmdir(remote_dir)
                    except Exception:
                        pass
                try:
                    sftp.close()
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass
            self._client = self._sftp = self._channel = None
            self._stdin = self._stdout = self._stderr = None
            if isinstance(error, (TimeoutError, RuntimeError)):
                raise
            raise RuntimeError(f"QDC GenieX worker startup failed ({type(error).__name__})") from None

    def describe(self, image_path: Path, prompt: str = CAPTION_PROMPT) -> CaptionResult:
        with self._request_lock:
            self._ensure_connected()
            assert self._sftp is not None and self._channel is not None
            caption_started = time.perf_counter_ns()
            image_path = image_path.expanduser().resolve(strict=True)
            image_bytes = image_path.read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            remote_image = f"{self._remote_work_dir}/{digest[:16]}-{uuid.uuid4().hex}.jpg"
            request_id = uuid.uuid4().hex
            transfer_started = time.perf_counter_ns()
            try:
                # Paramiko's put(confirm=True) performs one remote stat and
                # confirms the uploaded byte count. Reading the frame back over
                # SFTP adds another round-trip and provides no stronger
                # transport guarantee than the authenticated SSH channel.
                remote_stat = self._sftp.put(str(image_path), remote_image, confirm=True)
                if remote_stat.st_size != len(image_bytes):
                    raise RuntimeError("QDC frame upload size did not match")
                transfer_us = (time.perf_counter_ns() - transfer_started) // 1000

                request = (
                    f"REQ\t{request_id}\t{self.profile.max_tokens}\t"
                    f"{_hex(prompt)}\t{_hex(remote_image)}\n"
                ).encode("ascii")
                request_started = time.perf_counter_ns()
                try:
                    self._channel.sendall(request)
                    response_line = self._readline(self.profile.request_timeout_s)
                except (OSError, TimeoutError):
                    self._abort_connection()
                    raise
                text, metrics = parse_service_result(response_line, request_id)
                metrics["frame_transfer_us"] = int(transfer_us)
                metrics["service_roundtrip_us"] = int((time.perf_counter_ns() - request_started) // 1000)
                metrics["caption_total_us"] = int((time.perf_counter_ns() - caption_started) // 1000)
                metrics["frame_bytes"] = len(image_bytes)
                return CaptionResult(
                    text=text,
                    backend_id=self.backend_id,
                    model_id=self.model_id,
                    simulation_only=False,
                    inference_metrics=metrics,
                )
            finally:
                try:
                    self._sftp.remove(remote_image)
                except Exception:
                    pass

    def extract_video_frames(self, video_path: Path, artifact_dir: Path, policy, *,
                             progress_callback=None):
        """Upload a source video once, sample it on-device with GStreamer, and retain
        only selected timestamped evidence frames locally for replay and audit.
        """
        import cv2
        import numpy as np
        from .sampling import sha256_file
        from .types import FrameSample, VideoMetadata

        policy.validate()
        video_path = video_path.expanduser().resolve(strict=True)
        if not video_path.is_file():
            raise ValueError("video input must be a regular file")
        source_hash = sha256_file(video_path)
        media_id = source_hash[:24]
        artifact_dir = artifact_dir.expanduser().resolve()
        frame_dir = artifact_dir / "frames" / media_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(0, 0)

        with self._request_lock:
            self._ensure_connected()
            assert self._sftp is not None and self._client is not None
            remote_video = f"{self._remote_work_dir}/video-{source_hash[:16]}{video_path.suffix[:12]}"
            remote_script = f"{self._remote_work_dir}/gstreamer_video_sampler.py"
            remote_samples = f"{self._remote_work_dir}/samples-{uuid.uuid4().hex}"
            self._sftp.mkdir(remote_samples, mode=0o700)
            helper = ROOT / "tools" / "remote" / "gstreamer_video_sampler.py"
            if not helper.is_file():
                raise RuntimeError("on-device video sampler source is missing")
            script_transfer_started = time.perf_counter_ns()
            self._sftp.put(str(helper), remote_script, confirm=True)
            self._sftp.chmod(remote_script, 0o700)
            source_stat = video_path.stat()
            upload_started = time.perf_counter_ns()
            uploaded = self._sftp.put(str(video_path), remote_video, confirm=True)
            video_upload_us = (time.perf_counter_ns() - upload_started) // 1000
            if uploaded.st_size != source_stat.st_size:
                try:
                    self._sftp.remove(remote_video)
                except Exception:
                    pass
                raise RuntimeError("QDC video upload size did not match")

            interval_ms = max(100, round(policy.min_interval_s * 1000))
            arguments = [
                "python3", remote_script, "--video", remote_video,
                "--output", remote_samples, "--max-samples", str(policy.max_samples),
                "--interval-ms", str(interval_ms),
                "--max-frame-side", str(policy.max_frame_side),
                "--jpeg-quality", str(policy.jpeg_quality),
            ]
            command = "env PYTHONUNBUFFERED=1 " + " ".join(
                shlex.quote(argument) for argument in arguments)
            try:
                _stdin, stdout, _stderr = self._client.exec_command(
                    command, timeout=self.profile.request_timeout_s)
                stdout.channel.settimeout(self.profile.request_timeout_s)
                output = stdout.read(2 * 1024 * 1024)
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    raise RuntimeError(f"on-device GStreamer sampling failed ({exit_status})")
                try:
                    response = json.loads(output.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError("on-device sampler returned invalid metadata") from error
            finally:
                try:
                    self._sftp.remove(remote_video)
                except Exception:
                    pass
            if (not isinstance(response, dict) or response.get("schema_version") != 1 or
                    not isinstance(response.get("frames"), list) or
                    len(response["frames"]) > policy.max_samples):
                raise RuntimeError("on-device sampler returned an invalid frame manifest")
            self._remote_sample_roots.add(remote_samples)

            fps = float(response.get("fps", 0.0) or 0.0)
            duration_ms = int(response.get("duration_ms", 0) or 0)
            source_width = int(response.get("source_width", response.get("width", 0)) or 0)
            source_height = int(response.get("source_height", response.get("height", 0)) or 0)
            samples: list[FrameSample] = []
            previous_gray = None
            last_sample_ms: int | None = None
            for candidate in response["frames"]:
                if not isinstance(candidate, dict):
                    raise RuntimeError("on-device sampler frame row is invalid")
                remote_frame = candidate.get("path")
                expected_prefix = remote_samples.rstrip("/") + "/"
                if (not isinstance(remote_frame, str) or
                        not posixpath.normpath(remote_frame).startswith(expected_prefix)):
                    raise RuntimeError("on-device sampler frame path escaped its work directory")
                remote_frame = posixpath.normpath(remote_frame)
                with self._sftp.open(remote_frame, "rb") as remote_stream:
                    payload = remote_stream.read(16 * 1024 * 1024 + 1)
                if len(payload) > 16 * 1024 * 1024:
                    raise RuntimeError("on-device evidence frame exceeds the transfer limit")
                timestamp_ms = int(candidate.get("timestamp_ms", -1))
                if timestamp_ms < 0:
                    raise RuntimeError("on-device sampler returned an invalid timestamp")
                encoded = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("on-device sampler returned an unreadable JPEG")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                delta = (0.0 if previous_gray is None else
                         float(cv2.absdiff(gray, previous_gray).mean()) / 255.0)
                elapsed_ms = None if last_sample_ms is None else timestamp_ms - last_sample_ms
                reason = "first_frame" if last_sample_ms is None else ""
                if last_sample_ms is not None and elapsed_ms is not None:
                    if elapsed_ms >= round(policy.max_interval_s * 1000):
                        reason = "max_interval"
                    elif (elapsed_ms >= round(policy.min_interval_s * 1000) and
                          delta >= policy.scene_change_threshold):
                        reason = "scene_change"
                if reason:
                    frame_index = round(timestamp_ms * fps / 1000.0) if fps > 0 else int(candidate["frame_index"])
                    frame_path = frame_dir / f"frame_{frame_index:09d}_{timestamp_ms:012d}.jpg"
                    frame_path.write_bytes(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    samples.append(FrameSample(
                        media_id=media_id, frame_index=frame_index,
                        timestamp_ms=timestamp_ms, frame_path=frame_path,
                        frame_sha256=digest, scene_delta=delta,
                        sample_reason=reason, remote_path=remote_frame,
                    ))
                    last_sample_ms = timestamp_ms
                    previous_gray = gray
                elif previous_gray is None or delta >= policy.scene_change_threshold / 4:
                    previous_gray = gray

            self._video_ingestion_metadata = {
                "execution_location": "qdc_device",
                "video_decoder": "gstreamer_decodebin",
                "decoder_factories": response.get("decoder_factories", []),
                "source_upload_count": 1,
                "source_bytes": source_stat.st_size,
                "source_upload_us": int(video_upload_us),
                "sampler_script_upload_us": int((time.perf_counter_ns() - script_transfer_started) // 1000),
                "decode_elapsed_ms": int(response.get("decode_elapsed_ms", 0) or 0),
                "sampling_cadence_ms": interval_ms,
                "remote_frame_candidates": len(response["frames"]),
                "selected_evidence_frames": len(samples),
                "sampling_policy": "local_scene_selection_over_qdc_gstreamer_frames",
            }
            if progress_callback is not None:
                progress_callback(len(samples), len(samples))
            if not samples:
                self.cleanup_remote_frames([])
                raise RuntimeError("on-device GStreamer produced no selected evidence frames")
            metadata = VideoMetadata(
                media_id=media_id, source_path=video_path,
                source_sha256=source_hash, duration_ms=duration_ms,
                source_frame_count=round(duration_ms * fps / 1000.0) if fps > 0 else 0,
                fps=fps, width=source_width, height=source_height,
            )
            return metadata, samples

    def describe_remote(self, remote_image: str, image_path: Path,
                        prompt: str = CAPTION_PROMPT) -> CaptionResult:
        with self._request_lock:
            self._ensure_connected()
            assert self._sftp is not None and self._channel is not None
            normalized = posixpath.normpath(remote_image)
            if not normalized.startswith("/tmp/hexagonflow-edgememory-"):
                raise ValueError("remote evidence image is outside the disposable work area")
            image_path = image_path.expanduser().resolve(strict=True)
            image_bytes = image_path.read_bytes()
            remote_stat = self._sftp.stat(normalized)
            if remote_stat.st_size != len(image_bytes):
                raise RuntimeError("device-local evidence frame no longer matches its local copy")
            request_id = uuid.uuid4().hex
            request = (f"REQ\t{request_id}\t{self.profile.max_tokens}\t"
                       f"{_hex(prompt)}\t{_hex(normalized)}\n").encode("ascii")
            request_started = time.perf_counter_ns()
            try:
                self._channel.sendall(request)
                response_line = self._readline(self.profile.request_timeout_s)
            except (OSError, TimeoutError):
                self._abort_connection()
                raise
            text, metrics = parse_service_result(response_line, request_id)
            metrics["frame_transfer_us"] = 0
            metrics["service_roundtrip_us"] = int((time.perf_counter_ns() - request_started) // 1000)
            metrics["frame_bytes"] = len(image_bytes)
            metrics["device_local_frame"] = 1
            return CaptionResult(text=text, backend_id=self.backend_id, model_id=self.model_id,
                                 simulation_only=False, inference_metrics=metrics)

    def cleanup_remote_frames(self, samples) -> None:
        if self._sftp is None:
            return
        for remote_root in list(self._remote_sample_roots):
            if not remote_root.startswith(self._remote_work_dir + "/samples-"):
                continue
            try:
                for entry in self._sftp.listdir_attr(remote_root):
                    path = posixpath.join(remote_root, entry.filename)
                    try:
                        self._sftp.remove(path)
                    except Exception:
                        pass
                self._sftp.rmdir(remote_root)
            except Exception:
                pass
            self._remote_sample_roots.discard(remote_root)

    def _abort_connection(self) -> None:
        self._failed = True
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._sftp = self._client = self._channel = None
        self._stdin = self._stdout = self._stderr = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._channel is not None and not self._channel.exit_status_ready():
                self._channel.sendall(b"STOP\n")
                try:
                    self._readline(min(10.0, float(self.profile.request_timeout_s)))
                except Exception:
                    pass
        finally:
            if self._sftp is not None:
                if self._remote_work_dir:
                    try:
                        self._sftp.rmdir(self._remote_work_dir)
                    except Exception:
                        pass
                self._sftp.close()
            if self._client is not None:
                self._client.close()
            self._sftp = self._client = self._channel = None
            self._stdin = self._stdout = self._stderr = None

    def __enter__(self) -> "QdcGenieXVlmCaptioner":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
