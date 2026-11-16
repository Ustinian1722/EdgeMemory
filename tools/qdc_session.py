#!/usr/bin/env python3
"""Manual-session QDC helpers. No API calls and no connection data in run logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import hexagonflow

ROOT = hexagonflow.ROOT
RUNS = ROOT / "runs"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class SessionError(RuntimeError):
    pass


def checked_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise SessionError("run_id must contain only letters, digits, dot, underscore, or dash")
    return run_id


def _path_under_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise SessionError("session outputs must stay inside the D: workspace") from error
    return resolved


def _safe_remote_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise SessionError(f"profile requires an absolute {label}")
    if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        raise SessionError(f"invalid {label}")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise SessionError(f"invalid {label}")
    return value.rstrip("/") or "/"


@dataclass(frozen=True)
class ConnectionProfile:
    transport: str
    remote_root: str = ""
    host: str = ""
    port: int = 0
    username: str = ""
    identity_file: str = ""
    adb_port: int = 0
    adb_serial: str = ""


def load_profile(path: Path) -> ConnectionProfile:
    path = _path_under_root(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionError("local connection profile is missing or malformed") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SessionError("unsupported local connection profile schema")

    def forbidden_profile_fields(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in {
                    "password", "api_key", "session_id", "private_key", "key_content"
                } or forbidden_profile_fields(nested):
                    return True
        elif isinstance(value, list):
            return any(forbidden_profile_fields(item) for item in value)
        return False

    if forbidden_profile_fields(raw):
        raise SessionError("profile must not contain passwords, API keys, session IDs, or key material")
    transport = raw.get("transport")
    if transport not in {"ssh", "adb"}:
        raise SessionError("profile transport must be selected from the actual QDC Connect page")
    configured_remote_root = raw.get("remote_root")
    remote_root = (_safe_remote_path(configured_remote_root, "remote_root")
                   if configured_remote_root else "")
    if transport == "ssh":
        endpoint = raw.get("ssh", {})
        if not isinstance(endpoint, dict):
            raise SessionError("SSH profile section must be an object")
        host = endpoint.get("host")
        username = endpoint.get("username")
        port = endpoint.get("port")
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9._:-]+", host):
            raise SessionError("profile requires the host supplied by the QDC Connect page")
        if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", username):
            raise SessionError("profile requires the username shown by QDC")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise SessionError("profile requires the SSH port shown by QDC")
        identity_file = endpoint.get("identity_file", "")
        if identity_file:
            if not isinstance(identity_file, str) or not Path(identity_file).expanduser().is_absolute():
                raise SessionError("identity_file must be an absolute local path in the ignored profile")
        return ConnectionProfile(transport, remote_root, host, port, username,
                                 identity_file=identity_file)
    endpoint = raw.get("adb", {})
    if not isinstance(endpoint, dict):
        raise SessionError("ADB profile section must be an object")
    port = endpoint.get("port")
    serial = endpoint.get("serial", "")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise SessionError("profile requires the ADB port shown by QDC")
    if serial and (not isinstance(serial, str) or not re.fullmatch(r"[A-Za-z0-9._:-]+", serial)):
        raise SessionError("invalid locally discovered ADB serial")
    return ConnectionProfile(transport, remote_root, adb_port=port, adb_serial=serial)


class CommandRunner:
    """Runs only argv arrays, leaves auth prompts on the terminal, and hides args."""

    @staticmethod
    def run(argv: list[str], label: str, timeout_s: int,
            capture_stdout: bool = False) -> bytes:
        try:
            completed = subprocess.run(
                argv,
                stdin=None,
                stdout=subprocess.PIPE if capture_stdout else None,
                stderr=None,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SessionError(f"{label} failed or timed out") from error
        if completed.returncode != 0:
            raise SessionError(f"{label} exited with status {completed.returncode}")
        return completed.stdout or b""


class Transport:
    def __init__(self, profile: ConnectionProfile):
        self.profile = profile

    def execute(self, remote_argv: list[str], label: str, timeout_s: int,
                capture_stdout: bool = False) -> bytes:
        raise NotImplementedError

    def upload(self, local: Path, remote: str, label: str, timeout_s: int) -> None:
        raise NotImplementedError

    def download(self, remote: str, local: Path, label: str, timeout_s: int) -> None:
        raise NotImplementedError


class SshTransport(Transport):
    def _target(self) -> str:
        return f"{self.profile.username}@{self.profile.host}"

    def execute(self, remote_argv: list[str], label: str, timeout_s: int,
                capture_stdout: bool = False) -> bytes:
        command = shlex.join(remote_argv)
        identity_args = (["-i", str(Path(self.profile.identity_file).expanduser()),
                          "-o", "IdentitiesOnly=yes"]
                         if self.profile.identity_file else [])
        return CommandRunner.run(
            ["ssh", "-o", "ConnectTimeout=8", "-p", str(self.profile.port),
             *identity_args,
             self._target(), command], label, timeout_s, capture_stdout
        )

    def upload(self, local: Path, remote: str, label: str, timeout_s: int) -> None:
        identity_args = (["-i", str(Path(self.profile.identity_file).expanduser()),
                          "-o", "IdentitiesOnly=yes"]
                         if self.profile.identity_file else [])
        CommandRunner.run(
            ["scp", "-P", str(self.profile.port), *identity_args, str(local),
             f"{self._target()}:{remote}"], label, timeout_s
        )

    def download(self, remote: str, local: Path, label: str, timeout_s: int) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        identity_args = (["-i", str(Path(self.profile.identity_file).expanduser()),
                          "-o", "IdentitiesOnly=yes"]
                         if self.profile.identity_file else [])
        CommandRunner.run(
            ["scp", "-r", "-P", str(self.profile.port), *identity_args,
             f"{self._target()}:{remote}", str(local)], label, timeout_s
        )


class AdbTransport(Transport):
    def __init__(self, profile: ConnectionProfile):
        super().__init__(profile)
        self.serial = profile.adb_serial or self._single_device_serial()

    def _base(self) -> list[str]:
        return ["adb", "-P", str(self.profile.adb_port), "-s", self.serial]

    def _single_device_serial(self) -> str:
        output = CommandRunner.run(
            ["adb", "-P", str(self.profile.adb_port), "devices"],
            "ADB device discovery", 15, capture_stdout=True
        ).decode("utf-8", errors="replace")
        devices = [line.split()[0] for line in output.splitlines()[1:]
                   if len(line.split()) >= 2 and line.split()[1] == "device"]
        if len(devices) != 1:
            raise SessionError("ADB must report exactly one connected device, or profile a locally verified serial")
        return devices[0]

    def execute(self, remote_argv: list[str], label: str, timeout_s: int,
                capture_stdout: bool = False) -> bytes:
        return CommandRunner.run(
            self._base() + ["shell"] + remote_argv,
            label, timeout_s, capture_stdout
        )

    def upload(self, local: Path, remote: str, label: str, timeout_s: int) -> None:
        CommandRunner.run(self._base() + ["push", str(local), remote], label, timeout_s)

    def download(self, remote: str, local: Path, label: str, timeout_s: int) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        CommandRunner.run(self._base() + ["pull", remote, str(local)], label, timeout_s)


def transport_for(profile: ConnectionProfile) -> Transport:
    return SshTransport(profile) if profile.transport == "ssh" else AdbTransport(profile)


def run_directory(run_id: str) -> Path:
    return _path_under_root(RUNS / checked_run_id(run_id))


def _state_path(run_id: str) -> Path:
    return run_directory(run_id) / "session-budget.json"


def begin_session(run_id: str, budget_minutes: int) -> dict[str, object]:
    if not 1 <= budget_minutes <= 30:
        raise SessionError("P1 session budget must be in 1..30 minutes")
    root = run_directory(run_id)
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": checked_run_id(run_id),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "budget_minutes": budget_minutes,
        "stop_new_experiments_after_minutes": min(24, budget_minutes - 1),
        "connection_parameters_stored": False,
        "session_id_stored": False,
        "private_key_logged": False,
        "hardware_performance_claims": False,
    }
    _state_path(run_id).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (root / "raw").mkdir(exist_ok=True)
    return state


def load_budget(run_id: str) -> tuple[dict[str, Any], float]:
    try:
        state = json.loads(_state_path(run_id).read_text(encoding="utf-8"))
        started = datetime.fromisoformat(state["started_utc"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SessionError("run has no valid local session timer; start it after opening QDC") from error
    elapsed = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
    remaining = max(0.0, int(state["budget_minutes"]) * 60 - elapsed)
    return state, remaining


def _require_work_window(run_id: str, reserve_minutes: int = 5) -> float:
    state, remaining = load_budget(run_id)
    stop_after = int(state["stop_new_experiments_after_minutes"])
    elapsed = int(state["budget_minutes"]) * 60 - remaining
    if elapsed >= stop_after * 60 or remaining <= reserve_minutes * 60:
        raise SessionError("new experiments stopped to reserve time for result collection and checklist")
    return remaining


def local_connection_probe(run_id: str, profile_path: Path) -> dict[str, object]:
    profile = load_profile(profile_path)
    _require_work_window(run_id, reserve_minutes=5)
    transport = transport_for(profile)
    script = (
        "printf 'probe_mode=read_only\\n'; uname -a; id; "
        "printf 'machine='; uname -m; "
        "printf 'glibc='; getconf GNU_LIBC_VERSION 2>/dev/null || true; "
        "printf 'word_bits='; getconf LONG_BIT 2>/dev/null || true; "
        "for p in /tmp /var/tmp \"${HOME:-}\" /qdc/logs; do "
        "[ -e \"$p\" ] || continue; "
        "printf 'path=%s writable=' \"$p\"; "
        "if [ -w \"$p\" ]; then printf 'yes\\n'; else printf 'no\\n'; fi; "
        "df -Pk \"$p\" 2>/dev/null || true; done"
    )
    output = transport.execute(["sh", "-lc", script], "read-only connection probe", 45, True)
    root = run_directory(run_id)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "connection-probe.txt").write_bytes(output)
    return {"status": "passed", "run_id": run_id, "transport": profile.transport,
            "evidence": "raw/connection-probe.txt", "remote_mutations": False}


def _adapter(profile: ConnectionProfile) -> Transport:
    return transport_for(profile)


def deploy_probe(run_id: str, profile_path: Path, bundle_path: Path) -> dict[str, object]:
    remaining = _require_work_window(run_id, reserve_minutes=5)
    profile = load_profile(profile_path)
    if not profile.remote_root:
        raise SessionError("set remote_root only after the read-only probe confirms a writable path")
    bundle_path = _path_under_root(bundle_path)
    hexagonflow.read_and_verify_bundle(bundle_path)
    transport = _adapter(profile)
    remote_dir = f"{profile.remote_root}/{checked_run_id(run_id)}"
    log_dir = f"{remote_dir}/runtime-logs"
    timeout_s = max(10, min(240, int(remaining - 300)))
    transport.execute(["mkdir", "-p", remote_dir], "create temporary deployment directory", 30)
    transfer_start = time.monotonic()
    remote_bundle = f"{remote_dir}/first-probe.tar.gz"
    transport.upload(bundle_path, remote_bundle, "upload verified probe bundle", timeout_s)
    upload_seconds = max(0.001, time.monotonic() - transfer_start)
    bootstrap = (
        f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(remote_dir)} && "
        f"HEXAGONFLOW_REMOTE_LOG_ROOT={shlex.quote(log_dir)} "
        f"sh {shlex.quote(remote_dir + '/scripts/remote/bootstrap.sh')} "
        f"{shlex.quote(checked_run_id(run_id))}"
    )
    transport.execute(["sh", "-lc", bootstrap], "read-only device probe", timeout_s)
    result = {
        "run_id": checked_run_id(run_id),
        "transport": profile.transport,
        "bundle_path": bundle_path.relative_to(ROOT).as_posix(),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "upload_seconds": round(upload_seconds, 3),
        "upload_bytes_per_second": int(bundle_path.stat().st_size / upload_seconds),
        "hardware_performance_measurement": False,
        "connection_parameters_stored": False,
    }
    local = run_directory(run_id)
    (local / "deploy-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def transfer_smoke(run_id: str, profile_path: Path, size_mib: int = 1) -> dict[str, object]:
    remaining = _require_work_window(run_id, reserve_minutes=5)
    if not 1 <= size_mib <= 8:
        raise SessionError("transfer smoke payload must be 1..8 MiB")
    profile = load_profile(profile_path)
    if not profile.remote_root:
        raise SessionError("set remote_root only after the read-only probe confirms a writable path")
    local = run_directory(run_id)
    input_dir = local / "inputs"
    raw_dir = local / "raw"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = input_dir / "transfer-smoke.bin"
    returned = raw_dir / "transfer-smoke-return.bin"
    chunk = bytes(range(256)) * 4096
    with payload.open("wb") as stream:
        for _ in range(size_mib):
            stream.write(chunk)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    transport = _adapter(profile)
    remote_dir = f"{profile.remote_root}/{checked_run_id(run_id)}/transfer"
    transport.execute(["mkdir", "-p", remote_dir], "create transfer-test directory", 30)
    remote_file = f"{remote_dir}/transfer-smoke.bin"
    timeout_s = max(10, min(240, int(remaining - 300)))
    started = time.monotonic()
    transport.upload(payload, remote_file, "upload transfer smoke", timeout_s)
    upload_s = max(0.001, time.monotonic() - started)
    started = time.monotonic()
    transport.download(remote_file, returned, "download transfer smoke", timeout_s)
    download_s = max(0.001, time.monotonic() - started)
    returned_hash = hashlib.sha256(returned.read_bytes()).hexdigest()
    result = {
        "run_id": checked_run_id(run_id),
        "transport": profile.transport,
        "bytes": payload.stat().st_size,
        "sha256": digest,
        "download_sha256": returned_hash,
        "integrity_passed": digest == returned_hash,
        "upload_seconds": round(upload_s, 3),
        "download_seconds": round(download_s, 3),
        "upload_bytes_per_second": int(payload.stat().st_size / upload_s),
        "download_bytes_per_second": int(payload.stat().st_size / download_s),
        "hardware_performance_measurement": False,
    }
    (local / "transfer-test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if digest != returned_hash:
        raise SessionError("transfer smoke SHA256 mismatch; local evidence was retained")
    return result


def run_arm64_smoke(run_id: str, profile_path: Path) -> dict[str, object]:
    remaining = _require_work_window(run_id, reserve_minutes=5)
    profile = load_profile(profile_path)
    if not profile.remote_root:
        raise SessionError("set remote_root only after the read-only probe confirms a writable path")
    transport = _adapter(profile)
    remote_dir = f"{profile.remote_root}/{checked_run_id(run_id)}"
    log_dir = f"{remote_dir}/runtime-logs"
    command = (
        f"HEXAGONFLOW_REMOTE_LOG_ROOT={shlex.quote(log_dir)} "
        f"sh {shlex.quote(remote_dir + '/scripts/remote/run_experiment.sh')} "
        f"{shlex.quote(checked_run_id(run_id))} 120 sh "
        f"{shlex.quote(remote_dir + '/scripts/remote/arm64_smoke.sh')} "
        f"{shlex.quote(checked_run_id(run_id))}"
    )
    transport.execute(["sh", "-lc", command], "bounded ARM64 ABI smoke test",
                      max(10, min(150, int(remaining - 300))))
    result = {"run_id": checked_run_id(run_id), "transport": profile.transport,
              "status": "completed_or_skipped", "remote_status_file": "runtime-logs/arm64-smoke-status.txt",
              "hardware_performance_measurement": False}
    root = run_directory(run_id)
    (root / "arm64-smoke-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _digest_tree(directory: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not directory.exists():
        return entries
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        entries.append({"path": path.relative_to(directory).as_posix(), "bytes": size,
                        "sha256": digest.hexdigest()})
    return entries


def collect_results(run_id: str, profile_path: Path) -> dict[str, object]:
    profile = load_profile(profile_path)
    if not profile.remote_root:
        raise SessionError("set remote_root only after the read-only probe confirms a writable path")
    root = run_directory(run_id)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    transport = _adapter(profile)
    remote_base = f"{profile.remote_root}/{checked_run_id(run_id)}/runtime-logs"
    copied: list[str] = []
    missing: list[str] = []
    for label, remote, local_name in (
        ("remote run logs", remote_base, "remote-logs"),
        ("QDC log backup", f"/qdc/logs/hexagonflow/{checked_run_id(run_id)}", "qdc-logs"),
    ):
        try:
            transport.download(remote, raw / local_name, label, 90)
            copied.append(local_name)
        except SessionError:
            # QDC log backup is best effort; immediate host-side retrieval remains primary.
            missing.append(local_name)
    entries = _digest_tree(raw)
    entry_names = {entry["path"] for entry in entries}
    required_remote_logs = {
        "remote-logs/bootstrap-status.txt",
        "remote-logs/device-probe.txt",
    }
    collection_complete = required_remote_logs.issubset(entry_names)
    manifest = {
        "schema_version": 1,
        "project": "HexagonFlow",
        "run_id": checked_run_id(run_id),
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "copied_sources": copied,
        "missing_sources": missing,
        "connection_parameters_stored": False,
        "private_key_logged": False,
        "files": entries,
    }
    (root / "results-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "collection-status.json").write_text(
        json.dumps({"complete": collection_complete, "file_count": len(entries),
                    "missing_sources": missing}, indent=2) + "\n", encoding="utf-8"
    )
    return {"run_id": checked_run_id(run_id), "files": len(entries),
            "complete": collection_complete,
            "copied_sources": copied, "missing_sources": missing,
            "results_manifest": str(root / "results-manifest.json")}


def checklist(run_id: str) -> dict[str, object]:
    root = run_directory(run_id)
    raw = root / "raw"
    requirements = {
        "session_budget": _state_path(run_id).is_file(),
        "connection_probe": (raw / "connection-probe.txt").is_file(),
        "transfer_integrity": False,
        "deployment_result": (root / "deploy-result.json").is_file(),
        "arm64_smoke_recorded": (root / "arm64-smoke-result.json").is_file(),
        "results_manifest": (root / "results-manifest.json").is_file(),
        "collection_status": (root / "collection-status.json").is_file(),
    }
    transfer_path = root / "transfer-test.json"
    if transfer_path.is_file():
        transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
        requirements["transfer_integrity"] = (
            transfer.get("integrity_passed") is True
            and transfer.get("bytes") == 1024 * 1024
            and transfer.get("sha256") == transfer.get("download_sha256")
        )
    if requirements["results_manifest"]:
        manifest = json.loads((root / "results-manifest.json").read_text(encoding="utf-8"))
        files = manifest.get("files", [])
        valid = isinstance(files, list) and bool(files)
        for entry in files if isinstance(files, list) else []:
            relative = Path(entry.get("path", "")) if isinstance(entry, dict) else Path("")
            if relative.is_absolute() or not relative.parts or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                requirements["results_manifest"] = False
                break
            path = (raw / relative).resolve()
            try:
                path.relative_to(raw.resolve())
            except ValueError:
                requirements["results_manifest"] = False
                break
            if not path.is_file() or path.stat().st_size != entry.get("bytes"):
                requirements["results_manifest"] = False
                break
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
                requirements["results_manifest"] = False
                break
        if requirements["results_manifest"]:
            requirements["results_manifest"] = valid and any(
                entry.get("path") == "remote-logs/bootstrap-status.txt"
                for entry in files
            ) and any(
                entry.get("path") == "remote-logs/device-probe.txt"
                for entry in files
            )
    if requirements["collection_status"]:
        collection = json.loads((root / "collection-status.json").read_text(encoding="utf-8"))
        requirements["collection_status"] = collection.get("complete") is True
    if requirements["deployment_result"]:
        deployment = json.loads((root / "deploy-result.json").read_text(encoding="utf-8"))
        try:
            bundle = hexagonflow._workspace_path(ROOT / deployment["bundle_path"])
            requirements["deployment_result"] = (
                isinstance(deployment.get("bundle_sha256"), str)
                and len(deployment["bundle_sha256"]) == 64
                and bundle.is_file()
                and hashlib.sha256(bundle.read_bytes()).hexdigest() == deployment["bundle_sha256"]
            )
        except (KeyError, TypeError, hexagonflow.BundleError):
            requirements["deployment_result"] = False
    return {"run_id": checked_run_id(run_id), "ready_to_close": all(requirements.values()),
            "checks": requirements, "connection_parameters_stored": False,
            "private_key_logged": False}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdc_session.py")
    sub = parser.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("begin", help="start a local timer after manual QDC session start")
    begin.add_argument("--run-id", required=True)
    begin.add_argument("--budget-minutes", type=int, default=30)
    probe = sub.add_parser("connection-probe", help="run a read-only SSH/ADB connection check")
    probe.add_argument("--run-id", required=True)
    probe.add_argument("--profile", type=Path, required=True)
    deploy = sub.add_parser("deploy-probe", help="upload and execute the minimal probe bundle")
    deploy.add_argument("--run-id", required=True)
    deploy.add_argument("--profile", type=Path, required=True)
    deploy.add_argument("--bundle", type=Path, default=ROOT / "build" / "first-probe.tar.gz")
    transfer = sub.add_parser("transfer-test", help="round-trip a small SHA256-verified file")
    transfer.add_argument("--run-id", required=True)
    transfer.add_argument("--profile", type=Path, required=True)
    transfer.add_argument("--size-mib", type=int, default=1)
    smoke = sub.add_parser("arm64-smoke", help="run the bundled static ABI test or compile it when a target compiler exists")
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--profile", type=Path, required=True)
    experiment = sub.add_parser(
        "run-experiment", help="deploy pinned artifacts, run one bounded plan, and collect its logs")
    experiment.add_argument("--run-id", required=True, help="new, unique experiment run ID")
    experiment.add_argument("--session-run-id", required=True,
                            help="local ID for the freshly observed active QDC session budget")
    experiment.add_argument("--profile", type=Path, required=True,
                            help="ignored local profile built from the current Connect page")
    experiment.add_argument("--plan", type=Path, required=True,
                            help="pinned plan under ignored .local/experiment-plans/")
    collect = sub.add_parser("collect", help="pull per-run logs and hash them locally")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--profile", type=Path, required=True)
    check = sub.add_parser("checklist", help="verify local evidence before closing QDC")
    check.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "begin":
            result = begin_session(args.run_id, args.budget_minutes)
        elif args.command == "connection-probe":
            result = local_connection_probe(args.run_id, args.profile)
        elif args.command == "deploy-probe":
            result = deploy_probe(args.run_id, args.profile, args.bundle)
        elif args.command == "transfer-test":
            result = transfer_smoke(args.run_id, args.profile, args.size_mib)
        elif args.command == "arm64-smoke":
            result = run_arm64_smoke(args.run_id, args.profile)
        elif args.command == "run-experiment":
            from tools import experiment_runner
            result = experiment_runner.run_experiment(
                args.run_id, args.session_run_id, args.profile, args.plan)
        elif args.command == "collect":
            result = collect_results(args.run_id, args.profile)
        else:
            result = checklist(args.run_id)
    except (SessionError, hexagonflow.BundleError, OSError, ValueError) as error:
        # Do not print subprocess argv, Connect text, host, port, username, or key paths.
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "run-experiment" and result.get("status") != "completed":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
