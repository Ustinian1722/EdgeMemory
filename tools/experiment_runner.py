"""Execute a pinned, bounded experiment on a user-managed QDC session."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from tools import qdc_session, run_manifest, session_budget


ROOT = qdc_session.ROOT
PLAN_ROOT = (ROOT / ".local" / "experiment-plans").resolve()
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"\{artifact:([^{}]+)\}")
FORBIDDEN_KEYS = {"password", "api_key", "session_id", "private_key", "key_content"}


class ExperimentPlanError(qdc_session.SessionError):
    pass


@dataclass(frozen=True)
class Artifact:
    local_path: Path
    destination: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    program: str
    arguments: tuple[str, ...]
    working_directory: str
    timeout_seconds: int
    transfer_timeout_seconds: int
    expected_exit_codes: tuple[int, ...]
    artifacts: tuple[Artifact, ...]
    plan_sha256: str


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_KEYS or _contains_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExperimentPlanError(f"invalid {label}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        if value != "." or label != "working_directory":
            raise ExperimentPlanError(f"invalid {label}")
    if value != "." and not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                                  for part in path.parts):
        raise ExperimentPlanError(f"invalid {label}")
    return value


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def load_plan(path: Path) -> ExperimentPlan:
    resolved = path.resolve()
    if not resolved.is_relative_to(PLAN_ROOT):
        raise ExperimentPlanError("experiment plans must stay under ignored .local/experiment-plans")
    try:
        raw_bytes = resolved.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentPlanError("experiment plan is missing or malformed") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ExperimentPlanError("unsupported experiment plan schema")
    if _contains_forbidden_key(raw):
        raise ExperimentPlanError("experiment plans must not contain credentials or session IDs")
    allowed = {"schema_version", "plan_id", "program", "arguments", "working_directory",
               "timeout_seconds", "transfer_timeout_seconds", "expected_exit_codes", "artifacts"}
    if set(raw) - allowed:
        raise ExperimentPlanError("experiment plan contains unknown fields")

    plan_id = raw.get("plan_id")
    if not isinstance(plan_id, str) or not SAFE_TOKEN.fullmatch(plan_id):
        raise ExperimentPlanError("invalid experiment plan ID")
    program = _relative_path(raw.get("program"), "program")
    working_directory = _relative_path(raw.get("working_directory", "."),
                                       "working_directory")

    def bounded_seconds(value: object, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 900:
            raise ExperimentPlanError(f"{field} must be an integer in 1..900 seconds")
        return value

    timeout_seconds = bounded_seconds(raw.get("timeout_seconds"), "timeout_seconds")
    transfer_timeout_seconds = bounded_seconds(
        raw.get("transfer_timeout_seconds", 300), "transfer_timeout_seconds")

    raw_arguments = raw.get("arguments", [])
    if not isinstance(raw_arguments, list) or len(raw_arguments) > 256:
        raise ExperimentPlanError("arguments must be an array of at most 256 strings")
    arguments: list[str] = []
    for argument in raw_arguments:
        if not isinstance(argument, str) or len(argument) > 4096 or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in argument
        ):
            raise ExperimentPlanError("arguments must be bounded printable strings")
        arguments.append(argument)

    raw_codes = raw.get("expected_exit_codes", [0])
    if not isinstance(raw_codes, list) or not raw_codes or len(raw_codes) > 4 or any(
        isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255
        for code in raw_codes
    ):
        raise ExperimentPlanError("expected_exit_codes must contain one to four byte values")

    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, list) or not 1 <= len(raw_artifacts) <= 32:
        raise ExperimentPlanError("artifacts must contain 1..32 pinned files")
    artifacts: list[Artifact] = []
    destinations: set[str] = set()
    for item in raw_artifacts:
        if not isinstance(item, dict) or set(item) != {"source", "destination", "sha256"}:
            raise ExperimentPlanError("each artifact requires source, destination, and sha256")
        source = item["source"]
        if not isinstance(source, str) or not source or Path(source).is_absolute() \
                or "\\" in source:
            raise ExperimentPlanError("artifact source must be a workspace-relative path")
        local = (ROOT / source).resolve()
        if not local.is_relative_to(ROOT) or not local.is_file() or local.is_symlink():
            raise ExperimentPlanError("artifact source must be a regular file under D: workspace")
        destination = _relative_path(item["destination"], "artifact destination")
        if destination in destinations or destination == "scripts/remote/run_experiment.sh":
            raise ExperimentPlanError("duplicate or reserved artifact destination")
        destinations.add(destination)
        digest = item["sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ExperimentPlanError("artifact SHA256 must be lowercase hexadecimal")
        actual, byte_count = _hash_file(local)
        if actual != digest:
            raise ExperimentPlanError("artifact SHA256 differs from the pinned plan")
        if local.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise ExperimentPlanError("credential files cannot be experiment artifacts")
        artifacts.append(Artifact(local, destination, digest, byte_count))

    if program not in destinations:
        raise ExperimentPlanError("program must name one of the pinned artifact destinations")
    for argument in arguments:
        for match in PLACEHOLDER.finditer(argument):
            _relative_path(match.group(1), "artifact placeholder")
            if match.group(1) not in destinations:
                raise ExperimentPlanError("argument refers to an artifact not in the plan")
        without_artifact_tokens = PLACEHOLDER.sub("", argument)
        without_known_tokens = re.sub(
            r"\{(?:run_id|session_dir|run_log_dir)\}", "", without_artifact_tokens)
        if "{" in without_known_tokens or "}" in without_known_tokens:
            raise ExperimentPlanError("argument contains an unsupported placeholder")

    return ExperimentPlan(
        plan_id, program, tuple(arguments), working_directory, timeout_seconds,
        transfer_timeout_seconds, tuple(raw_codes), tuple(artifacts),
        hashlib.sha256(raw_bytes).hexdigest())


def _remote_hash(transport: qdc_session.Transport, remote_path: str,
                 timeout_seconds: int) -> str | None:
    try:
        output = transport.execute(["sha256sum", "--", remote_path],
                                   "verify remote artifact hash", timeout_seconds, True)
    except qdc_session.SessionError:
        return None
    fields = output.decode("ascii", errors="ignore").split()
    return fields[0] if fields and SHA256.fullmatch(fields[0]) else None


def _ensure_remote_file(transport: qdc_session.Transport, local_path: Path,
                        remote_path: str, expected_hash: str,
                        timeout_seconds: int, run_id: str) -> bool:
    if _remote_hash(transport, remote_path, min(30, timeout_seconds)) == expected_hash:
        return False
    parent = str(PurePosixPath(remote_path).parent)
    transport.execute(["mkdir", "-p", parent], "prepare remote artifact directory", 30)
    temporary = f"{remote_path}.upload-{run_id}"
    transport.upload(local_path, temporary, "upload pinned experiment artifact", timeout_seconds)
    if _remote_hash(transport, temporary, min(30, timeout_seconds)) != expected_hash:
        try:
            transport.execute(["rm", "-f", "--", temporary],
                              "remove incomplete experiment upload", 15)
        except qdc_session.SessionError:
            pass
        raise qdc_session.SessionError("uploaded experiment artifact failed SHA256 verification")
    transport.execute(["mv", "-f", "--", temporary, remote_path],
                      "install verified experiment artifact", 30)
    return True


def _expand_argument(argument: str, *, run_id: str, session_dir: str,
                     run_log_dir: str, payload_dir: str) -> str:
    value = argument.replace("{run_id}", run_id)
    value = value.replace("{session_dir}", session_dir)
    value = value.replace("{run_log_dir}", run_log_dir)
    value = PLACEHOLDER.sub(lambda match: f"{payload_dir}/{match.group(1)}", value)
    if "{" in value or "}" in value:
        raise ExperimentPlanError("unsupported experiment argument placeholder")
    return value


def _collected_log_state(raw_root: Path) -> tuple[bool, int | None]:
    files = [path for path in (raw_root / "remote-logs").rglob("*") if path.is_file()] \
        if (raw_root / "remote-logs").exists() else []
    names = {path.name for path in files}
    metadata = next((path for path in files if path.name == "experiment-metadata.txt"), None)
    has_output = {"experiment.stdout.log", "experiment.stderr.log"}.issubset(names) \
        or "experiment-error.log" in names
    exit_code: int | None = None
    if metadata:
        match = re.search(r"(?m)^exit_code=(\d+)\s*$",
                          metadata.read_text(encoding="utf-8", errors="replace"))
        if match:
            exit_code = int(match.group(1))
    return metadata is not None and has_output, exit_code


def run_experiment(run_id: str, session_run_id: str, profile_path: Path,
                   plan_path: Path) -> dict[str, object]:
    """Deploy pinned artifacts, execute one bounded trial, then recover its logs."""
    run_id = qdc_session.checked_run_id(run_id)
    session_run_id = qdc_session.checked_run_id(session_run_id)
    if run_id == session_run_id:
        raise qdc_session.SessionError("experiment and session run IDs must be distinct")

    plan = load_plan(plan_path)
    profile_resolved = qdc_session._path_under_root(profile_path)
    if not profile_resolved.is_relative_to((ROOT / ".local").resolve()):
        raise qdc_session.SessionError("QDC connection profile must stay under ignored .local/")
    profile = qdc_session.load_profile(profile_resolved)
    if not profile.remote_root:
        raise qdc_session.SessionError("profile requires a verified writable remote_root")

    budget = session_budget.status(session_run_id)
    remaining = budget["conservative_remaining_seconds"]
    required = (plan.timeout_seconds
                + (len(plan.artifacts) + 1) * plan.transfer_timeout_seconds
                + 180 + 45 * 60)
    if budget["phase"] != "research_window" or remaining <= required:
        raise qdc_session.SessionError(
            "observed QDC budget cannot cover this experiment, artifact transfers, and recovery reserve")

    root = qdc_session.run_directory(run_id)
    if root.exists() and any(root.iterdir()):
        raise qdc_session.SessionError("experiment run directory already contains data")
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "raw"
    raw_root.mkdir(exist_ok=True)

    command_spec = {"program": plan.program, "arguments": list(plan.arguments)}
    config = {
        "schema_version": 1,
        "run_id": run_id,
        "session_run_id": session_run_id,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "command_spec_sha256": hashlib.sha256(json.dumps(
            command_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "timeout_seconds": plan.timeout_seconds,
        "observed_budget_seconds_at_start": remaining,
        "artifacts": [{"destination": item.destination, "sha256": item.sha256,
                       "bytes": item.byte_count} for item in plan.artifacts],
        "connection_parameters_stored": False,
        "private_key_logged": False,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    transport = qdc_session.transport_for(profile)
    remote_session_dir = f"{profile.remote_root}/{session_run_id}"
    payload_dir = f"{remote_session_dir}/payload"
    runner_local = ROOT / "scripts" / "remote" / "run_experiment.sh"
    runner_hash, runner_size = _hash_file(runner_local)
    runner_remote = f"{payload_dir}/scripts/remote/run_experiment.sh"
    uploaded: list[str] = []
    remote_started = False
    preparation_error = False
    collect_error = False
    execution_returned_zero = False
    remote_log_dir = f"{remote_session_dir}/runtime-logs/{run_id}"
    qdc_log_dir = f"/qdc/logs/hexagonflow/{run_id}"
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        transport.execute(["mkdir", "-p", payload_dir], "prepare temporary experiment workspace", 30)
        for artifact in plan.artifacts:
            destination = f"{payload_dir}/{artifact.destination}"
            if _ensure_remote_file(transport, artifact.local_path, destination,
                                   artifact.sha256, plan.transfer_timeout_seconds, run_id):
                uploaded.append(artifact.destination)
        if _ensure_remote_file(transport, runner_local, runner_remote, runner_hash,
                               plan.transfer_timeout_seconds, run_id):
            uploaded.append("scripts/remote/run_experiment.sh")

        program_remote = f"{payload_dir}/{plan.program}"
        transport.execute(["chmod", "u+x", "--", program_remote],
                          "prepare experiment executable", 30)
        # A unique run directory prevents stale remote files from contaminating results.
        create_logs = (f"test ! -e {shlex.quote(remote_log_dir)} && "
                       f"mkdir -p {shlex.quote(remote_log_dir)}")
        transport.execute(["sh", "-lc", create_logs], "create isolated remote run directory", 30)
        args = [
            _expand_argument(value, run_id=run_id, session_dir=remote_session_dir,
                             run_log_dir=remote_log_dir, payload_dir=payload_dir)
            for value in plan.arguments
        ]
        executable_args = [program_remote, *args]
        wrapper_args = ["sh", runner_remote, run_id, str(plan.timeout_seconds),
                        *executable_args]
        command = (
            f"cd {shlex.quote(payload_dir)} && "
            f"HEXAGONFLOW_REMOTE_LOG_ROOT={shlex.quote(remote_log_dir)} "
            f"HEXAGONFLOW_QDC_LOG_ROOT={shlex.quote(qdc_log_dir)} "
            f"{shlex.join(wrapper_args)}"
        )
        remote_started = True
        try:
            transport.execute(["sh", "-lc", command], "bounded predefined experiment",
                              plan.timeout_seconds + 60)
            execution_returned_zero = True
        except qdc_session.SessionError:
            # Nonzero experiment exit and transport timeout both leave recoverable logs.
            pass
    except (qdc_session.SessionError, OSError):
        preparation_error = True
    finally:
        if remote_started:
            try:
                transport.download(remote_log_dir, raw_root / "remote-logs",
                                   "collect experiment logs", 120)
            except qdc_session.SessionError:
                collect_error = True
            try:
                transport.download(qdc_log_dir, raw_root / "qdc-logs",
                                   "collect QDC log backup", 90)
            except qdc_session.SessionError:
                pass

    collection_complete, remote_exit_code = _collected_log_state(raw_root)
    expected_exit = remote_exit_code in plan.expected_exit_codes
    if not collection_complete:
        collect_error = True
    if not execution_returned_zero and remote_exit_code in plan.expected_exit_codes:
        # The wrapper may return an expected nonzero code, which SSH reports as failure.
        expected_exit = True
    if not remote_started:
        status = "preparation_failed"
    elif not collection_complete:
        status = "collection_incomplete"
    elif not expected_exit or preparation_error:
        status = "failed"
    else:
        status = "completed"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "status": status,
        "remote_exit_code": remote_exit_code,
        "execution_transport_returned_zero": execution_returned_zero,
        "collection_complete": collection_complete,
        "collection_error": collect_error,
        "artifact_count": len(plan.artifacts) + 1,
        "uploaded_artifacts": uploaded,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "connection_parameters_stored": False,
        "private_key_logged": False,
        "device_execution": True,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = run_manifest.create(root)
    return {"run_id": run_id, "plan_id": plan.plan_id, "status": status,
            "remote_exit_code": remote_exit_code,
            "collection_complete": collection_complete,
            "verified_manifest_files": len(manifest["files"]),
            "result_directory": root.relative_to(ROOT).as_posix()}
