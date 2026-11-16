#!/usr/bin/env python3
"""Local-only bundle, integrity, recovery, and QDC Connect inspection tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = (
    "configs/first_probe.json",
    "scripts/remote/bootstrap.sh",
    "scripts/remote/device_probe.sh",
    "scripts/remote/arm64_smoke.sh",
    "scripts/remote/run_experiment.sh",
    "scripts/remote/session_checklist.sh",
    "src/arm64_smoke.cpp",
)
OPTIONAL_BUNDLE_FILES = ("build/arm64-smoke",)


class BundleError(RuntimeError):
    """A bundle is malformed, incomplete, or fails integrity checks."""


def _workspace_path(path: Path) -> Path:
    """Resolve an output path and keep all generated project data on D:."""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise BundleError("project outputs must stay inside the D: workspace") from error
    return resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise BundleError(f"unsafe bundle path: {name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"unsafe bundle path: {name!r}")
    if "\\" in name or ":" in name:
        raise BundleError(f"unsafe bundle path: {name!r}")
    return path


def _file_manifest() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for relative in (*BUNDLE_FILES, *(path for path in OPTIONAL_BUNDLE_FILES
                                       if (ROOT / path).is_file())):
        source = ROOT / relative
        if not source.is_file():
            raise BundleError(f"required bundle input missing: {relative}")
        data = source.read_bytes()
        entries.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "mode": 0o755 if source.suffix == ".sh" or relative == "build/arm64-smoke" else 0o644,
            }
        )
    return entries


def build_bundle(output_path: Path) -> dict[str, object]:
    output_path = _workspace_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = _file_manifest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": "HexagonFlow",
        "bundle_kind": "read-only-first-probe",
        "hardware_measurements": False,
        "entries": entries,
    }
    manifest_data = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in entries
    ).encode("utf-8")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        import gzip

        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for entry in entries:
                    name = str(entry["path"])
                    source = ROOT / name
                    data = source.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = int(entry["mode"])
                    import io

                    archive.addfile(info, io.BytesIO(data))
                for name, data in (
                    ("manifest.json", manifest_data),
                    ("manifest.sha256", checksums),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    import io

                    archive.addfile(info, io.BytesIO(data))
    os.replace(temporary, output_path)
    return {
        "bundle": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "entries": len(entries),
    }


@dataclass(frozen=True)
class VerifiedBundle:
    manifest: dict[str, object]
    payloads: dict[str, bytes]
    metadata: dict[str, bytes]


def read_and_verify_bundle(bundle_path: Path) -> VerifiedBundle:
    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = archive.getmembers()
            seen: set[str] = set()
            contents: dict[str, bytes] = {}
            for member in members:
                _safe_member_name(member.name)
                if member.name in seen:
                    raise BundleError(f"duplicate bundle path: {member.name}")
                seen.add(member.name)
                if not member.isfile():
                    raise BundleError(f"unsupported bundle entry type: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise BundleError(f"cannot read bundle entry: {member.name}")
                contents[member.name] = handle.read()
    except (OSError, tarfile.TarError) as error:
        raise BundleError(f"cannot read bundle: {error}") from error

    try:
        manifest = json.loads(contents.pop("manifest.json"))
        checksums = contents.pop("manifest.sha256").decode("utf-8")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError("manifest files are missing or malformed") from error
    if manifest.get("schema_version") != 1:
        raise BundleError("unsupported manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise BundleError("manifest entries must be an array")

    expected_lines: list[str] = []
    payloads: dict[str, bytes] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise BundleError("manifest entry must be an object")
        name = str(entry.get("path", ""))
        _safe_member_name(name)
        try:
            data = contents[name]
        except KeyError as error:
            raise BundleError(f"payload missing: {name}") from error
        if len(data) != int(entry.get("bytes", -1)):
            raise BundleError(f"payload size mismatch: {name}")
        digest = sha256_bytes(data)
        if digest != entry.get("sha256"):
            raise BundleError(f"payload checksum mismatch: {name}")
        payloads[name] = data
        expected_lines.append(f"{digest}  {name}\n")

    expected_names = {str(entry["path"]) for entry in entries}
    actual_payload_names = set(contents)
    if actual_payload_names != expected_names:
        raise BundleError("payload file set does not match manifest")
    if checksums != "".join(expected_lines):
        raise BundleError("sha256sum file does not match manifest")
    return VerifiedBundle(
        manifest=manifest,
        payloads=payloads,
        metadata={"manifest.json": json.dumps(manifest, indent=2).encode() + b"\n",
                  "manifest.sha256": checksums.encode()},
    )


def restore_bundle(bundle_path: Path, destination: Path) -> dict[str, object]:
    verified = read_and_verify_bundle(bundle_path)
    destination = _workspace_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise BundleError(f"restore destination must be empty: {destination}")

    try:
        for relative, data in verified.payloads.items():
            path = _safe_member_name(relative)
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.suffix == ".sh":
                target.chmod(0o755)
        for name, data in verified.metadata.items():
            (destination / name).write_bytes(data)
    except OSError as error:
        evidence = destination / "restore-error.json"
        evidence.write_text(
            json.dumps({"error": str(error), "bundle": str(bundle_path)}) + "\n",
            encoding="utf-8",
        )
        raise
    return {
        "destination": str(destination),
        "restored_files": len(verified.payloads),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }


def build_artifact_manifest(spec_path: Path, output_path: Path) -> dict[str, object]:
    """Hash only explicit, project-local artifacts; never records key material."""
    spec_path = spec_path.resolve()
    try:
        spec_path.relative_to(ROOT)
    except ValueError as error:
        raise BundleError("artifact spec must be inside the D: workspace") from error
    output_path = _workspace_path(output_path)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError("artifact spec is missing or malformed") from error
    if spec.get("schema_version") != 1 or not isinstance(spec.get("artifacts"), list):
        raise BundleError("artifact spec requires schema_version=1 and an artifacts array")

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in spec["artifacts"]:
        if not isinstance(item, dict):
            raise BundleError("artifact entry must be an object")
        artifact_id = str(item.get("id", ""))
        relative = str(item.get("path", ""))
        path = _safe_member_name(relative)
        if not artifact_id or artifact_id in seen:
            raise BundleError("artifact ids must be non-empty and unique")
        seen.add(artifact_id)
        candidate = ROOT.joinpath(*path.parts).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError as error:
            raise BundleError("artifact path resolves outside the D: workspace") from error
        if candidate.suffix.lower() in {".pem", ".key", ".p12", ".pfx"} or candidate.name.lower() in {
            "id_rsa", "id_ed25519", "id_ecdsa"
        }:
            raise BundleError("private-key material cannot be listed in an artifact manifest")
        if not candidate.is_file():
            raise BundleError(f"artifact file is missing: {relative}")

        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        entries.append({
            "id": artifact_id,
            "artifact_type": str(item.get("artifact_type", "unspecified")),
            "path": relative,
            "version": str(item.get("version", "unknown")),
            "source": str(item.get("source", "local")),
            "license": str(item.get("license", "unknown")),
            "bytes": size,
            "sha256": digest.hexdigest(),
        })

    manifest = {
        "schema_version": 1,
        "project": "HexagonFlow",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": str(spec.get("source_commit", "unknown")),
        "hardware_measurements": False,
        "artifacts": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": str(output_path),
        "artifacts": len(entries),
        "hardware_measurements": False,
    }


def inspect_connect_text(command_text: str) -> dict[str, object]:
    """Classify only; never return the command, host, key path, or session id."""
    tokens = shlex.split(command_text, posix=False)
    ssh_forward = False
    adb_forward = False
    ssh_commands = 0
    adb_commands = 0
    adb_port_option = False
    for index, token in enumerate(tokens):
        if token.lower().endswith("ssh") or token.lower() == "ssh":
            ssh_commands += 1
        if token.lower().endswith("adb") or token.lower() == "adb":
            adb_commands += 1
            if index + 1 < len(tokens) and tokens[index + 1] == "-P":
                adb_port_option = True
        if token == "-L" and index + 1 < len(tokens):
            mapping = tokens[index + 1]
        elif token.startswith("-L") and len(token) > 2:
            mapping = token[2:]
        else:
            continue
        fields = mapping.split(":")
        if len(fields) < 3:
            continue
        remote_port = fields[-1]
        if remote_port == "22":
            ssh_forward = True
        if remote_port == "5037":
            adb_forward = True

    if ssh_forward and adb_forward:
        transport = "ssh_tunnel_with_ssh_and_adb_forwards"
    elif ssh_forward:
        transport = "ssh_tunnel"
    elif adb_forward:
        transport = "ssh_tunnel_for_adb"
    elif adb_commands and adb_port_option:
        transport = "adb_custom_port"
    elif ssh_commands:
        transport = "ssh_command_present_unclassified"
    elif adb_commands:
        transport = "adb_command_present_unclassified"
    else:
        transport = "unknown"
    return {
        "transport": transport,
        "ssh_command_present": ssh_commands > 0,
        "ssh_forward_present": ssh_forward,
        "adb_forward_present": adb_forward,
        "adb_command_present": adb_commands > 0,
        "adb_custom_port_present": adb_port_option,
        "contains_private_key_argument": any(
            token == "-i" for token in tokens
        ),
        "command_saved": False,
        "hardware_connected": False,
    }


def _run_recovery_demo(run_root: Path) -> dict[str, object]:
    run_root = _workspace_path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    bundle_path = run_root / "first-probe.tar.gz"
    bundle_info = build_bundle(bundle_path)
    verified = read_and_verify_bundle(bundle_path)
    if not shutil.which("sh"):
        raise RuntimeError("sh is required for the local Linux recovery rehearsal")

    runs: list[dict[str, object]] = []
    for run_id in ("recovery-1", "recovery-2"):
        device_root = run_root / run_id / "empty-device-root"
        device_root.mkdir(parents=True, exist_ok=True)
        if any(device_root.iterdir()):
            raise BundleError("recovery rehearsal directory is not empty")
        restore_info = restore_bundle(bundle_path, device_root)
        scripts = device_root / "scripts" / "remote"
        remote_logs = run_root / run_id / "remote-tmp"
        qdc_logs = run_root / run_id / "qdc-logs-simulation"
        env = os.environ.copy()
        env["HEXAGONFLOW_REMOTE_LOG_ROOT"] = str(remote_logs)
        env["HEXAGONFLOW_QDC_LOG_ROOT"] = str(qdc_logs)
        completed = subprocess.run(
            ["sh", str(scripts / "bootstrap.sh"), run_id],
            cwd=device_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        evidence_dir = run_root / run_id / "local-evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "bootstrap.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (evidence_dir / "bootstrap.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        result = {
            "run_id": run_id,
            "environment": "local-wsl-recovery-simulation",
            "device_root": str(device_root),
            "exit_code": completed.returncode,
            "hardware_measurements": False,
            "bundle_sha256": restore_info["bundle_sha256"],
            "restored_files": restore_info["restored_files"],
        }
        (evidence_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"recovery bootstrap failed for {run_id}; evidence retained at "
                f"{evidence_dir}"
            )
        if not (remote_logs / "device-probe.txt").is_file():
            raise RuntimeError(f"probe evidence missing for {run_id}")
        if not (qdc_logs / "bootstrap-status.txt").is_file():
            raise RuntimeError(f"backup evidence missing for {run_id}")
        runs.append(result)

    failure_id = "recovery-failure"
    failure_root = run_root / failure_id / "empty-device-root"
    failure_root.mkdir(parents=True, exist_ok=True)
    restore_bundle(bundle_path, failure_root)
    failure_dir = run_root / failure_id / "local-evidence"
    failure_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HEXAGONFLOW_REMOTE_LOG_ROOT"] = str(run_root / failure_id / "remote-tmp")
    env["HEXAGONFLOW_QDC_LOG_ROOT"] = str(run_root / failure_id / "qdc-logs-simulation")
    failed = subprocess.run(
        ["sh", str(failure_root / "scripts/remote/run_experiment.sh"),
         failure_id, "10", "sh", "-c", "printf 'failure fixture reached\\n'; exit 7"],
        cwd=failure_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    failure_evidence = {
        "run_id": failure_id,
        "environment": "local-wsl-recovery-simulation",
        "expected_exit_code": 7,
        "observed_exit_code": failed.returncode,
        "hardware_measurements": False,
    }
    (failure_dir / "stdout.log").write_text(failed.stdout, encoding="utf-8")
    (failure_dir / "stderr.log").write_text(failed.stderr, encoding="utf-8")
    failure_log = run_root / failure_id / "remote-tmp" / "experiment.stdout.log"
    if failure_log.exists():
        shutil.copy2(failure_log, failure_dir / "experiment.stdout.log")
    (failure_dir / "result.json").write_text(
        json.dumps(failure_evidence, indent=2) + "\n", encoding="utf-8"
    )
    if failed.returncode != 7 or not (failure_dir / "result.json").is_file():
        raise RuntimeError(f"failure evidence was not retained at {failure_dir}")

    return {
        "status": "passed",
        "bundle": bundle_info,
        "recovery_runs": runs,
        "failure_evidence": str(failure_dir),
        "failure_exit_code": failed.returncode,
        "hardware_measurements": False,
        "output_directory": str(run_root),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hexagonflow.py")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle", help="build the minimal probe bundle")
    bundle.add_argument(
        "--output", type=Path, default=ROOT / "build" / "first-probe.tar.gz"
    )
    verify = commands.add_parser("verify", help="verify a bundle and its hashes")
    verify.add_argument("bundle", type=Path)
    restore = commands.add_parser("restore", help="restore a bundle to an empty dir")
    restore.add_argument("bundle", type=Path)
    restore.add_argument("destination", type=Path)
    demo = commands.add_parser(
        "recovery-demo", help="exercise two restores and preserve failure evidence"
    )
    demo.add_argument(
        "--output-dir", type=Path, default=ROOT / "build" / "recovery-demo"
    )
    manifest = commands.add_parser(
        "artifact-manifest", help="hash explicitly listed local artifacts"
    )
    manifest.add_argument("spec", type=Path)
    manifest.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser(
        "connect-inspect", help="classify a local QDC Connect command without echoing it"
    )
    inspect.add_argument(
        "--input", type=Path, help="read a temporary local command file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "bundle":
            result = build_bundle(args.output)
        elif args.command == "verify":
            verified = read_and_verify_bundle(args.bundle)
            result = {
                "status": "verified",
                "files": len(verified.payloads),
                "hardware_measurements": False,
            }
        elif args.command == "restore":
            result = restore_bundle(args.bundle, args.destination)
        elif args.command == "recovery-demo":
            result = _run_recovery_demo(args.output_dir)
        elif args.command == "artifact-manifest":
            result = build_artifact_manifest(args.spec, args.output)
        elif args.command == "connect-inspect":
            text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
            result = inspect_connect_text(text)
        else:
            raise RuntimeError("unhandled command")
    except (BundleError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
