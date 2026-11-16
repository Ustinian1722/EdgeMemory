#!/usr/bin/env python3
"""Safely install a SHA-verified GenieX v0.11.0 SDK ZIP or llama_cpp tar archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import stat
import shutil
import tarfile
import tempfile
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("runtime archive contains an unsafe path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("runtime archive contains an unsafe path")
    return relative


def _write_member(source, destination: Path, mode: int) -> int:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            size += len(block)
    os.chmod(destination, 0o755 if mode & 0o111 else 0o644)
    return size


def _extract_tar(archive: Path, temporary: Path) -> int:
    total = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            relative = _safe_member_path(member.name)
            if member.isdir():
                (temporary / Path(*relative.parts)).mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("runtime archive contains a link or special file")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("runtime archive member could not be read")
            with source:
                total += _write_member(source, temporary / Path(*relative.parts), member.mode)
            if total > 1024 * 1024 * 1024:
                raise ValueError("runtime archive expands beyond the 1 GiB safety limit")
    return total


def _extract_sdk_zip(archive: Path, temporary: Path) -> int:
    """Install only the Linux SDK library tree into the normalized runtime layout."""
    prefix = ("sdk-linux-arm64", "lib")
    total = 0
    seen: set[str] = set()
    with zipfile.ZipFile(archive, "r") as bundle:
        for member in bundle.infolist():
            relative = _safe_member_path(member.filename.rstrip("/"))
            if relative.parts[:len(prefix)] != prefix or len(relative.parts) <= len(prefix):
                continue
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or (file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("runtime SDK contains a link or special file")
            tail = relative.parts[len(prefix):]
            output_relative = PurePosixPath("geniex", "lib", *tail)
            output_key = output_relative.as_posix()
            if output_key in seen:
                raise ValueError("runtime SDK contains duplicate library paths")
            seen.add(output_key)
            output_path = temporary / Path(*output_relative.parts)
            if member.is_dir():
                output_path.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if member.file_size > 1024 * 1024 * 1024 or total + member.file_size > 1024 * 1024 * 1024:
                raise ValueError("runtime SDK expands beyond the 1 GiB safety limit")
            with bundle.open(member, "r") as source:
                total += _write_member(source, output_path, mode)
    return total


def restore(archive: Path, destination: Path, expected_sha256: str,
            expected_unpacked_bytes: int | None = None) -> dict:
    if sha256(archive) != expected_sha256.lower():
        raise ValueError("GenieX runtime archive SHA256 mismatch")
    archive_bytes = archive.stat().st_size
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime = destination / "geniex"
    if runtime.exists():
        manifest = destination / "runtime-manifest.json"
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("runtime destination exists without a verifiable manifest") from None
        if record.get("sha256") != expected_sha256.lower():
            raise RuntimeError("runtime destination belongs to a different archive")
        archive.unlink(missing_ok=True)
        return {"status": "already_restored", "runtime_root": str(runtime)}

    temporary = Path(tempfile.mkdtemp(prefix=".geniex-restore-", dir=destination))
    try:
        if zipfile.is_zipfile(archive):
            unpacked_bytes = _extract_sdk_zip(archive, temporary)
            source_format = "official_linux_arm64_sdk_zip"
        else:
            unpacked_bytes = _extract_tar(archive, temporary)
            source_format = "cached_llama_only_tar_gz"
        if expected_unpacked_bytes is not None and unpacked_bytes != expected_unpacked_bytes:
            raise ValueError("restored runtime size differs from the pinned SDK manifest")
        extracted = temporary / "geniex"
        required = (
            extracted / "lib" / "libgeniex.so",
            extracted / "lib" / "llama_cpp" / "libgeniex_plugin.so",
            extracted / "lib" / "llama_cpp" / "libggml-htp-v73.so",
        )
        if any(not path.is_file() for path in required):
            raise RuntimeError("restored package is missing a GenieX v0.11.0 runtime component")
        extracted.rename(runtime)
        record = {"schema_version": 1, "version": "v0.11.0",
                  "sha256": expected_sha256.lower(), "archive_bytes": archive_bytes,
                  "unpacked_bytes": unpacked_bytes, "source_format": source_format}
        temporary_manifest = destination / "runtime-manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(destination / "runtime-manifest.json")
        archive.unlink(missing_ok=True)
        return {"status": "restored", "runtime_root": str(runtime),
                "sha256": expected_sha256.lower(), "bytes": archive_bytes,
                "unpacked_bytes": unpacked_bytes, "source_format": source_format}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-unpacked-bytes", type=int)
    args = parser.parse_args()
    result = restore(args.archive, args.destination, args.expected_sha256,
                     args.expected_unpacked_bytes)
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
