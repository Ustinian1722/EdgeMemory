#!/usr/bin/env python3
"""Build a local GenieX llama.cpp-only bundle without QAIRT assets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workspace_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("input and output paths must stay on the D: project workspace")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(archive_path: Path, prefix: str, bundle_path: Path, manifest_path: Path) -> dict:
    archive_path = workspace_path(archive_path)
    bundle_path = workspace_path(bundle_path)
    manifest_path = workspace_path(manifest_path)
    if not archive_path.is_file() or not prefix.endswith("/"):
        raise ValueError("archive must exist and prefix must end with '/'")
    if bundle_path == archive_path or manifest_path == archive_path:
        raise ValueError("refusing to overwrite the source archive")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    with tarfile.open(archive_path, "r:gz") as original:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as target:
                    for member in original.getmembers():
                        name = member.name
                        if not member.isfile() or not name.startswith(prefix):
                            continue
                        relative = name[len(prefix):]
                        include = (relative == "bin/geniex-bench" or
                                   relative == "lib/libgeniex.so" or
                                   (relative.startswith("lib/llama_cpp/") and
                                    relative.endswith(".so")))
                        if not include:
                            continue
                        if relative.startswith("/") or ".." in Path(relative).parts:
                            raise ValueError("unsafe source archive member")
                        source = original.extractfile(member)
                        if source is None:
                            raise ValueError("archive entry cannot be read")
                        payload = source.read()
                        info = tarfile.TarInfo("geniex/" + relative)
                        info.size = len(payload)
                        info.mode = member.mode & 0o755
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        target.addfile(info, io.BytesIO(payload))
                        entries.append({"path": relative, "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest()})
    if not entries:
        temporary.unlink(missing_ok=True)
        raise ValueError("no supported GenieX llama.cpp files matched the selected prefix")
    temporary.replace(bundle_path)
    manifest = {
        "schema_version": 1,
        "source_archive": archive_path.relative_to(ROOT).as_posix(),
        "source_archive_sha256": sha256(archive_path),
        "source_prefix": prefix,
        "bundle": bundle_path.relative_to(ROOT).as_posix(),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256(bundle_path),
        "entries": sorted(entries, key=lambda entry: entry["path"]),
        "qairt_assets_included": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distribution": "local-only; Qualcomm release package and its terms apply",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.archive, args.prefix, args.bundle, args.manifest)
    except (OSError, tarfile.TarError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"bundle": result["bundle"], "bytes": result["bundle_bytes"],
                      "sha256": result["bundle_sha256"], "entries": len(result["entries"]),
                      "qairt_assets_included": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
