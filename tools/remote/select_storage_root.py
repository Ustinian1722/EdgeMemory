#!/usr/bin/env python3
"""Select a writable non-RAM filesystem for session-local models and state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


PERSISTENT_FS = {"ext2", "ext3", "ext4", "f2fs", "xfs", "btrfs",
                 "vfat", "exfat", "ntfs", "ntfs3", "fuseblk"}


def _unescape_mount(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")


def _mounts() -> list[tuple[str, str]]:
    rows = []
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            left_fields, right_fields = left.split(), right.split()
            if len(left_fields) >= 5 and len(right_fields) >= 1:
                rows.append((_unescape_mount(left_fields[4]), right_fields[0]))
    except OSError:
        pass
    return rows


def _filesystem(path: Path, mounts: list[tuple[str, str]]) -> str | None:
    target = PurePosixPath(os.path.realpath(str(path)))
    matches = [(len(PurePosixPath(mount).parts), fs_type)
               for mount, fs_type in mounts
               if target == PurePosixPath(mount) or PurePosixPath(mount) in target.parents]
    return max(matches)[1] if matches else None


def _candidates(explicit: list[str], mounts: list[tuple[str, str]]) -> list[Path]:
    result = [Path(item) for item in explicit]
    for mount, fs_type in mounts:
        if fs_type not in PERSISTENT_FS:
            continue
        point = Path(mount)
        if point == Path("/"):
            result.extend((Path("/root/hexagonflow"), Path("/var/lib/hexagonflow")))
        elif point == Path("/data") or point == Path("/mnt") or point == Path("/media") or \
                str(point).startswith(("/data/", "/mnt/", "/media/")):
            result.append(point / "hexagonflow")
        elif point in {Path("/var/lib"), Path("/home"), Path("/root")}:
            result.append(point / "hexagonflow")
    unique = []
    seen = set()
    for item in result:
        if not item.is_absolute():
            continue
        normalized = Path(os.path.normpath(str(item)))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--minimum-free-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.minimum_free_bytes < 0:
        parser.error("--minimum-free-bytes must be non-negative")
    mounts = _mounts()
    reports = []
    eligible = []
    for path in _candidates(args.candidate, mounts):
        fs_type = _filesystem(path, mounts)
        row = {"path": str(path), "filesystem_type": fs_type,
               "status": "rejected_ram_or_unknown_filesystem"}
        if fs_type not in PERSISTENT_FS:
            reports.append(row)
            continue
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".hexagonflow-write-check-",
                                             dir=path, delete=True):
                pass
            free = shutil.disk_usage(path).free
            row.update(free_bytes=free,
                       status="available" if free >= args.minimum_free_bytes
                       else "insufficient_free_space")
            if free >= args.minimum_free_bytes:
                eligible.append((free, path, fs_type))
        except OSError as error:
            row.update(status="not_writable", error_type=type(error).__name__)
        reports.append(row)
    if not eligible:
        print(json.dumps({"status": "no_eligible_persistent_filesystem",
                          "minimum_free_bytes": args.minimum_free_bytes,
                          "candidates": reports}, separators=(",", ":")))
        return 42
    free, path, fs_type = max(eligible, key=lambda item: item[0])
    print(json.dumps({"status": "selected", "root": str(path),
                      "filesystem_type": fs_type, "free_bytes": free,
                      "durability_scope": "non-RAM filesystem for this QDC session only",
                      "minimum_free_bytes": args.minimum_free_bytes,
                      "candidates": reports}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
