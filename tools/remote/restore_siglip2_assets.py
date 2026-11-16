#!/usr/bin/env python3
"""Validate and unpack the optional pinned SigLIP2 QNN assets on IQ-9075."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


ASSET_VERSION = "0.61.0"
REQUIRED_QAIRT = "2.45.0.260326154327"
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = destination.resolve()
    written = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("SigLIP2 archive contains an unsafe path")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError("SigLIP2 archive contains a symbolic link")
            target = (destination / Path(*path.parts)).resolve()
            if root != target and root not in target.parents:
                raise ValueError("SigLIP2 archive path escapes the asset directory")
            if info.is_dir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            target.chmod(0o600)
            written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qairt-version", required=True)
    parser.add_argument("--backend-library", default="/usr/lib/libQnnGpu.so")
    parser.add_argument("--model-id", default="siglip2-base-patch16-224-qnn-w8a16-v0.61.0")
    parser.add_argument("--minimum-free-bytes", type=int, default=1073741824,
                        help="free space to retain after ZIP extraction")
    parser.add_argument("--max-unpacked-bytes", type=int, default=MAX_UNPACKED_BYTES,
                        help="refuse archives whose expanded payload exceeds this limit")
    args = parser.parse_args()

    if args.qairt_version != REQUIRED_QAIRT:
        print(json.dumps({"status": "blocked_qairt_version_mismatch",
                          "required_qairt_version": REQUIRED_QAIRT,
                          "observed_qairt_version": args.qairt_version,
                          "downloaded_or_unpacked": False}, separators=(",", ":")))
        return 42
    if not args.archive.is_file() or not zipfile.is_zipfile(args.archive):
        parser.error("--archive must name the verified downloaded ZIP")
    if not args.tokenizer.is_file():
        parser.error("--tokenizer must name the verified pinned tokenizer.json")
    if args.minimum_free_bytes < 0 or args.max_unpacked_bytes <= 0:
        parser.error("space limits must be non-negative and max-unpacked-bytes must be positive")
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        plugins = {name: Gst.ElementFactory.find(name) is not None
                   for name in ("qtimlqnn", "qtimlvconverter")}
    except Exception:
        plugins = {"qtimlqnn": False, "qtimlvconverter": False}
    try:
        import tokenizers  # noqa: F401
        tokenizers_available = True
    except Exception:
        tokenizers_available = False
    if not plugins["qtimlqnn"] or not tokenizers_available:
        print(json.dumps({"status": "blocked_target_dependency",
                          "qim_plugins": plugins,
                          "tokenizers_available": tokenizers_available,
                          "downloaded_files_retained": True}, separators=(",", ":")))
        return 43

    output_root = args.output_root.resolve()
    with zipfile.ZipFile(args.archive) as archive:
        unpacked_bytes = sum(info.file_size for info in archive.infolist() if not info.is_dir())
    archive_bytes = args.archive.stat().st_size
    if unpacked_bytes > args.max_unpacked_bytes:
        print(json.dumps({"status": "blocked_asset_size_limit",
                          "unpacked_bytes": unpacked_bytes,
                          "max_unpacked_bytes": args.max_unpacked_bytes,
                          "archive_bytes_already_used": archive_bytes}, separators=(",", ":")))
        return 46
    disk = shutil.disk_usage(output_root)
    required_bytes = unpacked_bytes + args.minimum_free_bytes
    if disk.free < required_bytes:
        print(json.dumps({"status": "blocked_insufficient_disk",
                          "free_bytes": disk.free,
                          "unpacked_bytes": unpacked_bytes,
                          "minimum_free_bytes": args.minimum_free_bytes,
                          "archive_bytes_already_used": archive_bytes,
                          "required_additional_bytes": required_bytes}, separators=(",", ":")))
        return 45
    extract_root = output_root / "siglip2-qnn-dlc-w8a16-v0.61.0"
    files = _safe_extract(args.archive.resolve(), extract_root)
    model_files = [path for path in files if path.suffix.lower() in {".bin", ".so"}]
    image_candidates = [path for path in model_files
                        if any(part in path.name.casefold() for part in ("image", "vision"))]
    text_candidates = [path for path in model_files if "text" in path.name.casefold()]
    status = "ready" if len(image_candidates) == 1 and len(text_candidates) == 1 else "asset_layout_needs_review"
    manifest = {
        "schema_version": 1, "status": status,
        "model_id": args.model_id, "ai_hub_models_version": ASSET_VERSION,
        "required_qairt_version": REQUIRED_QAIRT,
        "observed_qairt_version": args.qairt_version,
        "requested_backend_library": args.backend_library,
        "backend_library_present": Path(args.backend_library).is_file(),
        "qim_plugins": plugins,
        "tokenizers_available": tokenizers_available,
        "image_model_candidates": [str(path) for path in image_candidates],
        "text_model_candidates": [str(path) for path in text_candidates],
        "image_model_path": str(image_candidates[0]) if len(image_candidates) == 1 else None,
        "text_model_path": str(text_candidates[0]) if len(text_candidates) == 1 else None,
        "unpacked_bytes": unpacked_bytes,
        "tokenizer_path": str(args.tokenizer.resolve()),
        "archive_sha256": _sha256(args.archive.resolve()),
        "tokenizer_sha256": _sha256(args.tokenizer.resolve()),
        "files": [{"path": str(path.relative_to(output_root)),
                   "bytes": path.stat().st_size, "sha256": _sha256(path)}
                  for path in files],
        "device_support_note": "SigLIP2 AI Hub page contains conflicting IoT support statements; IQ-9075 inference remains unverified.",
    }
    manifest_path = output_root / "siglip2-assets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status,
                      "manifest_path": str(manifest_path),
                      "image_model_candidates": len(image_candidates),
                      "text_model_candidates": len(text_candidates),
                      "backend_library_present": manifest["backend_library_present"],
                      "qim_plugins": plugins}, separators=(",", ":")))
    return 0 if status == "ready" and manifest["backend_library_present"] else 44


if __name__ == "__main__":
    raise SystemExit(main())
