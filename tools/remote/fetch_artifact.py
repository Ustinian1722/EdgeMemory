#!/usr/bin/env python3
"""Resume and verify pinned public model downloads on a temporary Linux device."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sys
import time
import urllib.error
import urllib.request
import re
import subprocess
import zipfile
from urllib.parse import urlsplit


SIGNED_CDN_HOSTS = {"us.aws.cdn.hf.co"}


def _read_url_overrides(chosen: list[dict], enabled: bool) -> dict[str, str]:
    if not enabled:
        return {}
    payload = sys.stdin.read(65537)
    if not payload or len(payload) > 65536:
        raise ValueError("temporary CDN URL handoff is empty or too large")
    try:
        overrides = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("temporary CDN URL handoff is invalid") from error
    allowed_ids = {item["id"] for item in chosen}
    if not isinstance(overrides, dict) or set(overrides) - allowed_ids:
        raise ValueError("temporary CDN URL handoff has unexpected asset IDs")
    for asset_id, value in overrides.items():
        parsed = urlsplit(value) if isinstance(value, str) else None
        source = next(item for item in chosen if item["id"] == asset_id).get("url", "")
        original = urlsplit(source)
        safe_mirror = (parsed is not None and parsed.hostname == "hf-mirror.com" and
                       original.hostname == "huggingface.co" and parsed.path == original.path and
                       not parsed.query and not parsed.fragment)
        if (parsed is None or parsed.scheme != "https" or
                (parsed.hostname not in SIGNED_CDN_HOSTS and not safe_mirror) or
                parsed.username or parsed.password):
            raise ValueError("temporary asset URL host or path is not allowlisted")
    return overrides


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_format(path: Path, format_name: str | None) -> None:
    if not format_name:
        return
    normalized = str(format_name).upper()
    if normalized == "GGUF":
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("artifact does not have the GGUF magic header")
        return
    if normalized == "ZIP":
        if not zipfile.is_zipfile(path):
            raise ValueError("artifact is not a valid ZIP archive")
        return
    if normalized == "ONNX":
        # ONNX is a protobuf stream without a fixed file magic; the pinned
        # byte count and SHA256 are the integrity checks for this asset.
        if path.stat().st_size < 1024 * 1024:
            raise ValueError("ONNX artifact is unexpectedly small")
        return
    if normalized == "JSON":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("tokenizer JSON artifact is not an object")
        return
    if normalized == "TEXT":
        if path.stat().st_size < 1024:
            raise ValueError("tokenizer merge table is unexpectedly small")
        return
    raise ValueError(f"unsupported artifact format check: {format_name}")


def _curl_resume(request_url: str, partial: Path, etag: str | None) -> int | None:
    """Use the board's curl -C - path for large CDN files, keeping URLs off argv/logs.

    The caller supplies HTTPS_PROXY only when the QDC subnet cannot egress directly.
    A missing curl binary returns None so the urllib fallback remains available.
    """
    config = "url = " + json.dumps(request_url) + "\n"
    if partial.is_file() and partial.stat().st_size and etag and "\r" not in etag and "\n" not in etag:
        config += "header = " + json.dumps("If-Range: " + etag) + "\n"
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "4", "--retry-all-errors", "--retry-delay", "1",
        "--connect-timeout", "30", "--continue-at", "-",
        "--output", str(partial), "--config", "-",
    ]
    try:
        completed = subprocess.run(
            command, input=config, text=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False,
        )
    except FileNotFoundError:
        return None
    return int(completed.returncode)


def fetch(item: dict, root: Path, *, ledger: dict, attempts: int = 5,
          url_override: str | None = None) -> dict:
    asset_id = item["id"]
    filename = item["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"unsafe artifact filename for {asset_id}")
    expected = str(item.get("sha256") or "").lower()
    policy = item.get("sha256_policy", "pinned")
    if expected and (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"artifact {asset_id} has an invalid pinned SHA256")
    if not expected and policy != "record_first_fetch_over_https":
        raise ValueError(f"artifact {asset_id} needs a pinned SHA256 or explicit first-fetch policy")
    prior = ledger.get(asset_id, {})
    if not expected and prior.get("sha256"):
        expected = str(prior["sha256"]).lower()
    url = item.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"artifact {asset_id} has no public HTTPS URL")
    request_url = url_override or url
    expected_size = item.get("size_bytes")
    if expected_size is None:
        prior_size = prior.get("bytes")
        if isinstance(prior_size, int) and prior_size > 0:
            expected_size = prior_size
    target = root / filename
    partial = target.with_name(target.name + ".part")
    partial_meta = partial.with_name(partial.name + ".json")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.is_file():
        observed = sha256(target)
        size_ok = expected_size is None or target.stat().st_size == int(expected_size)
        if not expected:
            raise ValueError(f"artifact {asset_id} exists without a checksum ledger; refusing to trust an untracked file")
        if observed == expected and size_ok:
            _verify_format(target, item.get("format"))
            return {"id": asset_id, "path": str(target), "bytes": target.stat().st_size,
                    "sha256": observed, "action": "already_verified"}
    if target.exists():
        target.unlink()

    if expected_size is None:
        # Query the CDN only when neither the manifest nor the durable first-fetch
        # ledger supplies a size. A verified existing asset can be reused offline.
        expected_size = preflight(item, root, url_override=request_url)["bytes"]

    last_error = "download failed"
    for attempt in range(attempts):
        metadata = {}
        if partial_meta.is_file():
            try:
                metadata = json.loads(partial_meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset > int(expected_size):
            partial.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            offset = 0
        if offset and (metadata.get("url") != url or not metadata.get("etag")):
            partial.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            offset = 0
        if offset and expected_size is not None and offset == int(expected_size):
            observed = sha256(partial)
            if not expected or observed == expected:
                _verify_format(partial, item.get("format"))
                partial.replace(target)
                partial_meta.unlink(missing_ok=True)
                if not item.get("sha256"):
                    ledger[asset_id] = {
                        "sha256": observed, "bytes": target.stat().st_size,
                        "url": url, "verification": "recorded_over_https_first_fetch",
                        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                return {"id": asset_id, "path": str(target), "bytes": target.stat().st_size,
                        "sha256": observed, "action": "resumed_complete_partial"}
            partial.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            offset = 0
        try:
            curl_status = _curl_resume(request_url, partial, metadata.get("etag"))
            if curl_status is not None:
                if curl_status != 0:
                    last_error = f"curl_exit_{curl_status}"
                    if attempt + 1 < attempts:
                        time.sleep(min(2 ** attempt, 12))
                        continue
                    raise RuntimeError(
                        f"artifact {asset_id} curl download stopped; partial bytes retained")
                # ETag is retained from preflight or the prior resumable part. The
                # pinned SHA256 and exact size below remain the final authority.
            else:
                headers = {"User-Agent": "HexagonFlow-device-bootstrap/1"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                    headers["If-Range"] = str(metadata["etag"])
                request = urllib.request.Request(request_url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = getattr(response, "status", response.getcode())
                    response_etag = response.headers.get("ETag", "")
                    append = (offset > 0 and status == 206
                              and response_etag == metadata.get("etag"))
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            append = False
                    mode = "ab" if append else "wb"
                    if not append:
                        offset = 0
                    if response_etag:
                        _write_json(partial_meta, {"url": url, "etag": response_etag})
                    else:
                        partial_meta.unlink(missing_ok=True)
                    with partial.open(mode) as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = type(error).__name__
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 12))
                continue
            raise RuntimeError(f"artifact {asset_id} download stopped; partial bytes retained") from error

        if expected_size is not None and partial.stat().st_size != int(expected_size):
            last_error = "size mismatch"
            if partial.stat().st_size > int(expected_size):
                partial.unlink()
            if attempt + 1 < attempts:
                continue
            raise RuntimeError(f"artifact {asset_id} size mismatch; partial bytes retained")
        observed = sha256(partial)
        if expected and observed != expected:
            last_error = "sha256 mismatch"
            # A complete hash mismatch cannot be fixed by appending the same source again.
            partial.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 12))
                continue
            raise RuntimeError(f"artifact {asset_id} SHA256 mismatch; refusing installation")
        try:
            _verify_format(partial, item.get("format"))
        except ValueError as error:
            partial.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            raise RuntimeError(f"artifact {asset_id} format check failed") from error
        partial.replace(target)
        partial_meta.unlink(missing_ok=True)
        if not item.get("sha256"):
            ledger[asset_id] = {
                "sha256": observed, "bytes": target.stat().st_size,
                "url": url, "verification": "recorded_over_https_first_fetch",
                "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        return {"id": asset_id, "path": str(target), "bytes": target.stat().st_size,
                "sha256": observed,
                "action": "downloaded_and_verified" if expected else "downloaded_and_sha256_recorded"}
    raise RuntimeError(f"artifact {asset_id}: {last_error}")


def preflight(item: dict, root: Path, *, url_override: str | None = None) -> dict:
    filename = item["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"unsafe artifact filename for {item.get('id', 'unknown')}")
    url = item.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"artifact {item.get('id', 'unknown')} has no public HTTPS URL")
    request_url = url_override or url
    request = urllib.request.Request(
        request_url, headers={"User-Agent": "HexagonFlow-device-bootstrap/1", "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+)", content_range)
        if match:
            total = int(match.group(1))
        elif status == 200 and response.headers.get("Content-Length", "").isdigit():
            total = int(response.headers["Content-Length"])
        elif item.get("size_bytes") is not None:
            total = int(item["size_bytes"])
        else:
            raise RuntimeError(f"artifact {item.get('id', 'unknown')} source did not report a bounded size")
        if item.get("size_bytes") is not None and total != int(item["size_bytes"]):
            raise RuntimeError(f"artifact {item.get('id', 'unknown')} source size differs from the pinned manifest")
        etag = response.headers.get("ETag", "")
        if status == 206:
            response.read(1)
    target = root / filename
    partial = target.with_name(target.name + ".part")
    metadata_path = partial.with_name(partial.name + ".json")
    prior_etag = ""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("url") == url:
            prior_etag = str(metadata.get("etag", ""))
    except (OSError, json.JSONDecodeError):
        pass
    partial_bytes = partial.stat().st_size if partial.is_file() and etag and etag == prior_etag else 0
    existing_bytes = target.stat().st_size if target.is_file() else 0
    return {
        "id": item.get("id"), "filename": filename, "bytes": total,
        "etag_available": bool(etag), "range_supported": status == 206,
        "partial_bytes_reusable": min(partial_bytes, total),
        "existing_target_bytes": existing_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path,
                        help="local device-side checksum ledger; defaults under --root")
    parser.add_argument("--preflight", action="store_true",
                        help="query source size/range metadata without downloading artifact contents")
    parser.add_argument("--include", action="append", default=[],
                        help="asset ID; repeat to select assets, default is all URL-backed assets")
    parser.add_argument("--url-overrides-stdin", action="store_true",
                        help="read ephemeral allowlisted CDN URLs from stdin; never persist them")
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported asset manifest schema")
    items = document.get("assets")
    if not isinstance(items, list):
        raise ValueError("asset manifest has no assets array")
    selected = set(args.include)
    by_id = {item.get("id"): item for item in items if isinstance(item, dict)}
    if len(by_id) != len(items):
        raise ValueError("asset IDs must be unique")
    unknown = selected - by_id.keys()
    if unknown:
        raise ValueError("unknown asset IDs: " + ", ".join(sorted(unknown)))
    chosen = [item for item in items if (not selected or item["id"] in selected)]
    url_overrides = _read_url_overrides(chosen, args.url_overrides_stdin)
    if args.preflight:
        args.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        rows = [preflight(item, args.root, url_override=url_overrides.get(item["id"]))
                for item in chosen]
        print(json.dumps({"schema_version": 1, "preflight_artifacts": rows}, separators=(",", ":")))
        return 0
    args.root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ledger_path = args.ledger or (args.root / "artifact-sha256-manifest.json")
    try:
        ledger_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        ledger_doc = {"schema_version": 1, "assets": {}}
    ledger = ledger_doc.get("assets", {})
    if not isinstance(ledger, dict):
        raise ValueError("checksum ledger assets must be an object")
    for item in chosen:
        if "url" not in item:
            raise ValueError(f"{item['id']} is a local-cache artifact and must use verified upload fallback")
    results_by_id = {}
    errors = []
    workers = min(3, len(chosen))
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch, item, args.root, ledger=dict(ledger),
                            url_override=url_overrides.get(item["id"])): item
                for item in chosen
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    results_by_id[item["id"]] = result
                    asset_id = item["id"]
                    if not item.get("sha256") and not ledger.get(asset_id, {}).get("sha256"):
                        ledger[asset_id] = {
                            "sha256": result["sha256"], "bytes": result["bytes"],
                            "url": item["url"], "verification": "recorded_over_https_first_fetch",
                            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                    ledger_doc["assets"] = ledger
                    _write_json(ledger_path, ledger_doc)
                except Exception as error:
                    errors.append(error)
    if errors:
        raise errors[0]
    results = [results_by_id[item["id"]] for item in chosen]
    print(json.dumps({"schema_version": 1, "verified_artifacts": results,
                      "checksum_ledger": str(ledger_path)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
