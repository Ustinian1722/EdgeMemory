#!/usr/bin/env python3
"""Deploy, exercise, collect, and forward the IQ-9075 device-resident product.

The SSH password is requested interactively and kept in process memory only.
This tool never logs the password, key material, or tunnel command.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import select
import shlex
import socket
import socketserver
import stat
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from getpass import getpass
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edgememory.qdc_geniex import load_qdc_geniex_profile


DEVICE_PACKAGE_FILES = (
    "edgememory/__init__.py",
    "edgememory/types.py",
    "edgememory/captions.py",
    "edgememory/qdc_geniex.py",
    "edgememory/scheduling.py",
    "edgememory/direct_execution.py",
    "edgememory/device_store.py",
    "edgememory/embeddings.py",
    "edgememory/hardware_profiler.py",
    "edgememory/llama_server.py",
    "edgememory/visual_embeddings.py",
    "edgememory/device_agent.py",
    "edgememory/static/index.html",
    "edgememory/static/app.css",
    "edgememory/static/app.js",
    "tools/remote/gstreamer_video_sampler.py",
    "tools/remote/fetch_artifact.py",
    "tools/remote/restore_geniex_runtime.py",
    "tools/remote/restore_siglip2_assets.py",
    "tools/remote/patch_clip_onnx_inputs.py",
    "tools/remote/select_storage_root.py",
    "configs/iq9075-board-assets.json",
    "scripts/remote/device_probe.sh",
)
SCHEDULER_LIBRARY = Path("build/arm64-device-scheduler/libhexagonflow_scheduler_c.so")
GENIEX_SERVICE = Path("build/arm64-geniex-run1/hexagonflow-geniex-vlm-service")
GENIEX_LLM_SERVICE = Path("build/arm64-geniex-run1/hexagonflow-geniex-llm-service")
GENIEX_RUNTIME_ARCHIVE = Path("third_party/downloads/geniex-v0.11.0/geniex-llama-only.tar.gz")
GENIEX_RUNTIME_SHA256 = "aed314b71b4baa394f3b34e3b6fcd5d2a9dc93d62035d7fb668c9dd5c31ef842"
GENIEX_SDK_ASSET_ID = "geniex-sdk-linux-arm64-v0.11.0"
GENIEX_SDK_SHA256 = "dd20ba09d1e00d3e584cff25042f040e542448f0e5e5f50023614af5eaa4e010"
GENIEX_SDK_UNPACKED_BYTES = 274962604
ASSET_MANIFEST = Path("configs/iq9075-board-assets.json")
LLAMA_SERVER = Path("build/llama-arm64/bin/llama-server")
MINILM_MODEL = Path("models/cache/all-MiniLM-L6-v2-q8_0-cstr-repaired-v3.gguf")
VLM_CACHE_ROOT = Path("models/cache/qwen3-vl-4b-qualcomm")
VLM_CACHE_FILES = {
    "qwen3-vl-4b-q4_0": ("Qwen3-VL-4B-Instruct-Q4_0.gguf",
        "09965766e92778b591ab9da5209463b843d000449b38d909055b399d8dbd3d76"),
    "qwen3-vl-4b-mmproj-f16": ("mmproj-F16.gguf",
        "1b9f4e92f0fbda14d7d7b58baed86039b8a980fe503d9d6a9393f25c0028f1fc"),
}
ASSET_PARTIAL_CACHE = ROOT / ".local" / "asset-cache" / "partials"
ASSET_LEDGER_CACHE = ROOT / ".local" / "asset-cache" / "artifact-sha256-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tar_unpacked_bytes(path: Path) -> int:
    with tarfile.open(path, "r:gz") as archive:
        return sum(member.size for member in archive.getmembers() if member.isfile())


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_bundle(output: Path) -> tuple[str, list[dict[str, str | int]]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in DEVICE_PACKAGE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"device package source is missing: {relative}")
        files.append((relative, path))
    scheduler = ROOT / SCHEDULER_LIBRARY
    if not scheduler.is_file():
        raise FileNotFoundError("ARM64 HexagonFlow scheduler library has not been built")
    files.append(("build/scheduler/libhexagonflow_scheduler_c.so", scheduler))
    service = ROOT / GENIEX_SERVICE
    if not service.is_file():
        raise FileNotFoundError("ARM64 GenieX VLM service has not been built")
    files.append(("assets/bin/hexagonflow-geniex-vlm-service", service))
    llm_service = ROOT / GENIEX_LLM_SERVICE
    if llm_service.is_file():
        files.append(("assets/bin/hexagonflow-geniex-llm-service", llm_service))
    with tarfile.open(output, "w:gz") as bundle:
        for relative, path in files:
            bundle.add(path, arcname=relative, recursive=False)
    inventory = [{"path": relative, "bytes": path.stat().st_size,
                  "sha256": _sha256(path)} for relative, path in files]
    return _sha256(output), inventory


def _put_verified(sftp, local: Path, remote: str) -> dict:
    sftp.put(str(local), remote, confirm=True)
    sftp.chmod(remote, 0o700 if os.access(local, os.X_OK) else 0o600)
    code = (
        "import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],'rb'); "
        "[h.update(b) for b in iter(lambda:f.read(1048576),b'')]; print(h.hexdigest())"
    )
    # Called only after an explicitly started session; the function name and output
    # intentionally avoid printing profile/connection data.
    expected = _sha256(local)
    return {"path": remote, "bytes": local.stat().st_size, "sha256": expected,
            "hash_check_command": code}


def _asset_catalog() -> dict[str, dict]:
    document = json.loads((ROOT / ASSET_MANIFEST).read_text(encoding="utf-8"))
    return {item["id"]: item for item in document["assets"] if isinstance(item, dict)}


def _resolve_hf_cdn_urls(asset_ids: list[str], catalog: dict[str, dict]) -> dict[str, str]:
    """Resolve short-lived HF redirects in memory for direct device-side CDN fetches."""
    resolved = {}
    for asset_id in asset_ids:
        item = catalog[asset_id]
        source = item.get("url")
        if not isinstance(source, str) or urlparse(source).hostname != "huggingface.co":
            continue
        request = Request(source, headers={
            "Range": "bytes=0-0", "Accept-Encoding": "identity",
            "User-Agent": "HexagonFlow-device-bootstrap/1",
        })
        try:
            with urlopen(request, timeout=30) as response:
                final_url = response.geturl()
                if urlparse(final_url).hostname != "us.aws.cdn.hf.co":
                    raise ValueError("unexpected redirect host")
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes\s+0-0/(\d+)", content_range)
                expected_size = item.get("size_bytes")
                if match and expected_size is not None and int(match.group(1)) != int(expected_size):
                    raise ValueError("asset size differs from manifest")
                if getattr(response, "status", response.getcode()) not in (200, 206):
                    raise ValueError("unexpected CDN response status")
                response.read(1)
                resolved[asset_id] = final_url
        except Exception as error:
            raise RuntimeError(
                f"could not resolve the allowlisted CDN route for asset {asset_id}: {type(error).__name__}") from None
    return resolved


def _remote_mkdir(sftp, path: str) -> None:
    try:
        sftp.mkdir(path, mode=0o700)
    except OSError:
        sftp.stat(path)


def _stage_asset_ledger(sftp, remote_root: str) -> bool:
    if not ASSET_LEDGER_CACHE.is_file():
        return False
    try:
        document = json.loads(ASSET_LEDGER_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("the D-drive artifact checksum ledger is invalid") from error
    if document.get("schema_version") != 1 or not isinstance(document.get("assets"), dict):
        raise RuntimeError("the D-drive artifact checksum ledger has an unsupported schema")
    remote_assets = f"{remote_root}/assets"
    _remote_mkdir(sftp, remote_assets)
    temporary = f"{remote_assets}/artifact-sha256-manifest.json.stage-{uuid.uuid4().hex}"
    sftp.put(str(ASSET_LEDGER_CACHE), temporary, confirm=True)
    sftp.chmod(temporary, 0o600)
    destination = f"{remote_assets}/artifact-sha256-manifest.json"
    try:
        sftp.remove(destination)
    except OSError:
        pass
    sftp.rename(temporary, destination)
    return True


def _collect_asset_ledger(sftp, remote_root: str, run_dir: Path | None = None) -> bool:
    remote = f"{remote_root}/assets/artifact-sha256-manifest.json"
    ASSET_LEDGER_CACHE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = ASSET_LEDGER_CACHE.with_name(ASSET_LEDGER_CACHE.name + ".download")
    try:
        sftp.get(remote, str(temporary))
        document = json.loads(temporary.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or not isinstance(document.get("assets"), dict):
            raise ValueError("device returned an invalid asset checksum ledger")
        ASSET_LEDGER_CACHE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.replace(ASSET_LEDGER_CACHE)
        if run_dir is not None:
            destination = run_dir / "artifact-sha256-manifest.json"
            stage = destination.with_suffix(".part")
            stage.write_bytes(ASSET_LEDGER_CACHE.read_bytes())
            stage.replace(destination)
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        temporary.unlink(missing_ok=True)
        return False


def _stage_asset_partials(sftp, remote_root: str, asset_ids: list[str]) -> list[dict]:
    """Restore validated host partials into a fresh session before size preflight."""
    catalog = _asset_catalog()
    staged = []
    remote_assets = f"{remote_root}/assets"
    _remote_mkdir(sftp, remote_assets)
    for asset_id in asset_ids:
        item = catalog[asset_id]
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", asset_id):
            raise ValueError("unsafe asset ID in recovery manifest")
        local_dir = ASSET_PARTIAL_CACHE / asset_id
        local_part = local_dir / f"{item['filename']}.part"
        local_meta = local_dir / f"{item['filename']}.part.json"
        if not local_part.is_file() or not local_meta.is_file() or local_part.stat().st_size <= 0:
            continue
        try:
            metadata = json.loads(local_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("url") != item.get("url") or not metadata.get("etag"):
            continue
        remote_part = f"{remote_assets}/{item['filename']}.part"
        remote_meta = f"{remote_part}.json"
        # A still-running or interrupted download in the same QDC session can be
        # newer than the D-drive recovery snapshot. Never replace its larger,
        # ETag-matched prefix with an older local partial.
        try:
            remote_size = int(sftp.stat(remote_part).st_size)
            with sftp.open(remote_meta, "rb") as remote_meta_file:
                remote_metadata = json.loads(remote_meta_file.read().decode("utf-8"))
            if (remote_metadata.get("url") == metadata.get("url") and
                    remote_metadata.get("etag") == metadata.get("etag") and
                    remote_size >= local_part.stat().st_size):
                continue
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
            pass
        for local, remote in ((local_part, remote_part), (local_meta, remote_meta)):
            temporary = f"{remote}.stage-{uuid.uuid4().hex}"
            sftp.put(str(local), temporary, confirm=True)
            sftp.chmod(temporary, 0o600)
            try:
                sftp.remove(remote)
            except OSError:
                pass
            sftp.rename(temporary, remote)
        staged.append({"asset_id": asset_id, "bytes": local_part.stat().st_size})
    return staged


def _collect_asset_partials(sftp, remote_root: str, asset_ids: list[str]) -> list[dict]:
    """Copy resumable device partials to ignored D-drive storage before session loss."""
    catalog = _asset_catalog()
    recovered = []
    remote_assets = f"{remote_root}/assets"
    for asset_id in asset_ids:
        item = catalog[asset_id]
        remote_part = f"{remote_assets}/{item['filename']}.part"
        remote_meta = f"{remote_part}.json"
        local_dir = ASSET_PARTIAL_CACHE / asset_id
        local_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        local_part = local_dir / f"{item['filename']}.part"
        local_meta = local_dir / f"{item['filename']}.part.json"
        part_tmp = local_part.with_name(local_part.name + ".download")
        meta_tmp = local_meta.with_name(local_meta.name + ".download")
        try:
            remote_part_size = int(sftp.stat(remote_part).st_size)
            if remote_part_size <= 0:
                continue
            sftp.get(remote_part, str(part_tmp))
            sftp.get(remote_meta, str(meta_tmp))
            metadata = json.loads(meta_tmp.read_text(encoding="utf-8"))
            if (part_tmp.stat().st_size != remote_part_size or
                    metadata.get("url") != item.get("url") or not metadata.get("etag")):
                continue
            part_tmp.replace(local_part)
            meta_tmp.replace(local_meta)
            recovered.append({"asset_id": asset_id, "bytes": remote_part_size,
                              "sha256": _sha256(local_part),
                              "cache": str(local_part.relative_to(ROOT))})
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        finally:
            part_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
    return recovered


def _delete_asset_partials(asset_ids: list[str]) -> None:
    catalog = _asset_catalog()
    for asset_id in asset_ids:
        item = catalog.get(asset_id)
        if not item:
            continue
        local_dir = ASSET_PARTIAL_CACHE / asset_id
        (local_dir / f"{item['filename']}.part").unlink(missing_ok=True)
        (local_dir / f"{item['filename']}.part.json").unlink(missing_ok=True)


def _connect(profile, password: str):
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("the existing Windows Python environment needs Paramiko") from error
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=profile.host, port=profile.port, username=profile.username,
        password=password, look_for_keys=False, allow_agent=False,
        timeout=15, banner_timeout=15, auth_timeout=15,
    )
    transport = client.get_transport()
    if transport is None or not transport.is_active():
        client.close()
        raise RuntimeError("current QDC SSH tunnel did not establish an active device transport")
    transport.set_keepalive(20)
    return client


def _remote_exec(client, command: str, *, timeout_s: int = 60,
                 log_path: Path | None = None,
                 stdin_data: str | None = None) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout_s, get_pty=False)
    if stdin_data is not None:
        stdin.write(stdin_data)
        stdin.flush()
        stdin.channel.shutdown_write()
    channel = stdout.channel
    channel.settimeout(1.0)
    out = bytearray()
    err = bytearray()
    log_stream = log_path.open("ab") if log_path else None
    try:
        while True:
            progressed = False
            while channel.recv_ready():
                block = channel.recv(65536)
                if not block:
                    break
                out.extend(block)
                progressed = True
                if log_stream:
                    log_stream.write(block)
            while channel.recv_stderr_ready():
                block = channel.recv_stderr(65536)
                if not block:
                    break
                err.extend(block)
                progressed = True
                if log_stream:
                    log_stream.write(block)
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if not progressed:
                time.sleep(0.05)
        status = channel.recv_exit_status()
    finally:
        if log_stream:
            log_stream.close()
    return status, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _device_https_proxy(command: str, port: int) -> str:
    """Supply an optional loopback CONNECT relay to target-side HTTPS fetches.

    The downloader still runs on the device and stores artifacts there; the
    relay is only a transport fallback when the QDC subnet lacks outbound access.
    """
    if not port:
        return command
    proxy = shlex.quote(f"http://127.0.0.1:{port}")
    return (f"env HTTPS_PROXY={proxy} HTTP_PROXY={proxy} "
            f"https_proxy={proxy} http_proxy={proxy} {command}")


def _collect_tree(sftp, remote: str, local: Path,
                  collected: dict[str, tuple[int, int]]) -> int:
    copied = 0
    local.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote)
    except FileNotFoundError:
        # The device agent may remove transient candidate frames while a
        # collection pass is enumerating them. A missing directory is not a
        # session failure; the next pass will collect any durable output.
        return 0
    for item in entries:
        remote_path = remote.rstrip("/") + "/" + item.filename
        local_path = local / item.filename
        mode = item.st_mode or 0
        if stat.S_ISLNK(mode):
            continue
        if stat.S_ISDIR(mode):
            copied += _collect_tree(sftp, remote_path, local_path, collected)
        elif stat.S_ISREG(mode):
            signature = (int(item.st_size or 0), int(item.st_mtime or 0))
            if collected.get(remote_path) == signature and local_path.is_file():
                continue
            temporary = local_path.with_name(local_path.name + ".part")
            try:
                sftp.get(remote_path, str(temporary))
            except FileNotFoundError:
                temporary.unlink(missing_ok=True)
                continue
            temporary.replace(local_path)
            collected[remote_path] = signature
            copied += 1
    return copied


def _collect_device(client, remote_root: str, local_root: Path,
                    collected: dict[str, tuple[int, int]]) -> int:
    sftp = client.open_sftp()
    count = 0
    try:
        for relative in ("state", "runs", "logs"):
            remote = f"{remote_root}/{relative}"
            try:
                sftp.stat(remote)
            except OSError:
                continue
            count += _collect_tree(sftp, remote, local_root / "device" / relative, collected)
        # Preserve the exact code copy that was installed on the device.
        for relative in ("edgememory", "tools", "build/scheduler"):
            remote = f"{remote_root}/{relative}"
            try:
                sftp.stat(remote)
            except OSError:
                continue
            count += _collect_tree(sftp, remote, local_root / "device-source" / relative, collected)
    finally:
        sftp.close()
    return count


def _forward_handler(transport, remote_port: int):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel = transport.open_channel(
                "direct-tcpip", ("127.0.0.1", remote_port), self.client_address)
            if channel is None:
                return
            left, right = self.request, channel

            def copy(source, destination):
                try:
                    while True:
                        ready, _, _ = select.select([source], [], [], 30.0)
                        if not ready:
                            continue
                        block = source.recv(65536)
                        if not block:
                            break
                        destination.sendall(block)
                except (OSError, EOFError):
                    pass
                try:
                    if hasattr(destination, "shutdown_write"):
                        destination.shutdown_write()
                except Exception:
                    pass

            outgoing = threading.Thread(target=copy, args=(left, right), daemon=True)
            incoming = threading.Thread(target=copy, args=(right, left), daemon=True)
            outgoing.start()
            incoming.start()
            outgoing.join()
            channel.close()
            incoming.join(timeout=2)

    return Handler


def _wait_for_health(client, *, timeout_s: int = 240) -> dict:
    command = (
        "python3 -c " + shlex.quote(
            "import json,urllib.request; "
            "print(urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=3).read().decode())")
    )
    deadline = time.monotonic() + timeout_s
    last_error = "service_not_ready"
    while time.monotonic() < deadline:
        status, stdout, _stderr = _remote_exec(client, command, timeout_s=5)
        if status == 0:
            try:
                return json.loads(stdout.strip())
            except json.JSONDecodeError:
                last_error = "invalid_health_json"
        else:
            last_error = "service_not_ready"
        time.sleep(1.0)
    raise TimeoutError(last_error)


def run_session(args) -> int:
    if not 0 <= args.device_https_proxy_port <= 65535 or (
            args.device_https_proxy_port and args.device_https_proxy_port < 1024):
        raise ValueError("--device-https-proxy-port must be 0 or between 1024 and 65535")
    if args.context is not None and not 256 <= args.context <= 131072:
        raise ValueError("--context must be between 256 and 131072")
    if args.hexagon_mbuf is not None and not re.fullmatch(
            r"[1-9][0-9]*[kmg]?(,[1-9][0-9]*[kmg]?){1,2}",
            args.hexagon_mbuf, flags=re.IGNORECASE):
        raise ValueError("--hexagon-mbuf must contain two or three positive comma-separated chunk sizes")
    if args.hexagon_devices is not None and not re.fullmatch(
            r"[A-Za-z0-9,:\[\]-]{1,255}", args.hexagon_devices):
        raise ValueError("--hexagon-devices contains unsupported selector characters")
    if args.vit_device_id is not None and not re.fullmatch(
            r"[A-Za-z0-9,:\[\]-]{1,127}", args.vit_device_id):
        raise ValueError("--vit-device-id contains unsupported selector characters")
    if args.assets_only and (not args.asset or args.restore_vlm_gguf or
                             args.restore_llm_gguf or args.restore_minilm or
                             args.restore_geniex_runtime or args.restore_siglip2 or
                             args.restore_openai_clip or
                             args.device_selftest):
        raise ValueError("--assets-only requires one or more --asset IDs and no restore shortcut")
    if args.restore_siglip2 and args.restore_openai_clip:
        raise ValueError("select only one visual embedding model restore path per session")
    if args.restore_siglip2 and args.siglip2_qairt_version != "2.45.0.260326154327":
        raise ValueError(
            "SigLIP2 assets require the exact target QAIRT version 2.45.0.260326154327; "
            "verify the current device runtime before passing --siglip2-qairt-version")
    if args.upload_local_vlm_cache and not args.restore_vlm_gguf:
        raise ValueError("--upload-local-vlm-cache requires --restore-vlm-gguf")
    if args.restore_llm_gguf and args.llm_min_free_bytes <= 0:
        raise ValueError("--restore-llm-gguf requires a positive --llm-min-free-bytes admission budget")
    if args.restore_llm_gguf and args.llm_backend != "cpu" and not (ROOT / GENIEX_LLM_SERVICE).is_file():
        raise FileNotFoundError("ARM64 GenieX LLM service build is missing")
    if not re.fullmatch(r"[A-Za-z0-9:_\[\]-]{1,127}", args.llm_device_id):
        raise ValueError("--llm-device-id contains unsupported selector characters")
    if not 256 <= args.llm_geniex_context <= 131072 or args.llm_geniex_threads < 1:
        raise ValueError("invalid GenieX LLM context or thread count")
    if args.restore_minilm and args.embedding_min_free_bytes <= 0:
        raise ValueError("--restore-minilm requires a positive --embedding-min-free-bytes admission budget")
    if args.visual_min_free_bytes < 0:
        raise ValueError("--visual-min-free-bytes must be non-negative")
    if not 1 <= args.video_workers <= 16 or not 0 <= args.pending_videos <= 64:
        raise ValueError("video worker and pending-job limits are outside the supported range")
    if (args.restore_llm_gguf or args.restore_minilm) and not LLAMA_SERVER.is_file():
        raise FileNotFoundError("ARM64 llama-server build is missing; prepare the existing local build first")
    profile_path = (ROOT / args.profile).resolve()
    profile = load_qdc_geniex_profile(profile_path)
    run_dir = ROOT / "runs" / f"iq9075-device-runtime-{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = ROOT / "build" / "hexagonflow-device.tar.gz"
    bundle_hash, source_files = _make_bundle(bundle_path)
    release_id = bundle_hash[:12]
    if args.release_root:
        remote_root = args.release_root.rstrip("/")
        remote_prefix = f"{profile.remote_work_root.rstrip('/')}/hexagonflow-device-"
        if (not remote_root.startswith(remote_prefix) or
                any(part == ".." for part in remote_root.split("/")) or
                not re.fullmatch(r"[A-Za-z0-9/_-]+", remote_root)):
            raise ValueError("session --release-root must be a safe release path under the configured work root")
    else:
        remote_root = f"{profile.remote_work_root.rstrip('/')}/hexagonflow-device-{release_id}"
    staging_root = remote_root
    report = {
        "schema_version": 1, "run_id": run_dir.name,
        "execution_target": "Qualcomm IQ-9075 via active QDC session",
        "remote_release_root": remote_root,
        "device_ids_configured": [item.strip() for item in args.devices.split(",") if item.strip()],
        "scheduler_policy": args.policy,
        "video_ingest": {
            "workers": args.video_workers,
            "pending_jobs": args.pending_videos,
            "preprocess_path_requested": args.video_preprocess,
            "decoder_factory_requested": args.decoder_factory,
            "decoder_io_mode_requested": args.decoder_io_mode,
        },
        "model_id": profile.model_id,
        "context_length": args.context or profile.context_length,
        "hexagon_mbuf_override": args.hexagon_mbuf,
        "hexagon_hostbuf_override": args.hexagon_hostbuf,
        "hexagon_devices_override": args.hexagon_devices,
        "vision_device_override": args.vit_device_id,
        "device_bundle_sha256": bundle_hash,
        "source_files": source_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "remote_storage_policy": "select a writable non-RAM filesystem before model downloads; remote assets remain session-local and are collected to D",
        "asset_download_transport": (
            "device_https_via_loopback_connect_relay" if args.device_https_proxy_port
            else "device_direct_https"),
        "asset_manifest": ASSET_MANIFEST.as_posix(),
        "restored_asset_ids": [],
    }
    (run_dir / "deployment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not sys.stdin.isatty():
        raise RuntimeError("run this session command in an interactive terminal for the hidden password prompt")
    password = getpass("QDC device password: ")
    if not password:
        raise RuntimeError("QDC device password was empty")
    client = None
    proxy_server = None
    collected_files: dict[str, tuple[int, int]] = {}
    try:
        client = _connect(profile, password)
        password = ""
        sftp = client.open_sftp()
        try:
            _remote_mkdir(sftp, profile.remote_work_root)
            remote_archive = f"{remote_root}.tar.gz"
            sftp.put(str(bundle_path), remote_archive, confirm=True)
            sftp.chmod(remote_archive, 0o600)
        finally:
            sftp.close()

        install_code = (
            "import os,tarfile; "
            f"root={remote_root!r}; archive={remote_archive!r}; "
            "os.makedirs(root,mode=0o700,exist_ok=True); "
            "tarfile.open(archive,'r:gz').extractall(root,filter='data'); "
            "os.chmod(os.path.join(root,'assets/bin/hexagonflow-geniex-vlm-service'),0o700); "
            "os.unlink(archive)"
        )
        status, _stdout, stderr = _remote_exec(
            client, "python3 -c " + shlex.quote(install_code), timeout_s=60,
            log_path=run_dir / "deployment-console.log")
        if status != 0:
            raise RuntimeError("device bundle extraction failed")

        probe_command = f"sh {shlex.quote(remote_root + '/scripts/remote/device_probe.sh')}"
        status, probe_stdout, probe_stderr = _remote_exec(
            client, probe_command, timeout_s=60,
            log_path=run_dir / "device-probe-console.log")
        (run_dir / "device-probe.txt").write_text(
            probe_stdout + ("\n" + probe_stderr if probe_stderr else ""), encoding="utf-8")
        if status != 0:
            raise RuntimeError("read-only device probe failed; model downloads were not started")
        report["device_probe_artifact"] = "device-probe.txt"

        if not args.assets_only:
            target_abi_code = (
                "import json,platform,re,sys;libc,version=platform.libc_ver(); "
                "parts=tuple(int(x) for x in re.findall(r'[0-9]+',version)[:2]); "
                "arch=platform.machine().lower(); compatible=(arch in ('aarch64','arm64') and "
                "libc=='glibc' and parts>=(2,34)); "
                "print(json.dumps({'architecture':arch,'libc':libc,'libc_version':version,"
                "'service_minimum_glibc':'2.34','compatible':compatible})); "
                "sys.exit(0 if compatible else 49)"
            )
            abi_status, abi_stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(target_abi_code), timeout_s=15,
                log_path=run_dir / "runtime-probe-console.log")
            try:
                target_abi = json.loads(abi_stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                raise RuntimeError("target ARM64/glibc preflight returned invalid metadata") from error
            report["target_abi_preflight"] = target_abi
            (run_dir / "deployment.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if abi_status != 0:
                raise RuntimeError("target ARM64/glibc is incompatible or unknown; SDK and model downloads were not started")

        asset_ids = list(args.asset or [])
        if args.restore_vlm_gguf:
            asset_ids.extend(("qwen3-vl-4b-q4_0", "qwen3-vl-4b-mmproj-f16"))
        if args.restore_llm_gguf:
            asset_ids.append("qwen3-4b-instruct-2507-q4_0")
        if args.restore_siglip2:
            asset_ids.extend(("siglip2-qnn-dlc-w8a16-v061-experimental",
                              "siglip2-tokenizer-google-pinned"))
        asset_ids = list(dict.fromkeys(asset_ids))
        catalog = _asset_catalog()
        unknown_assets = set(asset_ids) - catalog.keys()
        if unknown_assets:
            raise ValueError("unknown board asset IDs: " + ", ".join(sorted(unknown_assets)))
        local_only_assets = [asset_id for asset_id in asset_ids
                             if not catalog[asset_id].get("url")]
        if local_only_assets:
            raise ValueError(
                "local-cache assets must use their verified restore shortcut: " +
                ", ".join(sorted(local_only_assets)))

        expected_asset_bytes = sum(int(catalog[asset_id].get("size_bytes") or 0)
                                   for asset_id in asset_ids)
        expected_runtime_bytes = 0
        if args.restore_geniex_runtime:
            runtime_asset = catalog[GENIEX_SDK_ASSET_ID]
            expected_runtime_bytes = (int(runtime_asset.get("size_bytes") or 0) +
                                      int(runtime_asset.get("unpacked_size_bytes") or 0))
        expected_local_bytes = 0
        if args.restore_minilm or args.restore_llm_gguf:
            expected_local_bytes += LLAMA_SERVER.stat().st_size
        if args.restore_minilm:
            expected_local_bytes += MINILM_MODEL.stat().st_size
        if args.upload_local_vlm_cache:
            expected_local_bytes += sum((ROOT / VLM_CACHE_ROOT / name).stat().st_size
                                        for name, _sha in VLM_CACHE_FILES.values())
        storage_minimum = (args.asset_space_reserve_bytes + expected_asset_bytes +
                           expected_runtime_bytes + expected_local_bytes)
        storage_args = [
            "python3", f"{staging_root}/tools/remote/select_storage_root.py",
            "--minimum-free-bytes", str(storage_minimum),
        ]
        for candidate in (profile.remote_work_root, "/root/hexagonflow",
                          "/var/lib/hexagonflow", "/data/hexagonflow",
                          "/mnt/data/hexagonflow"):
            storage_args.extend(("--candidate", candidate))
        status, storage_stdout, _stderr = _remote_exec(
            client, " ".join(shlex.quote(item) for item in storage_args),
            timeout_s=30, log_path=run_dir / "storage-probe-console.log")
        try:
            storage_result = json.loads(storage_stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError("device persistent-storage selection returned invalid metadata") from error
        report["storage_preflight"] = storage_result
        if status != 0 or storage_result.get("status") != "selected":
            (run_dir / "deployment.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(
                "no writable non-RAM device filesystem has enough free space; model downloads were not started")

        selected_base = storage_result["root"].rstrip("/")
        if args.release_root:
            remote_root = args.release_root.rstrip("/")
            if not remote_root.startswith(f"{selected_base}/hexagonflow-device-"):
                raise ValueError("session --release-root must use the selected non-RAM storage base")
        else:
            remote_root = f"{selected_base}/hexagonflow-device-{release_id}"
        if remote_root != staging_root:
            relocate_code = (
                "import os,shutil,sys; src,dst,digest=sys.argv[1:4]; "
                "marker='.hexagonflow-bundle-sha256'; "
                "os.makedirs(os.path.dirname(dst),mode=0o700,exist_ok=True); "
                "if_exists=os.path.isdir(dst); "
                "existing=open(os.path.join(dst,marker)).read().strip() if if_exists and os.path.isfile(os.path.join(dst,marker)) else ''; "
                "(sys.exit(48) if if_exists and existing!=digest else None); "
                "shutil.copytree(src,dst,dirs_exist_ok=True) if not if_exists else None; "
                "open(os.path.join(dst,marker),'w').write(digest); "
                "shutil.rmtree(src) if os.path.realpath(src)!=os.path.realpath(dst) else None; "
                "print('storage_relocation=ready')"
            )
            status, _stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(relocate_code) + " " +
                shlex.quote(staging_root) + " " + shlex.quote(remote_root) + " " +
                shlex.quote(bundle_hash), timeout_s=60,
                log_path=run_dir / "storage-probe-console.log")
            if status != 0:
                raise RuntimeError("device bundle could not be relocated to the selected non-RAM filesystem")
        report["remote_release_root"] = remote_root
        report["remote_storage_policy"] = (
            "selected non-RAM filesystem; models, runtime, database and app are session-local; "
            "remote state is collected to D during the session")
        (run_dir / "deployment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Runtime assets are release-scoped and session-local. Do not probe a
        # stale path from an older QDC session/profile when resuming this release.
        runtime_root = f"{remote_root}/runtime-v0.11.0/geniex"
        service_binary = f"{remote_root}/assets/bin/hexagonflow-geniex-vlm-service"
        if not args.assets_only:
            if args.restore_geniex_runtime:
                current_runtime_code = (
                    "import json,os,sys;r=sys.argv[1];p=os.path.join(r,'lib','llama_cpp'); "
                    "c=dict(geniex_c_api=os.path.isfile(os.path.join(r,'lib','libgeniex.so')) ,"
                    "llama_cpp_plugin=os.path.isfile(os.path.join(p,'libgeniex_plugin.so')) ,"
                    "htp_v73_backend=os.path.isfile(os.path.join(p,'libggml-htp-v73.so'))); "
                    "print(json.dumps(c))"
                )
                status, current_stdout, _stderr = _remote_exec(
                    client, "python3 -c " + shlex.quote(current_runtime_code) + " " +
                    shlex.quote(runtime_root), timeout_s=20,
                    log_path=run_dir / "runtime-probe-console.log")
                try:
                    current_runtime_checks = json.loads(current_stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError) as error:
                    raise RuntimeError("current GenieX runtime probe returned invalid metadata") from error
                if status != 0 or not all(current_runtime_checks.values()):
                    catalog = _asset_catalog()
                    runtime_asset = catalog[GENIEX_SDK_ASSET_ID]
                    remote_assets = f"{remote_root}/assets"
                    remote_runtime_archive = f"{remote_assets}/{runtime_asset['filename']}"
                    runtime_download_ok = False
                    runtime_download_reason = "official_asset_preflight_failed"
                    sftp = client.open_sftp()
                    try:
                        _stage_asset_ledger(sftp, remote_root)
                        staged = _stage_asset_partials(sftp, remote_root, [GENIEX_SDK_ASSET_ID])
                    finally:
                        sftp.close()
                    preflight_args = [
                        "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                        "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                        "--root", remote_assets, "--preflight", "--include", GENIEX_SDK_ASSET_ID,
                    ]
                    status, preflight_stdout, _stderr = _remote_exec(
                        client, _device_https_proxy(
                            " ".join(shlex.quote(item) for item in preflight_args),
                            args.device_https_proxy_port),
                        timeout_s=60, log_path=run_dir / "runtime-probe-console.log")
                    if status == 0:
                        try:
                            preflight_rows = json.loads(preflight_stdout.strip().splitlines()[-1])["preflight_artifacts"]
                            row = preflight_rows[0]
                            expected_bytes = int(runtime_asset["size_bytes"])
                            reusable = min(int(row.get("partial_bytes_reusable", 0)), expected_bytes)
                            if int(row["bytes"]) != expected_bytes:
                                raise ValueError("official SDK size differs from its pinned manifest")
                            required_free = (expected_bytes - reusable + GENIEX_SDK_UNPACKED_BYTES +
                                             args.asset_space_reserve_bytes)
                            disk_code = (
                                "import json,shutil,sys;u=shutil.disk_usage(sys.argv[1]);r=int(sys.argv[2]);"
                                "print(json.dumps({'free_bytes':u.free,'required_bytes':r}));"
                                "sys.exit(0 if u.free>=r else 43)"
                            )
                            disk_status, disk_stdout, _stderr = _remote_exec(
                                client, "python3 -c " + shlex.quote(disk_code) + " " +
                                shlex.quote(remote_root) + " " + str(required_free), timeout_s=15,
                                log_path=run_dir / "runtime-probe-console.log")
                            disk = json.loads(disk_stdout.strip().splitlines()[-1])
                            report["runtime_asset_preflight"] = {
                                "asset_id": GENIEX_SDK_ASSET_ID,
                                "archive_bytes": expected_bytes,
                                "unpacked_bytes": GENIEX_SDK_UNPACKED_BYTES,
                                "required_free_bytes": required_free,
                                "device_free_bytes": disk.get("free_bytes"),
                                "resumed_host_partial_bytes": sum(int(item["bytes"]) for item in staged),
                            }
                            if disk_status == 0:
                                fetch_args = [
                                    "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                                    "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                                    "--root", remote_assets, "--include", GENIEX_SDK_ASSET_ID,
                                ]
                                fetch_status, _fetch_stdout, _stderr = _remote_exec(
                                    client, _device_https_proxy(
                                        " ".join(shlex.quote(item) for item in fetch_args),
                                        args.device_https_proxy_port),
                                    timeout_s=args.asset_timeout,
                                    log_path=run_dir / "runtime-download-console.log")
                                if fetch_status == 0:
                                    restore_args = [
                                        "python3", f"{remote_root}/tools/remote/restore_geniex_runtime.py",
                                        "--archive", remote_runtime_archive,
                                        "--destination", f"{remote_root}/runtime-v0.11.0",
                                        "--expected-sha256", GENIEX_SDK_SHA256,
                                        "--expected-unpacked-bytes", str(GENIEX_SDK_UNPACKED_BYTES),
                                    ]
                                    restore_status, restore_stdout, _stderr = _remote_exec(
                                        client, " ".join(shlex.quote(item) for item in restore_args),
                                        timeout_s=180, log_path=run_dir / "runtime-download-console.log")
                                    if restore_status == 0:
                                        try:
                                            runtime_restore = json.loads(restore_stdout.strip().splitlines()[-1])
                                        except (IndexError, json.JSONDecodeError):
                                            runtime_restore = {"status": "restored", "unpacked_bytes": GENIEX_SDK_UNPACKED_BYTES}
                                        runtime_root = f"{remote_root}/runtime-v0.11.0/geniex"
                                        report["runtime_restore"] = {
                                            **runtime_restore, "version": "v0.11.0",
                                            "asset_id": GENIEX_SDK_ASSET_ID,
                                            "source": "pinned official SDK downloaded and verified on device",
                                        }
                                        _delete_asset_partials([GENIEX_SDK_ASSET_ID])
                                        runtime_download_ok = True
                                    else:
                                        runtime_download_reason = "official_sdk_restore_failed"
                                else:
                                    runtime_download_reason = "official_sdk_device_download_failed"
                            else:
                                runtime_download_reason = "official_sdk_disk_preflight_failed"
                        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                            runtime_download_reason = "official_sdk_metadata_invalid"

                    if not runtime_download_ok:
                        sftp = client.open_sftp()
                        try:
                            report["runtime_partial_recovery"] = _collect_asset_partials(
                                sftp, remote_root, [GENIEX_SDK_ASSET_ID])
                            report["runtime_ledger_recovered"] = _collect_asset_ledger(
                                sftp, remote_root, run_dir)
                            for path in (f"{remote_runtime_archive}.part",
                                         f"{remote_runtime_archive}.part.json",
                                         remote_runtime_archive):
                                try:
                                    sftp.remove(path)
                                except OSError:
                                    pass
                        except Exception as recovery_error:
                            report["runtime_partial_recovery_error"] = type(recovery_error).__name__
                        finally:
                            sftp.close()

                        runtime_archive = ROOT / GENIEX_RUNTIME_ARCHIVE
                        if (not runtime_archive.is_file() or
                                _sha256(runtime_archive) != GENIEX_RUNTIME_SHA256):
                            raise RuntimeError(
                                "device-side GenieX SDK download failed and the pinned D-drive runtime fallback is unavailable")
                        fallback_unpacked = _tar_unpacked_bytes(runtime_archive)
                        fallback_required = (runtime_archive.stat().st_size + fallback_unpacked +
                                             args.asset_space_reserve_bytes)
                        disk_code = (
                            "import json,shutil,sys;u=shutil.disk_usage(sys.argv[1]);r=int(sys.argv[2]);"
                            "print(json.dumps({'free_bytes':u.free,'required_bytes':r}));"
                            "sys.exit(0 if u.free>=r else 43)"
                        )
                        disk_status, disk_stdout, _stderr = _remote_exec(
                            client, "python3 -c " + shlex.quote(disk_code) + " " +
                            shlex.quote(remote_root) + " " + str(fallback_required), timeout_s=15,
                            log_path=run_dir / "runtime-probe-console.log")
                        try:
                            disk = json.loads(disk_stdout.strip().splitlines()[-1])
                        except (IndexError, json.JSONDecodeError) as error:
                            raise RuntimeError("runtime fallback disk check returned invalid metadata") from error
                        if disk_status != 0:
                            raise RuntimeError("neither the official runtime archive nor its local fallback fits the target disk budget")
                        remote_fallback_archive = f"{remote_root}/runtime-package.tar.gz"
                        sftp = client.open_sftp()
                        try:
                            sftp.put(str(runtime_archive), remote_fallback_archive, confirm=True)
                            sftp.chmod(remote_fallback_archive, 0o600)
                        finally:
                            sftp.close()
                        remote_hash_code = (
                            "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');"
                            "[h.update(b) for b in iter(lambda:f.read(1048576),b'')];print(h.hexdigest())"
                        )
                        hash_status, hash_stdout, _stderr = _remote_exec(
                            client, "python3 -c " + shlex.quote(remote_hash_code) + " " +
                            shlex.quote(remote_fallback_archive), timeout_s=60,
                            log_path=run_dir / "deployment-console.log")
                        observed_hash = hash_stdout.strip().splitlines()[-1] if hash_stdout.strip() else ""
                        if hash_status != 0 or observed_hash != GENIEX_RUNTIME_SHA256:
                            raise RuntimeError("temporary GenieX runtime fallback upload hash mismatch")
                        restore_args = [
                            "python3", f"{remote_root}/tools/remote/restore_geniex_runtime.py",
                            "--archive", remote_fallback_archive,
                            "--destination", f"{remote_root}/runtime-v0.11.0",
                            "--expected-sha256", GENIEX_RUNTIME_SHA256,
                            "--expected-unpacked-bytes", str(fallback_unpacked),
                        ]
                        restore_status, restore_stdout, _stderr = _remote_exec(
                            client, " ".join(shlex.quote(item) for item in restore_args),
                            timeout_s=180, log_path=run_dir / "deployment-console.log")
                        if restore_status != 0:
                            raise RuntimeError("pinned D-drive GenieX runtime fallback extraction failed")
                        try:
                            runtime_restore = json.loads(restore_stdout.strip().splitlines()[-1])
                        except (IndexError, json.JSONDecodeError):
                            runtime_restore = {"status": "restored", "unpacked_bytes": fallback_unpacked}
                        runtime_root = f"{remote_root}/runtime-v0.11.0/geniex"
                        report["runtime_restore"] = {
                            **runtime_restore, "version": "v0.11.0",
                            "sha256": GENIEX_RUNTIME_SHA256,
                            "source": "SHA-verified D-drive fallback after " + runtime_download_reason,
                        }
                else:
                    report["runtime_restore"] = {
                        "source": "preinstalled target runtime",
                        "status": "required_files_present",
                        "version": "unknown_until_target_probe_or_startup_log",
                    }
            runtime_probe_code = (
                "import json,os,sys; r=sys.argv[1];s=sys.argv[2]; "
                "checks={'service_executable':os.path.isfile(s) and os.access(s,os.X_OK),"
                "'geniex_c_api':os.path.isfile(os.path.join(r,'lib','libgeniex.so')) ,"
                "'llama_cpp_plugin':os.path.isfile(os.path.join(r,'lib','llama_cpp','libgeniex_plugin.so')) ,"
                "'htp_v73_backend':os.path.isfile(os.path.join(r,'lib','llama_cpp','libggml-htp-v73.so'))}; "
                "print(json.dumps(checks)); sys.exit(0 if all(checks.values()) else 46)"
            )
            runtime_probe_command = (
                "python3 -c " + shlex.quote(runtime_probe_code) + " " +
                shlex.quote(runtime_root) + " " + shlex.quote(service_binary))
            status, runtime_stdout, _stderr = _remote_exec(
                client, runtime_probe_command, timeout_s=20,
                log_path=run_dir / "runtime-probe-console.log")
            try:
                runtime_checks = json.loads(runtime_stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                raise RuntimeError("GenieX runtime preflight returned invalid metadata") from error
            report["geniex_runtime_preflight"] = runtime_checks
            if status != 0:
                raise RuntimeError("GenieX v0.11.0 service/runtime files are unavailable; model downloads were stopped")

        visual_asset_paths: dict[str, str] | None = None
        if asset_ids or args.restore_minilm or args.restore_llm_gguf:
            sftp = client.open_sftp()
            try:
                report["device_asset_ledger_seeded"] = _stage_asset_ledger(sftp, remote_root)
            finally:
                sftp.close()
        remote_asset_ids = [asset_id for asset_id in asset_ids if not (
            args.upload_local_vlm_cache and asset_id in VLM_CACHE_FILES)]
        if asset_ids:
            if any(_asset_catalog()[asset_id].get("role", "").endswith("qairt_bundle")
                   for asset_id in asset_ids) and not args.assets_only:
                raise ValueError("QAIrt model bundles are download-only today; use --assets-only until the QAIrt adapter/runtime is verified")
        if remote_asset_ids:
            sftp = client.open_sftp()
            try:
                report["resumed_host_partials"] = _stage_asset_partials(
                    sftp, remote_root, remote_asset_ids)
            finally:
                sftp.close()
            preflight_args = [
                "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                "--root", f"{remote_root}/assets", "--preflight",
            ]
            for asset_id in remote_asset_ids:
                preflight_args.extend(("--include", asset_id))
            preflight_command = _device_https_proxy(
                " ".join(shlex.quote(item) for item in preflight_args),
                args.device_https_proxy_port)
            status, preflight_stdout, _stderr = _remote_exec(
                client, preflight_command, timeout_s=90,
                log_path=run_dir / "deployment-console.log")
            hf_asset_ids = [asset_id for asset_id in remote_asset_ids
                            if urlparse(str(catalog[asset_id].get("url", ""))).hostname ==
                            "huggingface.co"]
            # Resolve HF assets to their official short-lived CDN URLs on the host,
            # then let the target-side downloader issue the Range requests directly
            # to that CDN. The signed links stay in SSH stdin and are never logged.
            asset_cdn_urls = _resolve_hf_cdn_urls(hf_asset_ids, catalog) if hf_asset_ids else {}
            if len(asset_cdn_urls) != len(hf_asset_ids):
                raise RuntimeError("could not resolve every model asset to the allowlisted official CDN")
            preflight_stdin = json.dumps(asset_cdn_urls, separators=(",", ":")) if asset_cdn_urls else None
            preflight_command += " --url-overrides-stdin" if asset_cdn_urls else ""
            status, preflight_stdout, _stderr = _remote_exec(
                client, preflight_command, timeout_s=90,
                log_path=run_dir / "deployment-console.log", stdin_data=preflight_stdin)
            if status != 0:
                # Refresh CDN signatures once if the preflight raced their expiry.
                if not asset_cdn_urls:
                    raise RuntimeError("device-side asset size preflight failed; model downloads were not started")
                asset_cdn_urls = _resolve_hf_cdn_urls(hf_asset_ids, catalog)
                if len(asset_cdn_urls) != len(hf_asset_ids):
                    raise RuntimeError("could not refresh official CDN asset links")
                status, preflight_stdout, _stderr = _remote_exec(
                    client, preflight_command, timeout_s=90,
                    log_path=run_dir / "deployment-console.log",
                    stdin_data=json.dumps(asset_cdn_urls, separators=(",", ":")))
                if status != 0:
                    raise RuntimeError("device-side official CDN preflight failed; model downloads were not started")
            try:
                preflight = json.loads(preflight_stdout.strip().splitlines()[-1])
                rows = preflight["preflight_artifacts"]
                if not isinstance(rows, list) or len(rows) != len(remote_asset_ids):
                    raise ValueError("asset preflight row count mismatch")
            except (IndexError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
                raise RuntimeError("device-side asset preflight returned invalid metadata") from error
            remaining_bytes = sum(max(
                0, int(row["bytes"]) - int(row.get("partial_bytes_reusable", 0)) -
                min(int(row.get("existing_target_bytes", 0)), int(row["bytes"])))
                for row in rows)
            local_upload_bytes = 0
            if args.restore_llm_gguf or args.restore_minilm:
                local_upload_bytes += LLAMA_SERVER.stat().st_size
            if args.restore_minilm:
                local_upload_bytes += MINILM_MODEL.stat().st_size
            if args.upload_local_vlm_cache:
                local_upload_bytes += sum((ROOT / VLM_CACHE_ROOT / name).stat().st_size
                                          for name, _sha in VLM_CACHE_FILES.values())
            required_free = remaining_bytes + args.asset_space_reserve_bytes + local_upload_bytes
            disk_code = (
                "import json,shutil,sys;u=shutil.disk_usage(sys.argv[1]);r=int(sys.argv[2]); "
                "print(json.dumps({'free_bytes':u.free,'required_bytes':r})); sys.exit(0 if u.free>=r else 43)"
            )
            status, disk_stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(disk_code) + " " +
                shlex.quote(remote_root) + " " + str(required_free), timeout_s=15,
                log_path=run_dir / "deployment-console.log")
            try:
                disk = json.loads(disk_stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                raise RuntimeError("device-side asset disk check returned invalid metadata") from error
            report["asset_preflight"] = {"artifacts": rows, "required_free_bytes": required_free,
                                         "device_free_bytes": disk.get("free_bytes")}
            if status != 0:
                raise RuntimeError("not enough target disk space for selected assets plus safety reserve")
            (run_dir / "deployment.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            fetch_args = [
                "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                "--root", f"{remote_root}/assets",
            ]
            for asset_id in remote_asset_ids:
                fetch_args.extend(("--include", asset_id))
            fetch_stdin = None
            if asset_cdn_urls:
                refreshed_cdn_urls = _resolve_hf_cdn_urls(hf_asset_ids, catalog)
                if len(refreshed_cdn_urls) != len(hf_asset_ids):
                    raise RuntimeError("could not refresh temporary CDN handoff for model download")
                fetch_args.append("--url-overrides-stdin")
                fetch_stdin = json.dumps(refreshed_cdn_urls, separators=(",", ":"))
            fetch_command = _device_https_proxy(
                " ".join(shlex.quote(item) for item in fetch_args),
                args.device_https_proxy_port)
            status, _stdout, _stderr = _remote_exec(
                client, fetch_command, timeout_s=args.asset_timeout,
                log_path=run_dir / "deployment-console.log", stdin_data=fetch_stdin)
            if status != 0:
                sftp = client.open_sftp()
                try:
                    report["recovered_host_partials"] = _collect_asset_partials(
                        sftp, remote_root, remote_asset_ids)
                    report["checksum_ledger_recovered"] = _collect_asset_ledger(
                        sftp, remote_root, run_dir)
                except Exception as recovery_error:
                    report["partial_recovery_error"] = type(recovery_error).__name__
                finally:
                    sftp.close()
                (run_dir / "deployment.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError("device-side model asset restore failed; partial downloads remain resumable")

            if args.restore_siglip2:
                qnn_gpu_library = "/usr/lib/libQnnGpu.so"
                restore_visual_args = [
                    "python3", f"{remote_root}/tools/remote/restore_siglip2_assets.py",
                    "--archive", f"{remote_root}/assets/siglip2-qnn_dlc-w8a16-v0.61.0.zip",
                    "--tokenizer", f"{remote_root}/assets/siglip2-tokenizer.json",
                    "--output-root", f"{remote_root}/assets",
                    "--qairt-version", args.siglip2_qairt_version,
                    "--backend-library", qnn_gpu_library,
                    "--model-id", "siglip2-base-patch16-224-qnn-w8a16-v0.61.0",
                ]
                restore_visual_status, restore_visual_stdout, _stderr = _remote_exec(
                    client, " ".join(shlex.quote(item) for item in restore_visual_args),
                    timeout_s=600, log_path=run_dir / "visual-asset-restore-console.log")
                try:
                    visual_restore = json.loads(restore_visual_stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    visual_restore = {"status": "restore_result_unavailable"}
                report["siglip2_optional_restore"] = visual_restore
                if (restore_visual_status == 0 and visual_restore.get("status") == "ready"
                        and visual_restore.get("image_model_path")
                        and visual_restore.get("text_model_path")):
                    visual_asset_paths = {
                        "image": str(visual_restore["image_model_path"]),
                        "text": str(visual_restore["text_model_path"]),
                        "tokenizer": f"{remote_root}/assets/siglip2-tokenizer.json",
                        "model_id": "siglip2-base-patch16-224-qnn-w8a16-v0.61.0",
                        "model_family": "siglip2_qnn",
                        "backend_library": qnn_gpu_library,
                    }
                else:
                    report["siglip2_optional_restore"]["application_action"] = (
                        "existing VLM/LLM/text-memory path retained; visual channel not activated")

        if args.restore_openai_clip:
            clip_asset_ids = [
                "openai-clip-vit-b-32-vision-onnx",
                "openai-clip-vit-b-32-text-onnx",
                "openai-clip-vit-b-32-vocab",
                "openai-clip-vit-b-32-merges",
            ]
            clip_status: dict = {"status": "starting", "asset_ids": clip_asset_ids}
            clip_remote_assets = f"{remote_root}/assets"
            try:
                catalog = _asset_catalog()
                sftp = client.open_sftp()
                try:
                    _stage_asset_ledger(sftp, remote_root)
                    clip_status["resumed_host_partials"] = _stage_asset_partials(
                        sftp, remote_root, clip_asset_ids)
                finally:
                    sftp.close()

                hf_clip_ids = [asset_id for asset_id in clip_asset_ids
                               if urlparse(str(catalog[asset_id]["url"])).hostname ==
                               "huggingface.co"]
                # The board successfully downloaded these pinned assets from this
                # HTTPS mirror when direct access to Hugging Face was unavailable.
                clip_cdn_urls = {asset_id: str(catalog[asset_id]["url"]).replace(
                    "https://huggingface.co/", "https://hf-mirror.com/", 1)
                    for asset_id in hf_clip_ids}
                preflight_args = [
                    "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                    "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                    "--root", clip_remote_assets, "--preflight",
                ]
                for asset_id in clip_asset_ids:
                    preflight_args.extend(("--include", asset_id))
                if clip_cdn_urls:
                    preflight_args.append("--url-overrides-stdin")
                preflight_status, preflight_stdout, _stderr = _remote_exec(
                    client, _device_https_proxy(
                        " ".join(shlex.quote(item) for item in preflight_args),
                        args.device_https_proxy_port),
                    timeout_s=120, log_path=run_dir / "clip-asset-restore-console.log",
                    stdin_data=(json.dumps(clip_cdn_urls, separators=(",", ":"))
                                if clip_cdn_urls else None))
                if preflight_status != 0:
                    raise RuntimeError("CLIP board-side asset preflight failed")
                preflight = json.loads(preflight_stdout.strip().splitlines()[-1])
                rows = preflight["preflight_artifacts"]
                if not isinstance(rows, list) or len(rows) != len(clip_asset_ids):
                    raise ValueError("CLIP asset preflight row count mismatch")
                remaining_bytes = sum(max(
                    0, int(row["bytes"]) - int(row.get("partial_bytes_reusable", 0)) -
                    min(int(row.get("existing_target_bytes", 0)), int(row["bytes"])))
                    for row in rows)
                # The INT64-to-INT32 adapter writes one temporary text-model-sized
                # file before atomic replacement. Include it in target disk admission.
                patch_temporary_bytes = int(catalog["openai-clip-vit-b-32-text-onnx"]["size_bytes"])
                required_free = (remaining_bytes + patch_temporary_bytes +
                                 args.asset_space_reserve_bytes)
                disk_code = (
                    "import json,shutil,sys;u=shutil.disk_usage(sys.argv[1]);r=int(sys.argv[2]); "
                    "print(json.dumps({'free_bytes':u.free,'required_bytes':r})); "
                    "sys.exit(0 if u.free>=r else 43)"
                )
                disk_status, disk_stdout, _stderr = _remote_exec(
                    client, "python3 -c " + shlex.quote(disk_code) + " " +
                    shlex.quote(remote_root) + " " + str(required_free), timeout_s=15,
                    log_path=run_dir / "clip-asset-restore-console.log")
                disk = json.loads(disk_stdout.strip().splitlines()[-1])
                clip_status["preflight"] = {
                    "artifacts": rows, "required_free_bytes": required_free,
                    "device_free_bytes": disk.get("free_bytes"),
                }
                if disk_status != 0:
                    raise RuntimeError("insufficient target disk space for CLIP plus safety reserve")

                fetch_args = [
                    "python3", f"{remote_root}/tools/remote/fetch_artifact.py",
                    "--manifest", f"{remote_root}/{ASSET_MANIFEST.as_posix()}",
                    "--root", clip_remote_assets,
                ]
                for asset_id in clip_asset_ids:
                    fetch_args.extend(("--include", asset_id))
                refreshed_cdn_urls = clip_cdn_urls
                if refreshed_cdn_urls:
                    fetch_args.append("--url-overrides-stdin")
                fetch_status, _fetch_stdout, _stderr = _remote_exec(
                    client, _device_https_proxy(
                        " ".join(shlex.quote(item) for item in fetch_args),
                        args.device_https_proxy_port),
                    timeout_s=args.asset_timeout,
                    log_path=run_dir / "clip-asset-restore-console.log",
                    stdin_data=(json.dumps(refreshed_cdn_urls, separators=(",", ":"))
                                if refreshed_cdn_urls else None))
                if fetch_status != 0:
                    raise RuntimeError("board-side CLIP download or SHA256 verification failed")

                ort_root = f"{remote_root}/assets/ort-python"
                ort_check = "PYTHONPATH=" + shlex.quote(ort_root) + " python3 -c " + shlex.quote("import onnxruntime; print(onnxruntime.__version__)")
                ort_status, ort_stdout, _stderr = _remote_exec(client, ort_check, timeout_s=20)
                if ort_status != 0:
                    install = ["python3", "-m", "pip", "install", "--no-deps", "--target", ort_root,
                               "onnxruntime==1.24.2", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--timeout", "15"]
                    ort_status, _stdout, _stderr = _remote_exec(
                        client, _device_https_proxy(" ".join(shlex.quote(x) for x in install), args.device_https_proxy_port),
                        timeout_s=180, log_path=run_dir / "clip-asset-restore-console.log")
                    if ort_status != 0:
                        raise RuntimeError("isolated board ONNX Runtime installation failed")
                    ort_status, ort_stdout, _stderr = _remote_exec(client, ort_check, timeout_s=20)
                    if ort_status != 0:
                        raise RuntimeError("isolated board ONNX Runtime import failed")
                clip_status.update({"status": "ready", "onnxruntime_version": ort_stdout.strip().splitlines()[-1],
                                    "text_model_adaptation": "none; original INT64 graph"})
                visual_asset_paths = {
                    "image": f"{clip_remote_assets}/clip-vit-b-32-vision.onnx",
                    "text": f"{clip_remote_assets}/clip-vit-b-32-text.onnx",
                    "vocab": f"{clip_remote_assets}/clip-vit-b-32-vocab.json",
                    "merges": f"{clip_remote_assets}/clip-vit-b-32-merges.txt",
                    "model_id": "openai-clip-vit-b-32-onnx",
                    "model_family": "openai_clip_cpu", "ort_python_path": ort_root,
                    "backend_library": "/usr/lib/libQnnGpu.so",
                }
                _delete_asset_partials(clip_asset_ids)
                sftp = client.open_sftp()
                try:
                    clip_status["checksum_ledger_recovered"] = _collect_asset_ledger(
                        sftp, remote_root, run_dir)
                finally:
                    sftp.close()
                report["restored_asset_ids"] = list(dict.fromkeys(
                    [*report.get("restored_asset_ids", []), *clip_asset_ids]))
            except Exception as error:
                clip_status.update({
                    "status": "optional_restore_failed",
                    "error_type": type(error).__name__,
                    "application_action": "continue the existing VLM/LLM/MiniLM product; visual retrieval stays disabled",
                })
                try:
                    sftp = client.open_sftp()
                    try:
                        clip_status["recovered_partials"] = _collect_asset_partials(
                            sftp, remote_root, clip_asset_ids)
                        clip_status["checksum_ledger_recovered"] = _collect_asset_ledger(
                            sftp, remote_root, run_dir)
                    finally:
                        sftp.close()
                except Exception as recovery_error:
                    clip_status["partial_recovery_error"] = type(recovery_error).__name__
            report["openai_clip_optional_restore"] = clip_status
            (run_dir / "deployment.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.upload_local_vlm_cache:
            local_vlm_uploads = []
            for asset_id, (filename, expected_hash) in VLM_CACHE_FILES.items():
                local = ROOT / VLM_CACHE_ROOT / filename
                if not local.is_file() or _sha256(local) != expected_hash:
                    raise RuntimeError("the previously verified D-drive Qwen3-VL-4B cache is missing or changed")
                local_vlm_uploads.append((asset_id, local, expected_hash))
            upload_bytes = sum(local.stat().st_size for _asset_id, local, _digest in local_vlm_uploads)
            disk_code = (
                "import json,shutil,sys;u=shutil.disk_usage(sys.argv[1]);r=int(sys.argv[2]); "
                "print(json.dumps({'free_bytes':u.free,'required_bytes':r})); sys.exit(0 if u.free>=r else 43)"
            )
            required_free = upload_bytes + args.asset_space_reserve_bytes
            status, disk_stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(disk_code) + " " +
                shlex.quote(remote_root) + " " + str(required_free), timeout_s=15,
                log_path=run_dir / "deployment-console.log")
            try:
                disk = json.loads(disk_stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                raise RuntimeError("device-side local-cache disk check returned invalid metadata") from error
            if status != 0:
                raise RuntimeError("not enough target disk space for the local VLM cache and reserve")
            sftp = client.open_sftp()
            try:
                _remote_mkdir(sftp, f"{remote_root}/assets")
                for _asset_id, local, _digest in local_vlm_uploads:
                    uploaded = _put_verified(sftp, local, f"{remote_root}/assets/{local.name}")
                    sftp.chmod(uploaded["path"], 0o600)
                    uploaded["asset_id"] = _asset_id
                    uploaded["hash_check_command"] = ""
                    report.setdefault("local_vlm_cache_uploads", []).append(uploaded)
            finally:
                sftp.close()
            remote_hash_code = (
                "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');"
                "[h.update(b) for b in iter(lambda:f.read(1048576),b'')];print(h.hexdigest())"
            )
            for asset_id, local, expected_hash in local_vlm_uploads:
                remote_path = f"{remote_root}/assets/{local.name}"
                status, stdout, _stderr = _remote_exec(
                    client, "python3 -c " + shlex.quote(remote_hash_code) + " " +
                    shlex.quote(remote_path), timeout_s=90,
                    log_path=run_dir / "deployment-console.log")
                observed = stdout.strip().splitlines()[-1] if stdout.strip() else ""
                if status != 0 or observed != expected_hash:
                    raise RuntimeError(f"local cache upload SHA256 mismatch for {asset_id}")
            ledger_entries = {asset_id: {
                "sha256": digest, "bytes": local.stat().st_size,
                "url": _asset_catalog()[asset_id]["url"],
                "verification": "D_drive_cache_hash_verified_after_SFTP"}
                for asset_id, local, digest in local_vlm_uploads}
            ledger_code = (
                "import json,os,sys; p=sys.argv[1];d={'schema_version':1,'assets':{}}; "
                "d=json.load(open(p)) if os.path.isfile(p) else d; "
                "d['assets'].update(json.loads(sys.argv[2]));t=p+'.tmp'; "
                "json.dump(d,open(t,'w'),indent=2);os.replace(t,p)"
            )
            remote_ledger = f"{remote_root}/assets/artifact-sha256-manifest.json"
            status, _stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(ledger_code) + " " +
                shlex.quote(remote_ledger) + " " + shlex.quote(json.dumps(ledger_entries)),
                timeout_s=20, log_path=run_dir / "deployment-console.log")
            if status != 0:
                raise RuntimeError("could not record verified local VLM cache checksums")
            report["local_vlm_cache_disk_preflight"] = {
                "required_free_bytes": required_free, "device_free_bytes": disk.get("free_bytes")}

        if asset_ids:
            report["restored_asset_ids"] = asset_ids
            sftp = client.open_sftp()
            try:
                report["checksum_ledger_recovered"] = _collect_asset_ledger(
                    sftp, remote_root, run_dir)
            finally:
                sftp.close()
            _delete_asset_partials(remote_asset_ids)
            if args.assets_only:
                report["asset_restore_only"] = True
                (run_dir / "deployment.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                from tools.run_manifest import create as create_run_manifest
                create_run_manifest(run_dir)
                print(f"Selected board assets downloaded and checksummed; no inference service started. Results: {run_dir}", flush=True)
                return 0

        uploaded_assets: list[dict] = []
        if args.restore_minilm or args.restore_llm_gguf:
            if args.restore_minilm and (not MINILM_MODEL.is_file() or _sha256(MINILM_MODEL) != "4ef97c88f2c23733eac808305cd51a7c077a49462f7bf0f8c398caed694c143c"):
                raise RuntimeError("the previously verified D-drive MiniLM cache is missing or changed")
            if not LLAMA_SERVER.is_file():
                raise FileNotFoundError("ARM64 llama-server build is missing; prepare the existing local build first")
            sftp = client.open_sftp()
            try:
                for directory in (f"{remote_root}/assets/bin", f"{remote_root}/assets/models"):
                    try:
                        sftp.mkdir(directory, mode=0o700)
                    except OSError:
                        sftp.stat(directory)
                uploaded_assets.append(_put_verified(
                    sftp, LLAMA_SERVER, f"{remote_root}/assets/bin/llama-server"))
                if args.restore_llm_gguf and args.llm_backend != "cpu":
                    if not (ROOT / GENIEX_LLM_SERVICE).is_file():
                        raise FileNotFoundError("ARM64 GenieX LLM service build is missing")
                    uploaded_assets.append(_put_verified(
                        sftp, ROOT / GENIEX_LLM_SERVICE,
                        f"{remote_root}/assets/bin/hexagonflow-geniex-llm-service"))
                if args.restore_minilm:
                    uploaded_assets.append(_put_verified(
                        sftp, MINILM_MODEL,
                        f"{remote_root}/assets/models/{MINILM_MODEL.name}"))
                sftp.chmod(f"{remote_root}/assets/bin/llama-server", 0o700)
                if args.restore_llm_gguf and args.llm_backend != "cpu":
                    sftp.chmod(f"{remote_root}/assets/bin/hexagonflow-geniex-llm-service", 0o700)
            finally:
                sftp.close()
            for item in uploaded_assets:
                remote_hash_command = (
                    "python3 -c " + shlex.quote(
                        "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');"
                        "[h.update(b) for b in iter(lambda:f.read(1048576),b'')];print(h.hexdigest())") +
                    " " + shlex.quote(item["path"])
                )
                status, stdout, _stderr = _remote_exec(client, remote_hash_command, timeout_s=60)
                observed = stdout.strip().splitlines()[-1] if stdout.strip() else ""
                if status != 0 or observed != item["sha256"]:
                    raise RuntimeError("local-cache upload SHA256 mismatch")
                item.pop("hash_check_command", None)
            report["local_cache_uploads"] = uploaded_assets
            remote_ledger = f"{remote_root}/assets/artifact-sha256-manifest.json"
            ledger_code = (
                "import json,os,sys; p=sys.argv[1]; d={'schema_version':1,'assets':{}}; "
                "d=json.load(open(p)) if os.path.isfile(p) else d; "
                "d['assets'].update(json.loads(sys.argv[2])); "
                "t=p+'.tmp'; json.dump(d,open(t,'w'),indent=2); os.replace(t,p)"
            )
            ledger_entries = {Path(item["path"]).name: {
                "sha256": item["sha256"], "bytes": item["bytes"],
                "verification": "host_cache_hash_verified_after_SFTP"}
                for item in uploaded_assets}
            status, _stdout, _stderr = _remote_exec(
                client, "python3 -c " + shlex.quote(ledger_code) + " " +
                shlex.quote(remote_ledger) + " " + shlex.quote(json.dumps(ledger_entries)),
                timeout_s=20)
            if status != 0:
                raise RuntimeError("could not persist local-fallback artifact checksums on device")
            sftp = client.open_sftp()
            try:
                _collect_asset_ledger(sftp, remote_root, run_dir)
            finally:
                sftp.close()

        if args.restore_minilm or args.restore_llm_gguf:
            sftp = client.open_sftp()
            try:
                _collect_asset_ledger(sftp, remote_root, run_dir)
            finally:
                sftp.close()

        py_compile = (
            f"cd {shlex.quote(remote_root)} && python3 -m py_compile "
            "edgememory/device_agent.py edgememory/device_store.py "
            "edgememory/embeddings.py edgememory/llama_server.py "
            "edgememory/visual_embeddings.py edgememory/hardware_profiler.py "
            "tools/remote/gstreamer_video_sampler.py tools/remote/fetch_artifact.py "
            "tools/remote/restore_geniex_runtime.py tools/remote/restore_siglip2_assets.py "
            "tools/remote/select_storage_root.py "
            "edgememory/scheduling.py edgememory/direct_execution.py"
        )
        status, _stdout, _stderr = _remote_exec(
            client, py_compile, timeout_s=30, log_path=run_dir / "deployment-console.log")
        if status != 0:
            raise RuntimeError("device-side Python compile check failed")

        remote_args = [
            "python3", "-m", "edgememory.device_agent", "serve",
            "--root", f"{remote_root}/state", "--host", "127.0.0.1", "--port", "8765",
            "--service-binary", service_binary,
            "--runtime-root", runtime_root,
            "--model-path", (f"{remote_root}/assets/Qwen3-VL-4B-Instruct-Q4_0.gguf"
                             if args.restore_vlm_gguf else profile.remote_model_path),
            "--mmproj-path", (f"{remote_root}/assets/mmproj-F16.gguf"
                              if args.restore_vlm_gguf else profile.remote_mmproj_path),
            "--model-id", profile.model_id, "--devices", args.devices,
            "--resource-domains", (args.resource_domains or ",".join(
                f"{device}=htp-selector-topology-unverified" for device in
                (item.strip() for item in args.devices.split(",") if item.strip()))),
            "--policy", args.policy,
            "--context", str(args.context or profile.context_length), "--threads", str(profile.threads),
            "--max-tokens", str(profile.max_tokens),
            "--job-workers", str(args.video_workers),
            "--pending-limit", str(args.pending_videos),
            "--preprocess-path", args.video_preprocess,
            "--decoder-io-mode", args.decoder_io_mode,
        ]
        if args.decoder_factory:
            remote_args.extend(("--decoder-factory", args.decoder_factory))
        if visual_asset_paths:
            remote_args.extend((
                "--visual-image-model-path", visual_asset_paths["image"],
                "--visual-text-model-path", visual_asset_paths["text"],
                "--visual-model-family", visual_asset_paths["model_family"],
                "--visual-model-id", visual_asset_paths["model_id"],
                "--visual-backend-library", visual_asset_paths["backend_library"],
            ))
            if visual_asset_paths["model_family"] in {"openai_clip_onnx", "openai_clip_cpu"}:
                remote_args.extend((
                    "--visual-tokenizer-vocab-path", visual_asset_paths["vocab"],
                    "--visual-tokenizer-merges-path", visual_asset_paths["merges"],
                    "--visual-execution-provider", "cpu",
                    "--visual-ort-python-path", visual_asset_paths["ort_python_path"],
                    "--visual-backend-family", "4",
                    "--visual-resource-domain", "",
                    "--visual-threads", "2",
                    "--visual-min-free-bytes", str(args.visual_min_free_bytes),
                ))
            else:
                remote_args.extend(("--visual-tokenizer-path", visual_asset_paths["tokenizer"],))
        if args.hexagon_mbuf is not None:
            remote_args.extend(("--hexagon-mbuf", args.hexagon_mbuf))
        if args.hexagon_hostbuf is not None:
            remote_args.extend(("--hexagon-hostbuf", str(args.hexagon_hostbuf)))
        if args.hexagon_devices is not None:
            remote_args.extend(("--hexagon-devices", args.hexagon_devices))
        if args.vit_device_id is not None:
            remote_args.extend(("--vit-device-id", args.vit_device_id))
        if args.device_selftest:
            remote_args.append("--selftest")
        if args.restore_llm_gguf:
            remote_args.extend((
                "--llm-server-binary", f"{remote_root}/assets/bin/llama-server",
                "--llm-model-path", f"{remote_root}/assets/Qwen3-4B-Instruct-2507-Q4_0.gguf",
                "--llm-min-free-bytes", str(args.llm_min_free_bytes),
            ))
            if args.llm_backend != "cpu":
                remote_args.extend((
                    "--llm-geniex-service-binary",
                    f"{remote_root}/assets/bin/hexagonflow-geniex-llm-service",
                    "--llm-runtime-root", runtime_root,
                    "--llm-device-id", args.llm_device_id,
                    "--llm-geniex-context", str(args.llm_geniex_context),
                    "--llm-geniex-threads", str(args.llm_geniex_threads),
                    "--llm-geniex-max-tokens", str(args.llm_geniex_max_tokens),
                    "--llm-resource-domain", args.llm_resource_domain,
                ))
        if args.restore_minilm:
            remote_args.extend((
                "--embedding-server-binary", f"{remote_root}/assets/bin/llama-server",
                "--embedding-server-model-path", f"{remote_root}/assets/models/{MINILM_MODEL.name}",
                "--embedding-min-free-bytes", str(args.embedding_min_free_bytes),
            ))
        pid_file = f"{remote_root}/control/device-agent.pid"
        start_code = (
            f"cd {shlex.quote(remote_root)} || exit 1\n"
            "mkdir -p logs control || exit 1\n"
            "nohup env EDGEMEMORY_DIRECT_EXECUTION=" + ("1" if args.execution_mode == "direct" else "0") + " " + " ".join(shlex.quote(item) for item in remote_args) +
            " > logs/device-agent-stdout.log 2>&1 < /dev/null &\n"
            "service_pid=$!\n"
            f"printf '%s\\n' \"$service_pid\" > {shlex.quote(pid_file)}\n"
            f"chmod 600 {shlex.quote(pid_file)}\n"
            "printf '%s\\n' \"$service_pid\""
        )
        status, stdout, _stderr = _remote_exec(
            client, start_code, timeout_s=15, log_path=run_dir / "deployment-console.log")
        if status != 0:
            raise RuntimeError("device-resident HexagonFlow service did not start")
        pid_text = stdout.strip().splitlines()[-1] if stdout.strip() else ""
        report["device_agent_pid"] = int(pid_text) if pid_text.isdigit() else None
        (run_dir / "deployment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        try:
            health = _wait_for_health(client, timeout_s=args.start_timeout)
        except Exception as startup_error:
            report["startup_failure_type"] = type(startup_error).__name__
            try:
                report["startup_failure_files_recovered"] = _collect_device(
                    client, remote_root, run_dir, collected_files)
            except Exception as recovery_error:
                report["startup_failure_recovery_error_type"] = type(recovery_error).__name__
            (run_dir / "deployment.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            from tools.run_manifest import create as create_run_manifest
            create_run_manifest(run_dir)
            raise RuntimeError(
                "device service did not become healthy; startup logs were collected when reachable"
            ) from startup_error
        report["first_health_response"] = health
        (run_dir / "deployment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Device UI is running on IQ-9075; local forwarding will use 127.0.0.1:{args.local_port}", flush=True)
        print(f"Device health: {json.dumps(health, ensure_ascii=False)}", flush=True)

        transport = client.get_transport()
        assert transport is not None
        class ForwardServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
        try:
            proxy_server = ForwardServer(("127.0.0.1", args.local_port),
                                         _forward_handler(transport, 8765))
        except OSError:
            for local_port in range(args.local_port + 1, args.local_port + 10):
                try:
                    proxy_server = ForwardServer(("127.0.0.1", local_port),
                                                 _forward_handler(transport, 8765))
                    args.local_port = local_port
                    break
                except OSError:
                    continue
            if proxy_server is None:
                raise RuntimeError("no free local port is available for the device web UI")
        proxy_thread = threading.Thread(target=proxy_server.serve_forever,
                                        kwargs={"poll_interval": 0.5}, daemon=True)
        proxy_thread.start()
        print(f"Board-resident EdgeMemory UI: http://127.0.0.1:{args.local_port}/", flush=True)

        try:
            next_collect = time.monotonic() + args.collect_interval_s
            while client.get_transport() and client.get_transport().is_active():
                time.sleep(1.0)
                # Recover recent state and raw evidence to persistent D while
                # the QDC machine is still live; the caller can repeat later.
                if time.monotonic() >= next_collect:
                    _collect_device(client, remote_root, run_dir, collected_files)
                    next_collect = time.monotonic() + args.collect_interval_s
        except KeyboardInterrupt:
            pass
        finally:
            if proxy_server:
                proxy_server.shutdown()
                proxy_server.server_close()
            _collect_device(client, remote_root, run_dir, collected_files)
        from tools.run_manifest import create as create_run_manifest
        create_run_manifest(run_dir)
        print(f"Device results copied to {run_dir}", flush=True)
        return 0
    finally:
        password = ""
        if client:
            client.close()


def collect_release(args) -> int:
    """Recover a previous temporary-device release before it is replaced or cleared."""
    profile = load_qdc_geniex_profile((ROOT / args.profile).resolve())
    local_root = (ROOT / args.output).resolve()
    remote_root = args.release_root.rstrip("/")
    if not _is_allowed_release_root(remote_root, profile.remote_work_root):
        raise ValueError("release root must be a named HexagonFlow deployment under a configured storage candidate")
    if not sys.stdin.isatty():
        raise RuntimeError("run this command in an interactive terminal for the hidden password prompt")
    password = getpass("QDC device password: ")
    if not password:
        raise RuntimeError("QDC device password was empty")
    client = None
    try:
        client = _connect(profile, password)
        password = ""
        copied = _collect_device(client, remote_root, local_root, {})
        from tools.run_manifest import create as create_run_manifest
        manifest = create_run_manifest(local_root)
        print(f"Recovered {copied} device files to {local_root}", flush=True)
        print(f"Local manifest: {manifest}", flush=True)
        return 0
    finally:
        password = ""
        if client:
            client.close()


def stop_service(args) -> int:
    """Stop only the HexagonFlow agent recorded for one temporary release root."""
    remote_root = args.release_root.rstrip("/")
    profile = load_qdc_geniex_profile((ROOT / args.profile).resolve())
    if not _is_allowed_release_root(remote_root, profile.remote_work_root):
        raise ValueError("stop requires a named HexagonFlow release under a configured storage candidate")
    if not sys.stdin.isatty():
        raise RuntimeError("run stop in an interactive terminal for the hidden password prompt")
    password = getpass("QDC device password: ")
    if not password:
        raise RuntimeError("QDC device password was empty")

    run_dir = ROOT / args.output / f"device-stop-{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    stop_code = "\n".join((
        "import json,os,signal,sys,time",
        "root=sys.argv[1]; path=os.path.join(root,'control','device-agent.pid')",
        "try: raw=open(path,encoding='ascii').read().strip()",
        "except FileNotFoundError: print(json.dumps({'status':'already_stopped'})); raise SystemExit(0)",
        "if not raw.isdigit(): raise SystemExit('invalid HexagonFlow PID record')",
        "pid=int(raw); proc='/proc/%d/cmdline'%pid",
        "try: argv=open(proc,'rb').read().split(b'\\0')",
        "except FileNotFoundError: os.unlink(path); print(json.dumps({'status':'already_exited'})); raise SystemExit(0)",
        "expected=os.fsencode(os.path.join(root,'state'))",
        "if b'edgememory.device_agent' not in argv or b'serve' not in argv or expected not in argv: raise SystemExit('PID record does not identify this HexagonFlow release; refusing to signal')",
        "os.kill(pid,signal.SIGTERM); deadline=time.monotonic()+15",
        "while time.monotonic()<deadline:",
        " try: state=open('/proc/%d/stat'%pid,'rb').read().split(b')',1)[1].strip().split()[0]",
        " except FileNotFoundError: break",
        " if state==b'Z': break",
        " time.sleep(.2)",
        "else: raise SystemExit('HexagonFlow agent did not stop within 15 seconds; PID record retained')",
        "os.unlink(path); print(json.dumps({'status':'stopped','pid':pid}))",
    ))
    client = None
    try:
        client = _connect(profile, password)
        password = ""
        status, stdout, stderr = _remote_exec(
            client, "python3 -c " + shlex.quote(stop_code) + " " + shlex.quote(remote_root),
            timeout_s=25, log_path=run_dir / "stop-console.log")
        (run_dir / "stop-result.json").write_text(json.dumps({
            "schema_version": 1, "remote_release_root": remote_root,
            "exit_status": status, "stdout": stdout.strip(),
            "stderr_type": "" if not stderr else "remote_command_error",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied = _collect_device(client, remote_root, run_dir, {})
        from tools.run_manifest import create as create_run_manifest
        create_run_manifest(run_dir)
        print(f"Device agent stop status={status}; recovered {copied} files to {run_dir}", flush=True)
        return status
    finally:
        password = ""
        if client:
            client.close()


def _is_allowed_release_root(remote_root: str, configured_work_root: str) -> bool:
    path = PurePosixPath(remote_root)
    if not path.is_absolute() or ".." in path.parts or "" in path.parts:
        return False
    if not path.name.startswith("hexagonflow-device-"):
        return False
    allowed_parents = {
        "/tmp", "/root/hexagonflow", "/var/lib/hexagonflow",
        "/data/hexagonflow", "/mnt/data/hexagonflow",
    }
    configured = PurePosixPath(configured_work_root)
    if configured.is_absolute() and ".." not in configured.parts:
        allowed_parents.add(configured.as_posix().rstrip("/"))
    return path.parent.as_posix().rstrip("/") in allowed_parents


def push_artifacts(args) -> int:
    """Copy explicitly selected D-drive artifacts into a temporary QDC /tmp path."""
    if not args.local_file or not args.remote_path or len(args.local_file) != len(args.remote_path):
        raise ValueError("push requires matching repeated --local-file and --remote-path options")
    pairs = []
    for local_arg, remote in zip(args.local_file, args.remote_path):
        local = (ROOT / local_arg).resolve() if not local_arg.is_absolute() else local_arg.resolve()
        try:
            local.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("artifact source must remain within the D-drive project") from error
        if not local.is_file():
            raise FileNotFoundError(f"artifact source is not a file: {local.name}")
        remote = remote.replace("\\", "/")
        if (not remote.startswith("/tmp/hexagonflow-artifacts/")
                or ".." in remote.split("/") or remote.endswith("/")):
            raise ValueError("artifact destination must be a file under /tmp/hexagonflow-artifacts/")
        pairs.append((local, remote))

    profile = load_qdc_geniex_profile((ROOT / args.profile).resolve())
    if not sys.stdin.isatty():
        raise RuntimeError("run push in an interactive terminal for the hidden password prompt")
    password = getpass("QDC device password: ")
    if not password:
        raise RuntimeError("QDC device password was empty")
    client = None
    results = []
    try:
        client = _connect(profile, password)
        password = ""
        sftp = client.open_sftp()
        try:
            for local, remote in pairs:
                parent = posixpath.dirname(remote)
                missing = []
                while parent and parent != "/":
                    try:
                        sftp.stat(parent)
                        break
                    except OSError:
                        missing.append(parent)
                        parent = posixpath.dirname(parent)
                for directory in reversed(missing):
                    try:
                        sftp.mkdir(directory, mode=0o700)
                    except OSError:
                        sftp.stat(directory)
                sftp.put(str(local), remote, confirm=True)
                sftp.chmod(remote, 0o600)
                expected = _sha256(local)
                code = (
                    "import hashlib,sys; h=hashlib.sha256(); "
                    "f=open(sys.argv[1],'rb'); "
                    "[h.update(b) for b in iter(lambda:f.read(1048576),b'')]; "
                    "print(h.hexdigest())"
                )
                status, stdout, _stderr = _remote_exec(
                    client, "python3 -c " + shlex.quote(code) + " " + shlex.quote(remote),
                    timeout_s=60)
                observed = stdout.strip().splitlines()[-1] if stdout.strip() else ""
                if status != 0 or observed != expected:
                    raise RuntimeError(f"uploaded artifact hash verification failed: {local.name}")
                results.append({"name": local.name, "bytes": local.stat().st_size,
                                "sha256": expected, "verified": True})
        finally:
            sftp.close()
        print(json.dumps({"execution_target": "current user-started QDC device",
                          "artifacts": results}, ensure_ascii=False))
        return 0
    finally:
        password = ""
        if client:
            client.close()


def _parser():
    parser = argparse.ArgumentParser(prog="qdc_device")
    parser.add_argument("--profile", type=Path,
                        default=Path(".local/edgememory/qdc-profile-4b-live.json"))
    parser.add_argument("--devices", default="HTP0")
    parser.add_argument("--resource-domains",
                        default="")
    parser.add_argument("--context", type=int,
                        help="VLM context override for controlled single-device initialization checks")
    parser.add_argument("--hexagon-mbuf",
                        help="explicit GGML_HEXAGON_MBUF override (leave unset for GenieX v0.11.0 default)")
    parser.add_argument("--hexagon-hostbuf", type=int, choices=(0, 1),
                        help="explicit GGML_HEXAGON_HOSTBUF override for a controlled diagnostic run")
    parser.add_argument("--hexagon-devices",
                        help="explicit GGML_HEXAGON_DEVICES override for controlled selector isolation")
    parser.add_argument("--vit-device-id",
                        help="explicit GenieX vision-encoder selector, for example CPU or HTP0")
    parser.add_argument("--policy", choices=("fifo", "priority", "edf", "predicted_latency"), default="priority",
                        help="predicted_latency learns from real per-backend executions; cold start uses priority order")
    parser.add_argument("--execution-mode", choices=("direct", "scheduler"), default="direct",
                        help="direct backend admission or the legacy native scheduler")
    parser.add_argument("--local-port", type=int, default=8765)
    parser.add_argument("--start-timeout", type=int, default=240)
    parser.add_argument("--collect-interval-s", type=int, default=20)
    parser.add_argument("--video-workers", type=int, default=1,
                        help="bounded number of simultaneous video-ingest workers on IQ-9075")
    parser.add_argument("--pending-videos", type=int, default=2,
                        help="bounded video jobs queued behind active ingest workers")
    parser.add_argument("--video-preprocess", choices=("auto", "software", "qim"),
                        default="auto",
                        help="auto prefers QIM qtivtransform when available; software selects the explicit fallback")
    parser.add_argument("--decoder-factory",
                        help="optional GStreamer decoder factory confirmed on the current target")
    parser.add_argument("--decoder-io-mode", choices=("auto", "dmabuf"), default="auto",
                        help="auto requests V4L2 DMABUF when the selected decoder exposes both I/O properties")
    parser.add_argument("--restore-vlm-gguf", action="store_true",
                        help="resume target-side downloads of the previously verified VLM GGUF and projector")
    parser.add_argument("--upload-local-vlm-cache", action="store_true",
                        help="use the hash-verified D-drive VLM cache instead of downloading the two VLM files on-device")
    parser.add_argument("--restore-llm-gguf", action="store_true",
                        help="resume target-side download of Qualcomm's Qwen3-4B-Instruct-2507 GGUF fallback")
    parser.add_argument("--restore-minilm", action="store_true",
                        help="upload the SHA-verified 24 MB D-drive MiniLM cache and ARM64 llama-server")
    parser.add_argument("--restore-geniex-runtime", action="store_true",
                        help="reuse installed GenieX, otherwise fetch pinned v0.11.0 SDK on-device; use the 15 MB D-drive cache if target download fails")
    parser.add_argument("--restore-siglip2", action="store_true",
                        help="download and prepare optional IQ-9075 SigLIP2 W8A16 only after verifying the exact target QAIRT version")
    parser.add_argument("--restore-openai-clip", action="store_true",
                        help="download pinned paired CLIP ViT-B/32 ONNX encoders on-device; optional failures do not block the core app")
    parser.add_argument("--siglip2-qairt-version", default="",
                        help="target QAIRT version confirmed from this QDC session; must match 2.45.0.260326154327")
    parser.add_argument("--visual-min-free-bytes", type=int, default=2147483648,
                        help="minimum target MemAvailable required before initializing a visual encoder")
    parser.add_argument("--assets-only", action="store_true",
                        help="download and verify selected assets, then stop without starting model services")
    parser.add_argument("--device-selftest", action="store_true",
                        help="run the synthetic on-device service smoke after startup")
    parser.add_argument("--asset", action="append", default=[],
                        help="download a named artifact from configs/iq9075-board-assets.json")
    parser.add_argument("--asset-timeout", type=int, default=1800)
    parser.add_argument("--asset-space-reserve-bytes", type=int, default=1073741824)
    parser.add_argument("--device-https-proxy-port", type=int, default=0,
                        help="optional QDC-side loopback HTTPS CONNECT relay for device-side downloads")
    parser.add_argument("--llm-min-free-bytes", type=int, default=0)
    parser.add_argument("--llm-backend", choices=("auto", "geniex-htp", "cpu"), default="auto")
    parser.add_argument("--llm-device-id", default="HTP4")
    parser.add_argument("--llm-geniex-context", type=int, default=512)
    parser.add_argument("--llm-geniex-threads", type=int, default=8)
    parser.add_argument("--llm-geniex-max-tokens", type=int, default=96)
    parser.add_argument("--llm-resource-domain", default="htp-selector-topology-unverified")
    parser.add_argument("--embedding-min-free-bytes", type=int, default=0)
    parser.add_argument("--release-root", default="",
                        help="reuse a safe release path during session recovery; collect/stop also use it")
    parser.add_argument("--output", type=Path, default=Path("runs/device-recovery"))
    parser.add_argument("--local-file", type=Path, action="append")
    parser.add_argument("--remote-path", action="append")
    parser.add_argument("command", choices=("session", "collect", "stop", "push"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "collect":
        if not args.release_root:
            raise SystemExit("collect requires --release-root")
        return collect_release(args)
    if args.command == "stop":
        if not args.release_root:
            raise SystemExit("stop requires --release-root from the local deployment.json")
        return stop_service(args)
    if args.command == "push":
        return push_artifacts(args)
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
