#!/usr/bin/env python3
"""Build a private, matched GenieX VLM service-time profile from verified runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import run_manifest


ROOT = Path(__file__).resolve().parents[1]
COMMON_KEYS = (
    "model_id", "model_bundle", "source_video_sha256", "caption_prompt_sha256",
    "sampling_policy", "caption_configuration", "runtime_version", "plugin_version",
    "frame_evidence",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid SHA256 field: {field}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"invalid SHA256 field: {field}") from error
    return value.lower()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _model_bundle(value: object) -> list[dict[str, int | str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("calibration runs must record model bundle hashes")
    bundle: list[dict[str, int | str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("model bundle entries must be objects")
        digest = _sha256(item.get("sha256"), "model_bundle.sha256")
        size = _positive_int(item.get("bytes"), "model_bundle.bytes")
        if digest in seen:
            raise ValueError("model bundle contains a duplicate hash")
        seen.add(digest)
        bundle.append({"bytes": size, "sha256": digest})
    return bundle


def analyze(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two matched backend runs are required")

    reference: dict[str, Any] | None = None
    reference_run_id = ""
    evidence_ids: list[str] = []
    backends: list[dict[str, Any]] = []
    device_ids: set[str] = set()
    backend_ids: set[str] = set()

    for raw_path in run_dirs:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("run paths must stay inside the D: project workspace")
        manifest = run_manifest.verify(path)
        calibration = _json(path / "calibration-run.json")
        ingest = _json(path / "ingest.json")
        run_id = manifest["run_id"]
        if calibration.get("run_id") != run_id:
            raise ValueError(f"calibration identity mismatch in {path.name}")
        if calibration.get("warmup_completed") is not True:
            raise ValueError(f"missing completed warm-up in {path.name}")
        if calibration.get("simulation_only") is not False or ingest.get("simulation_only") is not False:
            raise ValueError(f"simulation data is not valid for a device service profile: {path.name}")
        if ingest.get("failed_events") != 0 or ingest.get("indexed_events") != ingest.get("sampled_frames"):
            raise ValueError(f"incomplete frame results in {path.name}")

        model_id = calibration.get("model_id")
        device_id = calibration.get("runtime_device_id")
        backend_id = calibration.get("backend_id")
        runtime = calibration.get("worker_metadata")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"missing model id in {path.name}")
        if not isinstance(device_id, str) or not device_id:
            raise ValueError(f"missing runtime device id in {path.name}")
        if not isinstance(backend_id, str) or not backend_id:
            raise ValueError(f"missing backend id in {path.name}")
        if not isinstance(runtime, dict) or runtime.get("device_id") != device_id \
                or runtime.get("model_id") != model_id:
            raise ValueError(f"worker metadata mismatch in {path.name}")
        if not isinstance(calibration.get("run_configuration"), dict) \
                or calibration["run_configuration"] != ingest.get("caption_configuration"):
            raise ValueError(f"caption configuration mismatch in {path.name}")

        model_bundle = _model_bundle(calibration.get("model_bundle"))
        source_sha = _sha256(calibration.get("source_video_sha256"), "source_video_sha256")
        prompt_sha = _sha256(calibration.get("caption_prompt_sha256"), "caption_prompt_sha256")
        warmup_frame_sha = _sha256(calibration.get("warmup_frame_sha256"), "warmup_frame_sha256")
        frame_evidence = ingest.get("frame_evidence")
        frame_results = ingest.get("frame_results")
        if not isinstance(frame_evidence, list) or not frame_evidence \
                or not isinstance(frame_results, list) or len(frame_results) != len(frame_evidence):
            raise ValueError(f"invalid frame evidence/results in {path.name}")
        hashes = []
        for item in frame_evidence:
            if not isinstance(item, dict):
                raise ValueError(f"invalid frame evidence item in {path.name}")
            hashes.append(_sha256(item.get("frame_sha256"), "frame_sha256"))
        if warmup_frame_sha != hashes[0]:
            raise ValueError(f"warm-up frame is not the first calibrated evidence frame in {path.name}")

        common = {
            "model_id": model_id,
            "model_bundle": model_bundle,
            "source_video_sha256": source_sha,
            "caption_prompt_sha256": prompt_sha,
            "sampling_policy": ingest.get("sampling_policy"),
            "caption_configuration": calibration["run_configuration"],
            "runtime_version": runtime.get("runtime_version"),
            "plugin_version": runtime.get("plugin_version"),
            "frame_evidence": frame_evidence,
        }
        if any(common[key] is None for key in COMMON_KEYS):
            raise ValueError(f"incomplete common run metadata in {path.name}")
        if reference is None:
            reference = common
            reference_run_id = run_id
        elif any(common[key] != reference[key] for key in COMMON_KEYS):
            raise ValueError(f"incomparable model, input, policy, runtime, or frames in {path.name}")

        if device_id in device_ids or backend_id in backend_ids:
            raise ValueError("each calibration run must use a distinct runtime device and backend ID")
        device_ids.add(device_id)
        backend_ids.add(backend_id)

        samples = []
        for result in frame_results:
            if not isinstance(result, dict) or result.get("status") != "completed" \
                    or result.get("simulation_only") is not False:
                raise ValueError(f"non-real or failed caption sample in {path.name}")
            metrics = result.get("inference_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"missing inference metrics in {path.name}")
            worker_us = _positive_int(metrics.get("worker_elapsed_us"), "worker_elapsed_us")
            ttft_us = _positive_int(metrics.get("ttft_us"), "ttft_us")
            decode_tps = metrics.get("decode_tokens_per_second")
            if isinstance(decode_tps, bool) or not isinstance(decode_tps, (int, float)) \
                    or not math.isfinite(float(decode_tps)) or decode_tps < 0:
                raise ValueError(f"invalid decode throughput in {path.name}")
            samples.append({
                "worker_elapsed_us": worker_us,
                "ttft_us": ttft_us,
                "decode_tokens_per_second": float(decode_tps),
                "output_tokens": _positive_int(metrics.get("output_tokens"), "output_tokens"),
            })

        warmup_metrics = calibration.get("warmup_metrics")
        if not isinstance(warmup_metrics, dict) \
                or _positive_int(warmup_metrics.get("worker_elapsed_us"), "warmup.worker_elapsed_us") <= 0:
            raise ValueError(f"missing warm-up timings in {path.name}")
        backends.append({
            "runtime_device_id": device_id,
            "backend_id": backend_id,
            "source_run_id": run_id,
            "sample_count": len(samples),
            "service_us": int(round(statistics.median(x["worker_elapsed_us"] for x in samples))),
            "ttft_us": int(round(statistics.median(x["ttft_us"] for x in samples))),
            "decode_tokens_per_second": statistics.median(
                x["decode_tokens_per_second"] for x in samples),
            "per_frame": samples,
            "warmup_worker_elapsed_us": warmup_metrics["worker_elapsed_us"],
        })
        evidence_ids.append(run_id)

    assert reference is not None
    signature = {
        "common": reference,
        "backends": sorted((x["runtime_device_id"], x["service_us"]) for x in backends),
    }
    calibration_id = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_class": "private_warmed_real_device_calibration",
        "simulation_only": False,
        "source_run_ids": evidence_ids,
        "reference_run_id": reference_run_id,
        "workload": {key: reference[key] for key in COMMON_KEYS if key != "frame_evidence"},
        "warmup_frame_sha256": reference["frame_evidence"][0]["frame_sha256"],
        "frame_evidence": reference["frame_evidence"],
        "service_metric": "median worker_elapsed_us from warmed persistent GenieX C API requests",
        "backends": backends,
        "limitations": [
            "Profile is valid only for the recorded model bundle, prompt, input, sampling policy, and runtime configuration.",
            "Service estimates are descriptive medians from the listed warmed samples; no confidence interval is computed.",
            "Runtime IDs do not prove physical accelerator topology or independent HTP capacity.",
            "Resident-memory estimates are separate from measured RSS or accelerator memory.",
            "The profile does not establish power, thermal steady state, quality, or deadline guarantees.",
        ],
    }


def _output_path(path: Path, run_dirs: list[Path]) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    allowed = any(resolved.is_relative_to((ROOT / directory).resolve())
                  for directory in ("runs", ".local"))
    if not allowed:
        raise ValueError("calibration profiles must be written under ignored runs/ or .local/")
    if any(resolved.is_relative_to(run.resolve()) for run in run_dirs):
        raise ValueError("profile output must not be inside an input run directory")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="verified warmed EdgeMemory run; provide one per backend")
    parser.add_argument("--output", type=Path, required=True,
                        help="profile path under runs/ or .local/")
    args = parser.parse_args()
    try:
        run_dirs = [(path if path.is_absolute() else ROOT / path).resolve()
                    for path in args.run]
        output = _output_path(args.output, run_dirs)
        profile = analyze(run_dirs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "backends": len(profile["backends"]),
                      "verified_source_runs": len(profile["source_run_ids"]),
                      "calibration_id": profile["calibration_id"],
                      "status": "analyzed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
