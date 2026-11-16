#!/usr/bin/env python3
"""Summarize private single-device versus configured split GenieX benchmark blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from tools import run_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = (
    "geniex_version", "llama_cpp_version", "model_sha256", "prompt_tokens",
    "generated_tokens", "repetitions", "warmup", "seed", "threads",
    "runner_sha256",
)
METRICS = ("ttft_ms", "prefill_tps", "decode_tps")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty condition")
    return float(statistics.median(values))


def _percent_change(value: float, baseline: float) -> float | None:
    return None if baseline == 0 else 100.0 * (value / baseline - 1.0)


def analyze(run_dirs: list[Path], device_a: str, device_b: str,
            split_device: str) -> dict[str, Any]:
    conditions = (device_a, device_b, split_device)
    if (any(not isinstance(item, str) or not item for item in conditions)
            or len(set(conditions)) != 3):
        raise ValueError("three distinct explicit runtime condition IDs are required")
    cells: dict[tuple[int, str], dict[str, Any]] = {}
    reference_config: dict[str, Any] | None = None
    for raw_path in run_dirs:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("run paths must stay inside the D: project workspace")
        run_manifest.verify(path)
        config = _json(path / "config.json")
        result = _json(path / "result.json")
        block = config.get("block")
        device_id = config.get("device_id")
        if not isinstance(block, int) or device_id not in conditions:
            raise ValueError(f"invalid block/device id in {path.name}")
        expected_mode = "split" if device_id == split_device else "single"
        if config.get("mode") != expected_mode:
            raise ValueError(f"mode does not match device id in {path.name}")
        if config.get("run_id") != path.name or result.get("run_id") != path.name:
            raise ValueError(f"run identity mismatch in {path.name}")
        if result.get("simulated") is not False or config.get("exit_code") != 0 \
                or result.get("request_status") != "process-completed":
            raise ValueError(f"run did not complete as real engine execution: {path.name}")
        if reference_config is None:
            reference_config = config
        else:
            for key in CONFIG_KEYS:
                if config.get(key) != reference_config.get(key):
                    raise ValueError(f"incomparable {key} in {path.name}")
        key = (block, device_id)
        if key in cells:
            raise ValueError(f"duplicate block/device cell: {key}")

        engine = _json(path / "bench.json")
        if engine.get("plugin") != "llama_cpp" or engine.get("device") != "npu" \
                or engine.get("device_id") != device_id:
            raise ValueError(f"engine JSON does not confirm requested device id in {path.name}")
        if engine.get("geniex_version") != config.get("geniex_version") \
                or engine.get("llama_cpp_version") != config.get("llama_cpp_version"):
            raise ValueError(f"engine version mismatch in {path.name}")
        agg = engine.get("agg")
        if not isinstance(agg, dict):
            raise ValueError(f"missing aggregate metrics in {path.name}")
        metrics = {}
        for metric in METRICS:
            entry = agg.get(metric)
            if not isinstance(entry, dict) or not isinstance(entry.get("median"), (int, float)):
                raise ValueError(f"missing {metric} median in {path.name}")
            metrics[metric] = float(entry["median"])
        cells[key] = {
            "run_id": path.name,
            "metrics": metrics,
            "wrapper_elapsed_ms": float(result["wrapper_elapsed_ms"]),
        }

    if reference_config is None:
        raise ValueError("at least one run is required")
    blocks = sorted({block for block, _ in cells})
    for block in blocks:
        if {(b, device) for b, device in cells if b == block} != {
            (block, device) for device in conditions
        }:
            raise ValueError(f"block {block} must contain both single-device IDs and the split ID")
    if len(blocks) < 2:
        raise ValueError("at least two complete paired blocks are required")

    per_condition: dict[str, Any] = {}
    split_changes: dict[str, Any] = {}
    for device in conditions:
        per_condition[device] = {}
        for metric in METRICS:
            values = [cells[(block, device)]["metrics"][metric] for block in blocks]
            per_condition[device][metric] = {
                "per_block": values,
                "median": _median(values),
                "min": min(values),
                "max": max(values),
            }
    for metric in METRICS:
        dual_values = [cells[(block, split_device)]["metrics"][metric] for block in blocks]
        split_changes[metric] = {}
        for single in (device_a, device_b):
            single_values = [cells[(block, single)]["metrics"][metric] for block in blocks]
            split_changes[metric][f"versus_{single}"] = {
                "per_block_percent": [
                    _percent_change(dual, baseline)
                    for dual, baseline in zip(dual_values, single_values)
                ],
                "median_percent": _percent_change(_median(dual_values), _median(single_values)),
            }

    wrapper_times = {
        device: [cells[(block, device)]["wrapper_elapsed_ms"] for block in blocks]
        for device in conditions
    }
    isolated_service_estimates_us = {
        f"geniex-process-{device_a}": int(round(_median(wrapper_times[device_a]) * 1000.0)),
        f"geniex-process-{device_b}": int(round(_median(wrapper_times[device_b]) * 1000.0)),
    }
    return {
        "analysis_schema_version": 1,
        "analyzer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_class": "private_real_device_benchmark",
        "simulated": False,
        "n_paired_blocks": len(blocks),
        "configuration": {key: reference_config.get(key) for key in CONFIG_KEYS},
        "conditions": per_condition,
        "split_change_percent": split_changes,
        "wrapper_process_elapsed_ms": {
            device: {"per_block": values, "median": _median(values)}
            for device, values in wrapper_times.items()
        },
        "scheduler_calibration_profile": {
            "policy": "predicted_latency",
            "service_estimates_by_backend_us": isolated_service_estimates_us,
            "basis": "median isolated process wall time, including process startup and configured repetitions",
            "compatible_workload_configuration": {
                key: reference_config.get(key)
                for key in ("model_sha256", "prompt_tokens", "generated_tokens",
                            "repetitions", "warmup", "seed", "threads")
            },
        },
        "source_run_ids": [cells[(block, device)]["run_id"]
                           for block in blocks for device in conditions],
        "limitations": [
            "Blocks are repeated in one device session and are not independent devices.",
            "GenieX engine JSON reports repeated benchmark measurements; wrapper time includes process and model-load overhead.",
            "The comparison covers one small GGUF and does not predict behavior for larger models.",
            "No steady-state thermal, power, quality, or deadline-miss claim is supported.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--device-a", required=True,
                        help="first explicit single-device runtime ID")
    parser.add_argument("--device-b", required=True,
                        help="second explicit single-device runtime ID")
    parser.add_argument("--split-device", required=True,
                        help="explicit runtime ID used for the split condition")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must stay inside the D: project workspace")
    try:
        report = analyze(args.run, args.device_a, args.device_b, args.split_device)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "paired_blocks": report["n_paired_blocks"],
                      "verified_runs": len(report["source_run_ids"]),
                      "status": "analyzed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
