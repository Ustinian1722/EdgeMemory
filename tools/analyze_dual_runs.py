#!/usr/bin/env python3
"""Analyze private paired GenieX runs on two explicitly selected runtime IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from tools import paired_statistics, performance_metrics, run_manifest

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
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty paired group")
    return float(statistics.median(values))


def _change_percent(value: float, baseline: float) -> float | None:
    return None if baseline == 0 else 100.0 * (value / baseline - 1.0)


def analyze(run_dirs: list[Path], device_ids: tuple[str, str],
            evidence_run_dirs: list[Path] | None = None) -> dict[str, Any]:
    if (len(device_ids) != 2 or any(not isinstance(item, str) or not item or "," in item
                                    for item in device_ids) or device_ids[0] == device_ids[1]):
        raise ValueError("two distinct explicit runtime device IDs are required")
    devices = device_ids
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    reference_config: dict[str, Any] | None = None

    for raw_path in run_dirs:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("run paths must stay inside the D: project workspace")
        run_manifest.verify(path)
        config = _json(path / "config.json")
        result = _json(path / "result.json")
        condition = config.get("condition")
        block = config.get("block")
        if condition not in ("isolated", "concurrent") or not isinstance(block, int):
            raise ValueError(f"invalid condition/block in {path.name}")
        if result.get("run_id") != path.name or result.get("condition") != condition:
            raise ValueError(f"run identity mismatch in {path.name}")
        if config.get("devices") != list(devices):
            raise ValueError(f"expected explicit device order {devices} in {path.name}")
        if result.get("simulated") is not False:
            raise ValueError(f"run is not marked as real execution in {path.name}")
        if config.get("exit_code") != 0 or result.get("request_a_status") != "process-completed" \
                or result.get("request_b_status") != "process-completed":
            raise ValueError(f"failed backend process in {path.name}")
        if reference_config is None:
            reference_config = config
        else:
            for key in CONFIG_KEYS:
                if config.get(key) != reference_config.get(key):
                    raise ValueError(f"incomparable {key} in {path.name}")

        key = (condition, block)
        if key in cells:
            raise ValueError(f"duplicate condition/block: {key}")
        per_device: dict[str, Any] = {}
        for tag, device in zip(("a", "b"), devices):
            engine = _json(path / f"llm-{tag}.json")
            if engine.get("plugin") != "llama_cpp" \
                    or not isinstance(engine.get("device"), str) or not engine.get("device") \
                    or engine.get("device_id") != device:
                raise ValueError(f"engine output does not confirm requested {device} in {path.name}")
            if engine.get("model_size_bytes") is None or engine.get("geniex_version") != config.get("geniex_version") \
                    or engine.get("llama_cpp_version") != config.get("llama_cpp_version"):
                raise ValueError(f"engine version/model metadata mismatch in {path.name}")
            agg = engine.get("agg")
            if not isinstance(agg, dict):
                raise ValueError(f"missing aggregate metrics for {device} in {path.name}")
            metrics: dict[str, float] = {}
            for name in METRICS:
                record = agg.get(name)
                if not isinstance(record, dict) or not isinstance(record.get("median"), (int, float)):
                    raise ValueError(f"missing {name} median for {device} in {path.name}")
                metrics[name] = float(record["median"])
            per_device[device] = metrics

        process_rss = performance_metrics.read_peak_process_rss(
            path / "metrics.jsonl", path.name,
            ("llm-request-a", "llm-request-b"))

        cells[key] = {
            "run_id": path.name,
            "devices": per_device,
            "peak_process_rss_bytes": {
                devices[0]: process_rss["llm-request-a"],
                devices[1]: process_rss["llm-request-b"],
            },
            "wrapper_elapsed_ms": {
                devices[0]: float(result["request_a_wrapper_elapsed_ms"]),
                devices[1]: float(result["request_b_wrapper_elapsed_ms"]),
            },
            "wrapper_interval_overlap_ms": float(result["wrapper_interval_overlap_ms"]),
        }

    if reference_config is None:
        raise ValueError("at least one run is required")
    isolated_blocks = {block for condition, block in cells if condition == "isolated"}
    concurrent_blocks = {block for condition, block in cells if condition == "concurrent"}
    if not isolated_blocks or isolated_blocks != concurrent_blocks:
        raise ValueError("isolated and concurrent runs must have the same non-empty paired blocks")

    summaries: dict[str, Any] = {}
    paired_deltas: dict[str, Any] = {}
    paired_delta_intervals: dict[str, Any] = {}
    process_rss_summaries: dict[str, Any] = {}
    for device in devices:
        summaries[device] = {}
        paired_deltas[device] = {}
        paired_delta_intervals[device] = {}
        process_rss_summaries[device] = {}
        for condition in ("isolated", "concurrent"):
            values = [cells[(condition, block)]["peak_process_rss_bytes"][device]
                      for block in sorted(isolated_blocks)]
            process_rss_summaries[device][condition] = performance_metrics.distribution_summary(
                value for value in values if value is not None)
        for metric in METRICS:
            isolated_values = [cells[("isolated", block)]["devices"][device][metric]
                               for block in sorted(isolated_blocks)]
            concurrent_values = [cells[("concurrent", block)]["devices"][device][metric]
                                 for block in sorted(concurrent_blocks)]
            isolated_median = _median(isolated_values)
            concurrent_median = _median(concurrent_values)
            delta = _change_percent(concurrent_median, isolated_median)
            summaries[device][metric] = {
                "isolated_per_block": isolated_values,
                "concurrent_per_block": concurrent_values,
                "isolated_median": isolated_median,
                "concurrent_median": concurrent_median,
                "concurrent_change_percent": delta,
            }
            paired_deltas[device][metric] = [
                _change_percent(concurrent, isolated)
                for isolated, concurrent in zip(isolated_values, concurrent_values)
            ]
            deltas = paired_deltas[device][metric]
            if any(delta is None for delta in deltas):
                paired_delta_intervals[device][metric] = {
                    "status": "undefined_zero_baseline",
                    "paired_blocks": len(deltas),
                    "lower": None,
                    "upper": None,
                }
            else:
                paired_delta_intervals[device][metric] = (
                    paired_statistics.bootstrap_median_interval(
                        (float(delta) for delta in deltas),
                        seed=paired_statistics.stable_seed(
                            *sorted(cells[("isolated", block)]["run_id"]
                                    for block in isolated_blocks),
                            device, metric, "paired-change-percent"),
                    )
                )

    serial_pair_wall = [
        sum(cells[("isolated", block)]["wrapper_elapsed_ms"].values())
        for block in sorted(isolated_blocks)
    ]
    concurrent_pair_wall = [
        max(cells[("concurrent", block)]["wrapper_elapsed_ms"].values())
        for block in sorted(concurrent_blocks)
    ]
    wall_speedup = _median(serial_pair_wall) / _median(concurrent_pair_wall)
    paired_wall_speedups = [
        serial / concurrent
        for serial, concurrent in zip(serial_pair_wall, concurrent_pair_wall)
        if concurrent > 0
    ]
    wall_speedup_interval = paired_statistics.bootstrap_median_interval(
        paired_wall_speedups,
        seed=paired_statistics.stable_seed(
            *sorted(cells[("isolated", block)]["run_id"]
                    for block in isolated_blocks),
            "paired-wall-speedup"),
    )
    concurrent_overlaps = [cells[("concurrent", block)]["wrapper_interval_overlap_ms"]
                           for block in sorted(concurrent_blocks)]

    evidence_ids = []
    for raw_path in evidence_run_dirs or []:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("evidence run paths must stay inside the D: project workspace")
        evidence_ids.append(run_manifest.verify(path)["run_id"])

    return {
        "analysis_schema_version": 3,
        "analyzer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_class": "private_real_device_benchmark",
        "simulated": False,
        "n_paired_blocks": len(isolated_blocks),
        "devices": list(devices),
        "configuration": {key: reference_config.get(key) for key in CONFIG_KEYS},
        "paired_device_metrics": summaries,
        "paired_per_block_change_percent": paired_deltas,
        "paired_median_change_percent_ci95": paired_delta_intervals,
        "peak_process_rss_bytes": {
            "measurement_scope": "direct child process only; null when unavailable",
            "by_device_and_condition": process_rss_summaries,
        },
        "pair_wrapper_wall_ms": {
            "isolated_serial_per_block": serial_pair_wall,
            "concurrent_per_block": concurrent_pair_wall,
            "isolated_median": _median(serial_pair_wall),
            "concurrent_median": _median(concurrent_pair_wall),
            "concurrent_pair_completion_speedup": wall_speedup,
            "paired_block_speedup_median_ci95": wall_speedup_interval,
            "concurrent_wrapper_overlap_ms": concurrent_overlaps,
        },
        "uncertainty_scope": (
            "Percentile bootstrap over matched paired blocks. Intervals are suppressed "
            "below 10 blocks and do not represent between-session or between-device variation."),
        "source_run_ids": [cells[(condition, block)]["run_id"]
                           for condition in ("isolated", "concurrent")
                           for block in sorted(isolated_blocks)],
        "supporting_trace_run_ids": evidence_ids,
        "limitations": [
            "Paired blocks are descriptive repetitions from one device session, not independent devices.",
            "GenieX JSON repetition medians report engine metrics; wrapper wall time includes process startup and shutdown.",
            "Overlapping process intervals and selected logical device IDs do not time-align individual HTP kernels.",
            "Results do not measure thermal steady state, power, model quality, or deadline miss rate.",
            "Peak process RSS is the direct child maximum; it excludes descendants and HTP memory.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="private run directory; provide every paired run")
    parser.add_argument("--evidence-run", type=Path, action="append", default=[],
                        help="optional supporting runtime trace run directory")
    parser.add_argument("--device-a", required=True,
                        help="runtime device ID used for request A in every run")
    parser.add_argument("--device-b", required=True,
                        help="runtime device ID used for request B in every run")
    parser.add_argument("--output", type=Path, required=True,
                        help="analysis JSON path under D:\\A-IQ-9075")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must stay inside the D: project workspace")
    try:
        result = analyze(args.run, (args.device_a, args.device_b), args.evidence_run)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "paired_blocks": result["n_paired_blocks"],
                      "verified_source_runs": len(result["source_run_ids"]),
                      "status": "analyzed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
