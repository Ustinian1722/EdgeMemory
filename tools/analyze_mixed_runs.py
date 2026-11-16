#!/usr/bin/env python3
"""Summarize private HexagonFlow mixed-load runs without publishing raw logs."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools import performance_metrics, run_manifest

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def embedding_batches(text: str) -> list[list[list[float]]]:
    """Extract llama.cpp JSON+ embedding batches from mixed diagnostic output."""
    decoder = json.JSONDecoder()
    batches: list[list[list[float]]] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict) and value.get("object") == "list":
            rows = value.get("data")
            if not isinstance(rows, list):
                raise ValueError("embedding JSON list has no data array")
            vectors = []
            for row in rows:
                vector = row.get("embedding") if isinstance(row, dict) else None
                if not isinstance(vector, list) or not vector:
                    raise ValueError("embedding JSON row has no non-empty vector")
                values = [float(item) for item in vector]
                if not all(math.isfinite(item) for item in values):
                    raise ValueError("embedding contains non-finite values")
                vectors.append(values)
            batches.append(vectors)
        cursor = start + offset
    return batches


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement group")
    return float(statistics.median(values))


def _percent_change(value: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (value / baseline - 1.0)


def analyze(run_dirs: list[Path]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    reference_vectors: list[list[float]] | None = None
    repeated_max_diffs: list[float] = []

    for raw_path in run_dirs:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("run paths must stay inside the D: project workspace")
        run_manifest.verify(path)
        config = _read_json(path / "config.json")
        result = _read_json(path / "result.json")
        bench = _read_json(path / "llm-benchmark.json")
        process_rss = performance_metrics.read_peak_process_rss(
            path / "metrics.jsonl", str(config["run_id"]),
            ("llm-request", "embedding-request"))
        batches = embedding_batches((path / "embedding-process.log").read_text(
            encoding="utf-8", errors="replace"))
        expected_batches = int(config["embedding_repeats"])
        expected_items = int(config["embedding_items_per_request"])
        if len(batches) != expected_batches:
            raise ValueError(f"{path.name}: expected {expected_batches} embedding batches, found {len(batches)}")
        if any(len(batch) != expected_items for batch in batches):
            raise ValueError(f"{path.name}: embedding item count differs from config")
        dimensions = {len(vector) for batch in batches for vector in batch}
        if len(dimensions) != 1:
            raise ValueError(f"{path.name}: mixed embedding dimensions")

        baseline = batches[0]
        max_repeat_diff = max(
            abs(left - right)
            for batch in batches[1:]
            for base_vector, vector in zip(baseline, batch)
            for left, right in zip(base_vector, vector)
        ) if len(batches) > 1 else 0.0
        repeated_max_diffs.append(max_repeat_diff)
        if reference_vectors is None:
            reference_vectors = baseline
        else:
            if len(reference_vectors) != len(baseline):
                raise ValueError("embedding item count changed across runs")
            repeated_max_diffs.append(max(
                abs(left - right)
                for base_vector, vector in zip(reference_vectors, baseline)
                for left, right in zip(base_vector, vector)
            ))

        norms = [math.sqrt(sum(value * value for value in vector))
                 for batch in batches for vector in batch]
        llm = result.get("llm_status") == "process-completed"
        embedding = result.get("embedding_status") == "process-completed"
        if not llm or not embedding:
            raise ValueError(f"{path.name}: one or more scheduled processes failed")
        runs = bench.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{path.name}: benchmark JSON has no repetitions")
        cell = {
            "run_id": config["run_id"],
            "block": config["block"],
            "condition": config["condition"],
            "embedding_threads": int(config["embedding_threads"]),
            "llm_device_id": config["device_id"],
            "llm_ttft_median_ms": float(bench["agg"]["ttft_ms"]["median"]),
            "llm_prefill_tps_median": float(bench["agg"]["prefill_tps"]["median"]),
            "llm_decode_tps_median": float(bench["agg"]["decode_tps"]["median"]),
            "llm_wrapper_elapsed_ms": float(result["llm_wrapper_elapsed_ms"]),
            "embedding_wrapper_elapsed_ms": float(result["embedding_wrapper_elapsed_ms"]),
            "wrapper_interval_overlap_ms": float(result["wrapper_interval_overlap_ms"]),
            "llm_peak_process_rss_bytes": process_rss["llm-request"],
            "embedding_peak_process_rss_bytes": process_rss["embedding-request"],
            "embedding_batches": len(batches),
            "embedding_vectors": sum(len(batch) for batch in batches),
            "embedding_dimensions": next(iter(dimensions)),
            "embedding_norm_range": [min(norms), max(norms)],
            "repeat_max_abs_difference": max_repeat_diff,
            "llm_repetitions": len(runs),
            "integrity_verified": True,
        }
        cells.append(cell)

    expected_config_keys = (
        "geniex_version", "llama_cpp_version", "llm_model_sha256", "embedding_model_sha256",
        "prompt_tokens", "generated_tokens", "benchmark_repetitions", "warmup", "seed",
        "llm_threads", "embedding_items_per_request", "embedding_repeats", "device_id",
    )
    configs = [_read_json(((path if path.is_absolute() else ROOT / path).resolve()) / "config.json")
               for path in run_dirs]
    for key in expected_config_keys:
        values = {json.dumps(config.get(key), sort_keys=True) for config in configs}
        if len(values) != 1:
            raise ValueError(f"incomparable experiment settings: {key}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        key = f"{cell['condition']}/embedding_threads={cell['embedding_threads']}"
        groups.setdefault(key, []).append(cell)
    grouped = {}
    for key, values in sorted(groups.items()):
        grouped[key] = {
            "runs": len(values),
            "llm_ttft_median_ms": _median([item["llm_ttft_median_ms"] for item in values]),
            "llm_prefill_tps_median": _median([item["llm_prefill_tps_median"] for item in values]),
            "llm_decode_tps_median": _median([item["llm_decode_tps_median"] for item in values]),
            "llm_wrapper_elapsed_median_ms": _median([item["llm_wrapper_elapsed_ms"] for item in values]),
            "embedding_wrapper_elapsed_median_ms": _median([item["embedding_wrapper_elapsed_ms"] for item in values]),
            "wrapper_overlap_median_ms": _median([item["wrapper_interval_overlap_ms"] for item in values]),
            "llm_peak_process_rss_bytes": performance_metrics.distribution_summary(
                item["llm_peak_process_rss_bytes"] for item in values
                if item["llm_peak_process_rss_bytes"] is not None),
            "embedding_peak_process_rss_bytes": performance_metrics.distribution_summary(
                item["embedding_peak_process_rss_bytes"] for item in values
                if item["embedding_peak_process_rss_bytes"] is not None),
        }

    comparison = {}
    isolated = grouped.get("isolated/embedding_threads=2")
    concurrent_2 = grouped.get("concurrent/embedding_threads=2")
    concurrent_1 = grouped.get("concurrent/embedding_threads=1")
    if isolated and concurrent_2:
        comparison["concurrent_t2_vs_isolated_t2"] = {
            "ttft_change_percent": _percent_change(concurrent_2["llm_ttft_median_ms"], isolated["llm_ttft_median_ms"]),
            "decode_tps_change_percent": _percent_change(concurrent_2["llm_decode_tps_median"], isolated["llm_decode_tps_median"]),
            "embedding_wall_change_percent": _percent_change(concurrent_2["embedding_wrapper_elapsed_median_ms"], isolated["embedding_wrapper_elapsed_median_ms"]),
        }
    if concurrent_1 and concurrent_2:
        comparison["concurrent_t1_vs_concurrent_t2"] = {
            "ttft_change_percent": _percent_change(concurrent_1["llm_ttft_median_ms"], concurrent_2["llm_ttft_median_ms"]),
            "decode_tps_change_percent": _percent_change(concurrent_1["llm_decode_tps_median"], concurrent_2["llm_decode_tps_median"]),
            "embedding_wall_change_percent": _percent_change(concurrent_1["embedding_wrapper_elapsed_median_ms"], concurrent_2["embedding_wrapper_elapsed_median_ms"]),
        }

    return {
        "schema_version": 2,
        "scope": "private-repository analysis; device output and raw logs remain in runs",
        "sample_design": "three run cells per group; five timed LLM repetitions per cell; descriptive medians only",
        "process_rss_scope": "direct child per process runner; null when unavailable; not accelerator memory",
        "all_embedding_vectors_repeat_exactly": max(repeated_max_diffs, default=0.0) == 0.0,
        "max_abs_difference_across_repeated_or_cross_condition_vectors": max(repeated_max_diffs, default=0.0),
        "groups": grouped,
        "comparisons": comparison,
        "cells": cells,
        "limitations": [
            "wrapper interval overlap proves concurrent process lifetimes, not exact HTP-kernel overlap",
            "three run cells per group do not establish a statistical significance claim",
            "legacy runs without the RSS field report no process-memory distribution",
            "direct-child RSS excludes descendants, persistent-worker lifetime and accelerator memory",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.runs)
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output = output.resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("analysis output must stay inside the D: project workspace")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "runs": len(report["cells"]),
                      "groups": len(report["groups"]),
                      "status": "analyzed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
