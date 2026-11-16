#!/usr/bin/env python3
"""Export scheduler observations from manifest-verified paired GenieX runs.

The TSV is a private, D-drive artifact. It contains derived service durations,
not raw engine logs or a public performance claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from tools import analyze_dual_runs, run_manifest

ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "schema_version", "source", "run_id", "resource_domain_id",
    "target_model_id", "target_backend_id", "target_kind",
    "target_input_units", "target_output_units", "target_model_resident",
    "isolated_latency_us", "concurrent_latency_us", "co_running",
)
TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
BACKENDS = ("geniex-device-a", "geniex-device-b")
KINDS = {"llm_decode": ("decode_tps", "generated_tokens"),
         "llm_prefill": ("prefill_tps", "prompt_tokens")}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise ValueError(f"{name} must use only ASCII letters, digits, dot, underscore, or dash")
    return value


def _duration_us(units: int, tokens_per_second: object, label: str) -> int:
    if isinstance(tokens_per_second, bool) or not isinstance(tokens_per_second, (int, float)):
        raise ValueError(f"{label} throughput is missing")
    rate = float(tokens_per_second)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{label} throughput must be finite and positive")
    duration = int(round(units * 1_000_000.0 / rate))
    if duration <= 0 or duration > (1 << 64) - 1:
        raise ValueError(f"{label} duration is outside the supported range")
    return duration


def build_profile(run_dirs: list[Path], resource_domain_id: str,
                  workload_kind: str,
                  observation_source: str) -> dict[str, Any]:
    domain = _token(resource_domain_id, "resource_domain_id")
    if workload_kind not in KINDS:
        raise ValueError("workload kind must be llm_decode or llm_prefill")
    if observation_source not in {"hardware", "mock_fixture"}:
        raise ValueError("observation source must be hardware or mock_fixture")
    resolved = []
    for raw in run_dirs:
        path = (raw if raw.is_absolute() else ROOT / raw).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("run paths must stay inside the D: project workspace")
        resolved.append(path)
    if not resolved:
        raise ValueError("at least one isolated/concurrent run pair is required")

    first_config = _json(resolved[0] / "config.json")
    devices = first_config.get("devices")
    if not isinstance(devices, list) or len(devices) != 2 or devices[0] == devices[1]:
        raise ValueError("source runs must record two distinct runtime devices")
    analysis = analyze_dual_runs.analyze(resolved, (devices[0], devices[1]))
    configuration = analysis["configuration"]
    model_id = configuration.get("model_sha256")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Fa-f0-9]{64}", model_id):
        raise ValueError("source runs must include a 64-digit model SHA256")
    model_id = model_id.lower()
    metric_name, size_key = KINDS[workload_kind]
    prompt_tokens = configuration.get("prompt_tokens")
    generated_tokens = configuration.get("generated_tokens")
    units = configuration.get(size_key)
    if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
        raise ValueError(f"source runs must record positive {size_key}")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError("source runs must record positive prompt_tokens")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int) or generated_tokens <= 0:
        raise ValueError("source runs must record positive generated_tokens")

    n_blocks = analysis["n_paired_blocks"]
    source_run_ids = analysis["source_run_ids"]
    isolated_ids = source_run_ids[:n_blocks]
    concurrent_ids = source_run_ids[n_blocks:]
    if len(isolated_ids) != n_blocks or len(concurrent_ids) != n_blocks:
        raise ValueError("paired analysis did not return complete source run IDs")

    observations = []
    for side, device in enumerate(devices):
        target_backend = BACKENDS[side]
        co_runner_backend = BACKENDS[1 - side]
        paired = analysis["paired_device_metrics"][device][metric_name]
        isolated_rates = paired["isolated_per_block"]
        concurrent_rates = paired["concurrent_per_block"]
        if len(isolated_rates) != n_blocks or len(concurrent_rates) != n_blocks:
            raise ValueError("paired analysis returned an incomplete throughput series")
        output_units = generated_tokens if workload_kind == "llm_decode" else 0
        co_runner = {
            "model_id": model_id,
            "backend_id": co_runner_backend,
            "kind": workload_kind,
            "input_units": prompt_tokens,
            "output_units": output_units,
            "model_resident": True,
        }
        for block in range(n_blocks):
            run_id = _token(concurrent_ids[block], "source run_id")
            observations.append({
                "schema_version": 1,
                "source": observation_source,
                "run_id": run_id,
                "resource_domain_id": domain,
                "target_model_id": model_id,
                "target_backend_id": target_backend,
                "target_kind": workload_kind,
                "target_input_units": prompt_tokens,
                "target_output_units": output_units,
                "target_model_resident": False,
                "isolated_latency_us": _duration_us(
                    units, isolated_rates[block], f"isolated block {block + 1}"),
                "concurrent_latency_us": _duration_us(
                    units, concurrent_rates[block], f"concurrent block {block + 1}"),
                "co_running": [co_runner],
            })
    return {
        "schema_version": 1,
        "profile_kind": "paired_geniex_interference_observations",
        "evidence_class": ("private_reported_hardware_run" if observation_source == "hardware"
                           else "simulation_fixture"),
        "simulation_only": observation_source == "mock_fixture",
        "hardware_origin_confirmation": (
            "operator-selected; run manifests verify file integrity, not device identity"
            if observation_source == "hardware" else "not applicable"),
        "resource_domain_id": domain,
        "target_model_id": model_id,
        "workload_kind": workload_kind,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "observation_count": len(observations),
        "paired_block_count": n_blocks,
        "source_analysis_sha256": analysis["analyzer_source_sha256"],
        "source_run_ids": source_run_ids,
        "service_time_derivation": (
            f"round({size_key} / median {metric_name} * 1,000,000) microseconds"),
        "residency_context": (
            "target is not resident at admission; each co-runner is active and "
            "resident in ResourceManager accounting"),
        "observations": observations,
        "limitations": [
            "Durations are derived from engine throughput, not wrapper wall time.",
            "The source runs use one model and a fixed paired workload configuration.",
            "The caller supplies the shared resource domain; runtime IDs do not prove topology.",
            "Each pairwise penalty is directional and additive only as a v1 approximation.",
            "Source runs are descriptive repetitions from one device session.",
        ],
    }


def _output_directory(path: Path, run_dirs: list[Path]) -> Path:
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not any(resolved.is_relative_to((ROOT / base).resolve())
               for base in ("runs", ".local")):
        raise ValueError("output directory must be under runs/ or .local/")
    if any(resolved == run or resolved.is_relative_to(run) for run in run_dirs):
        raise ValueError("output directory must not be inside a source run")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError("output directory must be new or empty")
    return resolved


def write_profile(profile: dict[str, Any], output_dir: Path,
                  source_runs: list[Path]) -> dict[str, Any]:
    output_dir = _output_directory(output_dir, source_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "interference-observations.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for item in profile["observations"]:
            actors = ",".join(
                f"{actor['model_id']}:{actor['backend_id']}:{actor['kind']}:"
                f"{actor['input_units']}:{actor['output_units']}:"
                f"{int(actor['model_resident'])}"
                for actor in item["co_running"])
            writer.writerow((
                item["schema_version"], item["source"], item["run_id"],
                item["resource_domain_id"], item["target_model_id"],
                item["target_backend_id"], item["target_kind"],
                item["target_input_units"], item["target_output_units"],
                int(item["target_model_resident"]), item["isolated_latency_us"],
                item["concurrent_latency_us"], actors,
            ))
    report_path = output_dir / "provenance.json"
    public_profile = {key: value for key, value in profile.items() if key != "observations"}
    public_profile["observation_file"] = tsv_path.name
    public_profile["observation_file_sha256"] = hashlib.sha256(tsv_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(public_profile, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    manifest = run_manifest.create(output_dir)
    return {"output": output_dir.relative_to(ROOT).as_posix(),
            "observations": profile["observation_count"],
            "paired_blocks": profile["paired_block_count"],
            "source_runs": len(profile["source_run_ids"]),
            "verified_manifest_files": len(manifest["files"]),
            "status": "exported_private_hardware_observations"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="verified private dual-run directory; provide all paired runs")
    parser.add_argument("--resource-domain-id", required=True,
                        help="explicit evidence-backed shared resource domain")
    parser.add_argument("--workload-kind", choices=tuple(KINDS), default="llm_decode")
    parser.add_argument("--observation-source", choices=("hardware", "mock_fixture"),
                        required=True,
                        help="select hardware only after confirming the source runs came from QDC")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory under runs/ or .local/")
    args = parser.parse_args()
    try:
        source_runs = [(run if run.is_absolute() else ROOT / run).resolve()
                       for run in args.run]
        profile = build_profile(source_runs, args.resource_domain_id,
                                args.workload_kind, args.observation_source)
        result = write_profile(profile, args.output_dir, source_runs)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
