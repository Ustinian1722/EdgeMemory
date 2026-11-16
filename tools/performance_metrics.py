"""Validated summaries for optional per-process telemetry in private runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def distribution_summary(values: Iterable[int | float]) -> dict[str, Any]:
    ordered: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("measurement values must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("measurement values must be finite and non-negative")
        ordered.append(number)
    ordered.sort()
    if not ordered:
        return {
            "count": 0, "min": None, "p50": None, "p95": None,
            "p99": None, "max": None,
            "percentile_method": "linear interpolation, type 7",
        }

    def percentile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "percentile_method": "linear interpolation, type 7",
    }


def read_peak_process_rss(metrics_path: Path, run_id: str,
                          task_ids: Iterable[str]) -> dict[str, int | None]:
    """Read direct-child RSS from new metrics; tolerate legacy rows without it."""
    expected = set(task_ids)
    if not expected:
        raise ValueError("at least one expected task ID is required")
    values: dict[str, int | None] = {task_id: None for task_id in expected}
    if not metrics_path.is_file():
        return values

    seen: set[str] = set()
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSON on line {line_number}") from error
        if not isinstance(row, dict) or row.get("run_id") != run_id:
            raise ValueError(f"metrics run identity mismatch on line {line_number}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected or task_id in seen:
            raise ValueError(f"unexpected or duplicate task in metrics on line {line_number}")
        seen.add(task_id)
        value = row.get("peak_process_rss_bytes")
        scope = row.get("peak_process_rss_scope")
        if value is None:
            if scope not in (None, "direct_child"):
                raise ValueError(f"invalid RSS scope on line {line_number}")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid peak process RSS on line {line_number}")
        if scope != "direct_child":
            raise ValueError(f"unrecognized RSS scope on line {line_number}")
        values[task_id] = value
    if seen != expected:
        raise ValueError("metrics file is missing expected task rows")
    return values
