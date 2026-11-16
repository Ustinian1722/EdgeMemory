"""Deterministic uncertainty summaries for matched benchmark blocks."""
from __future__ import annotations

import hashlib
import math
import random
import statistics
from typing import Iterable


DEFAULT_RESAMPLES = 10_000
DEFAULT_MINIMUM_BLOCKS = 10


def stable_seed(*parts: str) -> int:
    """Derive a stable, non-sensitive PRNG seed from analysis identifiers."""
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_median_interval(
    values: Iterable[float],
    *,
    seed: int,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    minimum_blocks: int = DEFAULT_MINIMUM_BLOCKS,
) -> dict[str, float | int | str | None]:
    """Return a percentile bootstrap CI for the median of paired block values.

    Small samples are reported without an interval. These intervals describe
    repeated blocks from the supplied run set; they do not estimate variation
    across devices or sessions.
    """
    sample: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("bootstrap values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("bootstrap values must be finite")
        sample.append(number)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if minimum_blocks < 1:
        raise ValueError("minimum_blocks must be positive")

    base: dict[str, float | int | str | None] = {
        "method": "paired-block percentile bootstrap of the median; type-7 quantiles",
        "confidence_level": confidence,
        "resamples": resamples,
        "seed": seed,
        "paired_blocks": len(sample),
        "minimum_blocks_for_interval": minimum_blocks,
        "estimate": statistics.median(sample) if sample else None,
        "lower": None,
        "upper": None,
    }
    if len(sample) < minimum_blocks:
        base["status"] = "insufficient_blocks"
        return base

    rng = random.Random(seed)
    estimates = sorted(
        statistics.median(rng.choices(sample, k=len(sample)))
        for _ in range(resamples)
    )

    def quantile(probability: float) -> float:
        position = (len(estimates) - 1) * probability
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        fraction = position - lower_index
        return estimates[lower_index] + fraction * (
            estimates[upper_index] - estimates[lower_index]
        )

    alpha = (1.0 - confidence) / 2.0
    base["lower"] = quantile(alpha)
    base["upper"] = quantile(1.0 - alpha)
    base["status"] = "estimated"
    return base
