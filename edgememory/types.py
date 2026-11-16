from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SamplingPolicy:
    min_interval_s: float = 1.5
    max_interval_s: float = 12.0
    scene_change_threshold: float = 0.16
    max_samples: int = 500
    max_frame_side: int = 768
    jpeg_quality: int = 88

    def validate(self) -> None:
        if self.min_interval_s <= 0 or self.max_interval_s < self.min_interval_s:
            raise ValueError("sampling intervals must satisfy 0 < min <= max")
        if not 0.0 <= self.scene_change_threshold <= 1.0:
            raise ValueError("scene_change_threshold must be in [0, 1]")
        if self.max_samples < 1 or self.max_frame_side < 32:
            raise ValueError("max_samples and max_frame_side are too small")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


@dataclass(frozen=True)
class VideoMetadata:
    media_id: str
    source_path: Path
    source_sha256: str
    duration_ms: int
    source_frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameSample:
    media_id: str
    frame_index: int
    timestamp_ms: int
    frame_path: Path
    frame_sha256: str
    scene_delta: float
    sample_reason: str
    # Optional path to the same encoded evidence frame on the remote inference
    # node. It is ephemeral and must never be serialized as a local artifact path.
    remote_path: str | None = None


@dataclass(frozen=True)
class CaptionResult:
    text: str
    backend_id: str
    model_id: str
    simulation_only: bool = False
    inference_metrics: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    event_id: str
    media_id: str
    media_title: str
    start_ms: int
    end_ms: int
    timestamp_ms: int
    caption: str
    backend_id: str
    model_id: str
    frame_path: str
    frame_sha256: str
    similarity: float | None
    simulation_only: bool
    inference_metrics: dict[str, int | float]


@dataclass(frozen=True)
class IngestSummary:
    media_id: str
    source_sha256: str
    sampled_frames: int
    indexed_events: int
    failed_events: int
    simulation_only: bool
    artifact_dir: Path
    run_dir: Path
    frame_samples: Sequence[FrameSample]
