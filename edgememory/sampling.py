from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .types import FrameSample, SamplingPolicy, VideoMetadata


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_frame(frame, max_side: int, cv2):
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest <= max_side:
        return frame
    scale = max_side / largest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def extract_adaptive_frames(
    video_path: Path,
    output_dir: Path,
    policy: SamplingPolicy,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[VideoMetadata, list[FrameSample]]:
    """Extract deterministic, budget-bounded evidence frames using OpenCV."""
    policy.validate()
    video_path = video_path.expanduser().resolve(strict=True)
    if not video_path.is_file():
        raise ValueError("video input must be a regular file")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video ingestion; install it in the active environment") from error

    source_hash = sha256_file(video_path)
    media_id = source_hash[:24]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the video input")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps or fps != fps or fps < 0.01:
        fps = 30.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_ms = round(source_frames * 1000.0 / fps) if source_frames > 0 else 0

    frame_dir = output_dir / "frames" / media_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    previous_gray = None
    last_sample_ms: int | None = None
    frame_index = 0
    image_params = [int(cv2.IMWRITE_JPEG_QUALITY), policy.jpeg_quality]
    try:
        while len(samples) < policy.max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = round(frame_index * 1000.0 / fps)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            delta = 0.0 if previous_gray is None else float(cv2.absdiff(gray, previous_gray).mean()) / 255.0
            elapsed_ms = None if last_sample_ms is None else timestamp_ms - last_sample_ms

            reason = "first_frame" if last_sample_ms is None else ""
            if last_sample_ms is not None and elapsed_ms is not None:
                if elapsed_ms >= round(policy.max_interval_s * 1000):
                    reason = "max_interval"
                elif (elapsed_ms >= round(policy.min_interval_s * 1000) and
                      delta >= policy.scene_change_threshold):
                    reason = "scene_change"

            if reason:
                resized = _resize_frame(frame, policy.max_frame_side, cv2)
                ok_jpeg, encoded = cv2.imencode(".jpg", resized, image_params)
                if not ok_jpeg:
                    raise RuntimeError("OpenCV failed to encode an evidence frame")
                payload = encoded.tobytes()
                frame_path = frame_dir / f"frame_{frame_index:09d}_{timestamp_ms:012d}.jpg"
                frame_path.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                samples.append(FrameSample(
                    media_id=media_id,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    frame_path=frame_path,
                    frame_sha256=digest,
                    scene_delta=delta,
                    sample_reason=reason,
                ))
                last_sample_ms = timestamp_ms
                previous_gray = gray
            elif previous_gray is None or delta >= policy.scene_change_threshold / 4:
                # Compare changes against the most recent meaningful visual state.
                previous_gray = gray
            frame_index += 1
            if progress_callback is not None and (frame_index % 30 == 0 or
                                                   (source_frames and frame_index >= source_frames)):
                progress_callback(frame_index, source_frames)
    finally:
        capture.release()

    if frame_index and duration_ms <= 0:
        duration_ms = round(frame_index * 1000.0 / fps)
    metadata = VideoMetadata(
        media_id=media_id,
        source_path=video_path,
        source_sha256=source_hash,
        duration_ms=duration_ms,
        source_frame_count=source_frames or frame_index,
        fps=fps,
        width=width,
        height=height,
    )
    return metadata, samples
