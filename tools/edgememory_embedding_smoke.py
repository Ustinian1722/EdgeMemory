#!/usr/bin/env python3
"""Exercise real llama.cpp embeddings against a synthetic EventStore fixture.

This is a local software integration check. The captions are hand-authored test
fixtures; they are not observations from a video or an IQ-9075 device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from edgememory.embeddings import LlamaCppEmbeddingProvider
from edgememory.store import EventStore
from edgememory.types import VideoMetadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CAPTIONS = (
    ("umbrella", "A commuter walks along a wet sidewalk carrying a red umbrella."),
    ("dog", "A dog rests on a rug beside a fireplace."),
    ("cycling", "A cyclist waits beside a traffic light at an intersection."),
    ("blocks", "A child stacks colorful wooden blocks on a table."),
)
QUERY = "Someone is outside during rainfall and staying dry beneath a handheld canopy."
EXPECTED_EVENT_ID = "fixture:umbrella"


def _project_output_path(path: Path) -> Path:
    root = PROJECT_ROOT.resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("--output must stay inside the D-drive project workspace") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_smoke(executable: Path, model_path: Path, threads: int) -> dict:
    if not executable.is_file():
        raise FileNotFoundError("llama-embedding executable does not exist")
    if not model_path.is_file():
        raise FileNotFoundError("embedding model file does not exist")

    embedder = LlamaCppEmbeddingProvider(str(executable), str(model_path), threads=threads)
    started = time.perf_counter()
    vectors = [(event_key, caption, embedder.embed(caption))
               for event_key, caption in FIXTURE_CAPTIONS]
    query_vector = embedder.embed(QUERY)
    elapsed_s = time.perf_counter() - started

    dimensions = len(query_vector)
    if dimensions == 0 or any(len(vector) != dimensions for _, _, vector in vectors):
        raise ValueError("embedding dimensions differ across fixture captions")
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if not math.isclose(query_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("query embedding is not L2-normalized")

    store = EventStore(":memory:")
    try:
        metadata = VideoMetadata(
            media_id="fixture:synthetic-captions",
            source_path=Path("fixture://synthetic-captions.mp4"),
            source_sha256=hashlib.sha256(b"synthetic-caption-fixture-v1").hexdigest(),
            duration_ms=4000,
            source_frame_count=len(vectors),
            fps=1.0,
            width=1,
            height=1,
        )
        store.register_media(metadata, "Synthetic caption fixture")
        for frame_index, (event_key, caption, vector) in enumerate(vectors):
            store.upsert_event(
                event_id=f"fixture:{event_key}",
                media_id=metadata.media_id,
                frame_index=frame_index,
                timestamp_ms=frame_index * 1000,
                start_ms=frame_index * 1000,
                end_ms=(frame_index + 1) * 1000,
                frame_path=Path(f"fixture://frame-{frame_index}.jpg"),
                frame_sha256=hashlib.sha256(caption.encode("utf-8")).hexdigest(),
                sample_reason="synthetic fixture",
                scene_delta=0.0,
                caption=caption,
                caption_status="completed",
                backend_id="synthetic-fixture",
                model_id="hand-authored-captions",
                embedding=vector,
                embedding_model_id=embedder.model_id,
                simulation_only=True,
            )
        hits = store.search(
            QUERY,
            query_embedding=query_vector,
            query_embedding_model_id=embedder.model_id,
            limit=len(vectors),
        )
    finally:
        store.close()

    if not hits:
        raise RuntimeError("EventStore returned no results for the fixture query")
    top_hit = hits[0]
    report = {
        "schema_version": 1,
        "record_type": "local_embedding_fixture_smoke",
        "execution_environment": "local CPU host",
        "backend": "official llama.cpp embedding executable",
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "dimensions": dimensions,
        "query_l2_norm": query_norm,
        "fixture_caption_count": len(vectors),
        "query": QUERY,
        "top_event_id": top_hit.event_id,
        "top_cosine_similarity": top_hit.similarity,
        "top_caption_is_fixture": top_hit.simulation_only,
        "expected_top_event_id": EXPECTED_EVENT_ID,
        "retrieval_match": top_hit.event_id == EXPECTED_EVENT_ID,
        "elapsed_seconds": round(elapsed_s, 6),
        "device_measurement": False,
        "note": "Captions are synthetic fixtures, not video observations; this is not IQ-9075 performance evidence.",
    }
    if not report["retrieval_match"] or not top_hit.simulation_only:
        raise RuntimeError("semantic retrieval did not return the expected fixture")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-executable", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    try:
        output_path = _project_output_path(args.output) if args.output else None
    except ValueError as error:
        parser.error(str(error))

    report = run_smoke(args.embedding_executable, args.model_path, args.threads)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
