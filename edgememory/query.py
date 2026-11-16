from __future__ import annotations

from .embeddings import EmbeddingProvider
from .store import EventStore
from .types import SearchHit


def search_events(store: EventStore, query: str, *, embedder: EmbeddingProvider | None = None,
                  start_ms: int | None = None, end_ms: int | None = None,
                  media_id: str | None = None, limit: int = 20) -> list[SearchHit]:
    vector = embedder.embed(query) if embedder is not None and query.strip() else None
    return store.search(
        query, query_embedding=vector,
        query_embedding_model_id=embedder.model_id if vector is not None else None,
        start_ms=start_ms, end_ms=end_ms, media_id=media_id, limit=limit)


def summary_events(store: EventStore, media_id: str, *, limit: int = 32) -> tuple[list[SearchHit], int]:
    """Select chronological evidence across a video without favoring its opening."""
    rows, total = store.summary_timeline(media_id, limit=limit)
    hits = [SearchHit(
        event_id=row["event_id"],
        media_id=row["media_id"],
        media_title=row["media_title"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        timestamp_ms=row["timestamp_ms"],
        caption=row["caption"],
        backend_id=row["backend_id"],
        model_id=row["model_id"],
        frame_path=row["frame_path"],
        frame_sha256=row["frame_sha256"],
        similarity=None,
        simulation_only=row["simulation_only"],
        inference_metrics=row["inference_metrics"],
    ) for row in rows]
    return hits, total
