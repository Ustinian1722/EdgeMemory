from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
import threading
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .types import SearchHit, VideoMetadata


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


def _fts_normalize(text: str) -> str:
    """Tokenize CJK character runs for useful SQLite FTS5 matching without extensions."""
    output: list[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold()):
        if token[0].isascii():
            output.append(token)
        else:
            output.extend(token)
            output.extend(token[i:i + 2] for i in range(len(token) - 1))
    return " ".join(output)


def _pack_vector(values: Sequence[float]) -> bytes:
    vector = [float(value) for value in values]
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding must contain finite values")
    return struct.pack("<" + "f" * len(vector), *vector)


def _unpack_vector(blob: bytes, dimension: int) -> list[float]:
    if dimension < 1 or len(blob) != dimension * 4:
        raise ValueError("stored embedding has an invalid dimension")
    return list(struct.unpack("<" + "f" * dimension, blob))


class EventStore:
    def __init__(self, db_path: Path | str):
        self._lock = threading.RLock()
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    @_serialized
    def close(self) -> None:
        self.connection.close()

    @_serialized
    def _initialize(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                fps REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                indexed_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL REFERENCES media(media_id) ON DELETE CASCADE,
                frame_index INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                frame_path TEXT NOT NULL,
                frame_sha256 TEXT NOT NULL,
                sample_reason TEXT NOT NULL,
                scene_delta REAL NOT NULL,
                caption TEXT NOT NULL DEFAULT '',
                caption_status TEXT NOT NULL,
                error_code TEXT NOT NULL DEFAULT '',
                backend_id TEXT NOT NULL DEFAULT '',
                model_id TEXT NOT NULL DEFAULT '',
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model_id TEXT NOT NULL DEFAULT '',
                simulation_only INTEGER NOT NULL DEFAULT 0,
                inference_metrics_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(media_id, frame_index, frame_sha256)
            );
            CREATE INDEX IF NOT EXISTS events_media_time
                ON events(media_id, timestamp_ms);
            CREATE INDEX IF NOT EXISTS events_status_time
                ON events(caption_status, timestamp_ms);
            CREATE TABLE IF NOT EXISTS ingest_jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                media_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                completed_units INTEGER NOT NULL DEFAULT 0,
                total_units INTEGER NOT NULL DEFAULT 0,
                error_code TEXT NOT NULL DEFAULT '',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                started_at_utc TEXT NOT NULL DEFAULT '',
                finished_at_utc TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS ingest_jobs_status_created
                ON ingest_jobs(status, created_at_utc DESC);
            CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(
                event_id UNINDEXED, caption_tokens, tokenize='unicode61'
            );
        """)
        event_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(events)")
        }
        if "inference_metrics_json" not in event_columns:
            self.connection.execute(
                "ALTER TABLE events ADD COLUMN inference_metrics_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "embedding_model_id" not in event_columns:
            self.connection.execute(
                "ALTER TABLE events ADD COLUMN embedding_model_id TEXT NOT NULL DEFAULT ''"
            )
        self.connection.commit()

    @_serialized
    def register_media(self, metadata: VideoMetadata, title: str) -> None:
        self.connection.execute("""
            INSERT INTO media(media_id,title,source_path,source_sha256,duration_ms,fps,width,height,indexed_at_utc)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(media_id) DO UPDATE SET
                title=excluded.title, source_path=excluded.source_path,
                duration_ms=excluded.duration_ms, fps=excluded.fps,
                width=excluded.width, height=excluded.height,
                indexed_at_utc=excluded.indexed_at_utc
        """, (metadata.media_id, title, str(metadata.source_path), metadata.source_sha256,
              metadata.duration_ms, metadata.fps, metadata.width, metadata.height,
              datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self.connection.commit()

    @_serialized
    def upsert_event(self, *, event_id: str, media_id: str, frame_index: int,
                     timestamp_ms: int, start_ms: int, end_ms: int,
                     frame_path: Path, frame_sha256: str, sample_reason: str,
                     scene_delta: float, caption: str = "", caption_status: str,
                     error_code: str = "", backend_id: str = "", model_id: str = "",
                     embedding: Sequence[float] | None = None,
                     embedding_model_id: str = "",
                     simulation_only: bool = False,
                     inference_metrics: dict[str, int | float] | None = None) -> None:
        if embedding is not None:
            if (not isinstance(embedding_model_id, str) or not embedding_model_id.strip()
                    or len(embedding_model_id) > 1024):
                raise ValueError("embedding_model_id is required and must be at most 1024 characters")
        elif embedding_model_id:
            raise ValueError("embedding_model_id requires an embedding")
        blob = _pack_vector(embedding) if embedding is not None else None
        dimension = len(embedding) if embedding is not None else 0
        metrics = inference_metrics or {}
        normalized_metrics: dict[str, int | float] = {}
        for key, value in metrics.items():
            if (not isinstance(key, str) or not key or len(key) > 80 or
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise ValueError("inference metrics must be finite numeric values with short string keys")
            normalized_metrics[key] = value
        metrics_json = json.dumps(normalized_metrics, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False)
        values = (event_id, media_id, frame_index, timestamp_ms, start_ms, end_ms,
                  str(frame_path), frame_sha256, sample_reason, scene_delta, caption,
                  caption_status, error_code, backend_id, model_id, blob, dimension,
                  embedding_model_id, int(simulation_only), metrics_json)
        self.connection.execute("""
            INSERT INTO events(event_id,media_id,frame_index,timestamp_ms,start_ms,end_ms,
                frame_path,frame_sha256,sample_reason,scene_delta,caption,caption_status,
                error_code,backend_id,model_id,embedding,embedding_dim,embedding_model_id,
                simulation_only,inference_metrics_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
                start_ms=excluded.start_ms, end_ms=excluded.end_ms,
                caption=excluded.caption, caption_status=excluded.caption_status,
                error_code=excluded.error_code, backend_id=excluded.backend_id,
                model_id=excluded.model_id, embedding=excluded.embedding,
                embedding_dim=excluded.embedding_dim,
                embedding_model_id=excluded.embedding_model_id,
                simulation_only=excluded.simulation_only,
                inference_metrics_json=excluded.inference_metrics_json
        """, values)
        self.connection.execute("DELETE FROM event_fts WHERE event_id=?", (event_id,))
        if caption and caption_status == "completed":
            self.connection.execute("INSERT INTO event_fts(event_id,caption_tokens) VALUES(?,?)",
                                    (event_id, _fts_normalize(caption)))
        self.connection.commit()

    @_serialized
    def set_event_bounds(self, media_id: str, duration_ms: int) -> None:
        rows = self.connection.execute(
            "SELECT event_id,timestamp_ms FROM events WHERE media_id=? ORDER BY timestamp_ms",
            (media_id,)).fetchall()
        if not rows:
            return
        times = [int(row["timestamp_ms"]) for row in rows]
        mids = [(times[i] + times[i + 1]) // 2 for i in range(len(times) - 1)]
        starts = [0, *mids]
        ends = [*mids, max(duration_ms, times[-1])]
        self.connection.executemany(
            "UPDATE events SET start_ms=?,end_ms=? WHERE event_id=?",
            [(starts[i], max(starts[i], ends[i]), rows[i]["event_id"])
             for i in range(len(rows))])
        self.connection.commit()

    @_serialized
    def event(self, event_id: str) -> dict | None:
        row = self.connection.execute("""
            SELECT e.*,m.title FROM events e JOIN media m USING(media_id) WHERE event_id=?
        """, (event_id,)).fetchone()
        return dict(row) if row else None

    @_serialized
    def media(self) -> list[dict]:
        rows = self.connection.execute("""
            SELECT m.media_id,m.title,m.duration_ms,m.source_sha256,m.indexed_at_utc,
                   COUNT(e.event_id) AS event_count,
                   SUM(CASE WHEN e.caption_status='failed' THEN 1 ELSE 0 END) AS failed_count
            FROM media m LEFT JOIN events e USING(media_id)
            GROUP BY m.media_id ORDER BY m.indexed_at_utc DESC
        """).fetchall()
        return [dict(row) for row in rows]

    @_serialized
    def media_item(self, media_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM media WHERE media_id=?", (media_id,)).fetchone()
        return dict(row) if row else None

    @_serialized
    def create_ingest_job(self, job_id: str, title: str, source_path: Path | str) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.connection.execute("""
            INSERT INTO ingest_jobs(job_id,title,source_path,status,stage,progress,
                created_at_utc,updated_at_utc)
            VALUES(?,?,?,'queued','waiting',0,?,?)
        """, (job_id, title, str(source_path), now, now))
        self.connection.commit()
        return self._ingest_job(job_id)

    @_serialized
    def update_ingest_job(self, job_id: str, **changes) -> dict | None:
        allowed = {
            "media_id", "status", "stage", "progress", "completed_units",
            "total_units", "error_code", "started_at_utc", "finished_at_utc",
        }
        if not changes or set(changes) - allowed:
            raise ValueError("invalid ingest job update")
        changes = dict(changes)
        if "progress" in changes:
            value = float(changes["progress"])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("job progress must be between 0 and 100")
            changes["progress"] = value
        changes["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        columns = tuple(changes)
        values = [changes[name] for name in columns]
        values.append(job_id)
        self.connection.execute(
            "UPDATE ingest_jobs SET " + ",".join(f"{name}=?" for name in columns)
            + " WHERE job_id=?", values)
        self.connection.commit()
        return self._ingest_job(job_id)

    def _ingest_job(self, job_id: str) -> dict | None:
        row = self.connection.execute("""
            SELECT job_id,title,media_id,status,stage,progress,completed_units,
                   total_units,error_code,created_at_utc,updated_at_utc,
                   started_at_utc,finished_at_utc
            FROM ingest_jobs WHERE job_id=?
        """, (job_id,)).fetchone()
        return dict(row) if row else None

    @_serialized
    def ingest_job(self, job_id: str) -> dict | None:
        return self._ingest_job(job_id)

    @_serialized
    def ingest_jobs(self, limit: int = 50) -> list[dict]:
        if not 1 <= limit <= 200:
            raise ValueError("job limit must be in 1..200")
        rows = self.connection.execute("""
            SELECT job_id,title,media_id,status,stage,progress,completed_units,
                   total_units,error_code,created_at_utc,updated_at_utc,
                   started_at_utc,finished_at_utc
            FROM ingest_jobs ORDER BY created_at_utc DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]

    @_serialized
    def recoverable_ingest_jobs(self, limit: int = 200) -> list[dict]:
        """Expose source paths only to the local recovery manager, never the UI."""
        if not 1 <= limit <= 1000:
            raise ValueError("recovery job limit must be in 1..1000")
        rows = self.connection.execute("""
            SELECT job_id,title,source_path,status
            FROM ingest_jobs WHERE status IN ('queued','running')
            ORDER BY created_at_utc LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]

    @_serialized
    def timeline(self, media_id: str, *, start_ms: int | None = None,
                 end_ms: int | None = None, limit: int = 500) -> list[dict]:
        if not 1 <= limit <= 2000:
            raise ValueError("timeline limit must be in 1..2000")
        where = ["e.media_id=?"]
        params: list[object] = [media_id]
        if start_ms is not None:
            where.append("e.end_ms>=?")
            params.append(start_ms)
        if end_ms is not None:
            where.append("e.start_ms<=?")
            params.append(end_ms)
        params.append(limit)
        rows = self.connection.execute("""
            SELECT e.event_id,e.media_id,m.title AS media_title,e.frame_index,
                   e.timestamp_ms,e.start_ms,e.end_ms,e.frame_path,e.frame_sha256,
                   e.sample_reason,e.scene_delta,e.caption,e.caption_status,
                   e.error_code,e.backend_id,e.model_id,e.simulation_only,
                   e.inference_metrics_json
            FROM events e JOIN media m USING(media_id)
            WHERE """ + " AND ".join(where) + " ORDER BY e.timestamp_ms LIMIT ?", params).fetchall()
        return [{**dict(row),
                 "simulation_only": bool(row["simulation_only"]),
                 "inference_metrics": json.loads(row["inference_metrics_json"] or "{}")}
                for row in rows]

    @_serialized
    def summary_timeline(self, media_id: str, *, limit: int = 32) -> tuple[list[dict], int]:
        """Return time-spaced, completed real captions for an evidence-grounded summary."""
        if not isinstance(media_id, str) or not media_id:
            raise ValueError("summary requires a media_id")
        if not 1 <= limit <= 64:
            raise ValueError("summary evidence limit must be in 1..64")
        eligible = "e.media_id=? AND e.caption_status='completed' AND " \
                   "e.simulation_only=0 AND TRIM(e.caption)<>''"
        total = int(self.connection.execute(
            "SELECT COUNT(*) FROM events e WHERE " + eligible, (media_id,)).fetchone()[0])
        if total == 0:
            return [], 0
        rows = self.connection.execute("""
            WITH ranked AS (
                SELECT e.event_id,e.media_id,m.title AS media_title,e.frame_index,
                       e.timestamp_ms,e.start_ms,e.end_ms,e.frame_path,e.frame_sha256,
                       e.sample_reason,e.scene_delta,e.caption,e.caption_status,
                       e.error_code,e.backend_id,e.model_id,e.simulation_only,
                       e.inference_metrics_json,
                       ROW_NUMBER() OVER (ORDER BY e.timestamp_ms,e.event_id) AS rn
                FROM events e JOIN media m USING(media_id)
                WHERE """ + eligible + """
            ), bucketed AS (
                SELECT *, CASE WHEN ? >= ? THEN rn - 1
                    ELSE CAST((rn - 1) * 1.0 * (? - 1) / MAX(? - 1, 1) AS INTEGER)
                    END AS sample_bucket
                FROM ranked
            ), selected AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY sample_bucket ORDER BY rn) AS bucket_rank
                FROM bucketed
            )
            SELECT * FROM selected WHERE bucket_rank=1 ORDER BY rn
        """, (media_id, limit, total, limit, total)).fetchall()
        events = [{**dict(row),
                   "simulation_only": bool(row["simulation_only"]),
                   "inference_metrics": json.loads(row["inference_metrics_json"] or "{}")}
                  for row in rows]
        return events, total

    @_serialized
    def dashboard_counts(self) -> dict[str, int]:
        media_count = int(self.connection.execute("SELECT COUNT(*) FROM media").fetchone()[0])
        event_counts = self.connection.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN caption_status='completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN caption_status='failed' THEN 1 ELSE 0 END) AS failed
            FROM events
        """).fetchone()
        job_counts = self.connection.execute("""
            SELECT SUM(CASE WHEN status IN ('queued','running') THEN 1 ELSE 0 END) AS active,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
            FROM ingest_jobs
        """).fetchone()
        return {
            "media_count": media_count,
            "event_count": int(event_counts["total"] or 0),
            "completed_event_count": int(event_counts["completed"] or 0),
            "failed_event_count": int(event_counts["failed"] or 0),
            "active_job_count": int(job_counts["active"] or 0),
            "failed_job_count": int(job_counts["failed"] or 0),
        }

    @_serialized
    def runtime_observations(self, limit: int = 20) -> list[dict]:
        """Return recent non-simulation inference records for the runtime view."""
        if not 1 <= limit <= 100:
            raise ValueError("runtime observation limit must be in 1..100")
        rows = self.connection.execute("""
            SELECT m.title AS media_title,e.event_id,e.timestamp_ms,
                   e.caption_status,e.backend_id,e.model_id,e.inference_metrics_json
            FROM events e JOIN media m USING(media_id)
            WHERE e.caption_status='completed' AND e.simulation_only=0
            ORDER BY m.indexed_at_utc DESC,e.timestamp_ms DESC LIMIT ?
        """, (limit,)).fetchall()
        return [{**dict(row),
                 "inference_metrics": json.loads(row["inference_metrics_json"] or "{}")}
                for row in rows]

    @_serialized
    def search(self, query: str, *, query_embedding: Sequence[float] | None = None,
               query_embedding_model_id: str | None = None,
               start_ms: int | None = None, end_ms: int | None = None,
               media_id: str | None = None, limit: int = 20) -> list[SearchHit]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be in 1..200")
        if query_embedding is not None and (
                not isinstance(query_embedding_model_id, str)
                or not query_embedding_model_id.strip()):
            raise ValueError("query_embedding_model_id is required when searching with a vector")
        where = ["e.caption_status='completed'"]
        params: list[object] = []
        if start_ms is not None:
            where.append("e.end_ms>=?")
            params.append(start_ms)
        if end_ms is not None:
            where.append("e.start_ms<=?")
            params.append(end_ms)
        if media_id is not None:
            where.append("e.media_id=?")
            params.append(media_id)
        base_sql = """
            SELECT e.*,m.title FROM events e JOIN media m USING(media_id)
            WHERE """ + " AND ".join(where)
        rows = self.connection.execute(base_sql, params).fetchall()
        if not rows:
            return []

        query_tokens = [token for token in _fts_normalize(query).split() if token]
        lexical_rank: dict[str, int] = {}
        if query_tokens:
            expression = " OR ".join('"' + token.replace('"', '""') + '"'
                                    for token in dict.fromkeys(query_tokens))
            try:
                fts = self.connection.execute("""
                    SELECT event_id FROM event_fts WHERE event_fts MATCH ? ORDER BY bm25(event_fts)
                """, (expression,)).fetchall()
                lexical_rank = {row["event_id"]: rank for rank, row in enumerate(fts)}
            except sqlite3.OperationalError:
                lexical_rank = {}

        qvector = _pack_vector(query_embedding) if query_embedding is not None else None
        qvalues = list(query_embedding) if query_embedding is not None else []
        qnorm = math.sqrt(sum(value * value for value in qvalues)) if qvalues else 0.0
        ranked: list[tuple[float, float | None, sqlite3.Row]] = []
        for row in rows:
            lexical = (1.0 - lexical_rank[row["event_id"]] / max(1, len(lexical_rank))) \
                if row["event_id"] in lexical_rank else 0.0
            similarity = None
            if (qvector is not None and row["embedding"] is not None
                    and row["embedding_model_id"] == query_embedding_model_id):
                values = _unpack_vector(row["embedding"], int(row["embedding_dim"]))
                if len(values) == len(qvalues) and qnorm:
                    norm = math.sqrt(sum(value * value for value in values))
                    if norm:
                        similarity = max(-1.0, min(1.0,
                            sum(a * b for a, b in zip(qvalues, values)) / (qnorm * norm)))
            if qvector is not None:
                semantic = (similarity + 1.0) / 2.0 if similarity is not None else 0.0
                score = 0.75 * semantic + 0.25 * lexical
            else:
                score = lexical
            ranked.append((score, similarity, row))
        ranked.sort(key=lambda item: (item[0], item[1] or -2.0,
                                      -int(item[2]["timestamp_ms"])), reverse=True)
        hits: list[SearchHit] = []
        for score, similarity, row in ranked:
            if score <= 0 and query.strip():
                continue
            hits.append(SearchHit(
                event_id=row["event_id"], media_id=row["media_id"],
                media_title=row["title"], start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]), timestamp_ms=int(row["timestamp_ms"]),
                caption=row["caption"], backend_id=row["backend_id"],
                model_id=row["model_id"], frame_path=row["frame_path"],
                frame_sha256=row["frame_sha256"], similarity=similarity,
                simulation_only=bool(row["simulation_only"]),
                inference_metrics=json.loads(row["inference_metrics_json"] or "{}"),
            ))
        return hits[:limit]
