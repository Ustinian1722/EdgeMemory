"""Small SQLite C-API bridge for target Python builds without _sqlite3.

IQ-9075 images may expose libsqlite3 while omitting CPython's optional
_sqlite3 extension. This module uses that system library directly; it installs
nothing and keeps the persistent event index on the device.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import re
import struct
import threading
import time
from pathlib import Path


class DeviceSQLite:
    SQLITE_OK = 0
    SQLITE_ROW = 100

    def __init__(self, path: Path):
        library = ctypes.util.find_library("sqlite3")
        if not library:
            raise RuntimeError("device libsqlite3 was not found")
        self._lib = ctypes.CDLL(library)
        self._configure_abi()
        self._db = ctypes.c_void_p()
        path.parent.mkdir(parents=True, exist_ok=True)
        result = self._lib.sqlite3_open_v2(
            str(path).encode(), ctypes.byref(self._db),
            0x00000002 | 0x00000004 | 0x00010000, None)
        if result != self.SQLITE_OK:
            raise RuntimeError("device SQLite database could not be opened")
        self._lock = threading.RLock()
        self._rows: list[dict[str, str | None]] = []
        self._callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p))
        self._callback = self._callback_type(self._collect_row)
        self.execute("PRAGMA journal_mode=WAL")
        self.execute("PRAGMA synchronous=NORMAL")
        self.execute("PRAGMA busy_timeout=5000")
        self.execute("""
            CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            )
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                caption TEXT NOT NULL,
                caption_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS events_media_time ON events(media_id, timestamp_ms)")
        self.execute("CREATE INDEX IF NOT EXISTS events_created ON events(created_ns)")
        self.execute("""
            CREATE TABLE IF NOT EXISTS visual_embeddings (
                event_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_hex TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS visual_embeddings_model ON visual_embeddings(model_id)")
        self.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_ns DESC)")
        self.execute("""
            CREATE TABLE IF NOT EXISTS inference (
                inference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id TEXT,
                event_id TEXT,
                stage TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)

    def _configure_abi(self) -> None:
        lib = self._lib
        lib.sqlite3_open_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_char_p]
        lib.sqlite3_open_v2.restype = ctypes.c_int
        lib.sqlite3_exec.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.sqlite3_exec.restype = ctypes.c_int
        lib.sqlite3_free.argtypes = [ctypes.c_void_p]
        lib.sqlite3_free.restype = None
        lib.sqlite3_errmsg.argtypes = [ctypes.c_void_p]
        lib.sqlite3_errmsg.restype = ctypes.c_char_p
        lib.sqlite3_close_v2.argtypes = [ctypes.c_void_p]
        lib.sqlite3_close_v2.restype = ctypes.c_int

    def _collect_row(self, _context, count, values, names):
        row: dict[str, str | None] = {}
        for index in range(count):
            key = names[index].decode("utf-8", "replace") if names[index] else str(index)
            value = values[index].decode("utf-8", "replace") if values[index] else None
            row[key] = value
        self._rows.append(row)
        return 0

    @staticmethod
    def _sql_text(value: object) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def execute(self, statement: str) -> list[dict[str, str | None]]:
        with self._lock:
            self._rows = []
            error = ctypes.c_void_p()
            result = self._lib.sqlite3_exec(
                self._db, statement.encode("utf-8"), self._callback,
                None, ctypes.byref(error))
            if result != self.SQLITE_OK:
                message = (ctypes.string_at(error).decode("utf-8", "replace")
                           if error.value else self._lib.sqlite3_errmsg(self._db).decode("utf-8", "replace"))
                if error.value:
                    self._lib.sqlite3_free(error)
                raise RuntimeError(f"device SQLite operation failed: {message[:240]}")
            return list(self._rows)

    def put_media(self, record: dict) -> None:
        sql = self._sql_text
        now = time.time_ns()
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.execute(
            "INSERT INTO media(media_id,title,source_path,source_sha256,payload_json,created_ns) "
            f"VALUES({sql(record['media_id'])},{sql(record['title'])},{sql(record['source_path'])},"
            f"{sql(record['source_sha256'])},{sql(payload)},{now}) "
            "ON CONFLICT(media_id) DO UPDATE SET title=excluded.title,source_path=excluded.source_path,"
            "source_sha256=excluded.source_sha256,payload_json=excluded.payload_json")

    def put_event(self, record: dict) -> None:
        sql = self._sql_text
        now = time.time_ns()
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.execute(
            "INSERT INTO events(event_id,media_id,timestamp_ms,caption,caption_status,payload_json,created_ns) "
            f"VALUES({sql(record['event_id'])},{sql(record['media_id'])},"
            f"{int(record['timestamp_ms'])},{sql(record.get('caption',''))},"
            f"{sql(record.get('caption_status','completed'))},{sql(payload)},{now}) "
            "ON CONFLICT(event_id) DO UPDATE SET media_id=excluded.media_id,"
            "timestamp_ms=excluded.timestamp_ms,caption=excluded.caption,"
            "caption_status=excluded.caption_status,payload_json=excluded.payload_json")

    def put_visual_embedding(self, event_id: str, model_id: str,
                             vector: list[float]) -> None:
        if not event_id or not model_id or not vector:
            raise ValueError("visual event ID, model ID and vector are required")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("visual embedding values must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("visual embedding must have non-zero norm")
        packed = struct.pack("<" + "f" * len(values), *(value / norm for value in values))
        self.execute(
            "INSERT INTO visual_embeddings(event_id,model_id,dimensions,vector_hex,created_ns) "
            f"VALUES({self._sql_text(event_id)},{self._sql_text(model_id)},{len(values)},"
            f"{self._sql_text(packed.hex())},{time.time_ns()}) "
            "ON CONFLICT(event_id) DO UPDATE SET model_id=excluded.model_id,"
            "dimensions=excluded.dimensions,vector_hex=excluded.vector_hex,created_ns=excluded.created_ns")

    def put_job(self, record: dict) -> None:
        sql = self._sql_text
        now = time.time_ns()
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.execute(
            "INSERT INTO jobs(job_id,status,created_ns,payload_json) "
            f"VALUES({sql(record['job_id'])},{sql(record['status'])},{now},{sql(payload)}) "
            "ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json")

    def add_inference(self, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.execute(
            "INSERT INTO inference(media_id,event_id,stage,created_ns,payload_json) VALUES(" +
            ",".join((self._sql_text(record.get("media_id") or ""),
                      self._sql_text(record.get("event_id") or ""),
                      self._sql_text(record.get("stage", "inference")),
                      str(time.time_ns()), self._sql_text(payload))) + ")")

    @staticmethod
    def _decode_payload(rows: list[dict[str, str | None]], field: str = "payload_json") -> list[dict]:
        return [json.loads(row[field]) for row in rows if row.get(field)]

    def media(self) -> list[dict]:
        rows = self.execute(
            "SELECT payload_json FROM media ORDER BY created_ns DESC")
        result = self._decode_payload(rows)
        for item in result:
            media_id = self._sql_text(item["media_id"])
            counts = self.execute(
                f"SELECT COUNT(*) AS total,SUM(CASE WHEN caption_status='failed' THEN 1 ELSE 0 END) AS failed "
                f"FROM events WHERE media_id={media_id}")[0]
            item["event_count"] = int(counts.get("total") or 0)
            item["failed_count"] = int(counts.get("failed") or 0)
        return result

    def events(self, media_id: str | None = None, *, limit: int = 1000) -> list[dict]:
        clause = f" WHERE media_id={self._sql_text(media_id)}" if media_id else ""
        rows = self.execute(
            f"SELECT payload_json FROM events{clause} ORDER BY timestamp_ms,event_id LIMIT {int(limit)}")
        return self._decode_payload(rows)

    def summary_timeline(self, media_id: str, *, start_ms: int | None = None,
                         end_ms: int | None = None, limit: int = 8) -> tuple[list[dict], int]:
        """Return segment-aware completed events without loading the full timeline."""
        if not media_id or not 1 <= limit <= 32:
            raise ValueError("summary timeline requires a media ID and limit in 1..32")
        clauses = ["media_id=" + self._sql_text(media_id),
                   "caption_status='completed'"]
        if start_ms is not None:
            clauses.append("timestamp_ms>=" + str(int(start_ms)))
        if end_ms is not None:
            clauses.append("timestamp_ms<=" + str(int(end_ms)))
        where = " AND ".join(clauses)
        row = self.execute(
            "SELECT COUNT(DISTINCT timestamp_ms) AS total FROM events WHERE " + where)[0]
        total = int(row.get("total") or 0)
        if total <= limit:
            offsets = list(range(total))
        elif limit == 1:
            offsets = [total - 1]
        else:
            offsets = [round(i * (total - 1) / (limit - 1)) for i in range(limit)]
        selected_times = set()
        if total > limit:
            segments = self.execute(
                "SELECT CAST(timestamp_ms / 60000 AS INTEGER) AS segment, "
                "MIN(timestamp_ms) AS first_ms, MAX(timestamp_ms) AS last_ms "
                "FROM events WHERE " + where + " GROUP BY segment ORDER BY segment")
            if len(segments) > limit:
                indexes = [round(i * (len(segments) - 1) / (limit - 1))
                           for i in range(limit)] if limit > 1 else [len(segments) - 1]
                segments = [segments[i] for i in indexes]
            for segment in segments:
                midpoint = (int(segment["first_ms"]) + int(segment["last_ms"])) // 2
                candidates = self.execute(
                    "SELECT timestamp_ms FROM events WHERE " + where +
                    " AND CAST(timestamp_ms / 60000 AS INTEGER)=" + str(int(segment["segment"])) +
                    " ORDER BY ABS(timestamp_ms - " + str(midpoint) + "), timestamp_ms LIMIT 1")
                if candidates:
                    selected_times.add(int(candidates[0]["timestamp_ms"]))
        for offset in offsets:
            if len(selected_times) >= limit:
                break
            timestamp_rows = self.execute(
                "SELECT timestamp_ms FROM events WHERE " + where +
                " GROUP BY timestamp_ms ORDER BY timestamp_ms LIMIT 1 OFFSET " + str(offset))
            if timestamp_rows:
                selected_times.add(int(timestamp_rows[0]["timestamp_ms"]))
        selected = []
        for timestamp in sorted(selected_times):
            rows = self.execute(
                "SELECT payload_json FROM events WHERE " + where +
                " AND timestamp_ms=" + str(timestamp) +
                " ORDER BY created_ns DESC,event_id DESC LIMIT 1")
            selected.extend(self._decode_payload(rows))
        return selected, total

    def jobs(self, limit: int = 50) -> list[dict]:
        rows = self.execute(f"SELECT payload_json FROM jobs ORDER BY created_ns DESC LIMIT {int(limit)}")
        return self._decode_payload(rows)

    def get_event(self, event_id: str) -> dict | None:
        rows = self.execute(
            "SELECT payload_json FROM events WHERE event_id=" + self._sql_text(event_id) + " LIMIT 1")
        decoded = self._decode_payload(rows)
        return decoded[0] if decoded else None

    def search(self, query: str, media_id: str | None = None, *, limit: int = 20,
               query_embedding: list[float] | None = None,
               query_embedding_model_id: str = "",
               start_ms: int | None = None, end_ms: int | None = None) -> list[dict]:
        query = query.strip()
        if not query or limit < 1:
            return []
        if query_embedding is not None and not query_embedding_model_id:
            raise ValueError("query embedding model ID is required for semantic search")
        candidates = [row for row in self.events(media_id, limit=10000)
                      if row.get("caption_status") == "completed"
                      and (start_ms is None or int(row.get("end_ms", row.get("timestamp_ms", 0))) >= start_ms)
                      and (end_ms is None or int(row.get("start_ms", row.get("timestamp_ms", 0))) <= end_ms)]
        normalized = query.casefold()
        qgrams = {normalized[i:i + 2] for i in range(max(0, len(normalized) - 1))}
        qnorm = math.sqrt(sum(float(value) ** 2 for value in query_embedding or []))
        ranked: list[tuple[float, dict]] = []
        for item in candidates:
            text = str(item.get("caption", "")).casefold()
            lexical = 1.0 if normalized in text else 0.0
            if lexical == 0.0 and qgrams:
                grams = {text[i:i + 2] for i in range(max(0, len(text) - 1))}
                lexical = len(qgrams & grams) / len(qgrams)
            semantic = 0.0
            stored = item.get("embedding")
            if (query_embedding is not None and isinstance(stored, list)
                    and item.get("embedding_model_id") == query_embedding_model_id
                    and len(stored) == len(query_embedding) and qnorm > 0):
                try:
                    dot = sum(float(a) * float(b) for a, b in zip(query_embedding, stored))
                    snorm = math.sqrt(sum(float(value) ** 2 for value in stored))
                    if snorm > 0:
                        semantic = max(-1.0, min(1.0, dot / (qnorm * snorm)))
                        semantic = max(0.0, semantic)
                except (TypeError, ValueError, OverflowError):
                    semantic = 0.0
            if query_embedding is not None:
                score = 0.55 * semantic + 0.45 * lexical
            else:
                score = lexical
            if score > 0:
                result = dict(item)
                result["lexical_score"] = lexical
                result["semantic_score"] = semantic
                result["similarity"] = score
                ranked.append((score, result))
        ranked.sort(key=lambda pair: (-pair[0], int(pair[1].get("timestamp_ms", 0))))
        unique = []
        seen_timestamps = set()
        for _score, row in ranked:
            key = (row.get("media_id"), int(row.get("timestamp_ms", 0)))
            if key in seen_timestamps:
                continue
            seen_timestamps.add(key)
            unique.append(row)
            if len(unique) >= limit:
                break
        return unique

    def search_visual(self, query_embedding: list[float], model_id: str, *,
                      media_id: str | None = None, start_ms: int | None = None,
                      end_ms: int | None = None, limit: int = 20) -> list[dict]:
        if not query_embedding or not model_id or limit < 1:
            return []
        values = [float(value) for value in query_embedding]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("visual query embedding values must be finite")
        qnorm = math.sqrt(sum(value * value for value in values))
        if qnorm <= 0:
            return []
        clauses = ["v.model_id=" + self._sql_text(model_id), "v.dimensions=" + str(len(values))]
        if media_id:
            clauses.append("e.media_id=" + self._sql_text(media_id))
        if start_ms is not None:
            clauses.append("e.timestamp_ms>=" + str(int(start_ms)))
        if end_ms is not None:
            clauses.append("e.timestamp_ms<=" + str(int(end_ms)))
        rows = self.execute(
            "SELECT e.payload_json,v.vector_hex FROM visual_embeddings v "
            "JOIN events e ON e.event_id=v.event_id WHERE " + " AND ".join(clauses))
        ranked: list[tuple[float, dict]] = []
        for row in rows:
            payload = row.get("payload_json")
            vector_hex = row.get("vector_hex")
            if not payload or not vector_hex:
                continue
            try:
                item = json.loads(payload)
                stored = struct.unpack("<" + "f" * len(values), bytes.fromhex(vector_hex))
                score = sum(a * b for a, b in zip(values, stored)) / qnorm
            except (ValueError, TypeError, struct.error, json.JSONDecodeError):
                continue
            if not math.isfinite(score):
                continue
            result = dict(item)
            result["visual_similarity"] = max(-1.0, min(1.0, score))
            result["visual_model_id"] = model_id
            ranked.append((score, result))
        ranked.sort(key=lambda pair: (-pair[0], int(pair[1].get("timestamp_ms", 0))))
        unique = []
        seen_timestamps = set()
        for score, row in ranked:
            key = (row.get("media_id"), int(row.get("timestamp_ms", 0)))
            if key in seen_timestamps:
                continue
            seen_timestamps.add(key)
            if score > 0:
                unique.append(row)
            if len(unique) >= limit:
                break
        return unique

    def latest_visual_similarity(self, query_embedding: list[float], model_id: str,
                                 media_id: str) -> tuple[dict | None, float | None]:
        if not query_embedding or not model_id or not media_id:
            return None, None
        row = self.execute(
            "SELECT e.payload_json,v.vector_hex,v.dimensions FROM visual_embeddings v "
            "JOIN events e ON e.event_id=v.event_id WHERE v.model_id=" +
            self._sql_text(model_id) + " AND e.media_id=" + self._sql_text(media_id) +
            " ORDER BY e.timestamp_ms DESC LIMIT 1")
        if not row:
            return None, None
        try:
            item = json.loads(row[0]["payload_json"] or "{}")
            stored = struct.unpack("<" + "f" * int(row[0]["dimensions"]),
                                   bytes.fromhex(row[0]["vector_hex"] or ""))
            query = [float(value) for value in query_embedding]
            if len(stored) != len(query):
                return item, None
            qnorm = math.sqrt(sum(value * value for value in query))
            snorm = math.sqrt(sum(value * value for value in stored))
            if not qnorm or not snorm:
                return item, None
            score = sum(a * b for a, b in zip(query, stored)) / (qnorm * snorm)
            return item, max(-1.0, min(1.0, score)) if math.isfinite(score) else None
        except (ValueError, TypeError, struct.error, json.JSONDecodeError):
            return None, None

    def visual_embedding_count(self) -> int:
        row = self.execute("SELECT COUNT(*) AS total FROM visual_embeddings")[0]
        return int(row.get("total") or 0)

    def counts(self) -> dict[str, int]:
        def count(table: str, where: str = "") -> int:
            row = self.execute(f"SELECT COUNT(*) AS n FROM {table}{where}")[0]
            return int(row.get("n") or 0)
        return {
            "media_count": count("media"),
            "event_count": count("events"),
            "completed_event_count": count("events", " WHERE caption_status='completed'"),
            "visual_only_event_count": count("events", " WHERE caption_status='visual_only'"),
            "failed_event_count": count("events", " WHERE caption_status='failed'"),
            "active_job_count": count("jobs", " WHERE status IN ('queued','running')"),
            "failed_job_count": count("jobs", " WHERE status='failed'"),
        }

    def inferences(self, limit: int = 20) -> list[dict]:
        rows = self.execute(
            f"SELECT payload_json FROM inference ORDER BY created_ns DESC LIMIT {int(limit)}")
        return self._decode_payload(rows)

    def close(self) -> None:
        if self._db:
            self._lib.sqlite3_close_v2(self._db)
            self._db = ctypes.c_void_p()
