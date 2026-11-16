from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from edgememory.captions import MockCaptioner, OpenAICompatibleVlmCaptioner
from edgememory.embeddings import DeterministicTestEmbedder
from edgememory.jobs import IngestJobManager
from edgememory.pipeline import EdgeMemoryPipeline
from edgememory.query import search_events
from edgememory.store import EventStore
from edgememory.types import FrameSample, VideoMetadata
from edgememory.web import make_handler, serve
from tools.run_manifest import verify as verify_manifest


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / ".local" / "test-tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class EdgeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _event(self, store: EventStore, event_id: str, caption: str,
               timestamp: int, vector: list[float] | None = None,
               embedding_model_id: str | None = None,
               simulation_only: bool = False,
               inference_metrics: dict[str, int | float] | None = None):
        media = VideoMetadata("media-1", self.root / "source.mp4", "source-hash",
                              20_000, 100, 5.0, 640, 480)
        store.register_media(media, "fixture-video")
        frame = self.root / f"{event_id}.jpg"
        frame.write_bytes(b"frame evidence")
        store.upsert_event(
            event_id=event_id, media_id=media.media_id, frame_index=timestamp,
            timestamp_ms=timestamp, start_ms=timestamp, end_ms=timestamp + 1000,
            frame_path=frame, frame_sha256="evidence-hash-" + event_id,
            sample_reason="fixture", scene_delta=0.2, caption=caption,
            caption_status="completed", backend_id="fixture", model_id="fixture-model",
            embedding=vector,
            embedding_model_id=(embedding_model_id or "fixture-embedding") if vector is not None else "",
            simulation_only=simulation_only,
            inference_metrics=inference_metrics,
        )
        return frame

    def test_chinese_fts_returns_frame_and_time_evidence(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            frame = self._event(store, "evt-red", "画面中有人拿起红色杯子并放在桌面", 5000)
            self._event(store, "evt-door", "一辆汽车驶过停车场入口", 15000)
            hits = store.search("红色杯子", start_ms=4000, end_ms=9000)
            self.assertEqual([hit.event_id for hit in hits], ["evt-red"])
            self.assertEqual(hits[0].timestamp_ms, 5000)
            self.assertEqual(Path(hits[0].frame_path), frame)
            self.assertTrue(hits[0].frame_sha256.startswith("evidence-hash-"))
            self.assertEqual(store.search("红色杯子", start_ms=10_000), [])
        finally:
            store.close()

    def test_semantic_ranking_can_retrieve_without_lexical_overlap(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            self._event(store, "evt-a", "A person opens a door", 1000, [1.0, 0.0])
            self._event(store, "evt-b", "A vehicle enters the scene", 2000, [0.0, 1.0])
            hits = store.search("something", query_embedding=[0.9, 0.1],
                                query_embedding_model_id="fixture-embedding")
            self.assertEqual(hits[0].event_id, "evt-a")
            self.assertGreater(hits[0].similarity, hits[1].similarity)
        finally:
            store.close()

    def test_pipeline_marks_mock_output_simulation_and_manifests_run(self):
        store = EventStore(self.root / "index.sqlite3")
        frame = self.root / "frame.jpg"
        frame.write_bytes(b"mock-image")
        metadata = VideoMetadata("media-2", self.root / "private-source.mp4", "video-hash",
                                 10_000, 100, 10.0, 640, 480)
        sample = FrameSample("media-2", 12, 1200, frame, "frame-hash", 0.4, "scene_change")
        run_dir = self.root / "run"
        try:
            with patch("edgememory.pipeline.extract_adaptive_frames", return_value=(metadata, [sample])):
                summary = EdgeMemoryPipeline(
                    store, MockCaptioner(["SIMULATED: person opens a cabinet"]),
                    DeterministicTestEmbedder()).ingest_video(
                        metadata.source_path, self.root / "frame-root", run_dir=run_dir)
            self.assertTrue(summary.simulation_only)
            self.assertEqual(summary.indexed_events, 1)
            self.assertEqual(summary.failed_events, 0)
            embedder = DeterministicTestEmbedder()
            hit = search_events(store, "cabinet", embedder=embedder)[0]
            self.assertTrue(hit.simulation_only)
            record = json.loads((run_dir / "ingest.json").read_text(encoding="utf-8"))
            self.assertTrue(record["simulation_only"])
            self.assertEqual(verify_manifest(run_dir)["status"], "verified")
        finally:
            store.close()

    def test_vlm_adapter_sends_image_and_returns_real_backend_metadata(self):
        observed: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                observed.update(json.loads(self.rfile.read(length)))
                payload = json.dumps({"choices": [{"message": {
                    "content": "A blue car stops near the building."}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        image = self.root / "evidence.jpg"
        image.write_bytes(b"small test image payload")
        try:
            captioner = OpenAICompatibleVlmCaptioner(
                f"http://127.0.0.1:{server.server_port}/v1", "fixture-vlm")
            result = captioner.describe(image, "caption the visible scene")
            self.assertEqual(result.text, "A blue car stops near the building.")
            self.assertFalse(result.simulation_only)
            self.assertEqual(result.model_id, "fixture-vlm")
            content = observed["messages"][0]["content"]
            self.assertEqual(content[0]["text"], "caption the visible scene")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_empty_query_returns_nothing_instead_of_arbitrary_rows(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            self._event(store, "evt-a", "a person opens a door", 1000)
            self.assertEqual(store.search("no matching phrase"), [])
        finally:
            store.close()

    def test_semantic_search_never_compares_vectors_from_another_model(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            self._event(store, "evt-other-model", "caption from another space", 2000,
                        [0.0, 1.0], embedding_model_id="embedding-model-a")
            self._event(store, "evt-query-model", "caption from query space", 1000,
                        [0.0, 1.0], embedding_model_id="embedding-model-b")
            hits = store.search("no lexical match", query_embedding=[0.0, 1.0],
                                query_embedding_model_id="embedding-model-b")
            self.assertEqual([hit.event_id for hit in hits], ["evt-query-model"])
            self.assertEqual(hits[0].similarity, 1.0)
        finally:
            store.close()

    def test_vector_search_requires_query_model_identity(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            with self.assertRaisesRegex(ValueError, "query_embedding_model_id is required"):
                store.search("query", query_embedding=[1.0, 0.0])
        finally:
            store.close()

    def test_old_index_adds_embedding_model_identity_column(self):
        db_path = self.root / "index.sqlite3"
        store = EventStore(db_path)
        self._event(store, "evt-legacy", "legacy searchable caption", 1000,
                    [1.0, 0.0], embedding_model_id="legacy-model")
        store.connection.execute("ALTER TABLE events DROP COLUMN embedding_model_id")
        store.connection.commit()
        store.close()

        migrated = EventStore(db_path)
        try:
            columns = {row["name"] for row in migrated.connection.execute("PRAGMA table_info(events)")}
            self.assertIn("embedding_model_id", columns)
            hits = migrated.search(
                "legacy searchable caption", query_embedding=[1.0, 0.0],
                query_embedding_model_id="legacy-model")
            self.assertEqual([hit.event_id for hit in hits], ["evt-legacy"])
            self.assertIsNone(hits[0].similarity)
        finally:
            migrated.close()

    def test_real_backend_metrics_survive_database_search(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            self._event(store, "evt-metrics", "red rectangle beside blue circle", 1000,
                        inference_metrics={"ttft_us": 1200000,
                                           "decode_tokens_per_second": 31.5})
            hit = store.search("red rectangle")[0]
            self.assertEqual(hit.inference_metrics["ttft_us"], 1200000)
            self.assertEqual(hit.inference_metrics["decode_tokens_per_second"], 31.5)
        finally:
            store.close()

    def test_video_endpoint_streams_ranges_and_rejects_paths_outside_allowlist(self):
        store = EventStore(self.root / "index.sqlite3")
        video = self.root / "source.mp4"
        video.write_bytes(b"0123456789abcdef")
        metadata = VideoMetadata("media-playback", video,
                                 hashlib.sha256(video.read_bytes()).hexdigest(), 1000,
                                 1, 1.0, 16, 16)
        store.register_media(metadata, "range fixture")
        frame = self.root / "playback-frame.jpg"
        frame.write_bytes(b"frame")
        store.upsert_event(
            event_id="evt-playback", media_id="media-playback", frame_index=0,
            timestamp_ms=400, start_ms=0, end_ms=1000, frame_path=frame,
            frame_sha256="frame-hash", sample_reason="fixture", scene_delta=0.0,
            caption="a car stops near the gate", caption_status="completed",
            backend_id="fixture", model_id="fixture-model")
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(store, self.root, video_root=self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base}/", timeout=3) as response:
                page = response.read()
                self.assertIn(b"/assets/app.js", page)
            with urllib.request.urlopen(f"{base}/assets/app.js", timeout=3) as response:
                script = response.read()
                self.assertIn(b"/api/video/", script)
                self.assertIn("定位回放".encode("utf-8"), script)
            query_request = urllib.request.Request(
                f"{base}/api/query",
                data=json.dumps({"query": "car stops"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(query_request, timeout=3) as response:
                query_result = json.loads(response.read())
            self.assertTrue(query_result["results"][0]["video_available"])

            request = urllib.request.Request(
                f"{base}/api/video/media-playback", headers={"Range": "bytes=4-7"})
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 4-7/16")
                self.assertEqual(response.read(), b"4567")
            invalid_range = urllib.request.Request(
                f"{base}/api/video/media-playback", headers={"Range": "bytes=16-"})
            with self.assertRaises(urllib.error.HTTPError) as range_error:
                urllib.request.urlopen(invalid_range, timeout=3)
            self.assertEqual(range_error.exception.code, 416)
            self.assertEqual(range_error.exception.headers["Content-Range"], "bytes */16")

            video.write_bytes(b"replaced source")
            with self.assertRaises(urllib.error.HTTPError) as changed_error:
                urllib.request.urlopen(f"{base}/api/video/media-playback", timeout=3)
            self.assertEqual(changed_error.exception.code, 409)

            outside = self.root.parent / f"{self.root.name}-outside.mp4"
            outside.write_bytes(b"private")
            try:
                outside_metadata = VideoMetadata("media-outside", outside, "outside-hash",
                                                 1000, 1, 1.0, 16, 16)
                store.register_media(outside_metadata, "outside fixture")
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(f"{base}/api/video/media-outside", timeout=3)
                self.assertEqual(error.exception.code, 403)
            finally:
                outside.unlink(missing_ok=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            store.close()

    def test_video_playback_refuses_non_loopback_binding(self):
        store = EventStore(self.root / "index.sqlite3")
        try:
            with self.assertRaisesRegex(ValueError, "loopback interface"):
                serve(store, self.root, video_root=self.root, host="0.0.0.0", port=0)
        finally:
            store.close()

    def test_workbench_status_timeline_runtime_and_evidence_only_answer(self):
        store = EventStore(self.root / "index.sqlite3")
        self._event(store, "evt-door", "A person opens the front door", 2400,
                    inference_metrics={"prefill_us": 3400,
                                       "decode_tokens_per_second": 17.5})
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(store, self.root, video_root=self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base}/api/status", timeout=3) as response:
                status = json.loads(response.read())
            self.assertEqual(status["counts"]["media_count"], 1)
            self.assertEqual(status["counts"]["completed_event_count"], 1)

            with urllib.request.urlopen(
                    f"{base}/api/timeline?media_id=media-1", timeout=3) as response:
                timeline = json.loads(response.read())["events"]
            self.assertEqual(timeline[0]["timestamp_ms"], 2400)
            self.assertNotIn("frame_path", timeline[0])

            with urllib.request.urlopen(f"{base}/api/runtime", timeout=3) as response:
                runtime = json.loads(response.read())
            self.assertEqual(runtime["latest_inference"]["prefill_us"], 3400)
            self.assertEqual(runtime["latest_inference"]["decode_tokens_per_second"], 17.5)

            answer_request = urllib.request.Request(
                f"{base}/api/answer",
                data=json.dumps({"query": "door", "media_id": "media-1"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(answer_request, timeout=3) as response:
                answer = json.loads(response.read())
            self.assertEqual(answer["status"], "not_configured")
            self.assertEqual(answer["evidence"][0]["event_id"], "evt-door")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            store.close()

    def test_ingest_manager_requeues_interrupted_job_from_local_video(self):
        store = EventStore(self.root / "index.sqlite3")
        video_root = self.root / "uploads"
        video_root.mkdir()
        source = video_root / "saved.mp4"
        source.write_bytes(b"local-source")
        store.create_ingest_job("job-restored", "saved.mp4", source)
        store.update_ingest_job("job-restored", status="running", stage="captioning")

        class ImmediatePipeline:
            def ingest_video(self, *args, progress_callback=None, **kwargs):
                if progress_callback:
                    progress_callback("completed", 1, 1)
                return SimpleNamespace(media_id="restored-media", failed_events=0,
                                       indexed_events=1, sampled_frames=1)

        manager = IngestJobManager(store, ImmediatePipeline(), self.root / "artifacts",
                                   video_root, workers=1, pending_limit=0)
        try:
            for _ in range(100):
                job = store.ingest_job("job-restored")
                if job["status"] == "completed":
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["media_id"], "restored-media")
        finally:
            manager.close()
            store.close()

    def test_local_video_upload_streams_to_allowlisted_root(self):
        store = EventStore(self.root / "index.sqlite3")

        class AcceptingJobs:
            def __init__(self):
                self.source = None

            def submit(self, source_path, title):
                self.source = Path(source_path)
                return {"job_id": "upload-job", "title": title}

        jobs = AcceptingJobs()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(
                store, self.root, video_root=self.root, job_manager=jobs))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/videos",
                data=b"video-bytes",
                headers={"Content-Type": "video/mp4", "X-File-Name": "sample.mp4"},
                method="POST")
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 202)
                payload = json.loads(response.read())
            self.assertEqual(payload["job"]["job_id"], "upload-job")
            self.assertEqual(jobs.source.read_bytes(), b"video-bytes")
            self.assertEqual(jobs.source.parent, self.root)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            store.close()


if __name__ == "__main__":
    unittest.main()
