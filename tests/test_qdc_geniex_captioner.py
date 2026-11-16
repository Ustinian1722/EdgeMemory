from __future__ import annotations

import json
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory

from edgememory.qdc_geniex import (
    QdcGenieXVlmCaptioner,
    QdcGenieXVlmProfile,
    load_qdc_geniex_profile,
    parse_service_ready,
    parse_service_result,
)


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / ".local" / "test-tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def valid_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": 2222,
        "username": "root",
        "device_id": "runtime-fixture",
        "model_id": "fixture-vlm",
        "remote_service_binary": "/tmp/hexagonflow/service",
        "remote_runtime_root": "/tmp/hexagonflow/runtime",
        "remote_model_path": "/tmp/hexagonflow/model.gguf",
        "remote_mmproj_path": "/tmp/hexagonflow/mmproj.gguf",
    }


class QdcGenieXCaptionerTests(unittest.TestCase):
    def test_profile_requires_observed_device_and_rejects_secret_fields(self):
        profile = QdcGenieXVlmProfile.from_mapping(valid_profile())
        self.assertEqual(profile.port, 2222)
        self.assertEqual(profile.device_id, "runtime-fixture")
        for key, secret in (("password", "do-not-print"), ("api_key", "private-token"),
                            ("session_id", "internal-session"), ("tunnel_command", "ssh secret")):
            with self.subTest(key=key):
                invalid = valid_profile()
                invalid[key] = secret
                with self.assertRaises(ValueError) as raised:
                    QdcGenieXVlmProfile.from_mapping(invalid)
                self.assertNotIn(secret, str(raised.exception))

    def test_profile_requires_absolute_remote_paths_and_tmp_work_root(self):
        invalid = valid_profile()
        invalid["remote_model_path"] = "model.gguf"
        with self.assertRaisesRegex(ValueError, "absolute"):
            QdcGenieXVlmProfile.from_mapping(invalid)
        invalid = valid_profile()
        invalid["remote_work_root"] = "/home/root"
        with self.assertRaisesRegex(ValueError, "must be /tmp"):
            QdcGenieXVlmProfile.from_mapping(invalid)

    def test_profile_file_must_live_under_local_ignored_storage(self):
        with TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory) / "qdc-profile.json"
            path.write_text(json.dumps(valid_profile()), encoding="utf-8")
            self.assertEqual(load_qdc_geniex_profile(path).model_id, "fixture-vlm")
        with self.assertRaisesRegex(ValueError, "under the ignored .local"):
            load_qdc_geniex_profile(ROOT / "docs" / "ARCHITECTURE.md")

    def test_service_protocol_parses_runtime_and_inference_metrics(self):
        ready = parse_service_ready("READY\t1\t666978747572652d766c6d\t72756e74696d652d66697874757265\t72756e74696d65\t706c7567696e\t12345\n")
        self.assertEqual(ready["model_id"], "fixture-vlm")
        self.assertEqual(ready["device_id"], "runtime-fixture")
        self.assertEqual(ready["estimated_resident_bytes"], 12345)

        response = "\t".join((
            "RES", "request-1", "1", "0", "23", "12", "1914000", "1160000",
            "660000", "489000", "731000", "660.2", "31.4",
            "A red block".encode().hex(), "636f6d706c65746564",
        ))
        text, metrics = parse_service_result(response, "request-1")
        self.assertEqual(text, "A red block")
        self.assertEqual(metrics["output_tokens"], 23)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 31.4)

    def test_service_protocol_rejects_mismatched_ids_and_nonfinite_metrics(self):
        with self.assertRaisesRegex(RuntimeError, "mismatched"):
            parse_service_result("RES\twrong\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\t", "right")
        bad = "\t".join(("RES", "id", "1", "0", "1", "0", "0", "0", "0", "0", "0",
                          "nan", "0", "61", "636f6d706c65746564"))
        with self.assertRaisesRegex(RuntimeError, "timing"):
            parse_service_result(bad, "id")

    def test_ssh_service_stream_wrappers_live_until_worker_shutdown(self):
        class Channel:
            def __init__(self):
                self.output = bytearray(
                    b"READY\t1\t666978747572652d766c6d\t72756e74696d652d66697874757265\t"
                    b"72756e74696d65\t706c7567696e\t42\n")
                self.input_closed = False

            def settimeout(self, _timeout):
                pass

            def recv_stderr_ready(self):
                return False

            def recv_ready(self):
                return bool(self.output)

            def recv(self, size):
                block = bytes(self.output[:size])
                del self.output[:size]
                return block

            def exit_status_ready(self):
                return self.input_closed

            def sendall(self, data):
                if data == b"STOP\n":
                    self.output.extend(b"BYE\n")

        class Stream:
            def __init__(self, channel=None):
                self.channel = channel

            def __del__(self):
                if self.channel is not None:
                    self.channel.input_closed = True

        class Transport:
            def is_active(self):
                return True

            def set_keepalive(self, _seconds):
                pass

        class Sftp:
            def mkdir(self, *_args, **_kwargs):
                pass

            def rmdir(self, _path):
                pass

            def close(self):
                pass

        class Client:
            def __init__(self):
                self.channel = Channel()
                self.stdin_ref = None

            def set_missing_host_key_policy(self, _policy):
                pass

            def connect(self, **_kwargs):
                pass

            def get_transport(self):
                return Transport()

            def open_sftp(self):
                return Sftp()

            def exec_command(self, *_args, **_kwargs):
                stdin = Stream()
                self.stdin_ref = weakref.ref(stdin)
                return stdin, Stream(self.channel), Stream()

            def close(self):
                pass

        client = Client()
        profile = QdcGenieXVlmProfile.from_mapping(valid_profile())
        with TemporaryDirectory(dir=TEMP_ROOT) as directory:
            captioner = QdcGenieXVlmCaptioner(
                profile, password_provider=lambda: "fixture-password",
                log_dir=Path(directory), ssh_client_factory=lambda: client)
            try:
                self.assertEqual(captioner.backend_id, "geniex-c-sdk-runtime-fixture")
                self.assertEqual(captioner.run_configuration["max_tokens"], 96)
                captioner._ensure_connected()
                self.assertIs(client.stdin_ref(), captioner._stdin)
                self.assertFalse(client.channel.input_closed)
            finally:
                captioner.close()
            self.assertTrue(captioner._closed)


if __name__ == "__main__":
    unittest.main()
