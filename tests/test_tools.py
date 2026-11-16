import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tools import hexagonflow
from tools import qdc_session
from tools import session_budget
from tools import analyze_logits
from tools import edgememory_embedding_smoke
from tools import analyze_mixed_runs
from tools import analyze_dual_runs
from tools import analyze_device_split
from tools import export_interference_observations
from tools import performance_metrics
from tools import paired_statistics
from tools import prepare_geniex_bundle
from tools import run_manifest
from tools import session_close_checklist

TEST_TEMP_ROOT = hexagonflow.ROOT / ".local" / "test-tmp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class EmbeddingSmokePathTests(unittest.TestCase):
    def test_relative_report_path_resolves_under_project_root(self):
        path = edgememory_embedding_smoke._project_output_path(
            Path("runs/embedding-smoke-fixture/summary.json")
        )
        self.assertEqual(
            path,
            (edgememory_embedding_smoke.PROJECT_ROOT
             / "runs/embedding-smoke-fixture/summary.json").resolve(),
        )

    def test_report_path_outside_project_root_is_rejected(self):
        outside = edgememory_embedding_smoke.PROJECT_ROOT.parent / "outside-smoke.json"
        with self.assertRaisesRegex(ValueError, "D-drive project workspace"):
            edgememory_embedding_smoke._project_output_path(outside)


class BundleToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (hexagonflow.ROOT / "build" / "test-artifacts").mkdir(
            parents=True, exist_ok=True
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=hexagonflow.ROOT / "build" / "test-artifacts"
        )
        self.root = Path(self.temp.name)
        self.bundle = self.root / "probe.tar.gz"
        hexagonflow.build_bundle(self.bundle)

    def tearDown(self):
        self.temp.cleanup()

    def test_geniex_local_bundle_excludes_qairt_and_is_deterministic(self):
        source_archive = self.root / "geniex-source.tar.gz"
        with tarfile.open(source_archive, "w:gz") as archive:
            for name, payload in (
                ("fake-prefix/bin/geniex-bench", b"cli"),
                ("fake-prefix/lib/libgeniex.so", b"geniex"),
                ("fake-prefix/lib/llama_cpp/libggml-hexagon.so", b"hexagon"),
                ("fake-prefix/lib/qairt/libQnnHtp.so", b"restricted-runtime-fixture"),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o755
                archive.addfile(member, io.BytesIO(payload))
        first = self.root / "local-only-one.tar.gz"
        manifest = self.root / "geniex-manifest.json"
        result = prepare_geniex_bundle.build(
            source_archive, "fake-prefix/", first, manifest
        )
        second = self.root / "local-only-two.tar.gz"
        prepare_geniex_bundle.build(
            source_archive, "fake-prefix/", second, self.root / "second.json"
        )
        with tarfile.open(first, "r:gz") as archive:
            names = archive.getnames()
        self.assertEqual(len(names), 3)
        self.assertFalse(any("qairt" in name.lower() or "qnn" in name.lower()
                             for name in names))
        self.assertFalse(result["qairt_assets_included"])
        self.assertEqual(result["bundle_sha256"], prepare_geniex_bundle.sha256(second))

    def test_geniex_bundle_rejects_paths_outside_workspace(self):
        outside = self.root.parents[3] / "outside-geniex.tar.gz"
        with self.assertRaisesRegex(ValueError, "D: project workspace"):
            prepare_geniex_bundle.workspace_path(outside)

    def test_mixed_run_embedding_parser_skips_diagnostics_and_similarity_matrix(self):
        embedding = {
            "object": "list",
            "data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}],
        }
        matrix = {"object": "matrix", "data": [[1.0, 0.0], [0.0, 1.0]]}
        output = "warning: backend diagnostic\n" + json.dumps(embedding) + "\n" + json.dumps(matrix)
        parsed = analyze_mixed_runs.embedding_batches(output)
        self.assertEqual(parsed, [[[1.0, 0.0], [0.0, 1.0]]])

    def test_mixed_run_embedding_parser_rejects_nonfinite_values(self):
        output = '{"object":"list","data":[{"embedding":[NaN]}]}'
        with self.assertRaisesRegex(ValueError, "non-finite"):
            analyze_mixed_runs.embedding_batches(output)

    def test_mixed_run_analysis_summarizes_process_rss_separately(self):
        run = self.root / "mixed-rss-fixture"
        run.mkdir()
        config = {
            "run_id": run.name, "block": 1, "condition": "concurrent",
            "embedding_threads": 2, "device_id": "runtime-test",
            "geniex_version": "v-test", "llama_cpp_version": "abc123",
            "llm_model_sha256": "a" * 64, "embedding_model_sha256": "b" * 64,
            "prompt_tokens": 8, "generated_tokens": 4,
            "benchmark_repetitions": 1, "warmup": 0, "seed": 42,
            "llm_threads": 2, "embedding_items_per_request": 1,
            "embedding_repeats": 1,
        }
        result = {
            "llm_status": "process-completed",
            "embedding_status": "process-completed",
            "llm_wrapper_elapsed_ms": 20,
            "embedding_wrapper_elapsed_ms": 10,
            "wrapper_interval_overlap_ms": 8,
        }
        benchmark = {
            "runs": [{"status": "ok"}],
            "agg": {
                "ttft_ms": {"median": 4.0},
                "prefill_tps": {"median": 5.0},
                "decode_tps": {"median": 6.0},
            },
        }
        embedding = {"object": "list", "data": [{"embedding": [1.0, 0.0]}]}
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (run / "llm-benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
        (run / "embedding-process.log").write_text(json.dumps(embedding), encoding="utf-8")
        (run / "metrics.jsonl").write_text("\n".join((
            json.dumps({"run_id": run.name, "task_id": "llm-request",
                        "peak_process_rss_bytes": 4096,
                        "peak_process_rss_scope": "direct_child"}),
            json.dumps({"run_id": run.name, "task_id": "embedding-request",
                        "peak_process_rss_bytes": 2048,
                        "peak_process_rss_scope": "direct_child"}),
        )) + "\n", encoding="utf-8")
        run_manifest.create(run)

        report = analyze_mixed_runs.analyze([run])
        group = report["groups"]["concurrent/embedding_threads=2"]
        self.assertEqual(group["llm_peak_process_rss_bytes"]["p50"], 4096.0)
        self.assertEqual(group["embedding_peak_process_rss_bytes"]["p99"], 2048.0)
        self.assertEqual(report["process_rss_scope"],
                         "direct child per process runner; null when unavailable; not accelerator memory")

    def test_performance_summary_reports_type7_tail_percentiles(self):
        summary = performance_metrics.distribution_summary([1, 2, 3, 4])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertAlmostEqual(summary["p99"], 3.97)
        self.assertEqual(performance_metrics.distribution_summary([])["p95"], None)

    def test_paired_bootstrap_is_repeatable_and_suppresses_small_samples(self):
        small = paired_statistics.bootstrap_median_interval([1.0, 2.0, 3.0], seed=7)
        self.assertEqual(small["status"], "insufficient_blocks")
        self.assertIsNone(small["lower"])
        sample = [float(value) for value in range(10)]
        first = paired_statistics.bootstrap_median_interval(sample, seed=7, resamples=500)
        second = paired_statistics.bootstrap_median_interval(sample, seed=7, resamples=500)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "estimated")
        self.assertLessEqual(first["lower"], first["estimate"])
        self.assertGreaterEqual(first["upper"], first["estimate"])

    def test_dual_run_analysis_requires_matched_engine_metadata_and_summarizes_pairing(self):
        root = self.root / "dual-fixtures"
        root.mkdir()

        def make_run(condition, device_a_tps, device_b_tps, block=1):
            run_id = f"fixture-{condition}-{block}"
            run = root / run_id
            run.mkdir()
            config = {
                "run_id": run_id, "condition": condition, "block": block,
                "geniex_version": "v-test", "llama_cpp_version": "abc123",
                "model_sha256": "a" * 64, "devices": ["runtime-a", "runtime-b"],
                "prompt_tokens": 8, "generated_tokens": 4, "repetitions": 2,
                "warmup": 1, "seed": 42, "threads": 2,
                "runner_sha256": "runner-hash", "exit_code": 0,
            }
            result = {
                "run_id": run_id, "condition": condition, "simulated": False,
                "request_a_status": "process-completed", "request_b_status": "process-completed",
                "request_a_wrapper_elapsed_ms": 120 if condition == "isolated" else 70,
                "request_b_wrapper_elapsed_ms": 130 if condition == "isolated" else 75,
                "wrapper_interval_overlap_ms": 0 if condition == "isolated" else 65,
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
            rss_scale = 1 if condition == "isolated" else 2
            metric_rows = [
                {"run_id": run_id, "task_id": "llm-request-a",
                 "peak_process_rss_bytes": 1000 * rss_scale,
                 "peak_process_rss_scope": "direct_child"},
                {"run_id": run_id, "task_id": "llm-request-b",
                 "peak_process_rss_bytes": 2000 * rss_scale,
                 "peak_process_rss_scope": "direct_child"},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metric_rows),
                encoding="utf-8",
            )
            for tag, device, tps in (("a", "runtime-a", device_a_tps), ("b", "runtime-b", device_b_tps)):
                engine = {
                    "plugin": "llama_cpp", "device": "npu", "device_id": device,
                    "model_size_bytes": 100, "geniex_version": "v-test",
                    "llama_cpp_version": "abc123",
                    "agg": {"ttft_ms": {"median": 12.0},
                            "prefill_tps": {"median": 900.0},
                            "decode_tps": {"median": tps}},
                }
                (run / f"llm-{tag}.json").write_text(json.dumps(engine), encoding="utf-8")
            run_manifest.create(run)
            return run

        isolated = make_run("isolated", 50.0, 60.0)
        concurrent = make_run("concurrent", 45.0, 54.0)
        result = analyze_dual_runs.analyze([isolated, concurrent], ("runtime-a", "runtime-b"))
        self.assertEqual(result["n_paired_blocks"], 1)
        self.assertEqual(
            result["paired_median_change_percent_ci95"]["runtime-a"]["decode_tps"]["status"],
            "insufficient_blocks",
        )
        rss = result["peak_process_rss_bytes"]["by_device_and_condition"]
        self.assertEqual(rss["runtime-a"]["isolated"]["p50"], 1000.0)
        self.assertEqual(rss["runtime-a"]["concurrent"]["p99"], 2000.0)
        self.assertEqual(rss["runtime-b"]["isolated"]["count"], 1)
        self.assertAlmostEqual(
            result["paired_device_metrics"]["runtime-a"]["decode_tps"]["concurrent_change_percent"],
            -10.0,
        )
        self.assertAlmostEqual(
            result["pair_wrapper_wall_ms"]["concurrent_pair_completion_speedup"],
            250 / 75,
        )

        profile = export_interference_observations.build_profile(
            [isolated, concurrent], "shared-domain", "llm_decode", "mock_fixture"
        )
        self.assertTrue(profile["simulation_only"])
        self.assertEqual(profile["observation_count"], 2)
        observation = profile["observations"][0]
        self.assertEqual(observation["source"], "mock_fixture")
        self.assertEqual(observation["target_model_id"], "a" * 64)
        self.assertEqual(observation["target_input_units"], 8)
        self.assertEqual(observation["target_output_units"], 4)
        self.assertFalse(observation["target_model_resident"])
        self.assertEqual(observation["co_running"][0]["backend_id"], "geniex-device-b")
        self.assertTrue(observation["co_running"][0]["model_resident"])
        self.assertEqual(observation["isolated_latency_us"], 80000)
        self.assertEqual(observation["concurrent_latency_us"], 88889)

        with tempfile.TemporaryDirectory(
            prefix="test-interference-export-", dir=TEST_TEMP_ROOT
        ) as temp_dir:
            output_dir = Path(temp_dir) / "profile"
            written = export_interference_observations.write_profile(
                profile, output_dir, [isolated, concurrent]
            )
            self.assertEqual(written["verified_manifest_files"], 2)
            self.assertEqual(run_manifest.verify(output_dir)["status"], "verified")
            exported = (output_dir / "interference-observations.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(exported), 3)
            self.assertEqual(exported[1].split("\t")[1], "mock_fixture")
            provenance = json.loads((output_dir / "provenance.json").read_text(
                encoding="utf-8"
            ))
            self.assertTrue(provenance["simulation_only"])

    def test_dual_run_analysis_rejects_unpaired_or_relabelled_devices(self):
        root = self.root / "bad-dual-fixtures"
        root.mkdir()

        def make_run(condition, devices, block=1):
            run_id = f"bad-{condition}-{block}"
            run = root / run_id
            run.mkdir()
            config = {
                "run_id": run_id, "condition": condition, "block": block,
                "geniex_version": "v-test", "llama_cpp_version": "abc123",
                "model_sha256": "model-hash", "devices": devices,
                "prompt_tokens": 8, "generated_tokens": 4, "repetitions": 2,
                "warmup": 1, "seed": 42, "threads": 2,
                "runner_sha256": "runner-hash", "exit_code": 0,
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "result.json").write_text(json.dumps({
                "run_id": run_id, "condition": condition, "simulated": False,
                "request_a_status": "process-completed", "request_b_status": "process-completed",
                "request_a_wrapper_elapsed_ms": 10, "request_b_wrapper_elapsed_ms": 10,
                "wrapper_interval_overlap_ms": 0,
            }), encoding="utf-8")
            for tag, device in (("a", "runtime-a"), ("b", "runtime-b")):
                engine = {
                    "plugin": "llama_cpp", "device": "npu", "device_id": device,
                    "model_size_bytes": 100, "geniex_version": "v-test",
                    "llama_cpp_version": "abc123",
                    "agg": {name: {"median": 1.0} for name in ("ttft_ms", "prefill_tps", "decode_tps")},
                }
                (run / f"llm-{tag}.json").write_text(json.dumps(engine), encoding="utf-8")
            run_manifest.create(run)
            return run

        isolated = make_run("isolated", ["runtime-a", "runtime-b"])
        reversed_run = make_run("concurrent", ["runtime-b", "runtime-a"])
        with self.assertRaisesRegex(ValueError, "device order"):
            analyze_dual_runs.analyze([isolated, reversed_run], ("runtime-a", "runtime-b"))

    def test_device_split_analysis_checks_complete_blocks_and_compares_medians(self):
        root = self.root / "split-fixtures"
        root.mkdir()
        runs = []
        for block in (1, 2):
            for device_id, decode_tps in (("runtime-a", 100.0), ("runtime-b", 95.0),
                                          ("runtime-a+runtime-b", 125.0)):
                run_id = f"split-{block}-{device_id.replace(',', '-') }"
                run = root / run_id
                run.mkdir()
                config = {
                    "run_id": run_id, "block": block, "device_id": device_id,
                    "mode": "split" if "+" in device_id else "single",
                    "geniex_version": "v-test", "llama_cpp_version": "abc123",
                    "model_sha256": "model-hash", "prompt_tokens": 8,
                    "generated_tokens": 4, "repetitions": 2, "warmup": 1,
                    "seed": 42, "threads": 2, "runner_sha256": "runner-hash",
                    "exit_code": 0,
                }
                result = {
                    "run_id": run_id, "simulated": False,
                    "request_status": "process-completed", "wrapper_elapsed_ms": 75.0,
                }
                engine = {
                    "plugin": "llama_cpp", "device": "npu", "device_id": device_id,
                    "geniex_version": "v-test", "llama_cpp_version": "abc123",
                    "agg": {"ttft_ms": {"median": 10.0},
                            "prefill_tps": {"median": 800.0},
                            "decode_tps": {"median": decode_tps}},
                }
                (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
                (run / "bench.json").write_text(json.dumps(engine), encoding="utf-8")
                run_manifest.create(run)
                runs.append(run)
        summary = analyze_device_split.analyze(
            runs, "runtime-a", "runtime-b", "runtime-a+runtime-b")
        self.assertEqual(summary["n_paired_blocks"], 2)
        self.assertAlmostEqual(
            summary["split_change_percent"]["decode_tps"]["versus_runtime-a"]["median_percent"],
            25.0,
        )
        self.assertEqual(
            summary["scheduler_calibration_profile"]["service_estimates_by_backend_us"],
            {"geniex-process-runtime-a": 75000, "geniex-process-runtime-b": 75000},
        )
        with self.assertRaisesRegex(ValueError, "both single-device IDs"):
            analyze_device_split.analyze(
                runs[:-1], "runtime-a", "runtime-b", "runtime-a+runtime-b")

    def test_bundle_manifest_and_file_hashes_verify(self):
        verified = hexagonflow.read_and_verify_bundle(self.bundle)
        self.assertEqual(len(verified.payloads), len(verified.manifest["entries"]))
        self.assertFalse(verified.manifest["hardware_measurements"])

    def test_restore_requires_empty_directory_and_restores_scripts(self):
        target = self.root / "restore"
        result = hexagonflow.restore_bundle(self.bundle, target)
        expected_files = len(
            hexagonflow.read_and_verify_bundle(self.bundle).manifest["entries"]
        )
        self.assertEqual(result["restored_files"], expected_files)
        self.assertTrue((target / "scripts/remote/device_probe.sh").is_file())
        (target / "keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(hexagonflow.BundleError):
            hexagonflow.restore_bundle(self.bundle, target)
        self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_checksum_corruption_is_rejected(self):
        broken = self.root / "broken.tar.gz"
        with tarfile.open(self.bundle, "r:gz") as source:
            members = [(member, source.extractfile(member).read())
                       for member in source.getmembers()]
        with tarfile.open(broken, "w:gz") as target:
            for member, data in members:
                if member.name not in ("manifest.json", "manifest.sha256"):
                    data = bytes([data[0] ^ 1]) + data[1:]
                info = tarfile.TarInfo(member.name)
                info.size = len(data)
                info.mode = member.mode
                target.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(hexagonflow.BundleError, "checksum mismatch"):
            hexagonflow.read_and_verify_bundle(broken)

    def test_path_traversal_member_is_rejected(self):
        unsafe = self.root / "unsafe.tar.gz"
        with tarfile.open(unsafe, "w:gz") as archive:
            payload = b"bad"
            member = tarfile.TarInfo("../outside")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        with self.assertRaisesRegex(hexagonflow.BundleError, "unsafe bundle path"):
            hexagonflow.read_and_verify_bundle(unsafe)

    def test_connect_classifier_never_echoes_connection_values(self):
        sample = (
            "ssh -i C:\\secure\\qdc-device.pem "
            "-L 15555:session-placeholder:22 "
            "-L 15556:session-placeholder:5037 "
            "-N sshtunnel@ssh.qdc.qualcomm.com"
        )
        result = hexagonflow.inspect_connect_text(sample)
        serialized = json.dumps(result)
        self.assertEqual(
            result["transport"], "ssh_tunnel_with_ssh_and_adb_forwards"
        )
        self.assertTrue(result["contains_private_key_argument"])
        self.assertNotIn("session-placeholder", serialized)
        self.assertNotIn("qdc-device.pem", serialized)
        self.assertNotIn("15555", serialized)

    def test_adb_connect_classifier_recognizes_qdc_custom_port_without_echoing(self):
        result = hexagonflow.inspect_connect_text("adb -P 15557 devices")
        self.assertEqual(result["transport"], "adb_custom_port")
        self.assertNotIn("15557", json.dumps(result))

    def test_artifact_manifest_hashes_explicit_files_and_rejects_keys(self):
        artifact = self.root / "probe.bin"
        artifact.write_bytes(b"small fixture")
        spec = self.root / "artifacts.json"
        spec.write_text(json.dumps({
            "schema_version": 1,
            "source_commit": "test-commit",
            "artifacts": [{
                "id": "fixture",
                "artifact_type": "smoke-test",
                "path": artifact.relative_to(hexagonflow.ROOT).as_posix(),
                "version": "test-1",
                "license": "MIT",
            }],
        }), encoding="utf-8")
        output = self.root / "manifest.json"
        result = hexagonflow.build_artifact_manifest(spec, output)
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["artifacts"], 1)
        self.assertEqual(data["artifacts"][0]["sha256"],
                         hexagonflow.sha256_bytes(b"small fixture"))
        spec.write_text(json.dumps({
            "schema_version": 1,
            "artifacts": [{"id": "key", "path": "secrets/private-key.pem"}],
        }), encoding="utf-8")
        with self.assertRaisesRegex(hexagonflow.BundleError, "private-key"):
            hexagonflow.build_artifact_manifest(spec, output)

    def test_outputs_outside_workspace_are_rejected(self):
        outside = self.root.parents[3] / "outside-probe.tar.gz"
        with self.assertRaisesRegex(hexagonflow.BundleError, "D: workspace"):
            hexagonflow.build_bundle(outside)

    def test_qdc_session_cli_help_runs_with_documented_script_invocation(self):
        script = qdc_session.ROOT / "tools" / "qdc_session.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=qdc_session.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("connection-probe", result.stdout)

    def test_qdc_profile_is_local_only_and_transport_args_are_not_logged(self):
        profile_path = self.root / "profile.json"
        identity_file = self.root / "mock-identity.pem"
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "transport": "ssh",
            "ssh": {"host": "localhost", "port": 18000, "username": "device-user",
                    "identity_file": str(identity_file)},
            "remote_root": "/tmp/hexagonflow-deploy",
        }), encoding="utf-8")
        profile = qdc_session.load_profile(profile_path)
        self.assertEqual(profile.transport, "ssh")
        self.assertEqual(profile.identity_file, str(identity_file))
        with patch("tools.qdc_session.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = b"ok\n"
            output = qdc_session.SshTransport(profile).execute(
                ["uname", "-m"], "test probe", 5, capture_stdout=True
            )
        argv = run.call_args.args[0]
        self.assertEqual(output, b"ok\n")
        self.assertIn("18000", argv)
        self.assertIn(str(identity_file), argv)
        self.assertNotIn("18000", output.decode("utf-8"))
        self.assertNotIn(str(identity_file), output.decode("utf-8"))

    def test_qdc_profile_rejects_secret_fields_recursively(self):
        profile_path = self.root / "secret-profile.json"
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "transport": "ssh",
            "ssh": {"host": "localhost", "port": 18000, "username": "device-user",
                    "password": "do-not-store"},
            "remote_root": "/tmp/hexagonflow-deploy",
        }), encoding="utf-8")
        with self.assertRaisesRegex(qdc_session.SessionError, "must not contain"):
            qdc_session.load_profile(profile_path)

    def test_session_budget_action_thresholds(self):
        self.assertEqual(session_budget.phase_for_seconds(2701), 'research_window')
        self.assertEqual(session_budget.phase_for_seconds(2700), 'finish_existing_experiments')
        self.assertEqual(session_budget.phase_for_seconds(1200), 'collect_and_verify')
        self.assertEqual(session_budget.phase_for_seconds(600), 'stop_remote_and_notify')

    def test_logit_comparison_rejects_mismatched_conditions(self):
        cpu = {'device': 'cpu', 'n_prompt': 2, 'vocab_size': 5, 'top_n': 2,
               'rows': [[[1, 2.0], [2, 1.0]]]}
        npu = {'device': 'npu', 'n_prompt': 2, 'vocab_size': 5, 'top_n': 2,
               'rows': [[[1, 1.9], [3, 0.8]]]}
        config = {'seed': 42, 'prompt_tokens': 2, 'model_sha256': 'fixture'}
        result = analyze_logits.compare(cpu, npu, config, config)
        self.assertTrue(result['top1_match'])
        self.assertEqual(result['shared_token_count'], 1)
        self.assertFalse(result['full_vocab_comparison'])
        with self.assertRaisesRegex(ValueError, 'conditions do not match'):
            analyze_logits.compare(cpu, npu, config, dict(config, seed=43))

    def test_run_manifest_detects_tampering_and_new_files(self):
        run = self.root / 'run-fixture'
        run.mkdir()
        (run / 'evidence.txt').write_text('original', encoding='utf-8')
        run_manifest.create(run)
        self.assertEqual(run_manifest.verify(run)['verified_files'], 1)
        (run / 'evidence.txt').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash or size mismatch'):
            run_manifest.verify(run)
        (run / 'evidence.txt').write_text('original', encoding='utf-8')
        (run / 'unlisted.txt').write_text('extra', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'file set changed'):
            run_manifest.verify(run)

    def test_run_manifest_cli_reports_created_and_verified_states(self):
        run = self.root / 'cli-manifest-fixture'
        run.mkdir()
        (run / 'evidence.txt').write_text('fixture', encoding='utf-8')

        with patch('sys.argv', ['run_manifest.py', 'create', str(run)]):
            with patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(run_manifest.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'created')

        with patch('sys.argv', ['run_manifest.py', 'verify', str(run)]):
            with patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(run_manifest.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'verified')

    def test_session_close_checklist_seals_runs_and_requires_remote_quiescence(self):
        runs_root = self.root / 'session-close-runs'
        runs_root.mkdir()
        primary = runs_root / 'session-fixture'
        primary.mkdir()
        (primary / 'result.json').write_text(json.dumps({
            'run_id': 'session-fixture', 'simulated': False,
            'request_status': 'process-completed',
        }), encoding='utf-8')
        failed = runs_root / 'captured-failure'
        failed.mkdir()
        (failed / 'config.json').write_text(json.dumps({'exit_code': 127}), encoding='utf-8')
        (failed / 'result.json').write_text(json.dumps({
            'run_id': 'captured-failure', 'simulated': False,
            'request_status': 'process-exit-127',
        }), encoding='utf-8')
        seal = session_close_checklist.seal_run_tree(runs_root)
        self.assertEqual(seal['verified_run_count'], 2)
        budget = {'conservative_remaining_seconds': 300, 'phase': 'finish_existing_experiments'}
        ready = session_close_checklist.create_checklist(
            runs_root, 'session-fixture',
            {'schema_version': 1, 'active_relevant_processes': 0}, budget,
        )
        self.assertTrue(ready['ready_for_user_to_close'])
        self.assertEqual(ready['failed_runs_with_collected_evidence'], ['captured-failure'])
        active = session_close_checklist.create_checklist(
            runs_root, 'session-fixture',
            {'schema_version': 1, 'active_relevant_processes': 1}, budget,
        )
        self.assertFalse(active['ready_for_user_to_close'])

    def test_session_checklist_requires_complete_hashed_probe_evidence(self):
        qdc_session.RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="checklist-", dir=qdc_session.RUNS) as temp:
            run_id = Path(temp).name
            run_root = qdc_session.run_directory(run_id)
            qdc_session.begin_session(run_id, 30)
            self.assertFalse(qdc_session.checklist(run_id)["ready_to_close"])

            raw = run_root / "raw"
            remote = raw / "remote-logs"
            remote.mkdir(parents=True, exist_ok=True)
            (raw / "connection-probe.txt").write_text("read-only\n", encoding="utf-8")
            (remote / "bootstrap-status.txt").write_text(
                "probe_status=complete\n", encoding="utf-8"
            )
            (remote / "device-probe.txt").write_text(
                "probe_mode=read_only\n", encoding="utf-8"
            )
            (run_root / "transfer-test.json").write_text(json.dumps({
                "bytes": 1024 * 1024,
                "sha256": "fixture-hash",
                "download_sha256": "fixture-hash",
                "integrity_passed": True,
            }), encoding="utf-8")
            bundle = hexagonflow.build_bundle(hexagonflow.ROOT / "build" / "first-probe.tar.gz")
            (run_root / "deploy-result.json").write_text(json.dumps({
                "bundle_path": "build/first-probe.tar.gz",
                "bundle_sha256": bundle["sha256"],
            }), encoding="utf-8")
            (run_root / "arm64-smoke-result.json").write_text(
                json.dumps({"status": "completed_or_skipped"}), encoding="utf-8"
            )
            files = []
            for file_path in sorted(path for path in raw.rglob("*") if path.is_file()):
                data = file_path.read_bytes()
                files.append({
                    "path": file_path.relative_to(raw).as_posix(),
                    "bytes": len(data),
                    "sha256": hexagonflow.sha256_bytes(data),
                })
            (run_root / "results-manifest.json").write_text(json.dumps({
                "run_id": run_id,
                "files": files,
            }), encoding="utf-8")
            (run_root / "collection-status.json").write_text(json.dumps({
                "complete": True,
            }), encoding="utf-8")
            self.assertTrue(qdc_session.checklist(run_id)["ready_to_close"])

            (remote / "device-probe.txt").write_text("corrupted\n", encoding="utf-8")
            self.assertFalse(qdc_session.checklist(run_id)["ready_to_close"])

    @unittest.skipUnless(shutil.which("sh"), "local shell needed for recovery rehearsal")
    def test_two_empty_restores_and_failure_evidence(self):
        result = hexagonflow._run_recovery_demo(self.root / "recovery")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["recovery_runs"]), 2)
        failure_result = Path(result["failure_evidence"]) / "result.json"
        self.assertTrue(failure_result.is_file())
        data = json.loads(failure_result.read_text(encoding="utf-8"))
        self.assertEqual(data["observed_exit_code"], 7)
        self.assertFalse(data["hardware_measurements"])

    @unittest.skipUnless(shutil.which("sh") and shutil.which("timeout"),
                         "local shell and timeout utility needed for runtime probe")
    def test_device_probe_limits_help_to_allowlisted_present_cli(self):
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        fake_tool = fake_bin / "llama-cli"
        fake_tool.write_text(
            "#!/bin/sh\nprintf 'llama-cli fixture help\\n'\n", encoding="utf-8"
        )
        fake_tool.chmod(0o755)
        log_root = self.root / "remote-logs"
        env = os.environ.copy()
        env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
        env["HEXAGONFLOW_REMOTE_LOG_ROOT"] = str(log_root)
        result = subprocess.run(
            ["sh", str(hexagonflow.ROOT / "scripts/remote/device_probe.sh")],
            cwd=hexagonflow.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("=== voice-ai-asr-assets ===", result.stdout)
        self.assertIn("runtime_help_status=llama-cli:completed", result.stdout)
        self.assertIn("llama-cli fixture help", (log_root / "runtime-help-llama-cli.log")
                        .read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
