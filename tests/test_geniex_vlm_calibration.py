import json
import tempfile
import unittest
from pathlib import Path

from tools import run_manifest
from tools.analyze_geniex_vlm_calibration import ROOT, _output_path, analyze

TEMP_ROOT = ROOT / ".local" / "test-tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class GenieXVlmCalibrationTests(unittest.TestCase):
    def test_simulation_fixture_cannot_become_device_profile(self):
        with tempfile.TemporaryDirectory(prefix="calibration-fixture-", dir=TEMP_ROOT) as temp:
            run = Path(temp) / "fixture-run"
            run.mkdir()
            (run / "calibration-run.json").write_text(
                json.dumps({"run_id": run.name, "warmup_completed": True,
                            "simulation_only": True}), encoding="utf-8")
            (run / "ingest.json").write_text(
                json.dumps({"simulation_only": True}), encoding="utf-8")
            run_manifest.create(run)

            with self.assertRaisesRegex(ValueError, "simulation data"):
                analyze([run, run])

    def test_profile_output_must_be_private_and_separate_from_input_runs(self):
        run = ROOT / "runs" / "example-run"
        with self.assertRaisesRegex(ValueError, "runs/ or .local/"):
            _output_path(ROOT / "docs" / "profile.json", [run])
        with self.assertRaisesRegex(ValueError, "must not be inside"):
            _output_path(run / "profile.json", [run])


if __name__ == "__main__":
    unittest.main()
