"""File-routing tests only; no scientific module is imported or executed."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))

from gray_whale_corridor.release import control, paths
from gray_whale_corridor.release.figure_adapters import plotting_imports

DATA = SOFTWARE.parent / "dataset"


class ReleasePathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".path-test-", dir=SOFTWARE.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mock_root = patch.object(paths, "RUNS_ROOT", self.root / "runs")
        self.mock_root.start()
        self.addCleanup(self.mock_root.stop)

    def task(self, task="S8", run="metadata_test"):
        return paths.TaskPaths(task, DATA, run)

    def test_preview_all_tasks_creates_nothing(self):
        before = set(sys.modules)
        policy = paths.load_policy()
        prey = policy['tasks']['prey_processing']['aliases']
        external = {prey['station_totals'], prey['station_matches'],
                    policy['tasks']['S11']['aliases']['observations']}
        for name in control.load_catalog()["tasks"]:
            result = self.task(name).preview()
            self.assertFalse(result["directories_created"])
            self.assertFalse(result["execution_enabled"])
            for row in result["inputs"]:
                if row["relative_path"] in external and not (DATA / row["relative_path"]).exists():
                    self.assertEqual(row["status"], "blocked")
                    self.assertIn("Missing V8 input; no fallback", row["reason"])
                else:
                    self.assertEqual(row["status"], "present_and_hash_verified", row)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse(any(name.split('.')[0] in {"numpy", "pandas", "scipy", "matplotlib", "geopandas"}
                             for name in set(sys.modules) - before))

    def test_invalid_run_identifiers_rejected(self):
        for value in ["../x", "/tmp/x", "a/b", "a\\b", ".", "", "a b"]:
            with self.assertRaises(ValueError):
                self.task(run=value)

    def test_traversal_and_unregistered_inputs_rejected(self):
        context = self.task()
        for value in ["../old.csv", "/old.csv", "unregistered.csv"]:
            with self.assertRaises(ValueError):
                context.input_path(value)
        for value in ["../old.csv", "/old.csv"]:
            with self.assertRaises(ValueError):
                context.output(value)

    def test_input_hash_is_enforced_without_modifying_input(self):
        context = self.task()
        with patch.object(paths, "digest", return_value="wrong-hash"):
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                context.input("correlations")

    def test_input_root_requires_exact_v8_index(self):
        with self.assertRaises(ValueError):
            paths.TaskPaths("S8", SOFTWARE, "test")

    def test_output_reservation_is_exclusive(self):
        context = self.task()
        output = context.reserve_output()
        self.assertTrue((output / "path_manifest.json").is_file())
        with self.assertRaises(FileExistsError):
            self.task()
        with self.assertRaises(FileExistsError):
            context.reserve_output()
        self.assertFalse(any(output.rglob("*.csv")))

    def test_held_inputs_and_tasks_cannot_reserve_outputs(self):
        context = self.task('prey_processing')
        context.contract['hold_reasons'] = ['explicit technical hold for test']
        with self.assertRaises(PermissionError):
            context.reserve_output()
        context.contract['hold_reasons'] = []
        relative = context.contract['aliases']['station_totals']
        context.policy['files'][relative]['hold'] = True
        with self.assertRaises(PermissionError):
            context.reserve_output()
        self.assertFalse((self.root / "runs").exists())

    def test_symlink_escape_is_rejected(self):
        context = self.task()
        context.task_root.mkdir(parents=True)
        (context.task_root / "figures").symlink_to(SOFTWARE, target_is_directory=True)
        with self.assertRaises(ValueError):
            context.output("figures/README.md")

    def test_namespaced_provenance_round_trip(self):
        context = self.task()
        for item in [context.input("correlations"), context.package("v8.py"), context.output("qa/check.json")]:
            self.assertEqual(context.resolve_label(context.label(item)), item)

    def test_asamm_input_names_and_sort_order_preserved(self):
        context = self.task("annual_rates")
        config = json.loads((control.CONFIG / "annual_effort.json").read_text())
        for pattern in ["*/_Output__*.csv", "*/_SummaryInfo__*.csv"]:
            values = context.input_glob(config["raw_root"], pattern)
            self.assertEqual(len(values), 4)
            self.assertEqual(values, sorted((DATA / config["raw_root"]).glob(pattern)))
            self.assertTrue(all(str(value.relative_to(context.data_root)).startswith("data/raw/validation/") for value in values))

    def test_import_setup_requires_reserved_output(self):
        with self.assertRaises(RuntimeError):
            with plotting_imports(self.task()):
                self.fail("Unreserved import setup should not proceed")

    def test_cli_path_preview_has_no_execution_option(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(control.cli(["paths", "S8", "--run-id", "cli_check"]), 0)
        self.assertFalse(json.loads(stream.getvalue())["directories_created"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                control.cli(["paths", "S8", "--run-id", "cli_check", "--run"])
        self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
