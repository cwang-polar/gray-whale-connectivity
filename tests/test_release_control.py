"""Test only metadata/entry controls, never the scientific pipeline."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))

from gray_whale_corridor.release import control


class ReleaseControlTests(unittest.TestCase):
    def test_frozen_scope_and_figure_ids(self):
        catalog = control.load_catalog()
        self.assertTrue(catalog["execution_enabled"])
        self.assertNotIn("all", catalog["tasks"])
        for key in [*(f"Fig{i}" for i in range(1, 7)),
                    *(f"S{i}" for i in range(1, 14)),
                    *(f"TableS{i}" for i in range(1, 6))]:
            self.assertIn(key, catalog["tasks"])

    def test_reject_parent_and_absolute_paths(self):
        for relative in ["../original/data.csv", "/original/data.csv"]:
            with self.assertRaises(ValueError):
                control.confined_path(SOFTWARE, relative)

    def test_missing_data_never_falls_back_to_originals(self):
        report = control.inspect_task("Fig5", SOFTWARE / "intentionally_absent_dataset")
        self.assertFalse(report["all_listed_unheld_inputs_present"])
        self.assertFalse(report["ready_for_execution"])

    def test_source_pending_does_not_block_station_checkpoint_entry(self):
        report = control.inspect_task("prey_processing", SOFTWARE.parent / "dataset")
        self.assertFalse(report["has_held_inputs"])
        self.assertFalse(report["ready_for_execution"])
        from gray_whale_corridor.release.paths import load_policy
        aliases = load_policy()["tasks"]["prey_processing"]["aliases"]
        external = {aliases["station_totals"], aliases["station_matches"]}
        self.assertTrue(all(row["status"] == "present" for row in report["inputs"] if row["path"] not in external))
        self.assertTrue(all(row["status"] in ("present", "missing") for row in report["inputs"] if row["path"] in external))
        self.assertFalse(any('data/raw/prey/' in row['path'] for row in report['inputs']))

    def test_inspection_does_not_import_scientific_modules(self):
        before = set(sys.modules)
        control.inspect_task("Fig5", SOFTWARE.parent / "dataset", hashes=True)
        control.inspect_task("S8", SOFTWARE.parent / "dataset", hashes=True)
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".")[0] in
                             {"numpy", "pandas", "scipy", "matplotlib", "geopandas"}
                             for name in imported))

    def test_changed_configuration_is_rejected(self):
        with patch.object(control, "digest", return_value="not-the-frozen-digest"):
            with self.assertRaisesRegex(ValueError, "frozen scope changed"):
                control.load_catalog()

    def test_no_default_or_execution_command(self):
        for argv in [[], ["run", "Fig5"], ["plan", "all"],
                     ["plan", "Phase8U"], ["plan", "Fig5", "--alpha", "0.25"]]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    control.cli(argv)
            self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
