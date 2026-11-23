"""Execution boundary checks without running scientific computations."""

import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))
from gray_whale_corridor.release.runtime import AccessGuard, ALLOWED, execute
from gray_whale_corridor.release.paths import load_policy
from gray_whale_corridor.release.table_adapters import decimal_half_up


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(data_root=SOFTWARE.parent / "dataset",
            run_root=SOFTWARE.parent / "runs/unit_test",
            task_root=SOFTWARE.parent / "runs/unit_test/Fig5")
        self.guard = AccessGuard(self.context)

    def test_only_35_explicit_adapters(self):
        policy = load_policy()
        self.assertEqual(set(ALLOWED), {k for k, v in policy["tasks"].items() if not v["hold_reasons"]})
        self.assertEqual(len(ALLOWED), 35)

    def test_unregistered_raw_processing_rejected_before_io(self):
        with self.assertRaises(PermissionError):
            execute("provider_raw_to_station", self.context.data_root, "not_created")

    def test_write_is_confined(self):
        self.guard("open", (str(self.context.task_root / "results.csv"), "w", os.O_WRONLY))
        for path in (SOFTWARE / "v8.py", self.context.data_root / "README.md", Path("/tmp/escape")):
            with self.assertRaises(PermissionError):
                self.guard("open", (str(path), "w", os.O_WRONLY))

    def test_original_read_and_network_rejected(self):
        if self.guard.project:
            with self.assertRaises(PermissionError):
                self.guard("open", (str(self.guard.project / "README.md"), "r", os.O_RDONLY))
        with self.assertRaises(PermissionError):
            self.guard("socket.connect", (None, ("example.org", 443)))

    def test_rename_source_and_target_confined(self):
        with self.assertRaises(PermissionError):
            self.guard("os.rename", (str(self.context.data_root / "README.md"), str(self.context.task_root / "x")))

    def test_environment_is_read_only(self):
        self.guard.environment = SOFTWARE.parent / "environments/test_venv"
        path = str(self.guard.environment / "lib/package.py")
        self.guard("open", (path, "r", os.O_RDONLY))
        with self.assertRaises(PermissionError):
            self.guard("open", (path, "w", os.O_WRONLY))

    def test_s4_display_rounding_preserves_source(self):
        source = "0.6625"
        self.assertEqual(decimal_half_up(source, 3), "0.663")
        self.assertEqual(source, "0.6625")
        self.assertEqual(decimal_half_up("0.6624", 3), "0.662")


if __name__ == "__main__":
    unittest.main()
