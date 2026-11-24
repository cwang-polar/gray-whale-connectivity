"""Metadata, binding and syntax tests only; never calculate spatial metrics."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))
from gray_whale_corridor.release import control, paths, spatial_v8

CONTRACTS = json.loads((SOFTWARE / "tests/fixtures/scientific_contracts.json").read_text())
LEGACY = "scripts/run_phase8t_primary_robustness_smoke_test.py"
ADAPTER = "src/gray_whale_corridor/release/spatial_v8.py"


def function(relative, name):
    tree = ast.parse((SOFTWARE / relative).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


class SpatialEntryTests(unittest.TestCase):
    def test_no_scientific_imports(self):
        self.assertFalse(any(name.split('.')[0] in {"numpy", "pandas", "scipy", "matplotlib", "shapely", "geopandas"}
                             for name in sys.modules))

    def test_scientific_functions_match_packaged_contracts(self):
        for name, expected in CONTRACTS["spatial_functions"].items():
            value = ast.dump(function(LEGACY, name), include_attributes=False)
            self.assertEqual(hashlib.sha256(value.encode()).hexdigest(), expected, name)

    def test_observation_filter_statements_identical(self):
        legacy = function(LEGACY, "p4_year_leverage")
        adapter = function(ADAPTER, "load_observations")
        dump = lambda nodes: [ast.dump(n, include_attributes=False) for n in nodes]
        self.assertEqual(dump(legacy.body[1:7]), dump(adapter.body[3:9]))

    def test_same_period_definitions(self):
        original = next(n.value for n in function(LEGACY, "p4_year_leverage").body
                        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "definitions")
        replacement = next(n.value for n in function(ADAPTER, "definitions").body if isinstance(n, ast.Return))

        class StripLegacy(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "legacy":
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return self.generic_visit(node)

        self.assertEqual(ast.dump(original), ast.dump(StripLegacy().visit(replacement)))

    def test_task_inputs_are_28_existing_grids_and_spatial_sources(self):
        policy = paths.load_policy()
        expected = {(s, y) for s in ("prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel")
                    for y in range(2011 if s in ("prey_only", "prey_ice") else 2015, 2020)}
        for name in ("spatial_matching", "sensitivity_survey_year"):
            task = policy["tasks"][name]
            self.assertFalse(task["hold_reasons"])
            self.assertEqual(task["adapter_status"], "input_paths_configured")
            self.assertEqual({a for a in task["aliases"] if a.startswith("grid_")},
                             {f"grid_{s}_{y}" for s, y in expected})
            self.assertEqual(len(task["inputs"]), 34)
            self.assertFalse(any("panel_C_annual" in p or "phase8u" in p.lower() or "prey_surface" in p
                                 for p in task["inputs"]))
            self.assertTrue(all(not policy["files"][p].get("hold") for p in task["inputs"]))

    def test_only_selected_functions_dispatched(self):
        before = set((SOFTWARE.parent / "runs").glob("*/*"))
        tree = ast.parse((SOFTWARE / ADAPTER).read_text())
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertTrue({"legacy.scenario_record_metrics", "legacy.summarize_records", "legacy.p4_year_leverage"} <= calls)
        for name in ("legacy.main", "legacy.ensure_dirs", "legacy.run_lcp", "legacy.load_topology", "legacy.finalize_integrity"):
            self.assertNotIn(name, calls)
        self.assertEqual(before, set((SOFTWARE.parent / "runs").glob("*/*")))

    def test_bindings_restore_after_exception(self):
        sentinel_loader, sentinel_status = object(), object()
        legacy = SimpleNamespace(YEARS=list(range(2011, 2020)), VESSEL_YEARS=list(range(2015, 2020)),
            PREY_VARIABLE="total_kJm2_frozen", FREQUENT_THRESHOLD=0.05,
            ASAMM_SIGHTINGS="old_sightings", ASAMM_EFFORT="old_effort", VALIDATION_OUT="old_output",
            load_primary_grid=sentinel_loader, set_stage=sentinel_status)
        with tempfile.TemporaryDirectory(prefix=".spatial-metadata-test-", dir=SOFTWARE.parent) as temporary:
            root = Path(temporary)
            (root / "qa").mkdir()
            context = SimpleNamespace(require_reserved=lambda: None, task="sensitivity_survey_year",
                input=lambda alias: root / alias, output=lambda name: root / name)
            with self.assertRaisesRegex(RuntimeError, "test restore"):
                with spatial_v8.spatial_bindings(context, legacy):
                    self.assertEqual(legacy.ASAMM_SIGHTINGS, root / "sightings")
                    self.assertEqual(legacy.VALIDATION_OUT, root / "source_data")
                    legacy.set_stage("p4", "metadata-test")
                    self.assertEqual(json.loads((root / "qa/spatial_stage.json").read_text())["state"], "metadata-test")
                    with self.assertRaises(ValueError):
                        legacy.set_stage("p1", "not-permitted")
                    raise RuntimeError("test restore")
            self.assertIs(legacy.load_primary_grid, sentinel_loader)
            self.assertIs(legacy.set_stage, sentinel_status)
            self.assertEqual(legacy.ASAMM_SIGHTINGS, "old_sightings")
            self.assertEqual(legacy.VALIDATION_OUT, "old_output")
            self.assertFalse(any(root.rglob("*.csv")))


if __name__ == "__main__":
    unittest.main()
