"""Static contracts only; no research modules, renderers or table builders run."""

import ast
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))
from gray_whale_corridor.release import control, paths, archived_figures, annual_v8, table_adapters

BACKUP = SOFTWARE.parent / "internal_provenance/path_adapter_before_v2/software"


def tree(relative, root=SOFTWARE):
    return ast.parse((root / relative).read_text())


def function(relative, name, root=SOFTWARE):
    return next(n for n in tree(relative, root).body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(nodes):
    return [ast.dump(n, include_attributes=False) for n in nodes]


class PreparedAdapterTests(unittest.TestCase):
    def test_imports_remain_non_scientific(self):
        self.assertFalse(any(name.split('.')[0] in {"numpy", "pandas", "geopandas", "scipy", "matplotlib", "shapely"}
                             for name in sys.modules))

    def test_all_ready_adapters_are_explicit_and_unexecuted(self):
        before = set((SOFTWARE.parent / "runs").glob("*/*"))
        policy = paths.load_policy()
        ready = {key for key, value in policy["tasks"].items() if value["adapter_status"] == "input_paths_configured"}
        self.assertEqual(len(ready), 23)
        self.assertEqual(set(archived_figures.MODULES), {"Fig1", "Fig2", "Fig3", "Fig4", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11", "S12", "S13"})
        self.assertEqual(set(table_adapters.BUILDERS), {"TableS1", "TableS2", "TableS3", "TableS4", "TableS5"})
        for name in ready:
            self.assertFalse(policy["tasks"][name]["hold_reasons"])
        self.assertTrue(control.load_catalog()["execution_enabled"])
        self.assertEqual(before, set((SOFTWARE.parent / "runs").glob("*/*")))

    def test_archived_prey_and_ice_entries_use_generated_results(self):
        policy = paths.load_policy()
        for task in ("Fig2", "Fig3"):
            contract = policy["tasks"][task]
            self.assertEqual(contract["adapter_status"], "archived_result_drawing_wired")
            self.assertFalse(contract["hold_reasons"])
            self.assertTrue(all(not policy["files"][p].get("hold") for p in contract["inputs"]))
            raw = [p for p in contract["inputs"] if p.startswith("data/raw/")]
            self.assertTrue(all("ne_10m_land" in p for p in raw))
        source = (SOFTWARE / "src/gray_whale_corridor/release/archived_figures.py").read_text()
        self.assertIn('prepared_ranges=read("ranges")', source)
        self.assertIn('module.draw_figure3(bundle, primary, late, ranges, grid, maps)', source)
        self.assertNotIn('module.prepare_figure3(', source)

    def test_draw_blocks_unchanged(self):
        names = ["build_si_b2_june_july_asamm_observations_v1", "build_si_b3_aug_oct_asamm_observations_v1",
                 "build_si_b4_pelagic_support_response_v1", "build_si_c2_prey_currency_sensitivity_v1"]
        for name in names:
            rel = f"scripts/manuscript_figures_v3/{name}.py"
            before = function(rel, "build_figure", BACKUP).body
            after = function(rel, "build_figure").body
            marker = "scenario_order" if name.endswith("currency_sensitivity_v1") else "fig"
            def drawing(body):
                for index, node in enumerate(body):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == marker or
                        isinstance(t, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == marker for x in t.elts)
                        for t in node.targets):
                        return body[index:]
                raise AssertionError(name)
            self.assertEqual(dump(drawing(before)), dump(drawing(after)), name)

    def test_regional_footprint_is_same_expression(self):
        rel = "scripts/run_phase8v_asamm_effort_rates.py"
        old = function(rel, "make_footprints", BACKUP)
        new = function(rel, "make_footprints")
        added = next(n for n in new.body if isinstance(n, ast.If) and ast.unparse(n.test) == "regional_only")
        self.assertEqual(dump(old.body), dump([n for n in new.body if n is not added]))
        original = next(n for n in old.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "footprints")
        self.assertEqual(ast.dump(original.value), ast.dump(added.body[0].value.elts[1]))

    def test_rate_math_and_missing_count_rule_unchanged(self):
        rel = "src/gray_whale_corridor/metrics/survey_effort.py"
        before = function(rel, "summarize_rates", BACKUP).body
        after = function(rel, "summarize_rates").body
        self.assertEqual(ast.dump(before[0]), ast.dump(after[0]))
        a = next(n for n in before if isinstance(n, ast.For))
        b = next(n for n in after if isinstance(n, ast.For))
        self.assertEqual(dump(a.body), dump(b.body[1:]))
        self.assertEqual(ast.unparse(b.body[0].test), "metrics is not None and name not in metrics")
        old_assignment = before[-2]
        new_assignment = after[-2]
        self.assertEqual(ast.dump(old_assignment.targets[0].slice.elts[0]),
                         ast.dump(new_assignment.targets[0].slice.elts[0]))
        self.assertEqual(ast.dump(old_assignment.value), ast.dump(new_assignment.value))
        self.assertEqual(ast.dump(old_assignment.targets[0].slice.elts[1]), ast.dump(after[-3].value.generators[0].iter))

    def test_v8_annual_selection_is_explicit(self):
        config = annual_v8.annual_contract()
        self.assertEqual(config["mode_sets"]["transect"], [2])
        source = (SOFTWARE / "src/gray_whale_corridor/release/annual_v8.py").read_text()
        parsed = ast.parse(source)
        calls = [ast.unparse(n.func) for n in ast.walk(parsed) if isinstance(n, ast.Call)]
        for forbidden in ("legacy.main", "legacy.flight_bootstrap", "legacy.make_correlations", "temporal.main", "temporal.load_sources"):
            self.assertNotIn(forbidden, calls)
        self.assertIn('metrics=("whales_per_100km",)', source)
        self.assertIn('regional_only=True', source)
        self.assertIn('2011 if scenario in ("prey_only", "prey_ice") else 2015', source)
        self.assertNotIn('np.random', source)

    def test_table_references_are_v8_not_old_word(self):
        refs = json.loads((control.CONFIG / "frozen_table_references.json").read_text())
        self.assertEqual({k: [len(p["rows"]) for p in v["parts"]] for k, v in refs["tables"].items()},
                         {"TableS1": [28, 19], "TableS2": [6, 6], "TableS3": [20], "TableS4": [8, 4]})
        self.assertNotIn("TableS5", refs["tables"])
        source = (SOFTWARE / "src/gray_whale_corridor/release/table_adapters.py").read_text()
        self.assertNotIn(".docx", source)
        self.assertIn('if parts != expected:', source)

    def test_context_entries_do_not_read_provider_raw_tables(self):
        policy = paths.load_policy()
        for task in ("Fig1", "S1", "TableS5"):
            contract = policy["tasks"][task]
            self.assertEqual(contract["adapter_status"], "archived_context_wired")
            self.assertFalse(contract["hold_reasons"])
            self.assertTrue(all(not policy["files"][p].get("hold") for p in contract["inputs"]))
            self.assertTrue(all("ne_10m_land" in p for p in contract["inputs"] if p.startswith("data/raw/")))
        refs = json.loads((control.CONFIG / "frozen_table_s5_reference.json").read_text())
        self.assertEqual(len(refs["tables"]["TableS5"]["parts"][0]["rows"]), 4)
        self.assertEqual(refs["source_sha256"], json.loads((control.CONFIG / "frozen_table_references.json").read_text())["source_sha256"])


if __name__ == "__main__":
    unittest.main()
