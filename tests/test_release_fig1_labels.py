"""Fig1 display regression tests; no data, model, Word or plot execution."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / "src"))
from gray_whale_corridor.release.archived_figures import cleanup_fig1_panel_c


class Text:
    def __init__(self, text, x=0.66, y=0.08):
        self.text, self.x, self.y = text, x, y
        self.fontsize, self.ha, self.removed = 6.3, "left", False

    def get_text(self):
        return self.text

    def remove(self):
        self.removed = True

    def set_x(self, x):
        self.x = x

    def set_horizontalalignment(self, ha):
        self.ha = ha


class Rectangle:
    def __init__(self, column):
        self.column = column

    def get_x(self):
        return 0.62 + self.column * 0.28 / 7

    def get_width(self):
        return 0.28 / 7


class Fig1LabelTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(texts=[Text("C"), Text("1,600 LCPs per year"),
            Text("frequency map"), Text("cell value = proportion\nof paths crossing cell")],
            patches=[Rectangle(j) for _ in range(7) for j in range(7)])

    def test_exactly_two_texts_change(self):
        ax = self.fixture()
        before = [dict(vars(t)) for t in ax.texts]
        cleanup_fig1_panel_c(ax)
        self.assertEqual(before[:2], [vars(t) for t in ax.texts[:2]])
        self.assertTrue(ax.texts[3].removed)
        self.assertAlmostEqual(ax.texts[2].x, 0.76)
        self.assertEqual(ax.texts[2].ha, "center")
        self.assertEqual((ax.texts[2].y, ax.texts[2].fontsize), (0.08, 6.3))
        self.assertEqual(len(ax.patches), 49)

    def test_unexpected_baseline_refused(self):
        ax = self.fixture()
        ax.texts.pop()
        with self.assertRaises(ValueError):
            cleanup_fig1_panel_c(ax)
        self.assertFalse(any(t.removed for t in ax.texts))

    def test_legacy_export_name_remains_default(self):
        p = SOFTWARE / "scripts/manuscript_figures_v3/build_figure1_three_panel_candidate_v2.py"
        fn = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "build_figure")
        self.assertEqual(fn.args.defaults[-1].value, "gray_whale_figure1_three_panel_candidate_v2")

    def test_only_fig1_has_display_amendment(self):
        refs = json.loads((SOFTWARE / "config/ncc_v8/frozen_figure_references.json").read_text())
        self.assertEqual([x["id"] for x in refs["figures"] if "display_revision" in x], ["Fig1"])
        fig = next(x for x in refs["figures"] if x["id"] == "Fig1")
        self.assertEqual(fig["display_revision"], "label_cleanup_v3")
        self.assertIn("The frequency map in C is illustrative, not an annual model output.", fig["caption"])
        self.assertIn("least-cost paths (LCPs)", fig["caption"])


if __name__ == "__main__":
    unittest.main()
