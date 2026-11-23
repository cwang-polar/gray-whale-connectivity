"""study drawing adapters, imported scientifically only on explicit task execution."""

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys

from .control import SOFTWARE


@contextmanager
def plotting_imports(paths):
    paths.require_reserved()
    previous_path = list(sys.path)
    previous_env = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for key, folder in (("MPLCONFIGDIR", "cache/matplotlib"), ("XDG_CACHE_HOME", "cache/xdg")):
            os.environ[key] = str(paths.output(folder))
        sys.path[:0] = [str(SOFTWARE / "scripts/manuscript_figures_v3"),
                       str(SOFTWARE / "scripts/manuscript_figures"), str(SOFTWARE / "scripts")]
        yield
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def render_fig5(paths):
    paths.require_task("Fig5")
    protected = paths.snapshot()
    paths.reserve_output()
    with plotting_imports(paths):
        module = importlib.import_module("build_figure5_6_split_20261117")
        run = module.CandidateRun(paths.output("results"), protected,
                                  release_paths=paths, figure_keys=["5"])
        frames = {"panels_A_B_unchanged_metrics": run.read(paths.input("metrics"), "Frozen V8 spatial metrics"),
                  "annual": run.read(paths.input("annual"), "Frozen V8 annual cost and effort-rate values")}
        annual = frames["annual"]
        band = run.read(paths.input("cost_band"), "Frozen Fig. 3 joint ice-anchor envelope")
        module.np.testing.assert_allclose(annual.mean_path_cost, band.mean_path_cost_b_primary, rtol=0, atol=1e-9)
        for name in ["mean_path_cost_b_minimum", "mean_path_cost_b_maximum"]:
            module.np.testing.assert_array_equal(annual[name], band[name])
        assert len(annual) == 9 and annual.year.tolist() == list(range(2011, 2020))
        for alias in ["metrics", "annual", "correlations", "cost_band"]:
            run.copy_source("5", paths.input(alias), alias, "Existing frozen source, no recalculation")
        module.workflow.style()
        module.figure5(run, frames)
        (run.paths["5"] / "caption.md").write_text(paths.figure_reference("Fig5")["caption"], encoding="utf-8")
        run.finish()


def render_fig6(paths):
    paths.require_task("Fig6")
    protected = paths.snapshot()
    paths.reserve_output()
    with plotting_imports(paths):
        module = importlib.import_module("build_figure6_tight_candidate_20261117")
        run = module.TightExport(2, 1.0, release_paths=paths)
        frames = {}
        for name in ["panel_C_unchanged_recurrence", "panel_D_unchanged_exposure",
                     "panel_C_survey_union_cells", "panel_D_survey_union_cells"]:
            frames[name] = module.pd.read_csv(run.source(paths.input(name), "Frozen V8 Fig. 6 table"))
        module.baseline.workflow.style()
        module.baseline.figure6(run, frames)
        changed = [key for key, value in protected.items()
                   if module.baseline.digest(paths.resolve_label(key)) != value]
        if changed:
            raise RuntimeError({"input_changes": changed})
        (run.bundle / "run_manifest.json").write_text(json.dumps({
            "sources": run.sources, "outputs": run.exports, "checks": run.checks,
            "input_hashes": protected, "input_changes": changed,
            "note": "Existing V8 drawing function and 1-mm tight export; no metrics recalculated."
        }, indent=2), encoding="utf-8")
        (run.out / "caption.md").write_text(paths.figure_reference("Fig6")["caption"], encoding="utf-8")


def render_s8(paths):
    paths.require_task("S8")
    paths.reserve_output()
    with plotting_imports(paths):
        module = importlib.import_module("build_si8_available_period_single_20261117")
        module.main(paths)
