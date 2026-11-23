#!/usr/bin/env python3
"""Split the frozen Fig. 5 into new candidates without changing legacy files."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gray_whale_fig56_split_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/gray_whale_fig56_split_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from shapely.geometry import box

import build_ncc_seasonality_and_footprint_candidates as workflow
import build_phase8v_temporal_metric_candidates_20261117 as temporal

OUT = ROOT / "outputs/manuscript_figures_v3"
V8_REFERENCES = ROOT / "config/ncc_v8/frozen_figure_references.json"
OLD = OUT / "figure_5"
OLD_STEM = "gray_whale_figure5_asamm_footprint_candidate_v5"
TEMP = ROOT / "outputs/asamm_effort_rates_phase8v/figure_candidates_20261117_v1"
STEMS = {
    "5": "gray_whale_figure5_spatial_annual_candidate_v1",
    "6": "gray_whale_figure6_recurrence_exposure_candidate_v1",
    "S8": "gray_whale_si_S8_temporal_sensitivity_candidate_v1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def final_text(element):
    raise RuntimeError("Word parsing is not a release dependency; use frozen V8 references")


def word_paragraphs(path):
    raise RuntimeError("Live Word documents are not accepted by the V8 release entry")


def snapshot():
    paths = set()
    for folder in [OUT, ROOT / "outputs/robustness_smoke_test_phase8t/figures"]:
        paths.update(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in
                     {".png", ".pdf", ".svg", ".tif", ".tiff"})
    for folder in [ROOT / "scripts", ROOT / "src"]:
        paths.update(folder.rglob("*.py"))
    paths.add(V8_REFERENCES)
    paths.discard(Path(__file__).resolve())
    return {rel(p): digest(p) for p in sorted(paths)}


class CandidateRun:
    def __init__(self, root, protected, review=False, *, release_paths=None, figure_keys=None):
        self.release_paths = release_paths
        self.root = root
        self.root.mkdir(exist_ok=review)
        (self.root / "baselines").mkdir(exist_ok=review)
        (self.root / "qa").mkdir(exist_ok=review)
        self.protected = protected
        self.sources = {}
        self.outputs = []
        self.checks = {}
        self.paths = {
            "5": OUT / "figure_5",
            "6": OUT / "figure_6",
            "S8": OUT / f"supplementary_figures_working/temporal_sensitivity_S8_20261117_v{STEMS['S8'].rsplit('_v', 1)[1]}",
        }
        if release_paths is not None:
            self.paths = {key: release_paths.output(f"figures/Fig{key}") for key in figure_keys}
        for key, path in self.paths.items():
            for child in ["candidates", "source_data", "qa"]:
                (path / child).mkdir(parents=True, exist_ok=True)
            if (path / "candidates" / (STEMS[key] + ".png")).exists() and not review:
                raise FileExistsError(f"Existing candidate is protected: {key}")

    def relative(self, path):
        return self.release_paths.label(path) if self.release_paths is not None else rel(path)

    def resolve(self, label):
        return self.release_paths.resolve_label(label) if self.release_paths is not None else ROOT / label

    def source(self, path, role):
        path = Path(path)
        if self.release_paths is not None:
            path = self.resolve(self.relative(path))
        self.sources[self.relative(path)] = {"file_path": self.relative(path), "sha256": digest(path), "role": role}
        return path

    def read(self, path, role):
        return pd.read_csv(self.source(path, role))

    def copy_source(self, key, source, name, role):
        source = self.source(source, role)
        target = self.paths[key] / "source_data" / f"{STEMS[key]}_{name}.csv"
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError(f"Existing source copy differs: {target}")
        else:
            copy2(source, target)
        assert digest(source) == digest(target)
        self.outputs.append(self.relative(target))

    def export(self, key, fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        outside = []
        for t in texts:
            bb = t.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1 + 1 or bb.y1 > fig.bbox.y1 + 1:
                outside.append(t.get_text())
        details = {"size_mm": (fig.get_size_inches() * 25.4).tolist(),
                   "minimum_text_pt": min(t.get_fontsize() for t in texts),
                   "canvas_text_outside": outside}
        folder = self.paths[key]
        for ext in ["png", "pdf", "svg"]:
            target = folder / "candidates" / f"{STEMS[key]}.{ext}"
            fig.savefig(target, dpi=600, facecolor="white")
            self.outputs.append(self.relative(target))
        review = folder / "qa" / f"{STEMS[key]}_review.png"
        fig.savefig(review, dpi=190, facecolor="white")
        self.outputs.append(self.relative(review))
        details["editable_svg_text_elements"] = (folder / "candidates" / f"{STEMS[key]}.svg").read_text().count("<text")
        details["pdf_font_embedded"] = b"/FontFile2" in (folder / "candidates" / f"{STEMS[key]}.pdf").read_bytes()
        self.checks[key] = details
        plt.close(fig)

    def finish(self):
        changed = [path for path, checksum in self.protected.items() if digest(self.resolve(path)) != checksum]
        changed_sources = [path for path, info in self.sources.items() if digest(self.resolve(path)) != info["sha256"]]
        if changed or changed_sources:
            raise RuntimeError({"protected_changed": changed, "source_changed": changed_sources})
        self.checks.update(protected_files_checked=len(self.protected), protected_files_changed=changed,
                           source_files_changed=changed_sources, models_run=False,
                           existing_spatial_metrics_recalculated=False, word_modified=False,
                           figure4_modified=False, status="Figure rendered from archived analysis inputs")
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "script": self.relative(Path(__file__).resolve()), "script_sha256": digest(Path(__file__).resolve()),
                    "workflow_thread": "Workflow and revising", "workflow_baseline": OLD_STEM,
                    "sources": list(self.sources.values()), "outputs": self.outputs, "checks": self.checks}
        (self.root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        (self.root / "qa/protected_before.json").write_text(json.dumps(self.protected, indent=2))
        pd.DataFrame(self.sources.values()).to_csv(self.root / "source_data_index.csv", index=False)
        for key in self.paths:
            q = self.paths[key] / "qa" / f"{STEMS[key]}_checks.json"
            q.write_text(json.dumps(self.checks, indent=2, ensure_ascii=False))
        print(json.dumps({"bundle": self.relative(self.root), "checks": self.checks}, indent=2), flush=True)


def preserve_baseline(run):
    source = run.source(V8_REFERENCES, "Archived figure references")
    references = json.loads(source.read_text(encoding="utf-8"))
    payload = {
        "basis": "frozen V8 reference metadata",
        "figures": [item for item in references["figures"] if item["id"] in {"Fig5", "Fig6"}],
        "live_word_required": False,
        "rendered_output_verified": False,
    }
    with (run.root / "baselines/v8_frozen_reference.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)
    run.checks["frozen_reference_registered"] = ["Fig5", "Fig6"]


def load_frozen(run):
    frames = {}
    for name in ["panels_A_B_unchanged_metrics", "panel_C_unchanged_recurrence",
                 "panel_D_unchanged_exposure", "panel_C_survey_union_cells", "panel_D_survey_union_cells"]:
        source = OLD / "source_data" / f"{OLD_STEM}_{name}.csv"
        frames[name] = run.read(source, "Unchanged Workflow v5 data")
        run.copy_source("5" if name.startswith("panels_A") else "6", source, name, "Frozen legacy table retained byte-identically")
    ranges = frames["panels_A_B_unchanged_metrics"]
    pd.testing.assert_frame_equal(ranges, pd.read_csv(workflow.legacy.P4_RANGE), check_exact=True)
    annual_path = TEMP / "source_data/line_candidate_annual_data.csv"
    frames["annual"] = run.read(annual_path, "Previously computed pure-transect annual encounter index and Fig. 3 cost envelope")
    run.copy_source("5", annual_path, "panel_C_annual_data", "Existing annual line data; no new analysis")
    band = run.read(temporal.BAND, "Existing Fig. 3A cost sensitivity envelope")
    annual = frames["annual"]
    np.testing.assert_allclose(annual.mean_path_cost, band.mean_path_cost_b_primary, rtol=0, atol=1e-9)
    for name in ["mean_path_cost_b_minimum", "mean_path_cost_b_maximum"]:
        np.testing.assert_array_equal(annual[name], band[name])
    assert len(annual) == 9 and annual.year.tolist() == list(range(2011, 2020))
    recurrence = frames["panel_C_unchanged_recurrence"]
    exposure = frames["panel_D_unchanged_exposure"]
    assert recurrence.recurrence_class.value_counts().to_dict() == {"Prey + ice only": 23, "Both": 20, "Prey only": 18}
    assert len(exposure) == 20 and exposure.exposure_priority.sum() == 9
    assert len(frames["panel_C_survey_union_cells"]) == 1220
    assert len(frames["panel_D_survey_union_cells"]) == 960
    run.checks["legacy_source_values_and_memberships_unchanged"] = True
    run.checks["new_C_reproduces_existing_Fig3_band"] = True
    return frames


def verify_metric_artists(axes, data):
    metrics = ["effort_footprint_percentile", "top5_overlap_percent", "distance_to_frequent_corridor_km"]
    for ax, metric in zip(axes, metrics):
        points = [c for c in ax.collections if isinstance(c, PathCollection)]
        actual = np.vstack([p.get_offsets() for p in points])
        expected = np.column_stack([data[f"{metric}_primary"].to_numpy(), np.arange(len(data))])
        np.testing.assert_array_equal(actual, expected)


def figure5(run, frames):
    f5 = workflow.legacy.fig5
    f5.base.SCENARIO_LABELS["prey_only"] = "Prey"
    fig = plt.figure(figsize=(183 / 25.4, 180 / 25.4), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[.62, 1.04, 1.08],
                             left=.14, right=.90, bottom=.095, top=.945, hspace=.75)
    ranges = frames["panels_A_B_unchanged_metrics"]
    for row, period, label, count in [(0, "full_2011_2019", "A", "1,179"),
                                      (1, "common_2015_2019", "B", "666")]:
        order = {scenario: i for i, scenario in enumerate(f5.base.SCENARIOS)}
        data = ranges.loc[ranges.analysis_period.eq(period)].sort_values("scenario", key=lambda x: x.map(order))
        axes = workflow.legacy.draw_metric_panel_intervals(fig, outer[row, 0], data, label,
                    f"{'2011' if row == 0 else '2015'}–2019 | {count} sightings", labels_below=True)
        container = fig.axes[-4]
        for text in container.texts:
            text.set_y(1.24 if text.get_text() == label else 1.22)
        verify_metric_artists(axes, data)
    data = frames["annual"]
    ax = fig.add_subplot(outer[2, 0])
    right = ax.twinx()
    green, neutral = "#3A8D6D", "#4F5C64"
    ax.fill_between(data.year, data.mean_path_cost_b_minimum, data.mean_path_cost_b_maximum,
                    color=green, alpha=.18, linewidth=0)
    cost, = ax.plot(data.year, data.mean_path_cost, marker="s", color=green, linewidth=1.4,
                    markersize=3.5, label="Prey + ice path cost")
    rate, = right.plot(data.year, data.whales_per_100km, marker="o", color=neutral, linewidth=1.4,
                       markersize=3.5, label="Reported whales / 100 transect km")
    ax.set_xlim(2010.7, 2019.3)
    ax.set_xticks(data.year)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_ylim(0, 3200)
    ax.set_yticks([0, 800, 1600, 2400, 3200])
    right.set_ylim(0, 4.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean modeled path cost\n(resistance-weighted km)", color=green)
    right.set_ylabel("Reported whales / 100 transect km", color=neutral)
    ax.tick_params(axis="y", colors=green)
    right.tick_params(axis="y", colors=neutral)
    ax.spines["left"].set_color(green)
    right.spines["left"].set_visible(False)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(neutral)
    ax.grid(axis="y", color="#E5E7E8", linewidth=.45)
    ax.set_axisbelow(True)
    ax.legend(handles=[cost, rate], loc="lower center", bbox_to_anchor=(.5, 1.10),
               ncol=2, fontsize=6.1, borderaxespad=0)
    primary_r = run.read(run.release_paths.input("correlations") if run.release_paths is not None else TEMP / "source_data/scenario_correlation_summary.csv", "Previously computed annual correlations")
    r = primary_r.loc[primary_r.period.eq("2011–2019") & primary_r.scenario.eq("prey_ice"), "pearson_r"].iloc[0]
    ax.text(.025, .97, f"Pearson r = {r:.2f} | 9 annual estimates", transform=ax.transAxes,
             fontsize=6.2, va="top")
    workflow.panel_label(fig, ax, "C", dx=-.035, dy=.012)
    np.testing.assert_array_equal(cost.get_ydata(), data.mean_path_cost)
    np.testing.assert_array_equal(rate.get_ydata(), data.whales_per_100km)
    run.checks["Fig5_AB_primary_marker_coordinates_unchanged"] = True
    run.export("5", fig)


def has_offsets(ax, coordinates):
    return any(isinstance(c, PathCollection) and np.array_equal(np.asarray(c.get_offsets()), coordinates)
               for c in ax.collections)


def figure6(run, frames):
    paths = getattr(run, "release_paths", None)
    f5 = workflow.legacy.fig5
    for module in [workflow, workflow.legacy, f5, f5.prior, f5.base]:
        run.source(Path(module.__file__), "Existing Workflow plotting helper; main entry point not executed")
    land, ocean, extent, dbo, boundary = f5.load_map_context_v3(paths=paths)
    land = land.clip(box(extent[0], extent[2], extent[1], extent[3]))
    run.source(paths.input("land") if paths is not None else f5.base.LAND, "Unchanged geographic land context")
    run.source(paths.input("dbo") if paths is not None else f5.prior.DBO, "Unchanged DBO2–4 reference polygons")
    run.source(paths.input("grid") if paths is not None else f5.base.GRID, "Unchanged study grid")
    f5.prior.add_geographic_labels = f5.add_geographic_labels
    fig = plt.figure(figsize=(183 / 25.4, 132 / 25.4), facecolor="white")
    outer = fig.add_gridspec(1, 2, left=.065, right=.915, bottom=.16, top=.955, wspace=.17)
    left_spec = outer[0, 0].subgridspec(1, 2, width_ratios=[1, .055], wspace=.035)
    right_spec = outer[0, 1].subgridspec(1, 2, width_ratios=[1, .055], wspace=.035)
    a = fig.add_subplot(left_spec[0, 0])
    fig.add_subplot(left_spec[0, 1]).set_axis_off()
    b = fig.add_subplot(right_spec[0, 0])
    cax = fig.add_subplot(right_spec[0, 1])
    recurrence = frames["panel_C_unchanged_recurrence"]
    exposure = frames["panel_D_unchanged_exposure"]
    f5.prior.draw_recurrence_map(a, land, ocean, extent, dbo, boundary, recurrence, exclusive_labels=True)
    f5.prior.draw_exposure_map(fig, b, cax, land, ocean, extent, dbo, boundary, exposure)
    for category in ["Prey only", "Prey + ice only", "Both"]:
        sub = recurrence.loc[recurrence.recurrence_class.eq(category)]
        assert has_offsets(a, sub[["lon", "lat"]].to_numpy())
    assert has_offsets(b, exposure[["lon", "lat"]].to_numpy())
    assert has_offsets(b, exposure.loc[exposure.exposure_priority, ["lon", "lat"]].to_numpy())
    assert any(np.array_equal(np.asarray(c.get_array()), exposure.mean_scaled_vessel_exposure.to_numpy())
               for c in b.collections if c.get_array() is not None)
    for ax, label, source_label, period in [(a, "A", "C", "2011–2019"), (b, "B", "D", "2015–2019")]:
        cells = frames[f"panel_{source_label}_survey_union_cells"]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        survey = ax.scatter(cells.lon, cells.lat, s=2.6, marker="s", facecolor="#999B9E",
                             edgecolor="none", alpha=.62, zorder=3)
        np.testing.assert_array_equal(np.asarray(survey.get_offsets()), cells[["lon", "lat"]].to_numpy())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        legend = ax.get_legend()
        ax.add_artist(legend)
        handle = Line2D([], [], marker="s", linestyle="", markersize=4,
                         markerfacecolor="#999B9E", markeredgecolor="none", alpha=.62)
        ax.legend([handle], [f"ASAMM survey footprint\n{period} union (surveyed in ≥1 year)"],
                   loc="upper left", bbox_to_anchor=(0, -.025), borderaxespad=0,
                   handlelength=1.7, handletextpad=.5, fontsize=5.5)
        f5.base.add_panel_label(ax, label)
        for text in ax.texts:
            text.set_fontsize(max(5.0, text.get_fontsize()))
            if "maritime boundary" in text.get_text():
                text.set_position((-167.7, 71.4))
            if text.get_path_effects():
                text.set_path_effects([])
                text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": .15})
    run.checks["Fig6_recurrence_locations_categories_and_exposure_values_unchanged"] = True
    run.checks["Fig6_survey_unions_unchanged"] = {"A": 1220, "B": 960}
    fig.canvas.draw()
    map_pos, bar_pos = b.get_position(), cax.get_position()
    cax.set_position([bar_pos.x0, map_pos.y0, bar_pos.width, map_pos.height])
    run.export("6", fig)


def figure_s8(run):
    summary_path = TEMP / "source_data/scenario_correlation_summary.csv"
    footprint_path = TEMP / "source_data/footprint_correlations.csv"
    summary = run.read(summary_path, "Existing temporal correlation estimates and separate sensitivity ranges")
    footprints = run.read(footprint_path, "Existing fixed-footprint correlation diagnostic")
    run.copy_source("S8", summary_path, "summary", "Previously computed correlations")
    run.copy_source("S8", footprint_path, "footprints", "Previously computed footprint sensitivities")
    fig = plt.figure(figsize=(183 / 25.4, 216 / 25.4), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=.19, right=.90, bottom=.08, top=.955, wspace=.95, hspace=.85)
    cmap = LinearSegmentedColormap.from_list("temporal_r", ["#356EA0", "#F7F7F5", "#C4833D"])
    for column, (period, (years, scenarios)) in enumerate(temporal.PERIODS.items()):
        data = summary.loc[summary.period.eq(period)]
        for row, method, lower, upper in [(0, "year omission", "omit_year_min", "omit_year_max"),
                                           (1, "flight bootstrap", "flight_bootstrap_lower", "flight_bootstrap_upper")]:
            ax = fig.add_subplot(grid[row, column])
            letter = [["A", "B"], ["C", "D"]][row][column]
            temporal.correlation_panel(ax, data, f"{period}\n{method}", letter, lower, upper)
            for text in ax.texts:
                if text.get_text() == letter:
                    text.set_x(-.14)
            ax.set_xlabel("Pearson r", fontsize=6.6)
            ax.set_title(f"{period}\n{method}", loc="left", fontsize=7.0, pad=10)
            ax.tick_params(axis="y", labelsize=6.2)
        matrix = footprints.loc[footprints.period.eq(period)].pivot(index="scenario", columns="footprint", values="pearson_r")
        matrix = matrix.loc[scenarios, ["regional", "stable_7", "stable_8", "stable_9"]]
        ax = fig.add_subplot(grid[2, column])
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(matrix)):
            for j in range(4):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if abs(value) > .7 else "#26343B")
        ax.set_xticks(range(4), ["Regional", "105", "52", "19"], fontsize=5.6)
        ax.set_xlabel("Survey footprint (fixed-mask cells)", fontsize=6.1)
        ax.set_yticks(range(len(scenarios)), [temporal.SCENARIOS[s][0] for s in scenarios], fontsize=6.2)
        ax.set_title(f"{period}\nsurvey footprint", loc="left", fontsize=7.0, pad=10)
        ax.text(-.14, 1.11, ["E", "F"][column], transform=ax.transAxes, fontsize=10, fontweight="bold")
    cax = fig.add_axes([.915, .08, .01, .2])
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label("Pearson r", fontsize=6.3, labelpad=3)
    colorbar.ax.tick_params(labelsize=5.6, length=2)
    run.export("S8", fig)


def write_reference_plan(run):
    """Record current identifiers; manuscript renumbering is not a release task."""
    source = run.source(V8_REFERENCES, "Fixed publication figure identifiers")
    references = json.loads(source.read_text(encoding="utf-8"))
    payload = [{"id": item["id"], "label": item["label"]} for item in references["figures"]]
    with (run.root / "v8_figure_identifiers.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--review-layout", action="store_true",
                        help="Refine only this new candidate bundle after validating its protected-file snapshot")
    args = parser.parse_args()
    if args.version < 1:
        raise ValueError("Version must be positive")
    for key, stem in STEMS.items():
        STEMS[key] = stem.rsplit("_v", 1)[0] + f"_v{args.version}"
    root = (args.run_dir or OUT / f"figure5_6_split_20261117_v{args.version}").resolve()
    if not root.is_relative_to(OUT) or (root.exists() and not args.review_layout):
        raise ValueError("Choose a new candidate bundle under manuscript_figures_v3")
    if args.review_layout:
        protected = json.loads((root / "qa/protected_before.json").read_text())
        assert all(digest(ROOT / path) == checksum for path, checksum in protected.items())
        backup = root / "qa/first_layout_run_manifest.json"
        if not backup.exists():
            copy2(root / "run_manifest.json", backup)
    else:
        protected = snapshot()
    run = CandidateRun(root, protected, review=args.review_layout)
    preserve_baseline(run)
    frames = load_frozen(run)
    workflow.style()
    figure5(run, frames)
    figure6(run, frames)
    figure_s8(run)
    write_reference_plan(run)
    run.finish()


if __name__ == "__main__":
    main()
