#!/usr/bin/env python3
"""Plot existing seasonal results and exact ASAMM cell unions; never run models.

Figure contract: Fig. 3 separates declining absolute cost from early-season ice
cost and spatial effects; Fig. 5 distinguishes surveyed areas from regional
screening. Both are mixed quantitative/map composites at 183 mm width. Existing
ice-anchor ranges and year-omission intervals are sensitivity ranges, not CIs.
All outputs are new, versioned candidates; legacy plotting entry points survive.
"""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gray-whale-ncc-candidate-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gray-whale-ncc-candidate-cache")
sys.path.insert(0, str(ROOT / "scripts/manuscript_figures"))
sys.path.insert(0, str(ROOT / "scripts/manuscript_figures_v3"))

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

import build_phase8t_robustness_candidates_v1 as legacy

OUT = ROOT / "outputs/manuscript_figures_v3"
EARLY = ROOT / "outputs/lcp_v2"
LATE = ROOT / "outputs/southward_connectivity_phase8q"
ROBUST = ROOT / "outputs/robustness_smoke_test_phase8t"
V8_REFERENCES = ROOT / "config/ncc_v8/frozen_figure_references.json"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
BASE_STEMS = {
    3: "gray_whale_figure3_joint_ice_anchors_prey_labels_annotation_free_candidate_v4",
    5: "gray_whale_figure5_year_leverage_intervals_prey_labels_candidate_v3",
}
NEW_STEMS = {3: "gray_whale_figure3_ncc_seasonality_candidate", 5: "gray_whale_figure5_asamm_footprint_candidate"}
PRIMARY = "primary_lower_0_15_upper_0_55"
ICE_COLOR, LATE_COLOR, PREY_COLOR = "#276C98", "#B5712D", "#62676B"
FOOTPRINT_COLOR = "#45454D"
YEARS = list(range(2011, 2020))
MAP_YEARS = [2011, 2013, 2015, 2017, 2019]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, content) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, data: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        data.to_csv(f, index=False)


def protected_snapshot() -> dict[str, str]:
    roots = [ROOT / "outputs/manuscript_figures/figure_3", ROOT / "outputs/manuscript_figures/figure_5", OUT, ROBUST / "figures"]
    assets = {p for r in roots for p in r.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".svg", ".tif", ".tiff"}}
    assets.add(V8_REFERENCES)
    assets.update((ROOT / "scripts/manuscript_figures").glob("*.py"))
    assets.update(p for p in (ROOT / "scripts/manuscript_figures_v3").glob("*.py") if p != Path(__file__))
    return {rel(p): sha256(p) for p in sorted(assets)}


class Bundle:
    def __init__(self, number: int):
        self.number = number
        self.root = OUT / f"figure_{number}"
        version = 1
        while any(self.root.rglob(f"{NEW_STEMS[number]}_v{version}*")):
            version += 1
        self.stem = f"{NEW_STEMS[number]}_v{version}"
        for sub in ("candidates", "source_data", "qa", "baselines"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.baseline = self.root / "baselines" / self.stem
        self.baseline.mkdir()
        self.sources = {}
        self.checks = {}
        self.outputs = []
        self.source(Path(__file__), "new plotting entry point")
        copy2(Path(__file__), self.root / "qa" / f"{self.stem}_plotting_entry.py")

    def source(self, path: Path, role: str):
        key = rel(path)
        if key not in self.sources:
            self.sources[key] = {"file_path": key, "sha256": sha256(path), "role": role}
        return path

    def read(self, path: Path, role: str) -> pd.DataFrame:
        return pd.read_csv(self.source(path, role))

    def table(self, suffix: str, data: pd.DataFrame):
        path = self.root / "source_data" / f"{self.stem}_{suffix}.csv"
        write_csv(path, data)
        self.outputs.append(rel(path))

    def preserve_baseline(self):
        source = self.source(V8_REFERENCES, "Archived figure reference")
        references = json.loads(source.read_text(encoding="utf-8"))
        identifier = f"Fig{self.number}"
        reference = next(item for item in references["figures"] if item["id"] == identifier)
        write_json(self.baseline / "baseline_manifest.json", {
            "basis": "frozen V8 reference metadata",
            "figure": reference,
            "live_word_required": False,
            "rendered_output_verified": False,
        })
        self.checks["frozen_reference_registered"] = identifier

    def export(self, fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        sizes = []
        for t in fig.findobj(Text):
            if not t.get_visible() or not t.get_text():
                continue
            sizes.append(t.get_fontsize())
            bb = t.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1 + 1 or bb.y1 > fig.bbox.y1 + 1:
                outside.append(t.get_text())
        self.checks["canvas_text_outside"] = outside
        self.checks["minimum_text_pt"] = min(sizes)
        self.checks["size_mm"] = (fig.get_size_inches() * 25.4).tolist()
        for ext in ("png", "pdf", "svg"):
            path = self.root / "candidates" / f"{self.stem}.{ext}"
            assert not path.exists()
            fig.savefig(path, dpi=600, facecolor="white")
            self.outputs.append(rel(path))
        preview = self.root / "qa" / f"{self.stem}_review.png"
        fig.savefig(preview, dpi=180, facecolor="white")
        self.checks["svg_editable_text_elements"] = (self.root / "candidates" / f"{self.stem}.svg").read_text().count("<text")
        plt.close(fig)

    def finish(self, protected: dict):
        self.checks["protected_files_checked"] = len(protected)
        self.checks["protected_files_changed"] = [p for p, h in protected.items() if sha256(ROOT / p) != h]
        self.checks["source_files_changed"] = [p for p, v in self.sources.items() if sha256(ROOT / p) != v["sha256"]]
        assert not self.checks["protected_files_changed"]
        assert not self.checks["source_files_changed"]
        self.checks.update(models_run=False, metrics_recalculated=False, phase8u_used=False, word_modified=False, status="Figure rendered from archived analysis inputs")
        write_json(self.root / "qa" / f"{self.stem}_checks.json", self.checks)
        write_csv(self.root / "qa" / f"{self.stem}_source_index.csv", pd.DataFrame(self.sources.values()))
        write_json(self.root / "qa" / f"{self.stem}_protected_files.json", protected)
        write_json(self.root / "qa" / f"{self.stem}_outputs.json", self.outputs)
        print(json.dumps({"figure": self.number, "candidate": self.stem, "checks": self.checks}, indent=2), flush=True)


def style():
    legacy.configure_style()
    mpl.rcParams.update({"axes.labelsize": 6.6, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.labelpad": 3, "savefig.bbox": None, "pdf.compression": 9})


def panel_label(fig, ax, label, dx=-0.035, dy=0.012):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, label, fontsize=9, weight="bold", va="bottom")


def annual_axis(ax, ylabel, ylim):
    ax.set_xlim(2010.65, 2019.35)
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y)[2:] for y in YEARS])
    ax.set_xlabel("Year (20xx)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.tick_params(length=2.5, pad=2)
    ax.grid(axis="y", color="#E5E7E8", lw=0.45)
    ax.set_axisbelow(True)


def seasonal_lines(ax, ranges, early_metric, late_values):
    x = ranges.target_year.to_numpy()
    ax.fill_between(x, ranges[f"{early_metric}_minimum"], ranges[f"{early_metric}_maximum"], color=ICE_COLOR, alpha=.15, lw=0)
    ax.plot(x, ranges[f"{early_metric}_primary"], "o-", color=ICE_COLOR, lw=1.1, ms=3, label="May-July")
    ax.plot(x, late_values, "s--", color=LATE_COLOR, lw=1, ms=2.8, label="August-October")


def early_path(scenario, year):
    return EARLY / f"{scenario}_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"


def prepare_figure3(b: Bundle):
    b.source(Path(legacy.__file__), "reused sensitivity range and direction-plot helpers")
    base = legacy.fig3.BASE
    b.source(Path(base.__file__), "reused geographic-aspect maps and common frequency rendering")
    joint = b.read(ROBUST / "ice_joint_anchor_v2/joint_ice_anchor_prey_comparison_metrics.csv", "A-E existing early-season primary and joint-anchor sensitivity summaries")
    late = b.read(LATE / "summaries/phase8q_aug_oct_scenario_comparisons.csv", "B-D within-year late ice versus prey comparisons")
    late = late.loc[late.comparison.eq("ice_effect_without_vessel")].sort_values("target_year").reset_index(drop=True)
    primary = joint.loc[joint.setting.eq(PRIMARY)].sort_values("target_year").reset_index(drop=True)
    assert list(late.target_year) == YEARS == list(primary.target_year)
    assert late.scenario_a.eq("prey_only").all() and late.scenario_b.eq("prey_ice").all()
    np.testing.assert_allclose(primary.mean_path_cost_a, late.mean_cost_a, rtol=1e-12)
    np.testing.assert_allclose(100 * (late.mean_cost_b - late.mean_cost_a) / late.mean_cost_a, late.mean_cost_percent_change, atol=1e-10)
    np.testing.assert_allclose(100 * (primary.mean_path_cost_b - primary.mean_path_cost_a) / primary.mean_path_cost_a, primary.percent_cost_change_b_vs_a, atol=1e-10)
    metrics = ["mean_path_cost_b", "percent_cost_change_b_vs_a", "jaccard_freq_gt_0_05", "centroid_displacement_km", "centroid_north_south_displacement_km"]
    ranges = legacy.range_table(joint, PRIMARY, metrics)
    b.table("panel_A_total_cost", ranges.filter(regex="target_year|mean_path_cost_b").assign(prey_mean_path_cost=primary.mean_path_cost_a.to_numpy()))
    for panel, metric, late_name in [("B", metrics[1], "mean_cost_percent_change"), ("C", metrics[2], "jaccard_freq_gt_0_05"), ("D", metrics[3], "centroid_shift_freq_gt_0_05_km")]:
        b.table(f"panel_{panel}_seasonal_comparison", ranges.filter(regex=f"target_year|{metric}").assign(aug_oct_primary=late[late_name].to_numpy(), reference="same-year prey; frequent cells F_s > 0.05"))
    b.table("panel_E_north_south", joint[["setting", "target_year", "centroid_north_south_displacement_km"]])
    b.table("panels_A_E_all_early_anchor_values", joint)
    b.source(LATE / "phase8q_config.json", "late-season fixed grid and parameter provenance")
    b.source(ROOT / "scripts/run_phase8q_aug_oct_late_season_connectivity.py", "read-only confirmation of common prey and endpoint inputs")
    b.source(ROOT / "scripts/run_prey_ice_lcp_brian_amphipod_C3_v2.py", "read-only early-season endpoint provenance")
    b.source(ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv", "shared fixed 40-by-40 termini")
    grid = b.read(GRID, "shared 2839-cell ocean grid")
    endpoint_paths = b.read(EARLY / "prey_only_brian_kJm2_frozen_C3_path_summary_v2.csv", "read-only endpoint identity check")
    checks = []
    for year in YEARS:
        early = b.read(early_path("prey_only", year), "shared annual prey baseline; read-only verification")
        late_prey = b.read(LATE / f"lcp/corridor_prey_only_aug_oct_target{year}_study_region.csv", "late prey baseline; read-only verification")
        paired = early.set_index("grid_id").sort_index()
        paired_late = late_prey.set_index("grid_id").sort_index()
        assert set(paired.index) == set(grid.grid_id) == set(paired_late.index)
        np.testing.assert_allclose(paired[["x", "y", "lcp_frequency_normalized"]], paired_late[["x", "y", "lcp_frequency_normalized"]], rtol=0, atol=1e-10)
        np.testing.assert_allclose(paired.U_amphipod_prey_global_q05_q95, paired_late.U_prey, rtol=0, atol=1e-12)
        raw_prey = b.read(ROOT / f"outputs/prey_v2/amphipod_prey_surface_total_kJm2_frozen_target{year}_full_trailing_C3_v2.csv", "annual unbounded prey utility before the shared 0.001 numerical floor").set_index("grid_id").sort_index()
        np.testing.assert_allclose(raw_prey.U_amphipod_prey_global_q05_q95, paired_late.U_prey_raw, rtol=0, atol=1e-12)
        ep = b.read(LATE / f"lcp/paths_prey_ice_aug_oct_target{year}_study_region.csv", "read-only late endpoint-pair identity check")
        original_pairs = set(map(tuple, endpoint_paths.loc[endpoint_paths.target_year.eq(year), ["source_grid_id", "destination_grid_id"]].to_numpy()))
        seasonal_pairs = set(map(tuple, ep[["computational_source_grid_id", "computational_destination_grid_id"]].to_numpy()))
        assert original_pairs == seasonal_pairs and len(original_pairs) == 1600
        checks.append({"year": year, "same_prey_utility": True, "same_grid": True, "same_prey_frequency": True, "same_1600_endpoint_pairs": True})
    b.table("seasonal_input_identity_checks", pd.DataFrame(checks))
    b.checks["same_year_prey_grid_endpoints_verified"] = True
    b.checks["early_ice_cost_decline_2011_to_2019_percent"] = float(100 * (1 - primary.mean_path_cost_b.iloc[-1] / primary.mean_path_cost_b.iloc[0]))
    b.checks["late_max_cost_increment_percent"] = float(late.mean_cost_percent_change.max())
    b.checks["late_max_centroid_displacement_km"] = float(late.centroid_shift_freq_gt_0_05_km.max())
    b.checks["late_jaccard_range"] = [float(late.jaccard_freq_gt_0_05.min()), float(late.jaccard_freq_gt_0_05.max())]
    b.checks["2017_north_south_by_setting"] = joint[joint.target_year.eq(2017)].set_index("setting").centroid_north_south_displacement_km.to_dict()
    return primary, late, ranges, grid


def figure3(b: Bundle):
    draw_figure3(b, *prepare_figure3(b))


def draw_figure3(b: Bundle, primary, late, ranges, grid, map_frames=None):
    """Original drawing block; optional archived map frames avoid analysis inputs."""
    base = legacy.fig3.BASE
    metrics = ["mean_path_cost_b", "percent_cost_change_b_vs_a", "jaccard_freq_gt_0_05", "centroid_displacement_km", "centroid_north_south_displacement_km"]

    fig = plt.figure(figsize=(183 / 25.4, 218 / 25.4), facecolor="white")
    a = fig.add_axes([.085, .80, .365, .16])
    bb = fig.add_axes([.565, .80, .365, .16])
    c = fig.add_axes([.085, .545, .225, .17])
    d = fig.add_axes([.385, .545, .225, .17])
    e = fig.add_axes([.735, .545, .195, .17])
    annual_axis(a, "Mean modeled path cost", (0, 3400))
    a.plot(YEARS, primary.mean_path_cost_a, "o-", c=PREY_COLOR, lw=1.1, ms=3, label="Prey")
    a.fill_between(YEARS, ranges.mean_path_cost_b_minimum, ranges.mean_path_cost_b_maximum, color=ICE_COLOR, alpha=.15, lw=0)
    a.plot(YEARS, primary.mean_path_cost_b, "o-", c=ICE_COLOR, lw=1.1, ms=3, label="May-July prey + ice")
    a.legend(loc="lower left", fontsize=5.9, handlelength=1.7, borderpad=.25)
    a.annotate("49% lower in 2019", xy=(2019, primary.mean_path_cost_b.iloc[-1]), xytext=(2014.4, 2600), fontsize=6.2, color=ICE_COLOR, arrowprops={"arrowstyle": "-", "color": ICE_COLOR, "lw": .6})
    annual_axis(bb, "Ice-related cost increase (%)", (-3, 65))
    seasonal_lines(bb, ranges, metrics[1], late.mean_cost_percent_change)
    bb.legend(loc="upper left", ncol=2, fontsize=5.8, columnspacing=.85, handlelength=1.6)
    bb.text(.97, .12, "August-October: 0-0.064%", transform=bb.transAxes, color=LATE_COLOR, fontsize=5.8, ha="right")
    annual_axis(c, "Frequent-corridor overlap\n(Jaccard)", (0, 1.07))
    seasonal_lines(c, ranges, metrics[2], late.jaccard_freq_gt_0_05)
    c.set_yticks([0, .25, .5, .75, 1])
    annual_axis(d, "Centroid displacement (km)", (-3, 100))
    seasonal_lines(d, ranges, metrics[3], late.centroid_shift_freq_gt_0_05_km)
    legacy.plot_fig3_direction(e, ranges)
    e.set_xlabel("North-south shift (km)", fontsize=6.2)
    e.set_xticks([-20, 0, 40, 80])
    for ax, label in zip([a, bb, c, d, e], "ABCDE"):
        panel_label(fig, ax, label)
    land = gpd.read_file(b.source(base.LAND, "Natural Earth land for existing cartographic styling")).to_crs(4326)
    for ext in (".shx", ".dbf", ".prj"):
        b.source(base.LAND.with_suffix(ext), "land shapefile companion")
    extent = base.extent_from_ocean_grid(grid)
    land = land.clip(box(extent[0], extent[2], extent[1], extent[3]))
    top, bottom = base.make_map_axes_grid(fig, extent, left=.085, right=.925, bottom=.035, row_gap=.014, col_gap=.008)
    map_records = []
    for row, axes, season in [(0, top, "May-July"), (1, bottom, "August-October")]:
        for index, year in enumerate(MAP_YEARS):
            path = early_path("prey_ice", year) if row == 0 else LATE / f"lcp/corridor_prey_ice_aug_oct_target{year}_study_region.csv"
            data = (b.read(path, "F primary prey+ice corridor frequencies; no vessels")
                    if map_frames is None else map_frames[(season, year)])
            assert set(data.grid_id) == set(grid.grid_id)
            sm = base.plot_corridor_surface(axes[index], data, grid, land, extent, title=str(year) if row == 0 else None)
            axes[index].title.set_fontsize(6.4)
            if map_frames is None:
                map_records.append(data[["grid_id", "lon", "lat", "lcp_count", "lcp_frequency_normalized"]].assign(year=year, season=season, source_file=rel(path)))
    if map_frames is None:
        b.table("panel_F_map_cells", pd.concat(map_records, ignore_index=True))
    for ax in (top[0], bottom[0]):
        # Labels sit mainly over land; leaders identify marine locations.
        ax.text(-159.2, 68.6, "Alaska", fontsize=5.1, ha="center", color="#514B44", zorder=10)
        ax.text(-173.1, 65.9, "Russia", fontsize=5.1, ha="center", color="#514B44", zorder=10)
        for text, xy, xytext in [("Bering\nStrait", (-168.8, 65.8), (-161.0, 66.3)), ("Chirikov\nBasin", (-169.8, 64.8), (-161.0, 64.7))]:
            ax.annotate(text, xy=xy, xytext=xytext, fontsize=5.0, ha="center", va="center", color="#454545", zorder=12, arrowprops={"arrowstyle": "-", "lw": .4, "color": "#666666"})
    cb = base.add_colorbar_for_axes(fig, sm, top + bottom, "Corridor frequency, $F_s$", width=.012, pad=.010, shrink=1)
    cb.ax.tick_params(labelsize=5.6)
    cb.set_label("Corridor frequency, $F_s$", fontsize=6.3, labelpad=3)
    fig.canvas.draw()
    for axes, text in [(top, "May-July prey + ice"), (bottom, "August-October prey + ice")]:
        y0, y1 = axes[0].get_position().y0, axes[0].get_position().y1
        fig.text(.033, (y0 + y1) / 2, text, rotation=90, fontsize=6.5, ha="center", va="center")
    panel_label(fig, top[0], "F", dx=-.035, dy=.012)
    b.checks["map_color_scale"] = {"vmin": 0, "vmax": 1, "PowerNorm_gamma": .65, "shared_across_all_maps": True}
    b.checks["map_bounds_identical"] = all(np.allclose(ax.get_xlim(), top[0].get_xlim()) and np.allclose(ax.get_ylim(), top[0].get_ylim()) for ax in top + bottom)
    b.export(fig)


def footprint_geometry(cells: pd.DataFrame, grid: pd.DataFrame):
    # Build exact shared edges from grid indices, avoiding floating-point slivers.
    x0 = float(np.median(grid.x - 10000 * grid.grid_x))
    y0 = float(np.median(grid.y - 10000 * grid.grid_y))
    np.testing.assert_allclose(grid.x, x0 + grid.grid_x * 10000, atol=1e-6, rtol=0)
    np.testing.assert_allclose(grid.y, y0 + grid.grid_y * 10000, atol=1e-6, rtol=0)
    polygons = [box(x0 + (r.grid_x - .5) * 10000, y0 + (r.grid_y - .5) * 10000, x0 + (r.grid_x + .5) * 10000, y0 + (r.grid_y + .5) * 10000) for r in cells.itertuples()]
    geom = unary_union(polygons)
    assert np.isclose(geom.area, len(cells) * 1e8, rtol=1e-10)
    return gpd.GeoSeries([geom], crs=3338).to_crs(4326)


def figure5(b: Bundle):
    f5 = legacy.fig5
    for mod in (legacy, f5, f5.prior, f5.base):
        b.source(Path(mod.__file__), "reused existing metric/map drawing helper; no analysis entry point executed")
    ranges = b.read(legacy.P4_RANGE, "A-B unchanged primary scores and year-omission intervals")
    data_dir = OUT / "figure_5/source_data"
    recurrence = b.read(data_dir / "figure5_panel_c_recurrent_cells_candidate_v2.csv", "C existing 61-cell recurrence classification")
    exposure = b.read(data_dir / "figure5_panel_d_exposure_priority_candidate_v2.csv", "D existing 20 shared cells and nine priority flags")
    assert recurrence.recurrence_class.value_counts().to_dict() == {"Prey + ice only": 23, "Both": 20, "Prey only": 18}
    assert len(exposure) == 20 and exposure.exposure_priority.sum() == 9
    b.table("panels_A_B_unchanged_metrics", ranges)
    b.table("panel_C_unchanged_recurrence", recurrence)
    b.table("panel_D_unchanged_exposure", exposure)
    grid = b.read(GRID, "study grid for exact 10-km cell boundaries")
    b.source(ROOT / "config/ncc_v8/spatial_reference.json", "Frozen V8 EPSG:3338 spatial reference")
    transformer = Transformer.from_crs(3338, 4326, always_xy=True)
    lon, lat = transformer.transform(grid.x.to_numpy(), grid.y.to_numpy())
    np.testing.assert_allclose(lon, grid.lon, atol=1e-9, rtol=0)
    np.testing.assert_allclose(lat, grid.lat, atol=1e-9, rtol=0)
    effort = b.read(EFFORT, "original annual strict-effort membership, not sightings")
    b.source(ROOT / "scripts/extract_asamm_to_brian_C3_lcp_phase8i2_v2.py", "read-only original annual background selection")
    b.source(ROOT / "scripts/run_phase8t_primary_robustness_smoke_test.py", "read-only unchanged mixed-domain metric implementation")
    effort = effort.loc[effort.year.between(2011, 2019) & effort.validation_window.eq("June_July") & effort.effort_background_type.eq("strict_effort")].copy()
    for year in YEARS:
        ids = set(b.read(early_path("prey_only", year), "annual evaluated grid membership").grid_id)
        assert ids == set(grid.grid_id)
        alternatives = [early_path("prey_ice", year)]
        if year >= 2015:
            alternatives.extend([
                ROOT / f"outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
                EARLY / f"prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
            ])
        for alternative in alternatives:
            assert set(b.read(alternative, "same evaluated grid membership across all compared scenarios").grid_id) == ids
    effort = effort.loc[effort.grid_id.isin(grid.grid_id)].drop_duplicates(["year", "grid_id"])
    b.table("effort_cells_by_year", effort.merge(grid, on="grid_id", validate="many_to_one"))
    counts = effort.groupby("year").grid_id.nunique()
    assert counts.tolist() == [634, 524, 391, 497, 433, 472, 452, 318, 338]
    geometries = []
    for label, first_year, expected_n in [("C", 2011, 1220), ("D", 2015, 960)]:
        sub = effort[effort.year.ge(first_year)]
        members = sub.groupby("grid_id").agg(effort_year_count=("year", "nunique"), effort_years=("year", lambda v: ";".join(map(str, sorted(set(v)))))).reset_index()
        members = members.merge(grid, on="grid_id", validate="one_to_one")
        assert len(members) == expected_n
        b.table(f"panel_{label}_survey_union_cells", members.assign(period=f"{first_year}-2019"))
        geom = footprint_geometry(members, grid)
        geo_path = b.root / "source_data" / f"{b.stem}_panel_{label}_survey_union.geojson"
        with geo_path.open("x") as f:
            f.write(geom.to_json())
        b.outputs.append(rel(geo_path))
        geometries.append(geom)
        b.checks[f"panel_{label}_union_cell_count"] = len(members)
    b.checks["annual_effort_cell_counts"] = counts.to_dict()

    land, ocean, extent, dbo, boundary = f5.load_map_context_v3()
    land = land.clip(box(extent[0], extent[2], extent[1], extent[3]))
    b.source(f5.base.LAND, "unchanged Natural Earth basemap")
    b.source(f5.prior.DBO, "unchanged DBO2-DBO4 geographic reference boxes")
    f5.prior.add_geographic_labels = f5.add_geographic_labels
    f5.base.SCENARIO_LABELS["prey_only"] = "Prey"
    fig = plt.figure(figsize=(183 / 25.4, 226 / 25.4), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[.72, 1.12, 2.2], left=.12, right=.925, bottom=.077, top=.955, hspace=.43)
    for row, period, label, count in [(0, "full_2011_2019", "A", "1,179"), (1, "common_2015_2019", "B", "666")]:
        order = {scenario: index for index, scenario in enumerate(f5.base.SCENARIOS)}
        sub = ranges.loc[ranges.analysis_period.eq(period)].sort_values("scenario", key=lambda x: x.map(order))
        legacy.draw_metric_panel_intervals(fig, outer[row, 0], sub, label, f"{'2011' if row == 0 else '2015'}-2019 | {count} sightings", labels_below=True)
    maps = outer[2, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=.17)
    csp = maps[0, 0].subgridspec(1, 2, width_ratios=[1, .055], wspace=.035)
    dsp = maps[0, 1].subgridspec(1, 2, width_ratios=[1, .055], wspace=.035)
    c = fig.add_subplot(csp[0, 0])
    dummy = fig.add_subplot(csp[0, 1]); dummy.set_axis_off()
    d = fig.add_subplot(dsp[0, 0]); dc = fig.add_subplot(dsp[0, 1])
    f5.prior.draw_recurrence_map(c, land, ocean, extent, dbo, boundary, recurrence, exclusive_labels=True)
    f5.prior.draw_exposure_map(fig, d, dc, land, ocean, extent, dbo, boundary, exposure)
    # Check the actual plotted marker coordinates and color values before overlay.
    assert any(np.array_equal(np.asarray(col.get_array()), exposure.mean_scaled_vessel_exposure.to_numpy()) for col in d.collections if col.get_array() is not None)
    prior_collections = [(ax, [(np.asarray(col.get_offsets()).copy(), None if col.get_array() is None else np.asarray(col.get_array()).copy()) for col in ax.collections]) for ax in (c, d)]
    for ax, geom, label, period in [(c, geometries[0], "C", "2011-2019"), (d, geometries[1], "D", "2015-2019")]:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        geom.boundary.plot(ax=ax, color=FOOTPRINT_COLOR, linewidth=.55, alpha=.8, zorder=4.8)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        f5.base.add_panel_label(ax, label)
        existing_legend = ax.get_legend()
        ax.add_artist(existing_legend)
        ax.legend([Line2D([0], [0], color=FOOTPRINT_COLOR, lw=.7)], [f"ASAMM survey footprint\n{period} union (surveyed in >=1 year)"], loc="upper left", bbox_to_anchor=(0, -.025), borderaxespad=0, handlelength=1.7, handletextpad=.5, fontsize=5.5, labelspacing=.25)
    for ax, before in prior_collections:
        for col, (xy, values) in zip(ax.collections, before):
            np.testing.assert_array_equal(np.asarray(col.get_offsets()), xy)
            if values is not None:
                np.testing.assert_array_equal(np.asarray(col.get_array()), values)
    b.checks["original_map_points_colors_unchanged_after_overlay"] = True
    b.checks["AB_scores_intervals_C_categories_D_exposure_read_without_recalculation"] = True
    b.checks["regional_recurrence_counts"] = recurrence.recurrence_class.value_counts().to_dict()
    b.checks["exposure_priority_count"] = int(exposure.exposure_priority.sum())
    b.checks["outline_definition"] = "Union of exact 10-km EPSG:3338 cell polygons; interior holes retained; no hull, smoothing, buffering or metric clipping"
    for ax in (c, d):
        for text in ax.texts:
            text.set_fontsize(max(5.0, text.get_fontsize()))
            if "maritime boundary" in text.get_text():
                text.set_position((-167.7, 71.4))
            if text.get_path_effects():
                # SVG path-effect text becomes outlines; keep geographic labels editable.
                text.set_path_effects([])
                text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": .15})
    b.export(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["3", "5", "both"], default="both")
    args = parser.parse_args()
    print("Hashing protected figures, scripts and Word documents", flush=True)
    protected = protected_snapshot()
    for number in ([3, 5] if args.figure == "both" else [int(args.figure)]):
        b = Bundle(number)
        write_json(b.root / "qa" / f"{b.stem}_protected_before.json", protected)
        b.preserve_baseline()
        style()
        (figure3 if number == 3 else figure5)(b)
        b.finish(protected)


if __name__ == "__main__":
    main()
