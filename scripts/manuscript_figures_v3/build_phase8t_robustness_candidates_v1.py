#!/usr/bin/env python3
"""Build isolated robustness candidates for manuscript Figures 2-5.

All outputs are written below outputs/robustness_smoke_test_phase8t/. Existing
figures and captions are read-only. Shading and whiskers show parameter or
year-leverage ranges, not sampling confidence intervals.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT / "scripts" / "manuscript_figures"))
sys.path.insert(0, str(ROOT / "scripts" / "manuscript_figures_v3"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import build_figure2_cee_v0 as fig2
import build_figure3_centroid_direction_candidate_v1 as fig3
import build_figure4_four_scenario_candidate_v2 as fig4
import build_figure5_dual_period_candidate_v3 as fig5


PHASE = ROOT / "outputs/robustness_smoke_test_phase8t"
CANDIDATE = PHASE / "figures/candidates"
SOURCE = PHASE / "figures/source_data"
QA = PHASE / "figures/qa"

P1_ROUTE = PHASE / "summaries/p1_idw_route_metrics.csv"
P2_COMPARE = PHASE / "summaries/p2_ice_comparison_metrics.csv"
P3_ROUTE = PHASE / "summaries/p3_vessel_route_metrics.csv"
P3_COMPARE = PHASE / "summaries/p3_vessel_comparison_metrics.csv"
P4_RANGE = PHASE / "validation/p4_asamm_year_leverage_ranges.csv"

PRIMARY_IDW = "primary_k8_p2"
PRIMARY_ICE = "primary_upper_0_55"
PRIMARY_VESSEL = "primary_alpha_0_50"


def _prepare_legacy_export_dirs():
    for directory in (CANDIDATE, SOURCE, QA):
        directory.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def export_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff")]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(QA / f"{stem.name}_review.png", dpi=180, bbox_inches="tight", facecolor="white")
    return outputs


def range_table(frame: pd.DataFrame, primary_setting: str, metrics: list[str]) -> pd.DataFrame:
    keys = ["target_year"]
    primary = frame.loc[frame["setting"].eq(primary_setting), keys + metrics].copy()
    primary = primary.rename(columns={metric: f"{metric}_primary" for metric in metrics})
    grouped = frame.groupby(keys, as_index=False).agg(
        **{
            f"{metric}_minimum": (metric, "min")
            for metric in metrics
        },
        **{
            f"{metric}_maximum": (metric, "max")
            for metric in metrics
        },
    )
    return primary.merge(grouped, on=keys, validate="one_to_one").sort_values("target_year")


def plot_fig2_cost_area(
    ax: plt.Axes, data: pd.DataFrame, *, show_range_annotation: bool = True
) -> None:
    years = data["target_year"].to_numpy()
    cost = "mean_path_cost"
    area = "corridor_area_cells_freq_gt_0_05"
    ax.fill_between(
        years,
        data[f"{cost}_minimum"],
        data[f"{cost}_maximum"],
        color=fig2.PALETTE["cost"],
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        years,
        data[f"{cost}_primary"],
        color=fig2.PALETTE["cost"],
        marker="o",
        markersize=3.6,
        linewidth=1.4,
        label="Mean cost",
        zorder=3,
    )
    ax.tick_params(axis="y", labelcolor=fig2.PALETTE["cost"], labelsize=6.5, pad=1.5)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax2 = ax.twinx()
    ax._fig2_twin_axis = ax2
    ax2.fill_between(
        years,
        data[f"{area}_minimum"],
        data[f"{area}_maximum"],
        color=fig2.PALETTE["area"],
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax2.plot(
        years,
        data[f"{area}_primary"],
        color=fig2.PALETTE["area"],
        marker="s",
        markersize=3.4,
        linewidth=1.2,
        label="Frequent-corridor cells",
        zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor=fig2.PALETTE["area"], labelsize=6.5, pad=1.5)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower center",
        bbox_to_anchor=(0.52, 1.025),
        ncol=2,
        fontsize=5.8,
        handlelength=1.25,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    if show_range_annotation:
        ax.text(
            0.02,
            0.965,
            "shading: IDW parameter range",
            transform=ax.transAxes,
            fontsize=5.2,
            color="#666666",
            ha="left",
            va="top",
        )
    ax.set_xlim(years.min() - 0.4, years.max() + 0.4)


def build_figure2(
    *,
    allowed_settings: tuple[str, ...] | None = None,
    stem_name: str = "gray_whale_figure2_idw_parameter_sensitivity_candidate_v1",
    source_name: str = "figure2_idw_parameter_sensitivity_source_data_v1.csv",
    show_range_annotation: bool = True,
    prepared_ranges: pd.DataFrame | None = None,
) -> list[Path]:
    configure_style()
    if prepared_ranges is None:
        routes = pd.read_csv(P1_ROUTE)
        routes = routes.loc[routes["scenario"].eq("prey_only")]
        if allowed_settings is not None:
            routes = routes.loc[routes["setting"].isin(allowed_settings)]
            missing = set(allowed_settings) - set(routes["setting"])
            if missing:
                raise ValueError(f"Missing IDW sensitivity settings: {sorted(missing)}")
        metrics = ["mean_path_cost", "corridor_area_cells_freq_gt_0_05"]
        ranges = range_table(routes, PRIMARY_IDW, metrics)
        ranges.to_csv(SOURCE / source_name, index=False)
    else:
        ranges = prepared_ranges.copy()

    ocean_grid = fig2.load_ocean_grid()
    land = fig2.gpd.read_file(fig2.LAND).to_crs(4326)
    extent = fig2.extent_from_ocean_grid(ocean_grid)
    grids = {year: fig2.load_year_grid(year) for year in fig2.YEARS}
    centroid = pd.read_csv(fig2.CENTROID_SUMMARY)
    centroid = centroid.loc[
        centroid["prey_variable"].eq("total_kJm2_frozen")
        & centroid["comparison_type"].eq("consecutive_year_prey_only")
    ].sort_values(["from_year", "to_year"])

    fig = plt.figure(figsize=(9.0, 7.4), constrained_layout=False)
    gs = GridSpec(3, 10, figure=fig, height_ratios=[1.25, 1.25, 1.05], hspace=0.30, wspace=0.18)
    prey_axes = [fig.add_subplot(gs[0, i * 2 : i * 2 + 2]) for i in range(5)]
    lcp_axes = [fig.add_subplot(gs[1, i * 2 : i * 2 + 2]) for i in range(5)]
    ax_cost = fig.add_subplot(gs[2, 0:4])
    ax_shift = fig.add_subplot(gs[2, 5:10])
    fig.subplots_adjust(left=0.062, right=0.945, top=0.965, bottom=0.072)
    prey_sc = None
    lcp_sc = None
    for index, year in enumerate(fig2.YEARS):
        prey_sc = fig2.plot_prey_map(prey_axes[index], grids[year], ocean_grid, land, extent, year)
        lcp_sc = fig2.plot_lcp_map(lcp_axes[index], grids[year], ocean_grid, land, extent, year)
    fig2.add_panel_label(prey_axes[0], "A")
    fig2.add_panel_label(lcp_axes[0], "B")
    fig2.add_outer_panel_label(ax_cost, "C")
    fig2.add_outer_panel_label(ax_shift, "D")
    fig2.add_colorbar_for_axes(fig, prey_sc, prey_axes, "Scaled prey support")
    fig2.add_colorbar_for_axes(fig, lcp_sc, lcp_axes, "Corridor frequency")
    plot_fig2_cost_area(ax_cost, ranges, show_range_annotation=show_range_annotation)
    fig2.plot_centroid_trajectory(ax_shift, centroid, land, ocean_grid=ocean_grid, full_extent=extent)
    fig2.align_outer_left_to_reference(fig, ax_cost, prey_axes[0])
    stem = CANDIDATE / stem_name
    outputs = export_figure(fig, stem)
    plt.close(fig)
    return outputs


def plot_fig3_cost(
    ax: plt.Axes, data: pd.DataFrame, *, show_range_annotation: bool = True
) -> None:
    years = data["target_year"].to_numpy()
    metric = "percent_cost_change_b_vs_a"
    primary = data[f"{metric}_primary"].to_numpy()
    lower = primary - data[f"{metric}_minimum"].to_numpy()
    upper = data[f"{metric}_maximum"].to_numpy() - primary
    ax.bar(years, primary, color="#75B7C8", edgecolor="#2B6F80", linewidth=0.5, width=0.72, zorder=2)
    ax.errorbar(years, primary, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#244E5B", elinewidth=0.85, capsize=2.0, capthick=0.75, zorder=4)
    ax.plot(years, primary, color="#2B6F80", marker="o", markersize=3.2, linewidth=1.0, zorder=3)
    ax.set_ylabel("Mean cost increase (%)", fontsize=7.3)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    if show_range_annotation:
        ax.text(0.14, 0.965, "whiskers: ice-threshold range", transform=ax.transAxes, fontsize=5.1, color="#666666", ha="left", va="top")
    ax.set_ylim(0, float(data[f"{metric}_maximum"].max()) + 8)


def plot_fig3_metrics(ax: plt.Axes, data: pd.DataFrame) -> None:
    years = data["target_year"].to_numpy()
    shift = "centroid_displacement_km"
    overlap = "jaccard_freq_gt_0_05"
    ax.fill_between(years, data[f"{shift}_minimum"], data[f"{shift}_maximum"], color=fig3.BASE.PALETTE["shift"], alpha=0.13, linewidth=0)
    ax.plot(years, data[f"{shift}_primary"], color=fig3.BASE.PALETTE["shift"], marker="o", linewidth=1.25, markersize=3.3, label="Centroid shift")
    ax.set_ylabel("Centroid shift (km)", color=fig3.BASE.PALETTE["shift"], fontsize=7.2)
    ax.tick_params(axis="y", labelcolor=fig3.BASE.PALETTE["shift"], labelsize=6.2)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=6.2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax2 = ax.twinx()
    ax2.fill_between(years, data[f"{overlap}_minimum"], data[f"{overlap}_maximum"], color=fig3.BASE.PALETTE["area"], alpha=0.12, linewidth=0)
    ax2.plot(years, data[f"{overlap}_primary"], color=fig3.BASE.PALETTE["area"], marker="s", linewidth=1.1, markersize=3.2, label="Jaccard overlap")
    ax2.set_ylabel("Jaccard overlap", color=fig3.BASE.PALETTE["area"], fontsize=7.2)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=fig3.BASE.PALETTE["area"], labelsize=6.2)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + [ax2.get_lines()[-1]]
    ax.legend(lines, [line.get_label() for line in lines], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=0.9)


def plot_fig3_direction(ax: plt.Axes, data: pd.DataFrame) -> None:
    years = data["target_year"].astype(int).to_numpy()
    metric = "centroid_north_south_displacement_km"
    primary = data[f"{metric}_primary"].to_numpy()
    minimum = data[f"{metric}_minimum"].to_numpy()
    maximum = data[f"{metric}_maximum"].to_numpy()
    ax.axvline(0, color="#666666", linewidth=0.75, zorder=1)
    for year, value, low, high in zip(years, primary, minimum, maximum):
        ax.hlines(year, low, high, color="#A7A7A7", linewidth=2.4, alpha=0.75, zorder=1)
        color = fig3.BASE.PALETTE["ice"] if value >= 0 else fig3.BASE.PALETTE["prey"]
        ax.hlines(year, min(0, value), max(0, value), color=color, linewidth=1.35, zorder=2)
        ax.scatter(value, year, s=23, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    low_lim = min(-25.0, float(minimum.min()) - 5)
    high_lim = max(80.0, float(maximum.max()) + 5)
    ax.set_xlim(low_lim, high_lim)
    ax.set_ylim(2019.65, 2010.25)
    ax.set_yticks(years)
    ax.set_yticklabels(years, fontsize=5.7)
    ax.tick_params(axis="x", labelsize=5.7, length=2)
    ax.tick_params(axis="y", length=2, pad=1.5)
    ax.set_xlabel("North-south displacement (km)", fontsize=6.3, labelpad=2.2)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.45, zorder=0)
    ax.text(0.02, 1.015, "South", transform=ax.transAxes, fontsize=5.6, color=fig3.BASE.PALETTE["prey"], ha="left", va="bottom")
    ax.text(0.98, 1.015, "North", transform=ax.transAxes, fontsize=5.6, color=fig3.BASE.PALETTE["ice"], ha="right", va="bottom")


def build_figure3(
    *,
    comparison_path: Path = P2_COMPARE,
    primary_setting: str = PRIMARY_ICE,
    stem_name: str = "gray_whale_figure3_ice_threshold_sensitivity_candidate_v1",
    source_name: str = "figure3_ice_threshold_sensitivity_source_data_v1.csv",
    row_labels: tuple[str, str] | None = None,
    show_range_annotation: bool = True,
) -> list[Path]:
    configure_style()
    comparison = pd.read_csv(comparison_path)
    metrics = [
        "percent_cost_change_b_vs_a",
        "centroid_displacement_km",
        "jaccard_freq_gt_0_05",
        "centroid_north_south_displacement_km",
    ]
    ranges = range_table(comparison, primary_setting, metrics)
    ranges.to_csv(SOURCE / source_name, index=False)
    base = fig3.BASE
    ocean_grid = base.load_ocean_grid()
    land = base.gpd.read_file(base.LAND).to_crs(4326)
    extent = base.extent_from_ocean_grid(ocean_grid)
    grids = {("prey_only", year): base.load_grid("prey_only", year) for year in base.YEARS}
    grids.update({("prey_ice", year): base.load_grid("prey_ice", year) for year in base.YEARS})

    fig = plt.figure(figsize=(10.3, 8.6), constrained_layout=False)
    ax_cost = fig.add_axes([0.075, 0.735, 0.33, 0.22])
    ax_metrics = fig.add_axes([0.425, 0.735, 0.36, 0.22])
    ax_direction = fig.add_axes([0.855, 0.735, 0.12, 0.22])
    map_axes_top, map_axes_bottom = base.make_map_axes_grid(fig, extent, left=0.080, right=0.925, bottom=0.065, row_gap=0.026, col_gap=0.010)
    plot_fig3_cost(ax_cost, ranges, show_range_annotation=show_range_annotation)
    plot_fig3_metrics(ax_metrics, ranges)
    sm = None
    for index, year in enumerate(base.YEARS):
        sm = base.plot_corridor_surface(map_axes_top[index], grids[("prey_only", year)], ocean_grid, land, extent, title=str(year))
        sm = base.plot_corridor_surface(map_axes_bottom[index], grids[("prey_ice", year)], ocean_grid, land, extent, title=None)
    base.add_colorbar_for_axes(fig, sm, map_axes_top + map_axes_bottom, "Corridor frequency", shrink=1.0)
    plot_fig3_direction(ax_direction, ranges)
    base.add_panel_label_inside(ax_cost, "A")
    base.add_panel_label_inside(map_axes_top[0], "B")
    base.add_panel_label_inside(ax_metrics, "C")
    base.add_panel_label_inside(ax_direction, "D")
    if row_labels is None:
        base.add_vertical_map_row_labels(fig, map_axes_top, map_axes_bottom)
    else:
        base.add_vertical_map_row_labels(
            fig, map_axes_top, map_axes_bottom, labels=row_labels
        )
    stem = CANDIDATE / stem_name
    outputs = export_figure(fig, stem)
    plt.close(fig)
    return outputs


def p3_range(metric: str) -> pd.DataFrame:
    comparison = pd.read_csv(P3_COMPARE)
    source_metric = {
        "centroid_shift_freq_gt_0_05_km": "centroid_displacement_km",
        "jaccard_freq_gt_0_05": "jaccard_freq_gt_0_05",
    }[metric]
    comparison["comparison"] = np.where(
        comparison["scenario_b"].eq("prey_vessel"),
        "vessel_effect_without_ice",
        "vessel_effect_with_ice",
    )
    primary = comparison.loc[
        comparison["setting"].eq(PRIMARY_VESSEL),
        ["target_year", "comparison", source_metric],
    ].rename(columns={source_metric: f"{metric}_primary"})
    grouped = comparison.groupby(["target_year", "comparison"], as_index=False).agg(
        **{
            f"{metric}_minimum": (source_metric, "min"),
            f"{metric}_maximum": (source_metric, "max"),
        }
    )
    return primary.merge(grouped, on=["target_year", "comparison"], validate="one_to_one")


def plot_fig4_scenario_lines(ax: plt.Axes, annual: pd.DataFrame, vessel_routes: pd.DataFrame, metric: str) -> None:
    for scenario in ("prey_vessel", "prey_ice_vessel"):
        subset = vessel_routes.loc[vessel_routes["scenario"].eq(scenario)]
        range_values = subset.groupby("target_year")[metric].agg(["min", "max"]).reset_index()
        ax.fill_between(
            range_values["target_year"],
            range_values["min"],
            range_values["max"],
            color=fig4.SCENARIO_COLORS[scenario],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
    fig4.plot_scenario_lines(ax, annual, metric)


def plot_fig4_comparison(ax: plt.Axes, spatial: pd.DataFrame, metric: str) -> None:
    ranges = p3_range(metric)
    for comparison in fig4.COMPARISONS:
        subset = ranges.loc[ranges["comparison"].eq(comparison)].sort_values("target_year")
        ax.fill_between(
            subset["target_year"],
            subset[f"{metric}_minimum"],
            subset[f"{metric}_maximum"],
            color=fig4.COMPARISON_COLORS[comparison],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
    fig4.plot_comparison_lines(ax, spatial, metric)


def build_figure4(
    *,
    stem_name: str = "gray_whale_figure4_vessel_alpha_sensitivity_candidate_v1",
    show_range_annotation: bool = True,
    write_source_data: bool = True,
) -> list[Path]:
    configure_style()
    annual, spatial = fig4.validate_inputs()
    vessel_routes = pd.read_csv(P3_ROUTE)
    if write_source_data:
        vessel_routes.to_csv(SOURCE / "figure4_vessel_alpha_sensitivity_route_source_data_v1.csv", index=False)
    comparison = pd.read_csv(P3_COMPARE)
    if write_source_data:
        comparison.to_csv(SOURCE / "figure4_vessel_alpha_sensitivity_spatial_source_data_v1.csv", index=False)
    fig, grid = plt.subplots(2, 2, figsize=(7.08, 5.0))
    ax_a, ax_b, ax_c, ax_d = grid.ravel()
    axes = [ax_a, ax_b, ax_c, ax_d]
    for ax in axes:
        fig4.style_axis(ax)
    plot_fig4_scenario_lines(ax_a, annual, vessel_routes, "mean_path_cost")
    ax_a.set_ylabel("Mean modeled path cost")
    ax_a.set_ylim(1050, 2070)
    ax_a.legend(loc="upper right", ncol=2, frameon=False, handlelength=2.2, columnspacing=0.85, handletextpad=0.45, labelspacing=0.35, borderaxespad=0.25)
    plot_fig4_scenario_lines(ax_b, annual, vessel_routes, "corridor_area_cells_freq_gt_0_05")
    ax_b.set_ylabel("Frequent-corridor cells (n)")
    ax_b.set_ylim(160, 285)
    plot_fig4_comparison(ax_c, spatial, "centroid_shift_freq_gt_0_05_km")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Corridor-centroid displacement (km)")
    ax_c.set_ylim(0, 130)
    ax_c.legend(loc="lower right", frameon=False, handlelength=2.1, handletextpad=0.45, labelspacing=0.35, borderaxespad=0.35)
    plot_fig4_comparison(ax_d, spatial, "jaccard_freq_gt_0_05")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Frequent-corridor overlap (Jaccard)")
    ax_d.set_ylim(0, 1)
    if show_range_annotation:
        ax_a.text(0.015, 0.025, "shading on vessel scenarios: alpha range", transform=ax_a.transAxes, fontsize=5.1, color="#666666", ha="left", va="bottom")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.98, wspace=0.30, hspace=0.34)
    fig4.add_panel_labels(fig, axes)
    stem = CANDIDATE / stem_name
    outputs = export_figure(fig, stem)
    plt.close(fig)
    return outputs


def draw_metric_panel_intervals(
    fig: plt.Figure,
    spec,
    data: pd.DataFrame,
    panel_label: str,
    period_label: str,
    *,
    labels_below: bool = False,
) -> list[plt.Axes]:
    base = fig5.base
    container = fig.add_subplot(spec)
    container.set_axis_off()
    container.text(-0.035, 1.12, panel_label, transform=container.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="top")
    container.text(0.018, 1.09, period_label, transform=container.transAxes, fontsize=6.6, fontweight="bold", color="#4B4B4B", ha="left", va="top")
    inner = spec.subgridspec(1, 3, width_ratios=[1.03, 0.98, 1.08], wspace=0.32)
    definitions = [
        ("effort_footprint_percentile", "Effort-footprint\npercentile", (45.0, 70.0), [50, 60, 70], "{:.1f}", "higher"),
        ("top5_overlap_percent", "Sightings in annual\ntop 5% (%)", (0.0, 25.0), [0, 10, 20], "{:.1f}%", "higher"),
        ("distance_to_frequent_corridor_km", "Median distance to\nfrequent corridor (km)", (0.0, 36.0), [0, 10, 20, 30], "{:.1f}", "lower"),
    ]
    axes: list[plt.Axes] = []
    y = np.arange(len(data), dtype=float)
    for index, (metric, xlabel, xlim, xticks, fmt, direction) in enumerate(definitions):
        ax = fig.add_subplot(inner[0, index])
        axes.append(ax)
        for yi, row in zip(y, data.itertuples(index=False)):
            scenario = str(row.scenario)
            value = float(getattr(row, f"{metric}_primary"))
            low = float(getattr(row, f"{metric}_minimum"))
            high = float(getattr(row, f"{metric}_maximum"))
            ax.hlines(yi, xlim[0], xlim[1], color="#EDF0F2", linewidth=0.9, zorder=0)
            ax.hlines(yi, low, high, color=base.SCENARIO_COLORS[scenario], linewidth=1.4, alpha=0.55, zorder=2)
            ax.vlines([low, high], yi - 0.08, yi + 0.08, color=base.SCENARIO_COLORS[scenario], linewidth=0.75, alpha=0.7, zorder=2)
            ax.scatter(value, yi, s=27, marker=base.SCENARIO_MARKERS[scenario], facecolor="white", edgecolor=base.SCENARIO_COLORS[scenario], linewidth=1.2, zorder=3)
            if labels_below:
                ax.text(
                    value,
                    yi + 0.17,
                    fmt.format(value),
                    fontsize=5.5,
                    color=base.TEXT_COLOR,
                    ha="center",
                    va="top",
                )
            else:
                offset = 1.10 if metric != "distance_to_frequent_corridor_km" else 1.40
                ax.text(value + offset, yi, fmt.format(value), fontsize=5.7, color=base.TEXT_COLOR, ha="left", va="center")
        if metric == "effort_footprint_percentile":
            ax.axvline(50, color="#AAB0B5", linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(len(data) - 0.5, -0.5)
        ax.set_xlabel(xlabel, labelpad=3.0)
        ax.tick_params(direction="out", length=2.5, width=0.65, pad=1.8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.99, 1.02, direction, transform=ax.transAxes, fontsize=5.1, color="#747474", ha="right", va="bottom")
        if index == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([base.SCENARIO_LABELS[str(value)] for value in data["scenario"]], fontsize=6.25)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
    return axes


def build_figure5(
    *,
    labels_below: bool = False,
    stem_name: str = "gray_whale_figure5_year_leverage_intervals_candidate_v1",
    exclusive_labels: bool = False,
    source_name: str = "figure5_year_leverage_intervals_source_data_v1.csv",
) -> list[Path]:
    configure_style()
    ranges = pd.read_csv(P4_RANGE)
    full = ranges.loc[ranges["analysis_period"].eq("full_2011_2019")].copy()
    common = ranges.loc[ranges["analysis_period"].eq("common_2015_2019")].copy()
    order_full = {"prey_only": 0, "prey_ice": 1}
    order_common = {scenario: index for index, scenario in enumerate(fig5.base.SCENARIOS)}
    full = full.sort_values("scenario", key=lambda values: values.map(order_full)).reset_index(drop=True)
    common = common.sort_values("scenario", key=lambda values: values.map(order_common)).reset_index(drop=True)
    ranges.to_csv(SOURCE / source_name, index=False)
    union, stable = fig5.base.load_recurrent_cells()
    exposure = fig5.base.load_exposure(stable)
    land, ocean_grid, extent, dbo, boundary = fig5.load_map_context_v3()
    fig5.prior.add_geographic_labels = fig5.add_geographic_labels

    fig = plt.figure(figsize=(7.2, 8.65), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.72, 1.12, 2.2], left=0.105, right=0.925, bottom=0.045, top=0.955, hspace=0.43)
    draw_metric_panel_intervals(
        fig,
        outer[0, 0],
        full,
        "A",
        "2011-2019 | 1,179 sightings",
        labels_below=labels_below,
    )
    draw_metric_panel_intervals(
        fig,
        outer[1, 0],
        common,
        "B",
        "2015-2019 | 666 sightings",
        labels_below=labels_below,
    )
    maps = outer[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.17)
    c_spec = maps[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    d_spec = maps[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    ax_c = fig.add_subplot(c_spec[0, 0])
    ax_c_dummy = fig.add_subplot(c_spec[0, 1])
    ax_c_dummy.set_axis_off()
    ax_d = fig.add_subplot(d_spec[0, 0])
    cax_d = fig.add_subplot(d_spec[0, 1])
    fig5.prior.draw_recurrence_map(
        ax_c, land, ocean_grid, extent, dbo, boundary, union,
        exclusive_labels=exclusive_labels,
    )
    fig5.prior.draw_exposure_map(fig, ax_d, cax_d, land, ocean_grid, extent, dbo, boundary, exposure)
    fig5.base.add_panel_label(ax_c, "C")
    fig5.base.add_panel_label(ax_d, "D")
    stem = CANDIDATE / stem_name
    outputs = export_figure(fig, stem)
    plt.close(fig)
    return outputs


def main() -> None:
    _prepare_legacy_export_dirs()
    outputs: dict[str, list[str]] = {}
    for label, builder in (
        ("figure2", build_figure2),
        ("figure3", build_figure3),
        ("figure4", build_figure4),
        ("figure5", build_figure5),
    ):
        built = builder()
        outputs[label] = [str(path.relative_to(ROOT)) for path in built]
        print(f"Built {label}: {built[0].name}", flush=True)
    (QA / "phase8t_candidate_export_manifest.json").write_text(
        json.dumps(outputs, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
