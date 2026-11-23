#!/usr/bin/env python3
"""Build the isolated V3 Figure 5 dual-period validation candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import geopandas as gpd
import matplotlib as mpl
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FULL_VALIDATION = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "asamm_phase8i2b_main_june_july_validation_summary.csv"
)
COMMON_VALIDATION = (
    ROOT
    / "outputs"
    / "prey_vessel_phase8k"
    / "validation"
    / "asamm_four_scenario_common_period_summary_v2.csv"
)
GRID = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
LAND = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"
PREY_ONLY_PERSISTENCE = (
    ROOT / "outputs" / "lcp_v2" / "prey_only_brian_C3_bottleneck_persistence_v2.csv"
)
PREY_ICE_PERSISTENCE = (
    ROOT / "outputs" / "lcp_v2" / "prey_ice_brian_C3_bottleneck_persistence_v2.csv"
)
BASELINE_FIGURE = (
    ROOT / "outputs" / "manuscript_figures" / "figure_5" / "gray_whale_cee_figure5_draft_v0.png"
)

OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_5"
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SCENARIOS = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIO_LABELS = {
    "prey_only": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}
SCENARIO_COLORS = {
    "prey_only": "#3B6EA8",
    "prey_ice": "#3A8D6D",
    "prey_vessel": "#D9822B",
    "prey_ice_vessel": "#A85A91",
}
SCENARIO_MARKERS = {
    "prey_only": "o",
    "prey_ice": "s",
    "prey_vessel": "^",
    "prey_ice_vessel": "D",
}

RECURRENCE_THRESHOLD = 5
EXPOSURE_THRESHOLD = 0.25
VESSEL_YEARS = range(2015, 2020)

OCEAN_COLOR = "#EAF2F5"
LAND_COLOR = "#D8D2C7"
GRID_COLOR = "#BDD2DB"
TEXT_COLOR = "#303030"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ordered(frame: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["scenario"] = pd.Categorical(result["scenario"], scenarios, ordered=True)
    return result.sort_values("scenario").reset_index(drop=True)


def load_validation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(FULL_VALIDATION)
    full = full.loc[
        full["prey_variable"].eq("total_kJm2_frozen")
        & full["validation_window"].eq("June_July")
        & full["model_layer"].isin(["prey_only", "prey_ice"]),
        [
            "model_layer",
            "record_count",
            "mean_lcp_percentile_within_effort_footprint",
            "percent_records_top5pct_lcp",
            "median_distance_to_core_corridor_km",
        ],
    ].rename(
        columns={
            "model_layer": "scenario",
            "mean_lcp_percentile_within_effort_footprint": "effort_footprint_percentile",
            "percent_records_top5pct_lcp": "top5_overlap_percent",
            "median_distance_to_core_corridor_km": "distance_to_frequent_corridor_km",
        }
    )
    full = ordered(full, ["prey_only", "prey_ice"])
    full["analysis_period"] = "2011-2019"

    common = pd.read_csv(COMMON_VALIDATION).rename(
        columns={
            "mean_effort_footprint_percentile": "effort_footprint_percentile",
            "percent_records_top5pct": "top5_overlap_percent",
            "median_distance_to_frequent_corridor_km": "distance_to_frequent_corridor_km",
        }
    )
    common = common.loc[
        common["scenario"].isin(SCENARIOS),
        [
            "scenario",
            "record_count",
            "effort_footprint_percentile",
            "top5_overlap_percent",
            "distance_to_frequent_corridor_km",
        ],
    ]
    common = ordered(common, SCENARIOS)
    common["analysis_period"] = "2015-2019"

    metrics = [
        "record_count",
        "effort_footprint_percentile",
        "top5_overlap_percent",
        "distance_to_frequent_corridor_km",
    ]
    if len(full) != 2 or full[metrics].isna().any().any():
        raise ValueError("Full-period validation input is incomplete")
    if len(common) != 4 or common[metrics].isna().any().any():
        raise ValueError("Common-period validation input is incomplete")
    if set(full["record_count"].astype(int)) != {1179}:
        raise ValueError("Full-period scenarios must use the same 1,179 records")
    if set(common["record_count"].astype(int)) != {666}:
        raise ValueError("Common-period scenarios must use the same 666 records")
    return full, common


def load_recurrent_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    prey = pd.read_csv(PREY_ONLY_PERSISTENCE)
    ice = pd.read_csv(PREY_ICE_PERSISTENCE)
    prey = prey.loc[
        prey["prey_variable"].eq("total_kJm2_frozen")
        & prey["years_in_top5pct"].ge(RECURRENCE_THRESHOLD),
        ["grid_id", "lon", "lat", "years_in_top5pct"],
    ].rename(
        columns={
            "lon": "prey_lon",
            "lat": "prey_lat",
            "years_in_top5pct": "prey_years_in_top5pct",
        }
    )
    ice = ice.loc[
        ice["prey_variable"].eq("total_kJm2_frozen")
        & ice["years_in_top5pct"].ge(RECURRENCE_THRESHOLD),
        ["grid_id", "lon", "lat", "years_in_top5pct"],
    ].rename(
        columns={
            "lon": "ice_lon",
            "lat": "ice_lat",
            "years_in_top5pct": "ice_years_in_top5pct",
        }
    )
    prey["grid_id"] = prey["grid_id"].astype(str)
    ice["grid_id"] = ice["grid_id"].astype(str)
    union = prey.merge(ice, on="grid_id", how="outer", indicator=True)
    union["lon"] = union[["prey_lon", "ice_lon"]].mean(axis=1)
    union["lat"] = union[["prey_lat", "ice_lat"]].mean(axis=1)
    union["recurrence_class"] = union["_merge"].map(
        {"left_only": "Prey only", "right_only": "Prey + ice only", "both": "Both"}
    )
    stable = union.loc[union["_merge"].eq("both")].copy()
    stable["minimum_years_in_top5pct"] = stable[
        ["prey_years_in_top5pct", "ice_years_in_top5pct"]
    ].min(axis=1)
    if len(prey) != 38 or len(ice) != 43 or len(stable) != 20:
        raise ValueError("Recurrent-cell counts differ from the frozen primary analysis")
    return union, stable


def vessel_surface_path(year: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "surfaces_v2"
        / f"prey_ice_ship_brian_kJm2_frozen_C3_utility_target{year}_v2.csv"
    )


def load_exposure(stable: pd.DataFrame) -> pd.DataFrame:
    stable_ids = set(stable["grid_id"])
    frames: list[pd.DataFrame] = []
    for year in VESSEL_YEARS:
        frame = pd.read_csv(
            vessel_surface_path(year),
            usecols=[
                "grid_id",
                "lon",
                "lat",
                "S_scaled_ship_primary_MayJul_mean",
                "D_ship_alpha0_5_MayJul_mean",
            ],
        )
        frame["grid_id"] = frame["grid_id"].astype(str)
        frame = frame.loc[frame["grid_id"].isin(stable_ids)].copy()
        frame["year"] = year
        frames.append(frame)
    annual = pd.concat(frames, ignore_index=True)
    exposure = annual.groupby("grid_id", as_index=False).agg(
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        mean_scaled_vessel_exposure=("S_scaled_ship_primary_MayJul_mean", "mean"),
        maximum_scaled_vessel_exposure=("S_scaled_ship_primary_MayJul_mean", "max"),
        mean_vessel_modifier=("D_ship_alpha0_5_MayJul_mean", "mean"),
    )
    exposure = exposure.merge(
        stable[
            [
                "grid_id",
                "prey_years_in_top5pct",
                "ice_years_in_top5pct",
                "minimum_years_in_top5pct",
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    exposure["exposure_priority"] = exposure["mean_scaled_vessel_exposure"].gt(
        EXPOSURE_THRESHOLD
    )
    if len(exposure) != 20 or int(exposure["exposure_priority"].sum()) != 9:
        raise ValueError("Exposure-priority counts differ from the frozen primary analysis")
    return exposure.sort_values("mean_scaled_vessel_exposure", ascending=False)


def load_map_context(*, paths=None) -> tuple[
    gpd.GeoDataFrame, pd.DataFrame, tuple[float, float, float, float]
]:
    land = gpd.read_file(paths.input("land") if paths is not None else LAND)
    grid = pd.read_csv(paths.input("grid") if paths is not None else GRID)
    grid = grid.loc[grid["ocean_flag"].astype(int).eq(1)].copy()
    extent = (
        float(grid["lon"].min() - 0.95),
        float(grid["lon"].max() + 0.95),
        float(grid["lat"].min() - 0.43),
        float(grid["lat"].max() + 0.24),
    )
    return land, grid, extent


def draw_metric_panel(
    fig: plt.Figure,
    spec,
    data: pd.DataFrame,
    panel_label: str,
    period_label: str,
) -> list[plt.Axes]:
    container = fig.add_subplot(spec)
    container.set_axis_off()
    container.text(
        -0.035,
        1.12,
        panel_label,
        transform=container.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    container.text(
        0.018,
        1.09,
        period_label,
        transform=container.transAxes,
        fontsize=6.6,
        fontweight="bold",
        color="#4B4B4B",
        ha="left",
        va="top",
    )
    inner = spec.subgridspec(1, 3, width_ratios=[1.03, 0.98, 1.08], wspace=0.32)
    definitions = [
        (
            "effort_footprint_percentile",
            "Effort-footprint\npercentile",
            (45.0, 70.0),
            [50, 60, 70],
            "{:.1f}",
            "higher",
        ),
        (
            "top5_overlap_percent",
            "Sightings in annual\ntop 5% (%)",
            (0.0, 25.0),
            [0, 10, 20],
            "{:.1f}%",
            "higher",
        ),
        (
            "distance_to_frequent_corridor_km",
            "Median distance to\nfrequent corridor (km)",
            (0.0, 36.0),
            [0, 10, 20, 30],
            "{:.1f}",
            "lower",
        ),
    ]
    axes: list[plt.Axes] = []
    y = np.arange(len(data), dtype=float)
    for index, (column, xlabel, xlim, xticks, fmt, direction) in enumerate(definitions):
        ax = fig.add_subplot(inner[0, index])
        axes.append(ax)
        for yi, row in zip(y, data.itertuples(index=False)):
            scenario = str(row.scenario)
            value = float(getattr(row, column))
            ax.hlines(yi, xlim[0], xlim[1], color="#EDF0F2", linewidth=0.9, zorder=0)
            ax.scatter(
                value,
                yi,
                s=27,
                marker=SCENARIO_MARKERS[scenario],
                facecolor="white",
                edgecolor=SCENARIO_COLORS[scenario],
                linewidth=1.2,
                zorder=3,
            )
            offset = 0.45 if column != "distance_to_frequent_corridor_km" else 0.7
            ax.text(
                value + offset,
                yi,
                fmt.format(value),
                fontsize=5.7,
                color=TEXT_COLOR,
                ha="left",
                va="center",
            )
        if column == "effort_footprint_percentile":
            ax.axvline(50, color="#AAB0B5", linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(len(data) - 0.5, -0.5)
        ax.set_xlabel(xlabel, labelpad=3.0)
        ax.tick_params(direction="out", length=2.5, width=0.65, pad=1.8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(
            0.99,
            1.02,
            direction,
            transform=ax.transAxes,
            fontsize=5.1,
            color="#747474",
            ha="right",
            va="bottom",
        )
        if index == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(
                [SCENARIO_LABELS[str(value)] for value in data["scenario"]],
                fontsize=6.25,
            )
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
    return axes


def draw_base_map(
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(OCEAN_COLOR)
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax,
        color=LAND_COLOR,
        edgecolor="white",
        linewidth=0.22,
        zorder=1,
    )
    ax.scatter(
        grid["lon"],
        grid["lat"],
        s=2.6,
        marker="s",
        color=GRID_COLOR,
        edgecolor="none",
        alpha=0.32,
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mean_lat = (ymin + ymax) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.55)


def add_geographic_labels(ax: plt.Axes) -> None:
    labels = [
        ("Russia", -173.2, 66.0, 5.2, "normal"),
        ("Alaska", -158.4, 68.5, 5.2, "normal"),
        ("St. Lawrence\nIsland", -170.1, 63.35, 4.6, "normal"),
        ("Chirikov\nBasin", -168.25, 64.40, 4.7, "italic"),
        ("Bering\nStrait", -167.20, 65.62, 4.7, "italic"),
        ("SE Chukchi\nSea", -168.1, 67.35, 4.7, "italic"),
    ]
    for label, lon, lat, size, style in labels:
        text = ax.text(
            lon,
            lat,
            label,
            fontsize=size,
            color="#4A4640",
            fontstyle=style,
            ha="center",
            va="center",
            zorder=8,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=1.25, foreground="white")])


def draw_recurrence_map(
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
    union: pd.DataFrame,
) -> None:
    draw_base_map(ax, land, grid, extent)
    add_geographic_labels(ax)
    specs = [
        ("Prey only", "#B7BCC2", "#80858A", 22),
        ("Prey + ice only", "#85B6D7", "#547E9B", 22),
        ("Both", "#168778", "#075E54", 39),
    ]
    for category, face, edge, size in specs:
        subset = union.loc[union["recurrence_class"].eq(category)]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=size,
            marker="s",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.45,
            alpha=0.96,
            zorder=6,
        )
    counts = union["recurrence_class"].value_counts()
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.45,
            markersize=5.3,
        )
        for _, face, edge, _ in specs
    ]
    labels = [
        f"Prey only (n={int(counts.get('Prey only', 0))})",
        f"Prey + ice only (n={int(counts.get('Prey + ice only', 0))})",
        f"Both (n={int(counts.get('Both', 0))})",
    ]
    ax.legend(
        handles,
        labels,
        loc="lower right",
        borderpad=0.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=5.3,
    )
    ax.text(
        0.98,
        0.98,
        "2011–2019",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        fontweight="bold",
        color="#4B4B4B",
    )


def draw_exposure_map(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
    exposure: pd.DataFrame,
) -> None:
    draw_base_map(ax, land, grid, extent)
    add_geographic_labels(ax)
    norm = colors.Normalize(vmin=0.0, vmax=0.5)
    cmap = colors.LinearSegmentedColormap.from_list(
        "exposure_magma", plt.cm.magma(np.linspace(0.12, 0.94, 256))
    )
    points = ax.scatter(
        exposure["lon"],
        exposure["lat"],
        c=exposure["mean_scaled_vessel_exposure"],
        cmap=cmap,
        norm=norm,
        s=43,
        marker="s",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.98,
        zorder=6,
    )
    candidates = exposure.loc[exposure["exposure_priority"]]
    ax.scatter(
        candidates["lon"],
        candidates["lat"],
        s=68,
        marker="s",
        facecolor="none",
        edgecolor="#C13E32",
        linewidth=1.0,
        zorder=7,
    )
    colorbar = fig.colorbar(points, cax=cax)
    colorbar.set_label("Mean scaled vessel exposure", fontsize=5.8, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.4, length=2, width=0.55)
    colorbar.outline.set_linewidth(0.55)
    stable_handle = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor="#4C1D73",
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=5.2,
    )
    priority_handle = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor="none",
        markeredgecolor="#C13E32",
        markeredgewidth=1.0,
        markersize=5.5,
    )
    ax.legend(
        [stable_handle, priority_handle],
        [
            f"Recurrent biological cells (n={len(exposure)})",
            f"Exposure priority >0.25 (n={len(candidates)})",
        ],
        loc="lower right",
        borderpad=0.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=5.1,
    )
    ax.text(
        0.98,
        0.98,
        "2015–2019",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        fontweight="bold",
        color="#4B4B4B",
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.07,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_source_data(
    full: pd.DataFrame,
    common: pd.DataFrame,
    union: pd.DataFrame,
    exposure: pd.DataFrame,
) -> list[Path]:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    panel_a = full.copy()
    panel_a["scenario"] = panel_a["scenario"].astype(str)
    panel_a["scenario_label"] = panel_a["scenario"].map(SCENARIO_LABELS)
    panel_a["observation_window"] = "June-July"
    panel_a["corridor_window"] = "May-July"
    panel_a_path = SOURCE_OUT / "figure5_panel_a_full_period_validation_candidate_v1.csv"
    panel_a.to_csv(panel_a_path, index=False, float_format="%.6f")

    panel_b = common.copy()
    panel_b["scenario"] = panel_b["scenario"].astype(str)
    panel_b["scenario_label"] = panel_b["scenario"].map(SCENARIO_LABELS)
    panel_b["observation_window"] = "June-July"
    panel_b["corridor_window"] = "May-July"
    panel_b_path = SOURCE_OUT / "figure5_panel_b_common_period_validation_candidate_v1.csv"
    panel_b.to_csv(panel_b_path, index=False, float_format="%.6f")

    panel_c = union[
        [
            "grid_id",
            "lon",
            "lat",
            "recurrence_class",
            "prey_years_in_top5pct",
            "ice_years_in_top5pct",
        ]
    ].copy()
    panel_c["analysis_period"] = "2011-2019"
    panel_c["annual_membership_definition"] = "annual top 5% of corridor frequency"
    panel_c["recurrence_threshold_years"] = RECURRENCE_THRESHOLD
    panel_c_path = SOURCE_OUT / "figure5_panel_c_recurrent_cells_candidate_v1.csv"
    panel_c.to_csv(panel_c_path, index=False, float_format="%.6f")

    panel_d = exposure.copy()
    panel_d["analysis_period"] = "2015-2019"
    panel_d["exposure_priority_threshold"] = EXPOSURE_THRESHOLD
    panel_d_path = SOURCE_OUT / "figure5_panel_d_exposure_priority_candidate_v1.csv"
    panel_d.to_csv(panel_d_path, index=False, float_format="%.6f")
    return [panel_a_path, panel_b_path, panel_c_path, panel_d_path]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(source_paths: list[Path]) -> Path:
    inputs = [
        (FULL_VALIDATION, "full_period_validation"),
        (COMMON_VALIDATION, "common_period_validation"),
        (GRID, "ocean_grid"),
        (LAND, "land_basemap"),
        (PREY_ONLY_PERSISTENCE, "prey_recurrence"),
        (PREY_ICE_PERSISTENCE, "prey_ice_recurrence"),
        (BASELINE_FIGURE, "immutable_visual_baseline"),
    ]
    inputs.extend((vessel_surface_path(year), f"vessel_surface_{year}") for year in VESSEL_YEARS)
    inputs.extend((path, "plot_ready_source_data") for path in source_paths)
    records = []
    for path, role in inputs:
        records.append(
            {
                "role": role,
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
    output = OUT_ROOT / "provenance_manifest_v1.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    return output


def write_qa_note(full: pd.DataFrame, common: pd.DataFrame, exposure: pd.DataFrame) -> Path:
    QA_OUT.mkdir(parents=True, exist_ok=True)
    output = QA_OUT / "figure5_dual_period_candidate_v1_qa.md"
    text = f"""# Figure 5 dual-period candidate V1 QA

- Backend: Python in the mypy10 environment.
- Figure archetype: mixed quantitative and spatial evidence.
- Panel A: 2011-2019, two scenarios, {int(full['record_count'].iloc[0]):,} identical sighting records per scenario.
- Panel B: 2015-2019, four scenarios, {int(common['record_count'].iloc[0]):,} identical sighting records per scenario.
- Panels A-B: identical metric definitions and axis limits.
- Panel C: 38 Prey recurrent cells, 43 Prey + ice recurrent cells and 20 cells recurrent in both.
- Panel D: {len(exposure)} recurrent biological cells and {int(exposure['exposure_priority'].sum())} exposure-priority cells.
- Recurrence threshold: annual top-5% membership in at least five years.
- Exposure-priority threshold: mean scaled vessel exposure greater than 0.25.
- No inferential intervals are shown; the three validation quantities are descriptive observed-use consistency summaries.
"""
    output.write_text(text, encoding="utf-8")
    return output


def export_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=600, facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return outputs


def build_figure() -> tuple[list[Path], list[Path]]:
    configure_style()
    full, common = load_validation_data()
    union, stable = load_recurrent_cells()
    exposure = load_exposure(stable)
    land, grid, extent = load_map_context()
    source_paths = export_source_data(full, common, union, exposure)

    fig = plt.figure(figsize=(7.2, 8.65), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.72, 1.12, 2.2],
        left=0.105,
        right=0.925,
        bottom=0.045,
        top=0.955,
        hspace=0.43,
    )
    draw_metric_panel(
        fig,
        outer[0, 0],
        full,
        "A",
        "2011–2019 | 1,179 sightings",
    )
    draw_metric_panel(
        fig,
        outer[1, 0],
        common,
        "B",
        "2015–2019 | 666 sightings",
    )

    maps = outer[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.17)
    c_spec = maps[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    d_spec = maps[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    ax_c = fig.add_subplot(c_spec[0, 0])
    ax_c_dummy = fig.add_subplot(c_spec[0, 1])
    ax_c_dummy.set_axis_off()
    ax_d = fig.add_subplot(d_spec[0, 0])
    cax_d = fig.add_subplot(d_spec[0, 1])
    draw_recurrence_map(ax_c, land, grid, extent, union)
    draw_exposure_map(fig, ax_d, cax_d, land, grid, extent, exposure)
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")

    stem = CANDIDATE_OUT / "gray_whale_figure5_dual_period_validation_candidate_v1"
    outputs = export_figure(fig, stem)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        QA_OUT / "gray_whale_figure5_dual_period_validation_candidate_v1_review.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)

    write_qa_note(full, common, exposure)
    write_provenance(source_paths)
    return outputs, source_paths


def main() -> None:
    outputs, source_paths = build_figure()
    for path in outputs + source_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
