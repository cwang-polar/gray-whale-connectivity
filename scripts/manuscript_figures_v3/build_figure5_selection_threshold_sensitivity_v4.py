#!/usr/bin/env python3
"""Build an isolated Figure 5C-D selection-threshold sensitivity diagnostic."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v4")

import geopandas as gpd
import matplotlib as mpl
from matplotlib import colors, patheffects
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "figure_5"
    / "selection_threshold_sensitivity_v4"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

GRID_PATH = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
LAND_PATH = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"
PRIMARY_PREY_PERSISTENCE = (
    ROOT / "outputs" / "lcp_v2" / "prey_only_brian_C3_bottleneck_persistence_v2.csv"
)
PRIMARY_ICE_PERSISTENCE = (
    ROOT / "outputs" / "lcp_v2" / "prey_ice_brian_C3_bottleneck_persistence_v2.csv"
)

YEARS = tuple(range(2011, 2020))
VESSEL_YEARS = tuple(range(2015, 2020))
CORE_PERCENTAGES = (2.5, 5.0, 7.5, 10.0)
RECURRENCE_YEARS = (4, 5, 6)
PRIMARY_CORE_PERCENT = 5.0
PRIMARY_RECURRENCE_YEARS = 5
PRIMARY_EXPOSURE_CUTOFF = 0.25
EXPOSURE_CUTOFFS = np.round(np.arange(0.0, 0.501, 0.01), 2)

SCENARIO_LABELS = {"prey": "Prey", "prey_ice": "Prey + ice"}
SCENARIO_PREFIXES = {
    "prey": "prey_only_brian_kJm2_frozen_C3_lcp_target",
    "prey_ice": "prey_ice_brian_kJm2_frozen_C3_lcp_target",
}

OCEAN_COLOR = "#EAF2F5"
LAND_COLOR = "#D8D2C7"
GRID_COLOR = "#BDD2DB"
TEXT_COLOR = "#303030"
PRIMARY_COLOR = "#C13E32"


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
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def lcp_path(scenario: str, year: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "lcp_v2"
        / f"{SCENARIO_PREFIXES[scenario]}{year}_v2.csv"
    )


def vessel_path(year: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "surfaces_v2"
        / f"prey_ice_ship_brian_kJm2_frozen_C3_utility_target{year}_v2.csv"
    )


def load_grid() -> pd.DataFrame:
    grid = pd.read_csv(GRID_PATH)
    grid["grid_id"] = grid["grid_id"].astype(str)
    grid = grid.loc[grid["ocean_flag"].astype(int).eq(1)].copy()
    if len(grid) != 2839 or grid["grid_id"].duplicated().any():
        raise ValueError("Unexpected regional ocean-grid dimensions")
    return grid


def load_annual_corridors(
    grid: pd.DataFrame,
) -> tuple[dict[str, dict[int, pd.Series]], list[Path]]:
    expected_ids = set(grid["grid_id"])
    annual: dict[str, dict[int, pd.Series]] = {}
    inputs: list[Path] = []
    for scenario in SCENARIO_PREFIXES:
        annual[scenario] = {}
        for year in YEARS:
            path = lcp_path(scenario, year)
            frame = pd.read_csv(
                path,
                usecols=["grid_id", "lcp_frequency_normalized", "ocean_flag"],
            )
            frame["grid_id"] = frame["grid_id"].astype(str)
            frame = frame.loc[frame["ocean_flag"].astype(int).eq(1)].copy()
            if set(frame["grid_id"]) != expected_ids:
                raise ValueError(f"Grid mismatch in {path}")
            frequency = frame.set_index("grid_id")["lcp_frequency_normalized"]
            frequency = pd.to_numeric(frequency, errors="raise").reindex(grid["grid_id"])
            if frequency.isna().any() or (frequency < 0).any():
                raise ValueError(f"Invalid corridor frequency in {path}")
            annual[scenario][year] = frequency
            inputs.append(path)
    return annual, inputs


def load_mean_vessel_exposure(
    grid: pd.DataFrame,
) -> tuple[pd.Series, list[Path]]:
    expected_ids = set(grid["grid_id"])
    frames: list[pd.DataFrame] = []
    inputs: list[Path] = []
    for year in VESSEL_YEARS:
        path = vessel_path(year)
        frame = pd.read_csv(
            path,
            usecols=["grid_id", "S_scaled_ship_primary_MayJul_mean"],
        )
        frame["grid_id"] = frame["grid_id"].astype(str)
        if set(frame["grid_id"]) != expected_ids:
            raise ValueError(f"Grid mismatch in {path}")
        frame["year"] = year
        frames.append(frame)
        inputs.append(path)
    annual = pd.concat(frames, ignore_index=True)
    exposure = annual.groupby("grid_id")["S_scaled_ship_primary_MayJul_mean"].mean()
    exposure = exposure.reindex(grid["grid_id"])
    if exposure.isna().any() or ((exposure < 0) | (exposure > 1)).any():
        raise ValueError("Invalid mean scaled vessel exposure")
    return exposure, inputs


def calculate_sensitivity(
    grid: pd.DataFrame,
    annual: dict[str, dict[int, pd.Series]],
    exposure: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    ids = pd.Index(grid["grid_id"], name="grid_id")
    audit_rows: list[dict[str, object]] = []
    membership_by_core: dict[tuple[str, float], pd.DataFrame] = {}

    for scenario in SCENARIO_PREFIXES:
        for core_percent in CORE_PERCENTAGES:
            annual_membership: dict[int, pd.Series] = {}
            for year in YEARS:
                frequency = annual[scenario][year].reindex(ids)
                threshold = float(frequency.quantile(1.0 - core_percent / 100.0))
                selected = frequency.ge(threshold)
                annual_membership[year] = selected.astype(int)
                audit_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "year": year,
                        "requested_annual_core_percent": core_percent,
                        "frequency_threshold": threshold,
                        "selected_cell_count": int(selected.sum()),
                        "total_ocean_cell_count": len(ids),
                        "effective_selected_percent": float(selected.mean() * 100.0),
                        "boundary_ties_included": True,
                        "boundary_tie_rule": "at_or_above_percentile",
                    }
                )
            membership_by_core[(scenario, core_percent)] = pd.DataFrame(
                annual_membership,
                index=ids,
            )

    summary_rows: list[dict[str, object]] = []
    cell_rows: list[pd.DataFrame] = []
    primary_set: set[str] | None = None
    specification_sets: dict[str, set[str]] = {}

    for core_percent in CORE_PERCENTAGES:
        prey_years = membership_by_core[("prey", core_percent)].sum(axis=1)
        ice_years = membership_by_core[("prey_ice", core_percent)].sum(axis=1)
        for recurrence_years in RECURRENCE_YEARS:
            prey_selected = prey_years.ge(recurrence_years)
            ice_selected = ice_years.ge(recurrence_years)
            shared_selected = prey_selected & ice_selected
            exposed_selected = shared_selected & exposure.reindex(ids).gt(
                PRIMARY_EXPOSURE_CUTOFF
            )
            specification_id = (
                f"annual_top{core_percent:g}pct_recurrence_ge{recurrence_years}yr"
            )
            selected_ids = set(ids[shared_selected])
            specification_sets[specification_id] = selected_ids
            if (
                core_percent == PRIMARY_CORE_PERCENT
                and recurrence_years == PRIMARY_RECURRENCE_YEARS
            ):
                primary_set = selected_ids

            summary_rows.append(
                {
                    "specification_id": specification_id,
                    "annual_core_percent": core_percent,
                    "minimum_recurrence_years": recurrence_years,
                    "prey_recurrent_cell_count": int(prey_selected.sum()),
                    "prey_ice_recurrent_cell_count": int(ice_selected.sum()),
                    "shared_recurrent_cell_count": int(shared_selected.sum()),
                    "exposure_cutoff": PRIMARY_EXPOSURE_CUTOFF,
                    "exposure_priority_cell_count": int(exposed_selected.sum()),
                    "exposure_priority_percent_of_shared": (
                        float(exposed_selected.sum() / shared_selected.sum() * 100.0)
                        if shared_selected.any()
                        else np.nan
                    ),
                    "is_primary_specification": bool(
                        core_percent == PRIMARY_CORE_PERCENT
                        and recurrence_years == PRIMARY_RECURRENCE_YEARS
                    ),
                }
            )

            cell = pd.DataFrame(
                {
                    "grid_id": ids,
                    "annual_core_percent": core_percent,
                    "minimum_recurrence_years": recurrence_years,
                    "prey_years_in_annual_core": prey_years.to_numpy(dtype=int),
                    "prey_ice_years_in_annual_core": ice_years.to_numpy(dtype=int),
                    "prey_recurrent": prey_selected.to_numpy(dtype=bool),
                    "prey_ice_recurrent": ice_selected.to_numpy(dtype=bool),
                    "shared_recurrent": shared_selected.to_numpy(dtype=bool),
                    "mean_scaled_vessel_exposure": exposure.reindex(ids).to_numpy(),
                    "exposure_priority_at_0_25": exposed_selected.to_numpy(dtype=bool),
                    "specification_id": specification_id,
                }
            )
            cell_rows.append(cell)

    if primary_set is None:
        raise RuntimeError("Primary specification was not evaluated")

    summary = pd.DataFrame(summary_rows)
    primary_exposed_set = {
        grid_id for grid_id in primary_set if float(exposure.loc[grid_id]) > 0.25
    }
    for index, row in summary.iterrows():
        selected = specification_sets[str(row["specification_id"])]
        intersection = len(primary_set & selected)
        union = len(primary_set | selected)
        selected_exposed = {
            grid_id for grid_id in selected if float(exposure.loc[grid_id]) > 0.25
        }
        exposed_intersection = len(primary_exposed_set & selected_exposed)
        exposed_union = len(primary_exposed_set | selected_exposed)
        summary.loc[index, "primary_recurrent_cells_retained"] = intersection
        summary.loc[index, "primary_recurrent_retention_percent"] = (
            intersection / len(primary_set) * 100.0
        )
        summary.loc[index, "jaccard_with_primary_recurrent_set"] = (
            intersection / union if union else np.nan
        )
        summary.loc[index, "primary_exposure_priority_cells_retained"] = (
            exposed_intersection
        )
        summary.loc[index, "jaccard_with_primary_exposure_priority_set"] = (
            exposed_intersection / exposed_union if exposed_union else np.nan
        )

    membership = pd.concat(cell_rows, ignore_index=True)
    recurrence_stability = (
        membership.groupby("grid_id", as_index=False)
        .agg(
            recurrent_specification_count=("shared_recurrent", "sum"),
            exposure_priority_specification_count=("exposure_priority_at_0_25", "sum"),
            mean_scaled_vessel_exposure=("mean_scaled_vessel_exposure", "first"),
        )
        .merge(grid[["grid_id", "lon", "lat", "x", "y"]], on="grid_id", how="left")
    )
    n_specs = len(CORE_PERCENTAGES) * len(RECURRENCE_YEARS)
    recurrence_stability["recurrent_selection_fraction"] = (
        recurrence_stability["recurrent_specification_count"] / n_specs
    )
    recurrence_stability["exposure_priority_selection_fraction"] = (
        recurrence_stability["exposure_priority_specification_count"] / n_specs
    )
    recurrence_stability["primary_shared_recurrent"] = recurrence_stability[
        "grid_id"
    ].isin(primary_set)
    recurrence_stability["primary_exposure_priority"] = recurrence_stability[
        "grid_id"
    ].isin(primary_exposed_set)

    threshold_rows: list[dict[str, object]] = []
    primary_exposure = exposure.loc[sorted(primary_set)]
    for cutoff in EXPOSURE_CUTOFFS:
        selected = primary_exposure.gt(cutoff)
        threshold_rows.append(
            {
                "mean_scaled_vessel_exposure_cutoff": cutoff,
                "exposure_priority_cell_count": int(selected.sum()),
                "exposure_priority_percent": float(selected.mean() * 100.0),
                "recurrent_cell_denominator": len(primary_exposure),
                "is_primary_cutoff": bool(np.isclose(cutoff, PRIMARY_EXPOSURE_CUTOFF)),
            }
        )
    threshold_response = pd.DataFrame(threshold_rows)
    audit = pd.DataFrame(audit_rows)
    return audit, summary, membership, recurrence_stability, threshold_response


def load_primary_frozen_counts() -> tuple[int, int, int]:
    prey = pd.read_csv(PRIMARY_PREY_PERSISTENCE)
    ice = pd.read_csv(PRIMARY_ICE_PERSISTENCE)
    prey_ids = set(
        prey.loc[
            prey["prey_variable"].eq("total_kJm2_frozen")
            & prey["years_in_top5pct"].ge(5),
            "grid_id",
        ].astype(str)
    )
    ice_ids = set(
        ice.loc[
            ice["prey_variable"].eq("total_kJm2_frozen")
            & ice["years_in_top5pct"].ge(5),
            "grid_id",
        ].astype(str)
    )
    return len(prey_ids), len(ice_ids), len(prey_ids & ice_ids)


def map_context(
    grid: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    land = gpd.read_file(LAND_PATH)
    extent = (
        float(grid["lon"].min() - 0.95),
        float(grid["lon"].max() + 0.95),
        float(grid["lat"].min() - 0.43),
        float(grid["lat"].max() + 0.24),
    )
    return land, extent


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
        s=2.2,
        marker="s",
        color=GRID_COLOR,
        edgecolor="none",
        alpha=0.28,
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mean_lat = (ymin + ymax) / 2.0
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#5A5A5A")
        spine.set_linewidth(0.55)


def add_geographic_labels(ax: plt.Axes) -> None:
    labels = [
        ("Russia", -173.2, 66.0, 4.9, "normal"),
        ("Alaska", -158.4, 68.5, 4.9, "normal"),
        ("St. Lawrence\nIsland", -170.1, 63.35, 4.3, "normal"),
        ("Chirikov\nBasin", -168.25, 64.40, 4.4, "italic"),
        ("Bering\nStrait", -167.20, 65.62, 4.4, "italic"),
        ("SE Chukchi\nSea", -168.1, 67.35, 4.4, "italic"),
    ]
    for label, lon, lat, size, style in labels:
        item = ax.text(
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
        item.set_path_effects(
            [patheffects.withStroke(linewidth=1.25, foreground="white")]
        )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.075) -> None:
    ax.text(
        x,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def draw_count_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    recurrent = summary.pivot(
        index="annual_core_percent",
        columns="minimum_recurrence_years",
        values="shared_recurrent_cell_count",
    ).reindex(index=CORE_PERCENTAGES, columns=RECURRENCE_YEARS)
    exposed = summary.pivot(
        index="annual_core_percent",
        columns="minimum_recurrence_years",
        values="exposure_priority_cell_count",
    ).reindex(index=CORE_PERCENTAGES, columns=RECURRENCE_YEARS)
    norm = colors.PowerNorm(gamma=0.52, vmin=0, vmax=float(recurrent.max().max()))
    image = ax.imshow(recurrent.to_numpy(), cmap="Blues", norm=norm, aspect="auto")
    for row in range(len(CORE_PERCENTAGES)):
        for col in range(len(RECURRENCE_YEARS)):
            total = int(recurrent.iloc[row, col])
            exposed_count = int(exposed.iloc[row, col])
            rgba = image.cmap(image.norm(total))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.52 else "#263238"
            ax.text(
                col,
                row,
                f"{total}\n({exposed_count})",
                ha="center",
                va="center",
                fontsize=6.2,
                fontweight="bold" if (col == 1 and row == 1) else "normal",
                color=text_color,
            )
    primary_rect = mpl.patches.Rectangle(
        (0.5, 0.5),
        1.0,
        1.0,
        fill=False,
        edgecolor=PRIMARY_COLOR,
        linewidth=1.45,
        clip_on=False,
    )
    ax.add_patch(primary_rect)
    ax.set_xticks(range(len(RECURRENCE_YEARS)), RECURRENCE_YEARS)
    ax.set_yticks(
        range(len(CORE_PERCENTAGES)),
        [f"{value:g}" for value in CORE_PERCENTAGES],
    )
    ax.set_xlabel("Minimum recurrence (years)", labelpad=3)
    ax.set_ylabel("Annual corridor core (%)", labelpad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(0.8)
    cax = inset_axes(
        ax,
        width="3.2%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.035, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_title("Cell\ncount", fontsize=5.4, pad=3)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)


def draw_stability_map(
    fig: plt.Figure,
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
    stability: pd.DataFrame,
    value_column: str,
    primary_column: str,
    period_label: str,
    cbar_label: str,
) -> None:
    draw_base_map(ax, land, grid, extent)
    subset = stability.loc[stability[value_column].gt(0)].copy()
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    points = ax.scatter(
        subset["lon"],
        subset["lat"],
        c=subset[value_column],
        cmap="viridis",
        norm=norm,
        s=23,
        marker="s",
        edgecolor="none",
        alpha=0.96,
        zorder=5,
    )
    primary = stability.loc[stability[primary_column]].copy()
    ax.scatter(
        primary["lon"],
        primary["lat"],
        s=43,
        marker="s",
        facecolor="none",
        edgecolor=PRIMARY_COLOR,
        linewidth=0.85,
        zorder=7,
    )
    add_geographic_labels(ax)
    cax = inset_axes(
        ax,
        width="4.0%",
        height="68%",
        loc="center left",
        bbox_to_anchor=(1.025, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(points, cax=cax)
    cbar.ax.set_title(cbar_label, fontsize=5.4, pad=3, loc="left")
    cbar.ax.tick_params(labelsize=5.2, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    ax.text(
        0.98,
        0.98,
        period_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        fontweight="bold",
        color="#4B4B4B",
    )
    ax.set_anchor("W")


def draw_exposure_response(ax: plt.Axes, response: pd.DataFrame) -> None:
    ax.axvspan(0.20, 0.30, color="#E8ECEF", alpha=0.75, zorder=0)
    ax.step(
        response["mean_scaled_vessel_exposure_cutoff"],
        response["exposure_priority_cell_count"],
        where="post",
        color="#6A3D9A",
        linewidth=1.55,
        zorder=3,
    )
    primary = response.loc[response["is_primary_cutoff"]].iloc[0]
    ax.axvline(
        PRIMARY_EXPOSURE_CUTOFF,
        color=PRIMARY_COLOR,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=2,
    )
    ax.scatter(
        [PRIMARY_EXPOSURE_CUTOFF],
        [primary["exposure_priority_cell_count"]],
        s=34,
        facecolor="white",
        edgecolor=PRIMARY_COLOR,
        linewidth=1.1,
        zorder=4,
    )
    ax.annotate(
        "0.25: 9 cells",
        xy=(PRIMARY_EXPOSURE_CUTOFF, primary["exposure_priority_cell_count"]),
        xytext=(0.287, 11.3),
        fontsize=5.7,
        color=TEXT_COLOR,
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        ha="left",
        va="bottom",
    )
    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(0, 20.8)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_yticks(np.arange(0, 21, 5))
    ax.set_xlabel("Mean scaled vessel-exposure cutoff", labelpad=3)
    ax.set_ylabel("Exposure-priority cells", labelpad=4)
    ax.grid(axis="y", color="#E3E6E8", linewidth=0.55, zorder=0)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=1.8)


def build_figure(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    threshold_response: pd.DataFrame,
) -> list[Path]:
    configure_style()
    land, extent = map_context(grid)
    fig = plt.figure(figsize=(6.65, 5.95), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 0.84],
        height_ratios=[1.0, 1.0],
        left=0.085,
        right=0.955,
        bottom=0.075,
        top=0.965,
        wspace=0.105,
        hspace=0.24,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    draw_count_heatmap(fig, ax_a, summary)
    draw_stability_map(
        fig,
        ax_b,
        land,
        grid,
        extent,
        stability,
        "recurrent_selection_fraction",
        "primary_shared_recurrent",
        "2011-2019",
        "Selection\nfrequency",
    )
    draw_exposure_response(ax_c, threshold_response)
    draw_stability_map(
        fig,
        ax_d,
        land,
        grid,
        extent,
        stability,
        "exposure_priority_selection_fraction",
        "primary_exposure_priority",
        "2015-2019 exposure",
        "Selection\nfrequency",
    )
    for ax, label, x in zip(
        (ax_a, ax_b, ax_c, ax_d),
        "ABCD",
        (-0.075, 0.0, -0.075, 0.0),
    ):
        add_panel_label(ax, label, x=x)

    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    stem = CANDIDATE_OUT / "gray_whale_figure5_selection_threshold_sensitivity_candidate_v4"
    outputs = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=400, facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    QA_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        QA_OUT / "gray_whale_figure5_selection_threshold_sensitivity_candidate_v4_review.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)
    return outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    membership: pd.DataFrame,
    stability: pd.DataFrame,
    threshold_response: pd.DataFrame,
    input_paths: list[Path],
    figure_paths: list[Path],
) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [
        SOURCE_OUT / "annual_core_effective_area_audit_v4.csv",
        SOURCE_OUT / "recurrence_specification_summary_v4.csv",
        SOURCE_OUT / "recurrence_specification_cell_membership_v4.csv",
        SOURCE_OUT / "recurrent_cell_selection_stability_v4.csv",
        SOURCE_OUT / "exposure_threshold_response_v4.csv",
    ]
    frames = [audit, summary, membership, stability, threshold_response]
    for frame, path in zip(frames, source_paths):
        frame.to_csv(path, index=False, float_format="%.8f")

    primary = summary.loc[summary["is_primary_specification"]].iloc[0]
    effective = (
        audit.groupby(["scenario_label", "requested_annual_core_percent"])[
            "effective_selected_percent"
        ]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    primary_exposure = stability.loc[stability["primary_shared_recurrent"]].sort_values(
        "mean_scaled_vessel_exposure", ascending=False
    )
    lower_block = float(
        primary_exposure.loc[
            primary_exposure["mean_scaled_vessel_exposure"].le(
                PRIMARY_EXPOSURE_CUTOFF
            ),
            "mean_scaled_vessel_exposure",
        ].max()
    )
    upper_block = float(
        primary_exposure.loc[
            primary_exposure["mean_scaled_vessel_exposure"].gt(
                PRIMARY_EXPOSURE_CUTOFF
            ),
            "mean_scaled_vessel_exposure",
        ].min()
    )
    counts = summary[
        [
            "annual_core_percent",
            "minimum_recurrence_years",
            "shared_recurrent_cell_count",
            "exposure_priority_cell_count",
        ]
    ].sort_values(["annual_core_percent", "minimum_recurrence_years"])
    primary_recurrent_stability = stability.loc[
        stability["primary_shared_recurrent"], "recurrent_selection_fraction"
    ]
    primary_priority_stability = stability.loc[
        stability["primary_exposure_priority"],
        "exposure_priority_selection_fraction",
    ]

    results_text = [
        "# Figure 5C-D selection-threshold sensitivity results",
        "",
        "This isolated analysis recalculates downstream cell-selection rules from frozen annual corridor and vessel-exposure products. No least-cost paths were rerun.",
        "",
        "## Primary-result reproduction",
        "",
        f"- Prey recurrent cells: {int(primary['prey_recurrent_cell_count'])}.",
        f"- Prey + ice recurrent cells: {int(primary['prey_ice_recurrent_cell_count'])}.",
        f"- Cells recurrent in both: {int(primary['shared_recurrent_cell_count'])}.",
        f"- Exposure-priority cells at mean scaled vessel exposure >0.25: {int(primary['exposure_priority_cell_count'])}.",
        "",
        "## Recurrence-definition response",
        "",
    ]
    for row in counts.itertuples(index=False):
        results_text.append(
            f"- Annual top {row.annual_core_percent:g}%, recurrence >= {row.minimum_recurrence_years} years: "
            f"{int(row.shared_recurrent_cell_count)} shared recurrent cells; "
            f"{int(row.exposure_priority_cell_count)} also exceeded 0.25."
        )
    results_text.extend(
        [
            "",
            "The count of recurrent cells is definition-sensitive, as expected for nested percentile and recurrence cutoffs. The primary setting lies near the middle of the tested grid. The 10% annual-core setting is an outer stress test because discrete corridor frequencies produce boundary ties and effective annual selections wider than exactly 10% in some years.",
            "",
            "## Vessel-exposure response",
            "",
            f"The primary cutoff lies in a natural gap: the largest non-priority value is {lower_block:.3f}, whereas the smallest priority value is {upper_block:.3f}. Consequently, the nine-cell result is unchanged for any cutoff from {lower_block:.3f} through values just below {upper_block:.3f}.",
            "",
            "## Spatial-selection stability",
            "",
            f"All 20 primary recurrent cells were selected by at least {primary_recurrent_stability.min() * 100:.1f}% of the 12 recurrence definitions (median {primary_recurrent_stability.median() * 100:.1f}%). One cell was selected by all 12 definitions.",
            f"All nine primary exposure-priority cells were selected by at least {primary_priority_stability.min() * 100:.1f}% of the recurrence definitions at the fixed 0.25 exposure cutoff (median {primary_priority_stability.median() * 100:.1f}%). One cell was selected by all 12 definitions.",
            "",
            "## Scientific interpretation",
            "",
            "The management-screening result has two distinct robustness properties. The absolute number of recurrent biological cells changes substantially when the operational annual-core and recurrence definitions change, so the count of 20 should remain explicitly tied to its prespecified rule. In contrast, the vessel-exposure cutoff of 0.25 does not divide a dense cluster of nearly identical values; it separates two groups with a large observed gap. Spatial maps show whether the same geographic clusters persist as the recurrence rule is tightened or relaxed.",
            "",
            "## Effective annual-core audit",
            "",
        ]
    )
    for row in effective.itertuples(index=False):
        results_text.append(
            f"- {row.scenario_label}, requested top {row.requested_annual_core_percent:g}%: "
            f"effective annual selection {row.min:.2f}-{row.max:.2f}% "
            f"(median {row.median:.2f}%)."
        )
    (OUT_ROOT / "results_summary.md").write_text(
        "\n".join(results_text) + "\n", encoding="utf-8"
    )

    caption = """# Caption candidate

**Supplementary Figure 10 | Sensitivity of recurrent-corridor and exposure-priority cell selection.** **A,** Numbers of cells recurrent in both benthic prey-only and prey+ice landscapes when annual high-frequency cores comprised cells at or above the 97.5th, 95th, 92.5th or 90th percentile of normalized corridor frequency and recurrence required four, five or six years. Because corridor frequency is discrete, all cells tied at each percentile boundary were retained, so realized core size could slightly exceed the nominal 2.5%, 5%, 7.5% or 10%. Values in parentheses also exceeded the primary mean scaled May-July vessel-exposure cutoff of 0.25 during 2015-2019. The red box marks the primary definition: the annual 95th percentile or higher in at least five years, yielding 20 shared recurrent cells, nine of which exceeded the exposure cutoff. **B,** Fraction of the 12 tested recurrence definitions selecting each cell; red outlines identify the primary 20-cell set. **C,** Number of those 20 cells exceeding alternative vessel-exposure cutoffs; shading spans 0.20-0.30, and the highlighted point marks the primary cutoff of 0.25. **D,** Fraction of the 12 definitions selecting each cell as both recurrent and exposure-priority at the 0.25 cutoff; red outlines identify the primary nine-cell set. Selection frequency measures robustness to alternative operational definitions and is not a biological probability.
"""
    (OUT_ROOT / "caption_candidate_v4.md").write_text(caption, encoding="utf-8")

    expected_counts = load_primary_frozen_counts()
    qa = {
        "backend": "Python",
        "environment": "mypy10",
        "ocean_cell_count": 2839,
        "annual_corridor_input_count": 18,
        "vessel_surface_input_count": 5,
        "sensitivity_specification_count": int(len(summary)),
        "expected_frozen_primary_counts": {
            "prey_recurrent": expected_counts[0],
            "prey_ice_recurrent": expected_counts[1],
            "shared_recurrent": expected_counts[2],
        },
        "recalculated_primary_counts": {
            "prey_recurrent": int(primary["prey_recurrent_cell_count"]),
            "prey_ice_recurrent": int(primary["prey_ice_recurrent_cell_count"]),
            "shared_recurrent": int(primary["shared_recurrent_cell_count"]),
            "exposure_priority": int(primary["exposure_priority_cell_count"]),
        },
        "primary_reproduction_pass": bool(
            (
                int(primary["prey_recurrent_cell_count"]),
                int(primary["prey_ice_recurrent_cell_count"]),
                int(primary["shared_recurrent_cell_count"]),
            )
            == expected_counts
            and int(primary["exposure_priority_cell_count"]) == 9
        ),
        "missing_values": {
            "audit": int(audit.isna().sum().sum()),
            "membership": int(membership.isna().sum().sum()),
            "stability": int(stability.isna().sum().sum()),
            "threshold_response": int(threshold_response.isna().sum().sum()),
        },
        "exposure_gap_around_primary_cutoff": {
            "largest_value_at_or_below_0_25": lower_block,
            "smallest_value_above_0_25": upper_block,
        },
        "existing_main_figure_modified": False,
        "manuscript_files_modified": False,
    }
    (QA_OUT / "selection_threshold_sensitivity_qa_v4.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    if not qa["primary_reproduction_pass"]:
        raise ValueError("Primary selection counts did not reproduce frozen results")

    all_inputs = [
        GRID_PATH,
        LAND_PATH,
        PRIMARY_PREY_PERSISTENCE,
        PRIMARY_ICE_PERSISTENCE,
        *input_paths,
    ]
    records = []
    for path in all_inputs:
        records.append(
            {
                "role": "read_only_input",
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
    for path in source_paths:
        records.append(
            {
                "role": "derived_source_data",
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
    for path in figure_paths:
        records.append(
            {
                "role": "candidate_figure",
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(records).to_csv(
        OUT_ROOT / "input_provenance_manifest_v4.csv", index=False
    )
    manifest = {
        "analysis": "Figure 5C-D selection-threshold sensitivity",
        "version": "v4",
        "annual_boundary_rule": "at or above each requested percentile; ties at the boundary retained",
        "core_percentages": list(CORE_PERCENTAGES),
        "recurrence_years": list(RECURRENCE_YEARS),
        "primary_exposure_cutoff": PRIMARY_EXPOSURE_CUTOFF,
        "years": list(YEARS),
        "vessel_years": list(VESSEL_YEARS),
        "figure_outputs": [str(path.relative_to(ROOT)) for path in figure_paths],
        "source_outputs": [str(path.relative_to(ROOT)) for path in source_paths],
    }
    (OUT_ROOT / "analysis_manifest_v4.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    grid = load_grid()
    annual, lcp_inputs = load_annual_corridors(grid)
    exposure, vessel_inputs = load_mean_vessel_exposure(grid)
    audit, summary, membership, stability, threshold_response = calculate_sensitivity(
        grid, annual, exposure
    )
    figure_paths = build_figure(grid, summary, stability, threshold_response)
    write_outputs(
        audit,
        summary,
        membership,
        stability,
        threshold_response,
        [*lcp_inputs, *vessel_inputs],
        figure_paths,
    )
    for path in figure_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
