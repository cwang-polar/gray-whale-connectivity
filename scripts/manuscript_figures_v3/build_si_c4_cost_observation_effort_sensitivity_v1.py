#!/usr/bin/env python3
"""Build the strict-effort cost-observation sensitivity candidate figure."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageStat
from pyproj import Transformer
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "manuscript_figures"))

import build_figure1_cee_v0 as figure1_base

OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "c4_cost_observation_effort_sensitivity_v1"
)
CANDIDATE_DIR = OUT_ROOT / "candidates"
SOURCE_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"
STEM = "gray_whale_si_c4_cost_observation_effort_sensitivity_candidate_v1"

GRID_PATH = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
EFFORT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
)
RECORDS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
)
ANNUAL_REFERENCE = (
    ROOT
    / "outputs"
    / "pelagic_prey_phase8m"
    / "figures"
    / "phase8m_asamm_behavior_summary_source_data_v1.csv"
)
NO_COVERAGE_REFERENCE = (
    ROOT
    / "outputs"
    / "annual_prey_sensitivity"
    / "figures"
    / "effort_footprint_sensitivity"
    / "gray_whale_asamm_effort_footprint_sensitivity_v4.csv"
)
LEGACY_COST_REFERENCE = (
    ROOT
    / "outputs"
    / "manuscript_figures"
    / "supplementary_cost_observation_overlay"
    / "gray_whale_cost_observation_cost_long_v0.csv"
)
LAND_PATH = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

PREY_VARIABLE = "total_kJm2_frozen"
MODELED_YEARS = list(range(2011, 2020))
CONTEXT_YEARS = list(range(2011, 2022))
STABLE_THRESHOLDS = [7, 8, 9]
EXPECTED_RECORDS = 1_179
EXPECTED_REPORTED_WHALES = 2_226.0
EXPECTED_EFFORT_UNION = 1_220
EXPECTED_STABLE_SIZES = {7: 105, 8: 52, 9: 19}
DBO_COLOR = figure1_base.PALETTE["dbo"]
DBO_LABEL_POSITIONS = {
    "DBO1": (-174.6, 62.8),
    "DBO2": (-169.2, 64.7),
    "DBO3": (-168.9, 67.7),
    "DBO4": (-162.5, 71.25),
    "DBO5": (-157.35, 71.45),
}

SCENARIO_ORDER = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIOS = {
    "prey_only": {
        "label": "Prey",
        "path": ROOT / "outputs" / "lcp_v2" / "prey_only_brian_C3_route_summary_v2.csv",
        "years": MODELED_YEARS,
        "color": "#3B6EA8",
        "marker": "o",
        "linestyle": "-",
    },
    "prey_ice": {
        "label": "Prey + ice",
        "path": ROOT / "outputs" / "lcp_v2" / "prey_ice_brian_C3_route_summary_v2.csv",
        "years": MODELED_YEARS,
        "color": "#3A8D6D",
        "marker": "s",
        "linestyle": "-",
    },
    "prey_vessel": {
        "label": "Prey + vessel",
        "path": (
            ROOT
            / "outputs"
            / "prey_vessel_phase8k"
            / "lcp"
            / "prey_vessel_brian_C3_route_summary_v2.csv"
        ),
        "years": list(range(2015, 2020)),
        "color": "#D9822B",
        "marker": "^",
        "linestyle": (0, (4, 2)),
    },
    "prey_ice_vessel": {
        "label": "Prey + ice + vessel",
        "path": ROOT / "outputs" / "lcp_v2" / "prey_ice_ship_brian_C3_route_summary_v2.csv",
        "years": list(range(2015, 2020)),
        "color": "#A85A91",
        "marker": "D",
        "linestyle": (0, (4, 2)),
    },
}

FOOTPRINT_ORDER = ["full", "stable_7", "stable_8", "stable_9"]
FOOTPRINTS = {
    "full": {
        "label": "Full annual footprint",
        "short_label": "Full annual",
        "color": "#5E6266",
        "marker": "o",
    },
    "stable_7": {
        "label": "Surveyed in at least 7 of 9 years (105 cells)",
        "short_label": "At least 7/9 years",
        "color": "#3A8D6D",
        "marker": "s",
    },
    "stable_8": {
        "label": "Surveyed in at least 8 of 9 years (52 cells)",
        "short_label": "At least 8/9 years",
        "color": "#3B6EA8",
        "marker": "^",
    },
    "stable_9": {
        "label": "Surveyed in all 9 years (19 cells)",
        "short_label": "All 9 years",
        "color": "#A85A91",
        "marker": "D",
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.1,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_costs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario in SCENARIO_ORDER:
        config = SCENARIOS[scenario]
        source = pd.read_csv(config["path"])
        subset = source.loc[
            source["prey_variable"].eq(PREY_VARIABLE)
            & source["target_year"].isin(config["years"])
        ].copy()
        expected_years = set(config["years"])
        if set(subset["target_year"].astype(int)) != expected_years:
            raise ValueError(f"Incomplete modeled years for {scenario}")
        if not (
            subset["attempted_path_count"].eq(1_600)
            & subset["successful_path_count"].eq(1_600)
            & subset["failed_path_count"].eq(0)
        ).all():
            raise ValueError(f"Path-completion failure for {scenario}")
        if subset["warning_flags"].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(f"Non-empty warning flags for {scenario}")
        subset["scenario"] = scenario
        subset["scenario_label"] = config["label"]
        frames.append(
            subset[
                [
                    "scenario",
                    "scenario_label",
                    "target_year",
                    "mean_path_cost",
                    "attempted_path_count",
                    "successful_path_count",
                    "failed_path_count",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 28:
        raise ValueError(f"Expected 28 scenario-year rows, found {len(result)}")
    return result


def verify_legacy_costs(costs: pd.DataFrame) -> float:
    legacy = pd.read_csv(LEGACY_COST_REFERENCE)
    legacy_map = {
        "Prey-only": "prey_only",
        "Prey+ice": "prey_ice",
        "Prey+ice+vessel": "prey_ice_vessel",
    }
    legacy["scenario"] = legacy["scenario"].map(legacy_map)
    check = costs.merge(
        legacy[["scenario", "year", "mean"]],
        left_on=["scenario", "target_year"],
        right_on=["scenario", "year"],
        how="inner",
        validate="one_to_one",
    )
    if len(check) != 23:
        raise ValueError(f"Expected 23 legacy cost rows, found {len(check)}")
    error = float(np.max(np.abs(check["mean_path_cost"] - check["mean"])))
    if error > 1e-9:
        raise ValueError(f"Modeled costs differ from legacy figure by {error}")
    return error


def load_observations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(GRID_PATH, dtype={"grid_id": str})
    effort = pd.read_csv(EFFORT_PATH, dtype={"grid_id": str})
    effort = effort.loc[
        effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
        & effort["year"].isin(MODELED_YEARS),
        ["year", "grid_id"],
    ].drop_duplicates()
    records = pd.read_csv(RECORDS_PATH, dtype={"nearest_grid_id": str})
    records = records.loc[
        records["use_for_C3_validation"].astype(bool)
        & records["month"].isin([6, 7])
        & records["year"].isin(MODELED_YEARS)
    ].copy()

    if len(grid) != 2_839 or grid["grid_id"].duplicated().any():
        raise ValueError("Unexpected regional ocean-grid inventory")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} sightings, found {len(records)}")
    if not np.isclose(records["count_or_best_count"].sum(), EXPECTED_REPORTED_WHALES):
        raise ValueError("Unexpected summed reported-whale count")
    annual_effort = effort.groupby("year")["grid_id"].nunique()
    if annual_effort.min() != 318 or annual_effort.max() != 634:
        raise ValueError("Strict-effort annual range does not reproduce 318-634 cells")
    effort_union = effort["grid_id"].nunique()
    if effort_union != EXPECTED_EFFORT_UNION:
        raise ValueError(f"Expected effort union of {EXPECTED_EFFORT_UNION}, found {effort_union}")

    record_effort_check = records.merge(
        effort.assign(in_strict_effort=True),
        left_on=["year", "nearest_grid_id"],
        right_on=["year", "grid_id"],
        how="left",
    )
    if not record_effort_check["in_strict_effort"].fillna(False).all():
        raise ValueError("At least one retained sighting lies outside its annual strict-effort footprint")

    annual_reference = pd.read_csv(ANNUAL_REFERENCE)
    record_counts = records.groupby("year").size()
    whale_counts = records.groupby("year")["count_or_best_count"].sum()
    expected_records = annual_reference.set_index("year")["record_count"].astype(int)
    expected_whales = annual_reference.set_index("year")["descriptive_observed_whale_sum"].astype(float)
    if not record_counts.equals(expected_records):
        raise ValueError("Annual sighting-record counts differ from the current ASAMM summary")
    if not np.allclose(whale_counts.loc[expected_whales.index], expected_whales):
        raise ValueError("Annual reported-whale counts differ from the current ASAMM summary")

    return grid, effort, records


def build_effort_masks(grid: pd.DataFrame, effort: pd.DataFrame) -> pd.DataFrame:
    years_surveyed = effort.groupby("grid_id")["year"].nunique().rename("years_surveyed")
    masks = grid[["grid_id", "x", "y", "lon", "lat"]].merge(
        years_surveyed, left_on="grid_id", right_index=True, how="left"
    )
    masks["years_surveyed"] = masks["years_surveyed"].fillna(0).astype(int)
    masks["in_any_strict_effort_2011_2019"] = masks["years_surveyed"].gt(0)
    for threshold in STABLE_THRESHOLDS:
        masks[f"in_stable_{threshold}_of_9"] = masks["years_surveyed"].ge(threshold)
        actual = int(masks[f"in_stable_{threshold}_of_9"].sum())
        if actual != EXPECTED_STABLE_SIZES[threshold]:
            raise ValueError(
                f"Expected {EXPECTED_STABLE_SIZES[threshold]} cells at {threshold}/9, found {actual}"
            )
    return masks


def load_dbo_reference_areas() -> gpd.GeoDataFrame:
    dbo = figure1_base.make_dbo_reference_polygons()
    expected = set(DBO_LABEL_POSITIONS)
    if set(dbo["name"]) != expected:
        raise ValueError(f"Expected DBO1-DBO5 reference areas, found {set(dbo['name'])}")
    return dbo


def summarize_effort_sensitivity(
    effort: pd.DataFrame, records: pd.DataFrame, masks: pd.DataFrame
) -> pd.DataFrame:
    mask_sets = {
        "full": set(masks.loc[masks["in_any_strict_effort_2011_2019"], "grid_id"]),
        **{
            f"stable_{threshold}": set(
                masks.loc[masks[f"in_stable_{threshold}_of_9"], "grid_id"]
            )
            for threshold in STABLE_THRESHOLDS
        },
    }
    rows: list[dict[str, object]] = []
    for footprint in FOOTPRINT_ORDER:
        config = FOOTPRINTS[footprint]
        fixed_mask = mask_sets[footprint]
        for year in CONTEXT_YEARS:
            if year not in MODELED_YEARS:
                rows.append(
                    {
                        "footprint": footprint,
                        "footprint_label": config["label"],
                        "year": year,
                        "fixed_mask_cells": len(fixed_mask),
                        "effort_covered_cells": np.nan,
                        "sighting_records": np.nan,
                        "reported_whales": np.nan,
                        "whales_per_effort_covered_cell": np.nan,
                        "has_june_july_coverage": False,
                    }
                )
                continue

            annual_effort = set(effort.loc[effort["year"].eq(year), "grid_id"])
            if footprint == "full":
                annual_mask = annual_effort
                annual_records = records.loc[records["year"].eq(year)]
            else:
                annual_mask = annual_effort & fixed_mask
                annual_records = records.loc[
                    records["year"].eq(year) & records["nearest_grid_id"].isin(fixed_mask)
                ]
            effort_count = len(annual_mask)
            rows.append(
                {
                    "footprint": footprint,
                    "footprint_label": config["label"],
                    "year": year,
                    "fixed_mask_cells": len(fixed_mask),
                    "effort_covered_cells": effort_count,
                    "sighting_records": len(annual_records),
                    "reported_whales": float(annual_records["count_or_best_count"].sum()),
                    "whales_per_effort_covered_cell": (
                        float(annual_records["count_or_best_count"].sum()) / effort_count
                        if effort_count
                        else np.nan
                    ),
                    "has_june_july_coverage": bool(effort_count),
                }
            )
    result = pd.DataFrame(rows)
    full = result.loc[result["footprint"].eq("full") & result["year"].isin(MODELED_YEARS)]
    if int(full["sighting_records"].sum()) != EXPECTED_RECORDS:
        raise ValueError("Full-footprint records do not sum to 1,179")
    if not np.isclose(full["reported_whales"].sum(), EXPECTED_REPORTED_WHALES):
        raise ValueError("Full-footprint reported whales do not sum to 2,226")
    return result


def verify_no_coverage_context() -> None:
    reference = pd.read_csv(NO_COVERAGE_REFERENCE)
    check = reference.loc[
        reference["case_id"].eq("full") & reference["target_year"].isin([2020, 2021])
    ]
    if len(check) != 2:
        raise ValueError("Missing 2020-2021 no-coverage audit rows")
    if not (
        check["effort_covered_cells_in_case"].eq(0)
        & check["observed_record_count"].eq(0)
        & check["observed_whale_count"].eq(0)
    ).all():
        raise ValueError("The ASAMM archive contains unexpected June-July coverage after 2019")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.5, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_costs(ax: plt.Axes, costs: pd.DataFrame) -> None:
    for scenario in SCENARIO_ORDER:
        config = SCENARIOS[scenario]
        subset = costs.loc[costs["scenario"].eq(scenario)].sort_values("target_year")
        ax.plot(
            subset["target_year"],
            subset["mean_path_cost"],
            color=config["color"],
            marker=config["marker"],
            linestyle=config["linestyle"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=config["label"],
            zorder=3,
        )
    style_axis(ax)
    ax.set_xlim(2010.75, 2019.25)
    ax.set_xticks(MODELED_YEARS)
    ax.set_ylabel("Mean modeled path cost")
    ax.set_xlabel("Year")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    add_panel_label(ax, "A", x=-0.047)


def plot_observation_panel(
    ax: plt.Axes, summary: pd.DataFrame, value: str, ylabel: str, panel: str
) -> None:
    for footprint in FOOTPRINT_ORDER:
        config = FOOTPRINTS[footprint]
        subset = summary.loc[summary["footprint"].eq(footprint)].sort_values("year")
        ax.plot(
            subset["year"],
            subset[value],
            color=config["color"],
            marker=config["marker"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            linewidth=1.25,
            label=config["short_label"],
            zorder=3,
        )
    ax.axvspan(2019.5, 2021.5, color="#F0F0F0", zorder=0)
    style_axis(ax)
    ax.set_xlim(2010.7, 2021.3)
    ax.set_xticks(CONTEXT_YEARS)
    ax.set_xticklabels(CONTEXT_YEARS, rotation=48, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_ylim(bottom=0)
    add_panel_label(ax, panel, x=-0.17)


def project_point(transformer: Transformer, lon: float, lat: float) -> tuple[float, float]:
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def plot_effort_map(
    ax: plt.Axes,
    grid: pd.DataFrame,
    masks: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
) -> None:
    dbo_projected = dbo.to_crs("EPSG:3338")
    dbo_xmin, dbo_ymin, dbo_xmax, dbo_ymax = dbo_projected.total_bounds
    xmin = min(float(grid["x"].min()), float(dbo_xmin))
    xmax = max(float(grid["x"].max()), float(dbo_xmax))
    ymin = min(float(grid["y"].min()), float(dbo_ymin))
    ymax = max(float(grid["y"].max()), float(dbo_ymax))
    xpad, ypad = 18_000, 18_000
    land = gpd.read_file(LAND_PATH).to_crs("EPSG:3338")
    land = land.clip(box(xmin - xpad, ymin - ypad, xmax + xpad, ymax + ypad))
    ax.set_facecolor("#EDF4F6")
    land.plot(ax=ax, color="#D8D3CA", edgecolor="white", linewidth=0.25, zorder=1)

    ax.scatter(
        grid["x"],
        grid["y"],
        s=1.1,
        color="#D7DEE2",
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    any_effort = masks.loc[masks["in_any_strict_effort_2011_2019"]]
    ax.scatter(
        any_effort["x"],
        any_effort["y"],
        s=2.0,
        color="#AEB8BD",
        linewidth=0,
        rasterized=True,
        zorder=3,
    )
    for threshold, size in [(7, 5.2), (8, 8.0), (9, 12.0)]:
        footprint = FOOTPRINTS[f"stable_{threshold}"]
        subset = masks.loc[masks[f"in_stable_{threshold}_of_9"]]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=size,
            marker=footprint["marker"],
            color=footprint["color"],
            edgecolor="white",
            linewidth=0.25,
            rasterized=True,
            zorder=3 + threshold,
        )

    dbo_projected.boundary.plot(
        ax=ax,
        color=DBO_COLOR,
        linewidth=0.72,
        linestyle=(0, (4.0, 2.2)),
        zorder=16,
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    for label, (lon, lat) in DBO_LABEL_POSITIONS.items():
        x, y = project_point(transformer, lon, lat)
        dbo_label = ax.text(
            x,
            y,
            label,
            fontsize=4.9,
            color=DBO_COLOR,
            ha="center",
            va="center",
            zorder=18,
        )
        dbo_label.set_path_effects(
            [path_effects.withStroke(linewidth=1.5, foreground="white")]
        )

    labels = [
        ("Russia", -173.8, 65.2, "#5E6266", 0),
        ("Alaska", -159.9, 69.0, "#5E6266", 0),
        ("St Lawrence\nIsland", -170.4, 63.5, "#5E6266", 0),
        ("Bering\nStrait", -168.6, 65.8, "#5E6266", 0),
        ("Chukchi Sea", -166.0, 69.7, "#5E6266", 0),
    ]
    for text, lon, lat, color, _ in labels:
        x, y = project_point(transformer, lon, lat)
        ax.text(x, y, text, fontsize=5.7, color=color, ha="center", va="center", zorder=20)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=3.2, markerfacecolor="#AEB8BD", markeredgewidth=0, label="Any strict effort"),
        *[
            Line2D(
                [0],
                [0],
                marker=FOOTPRINTS[f"stable_{threshold}"]["marker"],
                linestyle="",
                markersize=4.2,
                markerfacecolor=FOOTPRINTS[f"stable_{threshold}"]["color"],
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=(
                    f"At least {threshold}/9 ({EXPECTED_STABLE_SIZES[threshold]})"
                    if threshold < 9
                    else f"All 9 years ({EXPECTED_STABLE_SIZES[threshold]})"
                ),
            )
            for threshold in STABLE_THRESHOLDS
        ],
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        frameon=True,
        facecolor="white",
        edgecolor="#C7CDD0",
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.3,
        handletextpad=0.35,
        fontsize=5.0,
    )
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    add_panel_label(ax, "D", x=-0.08)


def draw_figure(
    costs: pd.DataFrame,
    summary: pd.DataFrame,
    grid: pd.DataFrame,
    masks: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 5.0), facecolor="white")
    grid_spec = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.86, 1.14],
        width_ratios=[1.05, 1.05, 0.95],
        left=0.075,
        right=0.99,
        bottom=0.17,
        top=0.96,
        hspace=0.37,
        wspace=0.31,
    )
    ax_a = fig.add_subplot(grid_spec[0, :])
    ax_b = fig.add_subplot(grid_spec[1, 0])
    ax_c = fig.add_subplot(grid_spec[1, 1])
    ax_d = fig.add_subplot(grid_spec[1, 2])

    plot_costs(ax_a, costs)
    plot_observation_panel(ax_b, summary, "reported_whales", "Reported whales", "B")
    plot_observation_panel(
        ax_c,
        summary,
        "whales_per_effort_covered_cell",
        "Whales per effort-covered cell",
        "C",
    )
    plot_effort_map(ax_d, grid, masks, dbo)

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.40, 0.025),
        ncol=4,
        frameon=False,
        fontsize=6.0,
        handlelength=1.5,
        columnspacing=0.85,
        handletextpad=0.3,
    )
    return fig


def save_figure(fig: plt.Figure) -> dict[str, Path]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": CANDIDATE_DIR / f"{STEM}.pdf",
        "png": CANDIDATE_DIR / f"{STEM}.png",
        "svg": CANDIDATE_DIR / f"{STEM}.svg",
        "tiff": CANDIDATE_DIR / f"{STEM}.tiff",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def image_nonblank(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(ImageStat.Stat(image.convert("L")).stddev[0] > 1.0)


def write_qa(
    paths: dict[str, Path],
    costs: pd.DataFrame,
    summary: pd.DataFrame,
    masks: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
    legacy_cost_error: float,
) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for file_type, path in paths.items():
        row: dict[str, object] = {
            "check": "export",
            "file_type": file_type,
            "file": path.name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        if file_type in {"png", "tiff"} and path.exists():
            with Image.open(path) as image:
                row.update(
                    {
                        "width_px": image.width,
                        "height_px": image.height,
                        "nonblank": image_nonblank(path),
                    }
                )
        rows.append(row)

    modeled = summary.loc[summary["year"].isin(MODELED_YEARS)]
    rows.extend(
        [
            {"check": "scenario_year_rows", "value": len(costs), "expected": 28, "passed": len(costs) == 28},
            {"check": "retained_sighting_records", "value": int(modeled.loc[modeled["footprint"].eq("full"), "sighting_records"].sum()), "expected": EXPECTED_RECORDS, "passed": int(modeled.loc[modeled["footprint"].eq("full"), "sighting_records"].sum()) == EXPECTED_RECORDS},
            {"check": "reported_whale_sum", "value": float(modeled.loc[modeled["footprint"].eq("full"), "reported_whales"].sum()), "expected": EXPECTED_REPORTED_WHALES, "passed": np.isclose(modeled.loc[modeled["footprint"].eq("full"), "reported_whales"].sum(), EXPECTED_REPORTED_WHALES)},
            {"check": "strict_effort_union_cells", "value": int(masks["in_any_strict_effort_2011_2019"].sum()), "expected": EXPECTED_EFFORT_UNION, "passed": int(masks["in_any_strict_effort_2011_2019"].sum()) == EXPECTED_EFFORT_UNION},
            {"check": "stable_7_of_9_cells", "value": int(masks["in_stable_7_of_9"].sum()), "expected": 105, "passed": int(masks["in_stable_7_of_9"].sum()) == 105},
            {"check": "stable_8_of_9_cells", "value": int(masks["in_stable_8_of_9"].sum()), "expected": 52, "passed": int(masks["in_stable_8_of_9"].sum()) == 52},
            {"check": "stable_9_of_9_cells", "value": int(masks["in_stable_9_of_9"].sum()), "expected": 19, "passed": int(masks["in_stable_9_of_9"].sum()) == 19},
            {"check": "dbo_reference_areas", "value": len(dbo), "expected": 5, "passed": len(dbo) == 5},
            {"check": "legacy_cost_max_error", "value": legacy_cost_error, "expected": "<=1e-9", "passed": legacy_cost_error <= 1e-9},
            {"check": "context_years_blank", "value": int(summary.loc[summary["year"].isin([2020, 2021]), "has_june_july_coverage"].sum()), "expected": 0, "passed": not summary.loc[summary["year"].isin([2020, 2021]), "has_june_july_coverage"].any()},
        ]
    )
    pd.DataFrame(rows).to_csv(
        QA_DIR / "gray_whale_si_c4_cost_observation_effort_sensitivity_qa_v1.csv",
        index=False,
    )


def format_markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_metadata(summary: pd.DataFrame, legacy_cost_error: float) -> None:
    modeled = summary.loc[summary["year"].isin(MODELED_YEARS)].copy()
    late = modeled.loc[modeled["year"].isin([2017, 2018, 2019])]
    late_table = late.pivot(
        index="footprint_label", columns="year", values="whales_per_effort_covered_cell"
    ).reset_index()
    lines = [
        "# Cost-observation and strict-effort sensitivity candidate v1",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: temporal comparison between modeled corridor cost and observed whale summaries must be interpreted together with annual survey coverage and stable-footprint sensitivity.",
        "- Evidence chain: panel A shows the four modeled cost trajectories; panels B and C show observed counts and effort-conditioned rates under four footprint definitions; panel D shows where those definitions occur.",
        "- DBO1-DBO5 outlines in panel D reproduce the geographic reference areas used in Figure 1 and do not enter any calculation.",
        "- Archetype: asymmetric quantitative composite with one temporal overview and three survey-footprint diagnostics.",
        "- Backend: Python/matplotlib only.",
        "",
        "## Corrected effort definition",
        "",
        "All observation summaries use the current strict-effort event classification. The earlier 99-cell display was not reused because it had treated all geolocated ASAMM event rows as effort. Under the strict definition, 1,220 study-region ocean cells were surveyed at least once during 2011-2019; 105, 52 and 19 cells were surveyed in at least seven, at least eight and all nine modeled years, respectively.",
        "",
        "## 2017-2019 rates",
        "",
        format_markdown_table(late_table),
        "",
        "The 105-cell at-least-7/9-years mask retains a 2017 peak followed by lower 2018-2019 values. The two stricter masks are much smaller and do not yield a monotonic 2017-2019 trajectory, so the figure presents footprint sensitivity rather than claiming footprint-invariant change.",
        "",
        "## Reproduction and interpretation checks",
        "",
        f"- Maximum modeled-cost difference from the three-scenario legacy display: {legacy_cost_error:.3g}.",
        "- Reported whales are sums of recorded group counts, not abundance estimates.",
        "- Whales per effort-covered cell does not correct for flight distance, detection probability or unequal effort within cells.",
        "- The 2020-2021 blank interval indicates no eligible June-July study-region effort in the available archive, not zero whale use.",
    ]
    (OUT_ROOT / "gray_whale_si_c4_cost_observation_effort_sensitivity_metadata_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    for directory in [CANDIDATE_DIR, SOURCE_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    costs = load_costs()
    legacy_cost_error = verify_legacy_costs(costs)
    grid, effort, records = load_observations()
    masks = build_effort_masks(grid, effort)
    dbo = load_dbo_reference_areas()
    summary = summarize_effort_sensitivity(effort, records, masks)
    verify_no_coverage_context()

    costs.to_csv(SOURCE_DIR / "c4_modeled_path_cost_source_data_v1.csv", index=False)
    summary.to_csv(SOURCE_DIR / "c4_strict_effort_sensitivity_source_data_v1.csv", index=False)
    masks.to_csv(SOURCE_DIR / "c4_strict_effort_masks_source_data_v1.csv", index=False)
    dbo_source = dbo.copy()
    dbo_source["geometry_wkt"] = dbo_source.geometry.to_wkt()
    dbo_source.drop(columns="geometry").to_csv(
        SOURCE_DIR / "c4_dbo_reference_areas_figure1_matched_v1.csv", index=False
    )

    figure = draw_figure(costs, summary, grid, masks, dbo)
    paths = save_figure(figure)
    plt.close(figure)
    write_qa(paths, costs, summary, masks, dbo, legacy_cost_error)
    write_metadata(summary, legacy_cost_error)

    modeled = summary.loc[summary["year"].isin([2017, 2018, 2019])]
    print(
        modeled.pivot(
            index="footprint_label",
            columns="year",
            values="whales_per_effort_covered_cell",
        ).round(3).to_string()
    )
    print(f"\nCandidate: {paths['pdf'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
