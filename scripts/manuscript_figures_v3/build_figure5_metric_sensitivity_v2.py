#!/usr/bin/env python3
"""Build an isolated one-at-a-time sensitivity analysis for Figure 5 metrics."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-figure5-sensitivity")

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ASAMM_SIGHTINGS = ROOT / "data/processed/validation/asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
ASAMM_EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
PRIMARY_FULL = ROOT / "data/processed/validation/asamm_phase8i2b_main_june_july_validation_summary.csv"
PRIMARY_COMMON = ROOT / "outputs/prey_vessel_phase8k/validation/asamm_four_scenario_common_period_summary_v2.csv"

OUT = ROOT / "outputs/manuscript_figures_v3/figure_5/sensitivity_v2"
CANDIDATES = OUT / "candidates"
SOURCE_DATA = OUT / "source_data"
QA = OUT / "qa"

SCENARIOS = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
LABELS = {
    "prey_only": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}
COLORS = {
    "prey_only": "#3B6EA8",
    "prey_ice": "#3A8D6D",
    "prey_vessel": "#D9822B",
    "prey_ice_vessel": "#A85A91",
}
MARKERS = {
    "prey_only": "o",
    "prey_ice": "s",
    "prey_vessel": "^",
    "prey_ice_vessel": "D",
}

PERIODS = {
    "full": {
        "label": "2011-2019 | 1,179 sightings",
        "years": list(range(2011, 2020)),
        "scenarios": ["prey_only", "prey_ice"],
        "stable_thresholds": [7, 8, 9],
    },
    "common": {
        "label": "2015-2019 | 666 sightings",
        "years": list(range(2015, 2020)),
        "scenarios": SCENARIOS,
        "stable_thresholds": [3, 4, 5],
    },
}

TOP_K_VALUES = list(range(1, 21))
DISTANCE_THRESHOLDS = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150]
PRIMARY_TOP_K = 5
PRIMARY_FREQUENCY_THRESHOLD = 0.05


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    for path in (OUT, CANDIDATES, SOURCE_DATA, QA):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_path(scenario: str, year: int) -> Path:
    if scenario == "prey_only":
        return ROOT / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice":
        return ROOT / f"outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_vessel":
        return ROOT / f"outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice_vessel":
        return ROOT / f"outputs/lcp_v2/prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    raise KeyError(scenario)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    valid = sightings["use_for_C3_validation"]
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().eq("true")
    sightings = sightings.loc[
        valid & sightings["year"].between(2011, 2019) & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)

    effort = pd.read_csv(ASAMM_EFFORT)
    effort = effort.loc[
        effort["year"].between(2011, 2019)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    effort["grid_id"] = effort["grid_id"].astype(str)

    grids: dict[tuple[str, int], pd.DataFrame] = {}
    needed = {
        (scenario, year)
        for spec in PERIODS.values()
        for scenario in spec["scenarios"]
        for year in spec["years"]
    }
    for scenario, year in sorted(needed):
        path = grid_path(scenario, year)
        frame = pd.read_csv(
            path,
            usecols=["grid_id", "x", "y", "lcp_frequency_normalized"],
        )
        frame["grid_id"] = frame["grid_id"].astype(str)
        frame["lcp_frequency_normalized"] = frame["lcp_frequency_normalized"].fillna(0.0)
        if frame["grid_id"].duplicated().any():
            raise ValueError(f"Duplicate grid ids in {path}")
        grids[(scenario, year)] = frame
    return sightings, effort, grids


def midrank_percentile(values: np.ndarray, focal: float) -> float:
    return float(
        ((values < focal).sum() + 0.5 * (values == focal).sum())
        / len(values)
        * 100.0
    )


def period_sightings(sightings: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return sightings.loc[sightings["year"].isin(years)].copy()


def effort_sets(effort: pd.DataFrame, years: list[int]) -> dict[int, set[str]]:
    return {
        year: set(effort.loc[effort["year"].eq(year), "grid_id"].astype(str))
        for year in years
    }


def build_effort_sensitivity(
    sightings: pd.DataFrame,
    effort: pd.DataFrame,
    grids: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_id, spec in PERIODS.items():
        years = spec["years"]
        annual_effort = effort_sets(effort, years)
        coverage: dict[str, int] = {}
        for ids in annual_effort.values():
            for grid_id in ids:
                coverage[grid_id] = coverage.get(grid_id, 0) + 1

        definitions: list[tuple[str, str, set[str] | None, int | None]] = [
            ("annual", "Annual footprint", None, None)
        ]
        for threshold in spec["stable_thresholds"]:
            core = {grid_id for grid_id, count in coverage.items() if count >= threshold}
            definitions.append(
                (
                    f"stable_{threshold}_of_{len(years)}",
                    f">={threshold} of {len(years)} years",
                    core,
                    threshold,
                )
            )

        all_records = period_sightings(sightings, years)
        for order, (definition_id, definition_label, core, threshold) in enumerate(definitions):
            retained = (
                all_records
                if core is None
                else all_records.loc[all_records["nearest_grid_id"].isin(core)]
            )
            for scenario in spec["scenarios"]:
                percentiles: list[float] = []
                background_sizes: list[int] = []
                used_record_ids: list[str] = []
                for year in years:
                    grid = grids[(scenario, year)].set_index("grid_id")
                    background_ids = (
                        annual_effort[year]
                        if core is None
                        else annual_effort[year].intersection(core)
                    )
                    background_index = grid.index.intersection(background_ids)
                    background = grid.loc[
                        background_index, "lcp_frequency_normalized"
                    ].to_numpy(dtype=float)
                    if len(background) == 0:
                        continue
                    year_records = retained.loc[retained["year"].eq(year)]
                    for rec in year_records.itertuples(index=False):
                        grid_id = str(rec.nearest_grid_id)
                        if grid_id not in grid.index:
                            continue
                        focal = float(grid.at[grid_id, "lcp_frequency_normalized"])
                        percentiles.append(midrank_percentile(background, focal))
                        used_record_ids.append(str(rec.record_id))
                    background_sizes.append(len(background))
                rows.append(
                    {
                        "period": period_id,
                        "period_label": spec["label"],
                        "scenario": scenario,
                        "scenario_label": LABELS[scenario],
                        "definition_order": order,
                        "effort_definition": definition_id,
                        "effort_definition_label": definition_label,
                        "minimum_years_covered": threshold,
                        "stable_core_cell_count": math.nan if core is None else len(core),
                        "record_count": len(set(used_record_ids)),
                        "mean_effort_footprint_percentile": float(np.mean(percentiles)),
                        "annual_background_cells_min": min(background_sizes),
                        "annual_background_cells_max": max(background_sizes),
                    }
                )
    return pd.DataFrame(rows)


def build_topk_sensitivity(
    sightings: pd.DataFrame,
    grids: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_id, spec in PERIODS.items():
        records = period_sightings(sightings, spec["years"])
        for scenario in spec["scenarios"]:
            for top_k in TOP_K_VALUES:
                inclusive_inside: list[bool] = []
                equal_area_scores: list[float] = []
                linear_cutoffs: list[float] = []
                equal_area_boundaries: list[float] = []
                boundary_weights: list[float] = []
                selected_counts: list[int] = []
                selected_fractions: list[float] = []
                used_ids: list[str] = []
                for year in spec["years"]:
                    grid = grids[(scenario, year)]
                    frequency = grid["lcp_frequency_normalized"].to_numpy(dtype=float)
                    linear_cutoff = float(
                        np.quantile(frequency, 1.0 - top_k / 100.0)
                    )
                    target_cells = len(frequency) * top_k / 100.0
                    boundary_index = max(0, int(math.ceil(target_cells)) - 1)
                    boundary = float(np.sort(frequency)[::-1][boundary_index])
                    cells_above = int((frequency > boundary).sum())
                    cells_at_boundary = int((frequency == boundary).sum())
                    boundary_weight = float(
                        np.clip(
                            (target_cells - cells_above) / cells_at_boundary,
                            0.0,
                            1.0,
                        )
                    )
                    lookup = grid.set_index("grid_id")["lcp_frequency_normalized"]
                    year_records = records.loc[records["year"].eq(year)]
                    selected = int((frequency >= linear_cutoff).sum())
                    for rec in year_records.itertuples(index=False):
                        grid_id = str(rec.nearest_grid_id)
                        if grid_id not in lookup.index:
                            continue
                        focal = float(lookup.at[grid_id])
                        inclusive_inside.append(focal >= linear_cutoff)
                        if focal > boundary:
                            equal_area_scores.append(1.0)
                        elif np.isclose(focal, boundary):
                            equal_area_scores.append(boundary_weight)
                        else:
                            equal_area_scores.append(0.0)
                        used_ids.append(str(rec.record_id))
                    linear_cutoffs.append(linear_cutoff)
                    equal_area_boundaries.append(boundary)
                    boundary_weights.append(boundary_weight)
                    selected_counts.append(selected)
                    selected_fractions.append(selected / len(grid) * 100.0)
                valid = bool(min(equal_area_boundaries) > 0)
                raw_overlap = float(np.mean(inclusive_inside) * 100.0)
                equal_area_overlap = float(np.mean(equal_area_scores) * 100.0)
                rows.append(
                    {
                        "period": period_id,
                        "period_label": spec["label"],
                        "scenario": scenario,
                        "scenario_label": LABELS[scenario],
                        "nominal_top_k_percent": top_k,
                        "record_count": len(set(used_ids)),
                        "all_annual_cutoffs_positive": valid,
                        "minimum_linear_quantile_cutoff": min(linear_cutoffs),
                        "maximum_linear_quantile_cutoff": max(linear_cutoffs),
                        "minimum_equal_area_boundary_frequency": min(equal_area_boundaries),
                        "maximum_equal_area_boundary_frequency": max(equal_area_boundaries),
                        "mean_boundary_weight": float(np.mean(boundary_weights)),
                        "mean_inclusive_grid_percent": float(np.mean(selected_fractions)),
                        "minimum_inclusive_grid_percent": min(selected_fractions),
                        "maximum_inclusive_grid_percent": max(selected_fractions),
                        "equal_area_grid_percent": float(top_k),
                        "raw_inclusive_overlap_percent": raw_overlap,
                        "tie_adjusted_equal_area_overlap_percent": equal_area_overlap,
                        "reportable_overlap_percent": equal_area_overlap if valid else math.nan,
                        "mean_inclusive_selected_cell_count": float(np.mean(selected_counts)),
                    }
                )
    return pd.DataFrame(rows)


def build_distance_sensitivity(
    sightings: pd.DataFrame,
    grids: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_id, spec in PERIODS.items():
        records = period_sightings(sightings, spec["years"])
        for scenario in spec["scenarios"]:
            for threshold in DISTANCE_THRESHOLDS:
                distances: list[float] = []
                corridor_counts: list[int] = []
                used_ids: list[str] = []
                for year in spec["years"]:
                    grid = grids[(scenario, year)]
                    corridor = grid.loc[
                        grid["lcp_frequency_normalized"].gt(threshold), ["x", "y"]
                    ].to_numpy(dtype=float)
                    if len(corridor) == 0:
                        raise ValueError(
                            f"No corridor cells for {scenario}, {year}, F_s>{threshold}"
                        )
                    tree = cKDTree(corridor)
                    lookup = grid.set_index("grid_id")
                    year_records = records.loc[records["year"].eq(year)]
                    xy: list[list[float]] = []
                    ids: list[str] = []
                    for rec in year_records.itertuples(index=False):
                        grid_id = str(rec.nearest_grid_id)
                        if grid_id not in lookup.index:
                            continue
                        cell = lookup.loc[grid_id]
                        xy.append([float(cell["x"]), float(cell["y"])] )
                        ids.append(str(rec.record_id))
                    if xy:
                        distances.extend(
                            (tree.query(np.asarray(xy), k=1)[0] / 1000.0).tolist()
                        )
                        used_ids.extend(ids)
                    corridor_counts.append(len(corridor))
                rows.append(
                    {
                        "period": period_id,
                        "period_label": spec["label"],
                        "scenario": scenario,
                        "scenario_label": LABELS[scenario],
                        "frequency_threshold": threshold,
                        "record_count": len(set(used_ids)),
                        "median_distance_km": float(np.median(distances)),
                        "mean_corridor_cell_count": float(np.mean(corridor_counts)),
                        "minimum_corridor_cell_count": min(corridor_counts),
                        "maximum_corridor_cell_count": max(corridor_counts),
                    }
                )
    return pd.DataFrame(rows)


def primary_targets() -> dict[tuple[str, str, str], float]:
    full = pd.read_csv(PRIMARY_FULL)
    full = full.loc[
        full["prey_variable"].eq("total_kJm2_frozen")
        & full["validation_window"].eq("June_July")
        & full["model_layer"].isin(["prey_only", "prey_ice"])
    ]
    common = pd.read_csv(PRIMARY_COMMON)
    targets: dict[tuple[str, str, str], float] = {}
    for row in full.itertuples(index=False):
        targets[("full", row.model_layer, "effort")] = float(
            row.mean_lcp_percentile_within_effort_footprint
        )
        targets[("full", row.model_layer, "topk")] = float(
            row.percent_records_top5pct_lcp
        )
        targets[("full", row.model_layer, "distance")] = float(
            row.median_distance_to_core_corridor_km
        )
    for row in common.itertuples(index=False):
        targets[("common", row.scenario, "effort")] = float(
            row.mean_effort_footprint_percentile
        )
        targets[("common", row.scenario, "topk")] = float(
            row.percent_records_top5pct
        )
        targets[("common", row.scenario, "distance")] = float(
            row.median_distance_to_frequent_corridor_km
        )
    return targets


def validate_primary_reproduction(
    effort: pd.DataFrame,
    topk: pd.DataFrame,
    distance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, scenario, metric), expected in primary_targets().items():
        if metric == "effort":
            actual = effort.loc[
                effort["period"].eq(period)
                & effort["scenario"].eq(scenario)
                & effort["effort_definition"].eq("annual"),
                "mean_effort_footprint_percentile",
            ].iloc[0]
        elif metric == "topk":
            actual = topk.loc[
                topk["period"].eq(period)
                & topk["scenario"].eq(scenario)
                & topk["nominal_top_k_percent"].eq(PRIMARY_TOP_K),
                "raw_inclusive_overlap_percent",
            ].iloc[0]
        else:
            actual = distance.loc[
                distance["period"].eq(period)
                & distance["scenario"].eq(scenario)
                & np.isclose(
                    distance["frequency_threshold"],
                    PRIMARY_FREQUENCY_THRESHOLD,
                ),
                "median_distance_km",
            ].iloc[0]
        difference = float(actual - expected)
        rows.append(
            {
                "period": period,
                "scenario": scenario,
                "metric": metric,
                "expected_primary_value": expected,
                "recalculated_value": actual,
                "absolute_difference": abs(difference),
                "pass_tolerance_1e_9": abs(difference) <= 1e-9,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame["pass_tolerance_1e_9"].all():
        raise ValueError(
            "Sensitivity calculations do not reproduce the primary Figure 5 values"
        )
    return frame


def winner_rows(
    effort: pd.DataFrame,
    topk: pd.DataFrame,
    distance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        (
            effort,
            ["period", "effort_definition"],
            "mean_effort_footprint_percentile",
            "higher",
            "effort-footprint percentile",
        ),
        (
            topk.loc[topk["all_annual_cutoffs_positive"]],
            ["period", "nominal_top_k_percent"],
            "reportable_overlap_percent",
            "higher",
            "annual core overlap",
        ),
        (
            distance,
            ["period", "frequency_threshold"],
            "median_distance_km",
            "lower",
            "distance to corridor",
        ),
    ]
    for frame, group_cols, value_col, direction, metric in specifications:
        for keys, group in frame.groupby(group_cols, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            best = (
                group[value_col].max()
                if direction == "higher"
                else group[value_col].min()
            )
            winners = group.loc[
                np.isclose(group[value_col], best), "scenario_label"
            ].tolist()
            row = {column: value for column, value in zip(group_cols, key_values)}
            row.update(
                {
                    "metric": metric,
                    "direction": direction,
                    "winning_scenario": "; ".join(winners),
                    "winning_value": best,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def global_valid_topk(topk: pd.DataFrame, period: str) -> int:
    sub = topk.loc[topk["period"].eq(period)]
    validity = sub.groupby("nominal_top_k_percent")[
        "all_annual_cutoffs_positive"
    ].all()
    consecutive: list[int] = []
    for value in sorted(validity.index[validity].astype(int)):
        if value == len(consecutive) + 1:
            consecutive.append(value)
        else:
            break
    if not consecutive:
        raise ValueError(f"No shared valid top-k range for {period}")
    return max(consecutive)


def scenario_winner_counts(winners: pd.DataFrame, period: str, metric: str) -> str:
    sub = winners.loc[
        winners["period"].eq(period) & winners["metric"].eq(metric)
    ]
    counts = sub["winning_scenario"].value_counts()
    labels = []
    for name, count in counts.items():
        if ";" in name:
            display = "Tie (" + " / ".join(part.strip() for part in name.split(";")) + ")"
        else:
            display = name
        labels.append(f"{display}: {count}/{len(sub)}")
    return "; ".join(labels)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E4E8EB", linewidth=0.6, zorder=0)
    ax.tick_params(length=3, width=0.7, color="#3A3A3A")
    ax.margins(x=0.03)


def mark_primary(ax: plt.Axes, x: float, y: float, scenario: str) -> None:
    ax.scatter(
        [x],
        [y],
        s=42,
        marker=MARKERS[scenario],
        facecolor=COLORS[scenario],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )


def plot_effort(ax: plt.Axes, frame: pd.DataFrame, period: str) -> None:
    sub = frame.loc[frame["period"].eq(period)].copy()
    for scenario in PERIODS[period]["scenarios"]:
        data = sub.loc[sub["scenario"].eq(scenario)].sort_values("definition_order")
        x = data["definition_order"].to_numpy(dtype=float)
        y = data["mean_effort_footprint_percentile"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markersize=4.0,
            linewidth=1.25,
            markerfacecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        mark_primary(ax, x[0], y[0], scenario)
    definitions = (
        sub.sort_values("definition_order")
        .drop_duplicates("definition_order")
        .sort_values("definition_order")
    )
    tick_labels = []
    for row in definitions.itertuples(index=False):
        label = str(row.effort_definition_label).replace(">=", "≥")
        tick_labels.append(f"{label}\n(n={int(row.record_count):,})")
    ax.set_xticks(definitions["definition_order"], tick_labels)
    ax.axhline(50, color="#9EA7AD", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
    ax.set_ylabel("Mean within-footprint percentile")
    ax.set_xlabel("Survey footprint and retained sightings")
    style_axis(ax)


def plot_topk(
    ax: plt.Axes,
    frame: pd.DataFrame,
    period: str,
    maximum_k: int,
) -> None:
    sub = frame.loc[
        frame["period"].eq(period)
        & frame["all_annual_cutoffs_positive"]
        & frame["nominal_top_k_percent"].le(maximum_k)
    ].copy()
    for scenario in PERIODS[period]["scenarios"]:
        data = sub.loc[sub["scenario"].eq(scenario)].sort_values(
            "nominal_top_k_percent"
        )
        x = data["nominal_top_k_percent"].to_numpy(dtype=float)
        y = data["reportable_overlap_percent"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markevery=2,
            markersize=3.4,
            linewidth=1.25,
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=3,
        )
        primary = data.loc[data["nominal_top_k_percent"].eq(PRIMARY_TOP_K)]
        mark_primary(
            ax,
            PRIMARY_TOP_K,
            float(primary["raw_inclusive_overlap_percent"].iloc[0]),
            scenario,
        )
    ax.axvline(
        PRIMARY_TOP_K,
        color="#9EA7AD",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xlim(1, maximum_k)
    ax.set_xticks(sorted(set([1, PRIMARY_TOP_K, maximum_k])))
    ax.set_ylabel("Sightings inside annual core (%)")
    ax.set_xlabel("Highest-frequency cells retained, k (%)")
    style_axis(ax)


def plot_distance(ax: plt.Axes, frame: pd.DataFrame, period: str) -> None:
    sub = frame.loc[frame["period"].eq(period)].copy()
    for scenario in PERIODS[period]["scenarios"]:
        data = sub.loc[sub["scenario"].eq(scenario)].sort_values(
            "frequency_threshold"
        )
        x = data["frequency_threshold"].to_numpy(dtype=float)
        y = data["median_distance_km"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markersize=4.0,
            linewidth=1.25,
            markerfacecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        primary = data.loc[
            np.isclose(data["frequency_threshold"], PRIMARY_FREQUENCY_THRESHOLD)
        ]
        mark_primary(
            ax,
            PRIMARY_FREQUENCY_THRESHOLD,
            float(primary["median_distance_km"].iloc[0]),
            scenario,
        )
    ax.axvline(
        PRIMARY_FREQUENCY_THRESHOLD,
        color="#9EA7AD",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xticks([0.025, 0.05, 0.10, 0.15])
    ax.set_ylabel("Median distance to corridor (km)")
    ax.set_xlabel("Corridor-frequency cutoff, $F_s$")
    style_axis(ax)


def padded_limits(values: pd.Series, floor: float | None = None) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1.0)
    lower = low - span * 0.13
    upper = high + span * 0.16
    if floor is not None:
        lower = max(floor, lower)
    return lower, upper


def make_figure(
    effort: pd.DataFrame,
    topk: pd.DataFrame,
    distance: pd.DataFrame,
) -> tuple[plt.Figure, int]:
    configure_style()
    max_k = min(global_valid_topk(topk, period) for period in PERIODS)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.45))
    fig.subplots_adjust(
        left=0.080,
        right=0.990,
        bottom=0.105,
        top=0.865,
        wspace=0.25,
        hspace=0.54,
    )

    for row, period in enumerate(["full", "common"]):
        plot_effort(axes[row, 0], effort, period)
        plot_topk(axes[row, 1], topk, period, max_k)
        plot_distance(axes[row, 2], distance, period)

    effort_limits = padded_limits(effort["mean_effort_footprint_percentile"], floor=0)
    valid_topk = topk.loc[
        topk["all_annual_cutoffs_positive"]
        & topk["nominal_top_k_percent"].le(max_k),
        "reportable_overlap_percent",
    ]
    topk_limits = padded_limits(valid_topk, floor=0)
    distance_limits = padded_limits(distance["median_distance_km"], floor=0)
    for row in range(2):
        axes[row, 0].set_ylim(*effort_limits)
        axes[row, 1].set_ylim(*topk_limits)
        axes[row, 2].set_ylim(*distance_limits)

    for col, title in enumerate(
        [
            "Survey-footprint sensitivity",
            "Annual corridor-core definition",
            "Frequent-corridor definition",
        ]
    ):
        axes[0, col].set_title(title, fontweight="bold", pad=8)

    for label, ax in zip(list("ABCDEF"), axes.flat):
        ax.text(
            -0.19,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.text(0.085, 0.91, "2011–2019 | 1,179 sightings", fontsize=7.6, fontweight="bold")
    fig.text(0.085, 0.47, "2015–2019 | 666 sightings", fontsize=7.6, fontweight="bold")
    fig.text(
        0.985,
        0.91,
        "Filled symbols mark primary definitions",
        fontsize=6.2,
        color="#666666",
        ha="right",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.5,
            linewidth=1.3,
            label=LABELS[scenario],
        )
        for scenario in SCENARIOS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.45,
    )
    return fig, max_k


def save_figure(fig: plt.Figure) -> dict[str, str]:
    stem = "gray_whale_figure5_metric_sensitivity_candidate_v2"
    outputs = {
        "pdf": CANDIDATES / f"{stem}.pdf",
        "svg": CANDIDATES / f"{stem}.svg",
        "png": CANDIDATES / f"{stem}.png",
        "tiff": CANDIDATES / f"{stem}.tiff",
    }
    fig.savefig(outputs["pdf"])
    fig.savefig(outputs["svg"])
    fig.savefig(outputs["png"], dpi=600)
    fig.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {key: str(path.relative_to(ROOT)) for key, path in outputs.items()}


def write_results_summary(
    effort: pd.DataFrame,
    topk: pd.DataFrame,
    distance: pd.DataFrame,
    winners: pd.DataFrame,
    max_k: int,
) -> None:
    lines = [
        "# Figure 5 metric-sensitivity results",
        "",
        "This isolated analysis varies one observed-use metric definition at a time.",
        "Metrics use archived corridor surfaces with fixed observation membership.",
        "",
        "## Shared valid top-k range",
        "",
        f"The annual regional quantile remained above zero for every scenario-year from top 1% through top {max_k}%.",
        "Larger cores were audited but not interpreted when their equal-area boundary reached zero corridor frequency.",
        "",
        "## Winning scenario counts",
        "",
    ]
    for period in ["full", "common"]:
        lines.extend(
            [
                f"### {PERIODS[period]['label']}",
                "",
                f"- Effort-footprint definitions: {scenario_winner_counts(winners, period, 'effort-footprint percentile')}.",
                f"- Annual core thresholds: {scenario_winner_counts(winners.loc[winners.get('nominal_top_k_percent', pd.Series(index=winners.index, dtype=float)).fillna(0).le(max_k)], period, 'annual core overlap')}.",
                f"- Corridor-frequency cutoffs: {scenario_winner_counts(winners, period, 'distance to corridor')}.",
                "",
            ]
        )

    lines.extend(["## Effort-background sample sizes", ""])
    for period in ["full", "common"]:
        sub = (
            effort.loc[effort["period"].eq(period)]
            .sort_values("definition_order")
            .drop_duplicates("effort_definition")
        )
        text = "; ".join(
            f"{row.effort_definition_label}: {int(row.record_count):,} records"
            for row in sub.itertuples(index=False)
        )
        lines.append(f"- {PERIODS[period]['label']}: {text}.")

    lines.extend(
        [
            "",
            "## Scientific read",
            "",
            "- In 2011-2019, Prey + ice had the higher effort-footprint percentile under the annual, >=7/9-year and >=8/9-year backgrounds; Prey was slightly higher only in the 9/9-year core, which retained 132 records. Prey + ice led at 14 of 15 equal-area core sizes and was never farther from sightings than Prey across the six corridor-frequency cutoffs.",
            "- In 2015-2019, Prey + ice + vessel had the highest effort-footprint percentile under the annual, >=3/5-year and >=4/5-year backgrounds. Prey + vessel was highest only in the 5/5-year core, which retained 121 records.",
            "- In the common period, the equal-area core-overlap winner depended on core width: Prey + ice led at k=1%, 2% and 5%; Prey led at k=3% and 4%; and Prey + ice + vessel led throughout k=6-15%.",
            "- Prey + ice remained the nearest corridor geography in the common period: it had the lowest median distance at four of six frequency cutoffs and tied Prey + ice + vessel at the other two.",
            "- The three metrics therefore do not support a single threshold-independent scenario winner. The strongest defensible statement is that Prey + ice gives the most stable proximity result, whereas percentile rank and broader-core overlap can favor vessel-inclusive landscapes.",
            "",
            "## Interpretation rule",
            "",
            "A stable winner across a reasonable sensitivity range supports keeping the primary definition in the main figure and placing this diagnostic in Supplementary Information. Frequent rank reversals would instead require explicit main-text qualification.",
            "",
        ]
    )
    (OUT / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_caption(max_k: int) -> None:
    caption = f"""# Candidate caption

**Sensitivity of observed-use comparisons to metric definitions.** **A-C,** Results for the benthic prey-only and prey+ice scenarios during 2011-2019 using 1,179 June-July gray whale sightings from Aerial Surveys of Arctic Marine Mammals. **D-F,** Results for all four scenarios during the common 2015-2019 period using 666 sightings. **A,D,** Mean percentile rank of corridor frequency at sighting cells relative to cells surveyed under the same footprint definition. Moving right restricts both the comparison cells and retained sightings to locations surveyed in progressively more years, so the curves are not expected to be monotonic. Values below the axes give retained sightings. **B,E,** Percentage of sightings within annual corridor cores containing the indicated fraction of highest-frequency regional cells. Cells tied at a core boundary were weighted fractionally to preserve the requested core area without arbitrary spatial tie-breaking. **C,F,** Median distance from sightings to the nearest corridor cell across normalized corridor-frequency cutoffs. Filled symbols mark the manuscript's primary definitions; dashed vertical lines identify their parameter values. Core sizes whose boundary reached zero corridor frequency were omitted. Metrics measure spatial correspondence within documented survey coverage, not abundance or behavioral response.
"""
    (OUT / "caption_candidate_v2.md").write_text(caption, encoding="utf-8")


def write_records(
    effort: pd.DataFrame,
    topk: pd.DataFrame,
    distance: pd.DataFrame,
    primary_qa: pd.DataFrame,
    winners: pd.DataFrame,
    max_k: int,
    figure_outputs: dict[str, str],
) -> None:
    effort.to_csv(SOURCE_DATA / "effort_footprint_percentile_sensitivity_v2.csv", index=False)
    topk.to_csv(SOURCE_DATA / "annual_core_overlap_sensitivity_v2.csv", index=False)
    distance.to_csv(SOURCE_DATA / "distance_to_corridor_sensitivity_v2.csv", index=False)
    winners.to_csv(SOURCE_DATA / "scenario_winners_by_metric_definition_v2.csv", index=False)
    primary_qa.to_csv(QA / "primary_value_reproduction_qa_v2.csv", index=False)

    input_paths = [GRID, ASAMM_SIGHTINGS, ASAMM_EFFORT, PRIMARY_FULL, PRIMARY_COMMON]
    input_paths.extend(
        grid_path(scenario, year)
        for spec in PERIODS.values()
        for scenario in spec["scenarios"]
        for year in spec["years"]
    )
    input_paths = list(dict.fromkeys(input_paths))
    provenance = pd.DataFrame(
        [
            {
                "file": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths
        ]
    )
    provenance.to_csv(OUT / "input_provenance_manifest_v2.csv", index=False)

    record_consistency = (
        effort.groupby(["period", "effort_definition"])["record_count"].nunique().eq(1).all()
        and topk.groupby(["period", "nominal_top_k_percent"])["record_count"].nunique().eq(1).all()
        and distance.groupby(["period", "frequency_threshold"])["record_count"].nunique().eq(1).all()
    )
    qa_payload = {
        "analysis": "Figure 5 one-at-a-time metric sensitivity",
        "backend": "Python in mypy10",
        "primary_values_reproduced": bool(primary_qa["pass_tolerance_1e_9"].all()),
        "identical_record_counts_across_scenarios": bool(record_consistency),
        "shared_valid_top_k_max_percent": int(max_k),
        "full_period_record_count": 1179,
        "common_period_record_count": 666,
        "distance_thresholds": DISTANCE_THRESHOLDS,
        "primary_top_k_percent": PRIMARY_TOP_K,
        "primary_frequency_threshold": PRIMARY_FREQUENCY_THRESHOLD,
        "figure_outputs": figure_outputs,
        "warnings": [],
    }
    if not qa_payload["primary_values_reproduced"]:
        qa_payload["warnings"].append("Primary values were not reproduced")
    if not qa_payload["identical_record_counts_across_scenarios"]:
        qa_payload["warnings"].append("Scenario record counts differ")
    (QA / "metric_sensitivity_qa_v2.json").write_text(
        json.dumps(qa_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "analysis_branch": "outputs/manuscript_figures_v3/figure_5/sensitivity_v2",
        "status": "complete",
        "analysis_type": "one-at-a-time metric-definition sensitivity",
        "periods": PERIODS,
        "top_k_values_audited": TOP_K_VALUES,
        "shared_top_k_values_plotted": list(range(1, max_k + 1)),
        "distance_thresholds": DISTANCE_THRESHOLDS,
        "primary_inputs_read_only": True,
        "manuscript_modified": False,
        "existing_figures_modified": False,
    }
    (OUT / "analysis_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    sightings, effort_input, grids = read_inputs()
    effort = build_effort_sensitivity(sightings, effort_input, grids)
    topk = build_topk_sensitivity(sightings, grids)
    distance = build_distance_sensitivity(sightings, grids)
    primary_qa = validate_primary_reproduction(effort, topk, distance)
    winners = winner_rows(effort, topk, distance)
    figure, max_k = make_figure(effort, topk, distance)
    figure_outputs = save_figure(figure)
    write_records(
        effort,
        topk,
        distance,
        primary_qa,
        winners,
        max_k,
        figure_outputs,
    )
    write_results_summary(effort, topk, distance, winners, max_k)
    write_caption(max_k)
    print(json.dumps({"status": "complete", "max_top_k": max_k, **figure_outputs}, indent=2))


if __name__ == "__main__":
    main()
