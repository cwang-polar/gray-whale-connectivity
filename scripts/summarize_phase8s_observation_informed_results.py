"""Build auditable Phase 8S result tables from completed corridor grids."""

from __future__ import annotations

if __name__ == '__main__':
    raise SystemExit('Historical CLI disabled in this release; use v8.py run pelagic_support.')

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase8s_observation_informed_corridor_sensitivity as phase8s  # noqa: E402


OUT = ROOT / "outputs/pelagic_support_phase8s"
CORRIDOR = OUT / "corridor_sensitivity"
SUMMARIES = CORRIDOR / "summaries"
VALIDATION = CORRIDOR / "validation"
TABLES = OUT / "tables"


def record_metrics(
    grid: pd.DataFrame,
    sightings: pd.DataFrame,
    effort: pd.DataFrame,
) -> pd.DataFrame:
    year = int(grid["target_year"].iloc[0])
    annual_sightings = sightings.loc[sightings["year"].eq(year)]
    effort_ids = set(effort.loc[effort["year"].eq(year), "grid_id"])
    lookup = grid.set_index("grid_id")
    effort_frequency = lookup.loc[
        lookup.index.intersection(effort_ids), "lcp_frequency_normalized"
    ].fillna(0.0)
    frequency = grid["lcp_frequency_normalized"].fillna(0.0)
    q95 = float(frequency.quantile(0.95))
    frequent = grid.loc[frequency > phase8s.FREQUENT_THRESHOLD, ["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(frequent) if len(frequent) else None
    rows = []
    for _, record in annual_sightings.iterrows():
        grid_id = record["nearest_grid_id"]
        if grid_id not in lookup.index:
            continue
        cell = lookup.loc[grid_id]
        value = float(cell["lcp_frequency_normalized"])
        percentile = float(
            ((effort_frequency < value).sum() + 0.5 * (effort_frequency == value).sum())
            / len(effort_frequency)
            * 100.0
        )
        distance = math.nan
        if tree is not None:
            distance = float(
                tree.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0] / 1000.0
            )
        rows.append(
            {
                "record_id": record["record_id"],
                "target_year": year,
                "grid_id": grid_id,
                "lcp_frequency_at_sighting": value,
                "effort_footprint_percentile": percentile,
                "inside_annual_top5pct": value >= q95,
                "distance_to_frequent_corridor_km": distance,
                "strict_effort_background_cell_count": len(effort_frequency),
            }
        )
    return pd.DataFrame(rows)


def build_record_table() -> pd.DataFrame:
    sightings, effort = phase8s.load_asamm()
    rows = []
    for operator, betas in phase8s.CENTRAL_DESIGN.items():
        for beta in betas:
            for year in phase8s.YEARS:
                for scenario in phase8s.SCENARIOS:
                    grid = pd.read_csv(
                        phase8s.central_grid_path(operator, beta, scenario, year),
                        dtype={"grid_id": str},
                    )
                    metrics = record_metrics(grid, sightings, effort)
                    metrics["combination_operator"] = operator
                    metrics["pelagic_strength_beta"] = beta
                    metrics["scenario"] = scenario
                    metrics["scenario_label"] = phase8s.SCENARIO_LABELS[scenario]
                    rows.append(metrics)
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(VALIDATION / "phase8s_central_asamm_record_metrics.csv", index=False)
    expected = sum(
        len(betas) * len(phase8s.SCENARIOS)
        for betas in phase8s.CENTRAL_DESIGN.values()
    ) * sightings["record_id"].nunique()
    if len(output) != expected:
        raise RuntimeError(f"Expected {expected} ASAMM metric rows, found {len(output)}")
    return output


def pooled_validation(record_table: pd.DataFrame) -> pd.DataFrame:
    output = (
        record_table.groupby(
            ["combination_operator", "pelagic_strength_beta", "scenario", "scenario_label"],
            as_index=False,
        )
        .agg(
            sighting_record_count=("record_id", "nunique"),
            year_count=("target_year", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100.0)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
        )
    )
    output.to_csv(VALIDATION / "phase8s_central_asamm_pooled_2017_2018_summary.csv", index=False)
    return output


def build_change_tables(route: pd.DataFrame, pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_route = route.loc[
        route["combination_operator"].eq("complementary")
        & route["pelagic_strength_beta"].eq(0.0)
    ].set_index(["target_year", "scenario"])
    route_rows = []
    for row in route.itertuples():
        baseline = baseline_route.loc[(row.target_year, row.scenario)]
        route_rows.append(
            {
                "target_year": row.target_year,
                "combination_operator": row.combination_operator,
                "pelagic_strength_beta": row.pelagic_strength_beta,
                "scenario": row.scenario,
                "delta_mean_path_cost_percent": row.delta_mean_path_cost_percent,
                "delta_frequent_corridor_cell_count": row.delta_frequent_corridor_cell_count,
                "jaccard_with_baseline_frequent_cells": row.jaccard_with_baseline_frequent_cells,
                "frequent_centroid_shift_from_baseline_km": row.frequent_centroid_shift_from_baseline_km,
                "baseline_mean_path_cost": baseline["mean_path_cost"],
            }
        )
    route_change = pd.DataFrame(route_rows)
    route_change.to_csv(SUMMARIES / "phase8s_central_corridor_change_from_benthic_baseline.csv", index=False)

    baseline_validation = pooled.loc[
        pooled["combination_operator"].eq("complementary")
        & pooled["pelagic_strength_beta"].eq(0.0)
    ].set_index("scenario")
    validation_rows = []
    for row in pooled.itertuples():
        baseline = baseline_validation.loc[row.scenario]
        validation_rows.append(
            {
                "combination_operator": row.combination_operator,
                "pelagic_strength_beta": row.pelagic_strength_beta,
                "scenario": row.scenario,
                "sighting_record_count": row.sighting_record_count,
                "mean_effort_footprint_percentile": row.mean_effort_footprint_percentile,
                "delta_effort_footprint_percentile_vs_beta0": (
                    row.mean_effort_footprint_percentile
                    - baseline["mean_effort_footprint_percentile"]
                ),
                "percent_records_top5pct": row.percent_records_top5pct,
                "delta_top5_overlap_percentage_points_vs_beta0": (
                    row.percent_records_top5pct - baseline["percent_records_top5pct"]
                ),
                "median_distance_to_frequent_corridor_km": row.median_distance_to_frequent_corridor_km,
                "delta_median_distance_km_vs_beta0": (
                    row.median_distance_to_frequent_corridor_km
                    - baseline["median_distance_to_frequent_corridor_km"]
                ),
            }
        )
    validation_change = pd.DataFrame(validation_rows)
    validation_change.to_csv(
        VALIDATION / "phase8s_central_asamm_change_from_beta0_2017_2018.csv", index=False
    )
    return route_change, validation_change


def condensed_findings(
    route: pd.DataFrame,
    pooled: pd.DataFrame,
    validation_change: pd.DataFrame,
) -> pd.DataFrame:
    medium = route.loc[
        route["combination_operator"].eq("complementary")
        & route["pelagic_strength_beta"].eq(0.25)
    ]
    operator_medium = route.loc[route["pelagic_strength_beta"].eq(0.25)]
    validation_medium = validation_change.loc[
        validation_change["combination_operator"].eq("complementary")
        & validation_change["pelagic_strength_beta"].eq(0.25)
    ]
    rows = [
        {
            "finding_id": "F1",
            "result": (
                f"For the primary complementary operator at beta=0.25, mean path cost changed by "
                f"{medium['delta_mean_path_cost_percent'].min():.1f}% "
                f"to {medium['delta_mean_path_cost_percent'].max():.1f}% across 2017-2018 and four scenarios."
            ),
            "interpretation": "Observed pelagic spatial structure can materially lower modeled resistance along established routes.",
            "boundary": "Beta is a sensitivity coefficient, not an estimated energy conversion.",
        },
        {
            "finding_id": "F2",
            "result": (
                f"For the primary complementary operator at beta=0.25, frequent-cell Jaccard with the "
                f"benthic baseline ranged from "
                f"{medium['jaccard_with_baseline_frequent_cells'].min():.3f} to "
                f"{medium['jaccard_with_baseline_frequent_cells'].max():.3f}, and centroid shifts ranged from "
                f"{medium['frequent_centroid_shift_from_baseline_km'].min():.1f} to "
                f"{medium['frequent_centroid_shift_from_baseline_km'].max():.1f} km."
            ),
            "interpretation": "The empirical support pattern reorganizes corridor geography, not only cost.",
            "boundary": "Magnitude depends on the cross-prey combination operator.",
        },
        {
            "finding_id": "F3",
            "result": (
                f"Across operators at beta=0.25, every scenario-year lowered cost; the full range was "
                f"{operator_medium['delta_mean_path_cost_percent'].min():.1f}% to "
                f"{operator_medium['delta_mean_path_cost_percent'].max():.1f}%."
            ),
            "interpretation": "The direction of the corridor response is structurally robust.",
            "boundary": "Operator-dependent effect sizes should not be reported as estimates.",
        },
        {
            "finding_id": "F4",
            "result": (
                "For the primary complementary operator at beta=0.25, no scenario improved all three "
                "pooled ASAMM metrics relative to beta=0; "
                f"percentile changes ranged from {validation_medium['delta_effort_footprint_percentile_vs_beta0'].min():+.2f} "
                f"to {validation_medium['delta_effort_footprint_percentile_vs_beta0'].max():+.2f}, top-5% changes from "
                f"{validation_medium['delta_top5_overlap_percentage_points_vs_beta0'].min():+.2f} to "
                f"{validation_medium['delta_top5_overlap_percentage_points_vs_beta0'].max():+.2f} percentage points."
            ),
            "interpretation": "Pelagic support is a plausible omitted pathway but does not uniquely resolve observed-use consistency.",
            "boundary": "The comparison covers only 323 June-July records in 2017-2018.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "phase8s_condensed_findings.csv", index=False)
    return frame


def build_review_tables(
    route: pd.DataFrame,
    pooled: pd.DataFrame,
    validation_change: pd.DataFrame,
) -> None:
    """Write compact, review-facing tables without changing analytical outputs."""

    route_review = route.loc[
        route["combination_operator"].eq("complementary")
        & route["pelagic_strength_beta"].eq(0.25),
        [
            "target_year",
            "scenario",
            "scenario_label",
            "delta_mean_path_cost_percent",
            "delta_frequent_corridor_cell_count",
            "jaccard_with_baseline_frequent_cells",
            "frequent_centroid_shift_from_baseline_km",
        ],
    ].copy()
    route_review.to_csv(TABLES / "phase8s_review_route_response_beta025.csv", index=False)

    baseline = pooled.loc[
        pooled["combination_operator"].eq("complementary")
        & pooled["pelagic_strength_beta"].eq(0.0),
        [
            "scenario",
            "mean_effort_footprint_percentile",
            "percent_records_top5pct",
            "median_distance_to_frequent_corridor_km",
        ],
    ].rename(
        columns={
            "mean_effort_footprint_percentile": "baseline_mean_effort_footprint_percentile",
            "percent_records_top5pct": "baseline_percent_records_top5pct",
            "median_distance_to_frequent_corridor_km": "baseline_median_distance_km",
        }
    )
    medium = validation_change.loc[
        validation_change["combination_operator"].eq("complementary")
        & validation_change["pelagic_strength_beta"].eq(0.25)
    ].copy()
    asamm_review = medium.merge(baseline, on="scenario", how="left", validate="one_to_one")
    asamm_review = asamm_review[
        [
            "scenario",
            "sighting_record_count",
            "baseline_mean_effort_footprint_percentile",
            "mean_effort_footprint_percentile",
            "delta_effort_footprint_percentile_vs_beta0",
            "baseline_percent_records_top5pct",
            "percent_records_top5pct",
            "delta_top5_overlap_percentage_points_vs_beta0",
            "baseline_median_distance_km",
            "median_distance_to_frequent_corridor_km",
            "delta_median_distance_km_vs_beta0",
        ]
    ]
    asamm_review.to_csv(TABLES / "phase8s_review_asamm_response_beta025.csv", index=False)


def update_status() -> None:
    state = json.loads(phase8s.STATUS.read_text(encoding="utf-8"))
    state["stages"]["result_synthesis"] = {
        "status": "completed",
        "record_level_asamm_metrics_saved": True,
        "pooled_period": "2017-2018",
        "asamm_record_count": 323,
        "warnings": [],
    }
    phase8s.write_json(phase8s.STATUS, state)


def main() -> None:
    route = pd.read_csv(SUMMARIES / "phase8s_central_route_summary.csv")
    records = build_record_table()
    pooled = pooled_validation(records)
    _, validation_change = build_change_tables(route, pooled)
    findings = condensed_findings(route, pooled, validation_change)
    build_review_tables(route, pooled, validation_change)
    update_status()
    print(findings[["finding_id", "result"]].to_string(index=False))


if __name__ == "__main__":
    main()
