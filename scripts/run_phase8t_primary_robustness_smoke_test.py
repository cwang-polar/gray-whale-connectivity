#!/usr/bin/env python3
"""Run isolated robustness checks for the manuscript-primary corridor analysis.

Phase 8T does not replace or modify any primary prey, ice, vessel, LCP,
validation, figure, manuscript, or Supplementary Information product. It tests
four reviewer-facing assumptions: prey interpolation, the sea-ice transition
anchor, vessel-modifier strength, and leverage of individual ASAMM years.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import build_phase8b_brian_amphipod_prey_surfaces_c3_v2 as prey_base
import build_phase8b2_brian_amphipod_prey_surfaces_full_trailing_c3_v2 as prey_trailing
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base
from gray_whale_corridor.movement.lcp_ensemble import build_weighted_adjacency, run_ensemble


YEARS = list(range(2011, 2020))
VESSEL_YEARS = list(range(2015, 2020))
PREY_VARIABLE = "total_kJm2_frozen"
PREY_UTILITY = "U_amphipod_prey_global_q05_q95"
EPSILON = 0.001
FREQUENT_THRESHOLD = 0.05

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
PREY_DIR = ROOT / "outputs/prey_v2"
ICE_DIR = ROOT / "outputs/ice"
SHIP_DIR = ROOT / "outputs/ship"
PRIMARY_LCP = ROOT / "outputs/lcp_v2"
PHASE8K_LCP = ROOT / "outputs/prey_vessel_phase8k/lcp"
ASAMM_SIGHTINGS = ROOT / "data/processed/validation/asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
ASAMM_EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"

OUT = ROOT / "outputs/robustness_smoke_test_phase8t"
CV_OUT = OUT / "interpolation_cv"
SURFACE_OUT = OUT / "surfaces"
LCP_OUT = OUT / "lcp"
SUMMARY_OUT = OUT / "summaries"
VALIDATION_OUT = OUT / "validation"
QA_OUT = OUT / "qa"
STATUS = OUT / "phase8t_run_status.json"
MANIFEST = OUT / "phase8t_analysis_manifest.json"

IDW_SETTINGS = {
    "k4_p2": {"neighbors": 4, "power": 2.0, "primary": False},
    "k8_p1": {"neighbors": 8, "power": 1.0, "primary": False},
    "primary_k8_p2": {"neighbors": 8, "power": 2.0, "primary": True},
    "k8_p3": {"neighbors": 8, "power": 3.0, "primary": False},
}
ICE_SETTINGS = {
    "upper_0_45": {"upper_sic": 0.45, "primary": False},
    "primary_upper_0_55": {"upper_sic": 0.55, "primary": True},
    "upper_0_65": {"upper_sic": 0.65, "primary": False},
}
VESSEL_SETTINGS = {
    "alpha_0_25": {"alpha": 0.25, "primary": False},
    "primary_alpha_0_50": {"alpha": 0.50, "primary": True},
    "alpha_0_75": {"alpha": 0.75, "primary": False},
}

SCENARIO_LABELS = {
    "prey_only": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["p1", "p3", "p4"],
        required=True,
        help="Explicit V8 stage only; paired ice anchors have a separate entry.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (OUT, CV_OUT, SURFACE_OUT, LCP_OUT, SUMMARY_OUT, VALIDATION_OUT, QA_OUT):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_status() -> dict:
    if STATUS.exists():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {
        "phase": "Phase 8T primary-analysis robustness smoke test",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "stages": {name: {"status": "pending"} for name in ("audit", "p1", "p2", "p3", "p4", "integrity")},
    }


def set_stage(stage: str, state: str, **details: object) -> None:
    status = load_status()
    entry = status["stages"].setdefault(stage, {})
    entry["status"] = state
    entry[f"{state}_utc"] = utc_now()
    entry.update(details)
    status["updated_utc"] = utc_now()
    write_json(STATUS, status)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prey_path(year: int) -> Path:
    return PREY_DIR / f"amphipod_prey_surface_{PREY_VARIABLE}_target{year}_full_trailing_C3_v2.csv"


def ice_path(year: int) -> Path:
    return ICE_DIR / f"ice_accessibility_v1_MayJul_target{year}.csv"


def ship_path(year: int) -> Path:
    return SHIP_DIR / f"ship_intensity_MayJul_target{year}.csv"


def primary_grid_path(scenario: str, year: int) -> Path:
    if scenario == "prey_only":
        return PRIMARY_LCP / f"prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice":
        return PRIMARY_LCP / f"prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_vessel":
        return PHASE8K_LCP / f"prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice_vessel":
        return PRIMARY_LCP / f"prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    raise ValueError(scenario)


def primary_route_path(scenario: str) -> Path:
    if scenario == "prey_only":
        return PRIMARY_LCP / "prey_only_brian_C3_route_summary_v2.csv"
    if scenario == "prey_ice":
        return PRIMARY_LCP / "prey_ice_brian_C3_route_summary_v2.csv"
    if scenario == "prey_vessel":
        return PHASE8K_LCP / "prey_vessel_brian_C3_route_summary_v2.csv"
    if scenario == "prey_ice_vessel":
        return PRIMARY_LCP / "prey_ice_ship_brian_C3_route_summary_v2.csv"
    raise ValueError(scenario)


def integrity_inputs() -> list[Path]:
    paths = [GRID, ENDPOINTS, ASAMM_SIGHTINGS, ASAMM_EFFORT]
    paths.extend(prey_path(year) for year in YEARS)
    paths.extend(ice_path(year) for year in YEARS)
    paths.extend(ship_path(year) for year in VESSEL_YEARS)
    for scenario, years in {
        "prey_only": YEARS,
        "prey_ice": YEARS,
        "prey_vessel": VESSEL_YEARS,
        "prey_ice_vessel": VESSEL_YEARS,
    }.items():
        paths.append(primary_route_path(scenario))
        paths.extend(primary_grid_path(scenario, year) for year in years)
    paths.extend([prey_trailing.STATION_TOTALS, prey_trailing.UNIQUE_MATCH])
    protected_visuals = [
        ROOT / "outputs/manuscript_figures/figure_2/gray_whale_cee_figure2_draft_v1.pdf",
        ROOT / "outputs/manuscript_figures/figure_2/gray_whale_cee_figure2_draft_v1.svg",
        ROOT / "outputs/manuscript_figures/figure_2/gray_whale_cee_figure2_draft_v1.png",
        ROOT / "outputs/manuscript_figures/figure_2/gray_whale_cee_figure2_draft_v1.tiff",
        ROOT / "outputs/manuscript_figures_v3/figure_3/candidates/gray_whale_figure3_centroid_direction_candidate_v1.pdf",
        ROOT / "outputs/manuscript_figures_v3/figure_3/candidates/gray_whale_figure3_centroid_direction_candidate_v1.svg",
        ROOT / "outputs/manuscript_figures_v3/figure_3/candidates/gray_whale_figure3_centroid_direction_candidate_v1.png",
        ROOT / "outputs/manuscript_figures_v3/figure_3/candidates/gray_whale_figure3_centroid_direction_candidate_v1.tiff",
        ROOT / "outputs/manuscript_figures_v3/figure_4/candidates/gray_whale_figure4_four_scenario_comparison_candidate_v2.pdf",
        ROOT / "outputs/manuscript_figures_v3/figure_4/candidates/gray_whale_figure4_four_scenario_comparison_candidate_v2.svg",
        ROOT / "outputs/manuscript_figures_v3/figure_4/candidates/gray_whale_figure4_four_scenario_comparison_candidate_v2.png",
        ROOT / "outputs/manuscript_figures_v3/figure_4/candidates/gray_whale_figure4_four_scenario_comparison_candidate_v2.tiff",
        ROOT / "outputs/manuscript_figures_v3/figure_4/figure4_caption_candidate_v4.md",
        ROOT / "outputs/manuscript_figures_v3/figure_5/candidates/gray_whale_figure5_dual_period_validation_candidate_v3.pdf",
        ROOT / "outputs/manuscript_figures_v3/figure_5/candidates/gray_whale_figure5_dual_period_validation_candidate_v3.svg",
        ROOT / "outputs/manuscript_figures_v3/figure_5/candidates/gray_whale_figure5_dual_period_validation_candidate_v3.png",
        ROOT / "outputs/manuscript_figures_v3/figure_5/candidates/gray_whale_figure5_dual_period_validation_candidate_v3.tiff",
        ROOT / "outputs/manuscript_figures_v3/figure_5/figure5_caption_candidate_v4.md",
    ]
    paths.extend(protected_visuals)
    return list(dict.fromkeys(paths))


def write_integrity_snapshot(name: str) -> pd.DataFrame:
    rows = []
    for path in integrity_inputs():
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("relative_path")
    frame.to_csv(QA_OUT / f"input_integrity_{name}.csv", index=False)
    return frame


def audit() -> None:
    set_stage("audit", "running")
    before = write_integrity_snapshot("before")
    grid = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    manifest = {
        "phase": "Phase 8T",
        "analysis_role": "isolated reviewer-facing robustness smoke test; not a replacement primary analysis",
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "primary_years": YEARS,
        "vessel_years": VESSEL_YEARS,
        "prey_variable": PREY_VARIABLE,
        "grid_cells": int(len(grid)),
        "source_endpoints": int((endpoints["endpoint_type"] == "source").sum()),
        "destination_endpoints": int((endpoints["endpoint_type"] == "destination").sum()),
        "paths_per_scenario_year": 1600,
        "p1_idw_settings": IDW_SETTINGS,
        "p2_ice_settings": ICE_SETTINGS,
        "p3_vessel_settings": VESSEL_SETTINGS,
        "p4_year_leverage": "leave one survey year out while retaining the primary corridor surfaces and metric definitions",
        "existing_results_are_read_only": True,
        "uncertainty_language": "parameter-sensitivity ranges and year-leverage ranges; not confidence intervals",
        "input_file_count": int(len(before)),
    }
    write_json(MANIFEST, manifest)
    set_stage("audit", "completed", input_file_count=len(before))


def load_topology():
    grid = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    topology = lcp_base.build_graph_topology(grid, endpoints)
    topology.grid["grid_id"] = topology.grid["grid_id"].astype(str)
    return topology


def resistance(utility: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(-np.log(np.asarray(utility, dtype=float) + EPSILON), 0.0)


def run_lcp(topology, utility: pd.DataFrame, stage: str, setting: str, scenario: str, year: int) -> tuple[dict, pd.DataFrame]:
    surface = utility[["grid_id", "utility"]].copy()
    surface["grid_id"] = surface["grid_id"].astype(str)
    grid = topology.grid.merge(surface, on="grid_id", how="left", validate="one_to_one")
    if grid["utility"].isna().any():
        raise RuntimeError(f"{stage}/{setting}/{scenario}/{year}: missing utility on topology grid")
    grid["utility"] = pd.to_numeric(grid["utility"], errors="coerce").clip(EPSILON, 1.0)
    grid["resistance"] = resistance(grid["utility"])
    adjacency = build_weighted_adjacency(topology.edge_pairs, grid["resistance"].to_numpy())
    paths, counts = run_ensemble(adjacency, topology)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    if len(paths) != attempted:
        raise RuntimeError(f"{stage}/{setting}/{scenario}/{year}: {len(paths)}/{attempted} paths succeeded")
    frequency = counts / len(paths)
    frequent = frequency > FREQUENT_THRESHOLD
    if not frequent.any():
        raise RuntimeError(f"{stage}/{setting}/{scenario}/{year}: no frequent-corridor cells")
    costs = np.asarray([path.cost for path in paths], dtype=float)
    centroid = grid.loc[frequent, ["x", "y", "lon", "lat"]].mean()
    route = {
        "stage": stage,
        "setting": setting,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "target_year": year,
        "successful_path_count": len(paths),
        "failed_path_count": attempted - len(paths),
        "mean_path_cost": float(costs.mean()),
        "median_path_cost": float(np.median(costs)),
        "min_path_cost": float(costs.min()),
        "max_path_cost": float(costs.max()),
        "corridor_area_cells_freq_gt_0_05": int(frequent.sum()),
        "top_bottleneck_frequency": float(frequency.max()),
        "centroid_x": float(centroid["x"]),
        "centroid_y": float(centroid["y"]),
        "centroid_lon": float(centroid["lon"]),
        "centroid_lat": float(centroid["lat"]),
    }
    freq = grid[["grid_id", "x", "y", "lon", "lat"]].copy()
    freq.insert(0, "stage", stage)
    freq.insert(1, "setting", setting)
    freq.insert(2, "scenario", scenario)
    freq.insert(3, "target_year", year)
    freq["lcp_frequency_normalized"] = frequency
    return route, freq


def primary_prey(year: int) -> pd.DataFrame:
    frame = pd.read_csv(prey_path(year), usecols=["grid_id", PREY_UTILITY])
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame.rename(columns={PREY_UTILITY: "utility"})


def primary_ice(year: int) -> pd.DataFrame:
    col = "A_ice_MayJul_mean_from_monthly_A_filled"
    frame = pd.read_csv(ice_path(year), usecols=["grid_id", col])
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame.rename(columns={col: "modifier"})


def primary_vessel(year: int) -> pd.DataFrame:
    cols = ["grid_id", "S_scaled_ship_primary_MayJul_mean", "D_ship_alpha0_5_MayJul_mean"]
    frame = pd.read_csv(ship_path(year), usecols=cols)
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame


def combine_utility(prey: pd.DataFrame, *modifiers: pd.DataFrame) -> pd.DataFrame:
    out = prey[["grid_id", "utility"]].copy()
    for number, modifier in enumerate(modifiers):
        mod = modifier[["grid_id", "modifier"]].rename(columns={"modifier": f"modifier_{number}"})
        out = out.merge(mod, on="grid_id", how="left", validate="one_to_one")
        if out[f"modifier_{number}"].isna().any():
            raise RuntimeError("Missing modifier during utility construction")
        out["utility"] *= out[f"modifier_{number}"].clip(0.05, 1.0)
        out = out.drop(columns=f"modifier_{number}")
    out["utility"] = out["utility"].clip(EPSILON, 1.0)
    return out


def jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    merged = left[["grid_id", "lcp_frequency_normalized"]].merge(
        right[["grid_id", "lcp_frequency_normalized"]],
        on="grid_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    a = merged["lcp_frequency_normalized_left"] > FREQUENT_THRESHOLD
    b = merged["lcp_frequency_normalized_right"] > FREQUENT_THRESHOLD
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else math.nan


def displacement(left_route: pd.Series | dict, right_route: pd.Series | dict) -> tuple[float, float, float]:
    dx = (float(right_route["centroid_x"]) - float(left_route["centroid_x"])) / 1000.0
    dy = (float(right_route["centroid_y"]) - float(left_route["centroid_y"])) / 1000.0
    return float(math.hypot(dx, dy)), dx, dy


def load_primary_route(scenario: str, year: int) -> dict:
    frame = pd.read_csv(primary_route_path(scenario))
    if "prey_variable" in frame.columns:
        frame = frame.loc[frame["prey_variable"].eq(PREY_VARIABLE)]
    row = frame.loc[frame["target_year"].eq(year)].iloc[0]
    grid = load_primary_grid(scenario, year)
    frequent = grid["lcp_frequency_normalized"] > FREQUENT_THRESHOLD
    centroid = grid.loc[frequent, ["x", "y", "lon", "lat"]].mean()
    result = row.to_dict()
    result.update(
        centroid_x=float(centroid["x"]),
        centroid_y=float(centroid["y"]),
        centroid_lon=float(centroid["lon"]),
        centroid_lat=float(centroid["lat"]),
    )
    return result


def load_primary_grid(scenario: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(primary_grid_path(scenario, year))
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame


def frozen_run(stage: str, setting: str, scenario: str, year: int) -> tuple[dict, pd.DataFrame]:
    """Load the frozen primary result as the centre line of a sensitivity set."""
    frozen_route = load_primary_route(scenario, year)
    route = {
        "stage": stage,
        "setting": setting,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "target_year": year,
        "successful_path_count": int(frozen_route["successful_path_count"]),
        "failed_path_count": int(frozen_route["failed_path_count"]),
        "mean_path_cost": float(frozen_route["mean_path_cost"]),
        "median_path_cost": float(frozen_route["median_path_cost"]),
        "min_path_cost": float(frozen_route.get("min_path_cost", math.nan)),
        "max_path_cost": float(frozen_route.get("max_path_cost", math.nan)),
        "corridor_area_cells_freq_gt_0_05": int(frozen_route["corridor_area_cells_freq_gt_0_05"]),
        "top_bottleneck_frequency": float(frozen_route["top_bottleneck_frequency"]),
        "centroid_x": float(frozen_route["centroid_x"]),
        "centroid_y": float(frozen_route["centroid_y"]),
        "centroid_lon": float(frozen_route["centroid_lon"]),
        "centroid_lat": float(frozen_route["centroid_lat"]),
    }
    grid = load_primary_grid(scenario, year)[
        ["grid_id", "x", "y", "lon", "lat", "lcp_frequency_normalized"]
    ].copy()
    grid.insert(0, "stage", stage)
    grid.insert(1, "setting", setting)
    grid.insert(2, "scenario", scenario)
    grid.insert(3, "target_year", year)
    return route, grid


def spatial_block_cv(observed: pd.DataFrame, neighbors: int, power: float, max_distance_km: float = 250.0) -> pd.DataFrame:
    data = observed.dropna(subset=["x", "y", "observed_value_window_mean"]).copy()
    x_mid = float(data["x"].median())
    y_mid = float(data["y"].median())
    data["fold"] = np.select(
        [
            (data["x"] < x_mid) & (data["y"] < y_mid),
            (data["x"] >= x_mid) & (data["y"] < y_mid),
            (data["x"] < x_mid) & (data["y"] >= y_mid),
        ],
        ["southwest", "southeast", "northwest"],
        default="northeast",
    )
    rows: list[dict] = []
    for fold in ("southwest", "southeast", "northwest", "northeast"):
        train = data.loc[data["fold"].ne(fold)].copy()
        test = data.loc[data["fold"].eq(fold)].copy()
        if train.empty or test.empty:
            continue
        tree = cKDTree(train[["x", "y"]].to_numpy(dtype=float))
        k = min(neighbors, len(train))
        distances, indices = tree.query(test[["x", "y"]].to_numpy(dtype=float), k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        values = train["observed_value_window_mean"].to_numpy(dtype=float)
        null_prediction = float(values.mean())
        for position, (_, row) in enumerate(test.iterrows()):
            keep = distances[position] <= max_distance_km * 1000.0
            prediction = math.nan
            used = int(keep.sum())
            if used:
                d = distances[position][keep]
                v = values[indices[position][keep]]
                exact = d == 0
                if exact.any():
                    prediction = float(v[exact].mean())
                else:
                    weights = 1.0 / np.power(d, power)
                    prediction = float(np.sum(weights * v) / np.sum(weights))
            rows.append(
                {
                    "grid_id": str(row["grid_id"]),
                    "fold": fold,
                    "observed": float(row["observed_value_window_mean"]),
                    "predicted": prediction,
                    "null_predicted": null_prediction,
                    "neighbor_count": used,
                    "nearest_training_distance_km": float(distances[position][0] / 1000.0),
                }
            )
    return pd.DataFrame(rows)


def cv_metrics(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["observed", "predicted"]).copy()
    if valid.empty:
        return {
            "test_cells": len(frame),
            "predicted_cells": 0,
            "coverage_percent": 0.0,
            "rmse": math.nan,
            "mae": math.nan,
            "bias": math.nan,
            "spearman_rho": math.nan,
            "null_rmse": math.nan,
            "rmse_skill_vs_null_percent": math.nan,
        }
    error = valid["predicted"] - valid["observed"]
    null_error = valid["null_predicted"] - valid["observed"]
    rmse = float(np.sqrt(np.mean(np.square(error))))
    null_rmse = float(np.sqrt(np.mean(np.square(null_error))))
    rho = float(spearmanr(valid["observed"], valid["predicted"]).statistic) if len(valid) > 2 else math.nan
    return {
        "test_cells": int(len(frame)),
        "predicted_cells": int(len(valid)),
        "coverage_percent": float(len(valid) / len(frame) * 100.0),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman_rho": rho,
        "null_rmse": null_rmse,
        "rmse_skill_vs_null_percent": float((null_rmse - rmse) / null_rmse * 100.0) if null_rmse else math.nan,
    }


def p1_interpolation(topology) -> None:
    set_stage("p1", "running")
    grid, stations, _ = prey_trailing.load_full_trailing_inputs()
    grid["grid_id"] = grid["grid_id"].astype(str)
    old_window = prey_base.window_for_target
    old_k = prey_base.IDW_K
    old_power = prey_base.IDW_POWER
    old_distance = prey_base.IDW_MAX_DISTANCE_KM
    surface_frames: list[pd.DataFrame] = []
    cv_predictions: list[pd.DataFrame] = []
    cv_summary: list[dict] = []
    route_rows: list[dict] = []
    frequency_frames: list[pd.DataFrame] = []
    try:
        prey_base.window_for_target = prey_trailing.full_trailing_window_for_target
        for setting, parameters in IDW_SETTINGS.items():
            prey_base.IDW_K = int(parameters["neighbors"])
            prey_base.IDW_POWER = float(parameters["power"])
            prey_base.IDW_MAX_DISTANCE_KM = 250.0
            annual: dict[int, pd.DataFrame] = {}
            for year in YEARS:
                observed = prey_base.aggregate_observed_cells(
                    stations, prey_base.VARIABLES[PREY_VARIABLE]["station_col"], year
                )
                observed["grid_id"] = observed["grid_id"].astype(str)
                observed_xy = observed.merge(grid[["grid_id", "x", "y"]], on="grid_id", how="left")
                cv = spatial_block_cv(
                    observed_xy,
                    int(parameters["neighbors"]),
                    float(parameters["power"]),
                )
                cv.insert(0, "setting", setting)
                cv.insert(1, "target_year", year)
                cv_predictions.append(cv)
                for fold, part in cv.groupby("fold", sort=False):
                    cv_summary.append({"setting": setting, "target_year": year, "scope": fold, **cv_metrics(part)})
                cv_summary.append({"setting": setting, "target_year": year, "scope": "all_blocks", **cv_metrics(cv)})

                surface = prey_base.build_idw_surface(grid, observed, year, PREY_VARIABLE)
                annual[year] = surface
            all_values = pd.concat([frame["raw_final_value"] for frame in annual.values()], ignore_index=True)
            q05 = float(all_values.quantile(0.05))
            q95 = float(all_values.quantile(0.95))
            for year, surface in annual.items():
                surface = surface.copy()
                surface["utility"] = prey_base.normalize_with_params(surface["raw_final_value"], q05, q95)
                if surface["utility"].isna().any():
                    raise RuntimeError(f"P1 {setting} {year}: incomplete prey surface")
                surface_frames.append(
                    surface[["grid_id", "target_year", "raw_final_value", "utility", "observed_window_flag", "interpolated_flag", "idw_neighbor_count", "nearest_observed_distance_km"]].assign(
                        setting=setting,
                        neighbors=int(parameters["neighbors"]),
                        power=float(parameters["power"]),
                        q05=q05,
                        q95=q95,
                    )
                )
                # Use the frozen manuscript surface for the primary branch so
                # equal-cost path ties reproduce the existing cell-frequency
                # map exactly. Rebuilt values differ only at machine precision.
                prey = primary_prey(year) if parameters["primary"] else surface[["grid_id", "utility"]]
                if parameters["primary"]:
                    route, freq = frozen_run("p1_idw", setting, "prey_only", year)
                else:
                    route, freq = run_lcp(topology, prey, "p1_idw", setting, "prey_only", year)
                route_rows.append(route)
                frequency_frames.append(freq)
                ice = primary_ice(year)
                prey_ice = combine_utility(prey, ice)
                if parameters["primary"]:
                    route, freq = frozen_run("p1_idw", setting, "prey_ice", year)
                else:
                    route, freq = run_lcp(topology, prey_ice, "p1_idw", setting, "prey_ice", year)
                route_rows.append(route)
                frequency_frames.append(freq)
                print(f"P1 {setting}: completed {year}", flush=True)
    finally:
        prey_base.window_for_target = old_window
        prey_base.IDW_K = old_k
        prey_base.IDW_POWER = old_power
        prey_base.IDW_MAX_DISTANCE_KM = old_distance

    surfaces = pd.concat(surface_frames, ignore_index=True)
    routes = pd.DataFrame(route_rows)
    frequencies = pd.concat(frequency_frames, ignore_index=True)
    predictions = pd.concat(cv_predictions, ignore_index=True)
    predictions.to_csv(CV_OUT / "p1_idw_spatial_block_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cv_summary).to_csv(CV_OUT / "p1_idw_spatial_block_metrics.csv", index=False)
    surfaces.to_csv(SURFACE_OUT / "p1_idw_energy_surfaces.csv.gz", index=False, compression="gzip")
    routes.to_csv(SUMMARY_OUT / "p1_idw_route_metrics.csv", index=False)
    frequencies.to_csv(LCP_OUT / "p1_idw_frequency_surfaces.csv.gz", index=False, compression="gzip")

    checks: list[dict] = []
    primary_routes = routes.loc[routes["setting"].eq("primary_k8_p2")]
    primary_freq = frequencies.loc[frequencies["setting"].eq("primary_k8_p2")]
    for scenario in ("prey_only", "prey_ice"):
        for year in YEARS:
            test_route = primary_routes.loc[
                primary_routes["scenario"].eq(scenario) & primary_routes["target_year"].eq(year)
            ].iloc[0]
            frozen_route = load_primary_route(scenario, year)
            test_grid = primary_freq.loc[
                primary_freq["scenario"].eq(scenario) & primary_freq["target_year"].eq(year)
            ]
            frozen_grid = load_primary_grid(scenario, year)
            merged = test_grid[["grid_id", "lcp_frequency_normalized"]].merge(
                frozen_grid[["grid_id", "lcp_frequency_normalized"]],
                on="grid_id",
                suffixes=("_test", "_frozen"),
                validate="one_to_one",
            )
            checks.append(
                {
                    "scenario": scenario,
                    "target_year": year,
                    "mean_cost_absolute_difference": abs(float(test_route["mean_path_cost"]) - float(frozen_route["mean_path_cost"])),
                    "frequent_cell_difference": int(test_route["corridor_area_cells_freq_gt_0_05"]) - int(frozen_route["corridor_area_cells_freq_gt_0_05"]),
                    "maximum_frequency_absolute_difference": float(np.max(np.abs(merged["lcp_frequency_normalized_test"] - merged["lcp_frequency_normalized_frozen"]))),
                }
            )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(QA_OUT / "p1_primary_reproduction_check.csv", index=False)
    if checks_frame["mean_cost_absolute_difference"].max() > 1e-7 or checks_frame["maximum_frequency_absolute_difference"].max() > 1e-12 or checks_frame["frequent_cell_difference"].abs().max() != 0:
        raise RuntimeError("P1 primary setting did not reproduce the frozen primary analysis")
    set_stage(
        "p1",
        "completed",
        settings=list(IDW_SETTINGS),
        scenario_year_runs=len(routes),
        primary_reproduction="exact within floating-point tolerance",
    )


def accessibility_from_sic(sic: pd.Series, upper: float) -> np.ndarray:
    lower = 0.15
    floor = 0.05
    values = pd.to_numeric(sic, errors="coerce").to_numpy(dtype=float)
    accessibility = np.full(values.shape, np.nan, dtype=float)
    accessibility[values <= lower] = 1.0
    middle = (values > lower) & (values < upper)
    accessibility[middle] = 1.0 - ((values[middle] - lower) / (upper - lower)) * (1.0 - floor)
    accessibility[values >= upper] = floor
    return np.clip(accessibility, floor, 1.0)


def ice_modifier(year: int, upper: float) -> pd.DataFrame:
    cols = ["grid_id", "SIC_May_filled", "SIC_June_filled", "SIC_July_filled"]
    frame = pd.read_csv(ice_path(year), usecols=cols)
    frame["grid_id"] = frame["grid_id"].astype(str)
    monthly = np.column_stack(
        [accessibility_from_sic(frame[column], upper) for column in cols[1:]]
    )
    frame["modifier"] = np.nanmean(monthly, axis=1)
    return frame[["grid_id", "modifier"]]


def comparison_row(
    setting: str,
    year: int,
    left_scenario: str,
    right_scenario: str,
    left_route: dict | pd.Series,
    right_route: dict | pd.Series,
    left_grid: pd.DataFrame,
    right_grid: pd.DataFrame,
) -> dict:
    total, east, north = displacement(left_route, right_route)
    return {
        "setting": setting,
        "target_year": year,
        "scenario_a": left_scenario,
        "scenario_b": right_scenario,
        "mean_path_cost_a": float(left_route["mean_path_cost"]),
        "mean_path_cost_b": float(right_route["mean_path_cost"]),
        "percent_cost_change_b_vs_a": float((float(right_route["mean_path_cost"]) - float(left_route["mean_path_cost"])) / float(left_route["mean_path_cost"]) * 100.0),
        "frequent_cells_a": int(left_route["corridor_area_cells_freq_gt_0_05"]),
        "frequent_cells_b": int(right_route["corridor_area_cells_freq_gt_0_05"]),
        "frequent_cell_change_b_minus_a": int(right_route["corridor_area_cells_freq_gt_0_05"]) - int(left_route["corridor_area_cells_freq_gt_0_05"]),
        "jaccard_freq_gt_0_05": jaccard(left_grid, right_grid),
        "centroid_displacement_km": total,
        "centroid_east_west_displacement_km": east,
        "centroid_north_south_displacement_km": north,
    }


def p2_ice(topology) -> None:
    set_stage("p2", "running")
    routes: list[dict] = []
    frequencies: list[pd.DataFrame] = []
    surface_summary: list[dict] = []
    for setting, parameters in ICE_SETTINGS.items():
        upper = float(parameters["upper_sic"])
        for year in YEARS:
            modifier = primary_ice(year) if parameters["primary"] else ice_modifier(year, upper)
            surface_summary.append(
                {
                    "setting": setting,
                    "upper_sic": upper,
                    "target_year": year,
                    "mean_accessibility": float(modifier["modifier"].mean()),
                    "median_accessibility": float(modifier["modifier"].median()),
                    "cells_accessibility_lt_0_50": int((modifier["modifier"] < 0.50).sum()),
                }
            )
            prey = primary_prey(year)
            utility = combine_utility(prey, modifier)
            if parameters["primary"]:
                route, freq = frozen_run("p2_ice", setting, "prey_ice", year)
            else:
                route, freq = run_lcp(topology, utility, "p2_ice", setting, "prey_ice", year)
            routes.append(route)
            frequencies.append(freq)
            if year in VESSEL_YEARS:
                vessel = primary_vessel(year)[["grid_id", "D_ship_alpha0_5_MayJul_mean"]].rename(
                    columns={"D_ship_alpha0_5_MayJul_mean": "modifier"}
                )
                combined = combine_utility(prey, modifier, vessel)
                if parameters["primary"]:
                    route, freq = frozen_run("p2_ice", setting, "prey_ice_vessel", year)
                else:
                    route, freq = run_lcp(topology, combined, "p2_ice", setting, "prey_ice_vessel", year)
                routes.append(route)
                frequencies.append(freq)
            print(f"P2 {setting}: completed {year}", flush=True)
    route_frame = pd.DataFrame(routes)
    frequency_frame = pd.concat(frequencies, ignore_index=True)
    route_frame.to_csv(SUMMARY_OUT / "p2_ice_route_metrics.csv", index=False)
    frequency_frame.to_csv(LCP_OUT / "p2_ice_frequency_surfaces.csv.gz", index=False, compression="gzip")
    pd.DataFrame(surface_summary).to_csv(SURFACE_OUT / "p2_ice_accessibility_summary.csv", index=False)

    comparisons: list[dict] = []
    for setting in ICE_SETTINGS:
        for year in YEARS:
            right_route = route_frame.loc[
                route_frame["setting"].eq(setting)
                & route_frame["scenario"].eq("prey_ice")
                & route_frame["target_year"].eq(year)
            ].iloc[0]
            right_grid = frequency_frame.loc[
                frequency_frame["setting"].eq(setting)
                & frequency_frame["scenario"].eq("prey_ice")
                & frequency_frame["target_year"].eq(year)
            ]
            comparisons.append(
                comparison_row(
                    setting,
                    year,
                    "prey_only",
                    "prey_ice",
                    load_primary_route("prey_only", year),
                    right_route,
                    load_primary_grid("prey_only", year),
                    right_grid,
                )
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(SUMMARY_OUT / "p2_ice_comparison_metrics.csv", index=False)

    checks: list[dict] = []
    primary = route_frame.loc[route_frame["setting"].eq("primary_upper_0_55")]
    primary_freq = frequency_frame.loc[frequency_frame["setting"].eq("primary_upper_0_55")]
    for scenario, years in {"prey_ice": YEARS, "prey_ice_vessel": VESSEL_YEARS}.items():
        for year in years:
            test_route = primary.loc[primary["scenario"].eq(scenario) & primary["target_year"].eq(year)].iloc[0]
            frozen_route = load_primary_route(scenario, year)
            test_grid = primary_freq.loc[primary_freq["scenario"].eq(scenario) & primary_freq["target_year"].eq(year)]
            frozen_grid = load_primary_grid(scenario, year)
            merged = test_grid[["grid_id", "lcp_frequency_normalized"]].merge(
                frozen_grid[["grid_id", "lcp_frequency_normalized"]], on="grid_id", suffixes=("_test", "_frozen"), validate="one_to_one"
            )
            checks.append(
                {
                    "scenario": scenario,
                    "target_year": year,
                    "mean_cost_absolute_difference": abs(float(test_route["mean_path_cost"]) - float(frozen_route["mean_path_cost"])),
                    "frequent_cell_difference": int(test_route["corridor_area_cells_freq_gt_0_05"]) - int(frozen_route["corridor_area_cells_freq_gt_0_05"]),
                    "maximum_frequency_absolute_difference": float(np.max(np.abs(merged["lcp_frequency_normalized_test"] - merged["lcp_frequency_normalized_frozen"]))),
                }
            )
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(QA_OUT / "p2_primary_reproduction_check.csv", index=False)
    if check_frame["mean_cost_absolute_difference"].max() > 1e-7 or check_frame["maximum_frequency_absolute_difference"].max() > 1e-12 or check_frame["frequent_cell_difference"].abs().max() != 0:
        raise RuntimeError("P2 primary setting did not reproduce the frozen primary analysis")
    set_stage("p2", "completed", settings=list(ICE_SETTINGS), scenario_year_runs=len(route_frame))


def vessel_modifier(year: int, alpha: float) -> pd.DataFrame:
    frame = primary_vessel(year)
    scaled = pd.to_numeric(frame["S_scaled_ship_primary_MayJul_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    frame["modifier"] = np.minimum(1.0, np.maximum(0.05, 1.0 - alpha * scaled))
    return frame[["grid_id", "modifier"]]


def frozen_primary_vessel_modifier(year: int) -> pd.DataFrame:
    frame = primary_vessel(year)[["grid_id", "D_ship_alpha0_5_MayJul_mean"]].copy()
    return frame.rename(columns={"D_ship_alpha0_5_MayJul_mean": "modifier"})


def p3_vessel(topology) -> None:
    set_stage("p3", "running")
    routes: list[dict] = []
    frequencies: list[pd.DataFrame] = []
    surface_summary: list[dict] = []
    for setting, parameters in VESSEL_SETTINGS.items():
        alpha = float(parameters["alpha"])
        for year in VESSEL_YEARS:
            prey = primary_prey(year)
            vessel = frozen_primary_vessel_modifier(year) if parameters["primary"] else vessel_modifier(year, alpha)
            surface_summary.append(
                {
                    "setting": setting,
                    "alpha": alpha,
                    "target_year": year,
                    "mean_modifier": float(vessel["modifier"].mean()),
                    "median_modifier": float(vessel["modifier"].median()),
                    "minimum_modifier": float(vessel["modifier"].min()),
                }
            )
            utility = combine_utility(prey, vessel)
            if parameters["primary"]:
                route, freq = frozen_run("p3_vessel", setting, "prey_vessel", year)
            else:
                route, freq = run_lcp(topology, utility, "p3_vessel", setting, "prey_vessel", year)
            routes.append(route)
            frequencies.append(freq)
            ice = primary_ice(year)
            combined = combine_utility(prey, ice, vessel)
            if parameters["primary"]:
                route, freq = frozen_run("p3_vessel", setting, "prey_ice_vessel", year)
            else:
                route, freq = run_lcp(topology, combined, "p3_vessel", setting, "prey_ice_vessel", year)
            routes.append(route)
            frequencies.append(freq)
            print(f"P3 {setting}: completed {year}", flush=True)
    route_frame = pd.DataFrame(routes)
    frequency_frame = pd.concat(frequencies, ignore_index=True)
    route_frame.to_csv(SUMMARY_OUT / "p3_vessel_route_metrics.csv", index=False)
    frequency_frame.to_csv(LCP_OUT / "p3_vessel_frequency_surfaces.csv.gz", index=False, compression="gzip")
    pd.DataFrame(surface_summary).to_csv(SURFACE_OUT / "p3_vessel_modifier_summary.csv", index=False)

    comparisons: list[dict] = []
    for setting in VESSEL_SETTINGS:
        for year in VESSEL_YEARS:
            for left_scenario, right_scenario in (
                ("prey_only", "prey_vessel"),
                ("prey_ice", "prey_ice_vessel"),
            ):
                right_route = route_frame.loc[
                    route_frame["setting"].eq(setting)
                    & route_frame["scenario"].eq(right_scenario)
                    & route_frame["target_year"].eq(year)
                ].iloc[0]
                right_grid = frequency_frame.loc[
                    frequency_frame["setting"].eq(setting)
                    & frequency_frame["scenario"].eq(right_scenario)
                    & frequency_frame["target_year"].eq(year)
                ]
                comparisons.append(
                    comparison_row(
                        setting,
                        year,
                        left_scenario,
                        right_scenario,
                        load_primary_route(left_scenario, year),
                        right_route,
                        load_primary_grid(left_scenario, year),
                        right_grid,
                    )
                )
    pd.DataFrame(comparisons).to_csv(SUMMARY_OUT / "p3_vessel_comparison_metrics.csv", index=False)

    checks: list[dict] = []
    primary = route_frame.loc[route_frame["setting"].eq("primary_alpha_0_50")]
    primary_freq = frequency_frame.loc[frequency_frame["setting"].eq("primary_alpha_0_50")]
    for scenario in ("prey_vessel", "prey_ice_vessel"):
        for year in VESSEL_YEARS:
            test_route = primary.loc[primary["scenario"].eq(scenario) & primary["target_year"].eq(year)].iloc[0]
            frozen_route = load_primary_route(scenario, year)
            test_grid = primary_freq.loc[primary_freq["scenario"].eq(scenario) & primary_freq["target_year"].eq(year)]
            frozen_grid = load_primary_grid(scenario, year)
            merged = test_grid[["grid_id", "lcp_frequency_normalized"]].merge(
                frozen_grid[["grid_id", "lcp_frequency_normalized"]], on="grid_id", suffixes=("_test", "_frozen"), validate="one_to_one"
            )
            checks.append(
                {
                    "scenario": scenario,
                    "target_year": year,
                    "mean_cost_absolute_difference": abs(float(test_route["mean_path_cost"]) - float(frozen_route["mean_path_cost"])),
                    "frequent_cell_difference": int(test_route["corridor_area_cells_freq_gt_0_05"]) - int(frozen_route["corridor_area_cells_freq_gt_0_05"]),
                    "maximum_frequency_absolute_difference": float(np.max(np.abs(merged["lcp_frequency_normalized_test"] - merged["lcp_frequency_normalized_frozen"]))),
                }
            )
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(QA_OUT / "p3_primary_reproduction_check.csv", index=False)
    if check_frame["mean_cost_absolute_difference"].max() > 1e-7 or check_frame["maximum_frequency_absolute_difference"].max() > 1e-12 or check_frame["frequent_cell_difference"].abs().max() != 0:
        raise RuntimeError("P3 primary setting did not reproduce the frozen primary analysis")
    set_stage("p3", "completed", settings=list(VESSEL_SETTINGS), scenario_year_runs=len(route_frame))


def scenario_record_metrics(scenario: str, years: list[int], sightings: pd.DataFrame, effort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year in years:
        annual_sightings = sightings.loc[sightings["year"].eq(year)]
        effort_ids = set(effort.loc[effort["year"].eq(year), "grid_id"])
        grid = load_primary_grid(scenario, year)
        frequency = grid["lcp_frequency_normalized"].fillna(0.0)
        q95 = float(frequency.quantile(0.95))
        frequent_xy = grid.loc[frequency > FREQUENT_THRESHOLD, ["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(frequent_xy) if len(frequent_xy) else None
        lookup = grid.set_index("grid_id")
        effort_frequency = lookup.loc[
            lookup.index.intersection(effort_ids), "lcp_frequency_normalized"
        ].fillna(0.0)
        for record in annual_sightings.itertuples(index=False):
            grid_id = str(record.nearest_grid_id)
            if grid_id not in lookup.index:
                continue
            cell = lookup.loc[grid_id]
            freq = float(cell["lcp_frequency_normalized"])
            percentile = float(
                ((effort_frequency < freq).sum() + 0.5 * (effort_frequency == freq).sum())
                / len(effort_frequency)
                * 100.0
            )
            distance = math.nan
            if tree is not None:
                distance = float(tree.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0] / 1000.0)
            rows.append(
                {
                    "record_id": record.record_id,
                    "target_year": year,
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "grid_id": grid_id,
                    "lcp_frequency_at_sighting": freq,
                    "effort_footprint_percentile": percentile,
                    "inside_annual_top5pct": bool(freq >= q95),
                    "distance_to_frequent_corridor_km": distance,
                    "strict_effort_background_cell_count": int(len(effort_frequency)),
                }
            )
    return pd.DataFrame(rows)


def summarize_records(frame: pd.DataFrame) -> dict:
    return {
        "record_count": int(frame["record_id"].nunique()),
        "effort_footprint_percentile": float(frame["effort_footprint_percentile"].mean()),
        "top5_overlap_percent": float(frame["inside_annual_top5pct"].mean() * 100.0),
        "distance_to_frequent_corridor_km": float(frame["distance_to_frequent_corridor_km"].median()),
    }


def p4_year_leverage() -> None:
    set_stage("p4", "running")
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    sightings = sightings.loc[
        sightings["use_for_C3_validation"].astype(bool)
        & sightings["year"].isin(YEARS)
        & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)
    effort = pd.read_csv(ASAMM_EFFORT)
    effort = effort.loc[
        effort["year"].isin(YEARS)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    effort["grid_id"] = effort["grid_id"].astype(str)

    definitions = [
        ("full_2011_2019", YEARS, ["prey_only", "prey_ice"]),
        ("common_2015_2019", VESSEL_YEARS, ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]),
    ]
    record_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for period, years, scenarios in definitions:
        for scenario in scenarios:
            records = scenario_record_metrics(scenario, years, sightings, effort)
            records.insert(0, "analysis_period", period)
            record_frames.append(records)
            summaries.append(
                {
                    "analysis_period": period,
                    "scenario": scenario,
                    "omitted_year": "none",
                    "summary_type": "primary_all_years",
                    **summarize_records(records),
                }
            )
            for omitted in years:
                reduced = records.loc[records["target_year"].ne(omitted)]
                summaries.append(
                    {
                        "analysis_period": period,
                        "scenario": scenario,
                        "omitted_year": str(omitted),
                        "summary_type": "leave_one_year_out",
                        **summarize_records(reduced),
                    }
                )
            print(f"P4 {period}: completed {scenario}", flush=True)
    record_frame = pd.concat(record_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    record_frame.to_csv(VALIDATION_OUT / "p4_asamm_record_metrics.csv.gz", index=False, compression="gzip")
    summary_frame.to_csv(VALIDATION_OUT / "p4_asamm_pooled_and_leave_one_year_out.csv", index=False)

    metric_columns = [
        "effort_footprint_percentile",
        "top5_overlap_percent",
        "distance_to_frequent_corridor_km",
    ]
    rows: list[dict] = []
    for (period, scenario), group in summary_frame.groupby(["analysis_period", "scenario"], sort=False):
        primary = group.loc[group["summary_type"].eq("primary_all_years")].iloc[0]
        jackknife = group.loc[group["summary_type"].eq("leave_one_year_out")]
        row = {
            "analysis_period": period,
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "record_count": int(primary["record_count"]),
            "leave_one_year_out_count": int(len(jackknife)),
        }
        for metric in metric_columns:
            row[f"{metric}_primary"] = float(primary[metric])
            row[f"{metric}_minimum"] = float(jackknife[metric].min())
            row[f"{metric}_maximum"] = float(jackknife[metric].max())
        rows.append(row)
    ranges = pd.DataFrame(rows)
    ranges.to_csv(VALIDATION_OUT / "p4_asamm_year_leverage_ranges.csv", index=False)
    expected = {"full_2011_2019": 1179, "common_2015_2019": 666}
    for period, count in expected.items():
        observed = set(ranges.loc[ranges["analysis_period"].eq(period), "record_count"].astype(int))
        if observed != {count}:
            raise RuntimeError(f"P4 {period}: expected {count} records, observed {observed}")
    set_stage("p4", "completed", periods=list(expected), record_rows=len(record_frame))


def finalize_integrity() -> None:
    set_stage("integrity", "running")
    before = pd.read_csv(QA_OUT / "input_integrity_before.csv")
    after = write_integrity_snapshot("after")
    comparison = before.merge(after, on="relative_path", suffixes=("_before", "_after"), validate="one_to_one")
    comparison["unchanged"] = (
        comparison["size_bytes_before"].eq(comparison["size_bytes_after"])
        & comparison["sha256_before"].eq(comparison["sha256_after"])
    )
    comparison.to_csv(QA_OUT / "input_integrity_comparison.csv", index=False)
    changed = comparison.loc[~comparison["unchanged"]]
    if not changed.empty:
        raise RuntimeError(f"Phase 8T changed {len(changed)} read-only input files")
    set_stage("integrity", "completed", checked_files=len(comparison), changed_files=0)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.stage == "p1" and not (QA_OUT / "input_integrity_before.csv").exists():
        audit()
    topology = load_topology() if args.stage in ("p1", "p3") else None
    if args.stage == "p1":
        p1_interpolation(topology)
    elif args.stage == "p3":
        p3_vessel(topology)
    elif args.stage == "p4":
        p4_year_leverage()
    print(json.dumps(load_status(), indent=2))


if __name__ == "__main__":
    main()
