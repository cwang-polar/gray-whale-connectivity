"""Run the isolated Phase 8Q August-October connectivity branch.

The established May-July scripts and outputs are read-only. Paths are solved
with the established south-to-north computational orientation to keep Dijkstra
tie handling fixed, then reported with north origins and south destinations.
The graph remains undirected; reported order does not create directional
movement inference.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_phase8o_season_scope_and_external_method as asamm_scope  # noqa: E402
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base  # noqa: E402
from gray_whale_corridor.movement.lcp_ensemble import (  # noqa: E402
    build_weighted_adjacency,
    path_length_km,
    run_ensemble,
)


PHASE = "Phase 8Q August-October late-season reciprocal connectivity"
SCRIPT = "scripts/run_phase8q_aug_oct_late_season_connectivity.py"
ALL_YEARS = list(range(2011, 2020))
VESSEL_YEARS = list(range(2015, 2020))
MONTHS = [8, 9, 10]
MONTH_NAMES = {8: "August", 9: "September", 10: "October"}
VESSEL_TYPES = ["Cargo", "Fishing", "Other", "Tanker"]
SCENARIO_YEARS = {
    "prey_only": ALL_YEARS,
    "prey_ice": ALL_YEARS,
    "prey_vessel": VESSEL_YEARS,
    "prey_ice_vessel": VESSEL_YEARS,
}
SCENARIO_LABELS = {
    "prey_only": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}

OUT = ROOT / "outputs/southward_connectivity_phase8q"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
ICE = ROOT / "data/raw/ice/cmems_mod_glo_phy_my_0.083deg_P1M-m_1779048578112.nc"
SHIP_DATA = ROOT / "data/raw/ship/resource_map_doi_10_18739_A2NZ80R4J/data"
SHIP_PARAMS = ROOT / "outputs/ship/ship_scaling_parameters_v0.json"
PHASE8P_SCALING = ROOT / "outputs/southward_connectivity_phase8p_audit/tables/phase8p_ship_scaling_decision.csv"
PREY_DIR = ROOT / "outputs/prey_v2"
PREY_PATTERN = "amphipod_prey_surface_total_kJm2_frozen_target{year}_full_trailing_C3_v2.csv"

EPSILON = 0.001
SIC_LOW = 0.15
SIC_HIGH = 0.55
A_MIN = 0.05
FREQUENT_THRESHOLD = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def output_tree(mode: str) -> dict[str, Path]:
    base = OUT / "smoke_tests/target2019" if mode == "smoke" else OUT
    return {
        "base": base,
        "surfaces": base / "surfaces",
        "lcp": base / "lcp",
        "summaries": base / "summaries",
        "validation": base / "validation",
        "status": base / ("phase8q_smoke_status_target2019.json" if mode == "smoke" else "phase8q_run_status.json"),
        "manifest": base / ("phase8q_smoke_manifest_target2019.json" if mode == "smoke" else "phase8q_analysis_manifest.json"),
    }


def ensure_tree(tree: dict[str, Path]) -> None:
    for key in ["base", "surfaces", "lcp", "summaries", "validation"]:
        tree[key].mkdir(parents=True, exist_ok=True)


def set_stage(tree: dict[str, Path], stage: str, status: str, **details: object) -> None:
    if tree["status"].exists():
        state = json.loads(tree["status"].read_text(encoding="utf-8"))
    else:
        state = {
            "phase": PHASE,
            "status": "running",
            "started_utc": utc_now(),
            "stages": {},
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
        }
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    if status == "failed":
        state["status"] = "failed"
    write_json(tree["status"], state)


def prey_path(year: int) -> Path:
    return PREY_DIR / PREY_PATTERN.format(year=year)


def ship_path(year: int, month: int, vessel_type: str) -> Path:
    return SHIP_DATA / f"Raster_{year}_{month:02d}_{vessel_type}_10km.tif"


def early_lcp_path(scenario: str, year: int) -> Path:
    if scenario == "prey_only":
        return ROOT / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice":
        return ROOT / f"outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_vessel":
        return ROOT / f"outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice_vessel":
        return ROOT / f"outputs/lcp_v2/prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    raise KeyError(scenario)


def input_paths(years: list[int], mode: str) -> list[Path]:
    paths = [GRID, ENDPOINTS, ICE, SHIP_PARAMS, PHASE8P_SCALING]
    paths.extend(prey_path(year) for year in years)
    for year in years:
        if year in VESSEL_YEARS:
            paths.extend(ship_path(year, month, vessel_type) for month in MONTHS for vessel_type in VESSEL_TYPES)
        for scenario, scenario_years in SCENARIO_YEARS.items():
            if year in scenario_years:
                paths.append(early_lcp_path(scenario, year))
    if mode == "batch":
        paths.extend(asamm_scope.ASAMM_OUTPUTS)
        paths.append(asamm_scope.DOMAIN_PATH)
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def snapshot(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def accessibility_from_sic(sic: np.ndarray) -> np.ndarray:
    values = np.asarray(sic, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    result[values <= SIC_LOW] = 1.0
    middle = (values > SIC_LOW) & (values < SIC_HIGH)
    result[middle] = 1.0 - ((values[middle] - SIC_LOW) / (SIC_HIGH - SIC_LOW)) * (1.0 - A_MIN)
    result[values >= SIC_HIGH] = A_MIN
    return np.clip(result, A_MIN, 1.0)


def sample_regular_grid(
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    points_lon: np.ndarray,
    points_lat: np.ndarray,
) -> np.ndarray:
    lon = np.where(longitude > 180, longitude - 360, longitude)
    lon_index = np.abs(lon[None, :] - points_lon[:, None]).argmin(axis=1)
    lat_index = np.abs(latitude[None, :] - points_lat[:, None]).argmin(axis=1)
    return values[lat_index, lon_index].astype(float)


def build_ice_surface(year: int, grid: pd.DataFrame, dataset: xr.Dataset) -> tuple[pd.DataFrame, dict]:
    dates = pd.to_datetime(dataset["time"].values)
    longitude = dataset["longitude"].values.astype(float)
    latitude = dataset["latitude"].values.astype(float)
    points_lon = grid["lon"].to_numpy(dtype=float)
    points_lat = grid["lat"].to_numpy(dtype=float)
    output = grid[["grid_id", "x", "y", "lon", "lat", "ocean_flag"]].copy()
    missing_counts = {}
    access_columns = []
    sic_columns = []
    for month in MONTHS:
        indices = np.where((dates.year == year) & (dates.month == month))[0]
        if len(indices) != 1:
            raise RuntimeError(f"Expected one SIC slice for {year}-{month:02d}; found {len(indices)}")
        sampled = sample_regular_grid(
            longitude,
            latitude,
            dataset["siconc"].isel(time=int(indices[0])).values.astype(float),
            points_lon,
            points_lat,
        )
        missing = ~np.isfinite(sampled)
        sic = np.clip(np.where(missing, 0.0, sampled), 0.0, 1.0)
        access = accessibility_from_sic(sic)
        name = MONTH_NAMES[month]
        output[f"SIC_{name}_filled"] = sic
        output[f"A_ice_{name}"] = access
        output[f"SIC_{name}_missing_filled_open_water"] = missing.astype(int)
        missing_counts[name] = int(missing.sum())
        access_columns.append(f"A_ice_{name}")
        sic_columns.append(f"SIC_{name}_filled")
    output["target_year"] = year
    output["SIC_AugOct_mean_filled"] = output[sic_columns].mean(axis=1)
    output["A_ice_AugOct_mean"] = output[access_columns].mean(axis=1)
    qa = {
        "target_year": year,
        "cell_count": len(output),
        "monthly_missing_filled_as_open_water": missing_counts,
        "mean_A_ice_AugOct": float(output["A_ice_AugOct_mean"].mean()),
        "min_A_ice_AugOct": float(output["A_ice_AugOct_mean"].min()),
        "cells_A_ice_lt_0_95": int((output["A_ice_AugOct_mean"] < 0.95).sum()),
        "cells_A_ice_lt_0_50": int((output["A_ice_AugOct_mean"] < 0.50).sum()),
    }
    return output, qa


def sample_vessel_raster(path: Path, grid: pd.DataFrame) -> np.ndarray:
    with rasterio.open(path) as source:
        transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
        xs, ys = transformer.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
        sampled = np.ma.asarray(list(source.sample(zip(xs, ys), indexes=1, masked=True))).reshape(-1)
        values = sampled.filled(np.nan).astype(float)
    if np.isnan(values).any():
        raise RuntimeError(f"Sampled vessel raster contains nodata: {path}")
    return values


def build_vessel_surface(
    year: int,
    grid: pd.DataFrame,
    primary_log_q99: float,
    sensitivity_log_q99: float,
) -> tuple[pd.DataFrame, dict]:
    output = grid[["grid_id", "x", "y", "lon", "lat", "ocean_flag"]].copy()
    primary_scaled_columns = []
    sensitivity_scaled_columns = []
    primary_modifier_columns = []
    sensitivity_modifier_columns = []
    raw_columns = []
    for month in MONTHS:
        name = MONTH_NAMES[month]
        raw_by_type = [sample_vessel_raster(ship_path(year, month, vessel_type), grid) for vessel_type in VESSEL_TYPES]
        raw = np.sum(np.vstack(raw_by_type), axis=0)
        log_raw = np.log1p(raw)
        scaled_primary = np.clip(log_raw, 0, primary_log_q99) / primary_log_q99
        scaled_sensitivity = np.clip(log_raw, 0, sensitivity_log_q99) / sensitivity_log_q99
        modifier_primary = np.clip(1 - 0.5 * scaled_primary, 0.05, 1.0)
        modifier_sensitivity = np.clip(1 - 0.5 * scaled_sensitivity, 0.05, 1.0)
        output[f"S_vessel_all_{name}"] = raw
        output[f"S_scaled_fixed_{name}"] = scaled_primary
        output[f"S_scaled_late_q99_{name}"] = scaled_sensitivity
        output[f"D_vessel_fixed_{name}"] = modifier_primary
        output[f"D_vessel_late_q99_{name}"] = modifier_sensitivity
        raw_columns.append(f"S_vessel_all_{name}")
        primary_scaled_columns.append(f"S_scaled_fixed_{name}")
        sensitivity_scaled_columns.append(f"S_scaled_late_q99_{name}")
        primary_modifier_columns.append(f"D_vessel_fixed_{name}")
        sensitivity_modifier_columns.append(f"D_vessel_late_q99_{name}")
    output["target_year"] = year
    output["S_vessel_all_AugOct_mean"] = output[raw_columns].mean(axis=1)
    output["S_scaled_fixed_AugOct_mean"] = output[primary_scaled_columns].mean(axis=1)
    output["S_scaled_late_q99_AugOct_mean"] = output[sensitivity_scaled_columns].mean(axis=1)
    output["D_vessel_fixed_AugOct_mean"] = output[primary_modifier_columns].mean(axis=1)
    output["D_vessel_late_q99_AugOct_mean"] = output[sensitivity_modifier_columns].mean(axis=1)
    qa = {
        "target_year": year,
        "cell_count": len(output),
        "primary_scaling_log_q99": primary_log_q99,
        "sensitivity_scaling_log_q99": sensitivity_log_q99,
        "mean_S_scaled_fixed_AugOct": float(output["S_scaled_fixed_AugOct_mean"].mean()),
        "mean_D_vessel_fixed_AugOct": float(output["D_vessel_fixed_AugOct_mean"].mean()),
        "nonzero_exposure_cells": int((output["S_vessel_all_AugOct_mean"] > 0).sum()),
    }
    return output, qa


def load_prey(year: int, grid: pd.DataFrame) -> pd.DataFrame:
    prey = pd.read_csv(prey_path(year), dtype={"grid_id": str})
    keep = prey[["grid_id", "U_amphipod_prey_global_q05_q95", "window_start", "window_end"]].copy()
    output = grid[["grid_id", "x", "y", "lon", "lat", "ocean_flag"]].merge(
        keep, on="grid_id", how="left", validate="one_to_one"
    )
    if output["U_amphipod_prey_global_q05_q95"].isna().any():
        raise RuntimeError(f"Prey surface does not cover the full grid for {year}")
    output.rename(columns={"U_amphipod_prey_global_q05_q95": "U_prey_raw"}, inplace=True)
    output["U_prey"] = pd.to_numeric(output["U_prey_raw"], errors="coerce").clip(EPSILON, 1.0)
    output["target_year"] = year
    return output


def build_scenario_surface(
    scenario: str,
    year: int,
    prey: pd.DataFrame,
    ice: pd.DataFrame,
    vessel: pd.DataFrame | None,
) -> pd.DataFrame:
    output = prey.copy()
    output = output.merge(
        ice[["grid_id", "A_ice_AugOct_mean", "SIC_AugOct_mean_filled"]],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    if vessel is None:
        output["S_scaled_vessel_AugOct_mean"] = 0.0
        output["D_vessel_AugOct_mean"] = 1.0
    else:
        output = output.merge(
            vessel[["grid_id", "S_scaled_fixed_AugOct_mean", "D_vessel_fixed_AugOct_mean"]],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        output.rename(
            columns={
                "S_scaled_fixed_AugOct_mean": "S_scaled_vessel_AugOct_mean",
                "D_vessel_fixed_AugOct_mean": "D_vessel_AugOct_mean",
            },
            inplace=True,
        )
    if scenario == "prey_only":
        utility = output["U_prey"]
    elif scenario == "prey_ice":
        utility = output["U_prey"] * output["A_ice_AugOct_mean"]
    elif scenario == "prey_vessel":
        utility = output["U_prey"] * output["D_vessel_AugOct_mean"]
    elif scenario == "prey_ice_vessel":
        utility = output["U_prey"] * output["A_ice_AugOct_mean"] * output["D_vessel_AugOct_mean"]
    else:
        raise KeyError(scenario)
    output["scenario"] = scenario
    output["scenario_label"] = SCENARIO_LABELS[scenario]
    output["U_scenario_raw"] = np.asarray(utility, dtype=float)
    output["U_scenario"] = output["U_scenario_raw"].clip(EPSILON, 1.0)
    output["R_scenario"] = np.maximum(-np.log(output["U_scenario"] + EPSILON), 0.0)
    output["analysis_window"] = "August-October"
    output["reported_movement_season"] = "late-season north-to-south interpretation"
    output["graph_direction"] = "undirected"
    output["computational_endpoint_orientation"] = "canonical south-to-north"
    return output


def align_surface(surface: pd.DataFrame, topology) -> pd.DataFrame:
    data = surface.copy()
    data["grid_id"] = data["grid_id"].astype(str)
    aligned = topology.grid[["grid_id"]].merge(data, on="grid_id", how="left", validate="one_to_one")
    if aligned["U_scenario"].isna().any():
        raise RuntimeError("Scenario surface failed topology alignment")
    return aligned


def run_scenario_lcp(scenario: str, year: int, surface: pd.DataFrame, topology) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aligned = align_surface(surface, topology)
    adjacency = build_weighted_adjacency(topology.edge_pairs, aligned["R_scenario"].to_numpy(dtype=float))
    paths, counts = run_ensemble(adjacency, topology)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    if len(paths) != attempted:
        raise RuntimeError(f"{scenario} {year}: {len(paths)}/{attempted} paths completed")
    aligned["lcp_count"] = counts
    aligned["lcp_frequency_normalized"] = counts / len(paths)
    path_rows = []
    for item in paths:
        nodes = item.node_indices
        node_ids = [topology.grid_id_by_node_index[index] for index in nodes]
        reported_ids = list(reversed(node_ids))
        node_values = aligned.iloc[list(nodes)]
        path_rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "target_year": year,
                "reported_origin_grid_id": reported_ids[0],
                "reported_destination_grid_id": reported_ids[-1],
                "reported_direction": "north-to-south",
                "computational_source_grid_id": node_ids[0],
                "computational_destination_grid_id": node_ids[-1],
                "computational_orientation": "canonical south-to-north",
                "path_sequence_reporting": "reversed after undirected shortest-path solution",
                "path_cost": item.cost,
                "path_length_km": path_length_km(nodes, topology.grid),
                "path_cell_count": len(nodes),
                "path_mean_U": float(node_values["U_scenario"].mean()),
                "path_min_U": float(node_values["U_scenario"].min()),
                "path_max_U": float(node_values["U_scenario"].max()),
                "reported_path_hash": hashlib.sha256("|".join(reported_ids).encode("utf-8")).hexdigest(),
            }
        )
    path_frame = pd.DataFrame(path_rows)
    southern_ids = {topology.grid_id_by_node_index[index] for index in topology.source_indices}
    northern_ids = {topology.grid_id_by_node_index[index] for index in topology.destination_indices}
    direction_qa = {
        "reported_origins_all_northern": set(path_frame["reported_origin_grid_id"]).issubset(northern_ids),
        "reported_destinations_all_southern": set(path_frame["reported_destination_grid_id"]).issubset(southern_ids),
        "computational_sources_all_southern": set(path_frame["computational_source_grid_id"]).issubset(southern_ids),
        "computational_destinations_all_northern": set(path_frame["computational_destination_grid_id"]).issubset(northern_ids),
        "duplicate_reported_path_hash_count": int(path_frame["reported_path_hash"].duplicated().sum()),
    }
    if not all(value is True for key, value in direction_qa.items() if key != "duplicate_reported_path_hash_count"):
        raise RuntimeError(f"{scenario} {year}: endpoint direction-label QA failed: {direction_qa}")
    if direction_qa["duplicate_reported_path_hash_count"]:
        raise RuntimeError(f"{scenario} {year}: duplicate path hashes detected: {direction_qa}")
    frequency = aligned["lcp_frequency_normalized"]
    frequent = aligned.loc[frequency > FREQUENT_THRESHOLD]
    top_index = frequency.idxmax()
    route = {
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "target_year": year,
        "analysis_window": "August-October",
        "reported_direction": "north-to-south seasonal interpretation",
        "graph_direction": "undirected",
        "computational_orientation": "canonical south-to-north",
        "origin_endpoint_count": len(topology.destination_indices),
        "destination_endpoint_count": len(topology.source_indices),
        "attempted_path_count": attempted,
        "successful_path_count": len(paths),
        "failed_path_count": attempted - len(paths),
        "mean_path_cost": float(path_frame["path_cost"].mean()),
        "median_path_cost": float(path_frame["path_cost"].median()),
        "min_path_cost": float(path_frame["path_cost"].min()),
        "max_path_cost": float(path_frame["path_cost"].max()),
        "mean_path_length_km": float(path_frame["path_length_km"].mean()),
        "corridor_area_cells_freq_gt_0": int((frequency > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int((frequency > FREQUENT_THRESHOLD).sum()),
        "corridor_area_cells_freq_gt_0_10": int((frequency > 0.10).sum()),
        "frequent_corridor_centroid_x": float(frequent["x"].mean()) if len(frequent) else math.nan,
        "frequent_corridor_centroid_y": float(frequent["y"].mean()) if len(frequent) else math.nan,
        "top_bottleneck_frequency": float(frequency.max()),
        "top_bottleneck_grid_id": str(aligned.loc[top_index, "grid_id"]),
        "zero_resistance_cell_count": int((aligned["R_scenario"] == 0).sum()),
        **direction_qa,
        "warning_flags": "",
    }
    keep = [
        "grid_id", "x", "y", "lon", "lat", "ocean_flag", "target_year", "scenario", "scenario_label",
        "U_prey_raw", "U_prey", "A_ice_AugOct_mean", "SIC_AugOct_mean_filled",
        "S_scaled_vessel_AugOct_mean", "D_vessel_AugOct_mean", "U_scenario_raw", "U_scenario",
        "R_scenario", "lcp_count", "lcp_frequency_normalized",
        "analysis_window", "reported_movement_season", "graph_direction", "computational_endpoint_orientation",
    ]
    return aligned[keep].copy(), path_frame, route


def jaccard(a: pd.Series, b: pd.Series) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else math.nan


def centroid_shift_km(a: pd.DataFrame, b: pd.DataFrame) -> float:
    aa = a.loc[a["lcp_frequency_normalized"] > FREQUENT_THRESHOLD, ["x", "y"]]
    bb = b.loc[b["lcp_frequency_normalized"] > FREQUENT_THRESHOLD, ["x", "y"]]
    if aa.empty or bb.empty:
        return math.nan
    return float(np.linalg.norm(aa.mean().to_numpy() - bb.mean().to_numpy()) / 1000.0)


def compare_grids(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    merged = a[["grid_id", "lcp_frequency_normalized"]].merge(
        b[["grid_id", "lcp_frequency_normalized"]],
        on="grid_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    mask_a = merged["lcp_frequency_normalized_a"] > FREQUENT_THRESHOLD
    mask_b = merged["lcp_frequency_normalized_b"] > FREQUENT_THRESHOLD
    return {
        "frequency_map_pearson_r": float(merged["lcp_frequency_normalized_a"].corr(merged["lcp_frequency_normalized_b"])),
        "jaccard_freq_gt_0_05": jaccard(mask_a, mask_b),
        "frequent_area_a": int(mask_a.sum()),
        "frequent_area_b": int(mask_b.sum()),
        "frequent_area_b_minus_a": int(mask_b.sum() - mask_a.sum()),
        "centroid_shift_freq_gt_0_05_km": centroid_shift_km(a, b),
        "maximum_absolute_frequency_difference": float(np.max(np.abs(merged["lcp_frequency_normalized_a"] - merged["lcp_frequency_normalized_b"]))),
    }


def write_outputs(
    tree: dict[str, Path],
    scenario: str,
    year: int,
    surface: pd.DataFrame,
    lcp_grid: pd.DataFrame,
    paths: pd.DataFrame,
) -> None:
    surface.to_csv(tree["surfaces"] / f"utility_{scenario}_aug_oct_target{year}_study_region.csv", index=False)
    lcp_grid.to_csv(tree["lcp"] / f"corridor_{scenario}_aug_oct_target{year}_study_region.csv", index=False)
    paths.to_csv(tree["lcp"] / f"paths_{scenario}_aug_oct_target{year}_study_region.csv", index=False)


def load_new_grid(tree: dict[str, Path], scenario: str, year: int) -> pd.DataFrame:
    data = pd.read_csv(tree["lcp"] / f"corridor_{scenario}_aug_oct_target{year}_study_region.csv", dtype={"grid_id": str})
    return data


def build_scenario_comparisons(tree: dict[str, Path], route_summary: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    pair_defs = [
        ("prey_only", "prey_ice", "ice_effect_without_vessel"),
        ("prey_only", "prey_vessel", "vessel_effect_without_ice"),
        ("prey_ice", "prey_ice_vessel", "vessel_effect_with_ice"),
        ("prey_vessel", "prey_ice_vessel", "ice_effect_with_vessel"),
        ("prey_only", "prey_ice_vessel", "combined_effect"),
    ]
    route_lookup = route_summary.set_index(["scenario", "target_year"])
    rows = []
    for year in years:
        for left, right, comparison in pair_defs:
            if year not in SCENARIO_YEARS[left] or year not in SCENARIO_YEARS[right]:
                continue
            left_grid = load_new_grid(tree, left, year)
            right_grid = load_new_grid(tree, right, year)
            metrics = compare_grids(left_grid, right_grid)
            cost_left = float(route_lookup.loc[(left, year), "mean_path_cost"])
            cost_right = float(route_lookup.loc[(right, year), "mean_path_cost"])
            rows.append(
                {
                    "target_year": year,
                    "comparison": comparison,
                    "scenario_a": left,
                    "scenario_b": right,
                    "mean_cost_a": cost_left,
                    "mean_cost_b": cost_right,
                    "mean_cost_b_minus_a": cost_right - cost_left,
                    "mean_cost_percent_change": (cost_right - cost_left) / cost_left * 100 if cost_left else math.nan,
                    **metrics,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(tree["summaries"] / "phase8q_aug_oct_scenario_comparisons.csv", index=False)
    return output


def compare_with_early_season(tree: dict[str, Path], route_summary: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    route_lookup = route_summary.set_index(["scenario", "target_year"])
    rows = []
    for scenario, available_years in SCENARIO_YEARS.items():
        for year in years:
            if year not in available_years:
                continue
            late = load_new_grid(tree, scenario, year)
            early = pd.read_csv(early_lcp_path(scenario, year), dtype={"grid_id": str})
            metrics = compare_grids(early, late)
            early_frequency = early["lcp_frequency_normalized"].fillna(0.0)
            early_path_file = {
                "prey_only": ROOT / "outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_path_summary_v2.csv",
                "prey_ice": ROOT / "outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_path_summary_v2.csv",
                "prey_vessel": ROOT / "outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_path_summary_v2.csv",
                "prey_ice_vessel": ROOT / "outputs/lcp_v2/prey_ice_ship_brian_kJm2_frozen_C3_path_summary_v2.csv",
            }[scenario]
            early_paths = pd.read_csv(early_path_file)
            early_cost = float(early_paths.loc[early_paths["target_year"] == year, "path_cost"].mean())
            late_cost = float(route_lookup.loc[(scenario, year), "mean_path_cost"])
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "target_year": year,
                    "early_window": "May-July",
                    "late_window": "August-October",
                    "mean_path_cost_early": early_cost,
                    "mean_path_cost_late": late_cost,
                    "late_minus_early_cost": late_cost - early_cost,
                    "late_vs_early_cost_percent_change": (late_cost - early_cost) / early_cost * 100 if early_cost else math.nan,
                    "early_frequent_area": int((early_frequency > FREQUENT_THRESHOLD).sum()),
                    "late_frequent_area": metrics["frequent_area_b"],
                    "late_minus_early_frequent_area": metrics["frequent_area_b_minus_a"],
                    "frequency_map_pearson_r": metrics["frequency_map_pearson_r"],
                    "jaccard_freq_gt_0_05": metrics["jaccard_freq_gt_0_05"],
                    "centroid_shift_freq_gt_0_05_km": metrics["centroid_shift_freq_gt_0_05_km"],
                    "maximum_absolute_frequency_difference": metrics["maximum_absolute_frequency_difference"],
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(tree["summaries"] / "phase8q_aug_oct_vs_may_july_comparison.csv", index=False)
    return output


def build_asamm_validation(tree: dict[str, Path], years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = asamm_scope.read_asamm_events()
    events, _ = asamm_scope.assign_spatial_context(events)
    events = events.loc[events["year"].isin(years) & events["month"].isin(MONTHS)].copy()
    sightings = events.loc[events["is_gray_whale"] & events["use_for_study_region_sighting_summary"]].copy()
    sightings["record_id"] = sightings["event_uid"].astype(str)
    effort_events = events.loc[events["inside_study_region"] & events["is_effort_entry"]].copy()
    effort_cells = (
        effort_events.groupby(["year", "nearest_grid_id"], as_index=False)
        .agg(
            strict_effort_event_count=("event_uid", "count"),
            strict_effort_unique_flights=("Flt_No", pd.Series.nunique),
            months_covered=("month", lambda x: ";".join(map(str, sorted(set(map(int, x)))))),
        )
        .rename(columns={"nearest_grid_id": "grid_id"})
    )
    sight_keep = [
        "record_id", "event_uid", "year", "month", "datetime", "Flt_No", "Event", "Entry", "lat", "lon",
        "nearest_grid_id", "nearest_grid_lon", "nearest_grid_lat", "distance_to_nearest_grid_cell_km",
        "count_or_best_count", "source_file",
    ]
    sightings[sight_keep].to_csv(tree["validation"] / "asamm_aug_oct_sightings_study_region.csv", index=False)
    effort_cells.to_csv(tree["validation"] / "asamm_aug_oct_strict_effort_cells.csv", index=False)

    rows = []
    for year in years:
        annual_sightings = sightings.loc[sightings["year"] == year]
        effort_ids = set(effort_cells.loc[effort_cells["year"] == year, "grid_id"].astype(str))
        for scenario, scenario_years in SCENARIO_YEARS.items():
            if year not in scenario_years:
                continue
            grid = load_new_grid(tree, scenario, year)
            grid["grid_id"] = grid["grid_id"].astype(str)
            frequency = grid["lcp_frequency_normalized"].fillna(0.0)
            q95 = float(frequency.quantile(0.95))
            frequent_xy = grid.loc[frequency > FREQUENT_THRESHOLD, ["x", "y"]].to_numpy(dtype=float)
            tree_search = cKDTree(frequent_xy) if len(frequent_xy) else None
            lookup = grid.set_index("grid_id")
            effort_frequency = lookup.loc[lookup.index.intersection(effort_ids), "lcp_frequency_normalized"].fillna(0.0)
            for _, record in annual_sightings.iterrows():
                grid_id = str(record["nearest_grid_id"])
                if grid_id not in lookup.index:
                    continue
                cell = lookup.loc[grid_id]
                value = float(cell["lcp_frequency_normalized"])
                percentile = math.nan
                if len(effort_frequency):
                    percentile = float(
                        ((effort_frequency < value).sum() + 0.5 * (effort_frequency == value).sum())
                        / len(effort_frequency)
                        * 100
                    )
                distance = math.nan
                if tree_search is not None:
                    distance = float(tree_search.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0] / 1000.0)
                rows.append(
                    {
                        "record_id": record["record_id"],
                        "target_year": year,
                        "month": int(record["month"]),
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "grid_id": grid_id,
                        "lcp_frequency_at_sighting": value,
                        "effort_footprint_percentile": percentile,
                        "inside_annual_top5pct": bool(value >= q95),
                        "distance_to_frequent_corridor_km": distance,
                        "strict_effort_background_cell_count": len(effort_frequency),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(tree["validation"] / "asamm_aug_oct_four_scenario_record_metrics.csv", index=False)
    annual = (
        metrics.groupby(["target_year", "scenario", "scenario_label"], as_index=False)
        .agg(
            record_count=("record_id", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
            strict_effort_background_cell_count=("strict_effort_background_cell_count", "first"),
        )
    )
    annual.to_csv(tree["validation"] / "asamm_aug_oct_four_scenario_by_year.csv", index=False)
    available = (
        metrics.groupby(["scenario", "scenario_label"], as_index=False)
        .agg(
            first_year=("target_year", "min"),
            last_year=("target_year", "max"),
            record_count=("record_id", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
        )
    )
    available.to_csv(tree["validation"] / "asamm_aug_oct_available_period_summary.csv", index=False)
    common = metrics.loc[metrics["target_year"].isin(VESSEL_YEARS)]
    common_summary = (
        common.groupby(["scenario", "scenario_label"], as_index=False)
        .agg(
            record_count=("record_id", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
        )
    )
    common_summary.to_csv(tree["validation"] / "asamm_aug_oct_common_2015_2019_summary.csv", index=False)
    qa = {
        "analysis_window": "August-October",
        "all_year_unique_sighting_records": int(sightings["record_id"].nunique()),
        "all_year_strict_effort_events": int(len(effort_events)),
        "common_2015_2019_unique_records": int(common["record_id"].nunique()),
        "common_record_count_by_scenario": {row.scenario: int(row.record_count) for row in common_summary.itertuples()},
        "common_record_counts_identical": bool(common_summary["record_count"].nunique() == 1),
        "validation_scope": "effort-conditioned spatial consistency; not direction, abundance, or behavioral response",
        "warnings": [],
    }
    if not qa["common_record_counts_identical"]:
        raise RuntimeError("Common-period ASAMM record counts differ among scenarios")
    write_json(tree["validation"] / "asamm_aug_oct_validation_qa.json", qa)
    return metrics, annual, common_summary


def run(mode: str) -> None:
    tree = output_tree(mode)
    ensure_tree(tree)
    years = [2019] if mode == "smoke" else ALL_YEARS
    paths = input_paths(years, mode)
    before = snapshot(paths)
    before.to_csv(tree["base"] / "input_integrity_before.csv", index=False)
    missing = before.loc[~before["exists"], "file_path"].tolist()
    if missing:
        raise FileNotFoundError(f"Missing Phase 8Q inputs: {missing}")
    write_json(
        tree["status"],
        {
            "phase": PHASE,
            "mode": mode,
            "status": "running",
            "started_utc": utc_now(),
            "stages": {},
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
        },
    )

    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    grid = grid.loc[grid["ocean_flag"] == 1].copy().reset_index(drop=True)
    endpoints = pd.read_csv(ENDPOINTS, dtype={"grid_id": str})
    topology = lcp_base.build_graph_topology(grid, endpoints)
    if topology.warnings:
        raise RuntimeError(f"Topology warnings: {topology.warnings}")
    primary_q99 = float(json.loads(SHIP_PARAMS.read_text(encoding="utf-8"))["log_q99"])
    scaling = pd.read_csv(PHASE8P_SCALING)
    sensitivity_q99 = float(scaling.loc[scaling["scaling_basis"] == "August-October only", "log_q99"].iloc[0])
    config = {
        "phase": PHASE,
        "mode": mode,
        "target_years": years,
        "vessel_years": [year for year in years if year in VESSEL_YEARS],
        "months": MONTHS,
        "prey_variable": "total_kJm2_frozen",
        "prey_temporal_resolution": "annual full trailing window",
        "ice_transform": {"sic_low": SIC_LOW, "sic_high": SIC_HIGH, "minimum_accessibility": A_MIN},
        "vessel_primary_log_q99": primary_q99,
        "vessel_sensitivity_log_q99": sensitivity_q99,
        "vessel_alpha": 0.5,
        "utility_floor_before_resistance": EPSILON,
        "resistance_formula": "max(-log(U + 0.001), 0)",
        "computational_orientation": "canonical south-to-north",
        "reported_orientation": "north-to-south after reversing path order",
        "graph_direction": "undirected",
        "directional_inference": False,
        "output_root": str(tree["base"].relative_to(ROOT)),
    }
    write_json(tree["base"] / "phase8q_config.json", config)
    set_stage(tree, "configuration", "completed", years=years, scenarios=SCENARIO_YEARS)

    dataset = xr.open_dataset(ICE)
    ice_cache = {}
    vessel_cache = {}
    ice_qa_rows = []
    vessel_qa_rows = []
    route_rows = []
    try:
        for year in years:
            ice, ice_qa = build_ice_surface(year, grid, dataset)
            ice_cache[year] = ice
            ice_qa_rows.append(ice_qa)
            ice.to_csv(tree["surfaces"] / f"ice_accessibility_aug_oct_target{year}_study_region.csv", index=False)
            if year in VESSEL_YEARS:
                vessel, vessel_qa = build_vessel_surface(year, grid, primary_q99, sensitivity_q99)
                vessel_cache[year] = vessel
                vessel_qa_rows.append(vessel_qa)
                vessel.to_csv(tree["surfaces"] / f"vessel_exposure_aug_oct_target{year}_study_region.csv", index=False)
        pd.DataFrame(ice_qa_rows).to_csv(tree["summaries"] / "phase8q_ice_surface_qa.csv", index=False)
        pd.DataFrame(vessel_qa_rows).to_csv(tree["summaries"] / "phase8q_vessel_surface_qa.csv", index=False)
        set_stage(tree, "seasonal_surfaces", "completed", ice_years=len(ice_cache), vessel_years=len(vessel_cache))

        for year in years:
            prey = load_prey(year, grid)
            for scenario, scenario_years in SCENARIO_YEARS.items():
                if year not in scenario_years:
                    continue
                surface = build_scenario_surface(scenario, year, prey, ice_cache[year], vessel_cache.get(year))
                lcp_grid, path_frame, route = run_scenario_lcp(scenario, year, surface, topology)
                write_outputs(tree, scenario, year, surface, lcp_grid, path_frame)
                route_rows.append(route)
        route_summary = pd.DataFrame(route_rows).sort_values(["scenario", "target_year"])
        route_summary.to_csv(tree["summaries"] / "phase8q_aug_oct_route_summary.csv", index=False)
        expected_runs = sum(year in scenario_years for year in years for scenario_years in SCENARIO_YEARS.values())
        if len(route_summary) != expected_runs or not (route_summary["successful_path_count"] == 1600).all():
            raise RuntimeError("Phase 8Q route count or path completion QA failed")
        direction_columns = [
            "reported_origins_all_northern",
            "reported_destinations_all_southern",
            "computational_sources_all_southern",
            "computational_destinations_all_northern",
        ]
        if not route_summary[direction_columns].all(axis=None):
            raise RuntimeError("Phase 8Q endpoint direction-label QA failed")
        if not (route_summary["duplicate_reported_path_hash_count"] == 0).all():
            raise RuntimeError("Phase 8Q duplicate path-hash QA failed")
        set_stage(tree, "lcp", "completed", scenario_year_runs=len(route_summary), successful_paths=int(route_summary["successful_path_count"].sum()))

        scenario_comparison = build_scenario_comparisons(tree, route_summary, years)
        seasonal_comparison = compare_with_early_season(tree, route_summary, years)
        prey_check = seasonal_comparison.loc[seasonal_comparison["scenario"] == "prey_only"]
        if not (prey_check["maximum_absolute_frequency_difference"] < 1e-12).all():
            raise RuntimeError("Canonical prey-only paths do not reproduce the established branch")
        set_stage(
            tree,
            "comparisons",
            "completed",
            scenario_comparison_rows=len(scenario_comparison),
            seasonal_comparison_rows=len(seasonal_comparison),
            prey_only_reproduction="pass",
        )

        if mode == "batch":
            _, annual_validation, common_validation = build_asamm_validation(tree, years)
            set_stage(
                tree,
                "validation",
                "completed",
                annual_rows=len(annual_validation),
                common_scenarios=len(common_validation),
            )
    except Exception as error:
        set_stage(tree, "failure", "failed", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        dataset.close()

    after = snapshot(paths)
    after.to_csv(tree["base"] / "input_integrity_after.csv", index=False)
    integrity = before.merge(after, on="file_path", suffixes=("_before", "_after"))
    integrity["unchanged"] = (
        integrity["exists_before"].eq(integrity["exists_after"])
        & integrity["size_bytes_before"].eq(integrity["size_bytes_after"])
        & integrity["mtime_ns_before"].eq(integrity["mtime_ns_after"])
    )
    integrity.to_csv(tree["base"] / "input_integrity_comparison.csv", index=False)
    if not integrity["unchanged"].all():
        raise RuntimeError("A read-only Phase 8Q input changed during execution")

    qa_name = "phase8q_smoke_qa_target2019.json" if mode == "smoke" else "phase8q_batch_qa.json"
    write_json(
        tree["base"] / qa_name,
        {
            "phase": PHASE,
            "mode": mode,
            "scenario_year_runs": int(len(route_summary)),
            "attempted_paths": int(route_summary["attempted_path_count"].sum()),
            "successful_paths": int(route_summary["successful_path_count"].sum()),
            "failed_paths": int(route_summary["failed_path_count"].sum()),
            "all_reported_origins_northern": bool(route_summary["reported_origins_all_northern"].all()),
            "all_reported_destinations_southern": bool(route_summary["reported_destinations_all_southern"].all()),
            "all_computational_sources_southern": bool(route_summary["computational_sources_all_southern"].all()),
            "all_computational_destinations_northern": bool(route_summary["computational_destinations_all_northern"].all()),
            "duplicate_reported_path_hash_count": int(route_summary["duplicate_reported_path_hash_count"].sum()),
            "prey_only_maximum_absolute_frequency_difference": float(
                seasonal_comparison.loc[
                    seasonal_comparison["scenario"] == "prey_only", "maximum_absolute_frequency_difference"
                ].max()
            ),
            "read_only_input_count": int(len(integrity)),
            "all_read_only_inputs_unchanged": bool(integrity["unchanged"].all()),
            "asamm_common_period_scenario_count": int(len(common_validation)) if mode == "batch" else None,
            "asamm_common_period_record_counts": (
                sorted(map(int, common_validation["record_count"].unique())) if mode == "batch" else None
            ),
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
            "warnings": [],
        },
    )

    output_files = sorted(path for path in tree["base"].rglob("*") if path.is_file() and path != tree["manifest"])
    write_json(
        tree["manifest"],
        {
            "phase": PHASE,
            "mode": mode,
            "created_utc": utc_now(),
            "created_by_script": SCRIPT,
            "input_files_unchanged": True,
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
            "output_files": [str(path.relative_to(ROOT)) for path in output_files],
        },
    )
    state = json.loads(tree["status"].read_text(encoding="utf-8"))
    state.update(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "input_files_unchanged": True,
            "decision": (
                "smoke_passed"
                if mode == "smoke"
                else "batch_complete_diagnostic_interpretation_documented_paper_integration_deferred"
            ),
        }
    )
    write_json(tree["status"], state)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": mode,
                "years": years,
                "scenario_year_runs": len(route_rows),
                "successful_paths": int(sum(row["successful_path_count"] for row in route_rows)),
                "inputs_unchanged": True,
                "output_root": str(tree["base"].relative_to(ROOT)),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "batch"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().mode)
