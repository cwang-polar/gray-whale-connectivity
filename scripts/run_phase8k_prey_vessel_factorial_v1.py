"""Run the isolated Phase 8K prey+vessel factorial diagnostic.

The branch uses the manuscript-primary full-trailing amphipod energy surface,
the existing May-July all-vessel modifier, the C3 grid, and the fixed 40 x 40
endpoint design. Existing scientific outputs are read-only; all new products
are written under outputs/prey_vessel_phase8k/.
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gray_whale_corridor.movement.lcp_ensemble import (
    build_weighted_adjacency,
    path_length_km,
    run_ensemble,
)
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base


TARGET_YEARS = list(range(2015, 2020))
PREY_VARIABLE = "total_kJm2_frozen"
PREY_LABEL = "kJm2_frozen"
EPSILON = 0.001
FREQUENT_THRESHOLD = 0.05

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
PREY_DIR = ROOT / "outputs/prey_v2"
SHIP_DIR = ROOT / "outputs/ship"
EXISTING_LCP = ROOT / "outputs/lcp_v2"
EXISTING_SURFACES = ROOT / "outputs/surfaces_v2"
ASAMM_SIGHTINGS = ROOT / "data/processed/validation/asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
ASAMM_EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"

OUT = ROOT / "outputs/prey_vessel_phase8k"
SURFACE_OUT = OUT / "surfaces"
LCP_OUT = OUT / "lcp"
COMPARE_OUT = OUT / "comparisons"
VALIDATION_OUT = OUT / "validation"
FIG_OUT = OUT / "figures"
STATUS_PATH = OUT / "phase8k_run_status.json"
MANIFEST_PATH = OUT / "phase8k_analysis_manifest.json"

SCENARIOS = {
    "prey_only": {
        "label": "Prey",
        "color": "#4C78A8",
        "grid": lambda y: EXISTING_LCP / f"prey_only_brian_{PREY_LABEL}_C3_lcp_target{y}_v2.csv",
        "path": EXISTING_LCP / f"prey_only_brian_{PREY_LABEL}_C3_path_summary_v2.csv",
        "route": EXISTING_LCP / "prey_only_brian_C3_route_summary_v2.csv",
    },
    "prey_ice": {
        "label": "Prey + ice",
        "color": "#59A14F",
        "grid": lambda y: EXISTING_LCP / f"prey_ice_brian_{PREY_LABEL}_C3_lcp_target{y}_v2.csv",
        "path": EXISTING_LCP / f"prey_ice_brian_{PREY_LABEL}_C3_path_summary_v2.csv",
        "route": EXISTING_LCP / "prey_ice_brian_C3_route_summary_v2.csv",
    },
    "prey_vessel": {
        "label": "Prey + vessel",
        "color": "#F28E2B",
        "grid": lambda y: LCP_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_lcp_target{y}_v2.csv",
        "path": LCP_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_path_summary_v2.csv",
        "route": LCP_OUT / "prey_vessel_brian_C3_route_summary_v2.csv",
    },
    "prey_ice_vessel": {
        "label": "Prey + ice + vessel",
        "color": "#B55A9D",
        "grid": lambda y: EXISTING_LCP / f"prey_ice_ship_brian_{PREY_LABEL}_C3_lcp_target{y}_v2.csv",
        "path": EXISTING_LCP / f"prey_ice_ship_brian_{PREY_LABEL}_C3_path_summary_v2.csv",
        "route": EXISTING_LCP / "prey_ice_ship_brian_C3_route_summary_v2.csv",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (OUT, SURFACE_OUT, LCP_OUT, COMPARE_OUT, VALIDATION_OUT, FIG_OUT):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {
        "phase": "Phase 8K prey+vessel factorial diagnostic",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "stages": {
            name: {"status": "pending"}
            for name in ("audit", "surfaces", "smoke", "batch", "comparisons", "validation", "figures", "records")
        },
    }


def set_stage(stage: str, state: str, **details: object) -> None:
    status = load_status()
    entry = status["stages"].setdefault(stage, {})
    entry["status"] = state
    entry[f"{state}_utc"] = utc_now()
    entry.update(details)
    status["updated_utc"] = utc_now()
    write_json(STATUS_PATH, status)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prey_path(year: int) -> Path:
    return PREY_DIR / f"amphipod_prey_surface_{PREY_VARIABLE}_target{year}_full_trailing_C3_v2.csv"


def ship_path(year: int) -> Path:
    return SHIP_DIR / f"ship_intensity_MayJul_target{year}.csv"


def integrity_inputs() -> list[Path]:
    paths = [GRID, ENDPOINTS, ASAMM_SIGHTINGS, ASAMM_EFFORT]
    paths.extend(prey_path(y) for y in TARGET_YEARS)
    paths.extend(ship_path(y) for y in TARGET_YEARS)
    for scenario in ("prey_only", "prey_ice", "prey_ice_vessel"):
        paths.append(SCENARIOS[scenario]["path"])
        paths.append(SCENARIOS[scenario]["route"])
        paths.extend(SCENARIOS[scenario]["grid"](y) for y in TARGET_YEARS)
    return list(dict.fromkeys(paths))


def write_integrity_snapshot(name: str) -> pd.DataFrame:
    rows = []
    for path in integrity_inputs():
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "file_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    frame = pd.DataFrame(rows).sort_values("file_path")
    frame.to_csv(OUT / f"input_integrity_{name}.csv", index=False)
    return frame


def audit_inputs() -> None:
    set_stage("audit", "running")
    before = write_integrity_snapshot("before")
    grid = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    summary = {
        "phase": "Phase 8K",
        "analysis_role": "Four-scenario comparison on the common 2015-2019 period",
        "target_years": TARGET_YEARS,
        "prey_variable": PREY_VARIABLE,
        "prey_surface_convention": "full trailing five-year support",
        "utility_formula": "U_prey_vessel = U_prey_global * D_vessel_alpha0_5_MayJul_mean",
        "resistance_formula": "R = max(-log(U + 0.001), 0)",
        "grid_rows": int(len(grid)),
        "ocean_domain_rows": int(((grid["ocean_flag"] == 1) & grid["inside_C3_domain"].astype(bool)).sum()),
        "source_endpoints": int((endpoints["endpoint_type"] == "source").sum()),
        "destination_endpoints": int((endpoints["endpoint_type"] == "destination").sum()),
        "attempted_paths_per_year": int((endpoints["endpoint_type"] == "source").sum() * (endpoints["endpoint_type"] == "destination").sum()),
        "input_file_count": int(len(before)),
        "python": sys.version,
        "platform": platform.platform(),
        "created_utc": utc_now(),
        "existing_results_are_read_only": True,
    }
    write_json(MANIFEST_PATH, summary)
    set_stage("audit", "completed", input_file_count=len(before), manifest=str(MANIFEST_PATH.relative_to(ROOT)))


def resistance(utility: pd.Series | np.ndarray) -> np.ndarray:
    return np.maximum(-np.log(np.asarray(utility, dtype=float) + EPSILON), 0.0)


def build_surface(year: int) -> tuple[pd.DataFrame, dict]:
    grid = pd.read_csv(GRID)
    prey = pd.read_csv(prey_path(year))
    vessel = pd.read_csv(ship_path(year))
    for frame in (grid, prey, vessel):
        frame["grid_id"] = frame["grid_id"].astype(str)
    base = grid.loc[(grid["ocean_flag"] == 1) & grid["inside_C3_domain"].astype(bool), [
        "grid_id", "grid_x", "grid_y", "x", "y", "lon", "lat", "ocean_flag", "inside_C3_domain"
    ]].copy()
    merged = base.merge(
        prey[["grid_id", "U_amphipod_prey_global_q05_q95", "supported_for_surface", "reliability_flag"]],
        on="grid_id",
        how="left",
        indicator="prey_merge",
    )
    vessel_columns = [
        "grid_id", "S_ship_all_MayJul_mean", "S_ship_all_MayJul_sum",
        "S_scaled_ship_primary_MayJul_mean", "D_ship_alpha0_5_MayJul_mean", "month_count",
    ]
    merged = merged.merge(vessel[vessel_columns], on="grid_id", how="left", indicator="vessel_merge")
    missing_prey = int((merged["prey_merge"] == "left_only").sum())
    missing_vessel = int((merged["vessel_merge"] == "left_only").sum())
    if missing_prey or missing_vessel:
        raise RuntimeError(f"{year}: missing prey={missing_prey}, vessel={missing_vessel}")

    u_prey = pd.to_numeric(merged["U_amphipod_prey_global_q05_q95"], errors="coerce").fillna(EPSILON).clip(EPSILON, 1.0)
    d_vessel = pd.to_numeric(merged["D_ship_alpha0_5_MayJul_mean"], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    s_scaled = pd.to_numeric(merged["S_scaled_ship_primary_MayJul_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    u_pv = (u_prey * d_vessel).clip(EPSILON, 1.0)
    r_prey = resistance(u_prey)
    r_pv = resistance(u_pv)
    out = pd.DataFrame({
        "grid_id": merged["grid_id"],
        "grid_x": merged["grid_x"],
        "grid_y": merged["grid_y"],
        "x": merged["x"],
        "y": merged["y"],
        "lon": merged["lon"],
        "lat": merged["lat"],
        "target_year": year,
        "prey_variable": PREY_VARIABLE,
        "U_prey_global": u_prey,
        "S_vessel_all_MayJul_mean": pd.to_numeric(merged["S_ship_all_MayJul_mean"], errors="coerce"),
        "S_vessel_all_MayJul_sum": pd.to_numeric(merged["S_ship_all_MayJul_sum"], errors="coerce"),
        "S_scaled_vessel_primary_MayJul_mean": s_scaled,
        "D_vessel_alpha0_5_MayJul_mean": d_vessel,
        "U_prey_vessel_alpha0_5": u_pv,
        "R_log_prey": r_prey,
        "R_log_prey_vessel_alpha0_5": r_pv,
        "delta_R_log_vessel_alpha0_5": r_pv - r_prey,
        "absolute_utility_loss_vessel_alpha0_5": u_prey - u_pv,
        "relative_utility_loss_vessel_alpha0_5": np.where(u_prey > 0, (u_prey - u_pv) / u_prey, np.nan),
        "month_count_vessel": pd.to_numeric(merged["month_count"], errors="coerce"),
        "supported_for_surface": merged["supported_for_surface"].astype(bool),
        "reliability_flag": merged["reliability_flag"],
        "ocean_flag": merged["ocean_flag"],
        "inside_C3_domain": merged["inside_C3_domain"].astype(bool),
    })
    qa = {
        "target_year": year,
        "grid_cells": int(len(out)),
        "missing_prey_cells": missing_prey,
        "missing_vessel_cells": missing_vessel,
        "utility_exceeds_prey_count": int((out["U_prey_vessel_alpha0_5"] > out["U_prey_global"] + 1e-12).sum()),
        "resistance_below_prey_count": int((out["R_log_prey_vessel_alpha0_5"] < out["R_log_prey"] - 1e-12).sum()),
        "mean_U_prey": float(out["U_prey_global"].mean()),
        "mean_D_vessel": float(out["D_vessel_alpha0_5_MayJul_mean"].mean()),
        "mean_U_prey_vessel": float(out["U_prey_vessel_alpha0_5"].mean()),
        "mean_delta_R_vessel": float(out["delta_R_log_vessel_alpha0_5"].mean()),
    }
    if qa["utility_exceeds_prey_count"] or qa["resistance_below_prey_count"]:
        raise RuntimeError(f"{year}: prey+vessel monotonicity check failed")
    return out, qa


def build_surfaces() -> None:
    set_stage("surfaces", "running")
    summaries = []
    for year in TARGET_YEARS:
        surface, qa = build_surface(year)
        surface.to_csv(SURFACE_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_utility_target{year}_v2.csv", index=False)
        summaries.append(qa)
    pd.DataFrame(summaries).to_csv(SURFACE_OUT / "prey_vessel_brian_C3_utility_summary_v2.csv", index=False)
    write_json(SURFACE_OUT / "prey_vessel_brian_C3_utility_qa_v2.json", {
        "phase": "Phase 8K.1",
        "target_years": TARGET_YEARS,
        "prey_variable": PREY_VARIABLE,
        "formula": "U_prey_vessel = U_prey_global * D_vessel_alpha0_5_MayJul_mean",
        "annual": summaries,
        "warnings": [],
    })
    set_stage("surfaces", "completed", years=TARGET_YEARS, warnings=[])


def load_topology():
    ocean = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    return lcp_base.build_graph_topology(ocean_mask=ocean, endpoints=endpoints), endpoints


def prepare_surface_for_topology(surface: pd.DataFrame, topology, utility_col: str, resistance_col: str) -> pd.DataFrame:
    frame = topology.grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"))
    frame = frame.loc[frame["ocean_flag"] == 1].copy().reset_index(drop=True)
    if frame[utility_col].isna().any():
        raise RuntimeError(f"Missing {utility_col} on topology grid")
    frame[utility_col] = pd.to_numeric(frame[utility_col], errors="coerce").fillna(EPSILON).clip(EPSILON, 1.0)
    frame[resistance_col] = resistance(frame[utility_col])
    return frame


def run_year(year: int, topology) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    surface = pd.read_csv(SURFACE_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_utility_target{year}_v2.csv")
    grid = prepare_surface_for_topology(surface, topology, "U_prey_vessel_alpha0_5", "R_log_prey_vessel_alpha0_5")
    adjacency = build_weighted_adjacency(topology.edge_pairs, grid["R_log_prey_vessel_alpha0_5"].to_numpy())
    paths, counts = run_ensemble(adjacency, topology)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    if len(paths) != attempted:
        raise RuntimeError(f"{year}: only {len(paths)}/{attempted} paths succeeded")
    grid["lcp_count"] = counts
    grid["lcp_frequency_normalized"] = counts / len(paths)
    path_rows = []
    for item in paths:
        sub = grid.iloc[list(item.node_indices)]
        path_rows.append({
            "prey_variable": PREY_VARIABLE,
            "target_year": year,
            "source_grid_id": topology.grid_id_by_node_index[item.source_index],
            "destination_grid_id": topology.grid_id_by_node_index[item.destination_index],
            "path_cost": item.cost,
            "path_length_km": path_length_km(item.node_indices, grid),
            "path_cell_count": len(item.node_indices),
            "path_mean_U_prey": float(sub["U_prey_global"].mean()),
            "path_mean_D_vessel": float(sub["D_vessel_alpha0_5_MayJul_mean"].mean()),
            "path_mean_S_scaled_vessel": float(sub["S_scaled_vessel_primary_MayJul_mean"].mean()),
            "path_mean_U_prey_vessel": float(sub["U_prey_vessel_alpha0_5"].mean()),
            "path_mean_delta_R_vessel": float(sub["delta_R_log_vessel_alpha0_5"].mean()),
            "path_min_D_vessel": float(sub["D_vessel_alpha0_5_MayJul_mean"].min()),
            "path_min_U_prey_vessel": float(sub["U_prey_vessel_alpha0_5"].min()),
            "path_S_scaled_vessel_gt_0_50_cell_count": int((sub["S_scaled_vessel_primary_MayJul_mean"] > 0.50).sum()),
        })
    path_frame = pd.DataFrame(path_rows)
    frequency = grid["lcp_frequency_normalized"]
    top_index = frequency.idxmax()
    route = {
        "prey_variable": PREY_VARIABLE,
        "target_year": year,
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "attempted_path_count": attempted,
        "successful_path_count": len(path_frame),
        "failed_path_count": attempted - len(path_frame),
        "mean_path_cost": float(path_frame["path_cost"].mean()),
        "median_path_cost": float(path_frame["path_cost"].median()),
        "min_path_cost": float(path_frame["path_cost"].min()),
        "max_path_cost": float(path_frame["path_cost"].max()),
        "mean_path_length_km": float(path_frame["path_length_km"].mean()),
        "median_path_length_km": float(path_frame["path_length_km"].median()),
        "mean_path_D_vessel": float(path_frame["path_mean_D_vessel"].mean()),
        "mean_path_S_scaled_vessel": float(path_frame["path_mean_S_scaled_vessel"].mean()),
        "mean_path_U_prey_vessel": float(path_frame["path_mean_U_prey_vessel"].mean()),
        "mean_path_delta_R_vessel": float(path_frame["path_mean_delta_R_vessel"].mean()),
        "corridor_area_cells_freq_gt_0": int((frequency > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int((frequency > FREQUENT_THRESHOLD).sum()),
        "corridor_area_cells_freq_gt_0_10": int((frequency > 0.10).sum()),
        "top_bottleneck_frequency": float(frequency.max()),
        "top_bottleneck_grid_id": str(grid.loc[top_index, "grid_id"]),
        "warning_flags": "",
    }
    keep = [
        "grid_id", "x", "y", "lon", "lat", "target_year", "prey_variable", "U_prey_global",
        "D_vessel_alpha0_5_MayJul_mean", "S_scaled_vessel_primary_MayJul_mean",
        "U_prey_vessel_alpha0_5", "R_log_prey_vessel_alpha0_5", "delta_R_log_vessel_alpha0_5",
        "lcp_count", "lcp_frequency_normalized", "ocean_flag",
    ]
    out_grid = grid[keep].copy()
    out_grid["route_pair"] = "main_south_to_north_C3_v2"
    return out_grid, path_frame, route


def smoke_test() -> None:
    set_stage("smoke", "running")
    topology, _ = load_topology()
    year = 2019
    existing_surface = pd.read_csv(EXISTING_SURFACES / f"prey_ice_ship_brian_{PREY_LABEL}_C3_utility_target{year}_v2.csv")
    existing_grid = prepare_surface_for_topology(existing_surface, topology, "U_prey_ice_ship_alpha0_5", "R_log_prey_ice_ship_alpha0_5")
    adjacency = build_weighted_adjacency(topology.edge_pairs, existing_grid["R_log_prey_ice_ship_alpha0_5"].to_numpy())
    paths, counts = run_ensemble(adjacency, topology)
    reference_route = pd.read_csv(EXISTING_LCP / "prey_ice_ship_brian_C3_route_summary_v2.csv")
    reference_route = reference_route.loc[(reference_route["prey_variable"] == PREY_VARIABLE) & (reference_route["target_year"] == year)].iloc[0]
    reference_grid = pd.read_csv(SCENARIOS["prey_ice_vessel"]["grid"](year))
    computed_frequency = counts / len(paths)
    freq_frame = topology.grid[["grid_id"]].copy()
    freq_frame["computed"] = computed_frequency
    check = freq_frame.merge(reference_grid[["grid_id", "lcp_frequency_normalized"]], on="grid_id", how="inner")
    mean_cost = float(np.mean([p.cost for p in paths]))
    cost_error = abs(mean_cost - float(reference_route["mean_path_cost"]))
    frequency_error = float(np.max(np.abs(check["computed"] - check["lcp_frequency_normalized"])))
    if cost_error > 1e-8 or frequency_error > 1e-12 or len(paths) != 1600:
        raise RuntimeError(f"Existing-engine equivalence failed: cost={cost_error}, freq={frequency_error}, paths={len(paths)}")

    pv_grid, pv_paths, pv_route = run_year(year, topology)
    prey_paths = pd.read_csv(SCENARIOS["prey_only"]["path"])
    prey_paths = prey_paths.loc[(prey_paths["prey_variable"] == PREY_VARIABLE) & (prey_paths["target_year"] == year), [
        "source_grid_id", "destination_grid_id", "path_cost"
    ]].rename(columns={"path_cost": "prey_path_cost"})
    paired = pv_paths.merge(prey_paths, on=["source_grid_id", "destination_grid_id"], how="inner")
    lower_count = int((paired["path_cost"] < paired["prey_path_cost"] - 1e-8).sum())
    if len(paired) != 1600 or lower_count:
        raise RuntimeError(f"Prey+vessel monotonicity failed: paired={len(paired)}, lower={lower_count}")
    smoke = {
        "target_year": year,
        "existing_engine_successful_paths": len(paths),
        "existing_engine_mean_cost_absolute_error": cost_error,
        "existing_engine_max_frequency_absolute_error": frequency_error,
        "prey_vessel_successful_paths": int(pv_route["successful_path_count"]),
        "prey_vessel_mean_path_cost": pv_route["mean_path_cost"],
        "prey_vessel_corridor_area_freq_gt_0_05": pv_route["corridor_area_cells_freq_gt_0_05"],
        "paired_paths": len(paired),
        "prey_vessel_paths_below_prey_only": lower_count,
        "warnings": [],
    }
    write_json(OUT / "phase8k_smoke_test_2019.json", smoke)
    set_stage("smoke", "completed", **smoke)


def batch_lcp() -> None:
    set_stage("batch", "running")
    topology, _ = load_topology()
    grids = []
    paths = []
    routes = []
    for year in TARGET_YEARS:
        grid, path_frame, route = run_year(year, topology)
        grid.to_csv(LCP_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_lcp_target{year}_v2.csv", index=False)
        grids.append(grid)
        paths.append(path_frame)
        routes.append(route)
        set_stage("batch", "running", last_completed_year=year, successful_paths=route["successful_path_count"])
    pd.concat(paths, ignore_index=True).to_csv(LCP_OUT / f"prey_vessel_brian_{PREY_LABEL}_C3_path_summary_v2.csv", index=False)
    route_frame = pd.DataFrame(routes)
    route_frame.to_csv(LCP_OUT / "prey_vessel_brian_C3_route_summary_v2.csv", index=False)
    qa = {
        "phase": "Phase 8K.3",
        "target_years": TARGET_YEARS,
        "prey_variable": PREY_VARIABLE,
        "successful_paths_by_year": dict(zip(route_frame["target_year"].astype(str), route_frame["successful_path_count"].astype(int))),
        "failed_paths_total": int(route_frame["failed_path_count"].sum()),
        "topology_warnings": list(topology.warnings),
        "warnings": [w for w in route_frame["warning_flags"].tolist() if w],
    }
    write_json(LCP_OUT / "prey_vessel_brian_C3_lcp_qa_v2.json", qa)
    set_stage("batch", "completed", years=TARGET_YEARS, successful_paths_total=int(route_frame["successful_path_count"].sum()), failed_paths_total=int(route_frame["failed_path_count"].sum()))


def load_route(scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(SCENARIOS[scenario]["route"])
    return frame.loc[(frame["prey_variable"] == PREY_VARIABLE) & frame["target_year"].isin(TARGET_YEARS)].copy()


def load_path_summary(scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(SCENARIOS[scenario]["path"])
    return frame.loc[(frame["prey_variable"] == PREY_VARIABLE) & frame["target_year"].isin(TARGET_YEARS), [
        "target_year", "source_grid_id", "destination_grid_id", "path_cost"
    ]].copy()


def load_grid(scenario: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(SCENARIOS[scenario]["grid"](year))
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame


def pct(new: float, old: float) -> float:
    return float((new - old) / old * 100.0) if old else math.nan


def jaccard(a: pd.Series, b: pd.Series) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else math.nan


def centroid_shift_km(a: pd.DataFrame, b: pd.DataFrame) -> float:
    aa = a.loc[a["lcp_frequency_normalized"] > FREQUENT_THRESHOLD, ["x", "y"]]
    bb = b.loc[b["lcp_frequency_normalized"] > FREQUENT_THRESHOLD, ["x", "y"]]
    if aa.empty or bb.empty:
        return math.nan
    return float(np.linalg.norm(aa.mean().to_numpy() - bb.mean().to_numpy()) / 1000.0)


def build_comparisons() -> None:
    set_stage("comparisons", "running")
    routes = {scenario: load_route(scenario).set_index("target_year") for scenario in SCENARIOS}
    annual_rows = []
    for year in TARGET_YEARS:
        for scenario in SCENARIOS:
            row = routes[scenario].loc[year]
            annual_rows.append({
                "target_year": year,
                "scenario": scenario,
                "scenario_label": SCENARIOS[scenario]["label"],
                "mean_path_cost": row["mean_path_cost"],
                "median_path_cost": row["median_path_cost"],
                "corridor_area_cells_freq_gt_0_05": row["corridor_area_cells_freq_gt_0_05"],
                "top_bottleneck_frequency": row["top_bottleneck_frequency"],
                "successful_path_count": row["successful_path_count"],
            })
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(COMPARE_OUT / "four_scenario_annual_summary_2015_2019_v2.csv", index=False)

    conditional = []
    for year in TARGET_YEARS:
        p = routes["prey_only"].loc[year]
        pi = routes["prey_ice"].loc[year]
        pv = routes["prey_vessel"].loc[year]
        piv = routes["prey_ice_vessel"].loc[year]
        vessel_without_ice = pct(pv["mean_path_cost"], p["mean_path_cost"])
        vessel_with_ice = pct(piv["mean_path_cost"], pi["mean_path_cost"])
        ice_without_vessel = pct(pi["mean_path_cost"], p["mean_path_cost"])
        ice_with_vessel = pct(piv["mean_path_cost"], pv["mean_path_cost"])
        conditional.append({
            "target_year": year,
            "vessel_cost_effect_without_ice_percent": vessel_without_ice,
            "vessel_cost_effect_with_ice_percent": vessel_with_ice,
            "conditional_difference_vessel_percent_points": vessel_with_ice - vessel_without_ice,
            "ice_cost_effect_without_vessel_percent": ice_without_vessel,
            "ice_cost_effect_with_vessel_percent": ice_with_vessel,
            "conditional_difference_ice_percent_points": ice_with_vessel - ice_without_vessel,
            "vessel_area_effect_without_ice_cells": pv["corridor_area_cells_freq_gt_0_05"] - p["corridor_area_cells_freq_gt_0_05"],
            "vessel_area_effect_with_ice_cells": piv["corridor_area_cells_freq_gt_0_05"] - pi["corridor_area_cells_freq_gt_0_05"],
            "ice_area_effect_without_vessel_cells": pi["corridor_area_cells_freq_gt_0_05"] - p["corridor_area_cells_freq_gt_0_05"],
            "ice_area_effect_with_vessel_cells": piv["corridor_area_cells_freq_gt_0_05"] - pv["corridor_area_cells_freq_gt_0_05"],
        })
    pd.DataFrame(conditional).to_csv(COMPARE_OUT / "vessel_conditional_effect_comparison_v2.csv", index=False)

    keys = ["target_year", "source_grid_id", "destination_grid_id"]
    paired = None
    for scenario in SCENARIOS:
        frame = load_path_summary(scenario).rename(columns={"path_cost": f"{scenario}_path_cost"})
        paired = frame if paired is None else paired.merge(frame, on=keys, how="inner", validate="one_to_one")
    paired["vessel_delta_without_ice"] = paired["prey_vessel_path_cost"] - paired["prey_only_path_cost"]
    paired["vessel_delta_with_ice"] = paired["prey_ice_vessel_path_cost"] - paired["prey_ice_path_cost"]
    paired["ice_delta_without_vessel"] = paired["prey_ice_path_cost"] - paired["prey_only_path_cost"]
    paired["ice_delta_with_vessel"] = paired["prey_ice_vessel_path_cost"] - paired["prey_vessel_path_cost"]
    paired.to_csv(COMPARE_OUT / "four_scenario_paired_path_differences_v2.csv", index=False)

    pair_defs = [
        ("prey_only", "prey_vessel", "vessel_effect_without_ice"),
        ("prey_ice", "prey_ice_vessel", "vessel_effect_with_ice"),
        ("prey_only", "prey_ice", "ice_effect_without_vessel"),
        ("prey_vessel", "prey_ice_vessel", "ice_effect_with_vessel"),
        ("prey_only", "prey_ice_vessel", "combined_effect"),
    ]
    spatial_rows = []
    for year in TARGET_YEARS:
        cache = {scenario: load_grid(scenario, year) for scenario in SCENARIOS}
        for left, right, comparison in pair_defs:
            a = cache[left]
            b = cache[right]
            merged = a[["grid_id", "lcp_frequency_normalized"]].merge(
                b[["grid_id", "lcp_frequency_normalized"]], on="grid_id", suffixes=("_a", "_b"), validate="one_to_one"
            )
            spatial_rows.append({
                "target_year": year,
                "comparison": comparison,
                "scenario_a": left,
                "scenario_b": right,
                "frequency_map_pearson_r": float(merged["lcp_frequency_normalized_a"].corr(merged["lcp_frequency_normalized_b"])),
                "jaccard_freq_gt_0_05": jaccard(merged["lcp_frequency_normalized_a"] > FREQUENT_THRESHOLD, merged["lcp_frequency_normalized_b"] > FREQUENT_THRESHOLD),
                "centroid_shift_freq_gt_0_05_km": centroid_shift_km(a, b),
            })
    pd.DataFrame(spatial_rows).to_csv(COMPARE_OUT / "four_scenario_corridor_overlap_centroid_v2.csv", index=False)
    set_stage("comparisons", "completed", annual_rows=len(annual), paired_path_rows=len(paired), spatial_rows=len(spatial_rows))


def validate_asamm() -> None:
    set_stage("validation", "running")
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    sightings = sightings.loc[
        sightings["use_for_C3_validation"].astype(bool)
        & sightings["year"].isin(TARGET_YEARS)
        & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)
    effort = pd.read_csv(ASAMM_EFFORT)
    effort = effort.loc[
        effort["year"].isin(TARGET_YEARS)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    effort["grid_id"] = effort["grid_id"].astype(str)
    rows = []
    for year in TARGET_YEARS:
        annual_sightings = sightings.loc[sightings["year"] == year]
        effort_ids = set(effort.loc[effort["year"] == year, "grid_id"])
        for scenario in SCENARIOS:
            grid = load_grid(scenario, year)
            frequency = grid["lcp_frequency_normalized"].fillna(0.0)
            q95 = float(frequency.quantile(0.95))
            frequent_xy = grid.loc[frequency > FREQUENT_THRESHOLD, ["x", "y"]].to_numpy(dtype=float)
            tree = cKDTree(frequent_xy) if len(frequent_xy) else None
            lookup = grid.set_index("grid_id")
            effort_frequency = lookup.loc[lookup.index.intersection(effort_ids), "lcp_frequency_normalized"].fillna(0.0)
            for _, rec in annual_sightings.iterrows():
                grid_id = rec["nearest_grid_id"]
                if grid_id not in lookup.index:
                    continue
                cell = lookup.loc[grid_id]
                freq = float(cell["lcp_frequency_normalized"])
                percentile = math.nan
                if len(effort_frequency):
                    percentile = float(((effort_frequency < freq).sum() + 0.5 * (effort_frequency == freq).sum()) / len(effort_frequency) * 100.0)
                distance = math.nan
                if tree is not None:
                    distance = float(tree.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0] / 1000.0)
                rows.append({
                    "record_id": rec["record_id"],
                    "target_year": year,
                    "scenario": scenario,
                    "scenario_label": SCENARIOS[scenario]["label"],
                    "grid_id": grid_id,
                    "lcp_frequency_at_sighting": freq,
                    "effort_footprint_percentile": percentile,
                    "inside_annual_top5pct": bool(freq >= q95),
                    "distance_to_frequent_corridor_km": distance,
                    "strict_effort_background_cell_count": len(effort_frequency),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(VALIDATION_OUT / "asamm_four_scenario_record_metrics_v2.csv", index=False)
    annual = (
        metrics.groupby(["target_year", "scenario", "scenario_label"], as_index=False)
        .agg(
            record_count=("record_id", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100.0)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
            strict_effort_background_cell_count=("strict_effort_background_cell_count", "first"),
        )
    )
    annual.to_csv(VALIDATION_OUT / "asamm_four_scenario_common_period_by_year_v2.csv", index=False)
    overall = (
        metrics.groupby(["scenario", "scenario_label"], as_index=False)
        .agg(
            record_count=("record_id", "nunique"),
            mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
            percent_records_top5pct=("inside_annual_top5pct", lambda x: float(np.mean(x) * 100.0)),
            median_distance_to_frequent_corridor_km=("distance_to_frequent_corridor_km", "median"),
        )
    )
    overall.to_csv(VALIDATION_OUT / "asamm_four_scenario_common_period_summary_v2.csv", index=False)
    counts = overall.set_index("scenario")["record_count"]
    qa = {
        "phase": "Phase 8K.5",
        "years": TARGET_YEARS,
        "months": [6, 7],
        "unique_sighting_records": int(sightings["record_id"].nunique()),
        "record_count_by_scenario": {k: int(v) for k, v in counts.items()},
        "identical_record_count_across_scenarios": bool(counts.nunique() == 1),
        "metric_scope": "effort-footprint-aware observed-use consistency; not abundance or behavioral response",
        "warnings": [],
    }
    if not qa["identical_record_count_across_scenarios"]:
        raise RuntimeError("ASAMM scenarios did not retain an identical record set")
    write_json(VALIDATION_OUT / "asamm_four_scenario_common_period_qa_v2.json", qa)
    set_stage("validation", "completed", records=qa["unique_sighting_records"], identical_records=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    base = FIG_OUT / stem
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_figures() -> None:
    set_stage("figures", "running")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    annual = pd.read_csv(COMPARE_OUT / "four_scenario_annual_summary_2015_2019_v2.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for scenario, meta in SCENARIOS.items():
        sub = annual.loc[annual["scenario"] == scenario]
        axes[0].plot(sub["target_year"], sub["mean_path_cost"], marker="o", ms=3.2, lw=1.4, color=meta["color"], label=meta["label"])
        axes[1].plot(sub["target_year"], sub["corridor_area_cells_freq_gt_0_05"], marker="o", ms=3.2, lw=1.4, color=meta["color"], label=meta["label"])
    axes[0].set_ylabel("Mean modeled path cost")
    axes[1].set_ylabel("Frequent-corridor cells")
    for ax in axes:
        ax.set_xlabel("Year")
        ax.set_xticks(TARGET_YEARS)
        ax.grid(color="#dddddd", linewidth=0.5)
    axes[0].legend(loc="best")
    axes[0].text(-0.12, 1.02, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=8)
    axes[1].text(-0.12, 1.02, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=8)
    save_figure(fig, "phase8k_four_scenario_cost_area_v1")
    plt.close(fig)

    grids = {year: load_grid("prey_vessel", year) for year in TARGET_YEARS}
    vmax = max(frame["lcp_frequency_normalized"].max() for frame in grids.values())
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.35), sharex=True, sharey=True, constrained_layout=True)
    artist = None
    for ax, year in zip(axes, TARGET_YEARS):
        frame = grids[year]
        artist = ax.scatter(frame["lon"], frame["lat"], c=frame["lcp_frequency_normalized"], s=4, cmap="magma", vmin=0, vmax=vmax, linewidth=0)
        ax.set_title(str(year), fontsize=7)
        ax.set_xlim(-176.1, -156.4)
        ax.set_ylim(62.7, 73.4)
        ax.set_xlabel("Longitude")
        ax.grid(color="#eeeeee", linewidth=0.4)
    axes[0].set_ylabel("Latitude")
    fig.colorbar(artist, ax=axes, fraction=0.025, pad=0.02, label="Normalized LCP frequency")
    save_figure(fig, "phase8k_prey_vessel_corridor_maps_2015_2019_v1")
    plt.close(fig)

    validation = pd.read_csv(VALIDATION_OUT / "asamm_four_scenario_common_period_summary_v2.csv")
    order = list(SCENARIOS)
    validation["order"] = validation["scenario"].map({s: i for i, s in enumerate(order)})
    validation = validation.sort_values("order")
    colors = [SCENARIOS[s]["color"] for s in validation["scenario"]]
    labels = [SCENARIOS[s]["label"] for s in validation["scenario"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    values = [
        ("mean_effort_footprint_percentile", "Mean effort-footprint\npercentile"),
        ("percent_records_top5pct", "Sightings in annual\ntop 5% (%)"),
        ("median_distance_to_frequent_corridor_km", "Median distance to\nfrequent corridor (km)"),
    ]
    for index, (column, ylabel) in enumerate(values):
        axes[index].bar(np.arange(len(validation)), validation[column], color=colors, width=0.72)
        axes[index].set_ylabel(ylabel)
        axes[index].set_xticks(np.arange(len(validation)), labels, rotation=28, ha="right")
        axes[index].grid(axis="y", color="#e1e1e1", linewidth=0.5)
        axes[index].text(-0.13, 1.02, chr(65 + index), transform=axes[index].transAxes, fontweight="bold", fontsize=8)
    save_figure(fig, "phase8k_asamm_four_scenario_common_period_v1")
    plt.close(fig)
    set_stage("figures", "completed", figure_stems=[
        "phase8k_four_scenario_cost_area_v1",
        "phase8k_prey_vessel_corridor_maps_2015_2019_v1",
        "phase8k_asamm_four_scenario_common_period_v1",
    ])


def finalize_records() -> None:
    set_stage("records", "running")
    before = pd.read_csv(OUT / "input_integrity_before.csv")
    after = write_integrity_snapshot("after")
    check = before.merge(after, on="file_path", suffixes=("_before", "_after"), validate="one_to_one")
    check["unchanged"] = (
        (check["size_bytes_before"] == check["size_bytes_after"])
        & (check["sha256_before"] == check["sha256_after"])
    )
    check.to_csv(OUT / "input_integrity_comparison.csv", index=False)
    if not bool(check["unchanged"].all()):
        raise RuntimeError("One or more frozen inputs changed during Phase 8K")
    inventory_paths = [
        path for path in sorted(OUT.rglob("*"))
        if path.is_file() and not path.name.endswith(".tmp") and path.name != "phase8k_output_inventory.csv"
    ]
    set_stage("records", "completed", frozen_inputs_unchanged=True, output_file_count=len(inventory_paths))
    rows = [{
        "file_path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in inventory_paths]
    pd.DataFrame(rows).to_csv(OUT / "phase8k_output_inventory.csv", index=False)


def run_all() -> None:
    ensure_dirs()
    audit_inputs()
    build_surfaces()
    smoke_test()
    batch_lcp()
    build_comparisons()
    validate_asamm()
    make_figures()
    finalize_records()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "audit", "surfaces", "smoke", "batch", "comparisons", "validation", "figures", "records"], default="all")
    args = parser.parse_args()
    ensure_dirs()
    actions = {
        "audit": audit_inputs,
        "surfaces": build_surfaces,
        "smoke": smoke_test,
        "batch": batch_lcp,
        "comparisons": build_comparisons,
        "validation": validate_asamm,
        "figures": make_figures,
        "records": finalize_records,
    }
    if args.stage == "all":
        run_all()
    else:
        actions[args.stage]()


if __name__ == "__main__":
    main()
