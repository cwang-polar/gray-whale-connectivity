"""Run the isolated Phase 8R pelagic-support structural sensitivity.

This experiment does not reconstruct annual euphausiid abundance. It adds a
time-invariant, explicitly hypothetical support potential within the existing
DBO3 anchor polygon and tests whether the established corridor results are
sensitive to an omitted alternative-prey pathway. All established inputs and
outputs are read-only; new products are written below
``outputs/pelagic_support_phase8r``.
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

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_prey_only_lcp_main_south_to_north_v0 as lcp_base  # noqa: E402
from gray_whale_corridor.movement.lcp_ensemble import (  # noqa: E402
    build_weighted_adjacency,
    path_length_km,
    run_ensemble,
)


PHASE = "Phase 8R pelagic-support structural sensitivity"
SCRIPT = "scripts/run_phase8r_pelagic_support_structural_sensitivity.py"
ALL_YEARS = list(range(2011, 2020))
VESSEL_YEARS = list(range(2015, 2020))
PREY_VARIABLE = "total_kJm2_frozen"
EPSILON = 0.001
FREQUENT_THRESHOLD = 0.05

STRENGTHS = {
    "baseline": 0.00,
    "low": 0.10,
    "medium": 0.25,
    "high": 0.50,
}

SCENARIO_YEARS = {
    "prey": ALL_YEARS,
    "prey_ice": ALL_YEARS,
    "prey_vessel": VESSEL_YEARS,
    "prey_ice_vessel": VESSEL_YEARS,
}

SCENARIO_LABELS = {
    "prey": "Benthic prey",
    "prey_ice": "Benthic prey + ice",
    "prey_vessel": "Benthic prey + vessel",
    "prey_ice_vessel": "Benthic prey + ice + vessel",
}

OUT = ROOT / "outputs/pelagic_support_phase8r"
SURFACE_OUT = OUT / "surfaces"
LCP_OUT = OUT / "lcp"
SUMMARY_OUT = OUT / "summaries"
VALIDATION_OUT = OUT / "validation"
FIGURE_OUT = OUT / "figures"
STATUS_PATH = OUT / "phase8r_run_status.json"
MANIFEST_PATH = OUT / "phase8r_analysis_manifest.json"

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
DBO = ROOT / "data/processed/study_domain/dbo_anchor_polygons.gpkg"
PREY_DIR = ROOT / "outputs/prey_v2"
ICE_SURFACE_DIR = ROOT / "outputs/surfaces_v2"
SHIP_DIR = ROOT / "outputs/ship"
ASAMM_SIGHTINGS = ROOT / "data/processed/validation/asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
ASAMM_EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
PHASE8N_GATE = ROOT / "outputs/pelagic_prey_phase8n/tables/phase8n_pelagic_surface_integration_gate.csv"
PHASE8N_ASGARD = ROOT / "outputs/pelagic_prey_phase8n/tables/phase8n_asgard_euphausiid_abundance_by_dbo_anchor.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (OUT, SURFACE_OUT, LCP_OUT, SUMMARY_OUT, VALIDATION_OUT, FIGURE_OUT):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {
        "phase": PHASE,
        "status": "running",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "stages": {},
        "established_output_modified": False,
        "manuscript_or_si_modified": False,
    }


def set_stage(stage: str, status: str, **details: object) -> None:
    state = load_status()
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    state["updated_utc"] = utc_now()
    if status == "failed":
        state["status"] = "failed"
    write_json(STATUS_PATH, state)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prey_path(year: int) -> Path:
    return PREY_DIR / f"amphipod_prey_surface_total_kJm2_frozen_target{year}_full_trailing_C3_v2.csv"


def ice_path(year: int) -> Path:
    return ICE_SURFACE_DIR / f"prey_ice_brian_kJm2_frozen_C3_utility_target{year}_v2.csv"


def ship_path(year: int) -> Path:
    return SHIP_DIR / f"ship_intensity_MayJul_target{year}.csv"


def established_grid_path(scenario: str, year: int) -> Path:
    if scenario == "prey":
        return ROOT / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice":
        return ROOT / f"outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_vessel":
        return ROOT / f"outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    if scenario == "prey_ice_vessel":
        return ROOT / f"outputs/lcp_v2/prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    raise KeyError(scenario)


def established_route_path(scenario: str) -> Path:
    if scenario == "prey":
        return ROOT / "outputs/lcp_v2/prey_only_brian_C3_route_summary_v2.csv"
    if scenario == "prey_ice":
        return ROOT / "outputs/lcp_v2/prey_ice_brian_C3_route_summary_v2.csv"
    if scenario == "prey_vessel":
        return ROOT / "outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_C3_route_summary_v2.csv"
    if scenario == "prey_ice_vessel":
        return ROOT / "outputs/lcp_v2/prey_ice_ship_brian_C3_route_summary_v2.csv"
    raise KeyError(scenario)


def input_paths() -> list[Path]:
    paths = [GRID, ENDPOINTS, DBO, ASAMM_SIGHTINGS, ASAMM_EFFORT, PHASE8N_GATE, PHASE8N_ASGARD]
    paths.extend(prey_path(year) for year in ALL_YEARS)
    paths.extend(ice_path(year) for year in ALL_YEARS)
    paths.extend(ship_path(year) for year in VESSEL_YEARS)
    for scenario, years in SCENARIO_YEARS.items():
        paths.append(established_route_path(scenario))
        paths.extend(established_grid_path(scenario, year) for year in years)
    return list(dict.fromkeys(paths))


def integrity_snapshot(label: str) -> pd.DataFrame:
    rows = []
    for path in input_paths():
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("file_path")
    frame.to_csv(OUT / f"phase8r_input_integrity_{label}.csv", index=False)
    return frame


def load_grid() -> pd.DataFrame:
    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    grid = grid.loc[(grid["ocean_flag"] == 1) & grid["inside_C3_domain"].astype(bool)].copy()
    return grid.reset_index(drop=True)


def build_support_template(grid: pd.DataFrame) -> pd.DataFrame:
    anchors = gpd.read_file(DBO)
    if anchors.crs is None:
        raise RuntimeError("DBO anchor polygons have no coordinate reference system")
    dbo3_rows = anchors.loc[anchors["name"].eq("DBO3")]
    if len(dbo3_rows) != 1:
        raise RuntimeError(f"Expected one DBO3 polygon; found {len(dbo3_rows)}")
    points = gpd.GeoDataFrame(
        grid[["grid_id", "x", "y", "lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(grid["x"], grid["y"]),
        crs="EPSG:3338",
    )
    if points.crs != anchors.crs:
        points = points.to_crs(anchors.crs)
    polygon = dbo3_rows.geometry.iloc[0]
    inside = points.geometry.intersects(polygon)
    output = grid[["grid_id", "x", "y", "lon", "lat"]].copy()
    output["inside_DBO3_anchor"] = inside.to_numpy(dtype=bool)
    output["P_pelagic_DBO3"] = output["inside_DBO3_anchor"].astype(float)
    if int(output["inside_DBO3_anchor"].sum()) == 0:
        raise RuntimeError("The DBO3 structural-support template contains no grid cells")
    return output


def audit_inputs() -> None:
    set_stage("audit", "running")
    before = integrity_snapshot("before")
    grid = load_grid()
    endpoints = pd.read_csv(ENDPOINTS)
    support = build_support_template(grid)
    support.to_csv(SURFACE_OUT / "phase8r_DBO3_structural_support_template_study_region.csv", index=False)
    gate = pd.read_csv(PHASE8N_GATE)
    manifest = {
        "phase": PHASE,
        "script": SCRIPT,
        "analysis_role": "structural sensitivity; not an empirical annual euphausiid surface",
        "target_season": "May-July",
        "target_years": ALL_YEARS,
        "vessel_years": VESSEL_YEARS,
        "prey_variable": PREY_VARIABLE,
        "support_zone": "DBO3 anchor polygon",
        "support_zone_cell_count": int(support["inside_DBO3_anchor"].sum()),
        "support_zone_definition": "P_pelagic_DBO3=1 inside the existing DBO3 anchor polygon and 0 outside",
        "support_is_time_invariant": True,
        "support_strengths": STRENGTHS,
        "combination_formula": "U_benthic_pelagic = 1 - (1 - U_benthic) * (1 - lambda * P_pelagic_DBO3)",
        "strength_interpretation": "dimensionless structural bounds; not estimated krill abundance or energy density",
        "scenario_years": SCENARIO_YEARS,
        "grid_cell_count": int(len(grid)),
        "source_endpoint_count": int((endpoints["endpoint_type"] == "source").sum()),
        "destination_endpoint_count": int((endpoints["endpoint_type"] == "destination").sum()),
        "paths_per_ensemble": int(
            (endpoints["endpoint_type"] == "source").sum()
            * (endpoints["endpoint_type"] == "destination").sum()
        ),
        "planned_ensemble_count": int(
            len(STRENGTHS) * sum(len(years) for years in SCENARIO_YEARS.values())
        ),
        "phase8n_integration_gate_rows": int(len(gate)),
        "phase8n_decision": "observational evidence only; no calibrated pelagic surface",
        "existing_results_are_read_only": True,
        "manuscript_or_si_modified": False,
        "input_file_count": int(len(before)),
        "created_utc": utc_now(),
    }
    write_json(MANIFEST_PATH, manifest)
    set_stage(
        "audit",
        "completed",
        input_file_count=len(before),
        support_zone_cell_count=int(support["inside_DBO3_anchor"].sum()),
        planned_ensemble_count=manifest["planned_ensemble_count"],
    )


def resistance(utility: pd.Series | np.ndarray) -> np.ndarray:
    return np.maximum(-np.log(np.asarray(utility, dtype=float) + EPSILON), 0.0)


def surface_path(strength_label: str, year: int) -> Path:
    return SURFACE_OUT / f"phase8r_pelagic_support_{strength_label}_target{year}_study_region.csv"


def lcp_grid_path(strength_label: str, scenario: str, year: int) -> Path:
    return LCP_OUT / f"phase8r_{strength_label}_{scenario}_corridor_target{year}_study_region.csv"


def path_summary_path(strength_label: str, scenario: str) -> Path:
    return LCP_OUT / f"phase8r_{strength_label}_{scenario}_path_summary_study_region.csv"


def build_surface(year: int, strength_label: str, strength: float, grid: pd.DataFrame, support: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prey = pd.read_csv(prey_path(year), dtype={"grid_id": str})
    ice = pd.read_csv(ice_path(year), dtype={"grid_id": str})
    keep_prey = prey[["grid_id", "U_amphipod_prey_global_q05_q95", "reliability_flag"]].copy()
    keep_prey.rename(columns={"U_amphipod_prey_global_q05_q95": "U_benthic"}, inplace=True)
    keep_ice = ice[["grid_id", "A_ice_v1_MayJul"]].copy()
    output = (
        grid[["grid_id", "grid_x", "grid_y", "x", "y", "lon", "lat", "ocean_flag"]]
        .merge(keep_prey, on="grid_id", how="left", validate="one_to_one")
        .merge(keep_ice, on="grid_id", how="left", validate="one_to_one")
        .merge(support[["grid_id", "inside_DBO3_anchor", "P_pelagic_DBO3"]], on="grid_id", how="left", validate="one_to_one")
    )
    if output[["U_benthic", "A_ice_v1_MayJul", "P_pelagic_DBO3"]].isna().any().any():
        raise RuntimeError(f"{year} {strength_label}: incomplete prey, ice, or support merge")
    output["target_year"] = year
    output["support_strength_label"] = strength_label
    output["pelagic_support_lambda"] = strength
    output["U_benthic"] = pd.to_numeric(output["U_benthic"], errors="coerce").clip(EPSILON, 1.0)
    output["A_ice_v1_MayJul"] = pd.to_numeric(output["A_ice_v1_MayJul"], errors="coerce").clip(0.05, 1.0)
    output["U_benthic_pelagic"] = 1.0 - (1.0 - output["U_benthic"]) * (
        1.0 - strength * output["P_pelagic_DBO3"]
    )
    output["U_benthic_pelagic"] = output["U_benthic_pelagic"].clip(EPSILON, 1.0)
    output["pelagic_utility_increment"] = output["U_benthic_pelagic"] - output["U_benthic"]
    output["S_scaled_vessel_MayJul"] = 0.0
    output["D_vessel_alpha0_5_MayJul"] = 1.0
    if year in VESSEL_YEARS:
        vessel = pd.read_csv(ship_path(year), dtype={"grid_id": str})
        keep_vessel = vessel[["grid_id", "S_scaled_ship_primary_MayJul_mean", "D_ship_alpha0_5_MayJul_mean"]].copy()
        keep_vessel.rename(
            columns={
                "S_scaled_ship_primary_MayJul_mean": "S_scaled_vessel_MayJul_input",
                "D_ship_alpha0_5_MayJul_mean": "D_vessel_MayJul_input",
            },
            inplace=True,
        )
        output = output.merge(keep_vessel, on="grid_id", how="left", validate="one_to_one")
        if output[["S_scaled_vessel_MayJul_input", "D_vessel_MayJul_input"]].isna().any().any():
            raise RuntimeError(f"{year}: incomplete vessel merge")
        output["S_scaled_vessel_MayJul"] = pd.to_numeric(
            output.pop("S_scaled_vessel_MayJul_input"), errors="coerce"
        ).clip(0.0, 1.0)
        output["D_vessel_alpha0_5_MayJul"] = pd.to_numeric(
            output.pop("D_vessel_MayJul_input"), errors="coerce"
        ).clip(0.05, 1.0)

    output["U_prey"] = output["U_benthic_pelagic"]
    output["U_prey_ice"] = output["U_benthic_pelagic"] * output["A_ice_v1_MayJul"]
    output["U_prey_vessel"] = output["U_benthic_pelagic"] * output["D_vessel_alpha0_5_MayJul"]
    output["U_prey_ice_vessel"] = (
        output["U_benthic_pelagic"]
        * output["A_ice_v1_MayJul"]
        * output["D_vessel_alpha0_5_MayJul"]
    )
    for scenario in SCENARIO_YEARS:
        output[f"R_{scenario}"] = resistance(output[f"U_{scenario}"])

    increment = output.loc[output["inside_DBO3_anchor"], "pelagic_utility_increment"]
    qa = {
        "target_year": year,
        "support_strength_label": strength_label,
        "pelagic_support_lambda": strength,
        "grid_cell_count": int(len(output)),
        "DBO3_cell_count": int(output["inside_DBO3_anchor"].sum()),
        "utility_below_benthic_count": int((output["U_benthic_pelagic"] < output["U_benthic"] - 1e-12).sum()),
        "utility_above_one_count": int((output["U_benthic_pelagic"] > 1.0 + 1e-12).sum()),
        "outside_DBO3_increment_count": int(
            ((~output["inside_DBO3_anchor"]) & (output["pelagic_utility_increment"].abs() > 1e-12)).sum()
        ),
        "mean_DBO3_utility_increment": float(increment.mean()),
        "max_DBO3_utility_increment": float(increment.max()),
        "mean_U_benthic": float(output["U_benthic"].mean()),
        "mean_U_benthic_pelagic": float(output["U_benthic_pelagic"].mean()),
    }
    if qa["utility_below_benthic_count"] or qa["utility_above_one_count"] or qa["outside_DBO3_increment_count"]:
        raise RuntimeError(f"{year} {strength_label}: structural-support surface QA failed")
    return output, qa


def build_surfaces() -> None:
    set_stage("surfaces", "running")
    grid = load_grid()
    support = build_support_template(grid)
    rows = []
    for year in ALL_YEARS:
        for strength_label, strength in STRENGTHS.items():
            surface, qa = build_surface(year, strength_label, strength, grid, support)
            surface.to_csv(surface_path(strength_label, year), index=False)
            rows.append(qa)
    summary = pd.DataFrame(rows)
    summary.to_csv(SURFACE_OUT / "phase8r_pelagic_support_surface_qa_summary.csv", index=False)
    write_json(
        SURFACE_OUT / "phase8r_pelagic_support_surface_qa.json",
        {
            "phase": PHASE,
            "formula": "U_benthic_pelagic = 1 - (1 - U_benthic) * (1 - lambda * P_pelagic_DBO3)",
            "annual_strength_rows": rows,
            "warnings": [],
        },
    )
    set_stage("surfaces", "completed", surface_file_count=len(rows), warnings=[])


def load_topology():
    ocean = pd.read_csv(GRID, dtype={"grid_id": str})
    endpoints = pd.read_csv(ENDPOINTS, dtype={"grid_id": str})
    return lcp_base.build_graph_topology(ocean_mask=ocean, endpoints=endpoints)


def run_scenario_year(
    topology,
    surface: pd.DataFrame,
    scenario: str,
    year: int,
    strength_label: str,
    strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    utility_column = f"U_{scenario}"
    merged = topology.grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"), validate="one_to_one")
    merged = merged.loc[merged["ocean_flag"] == 1].copy().reset_index(drop=True)
    if merged[utility_column].isna().any():
        raise RuntimeError(f"{year} {strength_label} {scenario}: utility missing on topology")
    merged["modeled_utility"] = pd.to_numeric(merged[utility_column], errors="coerce").clip(EPSILON, 1.0)
    merged["modeled_resistance"] = resistance(merged["modeled_utility"])
    adjacency = build_weighted_adjacency(topology.edge_pairs, merged["modeled_resistance"].to_numpy())
    paths, counts = run_ensemble(adjacency, topology)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    if len(paths) != attempted:
        raise RuntimeError(f"{year} {strength_label} {scenario}: {len(paths)}/{attempted} paths succeeded")
    merged["lcp_count"] = counts
    merged["lcp_frequency_normalized"] = counts / len(paths)

    path_rows = []
    for item in paths:
        sub = merged.iloc[list(item.node_indices)]
        path_rows.append(
            {
                "target_year": year,
                "support_strength_label": strength_label,
                "pelagic_support_lambda": strength,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "source_grid_id": topology.grid_id_by_node_index[item.source_index],
                "destination_grid_id": topology.grid_id_by_node_index[item.destination_index],
                "path_cost": item.cost,
                "path_length_km": path_length_km(item.node_indices, merged),
                "path_cell_count": len(item.node_indices),
                "path_intersects_DBO3": bool(sub["inside_DBO3_anchor"].any()),
                "path_DBO3_cell_count": int(sub["inside_DBO3_anchor"].sum()),
                "path_mean_U_benthic": float(sub["U_benthic"].mean()),
                "path_mean_U_benthic_pelagic": float(sub["U_benthic_pelagic"].mean()),
                "path_mean_pelagic_increment": float(sub["pelagic_utility_increment"].mean()),
            }
        )
    path_frame = pd.DataFrame(path_rows)
    frequency = merged["lcp_frequency_normalized"]
    frequent = frequency > FREQUENT_THRESHOLD
    frequent_cells = merged.loc[frequent]
    dbo3 = merged["inside_DBO3_anchor"].astype(bool)
    total_lcp_count = float(merged["lcp_count"].sum())
    route = {
        "target_year": year,
        "support_strength_label": strength_label,
        "pelagic_support_lambda": strength,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "attempted_path_count": attempted,
        "successful_path_count": len(path_frame),
        "failed_path_count": attempted - len(path_frame),
        "mean_path_cost": float(path_frame["path_cost"].mean()),
        "median_path_cost": float(path_frame["path_cost"].median()),
        "mean_path_length_km": float(path_frame["path_length_km"].mean()),
        "corridor_area_cells_freq_gt_0": int((frequency > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int(frequent.sum()),
        "corridor_area_cells_freq_gt_0_10": int((frequency > 0.10).sum()),
        "frequent_centroid_x": float(frequent_cells["x"].mean()) if len(frequent_cells) else math.nan,
        "frequent_centroid_y": float(frequent_cells["y"].mean()) if len(frequent_cells) else math.nan,
        "frequent_centroid_lon": float(frequent_cells["lon"].mean()) if len(frequent_cells) else math.nan,
        "frequent_centroid_lat": float(frequent_cells["lat"].mean()) if len(frequent_cells) else math.nan,
        "DBO3_cell_count": int(dbo3.sum()),
        "paths_intersecting_DBO3": int(path_frame["path_intersects_DBO3"].sum()),
        "percent_paths_intersecting_DBO3": float(path_frame["path_intersects_DBO3"].mean() * 100.0),
        "mean_path_DBO3_cell_count": float(path_frame["path_DBO3_cell_count"].mean()),
        "DBO3_cells_used": int((dbo3 & (frequency > 0)).sum()),
        "DBO3_frequent_cells": int((dbo3 & frequent).sum()),
        "percent_DBO3_cells_frequent": float((dbo3 & frequent).sum() / dbo3.sum() * 100.0),
        "mean_frequency_inside_DBO3": float(frequency.loc[dbo3].mean()),
        "share_total_lcp_count_inside_DBO3_percent": float(
            merged.loc[dbo3, "lcp_count"].sum() / total_lcp_count * 100.0
        ) if total_lcp_count else math.nan,
        "warning_flags": "",
    }
    keep = [
        "grid_id",
        "x",
        "y",
        "lon",
        "lat",
        "target_year",
        "inside_DBO3_anchor",
        "P_pelagic_DBO3",
        "U_benthic",
        "U_benthic_pelagic",
        "pelagic_utility_increment",
        "A_ice_v1_MayJul",
        "S_scaled_vessel_MayJul",
        "D_vessel_alpha0_5_MayJul",
        utility_column,
        "modeled_utility",
        "modeled_resistance",
        "lcp_count",
        "lcp_frequency_normalized",
        "ocean_flag",
    ]
    out_grid = merged[keep].copy()
    out_grid.rename(columns={utility_column: "scenario_utility_before_topology_merge"}, inplace=True)
    out_grid["support_strength_label"] = strength_label
    out_grid["pelagic_support_lambda"] = strength
    out_grid["scenario"] = scenario
    out_grid["scenario_label"] = SCENARIO_LABELS[scenario]
    return out_grid, path_frame, route


def compare_grid_to_established(computed: pd.DataFrame, scenario: str, year: int) -> tuple[float, int]:
    established = pd.read_csv(established_grid_path(scenario, year), dtype={"grid_id": str})
    check = computed[["grid_id", "lcp_frequency_normalized"]].merge(
        established[["grid_id", "lcp_frequency_normalized"]],
        on="grid_id",
        suffixes=("_computed", "_established"),
        validate="one_to_one",
    )
    error = float(
        np.max(
            np.abs(
                check["lcp_frequency_normalized_computed"]
                - check["lcp_frequency_normalized_established"]
            )
        )
    )
    return error, int(len(check))


def established_mean_cost(scenario: str, year: int) -> float:
    frame = pd.read_csv(established_route_path(scenario))
    if "prey_variable" in frame.columns:
        frame = frame.loc[frame["prey_variable"].eq(PREY_VARIABLE)]
    row = frame.loc[frame["target_year"].eq(year)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one established route row for {scenario} {year}; found {len(row)}")
    return float(row.iloc[0]["mean_path_cost"])


def smoke_test() -> None:
    set_stage("smoke", "running")
    topology = load_topology()
    grid = load_grid()
    support = build_support_template(grid)
    year = 2019
    rows = []
    for strength_label in ("baseline", "high"):
        strength = STRENGTHS[strength_label]
        surface, _ = build_surface(year, strength_label, strength, grid, support)
        for scenario in SCENARIO_YEARS:
            computed, _, route = run_scenario_year(
                topology, surface, scenario, year, strength_label, strength
            )
            row = {
                "target_year": year,
                "support_strength_label": strength_label,
                "scenario": scenario,
                "successful_path_count": route["successful_path_count"],
                "mean_path_cost": route["mean_path_cost"],
                "percent_paths_intersecting_DBO3": route["percent_paths_intersecting_DBO3"],
            }
            if strength_label == "baseline":
                frequency_error, matched_cells = compare_grid_to_established(computed, scenario, year)
                cost_error = abs(route["mean_path_cost"] - established_mean_cost(scenario, year))
                row.update(
                    {
                        "baseline_frequency_max_absolute_error": frequency_error,
                        "baseline_mean_cost_absolute_error": cost_error,
                        "baseline_matched_grid_cells": matched_cells,
                    }
                )
                if frequency_error > 1e-12 or cost_error > 1e-8:
                    raise RuntimeError(
                        f"Baseline equivalence failed for {scenario}: frequency={frequency_error}, cost={cost_error}"
                    )
            else:
                baseline_cost = next(
                    r["mean_path_cost"]
                    for r in rows
                    if r["support_strength_label"] == "baseline" and r["scenario"] == scenario
                )
                row["high_strength_cost_minus_baseline"] = route["mean_path_cost"] - baseline_cost
                if route["mean_path_cost"] > baseline_cost + 1e-8:
                    raise RuntimeError(f"High pelagic support increased optimal cost for {scenario}")
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "phase8r_smoke_test_target2019.csv", index=False)
    write_json(
        OUT / "phase8r_smoke_test_target2019.json",
        {
            "phase": PHASE,
            "target_year": year,
            "rows": rows,
            "warnings": [],
        },
    )
    set_stage("smoke", "completed", tested_rows=len(rows), baseline_equivalence=True, warnings=[])


def batch_lcp() -> None:
    set_stage("batch", "running")
    topology = load_topology()
    routes = []
    completed = 0
    for strength_label, strength in STRENGTHS.items():
        path_groups: dict[str, list[pd.DataFrame]] = {scenario: [] for scenario in SCENARIO_YEARS}
        for year in ALL_YEARS:
            surface = pd.read_csv(surface_path(strength_label, year), dtype={"grid_id": str})
            for scenario, years in SCENARIO_YEARS.items():
                if year not in years:
                    continue
                grid_out, paths_out, route = run_scenario_year(
                    topology, surface, scenario, year, strength_label, strength
                )
                grid_out.to_csv(lcp_grid_path(strength_label, scenario, year), index=False)
                path_groups[scenario].append(paths_out)
                routes.append(route)
                completed += 1
                set_stage(
                    "batch",
                    "running",
                    completed_ensembles=completed,
                    last_completed={"strength": strength_label, "scenario": scenario, "year": year},
                )
        for scenario, frames in path_groups.items():
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    path_summary_path(strength_label, scenario), index=False
                )
    route_frame = pd.DataFrame(routes)
    route_frame.to_csv(SUMMARY_OUT / "phase8r_route_summary.csv", index=False)
    failed = int(route_frame["failed_path_count"].sum())
    qa = {
        "phase": PHASE,
        "ensemble_count": int(len(route_frame)),
        "expected_ensemble_count": int(len(STRENGTHS) * sum(len(v) for v in SCENARIO_YEARS.values())),
        "successful_path_count_total": int(route_frame["successful_path_count"].sum()),
        "failed_path_count_total": failed,
        "successful_paths_by_ensemble": sorted(route_frame["successful_path_count"].unique().tolist()),
        "topology_warnings": list(topology.warnings),
        "warnings": [],
    }
    if qa["ensemble_count"] != qa["expected_ensemble_count"] or failed:
        raise RuntimeError(f"Phase 8R batch QA failed: {qa}")
    write_json(SUMMARY_OUT / "phase8r_lcp_qa.json", qa)
    set_stage(
        "batch",
        "completed",
        ensemble_count=qa["ensemble_count"],
        successful_path_count_total=qa["successful_path_count_total"],
        failed_path_count_total=failed,
    )


def jaccard(a: pd.Series, b: pd.Series) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else math.nan


def compare_to_baseline() -> None:
    set_stage("comparisons", "running")
    routes = pd.read_csv(SUMMARY_OUT / "phase8r_route_summary.csv")
    equivalence_rows = []
    for scenario, years in SCENARIO_YEARS.items():
        for year in years:
            computed_grid = pd.read_csv(
                lcp_grid_path("baseline", scenario, year), dtype={"grid_id": str}
            )
            frequency_error, matched_cells = compare_grid_to_established(
                computed_grid, scenario, year
            )
            computed_route = routes.loc[
                routes["scenario"].eq(scenario)
                & routes["target_year"].eq(year)
                & routes["support_strength_label"].eq("baseline")
            ].iloc[0]
            cost_error = abs(
                float(computed_route["mean_path_cost"])
                - established_mean_cost(scenario, year)
            )
            equivalence_rows.append(
                {
                    "target_year": year,
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "matched_grid_cells": matched_cells,
                    "max_frequency_absolute_error": frequency_error,
                    "mean_path_cost_absolute_error": cost_error,
                    "frequency_equivalent": frequency_error <= 1e-12,
                    "cost_equivalent": cost_error <= 1e-8,
                }
            )
    equivalence = pd.DataFrame(equivalence_rows)
    equivalence.to_csv(
        SUMMARY_OUT / "phase8r_baseline_equivalence_all_scenario_years.csv",
        index=False,
    )
    if not bool((equivalence["frequency_equivalent"] & equivalence["cost_equivalent"]).all()):
        raise RuntimeError("One or more Phase 8R baseline ensembles did not reproduce the established result")

    rows = []
    for scenario, years in SCENARIO_YEARS.items():
        for year in years:
            baseline_route = routes.loc[
                routes["scenario"].eq(scenario)
                & routes["target_year"].eq(year)
                & routes["support_strength_label"].eq("baseline")
            ].iloc[0]
            baseline_grid = pd.read_csv(lcp_grid_path("baseline", scenario, year), dtype={"grid_id": str})
            for strength_label, strength in STRENGTHS.items():
                if strength_label == "baseline":
                    continue
                variant_route = routes.loc[
                    routes["scenario"].eq(scenario)
                    & routes["target_year"].eq(year)
                    & routes["support_strength_label"].eq(strength_label)
                ].iloc[0]
                variant_grid = pd.read_csv(lcp_grid_path(strength_label, scenario, year), dtype={"grid_id": str})
                merged = baseline_grid[["grid_id", "lcp_frequency_normalized"]].merge(
                    variant_grid[["grid_id", "lcp_frequency_normalized"]],
                    on="grid_id",
                    suffixes=("_baseline", "_variant"),
                    validate="one_to_one",
                )
                base_frequent = merged["lcp_frequency_normalized_baseline"] > FREQUENT_THRESHOLD
                var_frequent = merged["lcp_frequency_normalized_variant"] > FREQUENT_THRESHOLD
                centroid_shift = float(
                    np.hypot(
                        variant_route["frequent_centroid_x"] - baseline_route["frequent_centroid_x"],
                        variant_route["frequent_centroid_y"] - baseline_route["frequent_centroid_y"],
                    ) / 1000.0
                )
                rows.append(
                    {
                        "target_year": year,
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "support_strength_label": strength_label,
                        "pelagic_support_lambda": strength,
                        "baseline_mean_path_cost": baseline_route["mean_path_cost"],
                        "variant_mean_path_cost": variant_route["mean_path_cost"],
                        "mean_path_cost_change": variant_route["mean_path_cost"] - baseline_route["mean_path_cost"],
                        "mean_path_cost_change_percent": (
                            (variant_route["mean_path_cost"] - baseline_route["mean_path_cost"])
                            / baseline_route["mean_path_cost"]
                            * 100.0
                        ),
                        "baseline_frequent_corridor_cells": baseline_route["corridor_area_cells_freq_gt_0_05"],
                        "variant_frequent_corridor_cells": variant_route["corridor_area_cells_freq_gt_0_05"],
                        "frequent_corridor_area_change_cells": (
                            variant_route["corridor_area_cells_freq_gt_0_05"]
                            - baseline_route["corridor_area_cells_freq_gt_0_05"]
                        ),
                        "frequency_map_pearson_r": float(
                            merged["lcp_frequency_normalized_baseline"].corr(
                                merged["lcp_frequency_normalized_variant"]
                            )
                        ),
                        "jaccard_frequent_corridor": jaccard(base_frequent, var_frequent),
                        "frequent_centroid_shift_km": centroid_shift,
                        "baseline_percent_paths_intersecting_DBO3": baseline_route["percent_paths_intersecting_DBO3"],
                        "variant_percent_paths_intersecting_DBO3": variant_route["percent_paths_intersecting_DBO3"],
                        "change_percent_paths_intersecting_DBO3_points": (
                            variant_route["percent_paths_intersecting_DBO3"]
                            - baseline_route["percent_paths_intersecting_DBO3"]
                        ),
                        "baseline_DBO3_frequent_cells": baseline_route["DBO3_frequent_cells"],
                        "variant_DBO3_frequent_cells": variant_route["DBO3_frequent_cells"],
                        "change_DBO3_frequent_cells": (
                            variant_route["DBO3_frequent_cells"] - baseline_route["DBO3_frequent_cells"]
                        ),
                        "baseline_share_lcp_count_inside_DBO3_percent": baseline_route[
                            "share_total_lcp_count_inside_DBO3_percent"
                        ],
                        "variant_share_lcp_count_inside_DBO3_percent": variant_route[
                            "share_total_lcp_count_inside_DBO3_percent"
                        ],
                        "change_share_lcp_count_inside_DBO3_points": (
                            variant_route["share_total_lcp_count_inside_DBO3_percent"]
                            - baseline_route["share_total_lcp_count_inside_DBO3_percent"]
                        ),
                    }
                )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(SUMMARY_OUT / "phase8r_pelagic_support_vs_baseline_annual.csv", index=False)
    aggregate = (
        comparison.groupby(
            ["scenario", "scenario_label", "support_strength_label", "pelagic_support_lambda"],
            as_index=False,
        )
        .agg(
            years=("target_year", "nunique"),
            mean_cost_change_percent=("mean_path_cost_change_percent", "mean"),
            min_cost_change_percent=("mean_path_cost_change_percent", "min"),
            max_cost_change_percent=("mean_path_cost_change_percent", "max"),
            mean_area_change_cells=("frequent_corridor_area_change_cells", "mean"),
            mean_jaccard=("jaccard_frequent_corridor", "mean"),
            min_jaccard=("jaccard_frequent_corridor", "min"),
            mean_centroid_shift_km=("frequent_centroid_shift_km", "mean"),
            max_centroid_shift_km=("frequent_centroid_shift_km", "max"),
            mean_DBO3_path_change_points=("change_percent_paths_intersecting_DBO3_points", "mean"),
            mean_DBO3_lcp_share_change_points=("change_share_lcp_count_inside_DBO3_points", "mean"),
        )
    )
    aggregate.to_csv(SUMMARY_OUT / "phase8r_pelagic_support_response_summary.csv", index=False)
    set_stage(
        "comparisons",
        "completed",
        baseline_equivalence_rows=len(equivalence),
        annual_rows=len(comparison),
        aggregate_rows=len(aggregate),
    )


def validation_period_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    common = metrics.loc[metrics["target_year"].between(2015, 2019)].copy()
    common["validation_period"] = "common_2015_2019"
    pieces.append(common)
    full = metrics.loc[
        metrics["scenario"].isin(["prey", "prey_ice"])
        & metrics["target_year"].between(2011, 2019)
    ].copy()
    full["validation_period"] = "full_2011_2019_nonvessel"
    pieces.append(full)
    return pd.concat(pieces, ignore_index=True)


def validate_asamm() -> None:
    set_stage("validation", "running")
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    sightings = sightings.loc[
        sightings["use_for_C3_validation"].astype(bool)
        & sightings["year"].isin(ALL_YEARS)
        & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)
    effort = pd.read_csv(ASAMM_EFFORT, dtype={"grid_id": str})
    effort = effort.loc[
        effort["year"].isin(ALL_YEARS)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    support = pd.read_csv(
        SURFACE_OUT / "phase8r_DBO3_structural_support_template_study_region.csv",
        dtype={"grid_id": str},
    ).set_index("grid_id")
    rows = []
    for strength_label, strength in STRENGTHS.items():
        for scenario, years in SCENARIO_YEARS.items():
            for year in years:
                annual_sightings = sightings.loc[sightings["year"].eq(year)]
                effort_ids = set(effort.loc[effort["year"].eq(year), "grid_id"])
                grid = pd.read_csv(lcp_grid_path(strength_label, scenario, year), dtype={"grid_id": str})
                lookup = grid.set_index("grid_id")
                frequency = grid["lcp_frequency_normalized"].fillna(0.0)
                q95 = float(frequency.quantile(0.95))
                frequent_xy = grid.loc[
                    frequency > FREQUENT_THRESHOLD, ["x", "y"]
                ].to_numpy(dtype=float)
                tree = cKDTree(frequent_xy) if len(frequent_xy) else None
                effort_frequency = lookup.loc[
                    lookup.index.intersection(effort_ids), "lcp_frequency_normalized"
                ].fillna(0.0)
                for _, record in annual_sightings.iterrows():
                    grid_id = record["nearest_grid_id"]
                    if grid_id not in lookup.index:
                        continue
                    cell = lookup.loc[grid_id]
                    freq = float(cell["lcp_frequency_normalized"])
                    percentile = math.nan
                    if len(effort_frequency):
                        percentile = float(
                            (
                                (effort_frequency < freq).sum()
                                + 0.5 * (effort_frequency == freq).sum()
                            )
                            / len(effort_frequency)
                            * 100.0
                        )
                    distance = math.nan
                    if tree is not None:
                        distance = float(
                            tree.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0]
                            / 1000.0
                        )
                    rows.append(
                        {
                            "record_id": record["record_id"],
                            "target_year": year,
                            "support_strength_label": strength_label,
                            "pelagic_support_lambda": strength,
                            "scenario": scenario,
                            "scenario_label": SCENARIO_LABELS[scenario],
                            "grid_id": grid_id,
                            "sighting_inside_DBO3": bool(support.loc[grid_id, "inside_DBO3_anchor"]),
                            "lcp_frequency_at_sighting": freq,
                            "effort_footprint_percentile": percentile,
                            "inside_annual_top5pct": bool(freq >= q95),
                            "distance_to_frequent_corridor_km": distance,
                            "strict_effort_background_cell_count": len(effort_frequency),
                        }
                    )
    record_metrics = pd.DataFrame(rows)
    record_metrics.to_csv(VALIDATION_OUT / "phase8r_asamm_record_metrics.csv", index=False)
    period_metrics = validation_period_rows(record_metrics)
    subsets = []
    for subset_label, subset_frame in (
        ("all_sightings", period_metrics),
        ("inside_DBO3", period_metrics.loc[period_metrics["sighting_inside_DBO3"]]),
        ("outside_DBO3", period_metrics.loc[~period_metrics["sighting_inside_DBO3"]]),
    ):
        summary = (
            subset_frame.groupby(
                [
                    "validation_period",
                    "scenario",
                    "scenario_label",
                    "support_strength_label",
                    "pelagic_support_lambda",
                ],
                as_index=False,
            )
            .agg(
                record_count=("record_id", "nunique"),
                mean_effort_footprint_percentile=("effort_footprint_percentile", "mean"),
                percent_records_top5pct=(
                    "inside_annual_top5pct",
                    lambda x: float(np.mean(x) * 100.0),
                ),
                median_distance_to_frequent_corridor_km=(
                    "distance_to_frequent_corridor_km",
                    "median",
                ),
                mean_lcp_frequency_at_sighting=("lcp_frequency_at_sighting", "mean"),
            )
        )
        summary["sighting_subset"] = subset_label
        subsets.append(summary)
    summary = pd.concat(subsets, ignore_index=True)
    summary.to_csv(VALIDATION_OUT / "phase8r_asamm_validation_summary.csv", index=False)

    common_all = summary.loc[
        summary["validation_period"].eq("common_2015_2019")
        & summary["sighting_subset"].eq("all_sightings")
    ].copy()
    ranks = []
    for strength_label in STRENGTHS:
        sub = common_all.loc[common_all["support_strength_label"].eq(strength_label)].copy()
        for metric, ascending in (
            ("mean_effort_footprint_percentile", False),
            ("percent_records_top5pct", False),
            ("median_distance_to_frequent_corridor_km", True),
        ):
            ranked = sub.sort_values(metric, ascending=ascending).reset_index(drop=True)
            for rank, row in ranked.iterrows():
                ranks.append(
                    {
                        "support_strength_label": strength_label,
                        "pelagic_support_lambda": row["pelagic_support_lambda"],
                        "metric": metric,
                        "rank": rank + 1,
                        "scenario": row["scenario"],
                        "scenario_label": row["scenario_label"],
                        "metric_value": row[metric],
                    }
                )
    rank_frame = pd.DataFrame(ranks)
    rank_frame.to_csv(VALIDATION_OUT / "phase8r_asamm_common_period_scenario_ranks.csv", index=False)
    counts = common_all.groupby("support_strength_label")["record_count"].nunique()
    if not bool((counts == 1).all()):
        raise RuntimeError("ASAMM common-period scenario record counts differ")
    qa = {
        "phase": PHASE,
        "validation_months": [6, 7],
        "all_sighting_record_count_2011_2019": int(sightings["record_id"].nunique()),
        "common_period_record_count": int(
            common_all.loc[common_all["support_strength_label"].eq("baseline"), "record_count"].iloc[0]
        ),
        "identical_common_period_record_count_across_scenarios_and_strengths": True,
        "metric_scope": "effort-conditioned observed-use consistency; not abundance or behavioral response",
        "warnings": [],
    }
    write_json(VALIDATION_OUT / "phase8r_asamm_validation_qa.json", qa)
    set_stage(
        "validation",
        "completed",
        record_metric_rows=len(record_metrics),
        summary_rows=len(summary),
        common_period_record_count=qa["common_period_record_count"],
    )


def write_results_summary() -> None:
    route = pd.read_csv(SUMMARY_OUT / "phase8r_route_summary.csv")
    response = pd.read_csv(SUMMARY_OUT / "phase8r_pelagic_support_response_summary.csv")
    validation = pd.read_csv(VALIDATION_OUT / "phase8r_asamm_validation_summary.csv")
    ranks = pd.read_csv(VALIDATION_OUT / "phase8r_asamm_common_period_scenario_ranks.csv")

    high = response.loc[response["support_strength_label"].eq("high")]
    common = validation.loc[
        validation["validation_period"].eq("common_2015_2019")
        & validation["sighting_subset"].eq("all_sightings")
    ]
    best = ranks.loc[ranks["rank"].eq(1)].copy()
    lines = [
        "# Phase 8R Pelagic-support Structural Sensitivity Results",
        "",
        "## Scope",
        "",
        "This is a structural sensitivity analysis, not an annual euphausiid reconstruction. A time-invariant support potential was assigned to the existing DBO3 anchor polygon and combined with the primary benthic amphipod utility at dimensionless low (0.10), medium (0.25), and high (0.50) strengths.",
        "",
        "## Execution",
        "",
        f"- Corridor ensembles completed: {len(route)}.",
        f"- Successful paths: {int(route['successful_path_count'].sum()):,}.",
        f"- Failed paths: {int(route['failed_path_count'].sum()):,}.",
        "- The baseline reproduction matched all established annual corridor grids and route costs within numerical tolerance.",
        "",
        "## High-strength response relative to the benthic baseline",
        "",
    ]
    for _, row in high.iterrows():
        lines.append(
            f"- {row['scenario_label']}: mean path cost changed by {row['mean_cost_change_percent']:.2f}% across {int(row['years'])} years; mean frequent-set Jaccard was {row['mean_jaccard']:.3f}; mean centroid shift was {row['mean_centroid_shift_km']:.1f} km; and the share of path use inside DBO3 changed by {row['mean_DBO3_lcp_share_change_points']:.1f} percentage points."
        )
    lines.extend(
        [
            "",
            "## ASAMM common-period comparison",
            "",
            "All four scenario families used the same June-July ASAMM record set during 2015-2019. The metrics remain observed-use consistency measures, not abundance or behavioral-displacement estimates.",
            "",
        ]
    )
    for strength_label in STRENGTHS:
        sub = common.loc[common["support_strength_label"].eq(strength_label)]
        lines.append(f"### {strength_label.title()} support")
        lines.append("")
        for _, row in sub.sort_values("scenario").iterrows():
            lines.append(
                f"- {row['scenario_label']}: percentile {row['mean_effort_footprint_percentile']:.1f}, top-5% overlap {row['percent_records_top5pct']:.1f}%, median distance {row['median_distance_to_frequent_corridor_km']:.1f} km."
            )
        winners = best.loc[best["support_strength_label"].eq(strength_label)]
        winner_text = "; ".join(
            f"{row['metric']} -> {row['scenario_label']}" for _, row in winners.iterrows()
        )
        lines.append(f"- Metric winners: {winner_text}.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The strength parameter is not calibrated to krill abundance, biomass, energy density, or transport. The experiment tests robustness to a localized omitted alternative-prey pathway. It cannot estimate annual pelagic-prey availability or identify pelagic feeding from whale behavior records.",
            "",
        ]
    )
    (OUT / "phase8r_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_records() -> None:
    set_stage("records", "running")
    write_results_summary()
    before = pd.read_csv(OUT / "phase8r_input_integrity_before.csv")
    after = integrity_snapshot("after")
    check = before.merge(after, on="file_path", suffixes=("_before", "_after"), validate="one_to_one")
    check["unchanged"] = (
        (check["size_bytes_before"] == check["size_bytes_after"])
        & (check["sha256_before"] == check["sha256_after"])
    )
    check.to_csv(OUT / "phase8r_input_integrity_comparison.csv", index=False)
    if not bool(check["unchanged"].all()):
        raise RuntimeError("One or more frozen inputs changed during Phase 8R")
    routes = pd.read_csv(SUMMARY_OUT / "phase8r_route_summary.csv")
    equivalence = pd.read_csv(
        SUMMARY_OUT / "phase8r_baseline_equivalence_all_scenario_years.csv"
    )
    annual_comparison = pd.read_csv(
        SUMMARY_OUT / "phase8r_pelagic_support_vs_baseline_annual.csv"
    )
    validation_records = pd.read_csv(
        VALIDATION_OUT / "phase8r_asamm_record_metrics.csv"
    )
    figure_stems = [
        "phase8r_pelagic_support_structural_response_v1",
        "phase8r_pelagic_support_asamm_robustness_v1",
        "phase8r_pelagic_support_corridor_maps_2017_2018_v1",
    ]
    figure_formats = ["pdf", "svg", "png", "tiff"]
    figure_exports = [
        FIGURE_OUT / f"{stem}.{suffix}"
        for stem in figure_stems
        for suffix in figure_formats
    ]
    final_qa = {
        "phase": PHASE,
        "surface_file_count": len(list(SURFACE_OUT.glob("phase8r_pelagic_support_*_target*_study_region.csv"))),
        "lcp_grid_file_count": len(list(LCP_OUT.glob("phase8r_*_corridor_target*_study_region.csv"))),
        "route_summary_rows": int(len(routes)),
        "successful_path_count_total": int(routes["successful_path_count"].sum()),
        "failed_path_count_total": int(routes["failed_path_count"].sum()),
        "baseline_equivalence_rows": int(len(equivalence)),
        "baseline_equivalence_all_pass": bool(
            (equivalence["frequency_equivalent"] & equivalence["cost_equivalent"]).all()
        ),
        "maximum_baseline_frequency_error": float(equivalence["max_frequency_absolute_error"].max()),
        "maximum_baseline_cost_error": float(equivalence["mean_path_cost_absolute_error"].max()),
        "experimental_comparison_rows": int(len(annual_comparison)),
        "asamm_record_metric_rows": int(len(validation_records)),
        "figure_export_count": int(sum(path.exists() for path in figure_exports)),
        "expected_figure_export_count": int(len(figure_exports)),
        "frozen_input_count": int(len(check)),
        "frozen_inputs_unchanged": bool(check["unchanged"].all()),
        "manuscript_or_si_modified": False,
        "warnings": [],
    }
    expected = {
        "surface_file_count": 36,
        "lcp_grid_file_count": 112,
        "route_summary_rows": 112,
        "successful_path_count_total": 179200,
        "failed_path_count_total": 0,
        "baseline_equivalence_rows": 28,
        "experimental_comparison_rows": 84,
        "asamm_record_metric_rows": 14760,
        "figure_export_count": 12,
    }
    failures = {
        key: {"observed": final_qa[key], "expected": value}
        for key, value in expected.items()
        if final_qa[key] != value
    }
    if failures or not final_qa["baseline_equivalence_all_pass"] or not final_qa["frozen_inputs_unchanged"]:
        raise RuntimeError(f"Phase 8R final QA failed: {failures}")
    write_json(OUT / "phase8r_final_qa_summary.json", final_qa)
    state = load_status()
    state["status"] = "completed"
    state["completed_utc"] = utc_now()
    state["established_output_modified"] = False
    state["manuscript_or_si_modified"] = False
    write_json(STATUS_PATH, state)
    inventory_paths = [
        path
        for path in sorted(OUT.rglob("*"))
        if path.is_file()
        and not path.name.endswith(".tmp")
        and path.name != "phase8r_output_inventory.csv"
    ]
    inventory = pd.DataFrame(
        [
            {
                "file_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inventory_paths
        ]
    )
    inventory.to_csv(OUT / "phase8r_output_inventory.csv", index=False)
    set_stage(
        "records",
        "completed",
        frozen_inputs_unchanged=True,
        output_file_count=len(inventory),
    )


def run_all_without_figures() -> None:
    ensure_dirs()
    audit_inputs()
    build_surfaces()
    smoke_test()
    batch_lcp()
    compare_to_baseline()
    validate_asamm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "audit",
            "surfaces",
            "smoke",
            "batch",
            "comparisons",
            "validation",
            "records",
        ],
        default="all",
    )
    args = parser.parse_args()
    ensure_dirs()
    actions = {
        "audit": audit_inputs,
        "surfaces": build_surfaces,
        "smoke": smoke_test,
        "batch": batch_lcp,
        "comparisons": compare_to_baseline,
        "validation": validate_asamm,
        "records": finalize_records,
    }
    try:
        if args.stage == "all":
            run_all_without_figures()
        else:
            actions[args.stage]()
    except Exception as exc:
        set_stage(args.stage, "failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
