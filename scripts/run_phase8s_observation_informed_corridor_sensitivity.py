"""Run isolated corridor sensitivity for validated Phase 8S support surfaces.

The empirical component is the spatial pattern of relative euphausiid support
in June 2017 and 2018. Its magnitude is not calibrated to benthic energy, so
the cross-prey combination strength remains an explicit beta sensitivity. No
ASAMM outcome is used to select beta, operator, or observation model.
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
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase8r_pelagic_support_structural_sensitivity as phase8r  # noqa: E402
from gray_whale_corridor.movement.lcp_ensemble import (  # noqa: E402
    build_weighted_adjacency,
    path_length_km,
    run_ensemble,
)


PHASE = "Phase 8S observation-informed pelagic-support corridor sensitivity"
SCRIPT = "scripts/run_phase8s_observation_informed_corridor_sensitivity.py"
YEARS = (2017, 2018)
SCENARIOS = ("prey", "prey_ice", "prey_vessel", "prey_ice_vessel")
SCENARIO_LABELS = {
    "prey": "Benthic + pelagic prey",
    "prey_ice": "Benthic + pelagic prey + ice",
    "prey_vessel": "Benthic + pelagic prey + vessel",
    "prey_ice_vessel": "Benthic + pelagic prey + ice + vessel",
}
CENTRAL_DESIGN = {
    "complementary": (0.00, 0.10, 0.25, 0.50, 1.00),
    "maximum": (0.25, 0.50),
    "additive_capped": (0.25, 0.50),
}
UNCERTAINTY_OPERATOR = "complementary"
UNCERTAINTY_BETA = 0.25
UNCERTAINTY_DRAWS = 100
EPSILON = 0.001
FREQUENT_THRESHOLD = 0.05

OUT = ROOT / "outputs/pelagic_support_phase8s"
MODEL_TABLES = OUT / "tables"
MODEL_SURFACES = OUT / "surfaces"
CORRIDOR_OUT = OUT / "corridor_sensitivity"
SURFACES = CORRIDOR_OUT / "surfaces"
LCP = CORRIDOR_OUT / "lcp"
SUMMARIES = CORRIDOR_OUT / "summaries"
VALIDATION = CORRIDOR_OUT / "validation"
STATUS = CORRIDOR_OUT / "phase8s_corridor_run_status.json"
MANIFEST = CORRIDOR_OUT / "phase8s_corridor_manifest.json"

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
ASAMM_SIGHTINGS = ROOT / "data/processed/validation/asamm_phase8i2_gray_whale_records_on_C3_grid.csv"
ASAMM_EFFORT = ROOT / "data/processed/validation/asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
MODEL_GATES = MODEL_TABLES / "phase8s_model_integration_gates.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    for path in (CORRIDOR_OUT, SURFACES, LCP, SUMMARIES, VALIDATION):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_stage(stage: str, status: str, **details: object) -> None:
    if STATUS.exists():
        state = json.loads(STATUS.read_text(encoding="utf-8"))
    else:
        state = {
            "phase": PHASE,
            "status": "running",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "stages": {},
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
        }
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    state["updated_utc"] = utc_now()
    if status == "failed":
        state["status"] = "failed"
    write_json(STATUS, state)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_path(year: int) -> Path:
    return MODEL_SURFACES / f"phase8s_observation_informed_pelagic_support_target{year}.csv"


def bootstrap_path(year: int) -> Path:
    return MODEL_SURFACES / f"phase8s_observation_informed_pelagic_support_bootstrap_target{year}.npz"


def input_paths() -> list[Path]:
    paths = [GRID, ENDPOINTS, ASAMM_SIGHTINGS, ASAMM_EFFORT, MODEL_GATES]
    for year in YEARS:
        paths.extend(
            [
                support_path(year),
                bootstrap_path(year),
                phase8r.prey_path(year),
                phase8r.ice_path(year),
                phase8r.ship_path(year),
            ]
        )
        for scenario in SCENARIOS:
            paths.append(phase8r.established_grid_path(scenario, year))
    paths.extend(phase8r.established_route_path(scenario) for scenario in SCENARIOS)
    return list(dict.fromkeys(paths))


def input_snapshot(label: str) -> pd.DataFrame:
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
    frame.to_csv(CORRIDOR_OUT / f"phase8s_corridor_input_integrity_{label}.csv", index=False)
    return frame


def beta_label(beta: float) -> str:
    return f"b{int(round(beta * 100)):03d}"


def central_surface_path(operator: str, beta: float, year: int) -> Path:
    return SURFACES / f"phase8s_{operator}_{beta_label(beta)}_utility_target{year}.csv"


def central_grid_path(operator: str, beta: float, scenario: str, year: int) -> Path:
    return LCP / f"phase8s_{operator}_{beta_label(beta)}_{scenario}_corridor_target{year}.csv"


def resistance(utility: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(-np.log(np.asarray(utility, dtype=float) + EPSILON), 0.0)


def combine_support(benthic: np.ndarray, pelagic: np.ndarray, beta: float, operator: str) -> np.ndarray:
    benthic = np.asarray(benthic, dtype=float)
    pelagic = np.asarray(pelagic, dtype=float)
    weighted = beta * pelagic
    if operator == "complementary":
        combined = 1.0 - (1.0 - benthic) * (1.0 - weighted)
    elif operator == "maximum":
        combined = np.maximum(benthic, weighted)
    elif operator == "additive_capped":
        combined = np.minimum(1.0, benthic + weighted)
    else:
        raise KeyError(operator)
    return np.clip(combined, EPSILON, 1.0)


def load_year_inputs(year: int, pelagic_support: np.ndarray | None = None) -> pd.DataFrame:
    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    grid = grid.loc[grid["ocean_flag"].eq(1)].copy()
    support = pd.read_csv(support_path(year), dtype={"grid_id": str})
    prey = pd.read_csv(phase8r.prey_path(year), dtype={"grid_id": str})
    ice = pd.read_csv(phase8r.ice_path(year), dtype={"grid_id": str})
    vessel = pd.read_csv(phase8r.ship_path(year), dtype={"grid_id": str})
    frame = (
        grid[["grid_id", "grid_x", "grid_y", "x", "y", "lon", "lat", "ocean_flag"]]
        .merge(
            prey[["grid_id", "U_amphipod_prey_global_q05_q95", "reliability_flag"]],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            ice[["grid_id", "A_ice_v1_MayJul"]],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            vessel[["grid_id", "S_scaled_ship_primary_MayJul_mean", "D_ship_alpha0_5_MayJul_mean"]],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            support[
                [
                    "grid_id",
                    "pelagic_evidence_domain",
                    "nearest_observation_distance_km",
                    "P_pelagic_central",
                    "P_pelagic_bootstrap_interval_width",
                ]
            ],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )
    required = [
        "U_amphipod_prey_global_q05_q95",
        "A_ice_v1_MayJul",
        "D_ship_alpha0_5_MayJul_mean",
        "P_pelagic_central",
    ]
    if frame[required].isna().any().any():
        raise RuntimeError(f"Incomplete Phase 8S input merge for {year}")
    frame.rename(
        columns={
            "U_amphipod_prey_global_q05_q95": "U_benthic",
            "S_scaled_ship_primary_MayJul_mean": "S_scaled_vessel_MayJul",
            "D_ship_alpha0_5_MayJul_mean": "D_vessel_MayJul",
        },
        inplace=True,
    )
    frame["target_year"] = year
    if pelagic_support is not None:
        if len(pelagic_support) != len(frame):
            raise RuntimeError("Bootstrap support length does not match the study grid")
        frame["P_pelagic"] = np.asarray(pelagic_support, dtype=float)
    else:
        frame["P_pelagic"] = frame["P_pelagic_central"]
    frame["U_benthic"] = frame["U_benthic"].clip(EPSILON, 1.0)
    frame["A_ice_v1_MayJul"] = frame["A_ice_v1_MayJul"].clip(0.05, 1.0)
    frame["D_vessel_MayJul"] = frame["D_vessel_MayJul"].clip(0.05, 1.0)
    frame["P_pelagic"] = frame["P_pelagic"].clip(0.0, 1.0)
    return frame


def build_utility_surface(base: pd.DataFrame, operator: str, beta: float) -> tuple[pd.DataFrame, dict]:
    output = base.copy()
    output["combination_operator"] = operator
    output["pelagic_strength_beta"] = beta
    output["U_benthic_pelagic"] = combine_support(
        output["U_benthic"].to_numpy(),
        output["P_pelagic"].to_numpy(),
        beta,
        operator,
    )
    output["pelagic_utility_increment"] = output["U_benthic_pelagic"] - output["U_benthic"]
    output["U_prey"] = output["U_benthic_pelagic"]
    output["U_prey_ice"] = output["U_benthic_pelagic"] * output["A_ice_v1_MayJul"]
    output["U_prey_vessel"] = output["U_benthic_pelagic"] * output["D_vessel_MayJul"]
    output["U_prey_ice_vessel"] = (
        output["U_benthic_pelagic"] * output["A_ice_v1_MayJul"] * output["D_vessel_MayJul"]
    )
    qa = {
        "target_year": int(output["target_year"].iloc[0]),
        "operator": operator,
        "beta": beta,
        "grid_cell_count": len(output),
        "evidence_domain_cell_count": int(output["pelagic_evidence_domain"].sum()),
        "support_nonzero_cell_count": int((output["P_pelagic"] > 0).sum()),
        "combined_below_benthic_cell_count": int(
            (output["U_benthic_pelagic"] < output["U_benthic"] - 1e-12).sum()
        ),
        "combined_above_one_cell_count": int((output["U_benthic_pelagic"] > 1 + 1e-12).sum()),
        "outside_evidence_increment_cell_count": int(
            ((~output["pelagic_evidence_domain"].astype(bool)) & (output["pelagic_utility_increment"] > 1e-12)).sum()
        ),
        "mean_utility_increment_inside_evidence_domain": float(
            output.loc[output["pelagic_evidence_domain"].astype(bool), "pelagic_utility_increment"].mean()
        ),
    }
    if any(
        qa[key] > 0
        for key in (
            "combined_below_benthic_cell_count",
            "combined_above_one_cell_count",
            "outside_evidence_increment_cell_count",
        )
    ):
        raise RuntimeError(f"Utility QA failed: {qa}")
    return output, qa


def run_lcp(
    topology,
    surface: pd.DataFrame,
    scenario: str,
    retain_paths: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    merged = topology.grid.merge(
        surface,
        on="grid_id",
        how="left",
        suffixes=("", "_surface"),
        validate="one_to_one",
    )
    merged = merged.loc[merged["ocean_flag"].eq(1)].copy().reset_index(drop=True)
    utility_column = f"U_{scenario}"
    if merged[utility_column].isna().any():
        raise RuntimeError("Scenario utility is incomplete after topology alignment")
    merged["modeled_utility"] = merged[utility_column].clip(EPSILON, 1.0)
    merged["modeled_resistance"] = resistance(merged["modeled_utility"])
    adjacency = build_weighted_adjacency(topology.edge_pairs, merged["modeled_resistance"].to_numpy())
    paths, counts = run_ensemble(adjacency, topology)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    if len(paths) != attempted:
        raise RuntimeError(f"Only {len(paths)} of {attempted} paths succeeded")
    merged["lcp_count"] = counts
    merged["lcp_frequency_normalized"] = counts / len(paths)
    frequent = merged["lcp_frequency_normalized"] > FREQUENT_THRESHOLD
    frequent_cells = merged.loc[frequent]
    total_count = float(merged["lcp_count"].sum())
    evidence = merged["pelagic_evidence_domain"].astype(bool)

    path_rows = []
    costs = []
    lengths = []
    for item in paths:
        costs.append(item.cost)
        length = path_length_km(item.node_indices, merged)
        lengths.append(length)
        if retain_paths:
            sub = merged.iloc[list(item.node_indices)]
            path_rows.append(
                {
                    "target_year": int(merged["target_year"].iloc[0]),
                    "combination_operator": merged["combination_operator"].iloc[0],
                    "pelagic_strength_beta": float(merged["pelagic_strength_beta"].iloc[0]),
                    "scenario": scenario,
                    "source_grid_id": topology.grid_id_by_node_index[item.source_index],
                    "destination_grid_id": topology.grid_id_by_node_index[item.destination_index],
                    "path_cost": item.cost,
                    "path_length_km": length,
                    "path_cell_count": len(item.node_indices),
                    "path_evidence_domain_cell_count": int(sub["pelagic_evidence_domain"].sum()),
                    "path_mean_pelagic_support": float(sub["P_pelagic"].mean()),
                }
            )
    route = {
        "target_year": int(merged["target_year"].iloc[0]),
        "combination_operator": merged["combination_operator"].iloc[0],
        "pelagic_strength_beta": float(merged["pelagic_strength_beta"].iloc[0]),
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "attempted_path_count": attempted,
        "successful_path_count": len(paths),
        "failed_path_count": attempted - len(paths),
        "mean_path_cost": float(np.mean(costs)),
        "median_path_cost": float(np.median(costs)),
        "mean_path_length_km": float(np.mean(lengths)),
        "frequent_corridor_cell_count": int(frequent.sum()),
        "frequent_centroid_x": float(frequent_cells["x"].mean()),
        "frequent_centroid_y": float(frequent_cells["y"].mean()),
        "frequent_centroid_lon": float(frequent_cells["lon"].mean()),
        "frequent_centroid_lat": float(frequent_cells["lat"].mean()),
        "evidence_domain_cells_used": int((evidence & merged["lcp_count"].gt(0)).sum()),
        "evidence_domain_frequent_cells": int((evidence & frequent).sum()),
        "share_total_lcp_count_inside_evidence_domain_percent": float(
            merged.loc[evidence, "lcp_count"].sum() / total_count * 100.0
        ),
        "mean_pelagic_support_weighted_by_lcp_count": float(
            np.average(merged["P_pelagic"], weights=merged["lcp_count"])
        ),
    }
    keep = [
        "grid_id",
        "x",
        "y",
        "lon",
        "lat",
        "target_year",
        "pelagic_evidence_domain",
        "P_pelagic",
        "P_pelagic_bootstrap_interval_width",
        "U_benthic",
        "U_benthic_pelagic",
        "pelagic_utility_increment",
        "A_ice_v1_MayJul",
        "S_scaled_vessel_MayJul",
        "D_vessel_MayJul",
        utility_column,
        "modeled_utility",
        "modeled_resistance",
        "lcp_count",
        "lcp_frequency_normalized",
        "combination_operator",
        "pelagic_strength_beta",
    ]
    return merged[keep].copy(), pd.DataFrame(path_rows) if retain_paths else None, route


def jaccard_from_frequency(first: pd.Series, second: pd.Series) -> float:
    first_set = first.to_numpy(dtype=float) > FREQUENT_THRESHOLD
    second_set = second.to_numpy(dtype=float) > FREQUENT_THRESHOLD
    union = np.logical_or(first_set, second_set).sum()
    return float(np.logical_and(first_set, second_set).sum() / union) if union else math.nan


def centroid_shift_km(current: dict, baseline: dict) -> float:
    return float(
        math.hypot(
            current["frequent_centroid_x"] - baseline["frequent_centroid_x"],
            current["frequent_centroid_y"] - baseline["frequent_centroid_y"],
        )
        / 1000.0
    )


def load_asamm() -> tuple[pd.DataFrame, pd.DataFrame]:
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    sightings = sightings.loc[
        sightings["use_for_C3_validation"].astype(bool)
        & sightings["year"].isin(YEARS)
        & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)
    effort = pd.read_csv(ASAMM_EFFORT, dtype={"grid_id": str})
    effort = effort.loc[
        effort["year"].isin(YEARS)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    return sightings, effort


def asamm_metrics(
    grid: pd.DataFrame,
    sightings: pd.DataFrame,
    effort: pd.DataFrame,
) -> dict[str, float | int]:
    year = int(grid["target_year"].iloc[0])
    annual_sightings = sightings.loc[sightings["year"].eq(year)]
    annual_effort = set(effort.loc[effort["year"].eq(year), "grid_id"])
    lookup = grid.set_index("grid_id")
    effort_ids = lookup.index.intersection(annual_effort)
    effort_frequency = lookup.loc[effort_ids, "lcp_frequency_normalized"].fillna(0.0)
    frequency = grid["lcp_frequency_normalized"].fillna(0.0)
    q95 = float(frequency.quantile(0.95))
    frequent_xy = grid.loc[frequency > FREQUENT_THRESHOLD, ["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(frequent_xy) if len(frequent_xy) else None
    percentiles = []
    top5 = []
    distances = []
    for _, record in annual_sightings.iterrows():
        grid_id = record["nearest_grid_id"]
        if grid_id not in lookup.index:
            continue
        cell = lookup.loc[grid_id]
        value = float(cell["lcp_frequency_normalized"])
        percentiles.append(
            float(
                ((effort_frequency < value).sum() + 0.5 * (effort_frequency == value).sum())
                / len(effort_frequency)
                * 100.0
            )
        )
        top5.append(value >= q95)
        if tree is not None:
            distances.append(
                float(tree.query([[float(cell["x"]), float(cell["y"])]], k=1)[0][0] / 1000.0)
            )
    return {
        "sighting_record_count": len(percentiles),
        "strict_effort_background_cell_count": len(effort_frequency),
        "mean_effort_footprint_percentile": float(np.mean(percentiles)),
        "percent_records_top5pct": float(np.mean(top5) * 100.0),
        "median_distance_to_frequent_corridor_km": float(np.median(distances)),
    }


def baseline_comparison(
    grid: pd.DataFrame,
    route: dict,
    scenario: str,
    year: int,
) -> dict:
    established = pd.read_csv(phase8r.established_grid_path(scenario, year), dtype={"grid_id": str})
    comparison = grid[["grid_id", "lcp_frequency_normalized"]].merge(
        established[["grid_id", "lcp_frequency_normalized"]],
        on="grid_id",
        suffixes=("_current", "_baseline"),
        validate="one_to_one",
    )
    baseline_route = {
        "mean_path_cost": phase8r.established_mean_cost(scenario, year),
    }
    baseline_frequency = comparison["lcp_frequency_normalized_baseline"]
    baseline_frequent = established.loc[
        established["lcp_frequency_normalized"] > FREQUENT_THRESHOLD
    ]
    baseline_route.update(
        {
            "frequent_corridor_cell_count": int(
                (established["lcp_frequency_normalized"] > FREQUENT_THRESHOLD).sum()
            ),
            "frequent_centroid_x": float(baseline_frequent["x"].mean()),
            "frequent_centroid_y": float(baseline_frequent["y"].mean()),
        }
    )
    return {
        "baseline_mean_path_cost": baseline_route["mean_path_cost"],
        "delta_mean_path_cost_percent": float(
            (route["mean_path_cost"] - baseline_route["mean_path_cost"])
            / baseline_route["mean_path_cost"]
            * 100.0
        ),
        "baseline_frequent_corridor_cell_count": baseline_route["frequent_corridor_cell_count"],
        "delta_frequent_corridor_cell_count": int(
            route["frequent_corridor_cell_count"] - baseline_route["frequent_corridor_cell_count"]
        ),
        "jaccard_with_baseline_frequent_cells": jaccard_from_frequency(
            comparison["lcp_frequency_normalized_current"], baseline_frequency
        ),
        "frequent_centroid_shift_from_baseline_km": centroid_shift_km(route, baseline_route),
        "maximum_frequency_absolute_error_vs_baseline": float(
            np.max(
                np.abs(
                    comparison["lcp_frequency_normalized_current"]
                    - comparison["lcp_frequency_normalized_baseline"]
                )
            )
        ),
    }


def run_central(topology, sightings: pd.DataFrame, effort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_stage("central_corridors", "running")
    route_rows = []
    validation_rows = []
    surface_qa_rows = []
    path_groups: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    run_count = 0
    for operator, betas in CENTRAL_DESIGN.items():
        for beta in betas:
            for year in YEARS:
                base = load_year_inputs(year)
                surface, surface_qa = build_utility_surface(base, operator, beta)
                surface.to_csv(central_surface_path(operator, beta, year), index=False)
                surface_qa_rows.append(surface_qa)
                for scenario in SCENARIOS:
                    grid, paths, route = run_lcp(topology, surface, scenario, retain_paths=True)
                    comparison = baseline_comparison(grid, route, scenario, year)
                    route.update(comparison)
                    route_rows.append(route)
                    metrics = asamm_metrics(grid, sightings, effort)
                    validation_rows.append({**route, **metrics})
                    grid.to_csv(central_grid_path(operator, beta, scenario, year), index=False)
                    key = (operator, beta_label(beta), scenario)
                    path_groups.setdefault(key, []).append(paths)
                    run_count += 1
    for (operator, beta_name, scenario), pieces in path_groups.items():
        pd.concat(pieces, ignore_index=True).to_csv(
            LCP / f"phase8s_{operator}_{beta_name}_{scenario}_path_summary.csv", index=False
        )
    route_frame = pd.DataFrame(route_rows)
    validation_frame = pd.DataFrame(validation_rows)
    route_frame.to_csv(SUMMARIES / "phase8s_central_route_summary.csv", index=False)
    validation_frame.to_csv(VALIDATION / "phase8s_central_asamm_validation_summary.csv", index=False)
    pd.DataFrame(surface_qa_rows).to_csv(SUMMARIES / "phase8s_central_utility_surface_qa.csv", index=False)

    beta_zero = route_frame.loc[
        route_frame["combination_operator"].eq("complementary")
        & route_frame["pelagic_strength_beta"].eq(0.0)
    ]
    if len(beta_zero) != len(YEARS) * len(SCENARIOS):
        raise RuntimeError("Incomplete beta-zero equivalence rows")
    if beta_zero["maximum_frequency_absolute_error_vs_baseline"].max() > 1e-12:
        raise RuntimeError("Beta-zero LCP frequency does not reproduce the established baseline")
    if beta_zero["delta_mean_path_cost_percent"].abs().max() > 1e-10:
        raise RuntimeError("Beta-zero path cost does not reproduce the established baseline")
    set_stage(
        "central_corridors",
        "completed",
        run_count=run_count,
        baseline_max_frequency_error=float(beta_zero["maximum_frequency_absolute_error_vs_baseline"].max()),
        warnings=[],
    )
    return route_frame, validation_frame


def align_bootstrap_draws(year: int, base: pd.DataFrame) -> np.ndarray:
    archive = np.load(bootstrap_path(year))
    archive_grid = archive["grid_id"].astype(str)
    draws = archive["support_draws"]
    if draws.shape[0] < UNCERTAINTY_DRAWS:
        raise RuntimeError("Insufficient saved bootstrap draws")
    order = pd.Series(np.arange(len(archive_grid)), index=archive_grid)
    if not set(base["grid_id"]).issubset(order.index):
        raise RuntimeError("Bootstrap archive grid IDs do not cover the LCP grid")
    positions = order.loc[base["grid_id"]].to_numpy(dtype=int)
    return draws[:UNCERTAINTY_DRAWS, positions]


def run_uncertainty(topology, sightings: pd.DataFrame, effort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_stage("bootstrap_corridors", "running")
    route_rows = []
    validation_rows = []
    baseline_cache = {}
    for year in YEARS:
        base = load_year_inputs(year)
        draws = align_bootstrap_draws(year, base)
        for draw_index, draw in enumerate(draws):
            draw_base = load_year_inputs(year, pelagic_support=draw)
            surface, _ = build_utility_surface(draw_base, UNCERTAINTY_OPERATOR, UNCERTAINTY_BETA)
            for scenario in SCENARIOS:
                grid, _, route = run_lcp(topology, surface, scenario, retain_paths=False)
                key = (scenario, year)
                if key not in baseline_cache:
                    baseline_cache[key] = baseline_comparison(grid, route, scenario, year)
                comparison = baseline_comparison(grid, route, scenario, year)
                route_rows.append({"bootstrap_draw": draw_index, **route, **comparison})
                validation_rows.append(
                    {"bootstrap_draw": draw_index, **route, **comparison, **asamm_metrics(grid, sightings, effort)}
                )
            if (draw_index + 1) % 10 == 0:
                print(f"Phase 8S bootstrap: year={year}, completed {draw_index + 1}/{UNCERTAINTY_DRAWS}", flush=True)
    routes = pd.DataFrame(route_rows)
    validation = pd.DataFrame(validation_rows)
    routes.to_csv(SUMMARIES / "phase8s_bootstrap_route_metrics_beta025.csv", index=False)
    validation.to_csv(VALIDATION / "phase8s_bootstrap_asamm_metrics_beta025.csv", index=False)

    route_metrics = [
        "delta_mean_path_cost_percent",
        "delta_frequent_corridor_cell_count",
        "jaccard_with_baseline_frequent_cells",
        "frequent_centroid_shift_from_baseline_km",
        "share_total_lcp_count_inside_evidence_domain_percent",
    ]
    validation_metrics = [
        "mean_effort_footprint_percentile",
        "percent_records_top5pct",
        "median_distance_to_frequent_corridor_km",
    ]

    def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        rows = []
        for (year, scenario), group in frame.groupby(["target_year", "scenario"]):
            for column in columns:
                rows.append(
                    {
                        "target_year": year,
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "metric": column,
                        "bootstrap_draw_count": group["bootstrap_draw"].nunique(),
                        "q05": float(group[column].quantile(0.05)),
                        "median": float(group[column].median()),
                        "q95": float(group[column].quantile(0.95)),
                    }
                )
        return pd.DataFrame(rows)

    summarize(routes, route_metrics).to_csv(
        SUMMARIES / "phase8s_bootstrap_route_metric_intervals_beta025.csv", index=False
    )
    summarize(validation, validation_metrics).to_csv(
        VALIDATION / "phase8s_bootstrap_asamm_metric_intervals_beta025.csv", index=False
    )
    set_stage(
        "bootstrap_corridors",
        "completed",
        bootstrap_draw_count=UNCERTAINTY_DRAWS,
        lcp_run_count=len(routes),
        warnings=[],
    )
    return routes, validation


def smoke_test(topology) -> None:
    set_stage("smoke_test", "running")
    year = 2018
    base = load_year_inputs(year)
    surface, qa = build_utility_surface(base, "complementary", 0.25)
    rows = []
    for scenario in SCENARIOS:
        grid, _, route = run_lcp(topology, surface, scenario, retain_paths=False)
        rows.append({**route, **baseline_comparison(grid, route, scenario, year)})
    frame = pd.DataFrame(rows)
    frame.to_csv(CORRIDOR_OUT / "phase8s_smoke_test_target2018.csv", index=False)
    set_stage(
        "smoke_test",
        "completed",
        successful_paths_min=int(frame["successful_path_count"].min()),
        failed_paths_max=int(frame["failed_path_count"].max()),
        utility_qa=qa,
        warnings=[],
    )


def write_summary(route: pd.DataFrame, validation: pd.DataFrame) -> None:
    central = route.loc[
        route["combination_operator"].eq("complementary")
        & route["pelagic_strength_beta"].eq(0.25)
    ].copy()
    validation_central = validation.loc[
        validation["combination_operator"].eq("complementary")
        & validation["pelagic_strength_beta"].eq(0.25)
    ].copy()
    route_lines = []
    for row in central.sort_values(["target_year", "scenario"]).itertuples():
        route_lines.append(
            f"- {row.target_year}, {row.scenario_label}: mean cost {row.delta_mean_path_cost_percent:+.2f}%, "
            f"frequent area {row.delta_frequent_corridor_cell_count:+d} cells, "
            f"Jaccard with benthic baseline {row.jaccard_with_baseline_frequent_cells:.3f}, "
            f"centroid shift {row.frequent_centroid_shift_from_baseline_km:.1f} km."
        )
    validation_lines = []
    for row in validation_central.sort_values(["target_year", "scenario"]).itertuples():
        validation_lines.append(
            f"- {row.target_year}, {row.scenario_label}: n={row.sighting_record_count}, "
            f"effort-footprint percentile {row.mean_effort_footprint_percentile:.1f}, "
            f"top-5% overlap {row.percent_records_top5pct:.1f}%, "
            f"median distance {row.median_distance_to_frequent_corridor_km:.1f} km."
        )
    text = f"""# Phase 8S corridor sensitivity summary

## Scope

The corridor experiment is restricted to June 2017 and 2018, the two years whose observation-informed euphausiid support surfaces passed the model gate. It does not create a 2011-2019 pelagic time series.

## Combination contract

The primary bounded operator is `U_BP = 1 - (1 - U_B)(1 - beta P)`, where `P` is the empirical relative-support surface and `beta` is an uncalibrated sensitivity parameter. `maximum` and capped-additive operators bracket stronger assumptions about substitution and additivity. ASAMM observations were not used to choose an operator or beta.

## Medium-strength central diagnostic (`beta = 0.25`)

{chr(10).join(route_lines)}

## Same-year ASAMM observed-use comparison

{chr(10).join(validation_lines)}

## Interpretation boundary

These are sensitivity results for whether the observed pelagic spatial pattern can reorganize established corridors. They do not estimate the energetic equivalence of pelagic and benthic prey, annual euphausiid dynamics outside 2017-2018, whale abundance or behavioral response.
"""
    (CORRIDOR_OUT / "phase8s_corridor_results_summary.md").write_text(text, encoding="utf-8")


def build_manifest() -> None:
    outputs = sorted(path for path in CORRIDOR_OUT.rglob("*") if path.is_file() and path != MANIFEST)
    payload = {
        "phase": PHASE,
        "script": SCRIPT,
        "created_utc": utc_now(),
        "central_design": {key: list(value) for key, value in CENTRAL_DESIGN.items()},
        "uncertainty_design": {
            "operator": UNCERTAINTY_OPERATOR,
            "beta": UNCERTAINTY_BETA,
            "bootstrap_draws": UNCERTAINTY_DRAWS,
            "reported_interval": "5th-95th percentile across spatial-block bootstrap surfaces",
        },
        "input_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in input_paths()
        ],
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
        "established_output_modified": False,
        "manuscript_or_si_modified": False,
    }
    write_json(MANIFEST, payload)


def verify_model_gates() -> None:
    gates = pd.read_csv(MODEL_GATES)
    for year in YEARS:
        row = gates.loc[gates["gate"].eq(f"year_specific_surface_{year}")]
        if len(row) != 1 or row.iloc[0]["decision"] != "pass":
            raise RuntimeError(f"Phase 8S model gate did not pass for {year}")
    annual = gates.loc[gates["gate"].eq("annual_2011_2019_dynamic_pelagic_surface")]
    if len(annual) != 1 or annual.iloc[0]["decision"] != "fail":
        raise RuntimeError("Annual extrapolation gate must remain closed")


def run_all() -> None:
    ensure_directories()
    set_stage("input_audit", "running")
    before = input_snapshot("before")
    verify_model_gates()
    topology = phase8r.load_topology()
    sightings, effort = load_asamm()
    set_stage(
        "input_audit",
        "completed",
        input_file_count=len(before),
        asamm_sighting_records=len(sightings),
        warnings=[],
    )
    smoke_test(topology)
    route, validation = run_central(topology, sightings, effort)
    run_uncertainty(topology, sightings, effort)
    after = input_snapshot("after")
    comparison = before.merge(after, on="file_path", suffixes=("_before", "_after"), validate="one_to_one")
    comparison["unchanged"] = (
        comparison["sha256_before"].eq(comparison["sha256_after"])
        & comparison["size_bytes_before"].eq(comparison["size_bytes_after"])
    )
    comparison.to_csv(CORRIDOR_OUT / "phase8s_corridor_input_integrity_comparison.csv", index=False)
    if not comparison["unchanged"].all():
        raise RuntimeError("An established input changed during Phase 8S")
    write_summary(route, validation)
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    state["status"] = "completed"
    state["updated_utc"] = utc_now()
    write_json(STATUS, state)
    build_manifest()
    print(route.loc[route["combination_operator"].eq("complementary") & route["pelagic_strength_beta"].eq(0.25)][
        [
            "target_year",
            "scenario",
            "delta_mean_path_cost_percent",
            "delta_frequent_corridor_cell_count",
            "jaccard_with_baseline_frequent_cells",
            "frequent_centroid_shift_from_baseline_km",
        ]
    ].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    verify_model_gates()
    topology = phase8r.load_topology()
    if args.mode == "smoke":
        smoke_test(topology)
    else:
        run_all()


if __name__ == "__main__":
    main()
