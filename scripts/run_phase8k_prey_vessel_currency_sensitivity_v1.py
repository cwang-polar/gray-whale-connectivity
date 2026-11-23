#!/usr/bin/env python3
"""Run an isolated prey-currency sensitivity extension of Phase 8K.

The extension recalculates the prey+vessel scenario for energy density,
wet-weight biomass, and carbon biomass during 2015-2019. Existing Phase 8K
outputs are read-only and are used to verify exact reproduction of the
energy-density branch.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase8k_prey_vessel_factorial_v1 as phase8k


TARGET_YEARS = list(range(2015, 2020))
VARIABLES = {
    "total_kJm2_frozen": {"short": "kJm2_frozen", "label": "Energy density"},
    "total_gww": {"short": "gww", "label": "Wet-weight biomass"},
    "total_gC": {"short": "gC", "label": "Carbon biomass"},
}

OUT = ROOT / "outputs" / "prey_vessel_phase8k" / "currency_sensitivity_v1"
SURFACE_OUT = OUT / "surfaces"
LCP_OUT = OUT / "lcp"
QA_OUT = OUT / "qa"
SMOKE_OUT = OUT / "smoke_test"
STATUS_PATH = OUT / "run_status_v1.json"
MANIFEST_PATH = OUT / "analysis_manifest_v1.json"

EXISTING_PHASE8K = ROOT / "outputs" / "prey_vessel_phase8k"
EXISTING_PREY = ROOT / "outputs" / "lcp_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (OUT, SURFACE_OUT, LCP_OUT, QA_OUT, SMOKE_OUT):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_status(stage: str, state: str, **details: object) -> None:
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        status = {
            "analysis": "Phase 8K prey+vessel prey-currency sensitivity",
            "created_utc": utc_now(),
            "stages": {},
        }
    status["updated_utc"] = utc_now()
    status["stages"][stage] = {"status": state, **details}
    write_json(STATUS_PATH, status)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prey_surface_path(variable: str, year: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "prey_v2"
        / f"amphipod_prey_surface_{variable}_target{year}_full_trailing_C3_v2.csv"
    )


def ship_surface_path(year: int) -> Path:
    return ROOT / "outputs" / "ship" / f"ship_intensity_MayJul_target{year}.csv"


def existing_prey_path_summary(variable: str) -> Path:
    short = VARIABLES[variable]["short"]
    return EXISTING_PREY / f"prey_only_brian_{short}_C3_path_summary_v2.csv"


def existing_energy_grid(year: int) -> Path:
    return (
        EXISTING_PHASE8K
        / "lcp"
        / f"prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    )


def existing_energy_surface(year: int) -> Path:
    return (
        EXISTING_PHASE8K
        / "surfaces"
        / f"prey_vessel_brian_kJm2_frozen_C3_utility_target{year}_v2.csv"
    )


def generated_surface_path(variable: str, year: int, root: Path = SURFACE_OUT) -> Path:
    short = VARIABLES[variable]["short"]
    return root / f"prey_vessel_brian_{short}_C3_utility_target{year}_v2.csv"


def generated_grid_path(variable: str, year: int, root: Path = LCP_OUT) -> Path:
    short = VARIABLES[variable]["short"]
    return root / f"prey_vessel_brian_{short}_C3_lcp_target{year}_currency_sensitivity_v1.csv"


def configure_phase8k(variable: str, surface_root: Path, lcp_root: Path) -> None:
    phase8k.PREY_VARIABLE = variable
    phase8k.PREY_LABEL = VARIABLES[variable]["short"]
    phase8k.SURFACE_OUT = surface_root
    phase8k.LCP_OUT = lcp_root


def audit_inputs() -> None:
    set_status("audit", "running")
    required = [phase8k.GRID, phase8k.ENDPOINTS]
    required.extend(ship_surface_path(year) for year in TARGET_YEARS)
    for variable in VARIABLES:
        required.extend(prey_surface_path(variable, year) for year in TARGET_YEARS)
        required.append(existing_prey_path_summary(variable))
    required.extend(existing_energy_grid(year) for year in TARGET_YEARS)
    required.extend(existing_energy_surface(year) for year in TARGET_YEARS)
    required.append(EXISTING_PHASE8K / "lcp" / "prey_vessel_brian_C3_route_summary_v2.csv")
    required = list(dict.fromkeys(required))

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    inventory = pd.DataFrame(
        [
            {
                "file": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
        ]
    ).sort_values("file")
    inventory.to_csv(OUT / "input_integrity_before_v1.csv", index=False)

    manifest = {
        "analysis": "isolated extension of Phase 8K",
        "role": "prey-currency robustness for the prey+vessel scenario",
        "target_years": TARGET_YEARS,
        "prey_variables": VARIABLES,
        "scenario": "prey+vessel",
        "vessel_window": "May-July",
        "vessel_modifier": "existing all-vessel alpha=0.5 modifier",
        "prey_surface_convention": "full trailing five-year surfaces",
        "utility_formula": "U_prey_vessel = U_prey * D_vessel",
        "resistance_formula": "R = max(-log(U + 0.001), 0)",
        "frequent_corridor_threshold": 0.05,
        "paths_per_variable_year": 1600,
        "existing_outputs_read_only": True,
        "created_utc": utc_now(),
    }
    write_json(MANIFEST_PATH, manifest)
    set_status("audit", "completed", input_files=len(inventory), missing_files=0)


def compare_paths_with_prey_only(variable: str, paths: pd.DataFrame) -> dict:
    prey = pd.read_csv(existing_prey_path_summary(variable), dtype={"source_grid_id": str, "destination_grid_id": str})
    prey = prey.loc[prey["target_year"].isin(TARGET_YEARS), [
        "target_year",
        "source_grid_id",
        "destination_grid_id",
        "path_cost",
    ]].rename(columns={"path_cost": "prey_only_path_cost"})
    candidate = paths.copy()
    candidate["source_grid_id"] = candidate["source_grid_id"].astype(str)
    candidate["destination_grid_id"] = candidate["destination_grid_id"].astype(str)
    paired = candidate.merge(
        prey,
        on=["target_year", "source_grid_id", "destination_grid_id"],
        how="inner",
    )
    expected = len(TARGET_YEARS) * 1600
    violations = int((paired["path_cost"] < paired["prey_only_path_cost"] - 1e-8).sum())
    if len(paired) != expected or violations:
        raise RuntimeError(
            f"{variable}: prey+vessel monotonicity failed; paired={len(paired)}, violations={violations}"
        )
    return {
        "paired_paths": int(len(paired)),
        "prey_vessel_cost_below_prey_only_count": violations,
        "minimum_added_path_cost": float((paired["path_cost"] - paired["prey_only_path_cost"]).min()),
    }


def compare_energy_reproduction(route_frame: pd.DataFrame, grid_root: Path, surface_root: Path) -> dict:
    existing_route = pd.read_csv(
        EXISTING_PHASE8K / "lcp" / "prey_vessel_brian_C3_route_summary_v2.csv"
    )
    existing_route = existing_route.loc[
        existing_route["prey_variable"].eq("total_kJm2_frozen")
        & existing_route["target_year"].isin(TARGET_YEARS)
    ]
    generated_route = route_frame.loc[route_frame["prey_variable"].eq("total_kJm2_frozen")]
    paired_route = generated_route.merge(
        existing_route[["target_year", "mean_path_cost", "corridor_area_cells_freq_gt_0_05"]],
        on="target_year",
        suffixes=("_generated", "_existing"),
    )
    cost_error = float(
        np.max(
            np.abs(
                paired_route["mean_path_cost_generated"]
                - paired_route["mean_path_cost_existing"]
            )
        )
    )
    area_error = int(
        np.max(
            np.abs(
                paired_route["corridor_area_cells_freq_gt_0_05_generated"]
                - paired_route["corridor_area_cells_freq_gt_0_05_existing"]
            )
        )
    )
    frequency_error = 0.0
    utility_error = 0.0
    for year in TARGET_YEARS:
        generated_grid = pd.read_csv(
            generated_grid_path("total_kJm2_frozen", year, grid_root), dtype={"grid_id": str}
        )
        existing_grid = pd.read_csv(existing_energy_grid(year), dtype={"grid_id": str})
        paired_grid = generated_grid[["grid_id", "lcp_frequency_normalized"]].merge(
            existing_grid[["grid_id", "lcp_frequency_normalized"]],
            on="grid_id",
            suffixes=("_generated", "_existing"),
        )
        frequency_error = max(
            frequency_error,
            float(
                np.max(
                    np.abs(
                        paired_grid["lcp_frequency_normalized_generated"]
                        - paired_grid["lcp_frequency_normalized_existing"]
                    )
                )
            ),
        )

        generated_surface = pd.read_csv(
            generated_surface_path("total_kJm2_frozen", year, surface_root), dtype={"grid_id": str}
        )
        existing_surface = pd.read_csv(existing_energy_surface(year), dtype={"grid_id": str})
        paired_surface = generated_surface[["grid_id", "U_prey_vessel_alpha0_5"]].merge(
            existing_surface[["grid_id", "U_prey_vessel_alpha0_5"]],
            on="grid_id",
            suffixes=("_generated", "_existing"),
        )
        utility_error = max(
            utility_error,
            float(
                np.max(
                    np.abs(
                        paired_surface["U_prey_vessel_alpha0_5_generated"]
                        - paired_surface["U_prey_vessel_alpha0_5_existing"]
                    )
                )
            ),
        )

    if cost_error > 1e-8 or area_error or frequency_error > 1e-12 or utility_error > 1e-12:
        raise RuntimeError(
            "Energy-density reproduction failed: "
            f"cost={cost_error}, area={area_error}, frequency={frequency_error}, utility={utility_error}"
        )
    return {
        "maximum_mean_path_cost_absolute_error": cost_error,
        "maximum_frequent_corridor_area_absolute_error_cells": area_error,
        "maximum_corridor_frequency_absolute_error": frequency_error,
        "maximum_utility_absolute_error": utility_error,
    }


def run_variables(
    variables: list[str],
    years: list[int],
    surface_root: Path,
    lcp_root: Path,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    topology, _ = phase8k.load_topology()
    all_routes: list[dict] = []
    qa: dict[str, dict] = {}

    for variable in variables:
        configure_phase8k(variable, surface_root, lcp_root)
        path_frames: list[pd.DataFrame] = []
        surface_rows: list[dict] = []
        for year in years:
            surface, surface_qa = phase8k.build_surface(year)
            surface.to_csv(generated_surface_path(variable, year, surface_root), index=False)
            grid, paths, route = phase8k.run_year(year, topology)
            grid.to_csv(generated_grid_path(variable, year, lcp_root), index=False)
            path_frames.append(paths)
            all_routes.append(route)
            surface_rows.append(surface_qa)
            print(
                f"{variable} {year}: {route['successful_path_count']}/"
                f"{route['attempted_path_count']} paths; mean cost={route['mean_path_cost']:.6f}; "
                f"frequent cells={route['corridor_area_cells_freq_gt_0_05']}",
                flush=True,
            )

        combined_paths = pd.concat(path_frames, ignore_index=True)
        short = VARIABLES[variable]["short"]
        combined_paths.to_csv(
            lcp_root / f"prey_vessel_brian_{short}_C3_path_summary_currency_sensitivity_v1.csv",
            index=False,
        )
        monotonicity = compare_paths_with_prey_only(variable, combined_paths) if years == TARGET_YEARS else {}
        qa[variable] = {
            "surface_checks": surface_rows,
            "successful_paths": int(len(combined_paths)),
            "failed_paths": 0,
            **monotonicity,
        }
    return pd.DataFrame(all_routes), qa


def run_smoke_test() -> None:
    set_status("smoke", "running")
    smoke_surfaces = SMOKE_OUT / "surfaces"
    smoke_lcp = SMOKE_OUT / "lcp"
    smoke_surfaces.mkdir(parents=True, exist_ok=True)
    smoke_lcp.mkdir(parents=True, exist_ok=True)
    routes, qa = run_variables(
        ["total_kJm2_frozen", "total_gww"],
        [2019],
        smoke_surfaces,
        smoke_lcp,
    )

    existing_route = pd.read_csv(
        EXISTING_PHASE8K / "lcp" / "prey_vessel_brian_C3_route_summary_v2.csv"
    )
    expected = existing_route.loc[
        existing_route["prey_variable"].eq("total_kJm2_frozen")
        & existing_route["target_year"].eq(2019)
    ].iloc[0]
    observed = routes.loc[routes["prey_variable"].eq("total_kJm2_frozen")].iloc[0]
    cost_error = abs(float(observed["mean_path_cost"]) - float(expected["mean_path_cost"]))
    area_error = abs(
        int(observed["corridor_area_cells_freq_gt_0_05"])
        - int(expected["corridor_area_cells_freq_gt_0_05"])
    )
    if cost_error > 1e-8 or area_error:
        raise RuntimeError(f"Smoke reproduction failed: cost={cost_error}, area={area_error}")
    payload = {
        "target_year": 2019,
        "variables": ["total_kJm2_frozen", "total_gww"],
        "energy_mean_path_cost_absolute_error": cost_error,
        "energy_frequent_area_absolute_error_cells": area_error,
        "successful_paths": {
            row.prey_variable: int(row.successful_path_count)
            for row in routes.itertuples(index=False)
        },
        "warnings": [],
    }
    write_json(QA_OUT / "smoke_test_2019_v1.json", payload)
    set_status("smoke", "completed", **payload)


def build_currency_comparison(route_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    energy = route_frame.loc[
        route_frame["prey_variable"].eq("total_kJm2_frozen")
    ].set_index("target_year")
    for row in route_frame.itertuples(index=False):
        baseline = energy.loc[row.target_year]
        grid = pd.read_csv(
            generated_grid_path(row.prey_variable, row.target_year), dtype={"grid_id": str}
        )[["grid_id", "lcp_frequency_normalized"]]
        energy_grid = pd.read_csv(
            generated_grid_path("total_kJm2_frozen", row.target_year), dtype={"grid_id": str}
        )[["grid_id", "lcp_frequency_normalized"]]
        paired = grid.merge(energy_grid, on="grid_id", suffixes=("_candidate", "_energy"))
        candidate_set = paired["lcp_frequency_normalized_candidate"] > phase8k.FREQUENT_THRESHOLD
        energy_set = paired["lcp_frequency_normalized_energy"] > phase8k.FREQUENT_THRESHOLD
        union = int((candidate_set | energy_set).sum())
        overlap = float((candidate_set & energy_set).sum() / union) if union else math.nan
        correlation = float(
            np.corrcoef(
                paired["lcp_frequency_normalized_candidate"],
                paired["lcp_frequency_normalized_energy"],
            )[0, 1]
        )
        rows.append(
            {
                "scenario": "prey+vessel",
                "target_year": int(row.target_year),
                "prey_variable": row.prey_variable,
                "prey_currency_label": VARIABLES[row.prey_variable]["label"],
                "mean_path_cost": float(row.mean_path_cost),
                "percent_cost_difference_vs_energy_density": float(
                    100.0 * (row.mean_path_cost - baseline["mean_path_cost"])
                    / baseline["mean_path_cost"]
                ),
                "corridor_area_freq_gt_0_05": int(row.corridor_area_cells_freq_gt_0_05),
                "energy_density_corridor_area_freq_gt_0_05": int(
                    baseline["corridor_area_cells_freq_gt_0_05"]
                ),
                "corridor_area_delta_vs_energy_density_cells": int(
                    row.corridor_area_cells_freq_gt_0_05
                    - baseline["corridor_area_cells_freq_gt_0_05"]
                ),
                "jaccard_vs_energy_density_freq_gt_0_05": overlap,
                "frequency_map_correlation_vs_energy_density": correlation,
                "successful_path_count": int(row.successful_path_count),
                "failed_path_count": int(row.failed_path_count),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_year", "prey_variable"])


def run_full() -> None:
    set_status("batch", "running")
    routes, variable_qa = run_variables(
        list(VARIABLES), TARGET_YEARS, SURFACE_OUT, LCP_OUT
    )
    routes = routes.sort_values(["prey_variable", "target_year"])
    routes.to_csv(LCP_OUT / "prey_vessel_brian_C3_route_summary_currency_sensitivity_v1.csv", index=False)

    reproduction = compare_energy_reproduction(routes, LCP_OUT, SURFACE_OUT)
    comparison = build_currency_comparison(routes)
    comparison.to_csv(OUT / "prey_vessel_currency_comparison_v1.csv", index=False)

    successful = int(routes["successful_path_count"].sum())
    failed = int(routes["failed_path_count"].sum())
    if successful != len(VARIABLES) * len(TARGET_YEARS) * 1600 or failed:
        raise RuntimeError(f"Unexpected path totals: successful={successful}, failed={failed}")

    payload = {
        "target_years": TARGET_YEARS,
        "prey_variables": list(VARIABLES),
        "variable_year_runs": int(len(routes)),
        "successful_paths": successful,
        "failed_paths": failed,
        "topology_warnings": list(phase8k.load_topology()[0].warnings),
        "energy_density_reproduction": reproduction,
        "variable_checks": variable_qa,
        "warnings": [],
    }
    write_json(QA_OUT / "prey_vessel_currency_sensitivity_qa_v1.json", payload)
    set_status(
        "batch",
        "completed",
        variable_year_runs=len(routes),
        successful_paths=successful,
        failed_paths=failed,
        energy_density_reproduction=reproduction,
    )


def verify_existing_inputs_unchanged() -> None:
    before = pd.read_csv(OUT / "input_integrity_before_v1.csv")
    after_rows = []
    for row in before.itertuples(index=False):
        path = ROOT / row.file
        after_rows.append(
            {
                "file": row.file,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    after = pd.DataFrame(after_rows).sort_values("file")
    after.to_csv(OUT / "input_integrity_after_v1.csv", index=False)
    check = before.merge(after, on="file", suffixes=("_before", "_after"))
    changed = check.loc[
        (check["size_bytes_before"] != check["size_bytes_after"])
        | (check["sha256_before"] != check["sha256_after"])
    ]
    if not changed.empty:
        raise RuntimeError(f"Existing inputs changed unexpectedly: {changed['file'].tolist()}")
    set_status("integrity", "completed", checked_files=len(check), changed_files=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("audit", "smoke", "full", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    audit_inputs()
    if args.stage == "audit":
        return
    if args.stage in {"smoke", "all"}:
        run_smoke_test()
    if args.stage in {"full", "all"}:
        run_full()
        verify_existing_inputs_unchanged()


if __name__ == "__main__":
    main()
