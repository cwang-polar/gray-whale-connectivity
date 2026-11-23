#!/usr/bin/env python3
"""Run isolated three-level, joint-anchor sea-ice sensitivity for Figure 3 v2."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_phase8t_primary_robustness_smoke_test as base


OUT = base.OUT / "ice_joint_anchor_v2"
SETTINGS = {
    "strict_lower_0_10_upper_0_45": (0.10, 0.45),
    "primary_lower_0_15_upper_0_55": (0.15, 0.55),
    "relaxed_lower_0_20_upper_0_65": (0.20, 0.65),
}
PRIMARY = "primary_lower_0_15_upper_0_55"


def protected_files() -> list[Path]:
    files = list(base.integrity_inputs())
    for label in ("figure2_idw_parameter_sensitivity", "figure3_ice_threshold_sensitivity"):
        stem = base.OUT / "figures/candidates" / f"gray_whale_{label}_candidate_v1"
        files.extend(stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff"))
    files.append(base.ROOT / "docs/coauthor_comment_resolution/phase8t_robustness_candidate_figure_captions_v1.md")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected input missing: {missing}")
    return sorted(set(files))


def integrity_snapshot(files: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(base.ROOT)): base.sha256(path) for path in files}


def accessibility(sic: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Ice anchors must satisfy 0 <= lower < upper <= 1")
    values = np.asarray(sic, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    result[values <= lower] = 1.0
    middle = (values > lower) & (values < upper)
    result[middle] = 1.0 - 0.95 * (values[middle] - lower) / (upper - lower)
    result[values >= upper] = 0.05
    return result


def annual_modifiers(year: int) -> dict[str, pd.DataFrame]:
    columns = ["grid_id", "SIC_May_filled", "SIC_June_filled", "SIC_July_filled"]
    sic = pd.read_csv(base.ice_path(year), usecols=columns)
    sic["grid_id"] = sic["grid_id"].astype(str)
    months = sic[columns[1:]].to_numpy(dtype=float)
    result: dict[str, pd.DataFrame] = {}
    for setting, (lower, upper) in SETTINGS.items():
        monthly = accessibility(months, lower, upper)
        if not np.isfinite(monthly).all():
            raise RuntimeError(f"Nonfinite ice accessibility in {setting}/{year}")
        result[setting] = pd.DataFrame(
            {"grid_id": sic["grid_id"], "modifier": monthly.mean(axis=1)}
        )
    strict = result["strict_lower_0_10_upper_0_45"]["modifier"].to_numpy()
    primary = result[PRIMARY]["modifier"].to_numpy()
    relaxed = result["relaxed_lower_0_20_upper_0_65"]["modifier"].to_numpy()
    if np.any(strict > primary + 1e-12) or np.any(primary > relaxed + 1e-12):
        raise RuntimeError(f"Joint ice settings are not ordered in {year}")
    frozen = base.primary_ice(year)
    check = result[PRIMARY].merge(frozen, on="grid_id", suffixes=("_reconstructed", "_frozen"), validate="one_to_one")
    delta = np.max(np.abs(check["modifier_reconstructed"] - check["modifier_frozen"]))
    if delta > 1e-12:
        raise RuntimeError(f"Primary ice accessibility differs from frozen input in {year}: {delta}")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = protected_files()
    before = integrity_snapshot(files)
    topology = base.load_topology()
    routes: list[dict] = []
    frequencies: list[pd.DataFrame] = []
    modifiers: list[dict] = []

    for year in base.YEARS:
        prey = base.primary_prey(year)
        ice_by_setting = annual_modifiers(year)
        vessel = None
        if year in base.VESSEL_YEARS:
            vessel = base.primary_vessel(year)[["grid_id", "D_ship_alpha0_5_MayJul_mean"]].rename(
                columns={"D_ship_alpha0_5_MayJul_mean": "modifier"}
            )
        for setting, (lower, upper) in SETTINGS.items():
            ice = ice_by_setting[setting]
            modifiers.append(
                {
                    "setting": setting,
                    "target_year": year,
                    "lower_sic": lower,
                    "upper_sic": upper,
                    "mean_accessibility": float(ice["modifier"].mean()),
                    "median_accessibility": float(ice["modifier"].median()),
                    "cells_accessibility_lt_0_50": int((ice["modifier"] < 0.50).sum()),
                }
            )
            for scenario in ("prey_ice", "prey_ice_vessel") if vessel is not None else ("prey_ice",):
                if setting == PRIMARY:
                    route, frequency = base.frozen_run("p2_joint_ice", setting, scenario, year)
                else:
                    utility = base.combine_utility(prey, ice, *( [vessel] if scenario == "prey_ice_vessel" else []))
                    route, frequency = base.run_lcp(topology, utility, "p2_joint_ice", setting, scenario, year)
                routes.append(route)
                frequencies.append(frequency)
        print(f"Joint ice anchors: completed {year}", flush=True)

    route_frame = pd.DataFrame(routes)
    frequency_frame = pd.concat(frequencies, ignore_index=True)
    comparisons = []
    for setting in SETTINGS:
        for year in base.YEARS:
            route = route_frame.loc[
                route_frame["setting"].eq(setting)
                & route_frame["scenario"].eq("prey_ice")
                & route_frame["target_year"].eq(year)
            ].iloc[0]
            frequency = frequency_frame.loc[
                frequency_frame["setting"].eq(setting)
                & frequency_frame["scenario"].eq("prey_ice")
                & frequency_frame["target_year"].eq(year)
            ]
            comparisons.append(
                base.comparison_row(
                    setting,
                    year,
                    "prey_only",
                    "prey_ice",
                    base.load_primary_route("prey_only", year),
                    route,
                    base.load_primary_grid("prey_only", year),
                    frequency,
                )
            )
    comparison_frame = pd.DataFrame(comparisons)

    cost_order = route_frame.pivot_table(index=["scenario", "target_year"], columns="setting", values="mean_path_cost")
    if not (
        (cost_order["strict_lower_0_10_upper_0_45"] >= cost_order[PRIMARY] - 1e-7).all()
        and (cost_order[PRIMARY] >= cost_order["relaxed_lower_0_20_upper_0_65"] - 1e-7).all()
    ):
        raise RuntimeError("Mean path costs violate the ordered ice-accessibility settings")
    if len(route_frame) != 42 or len(comparison_frame) != 27:
        raise RuntimeError("Joint-anchor run count is incomplete")

    route_frame.to_csv(OUT / "joint_ice_anchor_route_metrics.csv", index=False)
    frequency_frame.to_csv(OUT / "joint_ice_anchor_frequency_surfaces.csv.gz", index=False, compression="gzip")
    comparison_frame.to_csv(OUT / "joint_ice_anchor_prey_comparison_metrics.csv", index=False)
    pd.DataFrame(modifiers).to_csv(OUT / "joint_ice_anchor_accessibility_summary.csv", index=False)

    after = integrity_snapshot(files)
    changed = [path for path in before if before[path] != after[path]]
    if changed:
        raise RuntimeError(f"Protected inputs changed: {changed}")
    manifest = {
        "purpose": "isolated joint lower/upper SIC anchor sensitivity for Figure 3 candidate v2",
        "settings": {key: {"lower_sic": lower, "upper_sic": upper} for key, (lower, upper) in SETTINGS.items()},
        "route_scenario_years": len(route_frame),
        "prey_comparisons": len(comparison_frame),
        "paths_per_scenario_year": 1600,
        "protected_file_count": len(files),
        "protected_files_unchanged": True,
        "primary_ice_modifier_matches_frozen": True,
        "accessibility_and_path_cost_order_checks_passed": True,
        "interval_type": "joint-function parameter envelope; not a confidence interval",
    }
    (OUT / "joint_ice_anchor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(route_frame)} scenario-years; {len(files)} protected files unchanged", flush=True)


if __name__ == "__main__":
    main()
