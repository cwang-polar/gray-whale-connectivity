"""Phase 8D.1 benthic amphipod C3 prey + ice utility surfaces.

This phase combines the accepted benthic full-trailing C3 amphipod prey surfaces
with existing May-July v1 ice accessibility. It creates utility surfaces and
diagnostic figures only. It does not run LCP models and does not touch earlier
prey, ice, ship, validation, or LCP outputs.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_YEARS = list(range(2011, 2020))
PREY_VARIABLES = {
    "total_kJm2_frozen": {
        "short": "kJm2_frozen",
        "label": "kJ m-2 frozen",
        "role": "main_candidate",
    },
    "total_gww": {
        "short": "gww",
        "label": "g wet weight",
        "role": "sensitivity_candidate",
    },
    "total_gC": {
        "short": "gC",
        "label": "g C biomass",
        "role": "sensitivity_candidate",
    },
}
PREY_UTILITY_COL = "U_amphipod_prey_global_q05_q95"
ICE_COL = "A_ice_MayJul_mean_from_monthly_A_filled"
EPSILON = 0.001

PREY_DIR = ROOT / "outputs/prey_v2"
ICE_DIR = ROOT / "outputs/ice"
SURFACES_DIR = ROOT / "outputs/surfaces_v2"
FIG_DIR = ROOT / "outputs/figures/surfaces_v2"
GRID_PATH = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
MANIFEST_PATH = ROOT / "outputs/results_manifest.csv"
REGISTRY_PATH = ROOT / "docs/results_registry.md"
README_PATH = ROOT / "README.md"
CLAIMS_PATH = ROOT / "docs/paper_claims_evidence_matrix.md"
PHASE_MEMO = ROOT / "docs/phase_memos/phase8d1_brian_C3_prey_ice_utility_surfaces.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase8d1_brian_C3_prey_ice_utility_surfaces_paper_use.md"


def main() -> None:
    SURFACES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    c3_grid = pd.read_csv(GRID_PATH)
    c3_grid["grid_id"] = c3_grid["grid_id"].astype(str)
    c3_grid = c3_grid[(c3_grid["ocean_flag"] == 1) & (c3_grid["inside_C3_domain"].astype(bool))].copy()

    summary_rows: list[dict] = []
    annual_surfaces: dict[str, dict[int, pd.DataFrame]] = {v: {} for v in PREY_VARIABLES}
    qa = {
        "phase": "Phase 8D.1",
        "target_years": TARGET_YEARS,
        "domain": "C3 recurring-support v2 grid",
        "prey_surface_convention": "benthic amphipod full trailing 5-year C3 v2",
        "prey_variables": PREY_VARIABLES,
        "ice_layer": ICE_COL,
        "formula": "U_prey_ice = U_amphipod_prey_global_q05_q95 * A_ice_v1_MayJul",
        "resistance_formula": "R_log = max(-log(U + 0.001), 0) for non-negative LCP weights",
        "c3_grid_cell_count": int(len(c3_grid)),
        "merge_counts": {},
        "missing_ice_grid_id_counts": {},
        "missing_prey_grid_id_counts": {},
        "min_max": {},
        "warnings": [
            "This phase creates utility surfaces only; no LCP models are run.",
            "A_ice is an accessibility modifier, not mortality.",
            "Frozen-equivalent energy density is the primary prey variable; wet-weight and carbon biomass provide sensitivity comparisons.",
            "Resistance values are clipped at zero to avoid negative edge weights in later Dijkstra-based LCP runs.",
        ],
    }

    for prey_var, meta in PREY_VARIABLES.items():
        for year in TARGET_YEARS:
            surface, merge_info = build_surface_for_year_variable(year, prey_var, meta, c3_grid)
            out_path = SURFACES_DIR / f"prey_ice_brian_{meta['short']}_C3_utility_target{year}_v2.csv"
            surface.to_csv(out_path, index=False)
            annual_surfaces[prey_var][year] = surface
            summary_rows.append(summarize_surface(surface, prey_var, meta))
            qa["merge_counts"][f"{prey_var}_{year}"] = merge_info
            qa["missing_ice_grid_id_counts"][f"{prey_var}_{year}"] = merge_info["c3_cells_missing_ice"]
            qa["missing_prey_grid_id_counts"][f"{prey_var}_{year}"] = merge_info["c3_cells_missing_prey"]
            qa["min_max"][f"{prey_var}_{year}"] = {
                "U_prey": minmax(surface["U_prey_global"]),
                "A_ice": minmax(surface["A_ice_v1_MayJul"]),
                "U_prey_ice": minmax(surface["U_prey_ice_beta1"]),
                "delta_R_log_ice": minmax(surface["delta_R_log_ice_beta1"]),
            }

    summary = pd.DataFrame(summary_rows)
    summary_path = SURFACES_DIR / "prey_ice_brian_C3_utility_summary_v2.csv"
    summary.to_csv(summary_path, index=False)
    loss = build_loss_summary(summary)
    loss.to_csv(SURFACES_DIR / "prey_ice_brian_C3_utility_loss_by_year_v2.csv", index=False)
    comparison = build_variable_comparison(summary)
    comparison.to_csv(SURFACES_DIR / "prey_ice_brian_C3_variable_comparison_v2.csv", index=False)

    with (SURFACES_DIR / "prey_ice_brian_C3_utility_qa_v2.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)
        f.write("\n")

    make_figures(annual_surfaces, summary, comparison)
    write_docs(summary, loss, comparison, qa)
    update_registry_manifest_readme_claims()

    print(json.dumps({
        "phase": "Phase 8D.1",
        "surface_summary": str(summary_path),
        "qa": str(SURFACES_DIR / "prey_ice_brian_C3_utility_qa_v2.json"),
        "no_lcp_models_run": True,
        "no_ship_layer_used": True,
    }, indent=2))
    print(summary[summary["prey_variable"] == "total_kJm2_frozen"][
        ["target_year", "mean_U_prey", "mean_A_ice", "mean_U_prey_ice", "mean_delta_R_log_ice"]
    ].round(4).to_string(index=False))


def prey_path(prey_var: str, year: int) -> Path:
    return PREY_DIR / f"amphipod_prey_surface_{prey_var}_target{year}_full_trailing_C3_v2.csv"


def ice_path(year: int) -> Path:
    return ICE_DIR / f"ice_accessibility_v1_MayJul_target{year}.csv"


def build_surface_for_year_variable(year: int, prey_var: str, meta: dict, c3_grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prey = pd.read_csv(prey_path(prey_var, year))
    ice = pd.read_csv(ice_path(year))
    for df in (prey, ice):
        df["grid_id"] = df["grid_id"].astype(str)
    if PREY_UTILITY_COL not in prey.columns:
        raise ValueError(f"{PREY_UTILITY_COL} not found in {prey_path(prey_var, year)}")
    if ICE_COL not in ice.columns:
        candidates = [c for c in ice.columns if "A_ice" in c]
        raise ValueError(f"{ICE_COL} not found in {ice_path(year)}; candidates={candidates}")

    base = c3_grid[["grid_id", "grid_x", "grid_y", "x", "y", "lon", "lat", "ocean_flag", "inside_C3_domain"]].copy()
    merged = base.merge(
        prey[[
            "grid_id", "target_year", "window_start", "window_end", "prey_variable",
            "raw_final_value", PREY_UTILITY_COL, "support_station_year_count",
            "support_unique_station_count", "support_year_count", "support_years",
            "nearest_observed_distance_km", "reliability_flag", "supported_for_surface",
            "observed_window_flag", "observed_target_year_flag", "interpolated_flag",
        ]],
        on="grid_id",
        how="left",
        indicator="prey_merge",
    )
    merged = merged.merge(
        ice[[
            "grid_id", ICE_COL, "SIC_MayJul_mean_filled", "SIC_May_filled",
            "SIC_June_filled", "SIC_July_filled", "nan_open_water_fill_count",
            "nan_open_water_fill_percent", "sampling_method",
        ]],
        on="grid_id",
        how="left",
        indicator="ice_merge",
    )

    merge_info = {
        "c3_grid_cells": int(len(base)),
        "prey_rows": int(len(prey)),
        "ice_rows": int(len(ice)),
        "merged_rows": int(len(merged)),
        "c3_cells_missing_prey": int((merged["prey_merge"] == "left_only").sum()),
        "c3_cells_missing_ice": int((merged["ice_merge"] == "left_only").sum()),
    }
    if merge_info["c3_cells_missing_prey"] or merge_info["c3_cells_missing_ice"]:
        raise RuntimeError(f"Missing prey/ice coverage for {prey_var} {year}: {merge_info}")

    u_prey = pd.to_numeric(merged[PREY_UTILITY_COL], errors="coerce").fillna(EPSILON).clip(EPSILON, 1.0)
    a_ice = pd.to_numeric(merged[ICE_COL], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    u_prey_ice = (u_prey * a_ice).clip(EPSILON, 1.0)
    r_prey = resistance(u_prey)
    r_prey_ice = resistance(u_prey_ice)

    out = pd.DataFrame({
        "grid_id": merged["grid_id"],
        "grid_x": merged["grid_x"],
        "grid_y": merged["grid_y"],
        "x": merged["x"],
        "y": merged["y"],
        "lon": merged["lon"],
        "lat": merged["lat"],
        "target_year": year,
        "window_start": merged["window_start"].astype(int),
        "window_end": merged["window_end"].astype(int),
        "prey_variable": prey_var,
        "prey_variable_label": meta["label"],
        "prey_variable_role": meta["role"],
        "raw_amphipod_value": pd.to_numeric(merged["raw_final_value"], errors="coerce"),
        "U_prey_global": u_prey,
        "A_ice_v1_MayJul": a_ice,
        "SIC_MayJul_mean_filled": pd.to_numeric(merged["SIC_MayJul_mean_filled"], errors="coerce"),
        "SIC_May_filled": pd.to_numeric(merged["SIC_May_filled"], errors="coerce"),
        "SIC_June_filled": pd.to_numeric(merged["SIC_June_filled"], errors="coerce"),
        "SIC_July_filled": pd.to_numeric(merged["SIC_July_filled"], errors="coerce"),
        "U_prey_ice_beta1": u_prey_ice,
        "R_log_prey": r_prey,
        "R_log_prey_ice_beta1": r_prey_ice,
        "delta_R_log_ice_beta1": r_prey_ice - r_prey,
        "absolute_utility_loss_ice_beta1": u_prey - u_prey_ice,
        "relative_utility_loss_ice_beta1": np.where(u_prey > 0, (u_prey - u_prey_ice) / u_prey, np.nan),
        "support_station_year_count": pd.to_numeric(merged["support_station_year_count"], errors="coerce"),
        "support_unique_station_count": pd.to_numeric(merged["support_unique_station_count"], errors="coerce"),
        "support_year_count": pd.to_numeric(merged["support_year_count"], errors="coerce"),
        "support_years": merged["support_years"],
        "nearest_observed_distance_km": pd.to_numeric(merged["nearest_observed_distance_km"], errors="coerce"),
        "reliability_flag": merged["reliability_flag"],
        "supported_for_surface": merged["supported_for_surface"].astype(bool),
        "observed_window_flag": merged["observed_window_flag"].astype(bool),
        "observed_target_year_flag": merged["observed_target_year_flag"].astype(bool),
        "interpolated_flag": merged["interpolated_flag"].astype(bool),
        "nan_open_water_fill_count": pd.to_numeric(merged["nan_open_water_fill_count"], errors="coerce"),
        "nan_open_water_fill_percent": pd.to_numeric(merged["nan_open_water_fill_percent"], errors="coerce"),
        "ice_sampling_method": merged["sampling_method"],
        "ocean_flag": merged["ocean_flag"],
        "inside_C3_domain": merged["inside_C3_domain"].astype(bool),
    })
    return out, merge_info


def resistance(u: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=float)
    return np.maximum(-np.log(arr + EPSILON), 0.0)


def minmax(s: pd.Series) -> list[float | None]:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return [None, None]
    return [float(vals.min()), float(vals.max())]


def summarize_surface(surface: pd.DataFrame, prey_var: str, meta: dict) -> dict:
    delta = surface["delta_R_log_ice_beta1"]
    return {
        "target_year": int(surface["target_year"].iloc[0]),
        "prey_variable": prey_var,
        "prey_variable_label": meta["label"],
        "prey_variable_role": meta["role"],
        "window_start": int(surface["window_start"].iloc[0]),
        "window_end": int(surface["window_end"].iloc[0]),
        "grid_cell_count": int(len(surface)),
        "mean_U_prey": float(surface["U_prey_global"].mean()),
        "median_U_prey": float(surface["U_prey_global"].median()),
        "mean_A_ice": float(surface["A_ice_v1_MayJul"].mean()),
        "median_A_ice": float(surface["A_ice_v1_MayJul"].median()),
        "mean_U_prey_ice": float(surface["U_prey_ice_beta1"].mean()),
        "median_U_prey_ice": float(surface["U_prey_ice_beta1"].median()),
        "mean_R_log_prey": float(surface["R_log_prey"].mean()),
        "mean_R_log_prey_ice": float(surface["R_log_prey_ice_beta1"].mean()),
        "mean_delta_R_log_ice": float(delta.mean()),
        "median_delta_R_log_ice": float(delta.median()),
        "mean_absolute_utility_loss_ice": float(surface["absolute_utility_loss_ice_beta1"].mean()),
        "mean_relative_utility_loss_ice": float(surface["relative_utility_loss_ice_beta1"].mean()),
        "cells_Aice_lt_0_75": int((surface["A_ice_v1_MayJul"] < 0.75).sum()),
        "cells_Aice_lt_0_50": int((surface["A_ice_v1_MayJul"] < 0.50).sum()),
        "cells_Aice_lt_0_25": int((surface["A_ice_v1_MayJul"] < 0.25).sum()),
        "cells_delta_R_log_ice_gt_0_25": int((delta > 0.25).sum()),
        "cells_delta_R_log_ice_gt_0_50": int((delta > 0.50).sum()),
        "cells_delta_R_log_ice_gt_1": int((delta > 1.0).sum()),
        "cells_U_prey_ice_below_0_05": int((surface["U_prey_ice_beta1"] < 0.05).sum()),
        "cells_U_prey_ice_below_0_10": int((surface["U_prey_ice_beta1"] < 0.10).sum()),
        "mean_nearest_observed_distance_km": float(surface["nearest_observed_distance_km"].mean()),
    }


def build_loss_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "target_year": row["target_year"],
            "prey_variable": row["prey_variable"],
            "mean_delta_R_log_ice_beta1": row["mean_delta_R_log_ice"],
            "median_delta_R_log_ice_beta1": row["median_delta_R_log_ice"],
            "mean_U_prey": row["mean_U_prey"],
            "mean_U_prey_ice": row["mean_U_prey_ice"],
            "mean_absolute_utility_loss_ice": row["mean_absolute_utility_loss_ice"],
            "mean_relative_utility_loss_ice": row["mean_relative_utility_loss_ice"],
            "percent_cells_Aice_lt_0_50": row["cells_Aice_lt_0_50"] / row["grid_cell_count"] * 100,
            "percent_cells_delta_R_log_ice_gt_0_50": row["cells_delta_R_log_ice_gt_0_50"] / row["grid_cell_count"] * 100,
            "percent_cells_delta_R_log_ice_gt_1": row["cells_delta_R_log_ice_gt_1"] / row["grid_cell_count"] * 100,
        })
    return pd.DataFrame(rows)


def build_variable_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    main = summary[summary["prey_variable"] == "total_kJm2_frozen"][
        ["target_year", "mean_U_prey_ice", "mean_delta_R_log_ice", "cells_delta_R_log_ice_gt_0_50"]
    ].rename(columns={
        "mean_U_prey_ice": "main_kJ_mean_U_prey_ice",
        "mean_delta_R_log_ice": "main_kJ_mean_delta_R_log_ice",
        "cells_delta_R_log_ice_gt_0_50": "main_kJ_cells_delta_R_log_ice_gt_0_50",
    })
    rows = []
    for _, row in summary.merge(main, on="target_year", how="left").iterrows():
        rows.append({
            "target_year": row["target_year"],
            "prey_variable": row["prey_variable"],
            "mean_U_prey_ice": row["mean_U_prey_ice"],
            "mean_delta_R_log_ice": row["mean_delta_R_log_ice"],
            "mean_U_prey_ice_difference_vs_kJ": row["mean_U_prey_ice"] - row["main_kJ_mean_U_prey_ice"],
            "mean_delta_R_log_ice_difference_vs_kJ": row["mean_delta_R_log_ice"] - row["main_kJ_mean_delta_R_log_ice"],
            "cells_delta_R_log_ice_gt_0_50": row["cells_delta_R_log_ice_gt_0_50"],
            "cells_delta_R_log_ice_gt_0_50_difference_vs_kJ": row["cells_delta_R_log_ice_gt_0_50"] - row["main_kJ_cells_delta_R_log_ice_gt_0_50"],
        })
    return pd.DataFrame(rows)


def make_figures(annual_surfaces: dict[str, dict[int, pd.DataFrame]], summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    for prey_var, yearly in annual_surfaces.items():
        meta = PREY_VARIABLES[prey_var]
        plot_panel(yearly, "U_prey_ice_beta1", f"benthic prey + ice utility ({meta['label']})", FIG_DIR / f"prey_ice_brian_{meta['short']}_C3_utility_all_years_v2.png", 0, 1, "viridis")
        plot_panel(yearly, "delta_R_log_ice_beta1", f"Ice-added resistance ({meta['label']})", FIG_DIR / f"prey_ice_brian_{meta['short']}_C3_delta_R_ice_all_years_v2.png", 0, 3, "magma")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for prey_var, meta in PREY_VARIABLES.items():
        s = summary[summary["prey_variable"] == prey_var].sort_values("target_year")
        axes[0].plot(s["target_year"], s["mean_U_prey"], marker="o", label=meta["label"])
        axes[1].plot(s["target_year"], s["mean_U_prey_ice"], marker="o", label=meta["label"])
        axes[2].plot(s["target_year"], s["mean_delta_R_log_ice"], marker="o", label=meta["label"])
    axes[0].set_title("Prey utility")
    axes[1].set_title("Prey + ice utility")
    axes[2].set_title("Ice-added resistance")
    for ax in axes:
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean utility")
    axes[1].set_ylabel("Mean utility")
    axes[2].set_ylabel("Mean delta R_log")
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "prey_ice_brian_C3_temporal_summary_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prey_var, meta in PREY_VARIABLES.items():
        s = summary[summary["prey_variable"] == prey_var].sort_values("target_year")
        ax.plot(s["target_year"], s["cells_delta_R_log_ice_gt_0_50"], marker="o", label=meta["label"])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Cells with ice-added delta R_log > 0.50")
    ax.set_title("C3 cells with substantial ice penalty")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "prey_ice_brian_C3_ice_penalty_cell_counts_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    comp = comparison[comparison["prey_variable"] != "total_kJm2_frozen"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for prey_var, meta in PREY_VARIABLES.items():
        if prey_var == "total_kJm2_frozen":
            continue
        s = comp[comp["prey_variable"] == prey_var].sort_values("target_year")
        axes[0].plot(s["target_year"], s["mean_U_prey_ice_difference_vs_kJ"], marker="o", label=meta["label"])
        axes[1].plot(s["target_year"], s["mean_delta_R_log_ice_difference_vs_kJ"], marker="o", label=meta["label"])
    axes[0].axhline(0, color="0.35", lw=1)
    axes[1].axhline(0, color="0.35", lw=1)
    axes[0].set_title("Utility difference vs kJ")
    axes[1].set_title("Delta R difference vs kJ")
    for ax in axes:
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "prey_ice_brian_C3_variable_comparison_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_panel(yearly: dict[int, pd.DataFrame], col: str, title: str, path: Path, vmin: float, vmax: float, cmap: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, year in zip(axes.ravel(), TARGET_YEARS):
        df = yearly[year]
        last = ax.scatter(df["lon"], df["lat"], c=df[col], s=7, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(str(year), fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, lw=0.5)
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label(col)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_docs(summary: pd.DataFrame, loss: pd.DataFrame, comparison: pd.DataFrame, qa: dict) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_registry_manifest_readme_claims() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_once(path: Path, marker: str, text: str) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_manifest(paths: list[Path]) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


if __name__ == "__main__":
    main()
