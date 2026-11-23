"""Phase 8E.1 benthic amphipod C3 prey + ice + ship utility surfaces.

This phase combines Phase 8D.1 benthic C3 prey+ice utility surfaces with the
existing May-July all-vessel ship disturbance modifier for 2015-2019. It creates
utility surfaces and diagnostics only. It does not run LCP models and does not
modify previous prey, ice, ship, LCP, or validation outputs.
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


TARGET_YEARS = list(range(2015, 2020))
PREY_VARIABLES = {
    "total_kJm2_frozen": {"short": "kJm2_frozen", "label": "kJ m-2 frozen", "role": "main_candidate"},
    "total_gww": {"short": "gww", "label": "g wet weight", "role": "sensitivity_candidate"},
    "total_gC": {"short": "gC", "label": "g C biomass", "role": "sensitivity_candidate"},
}
EPSILON = 0.001
SHIP_D_COL = "D_ship_alpha0_5_MayJul_mean"
SHIP_S_COL = "S_scaled_ship_primary_MayJul_mean"

SURFACES = ROOT / "outputs/surfaces_v2"
SHIP = ROOT / "outputs/ship"
FIG = ROOT / "outputs/figures/surfaces_v2"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
MANIFEST = ROOT / "outputs/results_manifest.csv"
REGISTRY = ROOT / "docs/results_registry.md"
README = ROOT / "README.md"
CLAIMS = ROOT / "docs/paper_claims_evidence_matrix.md"
PHASE_MEMO = ROOT / "docs/phase_memos/phase8e1_brian_C3_prey_ice_ship_utility_surfaces.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase8e1_brian_C3_prey_ice_ship_utility_surfaces_paper_use.md"


def main() -> None:
    SURFACES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    c3 = pd.read_csv(GRID)
    c3["grid_id"] = c3["grid_id"].astype(str)
    c3 = c3[(c3["ocean_flag"] == 1) & (c3["inside_C3_domain"].astype(bool))].copy()

    annual: dict[str, dict[int, pd.DataFrame]] = {v: {} for v in PREY_VARIABLES}
    summary_rows: list[dict] = []
    qa = {
        "phase": "Phase 8E.1",
        "target_years": TARGET_YEARS,
        "domain": "C3 recurring-support v2 grid",
        "ship_modifier": SHIP_D_COL,
        "ship_intensity": SHIP_S_COL,
        "ship_interpretation": "all-vessel AIS-derived disturbance / movement impedance proxy, not ship-strike mortality",
        "formula": "U_prey_ice_ship = U_prey_ice_beta1 * D_ship_alpha0_5_MayJul_mean",
        "resistance_formula": "R_log = max(-log(U + 0.001), 0)",
        "c3_grid_cell_count": int(len(c3)),
        "merge_counts": {},
        "missing_ship_grid_id_counts": {},
        "min_max": {},
        "warnings": [
            "This phase creates utility surfaces only; no LCP models are run.",
            "Ship modifier represents disturbance / movement impedance proxy, not collision or fatality.",
            "Main model uses all-vessel exposure alpha0.5 only; no vessel-type weighting is imposed.",
            "Target years are 2015-2019 because ship data begin in 2015.",
        ],
    }

    for prey_var, meta in PREY_VARIABLES.items():
        for year in TARGET_YEARS:
            surface, merge_info = build_surface(prey_var, meta, year, c3)
            out = SURFACES / f"prey_ice_ship_brian_{meta['short']}_C3_utility_target{year}_v2.csv"
            surface.to_csv(out, index=False)
            annual[prey_var][year] = surface
            summary_rows.append(summarize(surface, prey_var, meta))
            qa["merge_counts"][f"{prey_var}_{year}"] = merge_info
            qa["missing_ship_grid_id_counts"][f"{prey_var}_{year}"] = merge_info["c3_cells_missing_ship"]
            qa["min_max"][f"{prey_var}_{year}"] = {
                "U_prey_ice": minmax(surface["U_prey_ice_beta1"]),
                "D_ship": minmax(surface["D_ship_alpha0_5_MayJul_mean"]),
                "U_prey_ice_ship": minmax(surface["U_prey_ice_ship_alpha0_5"]),
                "delta_R_log_ship": minmax(surface["delta_R_log_ship_alpha0_5"]),
            }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SURFACES / "prey_ice_ship_brian_C3_utility_summary_v2.csv", index=False)
    loss = build_loss_summary(summary)
    loss.to_csv(SURFACES / "prey_ice_ship_brian_C3_utility_loss_by_year_v2.csv", index=False)
    comparison = build_variable_comparison(summary)
    comparison.to_csv(SURFACES / "prey_ice_ship_brian_C3_variable_comparison_v2.csv", index=False)

    with (SURFACES / "prey_ice_ship_brian_C3_utility_qa_v2.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)
        f.write("\n")

    make_figures(annual, summary, comparison)
    write_docs(summary, loss, comparison, qa)
    update_docs_and_manifest()

    print(json.dumps({
        "phase": "Phase 8E.1",
        "summary": str(SURFACES / "prey_ice_ship_brian_C3_utility_summary_v2.csv"),
        "qa": str(SURFACES / "prey_ice_ship_brian_C3_utility_qa_v2.json"),
        "no_lcp_models_run": True,
        "ship_modifier": SHIP_D_COL,
    }, indent=2))
    print(summary[summary["prey_variable"] == "total_kJm2_frozen"][
        ["target_year", "mean_D_ship", "mean_U_prey_ice", "mean_U_prey_ice_ship", "mean_delta_R_log_ship", "cells_S_scaled_gt_0_50"]
    ].round(4).to_string(index=False))


def prey_ice_path(prey_var: str, meta: dict, year: int) -> Path:
    return SURFACES / f"prey_ice_brian_{meta['short']}_C3_utility_target{year}_v2.csv"


def ship_path(year: int) -> Path:
    return SHIP / f"ship_intensity_MayJul_target{year}.csv"


def build_surface(prey_var: str, meta: dict, year: int, c3: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prey_ice = pd.read_csv(prey_ice_path(prey_var, meta, year))
    ship = pd.read_csv(ship_path(year))
    prey_ice["grid_id"] = prey_ice["grid_id"].astype(str)
    ship["grid_id"] = ship["grid_id"].astype(str)
    base = c3[["grid_id", "grid_x", "grid_y", "x", "y", "lon", "lat", "ocean_flag", "inside_C3_domain"]].copy()
    merged = base.merge(prey_ice, on="grid_id", how="left", suffixes=("", "_preyice"), indicator="prey_ice_merge")
    merged = merged.merge(
        ship[[
            "grid_id", "S_ship_all_MayJul_mean", "S_ship_all_MayJul_sum", SHIP_S_COL,
            SHIP_D_COL, "D_ship_alpha0_7_MayJul_mean", "D_ship_exp_lambda1_MayJul_mean",
            "month_count",
        ]],
        on="grid_id",
        how="left",
        indicator="ship_merge",
    )
    merge_info = {
        "c3_grid_cells": int(len(base)),
        "prey_ice_rows": int(len(prey_ice)),
        "ship_rows": int(len(ship)),
        "merged_rows": int(len(merged)),
        "c3_cells_missing_prey_ice": int((merged["prey_ice_merge"] == "left_only").sum()),
        "c3_cells_missing_ship": int((merged["ship_merge"] == "left_only").sum()),
    }
    if merge_info["c3_cells_missing_prey_ice"] or merge_info["c3_cells_missing_ship"]:
        raise RuntimeError(f"Missing prey+ice or ship coverage for {prey_var} {year}: {merge_info}")

    u_prey_ice = pd.to_numeric(merged["U_prey_ice_beta1"], errors="coerce").fillna(EPSILON).clip(EPSILON, 1.0)
    d_ship = pd.to_numeric(merged[SHIP_D_COL], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    u_multi = (u_prey_ice * d_ship).clip(EPSILON, 1.0)
    r_multi = resistance(u_multi)
    r_prey_ice = pd.to_numeric(merged["R_log_prey_ice_beta1"], errors="coerce")
    r_prey_ice = r_prey_ice.fillna(pd.Series(resistance(u_prey_ice), index=merged.index)).clip(lower=0)

    out = pd.DataFrame({
        "grid_id": merged["grid_id"],
        "grid_x": merged["grid_x"],
        "grid_y": merged["grid_y"],
        "x": merged["x"],
        "y": merged["y"],
        "lon": merged["lon"],
        "lat": merged["lat"],
        "target_year": year,
        "prey_variable": prey_var,
        "prey_variable_label": meta["label"],
        "prey_variable_role": meta["role"],
        "U_prey_global": pd.to_numeric(merged["U_prey_global"], errors="coerce"),
        "A_ice_v1_MayJul": pd.to_numeric(merged["A_ice_v1_MayJul"], errors="coerce"),
        "U_prey_ice_beta1": u_prey_ice,
        "R_log_prey_ice_beta1": r_prey_ice,
        "S_ship_all_MayJul_mean": pd.to_numeric(merged["S_ship_all_MayJul_mean"], errors="coerce"),
        "S_ship_all_MayJul_sum": pd.to_numeric(merged["S_ship_all_MayJul_sum"], errors="coerce"),
        "S_scaled_ship_primary_MayJul_mean": pd.to_numeric(merged[SHIP_S_COL], errors="coerce").fillna(0.0).clip(0, 1),
        "D_ship_alpha0_5_MayJul_mean": d_ship,
        "D_ship_alpha0_7_MayJul_mean": pd.to_numeric(merged["D_ship_alpha0_7_MayJul_mean"], errors="coerce"),
        "D_ship_exp_lambda1_MayJul_mean": pd.to_numeric(merged["D_ship_exp_lambda1_MayJul_mean"], errors="coerce"),
        "U_prey_ice_ship_alpha0_5": u_multi,
        "R_log_prey_ice_ship_alpha0_5": r_multi,
        "delta_R_log_ship_alpha0_5": r_multi - r_prey_ice,
        "absolute_utility_loss_ship_alpha0_5": u_prey_ice - u_multi,
        "relative_utility_loss_ship_alpha0_5": np.where(u_prey_ice > 0, (u_prey_ice - u_multi) / u_prey_ice, np.nan),
        "month_count_ship": pd.to_numeric(merged["month_count"], errors="coerce"),
        "ocean_flag": merged["ocean_flag"],
        "inside_C3_domain": merged["inside_C3_domain"].astype(bool),
    })
    return out, merge_info


def resistance(u) -> np.ndarray:
    arr = np.asarray(u, dtype=float)
    return np.maximum(-np.log(arr + EPSILON), 0.0)


def minmax(s: pd.Series) -> list[float | None]:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return [None, None]
    return [float(vals.min()), float(vals.max())]


def summarize(surface: pd.DataFrame, prey_var: str, meta: dict) -> dict:
    delta = surface["delta_R_log_ship_alpha0_5"]
    return {
        "target_year": int(surface["target_year"].iloc[0]),
        "prey_variable": prey_var,
        "prey_variable_label": meta["label"],
        "prey_variable_role": meta["role"],
        "grid_cell_count": int(len(surface)),
        "mean_U_prey": float(surface["U_prey_global"].mean()),
        "mean_A_ice": float(surface["A_ice_v1_MayJul"].mean()),
        "mean_U_prey_ice": float(surface["U_prey_ice_beta1"].mean()),
        "mean_S_scaled_ship": float(surface["S_scaled_ship_primary_MayJul_mean"].mean()),
        "median_S_scaled_ship": float(surface["S_scaled_ship_primary_MayJul_mean"].median()),
        "mean_D_ship": float(surface["D_ship_alpha0_5_MayJul_mean"].mean()),
        "median_D_ship": float(surface["D_ship_alpha0_5_MayJul_mean"].median()),
        "mean_U_prey_ice_ship": float(surface["U_prey_ice_ship_alpha0_5"].mean()),
        "median_U_prey_ice_ship": float(surface["U_prey_ice_ship_alpha0_5"].median()),
        "mean_R_log_prey_ice": float(surface["R_log_prey_ice_beta1"].mean()),
        "mean_R_log_prey_ice_ship": float(surface["R_log_prey_ice_ship_alpha0_5"].mean()),
        "mean_delta_R_log_ship": float(delta.mean()),
        "median_delta_R_log_ship": float(delta.median()),
        "mean_absolute_utility_loss_ship": float(surface["absolute_utility_loss_ship_alpha0_5"].mean()),
        "mean_relative_utility_loss_ship": float(surface["relative_utility_loss_ship_alpha0_5"].mean()),
        "cells_S_scaled_gt_0_25": int((surface["S_scaled_ship_primary_MayJul_mean"] > 0.25).sum()),
        "cells_S_scaled_gt_0_50": int((surface["S_scaled_ship_primary_MayJul_mean"] > 0.50).sum()),
        "cells_S_scaled_gt_0_75": int((surface["S_scaled_ship_primary_MayJul_mean"] > 0.75).sum()),
        "cells_delta_R_log_ship_gt_0_25": int((delta > 0.25).sum()),
        "cells_delta_R_log_ship_gt_0_50": int((delta > 0.50).sum()),
        "cells_delta_R_log_ship_gt_1": int((delta > 1.0).sum()),
        "cells_U_prey_ice_ship_below_0_05": int((surface["U_prey_ice_ship_alpha0_5"] < 0.05).sum()),
        "cells_U_prey_ice_ship_below_0_10": int((surface["U_prey_ice_ship_alpha0_5"] < 0.10).sum()),
    }


def build_loss_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[[
        "target_year", "prey_variable", "mean_delta_R_log_ship", "median_delta_R_log_ship",
        "mean_U_prey_ice", "mean_U_prey_ice_ship", "mean_absolute_utility_loss_ship",
        "mean_relative_utility_loss_ship", "cells_S_scaled_gt_0_50",
        "cells_delta_R_log_ship_gt_0_50", "cells_delta_R_log_ship_gt_1",
    ]].copy()


def build_variable_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    main = summary[summary["prey_variable"] == "total_kJm2_frozen"][
        ["target_year", "mean_U_prey_ice_ship", "mean_delta_R_log_ship", "cells_delta_R_log_ship_gt_0_50"]
    ].rename(columns={
        "mean_U_prey_ice_ship": "main_kJ_mean_U_multi",
        "mean_delta_R_log_ship": "main_kJ_mean_delta_R_ship",
        "cells_delta_R_log_ship_gt_0_50": "main_kJ_cells_delta_R_gt_0_50",
    })
    out = summary.merge(main, on="target_year", how="left")
    out["mean_U_multi_difference_vs_kJ"] = out["mean_U_prey_ice_ship"] - out["main_kJ_mean_U_multi"]
    out["mean_delta_R_ship_difference_vs_kJ"] = out["mean_delta_R_log_ship"] - out["main_kJ_mean_delta_R_ship"]
    out["cells_delta_R_gt_0_50_difference_vs_kJ"] = out["cells_delta_R_log_ship_gt_0_50"] - out["main_kJ_cells_delta_R_gt_0_50"]
    return out


def make_figures(annual: dict[str, dict[int, pd.DataFrame]], summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    for prey_var, yearly in annual.items():
        meta = PREY_VARIABLES[prey_var]
        plot_panel(yearly, "U_prey_ice_ship_alpha0_5", f"benthic prey+ice+ship utility ({meta['label']})", FIG / f"prey_ice_ship_brian_{meta['short']}_C3_utility_all_years_v2.png", 0, 1, "viridis")
        plot_panel(yearly, "delta_R_log_ship_alpha0_5", f"Ship-added resistance ({meta['label']})", FIG / f"prey_ice_ship_brian_{meta['short']}_C3_delta_R_ship_all_years_v2.png", 0, 0.7, "magma")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for prey_var, meta in PREY_VARIABLES.items():
        s = summary[summary["prey_variable"] == prey_var].sort_values("target_year")
        axes[0].plot(s["target_year"], s["mean_U_prey_ice"], marker="o", label=meta["label"])
        axes[1].plot(s["target_year"], s["mean_U_prey_ice_ship"], marker="o", label=meta["label"])
        axes[2].plot(s["target_year"], s["mean_delta_R_log_ship"], marker="o", label=meta["label"])
    axes[0].set_title("Prey + ice utility")
    axes[1].set_title("Prey + ice + ship utility")
    axes[2].set_title("Ship-added resistance")
    for ax in axes:
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean utility")
    axes[1].set_ylabel("Mean utility")
    axes[2].set_ylabel("Mean delta R_log")
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "prey_ice_ship_brian_C3_temporal_summary_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prey_var, meta in PREY_VARIABLES.items():
        s = summary[summary["prey_variable"] == prey_var].sort_values("target_year")
        ax.plot(s["target_year"], s["cells_delta_R_log_ship_gt_0_50"], marker="o", label=meta["label"])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Cells with ship-added delta R_log > 0.50")
    ax.set_title("C3 cells with substantial ship penalty")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG / "prey_ice_ship_brian_C3_ship_penalty_cell_counts_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ship_ref = annual["total_kJm2_frozen"]
    plot_panel(ship_ref, "S_scaled_ship_primary_MayJul_mean", "Scaled May-July all-vessel ship exposure on C3", FIG / "ship_scaled_MayJul_C3_2015_2019_v2.png", 0, 1, "inferno")


def plot_panel(yearly: dict[int, pd.DataFrame], col: str, title: str, path: Path, vmin: float, vmax: float, cmap: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for ax, year in zip(axes.ravel(), TARGET_YEARS):
        df = yearly[year]
        scatter = ax.scatter(df["lon"], df["lat"], c=df[col], s=7, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(str(year), fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, lw=0.5)
        ax.tick_params(labelsize=7)
    axes.ravel()[-1].axis("off")
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label(col)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_docs(summary: pd.DataFrame, loss: pd.DataFrame, comparison: pd.DataFrame, qa: dict) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_docs_and_manifest() -> None:
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
