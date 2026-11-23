"""Phase 4B.1 QA patch for missing SIC and open-water treatment.

This script diagnoses missing sampled sea-ice concentration values from the
Phase 4B v0 outputs, then creates v1 ice-only surfaces where in-domain NaN SIC
values are interpreted as open water. It does not build prey+ice utility
surfaces and does not run LCP.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_ice_accessibility_surfaces_v0 import (
    A_MIN,
    FIGURES_DIR,
    ICE_DIR,
    MONTH_LABELS,
    MONTHS,
    OCEAN_GRID_PATH,
    PREY_QUALITY_OVERLAP_PATH,
    PREY_QUALITY_PATH,
    BOTTLENECK_PATH,
    SAMPLING_METHOD,
    SIC_FILE_NAME,
    SIC_PATH,
    TARGET_YEARS,
    accessibility_from_sic,
    locate_sic_file,
    read_ocean_grid,
    read_sic_dataset,
    sample_nearest,
)

MEMO_PATH = PROJECT_ROOT / "docs/phase_memos/phase4b_ice_accessibility_surfaces.md"
REGISTRY_PATH = PROJECT_ROOT / "docs/results_registry.md"
README_PATH = PROJECT_ROOT / "README.md"
MANIFEST_PATH = PROJECT_ROOT / "outputs/results_manifest.csv"


def main() -> None:
    ICE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sic_path = locate_sic_file()
    reader_info = read_sic_dataset(sic_path)
    ocean = read_ocean_grid()
    source_bounds = {
        "lon_min": float(np.nanmin(reader_info["longitude"])),
        "lon_max": float(np.nanmax(reader_info["longitude"])),
        "lat_min": float(np.nanmin(reader_info["latitude"])),
        "lat_max": float(np.nanmax(reader_info["latitude"])),
    }
    in_bounds = (
        ocean["lon"].between(source_bounds["lon_min"], source_bounds["lon_max"])
        & ocean["lat"].between(source_bounds["lat_min"], source_bounds["lat_max"])
    )

    missing_diag = build_missing_diagnostics(source_bounds)
    missing_diag.to_csv(ICE_DIR / "ice_sic_missing_diagnostics_v0.csv", index=False)
    plot_missing_panels()

    ocean_variable_support = diagnose_ocean_variable_support(sic_path, reader_info, ocean)
    monthly_v1 = build_monthly_v1(reader_info, ocean, in_bounds, sic_path)
    monthly_summary_v1 = summarize_monthly_v1(monthly_v1)
    monthly_summary_v1.to_csv(ICE_DIR / "ice_accessibility_monthly_summary_v1.csv", index=False)

    annual_surfaces = []
    for year in TARGET_YEARS:
        annual = build_annual_v1(year, monthly_v1)
        annual.to_csv(ICE_DIR / f"ice_accessibility_v1_MayJul_target{year}.csv", index=False)
        annual_surfaces.append(annual)
    annual_all = pd.concat(annual_surfaces, ignore_index=True)
    annual_summary_v1 = summarize_annual_v1(annual_all)
    annual_summary_v1.to_csv(ICE_DIR / "ice_accessibility_summary_v1.csv", index=False)

    comparison = compare_v0_v1(monthly_summary_v1, annual_summary_v1)
    comparison.to_csv(ICE_DIR / "ice_accessibility_v0_vs_v1_comparison.csv", index=False)

    persistence = build_persistence_v1(annual_all)
    persistence.to_csv(ICE_DIR / "ice_accessibility_persistence_v1.csv", index=False)
    monthly_persistence = build_monthly_persistence_v1(monthly_v1)
    monthly_persistence.to_csv(ICE_DIR / "ice_accessibility_monthly_persistence_v1.csv", index=False)

    overlap, overlap_summary, monthly_overlap_summary = build_overlap_outputs_v1(persistence, monthly_persistence)
    overlap.to_csv(ICE_DIR / "ice_x_prey_quality_bottleneck_overlap_v1.csv", index=False)
    overlap_summary.to_csv(ICE_DIR / "ice_x_prey_quality_bottleneck_overlap_summary_v1.csv", index=False)
    monthly_overlap_summary.to_csv(ICE_DIR / "ice_monthly_x_prey_quality_bottleneck_overlap_summary_v1.csv", index=False)

    plot_v1_panels(monthly_v1, annual_surfaces)
    plot_comparison(comparison)

    qa = {
        "source_sic_file": str(sic_path),
        "reader_used": reader_info["reader_used"],
        "sampling_method": SAMPLING_METHOD,
        "open_water_treatment": "In-source-bounds NaN siconc values on ocean grid are interpreted as open water: SIC_filled=0, A_ice_filled=1.",
        "source_bounds": source_bounds,
        "ocean_cells_in_source_bounds": int(in_bounds.sum()),
        "ocean_cell_count": int(len(ocean)),
        "missing_diagnostics": missing_diag.to_dict(orient="records"),
        "ocean_variable_support_for_missing_siconc": ocean_variable_support,
        "warnings": [
            "v1 changes only ice-only Phase 4B surfaces and diagnostics; no prey+ice utility or LCP was run.",
            "Open-water interpretation is supported by in-bounds coverage and ocean-variable support, but remains a modeling assumption to review before Phase 4C.",
        ],
    }
    with (ICE_DIR / "ice_accessibility_qa_v1.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)
        f.write("\n")

    update_phase_memo(missing_diag, ocean_variable_support, annual_summary_v1, comparison, overlap_summary)
    update_registry_and_readme()
    update_manifest()

    print(json.dumps({
        "missing_diagnostics": missing_diag.to_dict(orient="records"),
        "ocean_variable_support": ocean_variable_support,
        "annual_summary_v1": annual_summary_v1.to_dict(orient="records"),
        "overlap_summary_v1": overlap_summary.to_dict(orient="records"),
    }, indent=2))


def build_missing_diagnostics(source_bounds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for year in TARGET_YEARS:
        for month in MONTHS:
            path = ICE_DIR / f"ice_accessibility_target{year}_month{month:02d}.csv"
            df = pd.read_csv(path)
            missing = df["SIC"].isna()
            notes = []
            if df["lon"].between(source_bounds["lon_min"], source_bounds["lon_max"]).all() and df["lat"].between(source_bounds["lat_min"], source_bounds["lat_max"]).all():
                notes.append("all_ocean_grid_centroids_within_source_bounds")
            if month == 7:
                notes.append("July_lower_ice_month")
            if missing.any():
                notes.append(f"missing_mean_lat={df.loc[missing, 'lat'].mean():.3f}")
                notes.append(f"missing_mean_lon={df.loc[missing, 'lon'].mean():.3f}")
            rows.append({
                "target_year": year,
                "month": month,
                "total_ocean_cells": int(len(df)),
                "missing_SIC_cells": int(missing.sum()),
                "missing_SIC_percent": float(missing.mean() * 100),
                "nonmissing_SIC_cells": int((~missing).sum()),
                "mean_SIC_nonmissing": none_if_nan(df["SIC"].mean()),
                "mean_Aice_nonmissing": none_if_nan(df["A_ice"].mean()),
                "notes": "; ".join(notes),
            })
    return pd.DataFrame(rows)


def plot_missing_panels() -> None:
    for month in MONTHS:
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
        for ax, year in zip(axes.ravel(), TARGET_YEARS):
            df = pd.read_csv(ICE_DIR / f"ice_accessibility_target{year}_month{month:02d}.csv")
            missing = df["SIC"].isna().astype(int)
            ax.scatter(df["lon"], df["lat"], c=missing, s=4, cmap="coolwarm", vmin=0, vmax=1, linewidth=0)
            ax.set_title(str(year), fontsize=10)
        fig.suptitle(f"Missing SIC cells: {MONTH_LABELS[month]}")
        fig.savefig(FIGURES_DIR / f"ice_missing_SIC_month{month:02d}_all_targets_v0.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def diagnose_ocean_variable_support(sic_path: Path, reader_info: dict, ocean: pd.DataFrame) -> dict:
    lon_idx = np.abs(reader_info["longitude"][None, :] - ocean["lon"].to_numpy()[:, None]).argmin(axis=1)
    lat_idx = np.abs(reader_info["latitude"][None, :] - ocean["lat"].to_numpy()[:, None]).argmin(axis=1)
    dates = None
    results = {}
    with h5py.File(sic_path, "r") as f:
        dates = pd.to_datetime(f["time"][()], unit="s", origin="unix", utc=True)
        for year in TARGET_YEARS:
            for month in MONTHS:
                idx = np.where((dates.year == year) & (dates.month == month))[0][0]
                siconc = f["siconc"][idx, :, :].astype(float)[lat_idx, lon_idx]
                missing_sic = ~np.isfinite(siconc)
                entry = {"missing_siconc_cells": int(missing_sic.sum())}
                for var in ["thetao", "so", "mlotst", "zos"]:
                    if var not in f:
                        continue
                    data = f[var]
                    if data.ndim == 4:
                        sampled = data[idx, 0, :, :].astype(float)[lat_idx, lon_idx]
                    else:
                        sampled = data[idx, :, :].astype(float)[lat_idx, lon_idx]
                    present = np.isfinite(sampled) & missing_sic
                    entry[f"{var}_present_where_siconc_missing"] = int(present.sum())
                    entry[f"{var}_support_percent_of_missing_siconc"] = float(present.sum() / missing_sic.sum() * 100) if missing_sic.any() else 0.0
                results[f"{year}_{month:02d}"] = entry
    return results


def build_monthly_v1(reader_info: dict, ocean: pd.DataFrame, in_bounds: pd.Series, sic_path: Path) -> pd.DataFrame:
    surfaces = []
    for year in TARGET_YEARS:
        for month in MONTHS:
            raw_sampled = sample_nearest(
                lon_grid=reader_info["longitude"],
                lat_grid=reader_info["latitude"],
                values=reader_info["sic_by_year_month"][(year, month)],
                points_lon=ocean["lon"].to_numpy(dtype=float),
                points_lat=ocean["lat"].to_numpy(dtype=float),
            )
            sic_original = np.clip(raw_sampled, 0.0, 1.0)
            aice_original = accessibility_from_sic(sic_original)
            reason = np.where(np.isfinite(sic_original), "sampled", np.where(in_bounds.to_numpy(), "nan_interpreted_as_open_water", "outside_source_bounds"))
            sic_filled = sic_original.copy()
            sic_filled[reason == "nan_interpreted_as_open_water"] = 0.0
            aice_filled = accessibility_from_sic(sic_filled)
            surface = ocean.copy()
            surface["target_year"] = year
            surface["month"] = month
            surface["SIC_original"] = sic_original
            surface["SIC_filled"] = sic_filled
            surface["A_ice_original"] = aice_original
            surface["A_ice_filled"] = aice_filled
            surface["SIC_fill_reason"] = reason
            surface["low_access_Aice_lt_0_25"] = (surface["A_ice_filled"] < 0.25).astype(int)
            surface["moderate_access_Aice_lt_0_50"] = (surface["A_ice_filled"] < 0.50).astype(int)
            surface["high_SIC_gt_0_55"] = (surface["SIC_filled"] > 0.55).astype(int)
            surface["sampling_method"] = SAMPLING_METHOD
            surface["source_sic_file"] = str(sic_path)
            keep = [
                "grid_id", "x", "y", "lon", "lat", "ocean_flag", "target_year", "month",
                "SIC_original", "SIC_filled", "A_ice_original", "A_ice_filled", "SIC_fill_reason",
                "low_access_Aice_lt_0_25", "moderate_access_Aice_lt_0_50", "high_SIC_gt_0_55",
                "sampling_method", "source_sic_file",
            ]
            surface = surface[keep]
            surface.to_csv(ICE_DIR / f"ice_accessibility_v1_target{year}_month{month:02d}.csv", index=False)
            surfaces.append(surface)
    return pd.concat(surfaces, ignore_index=True)


def summarize_monthly_v1(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, month), group in monthly.groupby(["target_year", "month"]):
        n = len(group)
        fill = group["SIC_fill_reason"].eq("nan_interpreted_as_open_water")
        rows.append({
            "target_year": year,
            "month": month,
            "mean_SIC": group["SIC_filled"].mean(),
            "median_SIC": group["SIC_filled"].median(),
            "min_SIC": group["SIC_filled"].min(),
            "max_SIC": group["SIC_filled"].max(),
            "mean_A_ice": group["A_ice_filled"].mean(),
            "median_A_ice": group["A_ice_filled"].median(),
            "min_A_ice": group["A_ice_filled"].min(),
            "max_A_ice": group["A_ice_filled"].max(),
            "area_cells_Aice_lt_0_25": int(group["low_access_Aice_lt_0_25"].sum()),
            "area_cells_Aice_lt_0_50": int(group["moderate_access_Aice_lt_0_50"].sum()),
            "area_cells_SIC_gt_0_55": int(group["high_SIC_gt_0_55"].sum()),
            "percent_cells_Aice_lt_0_25": group["low_access_Aice_lt_0_25"].sum() / n * 100,
            "percent_cells_Aice_lt_0_50": group["moderate_access_Aice_lt_0_50"].sum() / n * 100,
            "percent_cells_SIC_gt_0_55": group["high_SIC_gt_0_55"].sum() / n * 100,
            "open_water_fill_cells": int(fill.sum()),
            "open_water_fill_percent": float(fill.mean() * 100),
        })
    return pd.DataFrame(rows).sort_values(["target_year", "month"])


def build_annual_v1(year: int, monthly: pd.DataFrame) -> pd.DataFrame:
    base = monthly[(monthly["target_year"] == year) & (monthly["month"] == 5)][["grid_id", "x", "y", "lon", "lat", "ocean_flag", "sampling_method", "source_sic_file"]].copy()
    parts = []
    fill_counts = []
    for month in MONTHS:
        name = MONTH_LABELS[month]
        part = monthly[(monthly["target_year"] == year) & (monthly["month"] == month)].copy()
        fill_counts.append(part["SIC_fill_reason"].eq("nan_interpreted_as_open_water").rename(f"fill_{name}"))
        parts.append(part[["grid_id", "SIC_filled", "A_ice_filled"]].rename(columns={"SIC_filled": f"SIC_{name}_filled", "A_ice_filled": f"A_ice_{name}_filled"}))
    annual = base.merge(parts[0], on="grid_id").merge(parts[1], on="grid_id").merge(parts[2], on="grid_id")
    annual["target_year"] = year
    annual["SIC_MayJul_mean_filled"] = annual[["SIC_May_filled", "SIC_June_filled", "SIC_July_filled"]].mean(axis=1)
    annual["A_ice_MayJul_mean_from_monthly_A_filled"] = annual[["A_ice_May_filled", "A_ice_June_filled", "A_ice_July_filled"]].mean(axis=1)
    annual["A_ice_from_MayJul_mean_SIC_filled"] = accessibility_from_sic(annual["SIC_MayJul_mean_filled"])
    year_monthly = monthly[monthly["target_year"] == year]
    fill_any = year_monthly.pivot(index="grid_id", columns="month", values="SIC_fill_reason").reindex(annual["grid_id"])
    fill_count_per_cell = fill_any.eq("nan_interpreted_as_open_water").sum(axis=1).to_numpy()
    annual["nan_open_water_fill_count"] = fill_count_per_cell
    annual["nan_open_water_fill_percent"] = fill_count_per_cell / len(MONTHS) * 100
    annual["low_access_Aice_lt_0_25"] = (annual["A_ice_MayJul_mean_from_monthly_A_filled"] < 0.25).astype(int)
    annual["moderate_access_Aice_lt_0_50"] = (annual["A_ice_MayJul_mean_from_monthly_A_filled"] < 0.50).astype(int)
    annual["high_SIC_gt_0_55"] = (annual["SIC_MayJul_mean_filled"] > 0.55).astype(int)
    keep = [
        "grid_id", "x", "y", "lon", "lat", "ocean_flag", "target_year",
        "SIC_May_filled", "SIC_June_filled", "SIC_July_filled", "SIC_MayJul_mean_filled",
        "A_ice_May_filled", "A_ice_June_filled", "A_ice_July_filled",
        "A_ice_MayJul_mean_from_monthly_A_filled", "A_ice_from_MayJul_mean_SIC_filled",
        "nan_open_water_fill_count", "nan_open_water_fill_percent",
        "low_access_Aice_lt_0_25", "moderate_access_Aice_lt_0_50", "high_SIC_gt_0_55",
        "sampling_method", "source_sic_file",
    ]
    return annual[keep]


def summarize_annual_v1(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in annual.groupby("target_year"):
        n = len(group)
        primary = group["A_ice_MayJul_mean_from_monthly_A_filled"]
        rows.append({
            "target_year": year,
            "mean_SIC_MayJul": group["SIC_MayJul_mean_filled"].mean(),
            "median_SIC_MayJul": group["SIC_MayJul_mean_filled"].median(),
            "min_SIC_MayJul": group["SIC_MayJul_mean_filled"].min(),
            "max_SIC_MayJul": group["SIC_MayJul_mean_filled"].max(),
            "mean_A_ice_primary": primary.mean(),
            "median_A_ice_primary": primary.median(),
            "min_A_ice_primary": primary.min(),
            "max_A_ice_primary": primary.max(),
            "mean_A_ice_from_mean_SIC": group["A_ice_from_MayJul_mean_SIC_filled"].mean(),
            "median_A_ice_from_mean_SIC": group["A_ice_from_MayJul_mean_SIC_filled"].median(),
            "area_cells_Aice_lt_0_25": int(group["low_access_Aice_lt_0_25"].sum()),
            "area_cells_Aice_lt_0_50": int(group["moderate_access_Aice_lt_0_50"].sum()),
            "area_cells_SIC_gt_0_55": int(group["high_SIC_gt_0_55"].sum()),
            "percent_cells_Aice_lt_0_25": group["low_access_Aice_lt_0_25"].sum() / n * 100,
            "percent_cells_Aice_lt_0_50": group["moderate_access_Aice_lt_0_50"].sum() / n * 100,
            "percent_cells_SIC_gt_0_55": group["high_SIC_gt_0_55"].sum() / n * 100,
            "open_water_fill_cells": int((group["nan_open_water_fill_count"] > 0).sum()),
            "open_water_fill_percent": float((group["nan_open_water_fill_count"] > 0).mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("target_year")


def compare_v0_v1(monthly_summary_v1: pd.DataFrame, annual_summary_v1: pd.DataFrame) -> pd.DataFrame:
    monthly_v0 = pd.read_csv(ICE_DIR / "ice_accessibility_monthly_summary_v0.csv")
    rows = []
    merged_m = monthly_v0.merge(monthly_summary_v1, on=["target_year", "month"], suffixes=("_v0", "_v1"))
    for _, row in merged_m.iterrows():
        rows.append({
            "scale": "monthly",
            "target_year": int(row["target_year"]),
            "month": int(row["month"]),
            "mean_SIC_v0": row["mean_SIC_v0"],
            "mean_SIC_v1": row["mean_SIC_v1"],
            "mean_Aice_v0": row["mean_A_ice_v0"],
            "mean_Aice_v1": row["mean_A_ice_v1"],
            "Aice_lt_0_50_cells_v0": int(row["area_cells_Aice_lt_0_50_v0"]),
            "Aice_lt_0_50_cells_v1": int(row["area_cells_Aice_lt_0_50_v1"]),
            "difference_mean_SIC": row["mean_SIC_v1"] - row["mean_SIC_v0"],
            "difference_mean_Aice": row["mean_A_ice_v1"] - row["mean_A_ice_v0"],
            "difference_low_access_cells": int(row["area_cells_Aice_lt_0_50_v1"] - row["area_cells_Aice_lt_0_50_v0"]),
        })
    annual_v0 = pd.read_csv(ICE_DIR / "ice_accessibility_summary_v0.csv")
    merged_a = annual_v0.merge(annual_summary_v1, on="target_year", suffixes=("_v0", "_v1"))
    for _, row in merged_a.iterrows():
        rows.append({
            "scale": "MayJul",
            "target_year": int(row["target_year"]),
            "month": np.nan,
            "mean_SIC_MayJul_v0": row["mean_SIC_MayJul_v0"],
            "mean_SIC_MayJul_v1": row["mean_SIC_MayJul_v1"],
            "mean_Aice_MayJul_v0": row["mean_A_ice_primary_v0"],
            "mean_Aice_MayJul_v1": row["mean_A_ice_primary_v1"],
            "Aice_lt_0_50_cells_v0": int(row["area_cells_Aice_lt_0_50_v0"]),
            "Aice_lt_0_50_cells_v1": int(row["area_cells_Aice_lt_0_50_v1"]),
            "difference_mean_SIC": row["mean_SIC_MayJul_v1"] - row["mean_SIC_MayJul_v0"],
            "difference_mean_Aice": row["mean_A_ice_primary_v1"] - row["mean_A_ice_primary_v0"],
            "difference_low_access_cells": int(row["area_cells_Aice_lt_0_50_v1"] - row["area_cells_Aice_lt_0_50_v0"]),
        })
    return pd.DataFrame(rows)


def build_persistence_v1(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in annual.groupby("grid_id", sort=False):
        first = group.iloc[0]
        years_low50 = int((group["A_ice_MayJul_mean_from_monthly_A_filled"] < 0.50).sum())
        cls = "persistent_low_access" if years_low50 >= 5 else ("intermittent_low_access" if years_low50 >= 1 else "rarely_or_never_low_access")
        rows.append({
            "grid_id": grid_id, "x": first["x"], "y": first["y"], "lon": first["lon"], "lat": first["lat"],
            "years_Aice_lt_0_25": int((group["A_ice_MayJul_mean_from_monthly_A_filled"] < 0.25).sum()),
            "years_Aice_lt_0_50": years_low50,
            "years_SIC_gt_0_55": int((group["SIC_MayJul_mean_filled"] > 0.55).sum()),
            "mean_SIC_MayJul": group["SIC_MayJul_mean_filled"].mean(),
            "mean_A_ice": group["A_ice_MayJul_mean_from_monthly_A_filled"].mean(),
            "min_A_ice": group["A_ice_MayJul_mean_from_monthly_A_filled"].min(),
            "max_A_ice": group["A_ice_MayJul_mean_from_monthly_A_filled"].max(),
            "ice_limitation_class": cls,
        })
    return pd.DataFrame(rows)


def build_monthly_persistence_v1(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grid_id, month), group in monthly.groupby(["grid_id", "month"], sort=False):
        first = group.iloc[0]
        years_low50 = int((group["A_ice_filled"] < 0.50).sum())
        cls = "persistent_low_access" if years_low50 >= 5 else ("intermittent_low_access" if years_low50 >= 1 else "rarely_or_never_low_access")
        rows.append({
            "grid_id": grid_id, "x": first["x"], "y": first["y"], "lon": first["lon"], "lat": first["lat"],
            "month": month,
            "years_Aice_lt_0_25": int((group["A_ice_filled"] < 0.25).sum()),
            "years_Aice_lt_0_50": years_low50,
            "years_SIC_gt_0_55": int((group["SIC_filled"] > 0.55).sum()),
            "mean_SIC": group["SIC_filled"].mean(),
            "mean_A_ice": group["A_ice_filled"].mean(),
            "min_A_ice": group["A_ice_filled"].min(),
            "max_A_ice": group["A_ice_filled"].max(),
            "ice_limitation_class": cls,
        })
    return pd.DataFrame(rows)


def build_overlap_outputs_v1(persistence: pd.DataFrame, monthly_persistence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlap = persistence.copy()
    if PREY_QUALITY_OVERLAP_PATH.exists():
        prey = pd.read_csv(PREY_QUALITY_OVERLAP_PATH)
        prey["grid_id"] = prey["grid_id"].astype(str)
        cols = [c for c in ["grid_id", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class", "bottleneck_years_in_top5pct"] if c in prey.columns]
        overlap = overlap.merge(prey[cols], on="grid_id", how="left")
    elif PREY_QUALITY_PATH.exists():
        prey = pd.read_csv(PREY_QUALITY_PATH)
        prey["grid_id"] = prey["grid_id"].astype(str)
        overlap = overlap.merge(prey[["grid_id", "prey_quality_class"]], on="grid_id", how="left")
    if BOTTLENECK_PATH.exists() and "bottleneck_years_in_top5pct" not in overlap.columns:
        bottleneck = pd.read_csv(BOTTLENECK_PATH)
        bottleneck["grid_id"] = bottleneck["grid_id"].astype(str)
        overlap = overlap.merge(bottleneck[["grid_id", "years_in_top5pct"]], on="grid_id", how="left")
        overlap["bottleneck_years_in_top5pct"] = overlap["years_in_top5pct"].fillna(0).astype(int)
        overlap = overlap.drop(columns=["years_in_top5pct"])
    if "bottleneck_years_in_top5pct" not in overlap.columns:
        overlap["bottleneck_years_in_top5pct"] = 0
    overlap["years_in_top5pct"] = overlap["bottleneck_years_in_top5pct"]
    if "bottleneck_persistence_class" not in overlap.columns:
        overlap["bottleneck_persistence_class"] = np.select(
            [overlap["bottleneck_years_in_top5pct"] >= 5, overlap["bottleneck_years_in_top5pct"].between(1, 4)],
            ["persistent_bottleneck", "intermittent_bottleneck"],
            default="non_bottleneck",
        )
    for col in ["prey_quality_class", "prey_bottleneck_class"]:
        if col not in overlap.columns:
            overlap[col] = "not_available"
        overlap[col] = overlap[col].fillna("not_available")
    overlap = overlap[[
        "grid_id", "lon", "lat", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class",
        "years_in_top5pct", "years_Aice_lt_0_25", "years_Aice_lt_0_50", "years_SIC_gt_0_55",
        "mean_A_ice", "mean_SIC_MayJul", "ice_limitation_class",
    ]]
    combo = overlap.groupby(["prey_bottleneck_class", "ice_limitation_class"], dropna=False).agg(
        cell_count=("grid_id", "count"),
        mean_A_ice=("mean_A_ice", "mean"),
        mean_SIC_MayJul=("mean_SIC_MayJul", "mean"),
    ).reset_index()
    special_rows = []
    masks = {
        "persistent_high_quality + persistent_low_access": (overlap["prey_quality_class"] == "persistent_high_quality") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_high_persistent_bottleneck + persistent_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_high_persistent_bottleneck + intermittent_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "intermittent_low_access"),
        "persistent_high_persistent_bottleneck + rarely_or_never_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "rarely_or_never_low_access"),
        "persistent_zero_proxy + persistent_low_access": (overlap["prey_quality_class"] == "persistent_zero_proxy") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_bottleneck + persistent_low_access": (overlap["bottleneck_persistence_class"] == "persistent_bottleneck") & (overlap["ice_limitation_class"] == "persistent_low_access"),
    }
    for label, mask in masks.items():
        part = overlap[mask]
        special_rows.append({"overlap_class": label, "cell_count": int(len(part)), "mean_A_ice": none_if_nan(part["mean_A_ice"].mean()), "mean_SIC_MayJul": none_if_nan(part["mean_SIC_MayJul"].mean())})
    combo["overlap_class"] = combo["prey_bottleneck_class"] + " + " + combo["ice_limitation_class"]
    summary = pd.concat([pd.DataFrame(special_rows), combo[["overlap_class", "cell_count", "mean_A_ice", "mean_SIC_MayJul"]]], ignore_index=True)

    monthly = monthly_persistence.merge(overlap[["grid_id", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class"]], on="grid_id", how="left")
    monthly_rows = []
    for month, group in monthly.groupby("month"):
        for label, mask in {
            "persistent_high_quality": group["prey_quality_class"] == "persistent_high_quality",
            "persistent_high_persistent_bottleneck": group["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck",
            "persistent_bottleneck": group["bottleneck_persistence_class"] == "persistent_bottleneck",
            "persistent_zero_proxy": group["prey_quality_class"] == "persistent_zero_proxy",
        }.items():
            for cls, part in group[mask].groupby("ice_limitation_class"):
                monthly_rows.append({"month": month, "overlay_group": label, "ice_limitation_class": cls, "cell_count": int(len(part)), "mean_A_ice": part["mean_A_ice"].mean(), "mean_SIC": part["mean_SIC"].mean()})
    return overlap, summary, pd.DataFrame(monthly_rows)


def plot_v1_panels(monthly: pd.DataFrame, annual_surfaces: list[pd.DataFrame]) -> None:
    for month in MONTHS:
        surfaces = [monthly[(monthly["target_year"] == y) & (monthly["month"] == month)] for y in TARGET_YEARS]
        plot_panel(surfaces, "A_ice_filled", f"{MONTH_LABELS[month]} A_ice v1", FIGURES_DIR / f"ice_accessibility_v1_month{month:02d}_all_targets.png", A_MIN, 1.0, "viridis")
    plot_panel(annual_surfaces, "A_ice_MayJul_mean_from_monthly_A_filled", "May-July A_ice v1", FIGURES_DIR / "ice_accessibility_v1_MayJul_all_targets.png", A_MIN, 1.0, "viridis")
    plot_panel(annual_surfaces, "SIC_MayJul_mean_filled", "May-July SIC v1", FIGURES_DIR / "ice_SIC_v1_MayJul_all_targets.png", 0.0, 1.0, "Blues_r")


def plot_panel(surfaces: list[pd.DataFrame], column: str, title: str, path: Path, vmin: float, vmax: float, cmap: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    sc = None
    for ax, surface in zip(axes.ravel(), surfaces):
        year = int(surface["target_year"].iloc[0])
        sc = ax.scatter(surface["lon"], surface["lat"], c=surface[column], s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
        ax.set_title(str(year), fontsize=10)
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75, label=column)
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(comparison: pd.DataFrame) -> None:
    annual = comparison[comparison["scale"] == "MayJul"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(annual["target_year"], annual["mean_Aice_MayJul_v0"], marker="o", label="v0")
    ax.plot(annual["target_year"], annual["mean_Aice_MayJul_v1"], marker="o", label="v1")
    ax.set_xlabel("Target year")
    ax.set_ylabel("Mean May-July A_ice")
    ax.set_title("v0 vs v1 mean ice accessibility")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ice_v0_vs_v1_mean_Aice_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(annual["target_year"], annual["Aice_lt_0_50_cells_v0"], marker="o", label="v0")
    ax.plot(annual["target_year"], annual["Aice_lt_0_50_cells_v1"], marker="o", label="v1")
    ax.set_xlabel("Target year")
    ax.set_ylabel("Cells with A_ice < 0.50")
    ax.set_title("v0 vs v1 low-access area")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ice_v0_vs_v1_low_access_area_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def update_phase_memo(missing_diag: pd.DataFrame, ocean_support: dict, annual_summary: pd.DataFrame, comparison: pd.DataFrame, overlap_summary: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_registry_and_readme() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_manifest() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def none_if_nan(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
