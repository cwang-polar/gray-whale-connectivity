"""Build monthly and May-July sea-ice accessibility surfaces.

Phase 4B creates ice-only SIC and A_ice surfaces on the existing 10 km ocean
grid. It does not create prey+ice utility surfaces and does not run LCP.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
from pathlib import Path
import re
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SIC_FILE_NAME = "cmems_mod_glo_phy_my_0.083deg_P1M-m_1779048578112.nc"
SIC_PATH = PROJECT_ROOT / "data/raw/ice" / SIC_FILE_NAME
OCEAN_GRID_PATH = PROJECT_ROOT / "data/processed/study_domain/ocean_mask_grid_10km_large_regional_bbox_v0.csv"
PREY_QUALITY_OVERLAP_PATH = PROJECT_ROOT / "outputs/prey_quality_x_bottleneck_overlap_clean_v1.csv"
PREY_QUALITY_PATH = PROJECT_ROOT / "outputs/prey_quality_persistence_clean_v1.csv"
BOTTLENECK_PATH = PROJECT_ROOT / "outputs/lcp/prey_only_bottleneck_persistence_clean_v0.csv"
ICE_DIR = PROJECT_ROOT / "outputs/ice"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
MEMO_PATH = PROJECT_ROOT / "docs/phase_memos/phase4b_ice_accessibility_surfaces.md"

TARGET_YEARS = list(range(2011, 2020))
MONTHS = [5, 6, 7]
MONTH_LABELS = {5: "May", 6: "June", 7: "July"}
A_MIN = 0.05
SIC_LOW = 0.15
SIC_HIGH = 0.55
SAMPLING_METHOD = "nearest_neighbor"


def main() -> None:
    ICE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sic_path = locate_sic_file()
    reader_info = read_sic_dataset(sic_path)
    ocean = read_ocean_grid()

    monthly_surfaces = []
    missing_by_year_month = {}
    mean_sic_by_year_month = {}
    mean_aice_by_year_month = {}
    before_minmax = [math.inf, -math.inf]
    after_minmax = [math.inf, -math.inf]

    for year in TARGET_YEARS:
        for month in MONTHS:
            sic_2d = reader_info["sic_by_year_month"][(year, month)]
            raw_sampled = sample_nearest(
                lon_grid=reader_info["longitude"],
                lat_grid=reader_info["latitude"],
                values=sic_2d,
                points_lon=ocean["lon"].to_numpy(dtype=float),
                points_lat=ocean["lat"].to_numpy(dtype=float),
            )
            if np.isfinite(raw_sampled).any():
                before_minmax[0] = min(before_minmax[0], float(np.nanmin(raw_sampled)))
                before_minmax[1] = max(before_minmax[1], float(np.nanmax(raw_sampled)))
            sic = np.clip(raw_sampled, 0.0, 1.0)
            if np.isfinite(sic).any():
                after_minmax[0] = min(after_minmax[0], float(np.nanmin(sic)))
                after_minmax[1] = max(after_minmax[1], float(np.nanmax(sic)))
            aice = accessibility_from_sic(sic)
            surface = ocean.copy()
            surface["target_year"] = year
            surface["month"] = month
            surface["SIC"] = sic
            surface["A_ice"] = aice
            surface["low_access_Aice_lt_0_25"] = (surface["A_ice"] < 0.25).astype(int)
            surface["moderate_access_Aice_lt_0_50"] = (surface["A_ice"] < 0.50).astype(int)
            surface["high_SIC_gt_0_55"] = (surface["SIC"] > 0.55).astype(int)
            surface["sampling_method"] = SAMPLING_METHOD
            surface["source_sic_file"] = str(sic_path)
            surface = surface[monthly_columns()]
            path = ICE_DIR / f"ice_accessibility_target{year}_month{month:02d}.csv"
            surface.to_csv(path, index=False)
            monthly_surfaces.append(surface)
            missing_by_year_month[f"{year}_{month:02d}"] = int(surface["SIC"].isna().sum())
            mean_sic_by_year_month[f"{year}_{month:02d}"] = none_if_nan(surface["SIC"].mean())
            mean_aice_by_year_month[f"{year}_{month:02d}"] = none_if_nan(surface["A_ice"].mean())
            plot_surface(surface, "SIC", f"SIC target {year} {MONTH_LABELS[month]}", FIGURES_DIR / f"ice_SIC_target{year}_month{month:02d}.png", 0.0, 1.0, "Blues_r")
            plot_surface(surface, "A_ice", f"Ice accessibility target {year} {MONTH_LABELS[month]}", FIGURES_DIR / f"ice_accessibility_target{year}_month{month:02d}.png", A_MIN, 1.0, "viridis")

    monthly = pd.concat(monthly_surfaces, ignore_index=True)
    monthly_summary = summarize_monthly(monthly)
    monthly_summary.to_csv(ICE_DIR / "ice_accessibility_monthly_summary_v0.csv", index=False)

    annual_surfaces = []
    for year in TARGET_YEARS:
        annual = build_annual_surface(year, monthly)
        annual.to_csv(ICE_DIR / f"ice_accessibility_MayJul_target{year}.csv", index=False)
        annual_surfaces.append(annual)
        plot_surface(annual, "SIC_MayJul_mean", f"May-July mean SIC target {year}", FIGURES_DIR / f"ice_SIC_MayJul_target{year}.png", 0.0, 1.0, "Blues_r")
        plot_surface(annual, "A_ice_MayJul_mean_from_monthly_A", f"May-July mean A_ice target {year}", FIGURES_DIR / f"ice_accessibility_MayJul_target{year}.png", A_MIN, 1.0, "viridis")

    annual_all = pd.concat(annual_surfaces, ignore_index=True)
    annual_summary = summarize_annual(annual_all)
    annual_summary.to_csv(ICE_DIR / "ice_accessibility_summary_v0.csv", index=False)
    plot_temporal_summaries(annual_summary)
    plot_panel(annual_surfaces, "SIC_MayJul_mean", "May-July mean SIC", FIGURES_DIR / "ice_SIC_MayJul_all_targets.png", 0.0, 1.0, "Blues_r")
    plot_panel(annual_surfaces, "A_ice_MayJul_mean_from_monthly_A", "May-July mean A_ice", FIGURES_DIR / "ice_accessibility_MayJul_all_targets.png", A_MIN, 1.0, "viridis")
    for month in MONTHS:
        plot_monthly_panel(monthly, month)

    persistence = build_persistence(annual_all)
    persistence.to_csv(ICE_DIR / "ice_accessibility_persistence_v0.csv", index=False)
    monthly_persistence = build_monthly_persistence(monthly)
    monthly_persistence.to_csv(ICE_DIR / "ice_accessibility_monthly_persistence_v0.csv", index=False)
    plot_categories(persistence, "ice_limitation_class", "May-July low-access persistence", FIGURES_DIR / "ice_low_access_persistence_v0.png")
    plot_monthly_persistence(monthly_persistence)

    overlap, overlap_summary, monthly_overlap_summary = build_overlap_outputs(persistence, monthly_persistence)
    overlap.to_csv(ICE_DIR / "ice_x_prey_quality_bottleneck_overlap_v0.csv", index=False)
    overlap_summary.to_csv(ICE_DIR / "ice_x_prey_quality_bottleneck_overlap_summary_v0.csv", index=False)
    monthly_overlap_summary.to_csv(ICE_DIR / "ice_monthly_x_prey_quality_bottleneck_overlap_summary_v0.csv", index=False)
    plot_overlap(overlap)

    qa = {
        "source_sic_file": str(sic_path),
        "reader_used": reader_info["reader_used"],
        "sampling_method": SAMPLING_METHOD,
        "sic_variable": "siconc",
        "sic_units": reader_info["sic_units"],
        "sic_min_max_before_clipping": before_minmax,
        "sic_min_max_after_clipping": after_minmax,
        "target_years": TARGET_YEARS,
        "months_used": MONTHS,
        "annual_surface_count": len(annual_surfaces),
        "monthly_surface_count": len(monthly_surfaces),
        "ocean_cell_count": int(len(ocean)),
        "missing_sic_cell_count_by_year_month": missing_by_year_month,
        "mean_sic_by_year_month": mean_sic_by_year_month,
        "mean_Aice_by_year_month": mean_aice_by_year_month,
        "warnings": [
            "Nearest-neighbor sampling used for v0; GLORYS grid is about 0.083 degrees and ocean grid is 10 km.",
            "SIC values clipped to [0, 1] before computing A_ice.",
            "This phase creates ice-only surfaces and diagnostics only; no prey+ice utility or LCP outputs.",
        ],
    }
    with (ICE_DIR / "ice_accessibility_qa_v0.json").open("w", encoding="utf-8") as file:
        json.dump(qa, file, indent=2)
        file.write("\n")

    write_phase_memo(reader_info, annual_summary, monthly_summary, persistence, monthly_persistence, overlap_summary, monthly_overlap_summary)
    print(json.dumps({
        "reader_used": reader_info["reader_used"],
        "monthly_surface_count": len(monthly_surfaces),
        "annual_surface_count": len(annual_surfaces),
        "annual_summary": annual_summary.to_dict(orient="records"),
        "monthly_summary": monthly_summary[["target_year", "month", "mean_SIC", "mean_A_ice", "area_cells_Aice_lt_0_50"]].to_dict(orient="records"),
        "overlap_summary": overlap_summary.to_dict(orient="records"),
    }, indent=2))


def locate_sic_file() -> Path:
    if SIC_PATH.exists():
        return SIC_PATH
    matches = list(PROJECT_ROOT.rglob(SIC_FILE_NAME))
    if not matches:
        raise FileNotFoundError(f"Could not locate {SIC_FILE_NAME}")
    return matches[0]


def read_sic_dataset(path: Path) -> dict:
    xarray_errors = []
    try:
        import xarray as xr
        for kwargs in [{}, {"engine": "netcdf4"}, {"engine": "h5netcdf"}]:
            try:
                ds = xr.open_dataset(path, **kwargs)
                reader = "xarray" if not kwargs else f"xarray/{kwargs['engine']}"
                data = extract_from_xarray(ds)
                ds.close()
                data["reader_used"] = reader
                return data
            except Exception as exc:
                xarray_errors.append(f"{kwargs or 'default'}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        xarray_errors.append(f"xarray import failed: {type(exc).__name__}: {exc}")
    data = extract_from_h5py(path)
    data["reader_used"] = "h5py fallback after xarray errors: " + " | ".join(xarray_errors)
    return data


def extract_from_xarray(ds) -> dict:
    if "siconc" not in ds:
        raise ValueError("siconc variable not found.")
    dates = pd.to_datetime(ds["time"].values)
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    sic = ds["siconc"].values.astype(float)
    by_ym = {}
    for idx, date in enumerate(dates):
        year, month = int(date.year), int(date.month)
        if year in TARGET_YEARS and month in MONTHS:
            by_ym[(year, month)] = sic[idx, :, :]
    validate_year_months(by_ym)
    return {
        "latitude": lat,
        "longitude": lon,
        "sic_by_year_month": by_ym,
        "sic_units": str(ds["siconc"].attrs.get("units", "unknown")),
        "time_range": [str(pd.Timestamp(dates.min()).date()), str(pd.Timestamp(dates.max()).date())],
    }


def extract_from_h5py(path: Path) -> dict:
    with h5py.File(path, "r") as file:
        lat = file["latitude"][()].astype(float)
        lon = file["longitude"][()].astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        dates = pd.to_datetime(file["time"][()], unit="s", origin="unix", utc=True)
        sic_data = file["siconc"]
        units = str(sic_data.attrs.get("units", "unknown"))
        by_ym = {}
        for idx, date in enumerate(dates):
            year, month = int(date.year), int(date.month)
            if year in TARGET_YEARS and month in MONTHS:
                by_ym[(year, month)] = sic_data[idx, :, :].astype(float)
        validate_year_months(by_ym)
        return {
            "latitude": lat,
            "longitude": lon,
            "sic_by_year_month": by_ym,
            "sic_units": units,
            "time_range": [str(dates.min().date()), str(dates.max().date())],
        }


def validate_year_months(by_ym: dict[tuple[int, int], np.ndarray]) -> None:
    missing = [(year, month) for year in TARGET_YEARS for month in MONTHS if (year, month) not in by_ym]
    if missing:
        raise ValueError(f"Missing target year/month SIC arrays: {missing}")


def read_ocean_grid() -> pd.DataFrame:
    grid = pd.read_csv(OCEAN_GRID_PATH)
    grid = grid[grid["ocean_flag"] == 1].copy().reset_index(drop=True)
    grid["grid_id"] = grid["grid_id"].astype(str)
    return grid[["grid_id", "x", "y", "lon", "lat", "ocean_flag"]]


def sample_nearest(lon_grid: np.ndarray, lat_grid: np.ndarray, values: np.ndarray, points_lon: np.ndarray, points_lat: np.ndarray) -> np.ndarray:
    lon_idx = np.abs(lon_grid[None, :] - points_lon[:, None]).argmin(axis=1)
    lat_idx = np.abs(lat_grid[None, :] - points_lat[:, None]).argmin(axis=1)
    return values[lat_idx, lon_idx].astype(float)


def accessibility_from_sic(sic: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(sic, dtype=float)
    aice = np.full(values.shape, np.nan, dtype=float)
    aice[values <= SIC_LOW] = 1.0
    mid = (values > SIC_LOW) & (values < SIC_HIGH)
    aice[mid] = 1.0 - ((values[mid] - SIC_LOW) / (SIC_HIGH - SIC_LOW)) * (1.0 - A_MIN)
    aice[values >= SIC_HIGH] = A_MIN
    return np.clip(aice, A_MIN, 1.0)


def monthly_columns() -> list[str]:
    return [
        "grid_id", "x", "y", "lon", "lat", "ocean_flag", "target_year", "month", "SIC", "A_ice",
        "low_access_Aice_lt_0_25", "moderate_access_Aice_lt_0_50", "high_SIC_gt_0_55",
        "sampling_method", "source_sic_file",
    ]


def build_annual_surface(year: int, monthly: pd.DataFrame) -> pd.DataFrame:
    parts = {}
    for month in MONTHS:
        part = monthly[(monthly["target_year"] == year) & (monthly["month"] == month)].copy()
        month_name = MONTH_LABELS[month]
        part = part.rename(columns={"SIC": f"SIC_{month_name}", "A_ice": f"A_ice_{month_name}"})
        parts[month] = part[["grid_id", f"SIC_{month_name}", f"A_ice_{month_name}"]]
    base = monthly[(monthly["target_year"] == year) & (monthly["month"] == 5)][["grid_id", "x", "y", "lon", "lat", "ocean_flag", "sampling_method", "source_sic_file"]].copy()
    annual = base.merge(parts[5], on="grid_id").merge(parts[6], on="grid_id").merge(parts[7], on="grid_id")
    annual["target_year"] = year
    annual["SIC_MayJul_mean"] = annual[["SIC_May", "SIC_June", "SIC_July"]].mean(axis=1)
    annual["A_ice_MayJul_mean_from_monthly_A"] = annual[["A_ice_May", "A_ice_June", "A_ice_July"]].mean(axis=1)
    annual["A_ice_from_MayJul_mean_SIC"] = accessibility_from_sic(annual["SIC_MayJul_mean"])
    annual["low_access_Aice_lt_0_25"] = (annual["A_ice_MayJul_mean_from_monthly_A"] < 0.25).astype(int)
    annual["moderate_access_Aice_lt_0_50"] = (annual["A_ice_MayJul_mean_from_monthly_A"] < 0.50).astype(int)
    annual["high_SIC_gt_0_55"] = (annual["SIC_MayJul_mean"] > 0.55).astype(int)
    keep = [
        "grid_id", "x", "y", "lon", "lat", "ocean_flag", "target_year",
        "SIC_May", "SIC_June", "SIC_July", "SIC_MayJul_mean",
        "A_ice_May", "A_ice_June", "A_ice_July",
        "A_ice_MayJul_mean_from_monthly_A", "A_ice_from_MayJul_mean_SIC",
        "low_access_Aice_lt_0_25", "moderate_access_Aice_lt_0_50", "high_SIC_gt_0_55",
        "sampling_method", "source_sic_file",
    ]
    return annual[keep]


def summarize_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, month), group in monthly.groupby(["target_year", "month"]):
        n = len(group)
        rows.append({
            "target_year": year,
            "month": month,
            "mean_SIC": group["SIC"].mean(),
            "median_SIC": group["SIC"].median(),
            "min_SIC": group["SIC"].min(),
            "max_SIC": group["SIC"].max(),
            "mean_A_ice": group["A_ice"].mean(),
            "median_A_ice": group["A_ice"].median(),
            "min_A_ice": group["A_ice"].min(),
            "max_A_ice": group["A_ice"].max(),
            "area_cells_Aice_lt_0_25": int(group["low_access_Aice_lt_0_25"].sum()),
            "area_cells_Aice_lt_0_50": int(group["moderate_access_Aice_lt_0_50"].sum()),
            "area_cells_SIC_gt_0_55": int(group["high_SIC_gt_0_55"].sum()),
            "percent_cells_Aice_lt_0_25": group["low_access_Aice_lt_0_25"].sum() / n * 100,
            "percent_cells_Aice_lt_0_50": group["moderate_access_Aice_lt_0_50"].sum() / n * 100,
            "percent_cells_SIC_gt_0_55": group["high_SIC_gt_0_55"].sum() / n * 100,
        })
    return pd.DataFrame(rows).sort_values(["target_year", "month"])


def summarize_annual(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in annual.groupby("target_year"):
        n = len(group)
        primary = group["A_ice_MayJul_mean_from_monthly_A"]
        rows.append({
            "target_year": year,
            "mean_SIC_MayJul": group["SIC_MayJul_mean"].mean(),
            "median_SIC_MayJul": group["SIC_MayJul_mean"].median(),
            "min_SIC_MayJul": group["SIC_MayJul_mean"].min(),
            "max_SIC_MayJul": group["SIC_MayJul_mean"].max(),
            "mean_A_ice_primary": primary.mean(),
            "median_A_ice_primary": primary.median(),
            "min_A_ice_primary": primary.min(),
            "max_A_ice_primary": primary.max(),
            "mean_A_ice_from_mean_SIC": group["A_ice_from_MayJul_mean_SIC"].mean(),
            "median_A_ice_from_mean_SIC": group["A_ice_from_MayJul_mean_SIC"].median(),
            "area_cells_Aice_lt_0_25": int(group["low_access_Aice_lt_0_25"].sum()),
            "area_cells_Aice_lt_0_50": int(group["moderate_access_Aice_lt_0_50"].sum()),
            "area_cells_SIC_gt_0_55": int(group["high_SIC_gt_0_55"].sum()),
            "percent_cells_Aice_lt_0_25": group["low_access_Aice_lt_0_25"].sum() / n * 100,
            "percent_cells_Aice_lt_0_50": group["moderate_access_Aice_lt_0_50"].sum() / n * 100,
            "percent_cells_SIC_gt_0_55": group["high_SIC_gt_0_55"].sum() / n * 100,
        })
    return pd.DataFrame(rows).sort_values("target_year")


def build_persistence(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in annual.groupby("grid_id", sort=False):
        first = group.iloc[0]
        years_low50 = int((group["A_ice_MayJul_mean_from_monthly_A"] < 0.50).sum())
        if years_low50 >= 5:
            cls = "persistent_low_access"
        elif years_low50 >= 1:
            cls = "intermittent_low_access"
        else:
            cls = "rarely_or_never_low_access"
        rows.append({
            "grid_id": grid_id, "x": first["x"], "y": first["y"], "lon": first["lon"], "lat": first["lat"],
            "years_Aice_lt_0_25": int((group["A_ice_MayJul_mean_from_monthly_A"] < 0.25).sum()),
            "years_Aice_lt_0_50": years_low50,
            "years_SIC_gt_0_55": int((group["SIC_MayJul_mean"] > 0.55).sum()),
            "mean_SIC_MayJul": group["SIC_MayJul_mean"].mean(),
            "mean_A_ice": group["A_ice_MayJul_mean_from_monthly_A"].mean(),
            "min_A_ice": group["A_ice_MayJul_mean_from_monthly_A"].min(),
            "max_A_ice": group["A_ice_MayJul_mean_from_monthly_A"].max(),
            "ice_limitation_class": cls,
        })
    return pd.DataFrame(rows)


def build_monthly_persistence(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grid_id, month), group in monthly.groupby(["grid_id", "month"], sort=False):
        first = group.iloc[0]
        years_low50 = int((group["A_ice"] < 0.50).sum())
        if years_low50 >= 5:
            cls = "persistent_low_access"
        elif years_low50 >= 1:
            cls = "intermittent_low_access"
        else:
            cls = "rarely_or_never_low_access"
        rows.append({
            "grid_id": grid_id, "x": first["x"], "y": first["y"], "lon": first["lon"], "lat": first["lat"],
            "month": month,
            "years_Aice_lt_0_25": int((group["A_ice"] < 0.25).sum()),
            "years_Aice_lt_0_50": years_low50,
            "years_SIC_gt_0_55": int((group["SIC"] > 0.55).sum()),
            "mean_SIC": group["SIC"].mean(),
            "mean_A_ice": group["A_ice"].mean(),
            "min_A_ice": group["A_ice"].min(),
            "max_A_ice": group["A_ice"].max(),
            "ice_limitation_class": cls,
        })
    return pd.DataFrame(rows)


def build_overlap_outputs(persistence: pd.DataFrame, monthly_persistence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlap = persistence.copy()
    if PREY_QUALITY_OVERLAP_PATH.exists():
        prey = pd.read_csv(PREY_QUALITY_OVERLAP_PATH)
        prey["grid_id"] = prey["grid_id"].astype(str)
        cols = ["grid_id", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class", "bottleneck_years_in_top5pct"]
        cols = [col for col in cols if col in prey.columns]
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
    keep = [
        "grid_id", "lon", "lat", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class",
        "years_in_top5pct", "years_Aice_lt_0_25", "years_Aice_lt_0_50", "years_SIC_gt_0_55",
        "mean_A_ice", "mean_SIC_MayJul", "ice_limitation_class",
    ]
    overlap = overlap[keep]

    combo = overlap.groupby(["prey_bottleneck_class", "ice_limitation_class"], dropna=False).agg(
        cell_count=("grid_id", "count"),
        mean_A_ice=("mean_A_ice", "mean"),
        mean_SIC_MayJul=("mean_SIC_MayJul", "mean"),
        mean_years_Aice_lt_0_50=("years_Aice_lt_0_50", "mean"),
    ).reset_index()
    special_rows = []
    for label, mask in {
        "persistent_high_quality + persistent_low_access": (overlap["prey_quality_class"] == "persistent_high_quality") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_high_persistent_bottleneck + persistent_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_high_persistent_bottleneck + intermittent_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "intermittent_low_access"),
        "persistent_high_persistent_bottleneck + rarely_or_never_low_access": (overlap["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (overlap["ice_limitation_class"] == "rarely_or_never_low_access"),
        "persistent_zero_proxy + persistent_low_access": (overlap["prey_quality_class"] == "persistent_zero_proxy") & (overlap["ice_limitation_class"] == "persistent_low_access"),
        "persistent_bottleneck + persistent_low_access": (overlap["bottleneck_persistence_class"] == "persistent_bottleneck") & (overlap["ice_limitation_class"] == "persistent_low_access"),
    }.items():
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
    monthly_summary = pd.DataFrame(monthly_rows)
    return overlap, summary, monthly_summary


def plot_surface(df: pd.DataFrame, column: str, title: str, path: Path, vmin: float, vmax: float, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df["lon"], df["lat"], c=df[column], s=7, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-177.5, -156.0)
    ax.set_ylim(62.5, 73.5)
    fig.colorbar(sc, ax=ax, label=column)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def plot_monthly_panel(monthly: pd.DataFrame, month: int) -> None:
    surfaces = [monthly[(monthly["target_year"] == year) & (monthly["month"] == month)] for year in TARGET_YEARS]
    plot_panel(surfaces, "A_ice", f"{MONTH_LABELS[month]} A_ice across target years", FIGURES_DIR / f"ice_accessibility_month{month:02d}_all_targets.png", A_MIN, 1.0, "viridis")


def plot_temporal_summaries(summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(summary["target_year"], summary["mean_SIC_MayJul"], marker="o", color="#1f77b4", label="mean SIC")
    ax1.set_ylabel("Mean SIC", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(summary["target_year"], summary["mean_A_ice_primary"], marker="o", color="#2ca02c", label="mean A_ice")
    ax2.set_ylabel("Mean A_ice", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(A_MIN, 1)
    ax1.set_xlabel("Target year")
    ax1.set_title("May-July mean SIC and ice accessibility")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ice_temporal_mean_SIC_Aice_v0.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["target_year"], summary["area_cells_Aice_lt_0_25"], marker="o", label="A_ice < 0.25")
    ax.plot(summary["target_year"], summary["area_cells_Aice_lt_0_50"], marker="o", label="A_ice < 0.50")
    ax.plot(summary["target_year"], summary["area_cells_SIC_gt_0_55"], marker="o", label="SIC > 0.55")
    ax.set_xlabel("Target year")
    ax.set_ylabel("Grid-cell count")
    ax.set_title("May-July low-access and high-SIC area")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ice_temporal_low_access_area_v0.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_categories(df: pd.DataFrame, column: str, title: str, path: Path) -> None:
    cats = sorted(df[column].dropna().unique())
    cmap = plt.get_cmap("tab10", max(len(cats), 1))
    colors = {cat: cmap(i) for i, cat in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in cats:
        part = df[df[column] == cat]
        ax.scatter(part["lon"], part["lat"], s=7, color=colors[cat], linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-177.5, -156.0)
    ax.set_ylim(62.5, 73.5)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[cat], markersize=6, label=cat) for cat in cats]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_persistence(monthly_persistence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for ax, month in zip(axes, MONTHS):
        part = monthly_persistence[monthly_persistence["month"] == month]
        sc = ax.scatter(part["lon"], part["lat"], c=part["years_Aice_lt_0_50"], s=6, cmap="magma", vmin=0, vmax=9, linewidth=0)
        ax.set_title(MONTH_LABELS[month])
    fig.colorbar(sc, ax=axes.tolist(), label="Years A_ice < 0.50")
    fig.suptitle("Monthly low-access persistence")
    fig.savefig(FIGURES_DIR / "ice_monthly_low_access_persistence_v0.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overlap(overlap: pd.DataFrame) -> None:
    target = overlap.copy()
    target["plot_class"] = np.where(
        (target["prey_bottleneck_class"] == "persistent_high_persistent_bottleneck") & (target["ice_limitation_class"] == "persistent_low_access"),
        "high_prey_bottleneck + persistent_low_access",
        np.where(target["ice_limitation_class"] == "persistent_low_access", "persistent_low_access", "other"),
    )
    plot_categories(target, "plot_class", "Ice x prey-bottleneck overlap diagnostic", FIGURES_DIR / "ice_x_prey_bottleneck_overlap_v0.png")


def write_phase_memo(reader_info: dict, annual_summary: pd.DataFrame, monthly_summary: pd.DataFrame, persistence: pd.DataFrame, monthly_persistence: pd.DataFrame, overlap_summary: pd.DataFrame, monthly_overlap_summary: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def none_if_nan(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
