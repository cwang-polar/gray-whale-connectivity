"""Build Phase 5B project-grid ship intensity and diagnostic disturbance surfaces."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inventory_ship_data_v0 import (
    CLAIMS,
    FILENAME_RE,
    MANIFEST,
    NODATA_SENTINEL,
    OCEAN_GRID,
    README,
    REGISTRY,
    ROOT,
    SHIP_CRS,
    SHIP_DATA,
    points_within_bounds,
    read_array,
    sample_nearest,
    tif_metadata,
    transform_ocean_points,
)


TARGET_YEARS = list(range(2015, 2020))
MONTHS = [5, 6, 7]
VESSEL_TYPES = ["Cargo", "Fishing", "Other", "Tanker"]
OUT_DIR = ROOT / "outputs/ship"
FIG_DIR = ROOT / "outputs/figures"
PHASE_MEMO = ROOT / "docs/phase_memos/phase5b_ship_disturbance_surfaces.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase5b_ship_disturbance_surfaces_paper_use.md"
QA_JSON = OUT_DIR / "ship_disturbance_qa_v0.json"

READER_USED = "tifffile+pyproj"
SAMPLING_METHOD = "project-grid centroid nearest raster-cell lookup"
PRIMARY_SCALING = "S_scaled_log_q99"
NODATA_TREATMENT = "Raw raster nodata values are treated as NaN during sampling; target QA found zero sampled nodata cells, so no nodata-to-zero fill was needed."


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_project_ocean_grid() -> pd.DataFrame:
    grid = pd.read_csv(OCEAN_GRID)
    required = ["grid_id", "x", "y", "lon", "lat", "ocean_flag"]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise ValueError(f"Project ocean grid missing required columns: {missing}")
    grid["grid_id"] = grid["grid_id"].astype(str)
    return grid[grid["ocean_flag"] == 1].copy()


def raster_path(year: int, month: int, vessel_type: str) -> Path:
    path = SHIP_DATA / f"Raster_{year}_{month:02d}_{vessel_type}_10km.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sample_file(path: Path, grid: pd.DataFrame, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, dict]:
    meta = tif_metadata(path)
    arr = read_array(path)
    within = points_within_bounds(xs, ys, meta["bounds_tuple"])
    values = sample_nearest(arr, xs, ys, meta)
    outside_count = int((~within).sum())
    nodata_count = int(np.isnan(values[within]).sum())
    return values, {
        "file": str(path.relative_to(ROOT)),
        "meta": meta,
        "outside_bounds_count": outside_count,
        "nodata_count": nodata_count,
        "sampled_cell_count": int(np.isfinite(values).sum()),
        "nonzero_sampled_cell_count": int(np.nansum(values > 0)),
        "sampled_min": float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
        "sampled_mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
        "sampled_median": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
        "sampled_max": float(np.nanmax(values)) if np.isfinite(values).any() else np.nan,
    }


def build_monthly_surfaces(grid: pd.DataFrame, xs: np.ndarray, ys: np.ndarray) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    monthly = []
    qa_rows = []
    for year in TARGET_YEARS:
        for month in MONTHS:
            df = grid[["grid_id", "x", "y", "lon", "lat", "ocean_flag"]].copy()
            source_files = []
            outside_total = 0
            nodata_total = 0
            for vessel_type in VESSEL_TYPES:
                path = raster_path(year, month, vessel_type)
                values, qa = sample_file(path, grid, xs, ys)
                # No sampled nodata appeared in Phase 5A/5B QA. If it appears later,
                # keep it as NaN so the problem is visible rather than silently zeroed.
                df[f"S_{vessel_type.lower()}"] = values
                source_files.append(str(path.relative_to(ROOT)))
                outside_total += qa["outside_bounds_count"]
                nodata_total += qa["nodata_count"]
                meta = qa["meta"]
                qa_rows.append(
                    {
                        "representative_file": str(path.relative_to(ROOT)),
                        "target_year": year,
                        "month": month,
                        "vessel_type": vessel_type,
                        "raster_crs": "NAD83 Albers Equal Area",
                        "raster_bounds": meta["bounds"],
                        "raster_resolution_x": meta["resolution_x"],
                        "raster_resolution_y": meta["resolution_y"],
                        "project_ocean_cell_count": len(grid),
                        "sampled_cell_count": qa["sampled_cell_count"],
                        "outside_bounds_count": qa["outside_bounds_count"],
                        "nodata_count": qa["nodata_count"],
                        "nonzero_sampled_cell_count": qa["nonzero_sampled_cell_count"],
                        "sampled_min": qa["sampled_min"],
                        "sampled_mean": qa["sampled_mean"],
                        "sampled_median": qa["sampled_median"],
                        "sampled_max": qa["sampled_max"],
                        "sampling_method": SAMPLING_METHOD,
                        "reader_used": READER_USED,
                        "notes": "Sampled to project ocean grid centroids; Kapsar raster row/column IDs were not used as analysis IDs.",
                    }
                )
            df["target_year"] = year
            df["month"] = month
            df["S_ship_all"] = df[[f"S_{v.lower()}" for v in VESSEL_TYPES]].sum(axis=1, skipna=False)
            df["sampling_method"] = SAMPLING_METHOD
            df["reader_used"] = READER_USED
            df["source_files"] = ";".join(source_files)
            df["outside_bounds_flag_count"] = outside_total
            df["nodata_flag_count"] = nodata_total
            monthly.append(df)
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUT_DIR / "ship_grid_harmonization_qa_v0.csv", index=False)
    return monthly, qa


def compute_scaling(monthly: list[pd.DataFrame]) -> dict:
    values = pd.concat([m["S_ship_all"] for m in monthly], ignore_index=True).astype(float)
    values = values[np.isfinite(values)]
    log_values = np.log1p(values)
    params = {
        "raw_q95": float(values.quantile(0.95)),
        "raw_q99": float(values.quantile(0.99)),
        "log_q95": float(log_values.quantile(0.95)),
        "log_q99": float(log_values.quantile(0.99)),
        "number_of_cell_months": int(len(values)),
        "zero_fraction": float((values <= 0).mean()),
        "reader_used": READER_USED,
        "sampling_method": SAMPLING_METHOD,
        "primary_scaling_choice": PRIMARY_SCALING,
        "nodata_treatment": NODATA_TREATMENT,
        "notes": "Primary scaling uses log1p all-vessel intensity and q99 clipping to reduce domination by extreme vessel-track cells while preserving high-use routes.",
    }
    (OUT_DIR / "ship_scaling_parameters_v0.json").write_text(json.dumps(params, indent=2))
    return params


def add_scaling_fields(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    raw = out["S_ship_all"].astype(float)
    log = np.log1p(raw)
    for name, vals, q in [
        ("S_scaled_raw_q95", raw, params["raw_q95"]),
        ("S_scaled_raw_q99", raw, params["raw_q99"]),
        ("S_scaled_log_q95", log, params["log_q95"]),
        ("S_scaled_log_q99", log, params["log_q99"]),
    ]:
        out[name] = 0.0 if q <= 0 else np.clip(vals, 0, q) / q
    out["S_scaled_ship_primary"] = out[PRIMARY_SCALING]
    out["D_ship_alpha0_5"] = np.clip(1 - 0.5 * out["S_scaled_ship_primary"], 0.05, 1.0)
    out["D_ship_alpha0_7"] = np.clip(1 - 0.7 * out["S_scaled_ship_primary"], 0.05, 1.0)
    out["D_ship_exp_lambda1"] = np.clip(np.exp(-out["S_scaled_ship_primary"]), 0.05, 1.0)
    return out


def write_monthly(monthly: list[pd.DataFrame], params: dict) -> list[pd.DataFrame]:
    scaled = []
    for df in monthly:
        out = add_scaling_fields(df, params)
        year = int(out["target_year"].iloc[0])
        month = int(out["month"].iloc[0])
        out.to_csv(OUT_DIR / f"ship_intensity_target{year}_month{month:02d}.csv", index=False)
        scaled.append(out)
    return scaled


def monthly_summary(monthly: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in monthly:
        s = df["S_ship_all"]
        scaled = df["S_scaled_ship_primary"]
        rows.append(
            {
                "target_year": int(df["target_year"].iloc[0]),
                "month": int(df["month"].iloc[0]),
                "mean_S_ship_all": s.mean(),
                "median_S_ship_all": s.median(),
                "max_S_ship_all": s.max(),
                "zero_cell_percent": (s <= 0).mean() * 100,
                "mean_S_scaled_ship_primary": scaled.mean(),
                "median_S_scaled_ship_primary": scaled.median(),
                "max_S_scaled_ship_primary": scaled.max(),
                "mean_D_ship_alpha0_5": df["D_ship_alpha0_5"].mean(),
                "mean_D_ship_alpha0_7": df["D_ship_alpha0_7"].mean(),
                "mean_D_ship_exp_lambda1": df["D_ship_exp_lambda1"].mean(),
                "cells_S_scaled_gt_0_25": int((scaled > 0.25).sum()),
                "cells_S_scaled_gt_0_50": int((scaled > 0.50).sum()),
                "cells_S_scaled_gt_0_75": int((scaled > 0.75).sum()),
                "percent_cells_S_scaled_gt_0_50": (scaled > 0.50).mean() * 100,
            }
        )
    out = pd.DataFrame(rows).sort_values(["target_year", "month"])
    out.to_csv(OUT_DIR / "ship_monthly_summary_v0.csv", index=False)
    return out


def build_mayjul(monthly: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    outputs = []
    rows = []
    for year in TARGET_YEARS:
        stack = pd.concat([m for m in monthly if int(m["target_year"].iloc[0]) == year], ignore_index=True)
        agg = (
            stack.groupby(["grid_id", "x", "y", "lon", "lat", "ocean_flag"], as_index=False)
            .agg(
                S_ship_all_MayJul_mean=("S_ship_all", "mean"),
                S_ship_all_MayJul_sum=("S_ship_all", "sum"),
                S_scaled_ship_primary_MayJul_mean=("S_scaled_ship_primary", "mean"),
                D_ship_alpha0_5_MayJul_mean=("D_ship_alpha0_5", "mean"),
                D_ship_alpha0_7_MayJul_mean=("D_ship_alpha0_7", "mean"),
                D_ship_exp_lambda1_MayJul_mean=("D_ship_exp_lambda1", "mean"),
                month_count=("month", "nunique"),
            )
        )
        agg["target_year"] = year
        agg.to_csv(OUT_DIR / f"ship_intensity_MayJul_target{year}.csv", index=False)
        outputs.append(agg)
        s = agg["S_ship_all_MayJul_mean"]
        scaled = agg["S_scaled_ship_primary_MayJul_mean"]
        rows.append(
            {
                "target_year": year,
                "mean_S_ship_all_MayJul": s.mean(),
                "median_S_ship_all_MayJul": s.median(),
                "max_S_ship_all_MayJul": s.max(),
                "mean_S_scaled_ship_primary_MayJul": scaled.mean(),
                "median_S_scaled_ship_primary_MayJul": scaled.median(),
                "mean_D_ship_alpha0_5_MayJul": agg["D_ship_alpha0_5_MayJul_mean"].mean(),
                "mean_D_ship_alpha0_7_MayJul": agg["D_ship_alpha0_7_MayJul_mean"].mean(),
                "mean_D_ship_exp_lambda1_MayJul": agg["D_ship_exp_lambda1_MayJul_mean"].mean(),
                "cells_S_scaled_gt_0_50_MayJul": int((scaled > 0.50).sum()),
                "percent_cells_S_scaled_gt_0_50_MayJul": (scaled > 0.50).mean() * 100,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "ship_MayJul_summary_v0.csv", index=False)
    return outputs, summary


def vessel_contribution(monthly: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for df in monthly:
        totals = {v.lower(): float(df[f"S_{v.lower()}"].sum()) for v in VESSEL_TYPES}
        total_all = sum(totals.values())
        row = {
            "target_year": int(df["target_year"].iloc[0]),
            "month": int(df["month"].iloc[0]),
            "total_cargo": totals["cargo"],
            "total_fishing": totals["fishing"],
            "total_other": totals["other"],
            "total_tanker": totals["tanker"],
            "total_all": total_all,
        }
        for k in ["cargo", "fishing", "other", "tanker"]:
            row[f"{k}_fraction"] = totals[k] / total_all if total_all > 0 else np.nan
        rows.append(row)
    monthly_out = pd.DataFrame(rows)
    monthly_out.to_csv(OUT_DIR / "ship_vessel_type_contribution_v0.csv", index=False)
    mayjul = (
        monthly_out.groupby("target_year", as_index=False)[
            ["total_cargo", "total_fishing", "total_other", "total_tanker", "total_all"]
        ]
        .sum()
    )
    for k in ["cargo", "fishing", "other", "tanker"]:
        mayjul[f"{k}_fraction"] = mayjul[f"total_{k}"] / mayjul["total_all"]
    mayjul.to_csv(OUT_DIR / "ship_vessel_type_contribution_MayJul_v0.csv", index=False)
    return monthly_out, mayjul


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def corridor_overlap(monthly: list[pd.DataFrame], mayjul: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for df in monthly:
        year = int(df["target_year"].iloc[0])
        lcp = pd.read_csv(ROOT / f"outputs/lcp/prey_ice_beta1_lcp_main_south_to_north_target{year}.csv")
        lcp["grid_id"] = lcp["grid_id"].astype(str)
        merged = df.merge(lcp[["grid_id", "lcp_frequency_normalized"]], on="grid_id", how="left")
        freq = merged["lcp_frequency_normalized"].fillna(0)
        for label, mask in [
            ("all_ocean", np.ones(len(merged), dtype=bool)),
            ("corridor", freq > 0),
            ("core_corridor", freq > 0.05),
            ("high_use_corridor", freq > 0.10),
        ]:
            sub = merged[mask]
            rows.append(
                {
                    "target_year": year,
                    "month": int(df["month"].iloc[0]),
                    "corridor_group": label,
                    "cell_count": int(len(sub)),
                    "mean_S_scaled_ship_primary": sub["S_scaled_ship_primary"].mean(),
                    "percent_cells_S_scaled_gt_0_25": (sub["S_scaled_ship_primary"] > 0.25).mean() * 100 if len(sub) else np.nan,
                    "percent_cells_S_scaled_gt_0_50": (sub["S_scaled_ship_primary"] > 0.50).mean() * 100 if len(sub) else np.nan,
                    "percent_cells_S_scaled_gt_0_75": (sub["S_scaled_ship_primary"] > 0.75).mean() * 100 if len(sub) else np.nan,
                    "frequency_weighted_mean_S_scaled_ship_primary": weighted_mean(
                        merged["S_scaled_ship_primary"], freq
                    )
                    if label == "corridor"
                    else np.nan,
                }
            )
    monthly_out = pd.DataFrame(rows)
    monthly_out.to_csv(OUT_DIR / "ship_overlap_with_prey_ice_corridor_v0.csv", index=False)

    rows = []
    for df in mayjul:
        year = int(df["target_year"].iloc[0])
        lcp = pd.read_csv(ROOT / f"outputs/lcp/prey_ice_beta1_lcp_main_south_to_north_target{year}.csv")
        lcp["grid_id"] = lcp["grid_id"].astype(str)
        merged = df.merge(lcp[["grid_id", "lcp_frequency_normalized"]], on="grid_id", how="left")
        freq = merged["lcp_frequency_normalized"].fillna(0)
        for label, mask in [
            ("all_ocean", np.ones(len(merged), dtype=bool)),
            ("corridor", freq > 0),
            ("core_corridor", freq > 0.05),
            ("high_use_corridor", freq > 0.10),
        ]:
            sub = merged[mask]
            rows.append(
                {
                    "target_year": year,
                    "corridor_group": label,
                    "cell_count": int(len(sub)),
                    "mean_S_scaled_ship_primary_MayJul": sub["S_scaled_ship_primary_MayJul_mean"].mean(),
                    "percent_cells_S_scaled_gt_0_25": (sub["S_scaled_ship_primary_MayJul_mean"] > 0.25).mean() * 100 if len(sub) else np.nan,
                    "percent_cells_S_scaled_gt_0_50": (sub["S_scaled_ship_primary_MayJul_mean"] > 0.50).mean() * 100 if len(sub) else np.nan,
                    "percent_cells_S_scaled_gt_0_75": (sub["S_scaled_ship_primary_MayJul_mean"] > 0.75).mean() * 100 if len(sub) else np.nan,
                    "frequency_weighted_mean_S_scaled_ship_primary_MayJul": weighted_mean(
                        merged["S_scaled_ship_primary_MayJul_mean"], freq
                    )
                    if label == "corridor"
                    else np.nan,
                }
            )
    mayjul_out = pd.DataFrame(rows)
    mayjul_out.to_csv(OUT_DIR / "ship_overlap_with_prey_ice_corridor_MayJul_v0.csv", index=False)
    return monthly_out, mayjul_out


def class_overlap(mayjul: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prey_path = ROOT / "outputs/prey_quality_x_bottleneck_overlap_clean_v1.csv"
    ice_bottleneck_path = ROOT / "outputs/lcp/prey_ice_beta1_bottleneck_persistence_v0.csv"
    frames = []
    prey = pd.read_csv(prey_path) if prey_path.exists() else pd.DataFrame()
    ice = pd.read_csv(ice_bottleneck_path) if ice_bottleneck_path.exists() else pd.DataFrame()
    if not prey.empty:
        prey["grid_id"] = prey["grid_id"].astype(str)
    if not ice.empty:
        ice["grid_id"] = ice["grid_id"].astype(str)
        ice = ice.rename(columns={"years_in_top5pct": "prey_ice_bottleneck_years_in_top5pct"})
    for df in mayjul:
        merged = df[[
            "grid_id",
            "lon",
            "lat",
            "target_year",
            "S_scaled_ship_primary_MayJul_mean",
            "D_ship_alpha0_5_MayJul_mean",
            "D_ship_alpha0_7_MayJul_mean",
            "D_ship_exp_lambda1_MayJul_mean",
        ]].copy()
        if not prey.empty:
            merged = merged.merge(
                prey[["grid_id", "prey_quality_class", "prey_bottleneck_class", "bottleneck_persistence_class"]],
                on="grid_id",
                how="left",
            )
        if not ice.empty:
            merged = merged.merge(
                ice[["grid_id", "prey_ice_bottleneck_years_in_top5pct"]],
                on="grid_id",
                how="left",
            )
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "ship_x_prey_quality_bottleneck_overlap_v0.csv", index=False)

    rows = []
    class_defs = {
        "persistent_high_persistent_bottleneck": out.get("prey_bottleneck_class", pd.Series(index=out.index, dtype=str)).eq("persistent_high_persistent_bottleneck"),
        "persistent_high_quality": out.get("prey_quality_class", pd.Series(index=out.index, dtype=str)).eq("persistent_high_quality"),
        "persistent_bottleneck": out.get("bottleneck_persistence_class", pd.Series(index=out.index, dtype=str)).eq("persistent_bottleneck"),
        "prey_ice_persistent_bottleneck": out.get("prey_ice_bottleneck_years_in_top5pct", pd.Series(0, index=out.index)).fillna(0) >= 5,
    }
    for year in TARGET_YEARS:
        yr = out[out["target_year"] == year]
        for label, mask_all in class_defs.items():
            sub = yr[mask_all.loc[yr.index]]
            rows.append(
                {
                    "target_year": year,
                    "class_label": label,
                    "cell_count": int(len(sub)),
                    "mean_S_scaled_ship_primary_MayJul": sub["S_scaled_ship_primary_MayJul_mean"].mean(),
                    "median_S_scaled_ship_primary_MayJul": sub["S_scaled_ship_primary_MayJul_mean"].median(),
                    "percent_cells_S_scaled_gt_0_50": (sub["S_scaled_ship_primary_MayJul_mean"] > 0.50).mean() * 100 if len(sub) else np.nan,
                    "mean_D_ship_alpha0_5": sub["D_ship_alpha0_5_MayJul_mean"].mean(),
                    "mean_D_ship_alpha0_7": sub["D_ship_alpha0_7_MayJul_mean"].mean(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "ship_x_prey_quality_bottleneck_overlap_summary_v0.csv", index=False)
    return out, summary


def scatter_map(df: pd.DataFrame, value: str, path: Path, title: str, vmin=None, vmax=None, cmap="viridis") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["lon"], df["lat"], c=df[value], s=5, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def panel_maps(frames: list[pd.DataFrame], value: str, path: Path, title: str, vmin=None, vmax=None, cmap="viridis") -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 4.5), sharex=True, sharey=True)
    if len(frames) == 1:
        axes = [axes]
    sc = None
    for ax, df in zip(axes, frames):
        sc = ax.scatter(df["lon"], df["lat"], c=df[value], s=3, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(str(int(df["target_year"].iloc[0])))
        ax.set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    fig.suptitle(title)
    fig.colorbar(sc, ax=axes, shrink=0.72)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figures(monthly: list[pd.DataFrame], mayjul: list[pd.DataFrame], monthly_sum: pd.DataFrame, mayjul_sum: pd.DataFrame, contrib: pd.DataFrame, overlap_mj: pd.DataFrame, class_summary: pd.DataFrame, params: dict, qa: pd.DataFrame) -> list[str]:
    created = []
    vmax = pd.concat([m["S_scaled_ship_primary"] for m in monthly]).quantile(0.995)
    for df in monthly:
        year = int(df["target_year"].iloc[0])
        month = int(df["month"].iloc[0])
        p = FIG_DIR / f"ship_intensity_target{year}_month{month:02d}.png"
        scatter_map(df, "S_scaled_ship_primary", p, f"Ship intensity {year}-{month:02d}", 0, max(vmax, 0.01), "magma")
        created.append(str(p.relative_to(ROOT)))
    for month in MONTHS:
        frames = [m for m in monthly if int(m["month"].iloc[0]) == month]
        p = FIG_DIR / f"ship_intensity_month{month:02d}_all_targets.png"
        panel_maps(frames, "S_scaled_ship_primary", p, f"Ship intensity month {month:02d}", 0, max(vmax, 0.01), "magma")
        created.append(str(p.relative_to(ROOT)))
    p = FIG_DIR / "ship_intensity_MayJul_all_targets.png"
    panel_maps(mayjul, "S_scaled_ship_primary_MayJul_mean", p, "Ship intensity May-Jul", 0, max(vmax, 0.01), "magma")
    created.append(str(p.relative_to(ROOT)))
    for col, name in [
        ("D_ship_alpha0_5_MayJul_mean", "alpha0_5"),
        ("D_ship_alpha0_7_MayJul_mean", "alpha0_7"),
    ]:
        p = FIG_DIR / f"ship_disturbance_Dship_MayJul_{name}_all_targets.png"
        panel_maps(mayjul, col, p, f"D_ship May-Jul {name}", 0.3, 1.0, "viridis")
        created.append(str(p.relative_to(ROOT)))

    all_raw = pd.concat([m["S_ship_all"] for m in monthly], ignore_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(all_raw, bins=80, color="#4c78a8")
    ax[0].set_title("Raw S_ship_all")
    ax[1].hist(np.log1p(all_raw), bins=80, color="#f58518")
    ax[1].set_title("log1p(S_ship_all)")
    fig.tight_layout()
    p = FIG_DIR / "ship_intensity_distribution_raw_vs_log_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["raw q95", "raw q99", "log q95", "log q99"], [params["raw_q95"], params["raw_q99"], params["log_q95"], params["log_q99"]])
    ax.set_title("Ship scaling thresholds")
    fig.tight_layout()
    p = FIG_DIR / "ship_scaling_q95_q99_comparison_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for month in MONTHS:
        sub = monthly_sum[monthly_sum["month"] == month]
        ax.plot(sub["target_year"], sub["mean_S_scaled_ship_primary"], marker="o", label=f"Month {month:02d}")
    ax.legend()
    ax.set_ylabel("Mean scaled ship intensity")
    ax.set_title("Monthly ship intensity")
    fig.tight_layout()
    p = FIG_DIR / "ship_temporal_monthly_intensity_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mayjul_sum["target_year"], mayjul_sum["mean_S_scaled_ship_primary_MayJul"], marker="o")
    ax.set_ylabel("Mean scaled ship intensity")
    ax.set_title("May-Jul ship intensity")
    fig.tight_layout()
    p = FIG_DIR / "ship_temporal_MayJul_intensity_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    contrib_mj = pd.read_csv(OUT_DIR / "ship_vessel_type_contribution_MayJul_v0.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(contrib_mj))
    for col in ["cargo_fraction", "fishing_fraction", "other_fraction", "tanker_fraction"]:
        ax.bar(contrib_mj["target_year"], contrib_mj[col], bottom=bottom, label=col.replace("_fraction", ""))
        bottom += contrib_mj[col].to_numpy()
    ax.legend()
    ax.set_ylabel("Fraction")
    ax.set_title("May-Jul vessel-type contribution")
    fig.tight_layout()
    p = FIG_DIR / "ship_vessel_type_contribution_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    corr = overlap_mj[overlap_mj["corridor_group"].isin(["all_ocean", "corridor", "core_corridor"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, sub in corr.groupby("corridor_group"):
        ax.plot(sub["target_year"], sub["mean_S_scaled_ship_primary_MayJul"], marker="o", label=group)
    ax.legend()
    ax.set_title("Ship overlap with prey+ice corridors")
    ax.set_ylabel("Mean May-Jul scaled ship intensity")
    fig.tight_layout()
    p = FIG_DIR / "ship_overlap_with_prey_ice_corridor_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = class_summary.pivot(index="target_year", columns="class_label", values="mean_S_scaled_ship_primary_MayJul")
    pivot.plot(ax=ax, marker="o")
    ax.set_ylabel("Mean scaled ship intensity")
    ax.set_title("Ship exposure by prey/bottleneck class")
    fig.tight_layout()
    p = FIG_DIR / "ship_x_prey_bottleneck_overlap_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))

    # Grid harmonization QA map using representative Cargo 2019-07.
    rep = [m for m in monthly if int(m["target_year"].iloc[0]) == 2019 and int(m["month"].iloc[0]) == 7][0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rep["lon"], rep["lat"], c="lightgray", s=3, label="Project ocean grid")
    nonzero = rep[rep["S_cargo"] > 0]
    ax.scatter(nonzero["lon"], nonzero["lat"], c=nonzero["S_cargo"], s=6, cmap="magma", label="Sampled nonzero Cargo")
    ax.set_title("Ship grid harmonization QA, 2019-07 Cargo")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    p = FIG_DIR / "ship_grid_harmonization_qa_v0.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(str(p.relative_to(ROOT)))
    return created


def qa_json(params: dict, qa: pd.DataFrame, figure_paths: list[str]) -> dict:
    missing_files = []
    for year in TARGET_YEARS:
        for month in MONTHS:
            for vessel in VESSEL_TYPES:
                if not raster_path(year, month, vessel).exists():
                    missing_files.append(f"{year}-{month:02d}-{vessel}")
    qa_data = {
        "source_dataset": "ArcticData/Kapsar 2022 North Pacific and Arctic Marine Vessel Traffic Dataset (2015-2020), 10 km resolution, DOI 10.18739/A2NZ80R4J",
        "years": TARGET_YEARS,
        "months": MONTHS,
        "vessel_types": VESSEL_TYPES,
        "reader_used": READER_USED,
        "sampling_method": SAMPLING_METHOD,
        "cell_count": int(qa["project_ocean_cell_count"].iloc[0]),
        "zero_fraction": params["zero_fraction"],
        "raw_q95": params["raw_q95"],
        "raw_q99": params["raw_q99"],
        "log_q95": params["log_q95"],
        "log_q99": params["log_q99"],
        "primary_scaling_choice": PRIMARY_SCALING,
        "monthly_file_count": int(len(TARGET_YEARS) * len(MONTHS) * len(VESSEL_TYPES)),
        "missing_file_count": int(len(missing_files)),
        "missing_files": missing_files,
        "missing_sample_count": int(qa["nodata_count"].sum()),
        "coverage_percent": float(qa["sampled_cell_count"].sum() / (qa["project_ocean_cell_count"].sum()) * 100),
        "nodata_treatment": NODATA_TREATMENT,
        "figures": figure_paths,
        "warnings": [
            "Ship layer is a vessel disturbance / noise-related movement impedance proxy, not ship-strike mortality.",
            "Centroid nearest-cell sampling is a v0 grid harmonization method; area-weighted remapping can be added later if ship conclusions become central.",
        ],
    }
    QA_JSON.write_text(json.dumps(qa_data, indent=2))
    return qa_data


def append_manifest(paths: list[str], figure_paths: list[str]) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_docs(params: dict, monthly_sum: pd.DataFrame, mayjul_sum: pd.DataFrame, contrib_mj: pd.DataFrame, overlap_mj: pd.DataFrame, class_summary: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_registry_and_claims() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def main() -> None:
    ensure_dirs()
    grid = load_project_ocean_grid()
    xs, ys = transform_ocean_points(grid)
    monthly_raw, harmonization_qa = build_monthly_surfaces(grid, xs, ys)
    params = compute_scaling(monthly_raw)
    monthly = write_monthly(monthly_raw, params)
    monthly_sum = monthly_summary(monthly)
    mayjul, mayjul_sum = build_mayjul(monthly)
    contrib, contrib_mj = vessel_contribution(monthly)
    overlap_monthly, overlap_mj = corridor_overlap(monthly, mayjul)
    class_overlap_table, class_summary = class_overlap(mayjul)
    figure_paths = create_figures(monthly, mayjul, monthly_sum, mayjul_sum, contrib, overlap_mj, class_summary, params, harmonization_qa)
    qa = qa_json(params, harmonization_qa, figure_paths)

    output_paths = [
        *(f"outputs/ship/ship_intensity_target{year}_month{month:02d}.csv" for year in TARGET_YEARS for month in MONTHS),
        *(f"outputs/ship/ship_intensity_MayJul_target{year}.csv" for year in TARGET_YEARS),
        "outputs/ship/ship_grid_harmonization_qa_v0.csv",
        "outputs/ship/ship_scaling_parameters_v0.json",
        "outputs/ship/ship_monthly_summary_v0.csv",
        "outputs/ship/ship_MayJul_summary_v0.csv",
        "outputs/ship/ship_vessel_type_contribution_v0.csv",
        "outputs/ship/ship_vessel_type_contribution_MayJul_v0.csv",
        "outputs/ship/ship_overlap_with_prey_ice_corridor_v0.csv",
        "outputs/ship/ship_overlap_with_prey_ice_corridor_MayJul_v0.csv",
        "outputs/ship/ship_x_prey_quality_bottleneck_overlap_v0.csv",
        "outputs/ship/ship_x_prey_quality_bottleneck_overlap_summary_v0.csv",
        "outputs/ship/ship_disturbance_qa_v0.json",
        "docs/phase_memos/phase5b_ship_disturbance_surfaces.md",
        "docs/paper_use_decisions/phase5b_ship_disturbance_surfaces_paper_use.md",
        "scripts/build_ship_disturbance_surfaces_v0.py",
    ]
    append_manifest(output_paths, figure_paths)
    write_docs(params, monthly_sum, mayjul_sum, contrib_mj, overlap_mj, class_summary)
    write_registry_and_claims()
    print(json.dumps(qa, indent=2))
    print(mayjul_sum.round(4).to_string(index=False))
    print(contrib_mj[["target_year", "cargo_fraction", "fishing_fraction", "other_fraction", "tanker_fraction"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
