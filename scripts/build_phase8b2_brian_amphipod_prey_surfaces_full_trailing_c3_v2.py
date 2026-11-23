"""Phase 8B.2 full trailing-window benthic amphipod prey surfaces on C3.

Builds old-workflow-style trailing 5-year candidate prey surfaces for
2011-2019, using 2007-2010 benthic data where needed for early target years.
No LCP or connectivity models are run.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_phase8b_brian_amphipod_prey_surfaces_c3_v2 as base  # noqa: E402


OUT = ROOT / "outputs" / "prey_v2"
FIG = ROOT / "outputs" / "figures" / "prey_v2"
STUDY = ROOT / "data" / "processed" / "study_domain"
GRID = STUDY / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
C3_DOMAIN = STUDY / "study_domain_C3_recurring_support_v2.geojson"
STATION_TOTALS = OUT / "brian_total_amphipod_station_year_values_v2.csv"
UNIQUE_MATCH = OUT / "brian_amphipod_unique_station_match_results_v2.csv"

TARGET_YEARS = list(range(2011, 2020))
VARIABLES = base.VARIABLES


def clean_key(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def station_key(year, cruise, station_number, station_name) -> str:
    return "|".join([clean_key(year), clean_key(cruise), clean_key(station_number), clean_key(station_name)])


def full_trailing_window_for_target(target_year: int) -> tuple[int, int]:
    return target_year - 4, target_year


def load_full_trailing_inputs():
    grid = pd.read_csv(GRID)
    grid["lon_norm"] = grid["lon"].map(base.normalize_lon)
    grid["x_plot"], grid["y_plot"] = base.project_points(grid["lon_norm"], grid["lat"])
    c3 = base.read_geojson_union(C3_DOMAIN)
    c3 = base.normalize_geom_lons(c3)

    totals = pd.read_csv(STATION_TOTALS)
    unique = pd.read_csv(UNIQUE_MATCH)
    totals["brian_unique_station_key"] = totals.apply(
        lambda r: station_key(r["Year"], r["Cruise"], r["StationNumber"], r["StationName"]), axis=1
    )
    unique["brian_unique_station_key"] = unique.apply(
        lambda r: station_key(
            r["Year_clean"], r["Cruise_clean"], r["StationNumber_clean"], r["StationName_clean"]
        ),
        axis=1,
    )
    stations = totals[totals["Year"].between(2007, 2019)].merge(
        unique[
            [
                "brian_unique_station_key",
                "match_method",
                "number_of_candidate_matches",
                "candidate_cruises",
                "match_status",
                "matched_yes_no",
            ]
        ],
        on="brian_unique_station_key",
        how="left",
    )
    stations["Longitude_norm"] = stations["Longitude"].map(base.normalize_lon)
    stations["has_coordinates"] = stations[["Latitude", "Longitude_norm"]].notna().all(axis=1)
    stations["latest_match_usable"] = (
        stations["matched_yes_no"].astype(bool) & stations["match_status"].isin(["exact", "fallback"])
    )
    stations = stations[stations["has_coordinates"] & stations["latest_match_usable"]].copy()
    stations["inside_C3_domain"] = [
        c3.covers(Point(lon, lat)) for lon, lat in zip(stations["Longitude_norm"], stations["Latitude"])
    ]
    stations = stations[stations["inside_C3_domain"]].copy()

    gx, gy = base.project_points(grid["lon_norm"], grid["lat"])
    tree = cKDTree(np.column_stack([gx, gy]))
    sx, sy = base.project_points(stations["Longitude_norm"], stations["Latitude"])
    dist, idx = tree.query(np.column_stack([sx, sy]), k=1)
    nearest = grid.iloc[idx].reset_index(drop=True)
    stations["nearest_C3_grid_id"] = nearest["grid_id"].to_numpy()
    stations["nearest_C3_grid_lon"] = nearest["lon_norm"].to_numpy()
    stations["nearest_C3_grid_lat"] = nearest["lat"].to_numpy()
    stations["distance_to_nearest_C3_grid_cell_km"] = dist / 1000.0
    for col in [v["station_col"] for v in VARIABLES.values()]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce").clip(lower=0)
    return grid, stations, c3


def build_surfaces(grid, stations):
    original_window = base.window_for_target
    base.window_for_target = full_trailing_window_for_target
    try:
        surfaces, support = base.build_all_surfaces(grid, stations)
    finally:
        base.window_for_target = original_window
    return surfaces, support


def apply_global_scaling_full(surfaces):
    scaling = {
        "normalization_mode": "global_2011_2019_C3_q05_q95_full_trailing_5yr",
        "q99_recorded_for_QA_only": True,
        "variables": {},
    }
    for var_name, yearly in surfaces.items():
        all_values = pd.concat([df["raw_final_value"] for df in yearly.values()], ignore_index=True)
        params = base.robust_params(all_values)
        scaling["variables"][var_name] = params | VARIABLES[var_name]
        for df in yearly.values():
            df["U_amphipod_prey_global_q05_q95"] = base.normalize_with_params(
                df["raw_final_value"], params["q05"], params["q95"]
            )
            df["raw_value_clipped_q05_q95"] = pd.to_numeric(df["raw_final_value"], errors="coerce").clip(
                params["q05"], params["q95"]
            )
    (OUT / "amphipod_prey_surface_C3_v2_full_trailing_scaling_parameters.json").write_text(
        json.dumps(scaling, indent=2), encoding="utf-8"
    )
    return scaling


def write_surfaces_full(surfaces):
    cols = [
        "grid_id",
        "grid_x",
        "grid_y",
        "x",
        "y",
        "lon",
        "lat",
        "target_year",
        "window_start",
        "window_end",
        "prey_variable",
        "observed_value_target_year_mean",
        "target_year_station_count",
        "observed_value_window_mean",
        "observed_value_window_median",
        "support_station_year_count",
        "support_unique_station_count",
        "support_year_count",
        "support_years",
        "raw_idw_value",
        "raw_final_value",
        "raw_value_clipped_q05_q95",
        "U_amphipod_prey_global_q05_q95",
        "observed_target_year_flag",
        "observed_window_flag",
        "interpolated_flag",
        "supported_for_surface",
        "idw_neighbor_count",
        "nearest_observed_distance_km",
        "reliability_flag",
        "ocean_flag",
        "inside_C3_domain",
    ]
    for var_name, yearly in surfaces.items():
        for target, df in yearly.items():
            df[cols].to_csv(
                OUT / f"amphipod_prey_surface_{var_name}_target{target}_full_trailing_C3_v2.csv",
                index=False,
            )


def summarize_full(surfaces, support):
    rows = []
    for var_name, yearly in surfaces.items():
        for target, df in yearly.items():
            u = pd.to_numeric(df["U_amphipod_prey_global_q05_q95"], errors="coerce")
            raw = pd.to_numeric(df["raw_final_value"], errors="coerce")
            rows.append(
                {
                    "target_year": target,
                    "prey_variable": var_name,
                    "window_start": full_trailing_window_for_target(target)[0],
                    "window_end": full_trailing_window_for_target(target)[1],
                    "grid_cell_count": len(df),
                    "supported_cell_count": int(df["supported_for_surface"].sum()),
                    "unsupported_cell_count": int((~df["supported_for_surface"]).sum()),
                    "observed_target_year_cell_count": int(df["observed_target_year_flag"].sum()),
                    "observed_window_cell_count": int(df["observed_window_flag"].sum()),
                    "interpolated_cell_count": int(df["interpolated_flag"].sum()),
                    "percent_supported_cells": float(df["supported_for_surface"].mean() * 100),
                    "percent_interpolated_gt_150km": float(
                        (
                            df.loc[df["interpolated_flag"], "nearest_observed_distance_km"]
                            > base.IDW_LONG_DISTANCE_QA_KM
                        ).mean()
                        * 100
                    )
                    if df["interpolated_flag"].any()
                    else math.nan,
                    "mean_raw_final_value": float(raw.mean(skipna=True)),
                    "median_raw_final_value": float(raw.median(skipna=True)),
                    "max_raw_final_value": float(raw.max(skipna=True)),
                    "mean_U": float(u.mean(skipna=True)),
                    "median_U": float(u.median(skipna=True)),
                    "max_U": float(u.max(skipna=True)),
                    "mean_nearest_observed_distance_km": float(df["nearest_observed_distance_km"].mean(skipna=True)),
                    "median_nearest_observed_distance_km": float(df["nearest_observed_distance_km"].median(skipna=True)),
                }
            )
    summary = pd.DataFrame(rows).merge(
        support, on=["target_year", "prey_variable", "window_start", "window_end"], how="left"
    )
    summary.to_csv(OUT / "amphipod_prey_surface_C3_v2_full_trailing_summary_by_year.csv", index=False)
    return summary


def compare_to_contemporary(surfaces):
    rows = []
    top_rows = []
    for var_name in VARIABLES:
        for target in TARGET_YEARS:
            full = surfaces[var_name][target][
                ["grid_id", "raw_final_value", "U_amphipod_prey_global_q05_q95", "reliability_flag"]
            ].rename(
                columns={
                    "raw_final_value": "raw_full_trailing",
                    "U_amphipod_prey_global_q05_q95": "U_full_trailing",
                    "reliability_flag": "reliability_full_trailing",
                }
            )
            contemporary_path = OUT / f"amphipod_prey_surface_{var_name}_target{target}_C3_v2.csv"
            contemporary = pd.read_csv(
                contemporary_path,
                usecols=["grid_id", "raw_final_value", "U_amphipod_prey_global_q05_q95", "reliability_flag"],
            ).rename(
                columns={
                    "raw_final_value": "raw_contemporary_only",
                    "U_amphipod_prey_global_q05_q95": "U_contemporary_only",
                    "reliability_flag": "reliability_contemporary_only",
                }
            )
            m = full.merge(contemporary, on="grid_id", how="inner").dropna(
                subset=["U_full_trailing", "U_contemporary_only"]
            )
            rows.append(
                {
                    "target_year": target,
                    "prey_variable": var_name,
                    "cell_count": len(m),
                    "pearson_U_full_vs_contemporary": float(m["U_full_trailing"].corr(m["U_contemporary_only"])),
                    "pearson_raw_full_vs_contemporary": float(m["raw_full_trailing"].corr(m["raw_contemporary_only"])),
                    "mean_U_full_trailing": float(m["U_full_trailing"].mean()),
                    "mean_U_contemporary_only": float(m["U_contemporary_only"].mean()),
                    "mean_delta_U_full_minus_contemporary": float(
                        (m["U_full_trailing"] - m["U_contemporary_only"]).mean()
                    ),
                    "mean_abs_delta_U": float((m["U_full_trailing"] - m["U_contemporary_only"]).abs().mean()),
                    "median_abs_delta_U": float((m["U_full_trailing"] - m["U_contemporary_only"]).abs().median()),
                    "cells_abs_delta_U_gt_0_10": int(
                        ((m["U_full_trailing"] - m["U_contemporary_only"]).abs() > 0.10).sum()
                    ),
                    "cells_abs_delta_U_gt_0_25": int(
                        ((m["U_full_trailing"] - m["U_contemporary_only"]).abs() > 0.25).sum()
                    ),
                }
            )
            for q, label in [(0.90, "top10pct"), (0.95, "top5pct")]:
                ft_thr = m["U_full_trailing"].quantile(q)
                co_thr = m["U_contemporary_only"].quantile(q)
                ft_set = set(m.loc[m["U_full_trailing"] >= ft_thr, "grid_id"])
                co_set = set(m.loc[m["U_contemporary_only"] >= co_thr, "grid_id"])
                union = len(ft_set | co_set)
                top_rows.append(
                    {
                        "target_year": target,
                        "prey_variable": var_name,
                        "threshold": label,
                        "intersection_cell_count": len(ft_set & co_set),
                        "union_cell_count": union,
                        "jaccard_overlap": len(ft_set & co_set) / union if union else math.nan,
                    }
                )
    comp = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    comp.to_csv(OUT / "amphipod_full_trailing_vs_contemporary_C3_comparison_v2.csv", index=False)
    top.to_csv(OUT / "amphipod_full_trailing_vs_contemporary_C3_top_cell_overlap_v2.csv", index=False)
    return comp, top


def compare_variables_full(surfaces):
    rows = []
    overlap_rows = []
    for target in TARGET_YEARS:
        merged = None
        for var_name in VARIABLES:
            sub = surfaces[var_name][target][["grid_id", "U_amphipod_prey_global_q05_q95"]].rename(
                columns={"U_amphipod_prey_global_q05_q95": f"U_{var_name}"}
            )
            merged = sub if merged is None else merged.merge(sub, on="grid_id", how="outer")
        for a, b in [("total_kJm2_frozen", "total_gww"), ("total_kJm2_frozen", "total_gC"), ("total_gww", "total_gC")]:
            pair = merged[[f"U_{a}", f"U_{b}"]].dropna()
            rows.append(
                {
                    "target_year": target,
                    "variable_a": a,
                    "variable_b": b,
                    "pearson_correlation": float(pair[f"U_{a}"].corr(pair[f"U_{b}"])) if len(pair) > 1 else math.nan,
                    "cell_count": int(len(pair)),
                }
            )
        for q, label in [(0.90, "top10pct"), (0.95, "top5pct")]:
            sets = {}
            for var in VARIABLES:
                vals = merged[["grid_id", f"U_{var}"]].dropna()
                thr = vals[f"U_{var}"].quantile(q)
                sets[var] = set(vals.loc[vals[f"U_{var}"] >= thr, "grid_id"])
            for a, b in [("total_kJm2_frozen", "total_gww"), ("total_kJm2_frozen", "total_gC"), ("total_gww", "total_gC")]:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                overlap_rows.append(
                    {
                        "target_year": target,
                        "threshold": label,
                        "variable_a": a,
                        "variable_b": b,
                        "intersection_cell_count": inter,
                        "union_cell_count": union,
                        "jaccard_overlap": inter / union if union else math.nan,
                    }
                )
    corr = pd.DataFrame(rows)
    top = pd.DataFrame(overlap_rows)
    corr.to_csv(OUT / "amphipod_prey_variable_correlation_full_trailing_C3_v2.csv", index=False)
    top.to_csv(OUT / "amphipod_prey_variable_top_cell_overlap_full_trailing_C3_v2.csv", index=False)
    return corr, top


def plot_panel_full(surfaces, var_name, c3, land):
    bounds = base.padded_bounds_from_geom(c3)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.5), dpi=220)
    for ax, target in zip(axes.flat, TARGET_YEARS):
        df = surfaces[var_name][target]
        ax.set_facecolor("#eaf4f7")
        base.draw_land(ax, land)
        sc = ax.scatter(
            df["x_plot"],
            df["y_plot"],
            c=df["U_amphipod_prey_global_q05_q95"],
            s=5,
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidth=0,
            zorder=3,
        )
        base.draw_domain(ax, c3)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(target), fontsize=9)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.74, pad=0.02)
    cbar.set_label("Global q05-q95 scaled amphipod prey utility")
    fig.suptitle(
        f"C3 full trailing 5-year benthic amphipod surface: {VARIABLES[var_name]['pretty']}",
        fontsize=12,
        y=0.98,
    )
    fig.text(0.5, 0.02, "Diagnostic IDW surfaces; early windows include 2007-2010 support.", ha="center", fontsize=8)
    fig.savefig(FIG / f"amphipod_prey_C3_{var_name}_full_trailing_all_years_v2.png", bbox_inches="tight")
    plt.close(fig)


def plot_comparison_figures(comp):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=220)
    for var, sub in comp.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["mean_abs_delta_U"], marker="o", label=var)
    ax.set_xlabel("Target year")
    ax.set_ylabel("Mean absolute delta U")
    ax.set_title("Full trailing vs contemporary-only C3 prey surfaces")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "amphipod_full_trailing_vs_contemporary_mean_abs_delta_U_v2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=220)
    for var, sub in comp.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["pearson_U_full_vs_contemporary"], marker="o", label=var)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Target year")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Correlation: full trailing vs contemporary-only")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "amphipod_full_trailing_vs_contemporary_correlation_v2.png", bbox_inches="tight")
    plt.close(fig)


def write_qa(summary, support, scaling, comp, var_corr):
    qa = {
        "phase": "Phase 8B.2 full trailing C3 benthic amphipod prey surfaces",
        "no_lcp_models_run": True,
        "no_connectivity_models_run": True,
        "target_years": TARGET_YEARS,
        "window_rule": "target_year-4 through target_year",
        "variables": VARIABLES,
        "normalization_mode": scaling["normalization_mode"],
        "idw": {
            "k": base.IDW_K,
            "power": base.IDW_POWER,
            "max_distance_km": base.IDW_MAX_DISTANCE_KM,
            "long_distance_QA_km": base.IDW_LONG_DISTANCE_QA_KM,
        },
        "summary": summary.to_dict(orient="records"),
        "support": support.to_dict(orient="records"),
        "comparison_with_contemporary_only": comp.to_dict(orient="records"),
        "variable_correlations": var_corr.to_dict(orient="records"),
        "warnings": [
            "Full trailing early windows include 2007-2010 benthic records.",
            "This phase builds prey surfaces only and does not run LCP/connectivity models.",
            "Use as sensitivity or replacement only after reviewing comparison with contemporary-only Phase 8B surfaces.",
        ],
    }
    (OUT / "amphipod_prey_surface_C3_v2_full_trailing_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")


def main():
    grid, stations, c3 = load_full_trailing_inputs()
    surfaces, support = build_surfaces(grid, stations)
    scaling = apply_global_scaling_full(surfaces)
    write_surfaces_full(surfaces)
    summary = summarize_full(surfaces, support)
    comp, top = compare_to_contemporary(surfaces)
    var_corr, var_top = compare_variables_full(surfaces)
    land = base.read_polygon_shapefile(base.LAND_SHP)
    for var in VARIABLES:
        plot_panel_full(surfaces, var, c3, land)
    plot_comparison_figures(comp)
    write_qa(summary, support, scaling, comp, var_corr)
    print(
        json.dumps(
            {
                "phase": "Phase 8B.2",
                "surfaces_written": len(TARGET_YEARS) * len(VARIABLES),
                "station_years_inside_C3_used_2007_2019": int(len(stations)),
                "no_lcp_models_run": True,
                "summary": str(OUT / "amphipod_prey_surface_C3_v2_full_trailing_summary_by_year.csv"),
                "comparison": str(OUT / "amphipod_full_trailing_vs_contemporary_C3_comparison_v2.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
