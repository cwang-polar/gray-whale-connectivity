"""Phase 8B benthic amphipod prey surfaces on final C3 v2 grid.

Builds candidate 2011-2019 amphipod prey surfaces for kJm2.frozen, gWW, and
gC on the selected C3 v2 ocean grid. This script does not run LCP or
connectivity models.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "prey_v2"
FIG = ROOT / "outputs" / "figures" / "prey_v2"
STUDY = ROOT / "data" / "processed" / "study_domain"
GRID = STUDY / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
STATION_COVERAGE = OUT / "brian_station_coverage_on_C3_v2_grid.csv"
C3_DOMAIN = STUDY / "study_domain_C3_recurring_support_v2.geojson"
LAND_SHP = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

TARGET_YEARS = list(range(2011, 2020))
WINDOW_YEARS = 5
IDW_K = 8
IDW_POWER = 2.0
IDW_MAX_DISTANCE_KM = 250.0
IDW_LONG_DISTANCE_QA_KM = 150.0

CRS_WGS84 = "EPSG:4326"
CRS_WORK = "+proj=laea +lat_0=68 +lon_0=-170 +datum=WGS84 +units=m +no_defs"
CRS_WORK_LABEL = "Pacific Arctic LAEA (lon0=-170, lat0=68)"
TO_WORK = Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform

VARIABLES = {
    "total_kJm2_frozen": {
        "station_col": "total_amphipod_kJm2_frozen",
        "pretty": "Amphipod energy density (kJ m-2 frozen)",
        "unit": "kJ m-2",
    },
    "total_gww": {
        "station_col": "total_amphipod_gww",
        "pretty": "Amphipod wet-weight biomass (gWW)",
        "unit": "g wet weight",
    },
    "total_gC": {
        "station_col": "total_amphipod_gC",
        "pretty": "Amphipod carbon biomass (gC)",
        "unit": "g C",
    },
}


@dataclass(frozen=True)
class LandFeature:
    bbox: tuple[float, float, float, float]
    rings: list[list[tuple[float, float]]]


def normalize_lon(lon: float) -> float:
    if pd.isna(lon):
        return math.nan
    lon = float(lon)
    while lon > 180:
        lon -= 360
    while lon <= -180:
        lon += 360
    if lon > 0:
        lon -= 360
    return lon


def normalize_geom_lons(geom):
    def _norm(x, y, z=None):
        arr = np.asarray(x, dtype=float)
        arr = np.where(arr > 0, arr - 360.0, arr)
        return (arr, y) if z is None else (arr, y, z)

    return transform(_norm, geom)


def to_work_geom(geom):
    return transform(TO_WORK, normalize_geom_lons(geom))


def project_points(lon, lat):
    return TO_WORK(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def robust_params(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {"q05": math.nan, "q95": math.nan, "q99_QA": math.nan, "count": 0}
    return {
        "q05": float(numeric.quantile(0.05)),
        "q95": float(numeric.quantile(0.95)),
        "q99_QA": float(numeric.quantile(0.99)),
        "count": int(len(numeric)),
    }


def normalize_with_params(values: pd.Series, q05: float, q95: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = numeric.notna() & np.isfinite(numeric)
    if not ok.any() or pd.isna(q05) or pd.isna(q95) or math.isclose(q05, q95):
        return out
    clipped = numeric.loc[ok].clip(lower=q05, upper=q95)
    out.loc[ok] = ((clipped - q05) / (q95 - q05)).clip(lower=0.0, upper=1.0)
    return out


def read_geojson_union(path: Path):
    data = json.loads(path.read_text())
    return unary_union([shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")])


def read_polygon_shapefile(path: Path) -> list[LandFeature]:
    if not path.exists():
        return []
    features: list[LandFeature] = []
    with path.open("rb") as file:
        header = file.read(100)
        shape_type = struct.unpack("<i", header[32:36])[0]
        if shape_type not in {5, 15, 25, 31}:
            return []
        while True:
            record_header = file.read(8)
            if not record_header:
                break
            content_length_words = struct.unpack(">i", record_header[4:8])[0]
            content = file.read(content_length_words * 2)
            if len(content) < 44:
                continue
            rec_shape_type = struct.unpack("<i", content[:4])[0]
            if rec_shape_type == 0 or rec_shape_type not in {5, 15, 25, 31}:
                continue
            xmin, ymin, xmax, ymax = struct.unpack("<dddd", content[4:36])
            num_parts, num_points = struct.unpack("<ii", content[36:44])
            parts_start = 44
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_start : parts_start + 4 * num_parts]))
            points_start = parts_start + 4 * num_parts
            points = [
                struct.unpack("<dd", content[points_start + i * 16 : points_start + (i + 1) * 16])
                for i in range(num_points)
            ]
            rings = []
            for idx, start in enumerate(parts):
                end = parts[idx + 1] if idx + 1 < len(parts) else num_points
                ring = points[start:end]
                if len(ring) >= 3:
                    rings.append(ring)
            features.append(LandFeature(bbox=(xmin, ymin, xmax, ymax), rings=rings))
    return features


def downsample_ring(ring: list[tuple[float, float]], max_points: int = 1000) -> list[tuple[float, float]]:
    if len(ring) <= max_points:
        return ring
    step = math.ceil(len(ring) / max_points)
    out = ring[::step]
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def draw_land(ax, land_features, lon_bounds=(-181, -150), lat_bounds=(58, 76)):
    for feature in land_features:
        lons = [normalize_lon(feature.bbox[0]), normalize_lon(feature.bbox[2])]
        if max(lons) < lon_bounds[0] or min(lons) > lon_bounds[1]:
            continue
        if feature.bbox[3] < lat_bounds[0] or feature.bbox[1] > lat_bounds[1]:
            continue
        for ring in feature.rings:
            rr = [(normalize_lon(lon), lat) for lon, lat in downsample_ring(ring)]
            xs_lon = [p[0] for p in rr]
            ys_lat = [p[1] for p in rr]
            if not xs_lon or max(xs_lon) < lon_bounds[0] or min(xs_lon) > lon_bounds[1]:
                continue
            if max(ys_lat) < lat_bounds[0] or min(ys_lat) > lat_bounds[1]:
                continue
            xs, ys = project_points(xs_lon, ys_lat)
            ax.add_patch(
                MplPolygon(
                    np.column_stack([xs, ys]),
                    closed=True,
                    facecolor="#d6d1c8",
                    edgecolor="#8f887b",
                    linewidth=0.2,
                    alpha=0.95,
                    zorder=1,
                )
            )


def draw_domain(ax, geom, color="#003f5c", lw=1.4):
    g = to_work_geom(geom)
    geoms = list(g.geoms) if hasattr(g, "geoms") else [g]
    for gg in geoms:
        if gg.geom_type == "Polygon":
            x, y = gg.exterior.xy
            ax.plot(x, y, color=color, lw=lw, zorder=5)


def padded_bounds_from_geom(geom, frac=0.06):
    minx, miny, maxx, maxy = to_work_geom(geom).bounds
    dx = maxx - minx
    dy = maxy - miny
    return minx - dx * frac, miny - dy * frac, maxx + dx * frac, maxy + dy * frac


def prepare_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(GRID)
    grid["lon_norm"] = grid["lon"].map(normalize_lon)
    grid["x_plot"], grid["y_plot"] = project_points(grid["lon_norm"], grid["lat"])
    stations = pd.read_csv(STATION_COVERAGE)
    stations = stations[stations["has_usable_coordinates"].astype(bool) & stations["inside_C3_domain"].astype(bool)].copy()
    stations["Longitude_norm"] = stations["Longitude"].map(normalize_lon)
    stations["x_plot"], stations["y_plot"] = project_points(stations["Longitude_norm"], stations["Latitude"])
    for col in [v["station_col"] for v in VARIABLES.values()]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce").clip(lower=0)
    c3 = read_geojson_union(C3_DOMAIN)
    return grid, stations, c3


def window_for_target(target_year: int) -> tuple[int, int]:
    # Keep the main contemporary analysis strictly within 2011-2019.
    return max(2011, target_year - WINDOW_YEARS + 1), target_year


def aggregate_observed_cells(stations: pd.DataFrame, var_col: str, target_year: int) -> pd.DataFrame:
    start, end = window_for_target(target_year)
    window = stations[stations["Year"].between(start, end)].copy()
    target = stations[stations["Year"].eq(target_year)].copy()
    if window.empty:
        return pd.DataFrame()
    grouped = (
        window.groupby("nearest_C3_grid_id")
        .agg(
            observed_value_window_mean=(var_col, "mean"),
            observed_value_window_median=(var_col, "median"),
            support_station_year_count=(var_col, "size"),
            support_unique_station_count=("brian_unique_station_key", "nunique"),
            support_year_count=("Year", "nunique"),
            support_years=("Year", lambda s: ",".join(str(int(y)) for y in sorted(set(s)))),
        )
        .reset_index()
    )
    if not target.empty:
        tg = (
            target.groupby("nearest_C3_grid_id")
            .agg(
                observed_value_target_year_mean=(var_col, "mean"),
                target_year_station_count=(var_col, "size"),
            )
            .reset_index()
        )
        grouped = grouped.merge(tg, on="nearest_C3_grid_id", how="left")
    else:
        grouped["observed_value_target_year_mean"] = np.nan
        grouped["target_year_station_count"] = 0
    grouped = grouped.rename(columns={"nearest_C3_grid_id": "grid_id"})
    return grouped


def build_idw_surface(grid: pd.DataFrame, observed: pd.DataFrame, target_year: int, variable: str) -> pd.DataFrame:
    start, end = window_for_target(target_year)
    result = grid.copy()
    result["target_year"] = target_year
    result["window_start"] = start
    result["window_end"] = end
    result["prey_variable"] = variable
    result = result.merge(observed, on="grid_id", how="left")
    for col in [
        "observed_value_window_mean",
        "observed_value_window_median",
        "support_station_year_count",
        "support_unique_station_count",
        "support_year_count",
        "support_years",
        "observed_value_target_year_mean",
        "target_year_station_count",
    ]:
        if col not in result.columns:
            result[col] = np.nan
    result["observed_window_flag"] = result["observed_value_window_mean"].notna()
    result["observed_target_year_flag"] = result["observed_value_target_year_mean"].notna()
    result["raw_idw_value"] = np.nan
    result["raw_final_value"] = result["observed_value_window_mean"]
    result["idw_neighbor_count"] = 0
    result["nearest_observed_distance_km"] = np.nan

    obs = observed.dropna(subset=["observed_value_window_mean"]).copy()
    if obs.empty:
        result["reliability_flag"] = "unsupported_no_observations"
        return result

    obs_xy = result.loc[result["observed_window_flag"], ["x", "y", "observed_value_window_mean"]].dropna()
    if obs_xy.empty:
        result["reliability_flag"] = "unsupported_no_observed_cells"
        return result
    tree = cKDTree(obs_xy[["x", "y"]].to_numpy(dtype=float))
    all_xy = result[["x", "y"]].to_numpy(dtype=float)
    obs_values = obs_xy["observed_value_window_mean"].to_numpy(dtype=float)

    k_query = min(IDW_K, len(obs_xy))
    distances, indices = tree.query(all_xy, k=k_query)
    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    result["nearest_observed_distance_km"] = distances[:, 0] / 1000.0

    for idx in range(len(result)):
        if result.at[idx, "observed_window_flag"]:
            result.at[idx, "idw_neighbor_count"] = int((distances[idx] == 0).sum())
            continue
        within_mask = distances[idx] <= (IDW_MAX_DISTANCE_KM * 1000.0)
        if not within_mask.any():
            continue
        use_dist = distances[idx][within_mask]
        use_idx = indices[idx][within_mask]
        use_vals = obs_values[use_idx]
        exact = use_dist == 0
        if exact.any():
            val = float(np.mean(use_vals[exact]))
        else:
            weights = 1.0 / np.power(use_dist, IDW_POWER)
            val = float(np.sum(weights * use_vals) / np.sum(weights))
        result.at[idx, "raw_idw_value"] = val
        result.at[idx, "raw_final_value"] = val
        result.at[idx, "idw_neighbor_count"] = int(len(use_vals))

    result["interpolated_flag"] = result["raw_idw_value"].notna()
    result["supported_for_surface"] = result["raw_final_value"].notna()
    result["reliability_flag"] = np.select(
        [
            result["observed_target_year_flag"],
            result["observed_window_flag"],
            result["interpolated_flag"] & (result["nearest_observed_distance_km"] <= IDW_LONG_DISTANCE_QA_KM),
            result["interpolated_flag"] & (result["nearest_observed_distance_km"] > IDW_LONG_DISTANCE_QA_KM),
        ],
        [
            "direct_target_year_observation",
            "rolling_window_observed_cell",
            "idw_supported_within_150km",
            "idw_supported_150_250km",
        ],
        default="unsupported_gt_250km_or_no_support",
    )
    return result


def build_all_surfaces(grid: pd.DataFrame, stations: pd.DataFrame) -> tuple[dict[str, dict[int, pd.DataFrame]], pd.DataFrame]:
    surfaces: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in VARIABLES}
    support_rows = []
    for var_name, meta in VARIABLES.items():
        col = meta["station_col"]
        for target in TARGET_YEARS:
            observed = aggregate_observed_cells(stations, col, target)
            surf = build_idw_surface(grid, observed, target, var_name)
            surfaces[var_name][target] = surf
            support_rows.append(
                {
                    "target_year": target,
                    "prey_variable": var_name,
                    "window_start": window_for_target(target)[0],
                    "window_end": window_for_target(target)[1],
                    "station_year_rows_in_window": int(
                        len(stations[stations["Year"].between(*window_for_target(target))])
                    ),
                    "unique_station_count_in_window": int(
                        stations[stations["Year"].between(*window_for_target(target))]["brian_unique_station_key"].nunique()
                    ),
                    "observed_grid_cells_in_window": int(len(observed)),
                }
            )
    return surfaces, pd.DataFrame(support_rows)


def apply_global_scaling(surfaces: dict[str, dict[int, pd.DataFrame]]) -> dict:
    scaling = {
        "normalization_mode": "global_2011_2019_C3_q05_q95",
        "q99_recorded_for_QA_only": True,
        "variables": {},
    }
    for var_name, yearly in surfaces.items():
        all_values = pd.concat([df["raw_final_value"] for df in yearly.values()], ignore_index=True)
        params = robust_params(all_values)
        scaling["variables"][var_name] = params | VARIABLES[var_name]
        q05, q95 = params["q05"], params["q95"]
        for target, df in yearly.items():
            df["U_amphipod_prey_global_q05_q95"] = normalize_with_params(df["raw_final_value"], q05, q95)
            df["raw_value_clipped_q05_q95"] = pd.to_numeric(df["raw_final_value"], errors="coerce").clip(q05, q95)
    (OUT / "amphipod_prey_surface_C3_v2_scaling_parameters.json").write_text(
        json.dumps(scaling, indent=2), encoding="utf-8"
    )
    return scaling


def write_surfaces(surfaces: dict[str, dict[int, pd.DataFrame]]):
    for var_name, yearly in surfaces.items():
        for target, df in yearly.items():
            out = df[
                [
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
            ].copy()
            out.to_csv(OUT / f"amphipod_prey_surface_{var_name}_target{target}_C3_v2.csv", index=False)


def summarize_surfaces(surfaces: dict[str, dict[int, pd.DataFrame]], support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var_name, yearly in surfaces.items():
        for target, df in yearly.items():
            u = pd.to_numeric(df["U_amphipod_prey_global_q05_q95"], errors="coerce")
            raw = pd.to_numeric(df["raw_final_value"], errors="coerce")
            rows.append(
                {
                    "target_year": target,
                    "prey_variable": var_name,
                    "window_start": window_for_target(target)[0],
                    "window_end": window_for_target(target)[1],
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
                            > IDW_LONG_DISTANCE_QA_KM
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
    summary = pd.DataFrame(rows).merge(support, on=["target_year", "prey_variable", "window_start", "window_end"], how="left")
    summary.to_csv(OUT / "amphipod_prey_surface_C3_v2_summary_by_year.csv", index=False)
    return summary


def compare_variables(surfaces: dict[str, dict[int, pd.DataFrame]]):
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
    comp = pd.DataFrame(rows)
    top = pd.DataFrame(overlap_rows)
    comp.to_csv(OUT / "amphipod_prey_variable_correlation_C3_v2.csv", index=False)
    top.to_csv(OUT / "amphipod_prey_variable_top_cell_overlap_C3_v2.csv", index=False)
    return comp, top


def old_proxy_path(target: int) -> Path | None:
    start = target - 4
    name = f"prey_proxy_grid_idw_global_rolling5_{start}_{target}_target{target}.csv"
    path = ROOT / "outputs" / name
    return path if path.exists() else None


def compare_old_proxy(surfaces: dict[str, dict[int, pd.DataFrame]]):
    rows = []
    overlap = []
    for target in TARGET_YEARS:
        path = old_proxy_path(target)
        if path is None:
            continue
        old = pd.read_csv(path, usecols=["grid_id", "combined_prey_proxy_final_global"])
        for var in VARIABLES:
            new = surfaces[var][target][["grid_id", "U_amphipod_prey_global_q05_q95"]].dropna()
            m = new.merge(old, on="grid_id", how="inner").dropna()
            rows.append(
                {
                    "target_year": target,
                    "prey_variable": var,
                    "old_proxy_file": str(path),
                    "cell_count": len(m),
                    "pearson_correlation_with_old_proxy": float(
                        m["U_amphipod_prey_global_q05_q95"].corr(m["combined_prey_proxy_final_global"])
                    )
                    if len(m) > 1
                    else math.nan,
                    "mean_new_U": float(m["U_amphipod_prey_global_q05_q95"].mean()),
                    "mean_old_proxy": float(m["combined_prey_proxy_final_global"].mean()),
                }
            )
            for q, label in [(0.90, "top10pct"), (0.95, "top5pct")]:
                new_thr = m["U_amphipod_prey_global_q05_q95"].quantile(q)
                old_thr = m["combined_prey_proxy_final_global"].quantile(q)
                new_set = set(m.loc[m["U_amphipod_prey_global_q05_q95"] >= new_thr, "grid_id"])
                old_set = set(m.loc[m["combined_prey_proxy_final_global"] >= old_thr, "grid_id"])
                union = len(new_set | old_set)
                overlap.append(
                    {
                        "target_year": target,
                        "prey_variable": var,
                        "threshold": label,
                        "intersection_cell_count": len(new_set & old_set),
                        "union_cell_count": union,
                        "jaccard_overlap_with_old_proxy": len(new_set & old_set) / union if union else math.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "amphipod_vs_old_proxy_C3_surface_comparison_v2.csv", index=False)
    pd.DataFrame(overlap).to_csv(OUT / "amphipod_vs_old_proxy_C3_top_cell_overlap_v2.csv", index=False)


def plot_panel_for_variable(surfaces, var_name: str, c3, land):
    vals = pd.concat([df["U_amphipod_prey_global_q05_q95"] for df in surfaces[var_name].values()])
    vmax = 1.0
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.5), dpi=220)
    meta = VARIABLES[var_name]
    bounds = padded_bounds_from_geom(c3)
    for ax, target in zip(axes.flat, TARGET_YEARS):
        df = surfaces[var_name][target]
        ax.set_facecolor("#eaf4f7")
        draw_land(ax, land)
        sc = ax.scatter(
            df["x_plot"],
            df["y_plot"],
            c=df["U_amphipod_prey_global_q05_q95"],
            s=5,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            linewidth=0,
            zorder=3,
        )
        draw_domain(ax, c3)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(str(target), fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.74, pad=0.02)
    cbar.set_label("Global q05-q95 scaled amphipod prey utility")
    fig.suptitle(f"C3 benthic amphipod candidate surface: {meta['pretty']}", fontsize=12, y=0.98)
    fig.text(0.5, 0.02, "Diagnostic IDW surfaces; early rolling windows are truncated to 2011 start.", ha="center", fontsize=8)
    fig.savefig(FIG / f"amphipod_prey_C3_{var_name}_all_years_v2.png", bbox_inches="tight")
    plt.close(fig)


def plot_support_figures(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=220)
    for var, sub in summary.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["percent_supported_cells"], marker="o", label=var)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Target year")
    ax.set_ylabel("C3 grid cells supported (%)")
    ax.set_title("C3 benthic amphipod IDW support by candidate variable")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "amphipod_prey_C3_surface_support_by_year_v2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=220)
    for var, sub in summary.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["mean_U"], marker="o", label=var)
    ax.set_xlabel("Target year")
    ax.set_ylabel("Mean scaled prey utility")
    ax.set_title("Mean C3 amphipod prey utility by variable")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "amphipod_prey_C3_mean_utility_by_variable_v2.png", bbox_inches="tight")
    plt.close(fig)


def write_qa(summary, support, scaling, comp, top):
    warnings = [
        "No LCP or connectivity models were run.",
        "No 2010 or pre-2011 historical records were used in main annual prey surfaces.",
        "Early target-year rolling windows are truncated to the 2011 start year.",
        "Unsupported cells retain NaN prey utility and are not treated as true zero prey.",
        "C3 excludes 74 usable benthic 2011-2019 stations from surface construction by domain decision.",
    ]
    qa = {
        "phase": "Phase 8B benthic amphipod C3 prey surfaces",
        "target_years": TARGET_YEARS,
        "variables": VARIABLES,
        "normalization_mode": scaling["normalization_mode"],
        "idw": {
            "k": IDW_K,
            "power": IDW_POWER,
            "max_distance_km": IDW_MAX_DISTANCE_KM,
            "long_distance_QA_km": IDW_LONG_DISTANCE_QA_KM,
        },
        "surface_summary": summary.to_dict(orient="records"),
        "support_summary": support.to_dict(orient="records"),
        "warnings": warnings,
    }
    (OUT / "amphipod_prey_surface_C3_v2_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")


def main():
    grid, stations, c3 = prepare_inputs()
    surfaces, support = build_all_surfaces(grid, stations)
    scaling = apply_global_scaling(surfaces)
    write_surfaces(surfaces)
    summary = summarize_surfaces(surfaces, support)
    comp, top = compare_variables(surfaces)
    compare_old_proxy(surfaces)
    land = read_polygon_shapefile(LAND_SHP)
    for var in VARIABLES:
        plot_panel_for_variable(surfaces, var, c3, land)
    plot_support_figures(summary)
    write_qa(summary, support, scaling, comp, top)
    print(
        json.dumps(
            {
                "phase": "Phase 8B",
                "target_years": TARGET_YEARS,
                "variables": list(VARIABLES),
                "surfaces_written": len(TARGET_YEARS) * len(VARIABLES),
                "no_lcp_models_run": True,
                "no_connectivity_models_run": True,
                "summary": str(OUT / "amphipod_prey_surface_C3_v2_summary_by_year.csv"),
                "qa": str(OUT / "amphipod_prey_surface_C3_v2_qa.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
