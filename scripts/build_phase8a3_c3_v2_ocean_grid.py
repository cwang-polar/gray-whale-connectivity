"""Phase 8A.3 final C3 v2 ocean grid and QA.

Decision/documentation branch only. This script turns the selected C3
recurring-support candidate domain into a v2 ocean analysis grid by subsetting
the existing 10 km ocean grid. It does not build prey surfaces and does not run
LCP/connectivity models.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape, mapping
from shapely.ops import transform, unary_union


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "processed" / "study_domain"
OUT = ROOT / "outputs" / "prey_v2"
FIG = ROOT / "outputs" / "figures" / "prey_v2"

C3_SOURCE = OUT / "candidate_study_region_C_recurring_support_3yr_v2.geojson"
C3_FINAL = STUDY / "study_domain_C3_recurring_support_v2.geojson"
GRID_V0 = STUDY / "ocean_mask_grid_10km_large_regional_bbox_v0.csv"
GRID_C3 = STUDY / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
GRID_C3_POINTS = STUDY / "ocean_mask_grid_10km_C3_recurring_support_v2.geojson"
STATIONS = OUT / "brian_amphipod_station_extent_2011_2019_v2.csv"
LAND_SHP = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

CRS_WGS84 = "EPSG:4326"
CRS_WORK = "+proj=laea +lat_0=68 +lon_0=-170 +datum=WGS84 +units=m +no_defs"
CRS_WORK_LABEL = "Pacific Arctic LAEA (lon0=-170, lat0=68)"
TO_WORK = Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform


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


def read_geojson_union(path: Path):
    data = json.loads(path.read_text())
    geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
    if not geoms:
        raise ValueError(f"No geometries found in {path}")
    return unary_union(geoms)


def write_geojson_feature_collection(path: Path, features: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def geojson_feature(geom, props: dict) -> dict:
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


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
                    linewidth=0.25,
                    alpha=0.95,
                    zorder=1,
                )
            )


def draw_domain(ax, geom, label, color, linewidth=1.8, linestyle="-", zorder=5):
    g = to_work_geom(geom)
    geoms = list(g.geoms) if hasattr(g, "geoms") else [g]
    for gg in geoms:
        if gg.geom_type == "Polygon":
            x, y = gg.exterior.xy
            ax.plot(x, y, color=color, lw=linewidth, ls=linestyle, zorder=zorder, label=label)
            label = None


def add_geo_labels(ax):
    labels = {
        "Chirikov Basin": (-170.5, 63.8),
        "Bering Strait": (-168.5, 66.0),
        "St. Lawrence Island": (-170.1, 63.2),
        "Southern Chukchi": (-168.2, 68.4),
        "Wainwright": (-160.0, 70.8),
        "Barrow Canyon": (-156.3, 72.0),
        "DBO2": (-169.3, 64.8),
        "DBO3": (-169.4, 67.4),
        "DBO4": (-162.0, 71.2),
        "DBO5": (-157.0, 72.35),
    }
    for text, (lon, lat) in labels.items():
        x, y = project_points([lon], [lat])
        ax.text(
            x[0],
            y[0],
            text,
            fontsize=7.3,
            color="#353535",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.2),
        )


def setup_map(ax, bounds, land_features):
    ax.set_facecolor("#eaf4f7")
    draw_land(ax, land_features)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.grid(color="#b8c6cc", linewidth=0.35, alpha=0.55)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(f"x ({CRS_WORK_LABEL})", fontsize=8)
    ax.set_ylabel(f"y ({CRS_WORK_LABEL})", fontsize=8)


def padded_bounds(*frames_or_geoms):
    xs, ys = [], []
    for obj in frames_or_geoms:
        if obj is None:
            continue
        if isinstance(obj, pd.DataFrame):
            if {"x_plot", "y_plot"}.issubset(obj.columns) and len(obj):
                xs.extend([obj["x_plot"].min(), obj["x_plot"].max()])
                ys.extend([obj["y_plot"].min(), obj["y_plot"].max()])
        else:
            b = to_work_geom(obj).bounds
            xs.extend([b[0], b[2]])
            ys.extend([b[1], b[3]])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    dx, dy = maxx - minx, maxy - miny
    return (minx - dx * 0.08, miny - dy * 0.08, maxx + dx * 0.08, maxy + dy * 0.08)


def load_and_prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    c3 = read_geojson_union(C3_SOURCE)
    c3 = normalize_geom_lons(c3)
    if not c3.is_valid:
        c3 = c3.buffer(0)
    write_geojson_feature_collection(
        C3_FINAL,
        [geojson_feature(c3, {"name": "C3_recurring_support_v2", "source": str(C3_SOURCE)})],
    )

    grid = pd.read_csv(GRID_V0)
    grid["lon_norm"] = grid["lon"].map(normalize_lon)
    grid["point_inside_C3"] = [c3.covers(Point(xy)) for xy in zip(grid["lon_norm"], grid["lat"])]
    grid["ocean_flag"] = grid["ocean_flag"].astype(int)
    grid["inside_C3_domain"] = grid["point_inside_C3"]
    c3_grid = grid[grid["ocean_flag"].eq(1) & grid["inside_C3_domain"]].copy()
    c3_grid = c3_grid.drop(columns=["point_inside_C3"])
    c3_grid.to_csv(GRID_C3, index=False)

    features = [
        geojson_feature(
            Point(row["lon_norm"], row["lat"]),
            {
                "grid_id": row["grid_id"],
                "grid_x": int(row["grid_x"]),
                "grid_y": int(row["grid_y"]),
                "ocean_flag": int(row["ocean_flag"]),
            },
        )
        for _, row in c3_grid.iterrows()
    ]
    write_geojson_feature_collection(GRID_C3_POINTS, features)
    return c3, grid, c3_grid


def grid_qa(c3, grid, c3_grid):
    c3_proj = to_work_geom(c3)
    current_ocean = grid[grid["ocean_flag"].eq(1)]
    summary = pd.DataFrame(
        [
            {
                "domain_name": "C3_recurring_support_v2",
                "source_domain_file": str(C3_SOURCE),
                "final_domain_file": str(C3_FINAL),
                "source_grid_file": str(GRID_V0),
                "c3_grid_file": str(GRID_C3),
                "c3_polygon_valid": bool(c3.is_valid),
                "c3_polygon_area_km2": c3_proj.area / 1_000_000,
                "current_ocean_grid_cell_count": int(len(current_ocean)),
                "c3_ocean_grid_cell_count": int(len(c3_grid)),
                "c3_grid_area_approx_km2": float(len(c3_grid) * 100),
                "percent_current_ocean_cells_retained": float(len(c3_grid) / len(current_ocean) * 100),
                "min_lon": float(c3_grid["lon_norm"].min()),
                "max_lon": float(c3_grid["lon_norm"].max()),
                "min_lat": float(c3_grid["lat"].min()),
                "max_lat": float(c3_grid["lat"].max()),
            }
        ]
    )
    summary.to_csv(OUT / "C3_v2_grid_qa_summary.csv", index=False)
    qa = summary.iloc[0].to_dict()
    qa["notes"] = [
        "C3 grid was created by subsetting existing 10 km ocean grid centroids to the selected C3 recurring-support polygon.",
        "Approximate grid area uses 100 km2 per retained 10 km cell.",
        "No prey surfaces or LCP models were run.",
    ]
    (OUT / "C3_v2_grid_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return summary


def station_coverage(c3, c3_grid):
    stations = pd.read_csv(STATIONS)
    stations["Longitude_norm"] = stations["Longitude"].map(normalize_lon)
    stations["has_usable_coordinates"] = stations["has_usable_coordinates"].astype(bool)
    usable = stations[stations["has_usable_coordinates"]].copy()
    usable["inside_C3_domain"] = [c3.covers(Point(xy)) for xy in zip(usable["Longitude_norm"], usable["Latitude"])]

    gx, gy = project_points(c3_grid["lon_norm"], c3_grid["lat"])
    tree = cKDTree(np.column_stack([gx, gy]))
    sx, sy = project_points(usable["Longitude_norm"], usable["Latitude"])
    dist_m, idx = tree.query(np.column_stack([sx, sy]), k=1)
    nearest = c3_grid.iloc[idx].reset_index(drop=True)
    usable["nearest_C3_grid_id"] = nearest["grid_id"].to_numpy()
    usable["nearest_C3_grid_lon"] = nearest["lon_norm"].to_numpy()
    usable["nearest_C3_grid_lat"] = nearest["lat"].to_numpy()
    usable["distance_to_nearest_C3_grid_cell_km"] = dist_m / 1000
    usable["assignable_to_C3_grid"] = np.isfinite(dist_m)
    usable.to_csv(OUT / "brian_station_coverage_on_C3_v2_grid.csv", index=False)

    by_year = (
        usable.groupby("Year")
        .agg(
            usable_station_count=("brian_unique_station_key", "size"),
            stations_inside_C3=("inside_C3_domain", "sum"),
            stations_outside_C3=("inside_C3_domain", lambda s: int((~s).sum())),
            mean_distance_to_nearest_C3_grid_cell_km=("distance_to_nearest_C3_grid_cell_km", "mean"),
            median_distance_to_nearest_C3_grid_cell_km=("distance_to_nearest_C3_grid_cell_km", "median"),
            max_distance_to_nearest_C3_grid_cell_km=("distance_to_nearest_C3_grid_cell_km", "max"),
        )
        .reset_index()
    )
    by_year["percent_inside_C3"] = by_year["stations_inside_C3"] / by_year["usable_station_count"] * 100
    by_year.to_csv(OUT / "brian_station_coverage_on_C3_v2_grid_by_year.csv", index=False)

    outside = usable[~usable["inside_C3_domain"]].copy()
    outside.to_csv(OUT / "brian_station_outside_C3_v2_domain.csv", index=False)
    return usable, by_year, outside


def layer_compatibility(c3_grid):
    checks = [
        (
            "prey_ice_utility_2011",
            ROOT / "outputs" / "surfaces" / "prey_ice_utility_target2011.csv",
            "existing prey+ice old-domain surface",
        ),
        (
            "prey_ice_utility_2019",
            ROOT / "outputs" / "surfaces" / "prey_ice_utility_target2019.csv",
            "existing prey+ice old-domain surface",
        ),
        (
            "ship_MayJul_2015",
            ROOT / "outputs" / "ship" / "ship_intensity_MayJul_target2015.csv",
            "existing Kapsar ship project-grid surface",
        ),
        (
            "ship_MayJul_2019",
            ROOT / "outputs" / "ship" / "ship_intensity_MayJul_target2019.csv",
            "existing Kapsar ship project-grid surface",
        ),
        (
            "prey_proxy_global_rolling5_2011",
            ROOT / "outputs" / "prey_proxy_grid_idw_global_rolling5_2007_2011_target2011.csv",
            "old global rolling-window prey proxy surface",
        ),
        (
            "prey_proxy_regional_rolling5_2011",
            ROOT / "outputs" / "prey_proxy_grid_idw_rolling5_2007_2011_target2011.csv",
            "old regional rolling-window prey proxy surface",
        ),
    ]
    ids = set(c3_grid["grid_id"].astype(str))
    rows = []
    for name, path, desc in checks:
        row = {
            "layer_name": name,
            "file_path": str(path),
            "description": desc,
            "file_exists": path.exists(),
            "c3_grid_cells": len(ids),
            "matched_c3_grid_ids": math.nan,
            "missing_c3_grid_ids": math.nan,
            "percent_c3_grid_ids_present": math.nan,
            "compatibility_status": "not_checked_missing_file",
        }
        if path.exists():
            try:
                df = pd.read_csv(path, usecols=["grid_id"])
                layer_ids = set(df["grid_id"].astype(str))
                matched = len(ids & layer_ids)
                row.update(
                    {
                        "matched_c3_grid_ids": matched,
                        "missing_c3_grid_ids": len(ids - layer_ids),
                        "percent_c3_grid_ids_present": matched / len(ids) * 100 if ids else math.nan,
                        "compatibility_status": "all_c3_grid_ids_present" if matched == len(ids) else "partial",
                    }
                )
            except Exception as exc:
                row["compatibility_status"] = f"read_error: {exc}"
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "C3_v2_layer_compatibility_qa.csv", index=False)
    return comp


def plot_domain_stations(c3, grid, c3_grid, stations, land_features):
    gplot = c3_grid.copy()
    gplot["x_plot"], gplot["y_plot"] = project_points(gplot["lon_norm"], gplot["lat"])
    s = stations.copy()
    s["x_plot"], s["y_plot"] = project_points(s["Longitude_norm"], s["Latitude"])
    bounds = padded_bounds(gplot, c3)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=220)
    setup_map(ax, bounds, land_features)
    ax.scatter(gplot["x_plot"], gplot["y_plot"], s=4, color="#79a9c5", alpha=0.28, lw=0, label="C3 ocean grid", zorder=3)
    ax.scatter(
        s.loc[s["inside_C3_domain"], "x_plot"],
        s.loc[s["inside_C3_domain"], "y_plot"],
        s=14,
        color="#274c77",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
        label="benthic stations inside C3",
        zorder=6,
    )
    ax.scatter(
        s.loc[~s["inside_C3_domain"], "x_plot"],
        s.loc[~s["inside_C3_domain"], "y_plot"],
        s=18,
        color="#c44536",
        marker="x",
        alpha=0.9,
        label="benthic stations outside C3",
        zorder=7,
    )
    draw_domain(ax, c3, "C3 recurring-support domain", "#003f5c", linewidth=2.2)
    add_geo_labels(ax)
    ax.set_title("C3 v2 prey-inference domain and benthic amphipod stations", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "C3_v2_domain_with_brian_stations.png", bbox_inches="tight")
    plt.close(fig)


def plot_grid_map(c3, c3_grid, land_features):
    gplot = c3_grid.copy()
    gplot["x_plot"], gplot["y_plot"] = project_points(gplot["lon_norm"], gplot["lat"])
    bounds = padded_bounds(gplot, c3)
    fig, ax = plt.subplots(figsize=(8.1, 7), dpi=220)
    setup_map(ax, bounds, land_features)
    ax.scatter(gplot["x_plot"], gplot["y_plot"], s=5, color="#2a9d8f", alpha=0.45, lw=0, label="C3 ocean grid cells", zorder=3)
    draw_domain(ax, c3, "C3 recurring-support domain", "#003f5c", linewidth=2.2)
    add_geo_labels(ax)
    ax.set_title("C3 v2 10 km ocean grid", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "C3_v2_ocean_grid_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_station_coverage_by_year(by_year):
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=220)
    ax.bar(by_year["Year"].astype(str), by_year["percent_inside_C3"], color="#4c78a8", alpha=0.88)
    ax.set_ylim(0, 105)
    ax.set_ylabel("benthic stations inside C3 (%)")
    ax.set_xlabel("Target year")
    ax.set_title("benthic amphipod station coverage inside C3 by year", fontsize=11)
    for _, row in by_year.iterrows():
        ax.text(
            str(int(row["Year"])),
            row["percent_inside_C3"] + 1.5,
            f"{int(row['stations_inside_C3'])}/{int(row['usable_station_count'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "C3_v2_station_coverage_by_year.png", bbox_inches="tight")
    plt.close(fig)


def plot_vs_old_domain(c3, grid, c3_grid, land_features):
    ocean = grid[grid["ocean_flag"].eq(1)].copy()
    ocean["x_plot"], ocean["y_plot"] = project_points(ocean["lon_norm"], ocean["lat"])
    c3p = c3_grid.copy()
    c3p["x_plot"], c3p["y_plot"] = project_points(c3p["lon_norm"], c3p["lat"])
    bounds = padded_bounds(ocean, c3)
    fig, ax = plt.subplots(figsize=(8.2, 7.1), dpi=220)
    setup_map(ax, bounds, land_features)
    ax.scatter(ocean["x_plot"], ocean["y_plot"], s=2.8, color="#b7c6cc", alpha=0.25, lw=0, label="current ocean grid", zorder=2)
    ax.scatter(c3p["x_plot"], c3p["y_plot"], s=4.5, color="#2a9d8f", alpha=0.6, lw=0, label="C3 retained ocean grid", zorder=4)
    draw_domain(ax, c3, "C3 recurring-support domain", "#003f5c", linewidth=2.2)
    add_geo_labels(ax)
    ax.set_title("C3 v2 grid subset relative to current ocean grid", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "C3_v2_vs_old_domain.png", bbox_inches="tight")
    plt.close(fig)


def main():
    c3, grid, c3_grid = load_and_prepare()
    summary = grid_qa(c3, grid, c3_grid)
    stations, by_year, outside = station_coverage(c3, c3_grid)
    comp = layer_compatibility(c3_grid)
    land = read_polygon_shapefile(LAND_SHP)
    plot_domain_stations(c3, grid, c3_grid, stations, land)
    plot_grid_map(c3, c3_grid, land)
    plot_station_coverage_by_year(by_year)
    plot_vs_old_domain(c3, grid, c3_grid, land)

    qa = {
        "phase": "Phase 8A.3 C3 v2 ocean grid / mask",
        "no_prey_surfaces_generated": True,
        "no_lcp_models_run": True,
        "c3_grid_cell_count": int(summary.loc[0, "c3_ocean_grid_cell_count"]),
        "c3_polygon_area_km2": float(summary.loc[0, "c3_polygon_area_km2"]),
        "brian_usable_station_count": int(len(stations)),
        "brian_stations_inside_C3": int(stations["inside_C3_domain"].sum()),
        "brian_stations_outside_C3": int((~stations["inside_C3_domain"]).sum()),
        "layer_compatibility": comp.to_dict(orient="records"),
        "outputs": {
            "final_domain": str(C3_FINAL),
            "c3_grid": str(GRID_C3),
            "grid_points_geojson": str(GRID_C3_POINTS),
            "grid_qa_summary": str(OUT / "C3_v2_grid_qa_summary.csv"),
            "grid_qa_json": str(OUT / "C3_v2_grid_qa.json"),
            "station_coverage": str(OUT / "brian_station_coverage_on_C3_v2_grid.csv"),
            "station_coverage_by_year": str(OUT / "brian_station_coverage_on_C3_v2_grid_by_year.csv"),
            "stations_outside_C3": str(OUT / "brian_station_outside_C3_v2_domain.csv"),
            "layer_compatibility": str(OUT / "C3_v2_layer_compatibility_qa.csv"),
        },
    }
    (OUT / "C3_v2_grid_phase8a3_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
