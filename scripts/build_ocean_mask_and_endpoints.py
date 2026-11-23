"""Build v0 ocean/land mask QA and fixed endpoint candidates.

Geopandas is preferred for long-term spatial workflows, but this script keeps a
small shapefile reader so the current repository can run with only the existing
lightweight dependencies. It does not run LCP.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from gray_whale_corridor.spatial import GridDefinition  # noqa: E402


SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:3338"
LAND_DIR = PROJECT_ROOT / "data/raw/prey/ne_10m_land"
LAND_SHP = LAND_DIR / "ne_10m_land.shp"
LARGE_REGIONAL_BBOX = {
    "lat_min": 63.0,
    "lat_max": 73.0,
    "lon_min": -176.5,
    "lon_max": -157.0,
}
SOURCE_ZONE = {
    "lat_min": 63.0,
    "lat_max": 65.8,
    "lon_min": -173.5,
    "lon_max": -166.0,
}
DESTINATION_ZONE = {
    "lat_min": 70.3,
    "lat_max": 73.0,
    "lon_min": -166.5,
    "lon_max": -157.0,
}
DBO_ANCHORS = {
    "DBO2": [
        (65.1077470, -170.8651760),
        (65.0970010, -167.8595490),
        (64.4818280, -167.9077580),
        (64.4936580, -170.8406690),
    ],
    "DBO3": [
        (66.7858800, -171.3300080),
        (68.6088690, -171.4193340),
        (68.5724120, -166.4810290),
        (66.7522810, -166.7558080),
    ],
    "DBO4": [
        (71.9940000, -164.2260000),
        (71.8010000, -159.7290000),
        (70.6410000, -160.3050000),
        (70.8210000, -164.5560000),
    ],
}
ROLLING_GLOBAL_PATTERN = "prey_proxy_grid_idw_global_rolling5_*_target*.csv"


@dataclass(frozen=True)
class LandFeature:
    bbox: tuple[float, float, float, float]
    rings: list[list[tuple[float, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ocean/land mask QA and fixed endpoint candidates."
    )
    parser.add_argument("--cell-size", type=float, default=10_000.0)
    parser.add_argument("--max-endpoints", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not LAND_SHP.exists():
        raise FileNotFoundError(f"Natural Earth land shapefile not found: {LAND_SHP}")

    processed_dir = PROJECT_ROOT / "data/processed/study_domain"
    figures_dir = PROJECT_ROOT / "outputs/figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    land_crs = read_prj_crs(LAND_SHP.with_suffix(".prj"))
    land_features = read_polygon_shapefile(LAND_SHP)
    land_bounds = combined_bounds([feature.bbox for feature in land_features])
    coverage = bbox_intersects(land_bounds, bbox_tuple(LARGE_REGIONAL_BBOX))
    regional_land_features = [
        feature
        for feature in land_features
        if bbox_intersects(feature.bbox, bbox_tuple_with_padding(LARGE_REGIONAL_BBOX, 2.0))
    ]

    grid = grid_for_large_regional_bbox(args.cell_size)
    ocean_grid = build_ocean_mask_grid(grid, regional_land_features)
    mask_path = processed_dir / "ocean_mask_grid_10km_large_regional_bbox_v0.csv"
    ocean_grid.to_csv(mask_path, index=False)

    dbo_qa = dbo_anchor_ocean_qa(ocean_grid)
    ranking = build_pooled_ranking_surface()
    endpoints, endpoint_warnings, zone_counts = build_endpoint_candidates(
        ocean_grid=ocean_grid,
        ranking=ranking,
        max_endpoints=args.max_endpoints,
    )
    endpoint_path = processed_dir / "fixed_endpoint_candidates_main_south_to_north.csv"
    endpoints.to_csv(endpoint_path, index=False)

    plot_ocean_land_mask(
        ocean_grid=ocean_grid,
        land_features=regional_land_features,
        figure_path=figures_dir / "ocean_land_mask_v0.png",
    )
    plot_source_destination_zones(
        ocean_grid=ocean_grid,
        endpoints=endpoints,
        land_features=regional_land_features,
        figure_path=figures_dir / "source_destination_zones_v0.png",
    )

    total_cells = len(ocean_grid)
    land_cells = int(ocean_grid["land_flag"].sum())
    ocean_cells = int(ocean_grid["ocean_flag"].sum())
    qa = {
        "land_mask_file": str(LAND_SHP),
        "land_crs": land_crs,
        "land_feature_count": len(land_features),
        "regional_land_feature_count": len(regional_land_features),
        "land_bounds": land_bounds,
        "land_bounds_cover_study_domain": bool(coverage),
        "total_grid_cells": total_cells,
        "land_cells": land_cells,
        "ocean_cells": ocean_cells,
        "percent_ocean": float(100.0 * ocean_cells / total_cells),
        "source_zone_ocean_cell_count": int(zone_counts["source"]),
        "destination_zone_ocean_cell_count": int(zone_counts["destination"]),
        "selected_source_endpoint_count": int((endpoints["endpoint_type"] == "source").sum()),
        "selected_destination_endpoint_count": int((endpoints["endpoint_type"] == "destination").sum()),
        "dbo_anchor_ocean_qa": dbo_qa,
        "warnings": endpoint_warnings + [
            "geopandas is not installed; Natural Earth shapefile was read with a lightweight local parser.",
            "Land/ocean mask is centroid-based at 10 km resolution.",
            "TODO: move land mask to data/raw/spatial/land/ for long-term repo organization.",
        ],
    }
    qa_path = processed_dir / "ocean_mask_endpoint_qa_v0.json"
    with qa_path.open("w", encoding="utf-8") as file:
        json.dump(qa, file, indent=2)
        file.write("\n")

    print(json.dumps(qa, indent=2))


def read_prj_crs(path: Path) -> str:
    if not path.exists():
        return "Unknown"
    text = path.read_text(errors="ignore")
    try:
        return CRS.from_wkt(text).to_string()
    except Exception:
        return text.strip()[:200]


def read_polygon_shapefile(path: Path) -> list[LandFeature]:
    features: list[LandFeature] = []
    with path.open("rb") as file:
        header = file.read(100)
        shape_type = struct.unpack("<i", header[32:36])[0]
        if shape_type not in {5, 15, 25, 31}:
            raise ValueError(f"Unsupported shapefile shape type: {shape_type}")
        while True:
            record_header = file.read(8)
            if not record_header:
                break
            content_length_words = struct.unpack(">i", record_header[4:8])[0]
            content = file.read(content_length_words * 2)
            if len(content) < 44:
                continue
            rec_shape_type = struct.unpack("<i", content[:4])[0]
            if rec_shape_type == 0:
                continue
            if rec_shape_type not in {5, 15, 25, 31}:
                continue
            xmin, ymin, xmax, ymax = struct.unpack("<dddd", content[4:36])
            num_parts, num_points = struct.unpack("<ii", content[36:44])
            parts_start = 44
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:parts_start + 4 * num_parts]))
            points_start = parts_start + 4 * num_parts
            points = [
                struct.unpack("<dd", content[points_start + i * 16:points_start + (i + 1) * 16])
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


def grid_for_large_regional_bbox(cell_size: float) -> GridDefinition:
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    lonlat = densify_bbox_boundary(LARGE_REGIONAL_BBOX, points_per_side=100)
    projected = [transformer.transform(lon, lat) for lon, lat in lonlat]
    xs = np.array([point[0] for point in projected], dtype=float)
    ys = np.array([point[1] for point in projected], dtype=float)
    return GridDefinition(
        origin_x=float(xs.min()),
        origin_y=float(ys.min()),
        cols=math.ceil((float(xs.max()) - float(xs.min())) / cell_size),
        rows=math.ceil((float(ys.max()) - float(ys.min())) / cell_size),
        cell_size=cell_size,
        crs=TARGET_CRS,
    )


def build_ocean_mask_grid(
    grid: GridDefinition,
    land_features: list[LandFeature],
) -> pd.DataFrame:
    inverse = Transformer.from_crs(TARGET_CRS, SOURCE_CRS, always_xy=True)
    records = []
    for grid_y in range(grid.rows):
        y = grid.origin_y + (grid_y + 0.5) * grid.cell_size
        for grid_x in range(grid.cols):
            x = grid.origin_x + (grid_x + 0.5) * grid.cell_size
            lon, lat = inverse.transform(x, y)
            if not point_in_bbox(lon, lat, LARGE_REGIONAL_BBOX):
                continue
            land = point_on_land(lon, lat, land_features)
            records.append(
                {
                    "grid_id": f"({grid_x}, {grid_y})",
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "x": x,
                    "y": y,
                    "lon": lon,
                    "lat": lat,
                    "land_flag": int(land),
                    "ocean_flag": int(not land),
                }
            )
    return pd.DataFrame.from_records(records)


def point_on_land(lon: float, lat: float, land_features: list[LandFeature]) -> bool:
    for feature in land_features:
        if not bbox_contains_point(feature.bbox, lon, lat):
            continue
        inside = False
        for ring in feature.rings:
            if not ring_bbox_contains_point(ring, lon, lat):
                continue
            if point_in_ring(lon, lat, ring):
                inside = not inside
        if inside:
            return True
    return False


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def ring_bbox_contains_point(ring: list[tuple[float, float]], lon: float, lat: float) -> bool:
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return min(xs) <= lon <= max(xs) and min(ys) <= lat <= max(ys)


def dbo_anchor_ocean_qa(ocean_grid: pd.DataFrame) -> dict:
    qa = {}
    for name, coords in DBO_ANCHORS.items():
        polygon = [(lon, lat) for lat, lon in coords]
        in_anchor = ocean_grid.apply(lambda row: point_in_ring(row["lon"], row["lat"], polygon), axis=1)
        subset = ocean_grid[in_anchor]
        total = len(subset)
        ocean = int(subset["ocean_flag"].sum()) if total else 0
        ocean_pct = float(100.0 * ocean / total) if total else 0.0
        qa[name] = {
            "centroid_cells_in_anchor": int(total),
            "ocean_cells": ocean,
            "ocean_percent": ocean_pct,
            "mainly_ocean": bool(ocean_pct >= 80.0 and ocean >= 5),
        }
    return qa


def build_pooled_ranking_surface() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "outputs").glob(ROLLING_GLOBAL_PATTERN))
    if not files:
        raise FileNotFoundError(
            "No rolling global prey-proxy surfaces found. Run build_dominant_macroinfauna_prey_proxy.py first."
        )
    frames = []
    for path in files:
        df = pd.read_csv(path)
        frames.append(
            df[
                [
                    "grid_id",
                    "combined_prey_proxy_final_global",
                ]
            ].rename(columns={"combined_prey_proxy_final_global": path.stem})
        )
    pooled = frames[0]
    for frame in frames[1:]:
        pooled = pooled.merge(frame, on="grid_id", how="outer")
    value_cols = [col for col in pooled.columns if col != "grid_id"]
    pooled["pooled_prey_proxy"] = pooled[value_cols].mean(axis=1, skipna=True)
    return pooled[["grid_id", "pooled_prey_proxy"]]


def build_endpoint_candidates(
    ocean_grid: pd.DataFrame,
    ranking: pd.DataFrame,
    max_endpoints: int,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    warnings = []
    candidates = ocean_grid[ocean_grid["ocean_flag"] == 1].merge(ranking, on="grid_id", how="left")
    source_pool = candidates[candidates.apply(lambda row: point_in_bbox(row["lon"], row["lat"], SOURCE_ZONE), axis=1)].copy()
    dest_pool = candidates[candidates.apply(lambda row: point_in_bbox(row["lon"], row["lat"], DESTINATION_ZONE), axis=1)].copy()
    zone_counts = {"source": len(source_pool), "destination": len(dest_pool)}
    selected = []
    for endpoint_type, pool in [("source", source_pool), ("destination", dest_pool)]:
        pool = pool.dropna(subset=["pooled_prey_proxy"]).sort_values("pooled_prey_proxy", ascending=False)
        thinned = spatial_thin(pool, max_count=max_endpoints, min_distance_m=25_000.0)
        if len(thinned) < 20:
            warnings.append(
                f"{endpoint_type} endpoint thinning at 25 km retained {len(thinned)} cells; relaxed to 15 km."
            )
            thinned = spatial_thin(pool, max_count=max_endpoints, min_distance_m=15_000.0)
        thinned = thinned.copy()
        thinned["endpoint_type"] = endpoint_type
        selected.append(thinned)
    endpoints = pd.concat(selected, ignore_index=True)
    return (
        endpoints[
            [
                "grid_id",
                "x",
                "y",
                "lon",
                "lat",
                "endpoint_type",
                "pooled_prey_proxy",
                "ocean_flag",
            ]
        ],
        warnings,
        zone_counts,
    )


def spatial_thin(pool: pd.DataFrame, max_count: int, min_distance_m: float) -> pd.DataFrame:
    selected_rows = []
    selected_xy: list[tuple[float, float]] = []
    for _, row in pool.iterrows():
        xy = (float(row["x"]), float(row["y"]))
        if all(math.hypot(xy[0] - sx, xy[1] - sy) >= min_distance_m for sx, sy in selected_xy):
            selected_rows.append(row)
            selected_xy.append(xy)
        if len(selected_rows) >= max_count:
            break
    if not selected_rows:
        return pool.iloc[0:0].copy()
    return pd.DataFrame(selected_rows)


def plot_ocean_land_mask(
    ocean_grid: pd.DataFrame,
    land_features: list[LandFeature],
    figure_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_land(ax, land_features)
    land = ocean_grid[ocean_grid["land_flag"] == 1]
    ocean = ocean_grid[ocean_grid["ocean_flag"] == 1]
    ax.scatter(ocean["lon"], ocean["lat"], s=2, color="#4c78a8", alpha=0.18, label="ocean grid")
    ax.scatter(land["lon"], land["lat"], s=4, color="#d62728", alpha=0.5, label="land grid")
    draw_context(ax)
    ax.set_title("Ocean/land mask QA at 10 km grid resolution")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_source_destination_zones(
    ocean_grid: pd.DataFrame,
    endpoints: pd.DataFrame,
    land_features: list[LandFeature],
    figure_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_land(ax, land_features)
    ocean = ocean_grid[ocean_grid["ocean_flag"] == 1]
    ax.scatter(ocean["lon"], ocean["lat"], s=2, color="#4c78a8", alpha=0.12, label="ocean grid")
    add_bbox_patch(ax, SOURCE_ZONE, "source zone", "#2ca02c")
    add_bbox_patch(ax, DESTINATION_ZONE, "destination zone", "#9467bd")
    source = endpoints[endpoints["endpoint_type"] == "source"]
    dest = endpoints[endpoints["endpoint_type"] == "destination"]
    ax.scatter(source["lon"], source["lat"], s=36, color="#2ca02c", edgecolor="black", label="fixed source endpoints")
    ax.scatter(dest["lon"], dest["lat"], s=36, color="#9467bd", edgecolor="black", label="fixed destination endpoints")
    draw_context(ax)
    ax.set_title("Fixed ocean-only source/destination endpoint candidates")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_land(ax: plt.Axes, land_features: list[LandFeature]) -> None:
    for feature in land_features:
        if not bbox_intersects(feature.bbox, bbox_tuple_with_padding(LARGE_REGIONAL_BBOX, 1.5)):
            continue
        for ring in feature.rings:
            if len(ring) < 3:
                continue
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            if max(xs) < LARGE_REGIONAL_BBOX["lon_min"] - 2 or min(xs) > LARGE_REGIONAL_BBOX["lon_max"] + 2:
                continue
            if max(ys) < LARGE_REGIONAL_BBOX["lat_min"] - 2 or min(ys) > LARGE_REGIONAL_BBOX["lat_max"] + 2:
                continue
            plot_ring = downsample_ring(ring, max_points=1200)
            patch = MplPolygon(plot_ring, closed=True, facecolor="#d9c8a9", edgecolor="#8c7b66", linewidth=0.3, alpha=0.8)
            ax.add_patch(patch)


def downsample_ring(
    ring: list[tuple[float, float]],
    max_points: int,
) -> list[tuple[float, float]]:
    if len(ring) <= max_points:
        return ring
    step = math.ceil(len(ring) / max_points)
    sampled = ring[::step]
    if sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    return sampled


def draw_context(ax: plt.Axes) -> None:
    add_bbox_patch(ax, LARGE_REGIONAL_BBOX, "study domain", "#333333", linestyle="--", fill=False)
    for name, coords in DBO_ANCHORS.items():
        lonlat = [(lon, lat) for lat, lon in coords]
        patch = MplPolygon(lonlat, closed=True, facecolor="none", edgecolor="#111111", linewidth=1.0, label=name)
        ax.add_patch(patch)
    ax.set_xlim(LARGE_REGIONAL_BBOX["lon_min"] - 1, LARGE_REGIONAL_BBOX["lon_max"] + 1)
    ax.set_ylim(LARGE_REGIONAL_BBOX["lat_min"] - 0.5, LARGE_REGIONAL_BBOX["lat_max"] + 0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def add_bbox_patch(
    ax: plt.Axes,
    bbox: dict[str, float],
    label: str,
    color: str,
    linestyle: str = "-",
    fill: bool = False,
) -> None:
    coords = [
        (bbox["lon_min"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_max"]),
        (bbox["lon_min"], bbox["lat_max"]),
    ]
    patch = MplPolygon(
        coords,
        closed=True,
        facecolor=color if fill else "none",
        edgecolor=color,
        linewidth=1.4,
        linestyle=linestyle,
        alpha=0.15 if fill else 1.0,
        label=label,
    )
    ax.add_patch(patch)


def densify_bbox_boundary(bbox: dict[str, float], points_per_side: int) -> list[tuple[float, float]]:
    west = [(bbox["lon_min"], lat) for lat in np.linspace(bbox["lat_min"], bbox["lat_max"], points_per_side)]
    north = [(lon, bbox["lat_max"]) for lon in np.linspace(bbox["lon_min"], bbox["lon_max"], points_per_side)]
    east = [(bbox["lon_max"], lat) for lat in np.linspace(bbox["lat_max"], bbox["lat_min"], points_per_side)]
    south = [(lon, bbox["lat_min"]) for lon in np.linspace(bbox["lon_max"], bbox["lon_min"], points_per_side)]
    return west + north + east + south


def point_in_bbox(lon: float, lat: float, bbox: dict[str, float]) -> bool:
    return bbox["lon_min"] <= lon <= bbox["lon_max"] and bbox["lat_min"] <= lat <= bbox["lat_max"]


def bbox_tuple(bbox: dict[str, float]) -> tuple[float, float, float, float]:
    return bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]


def bbox_tuple_with_padding(bbox: dict[str, float], padding: float) -> tuple[float, float, float, float]:
    return (
        bbox["lon_min"] - padding,
        bbox["lat_min"] - padding,
        bbox["lon_max"] + padding,
        bbox["lat_max"] + padding,
    )


def bbox_contains_point(bbox: tuple[float, float, float, float], lon: float, lat: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def combined_bounds(bounds: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


if __name__ == "__main__":
    main()
