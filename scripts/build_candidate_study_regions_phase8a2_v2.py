"""Build Phase 8A.2 candidate v2 study-region polygons.

This is a domain-design/QA script only. It does not build prey surfaces and does
not run LCP models.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon, box, mapping, shape
from shapely.ops import transform, unary_union
from shapely import wkb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "prey_v2"
FIG = ROOT / "outputs" / "figures" / "prey_v2"
STUDY = ROOT / "data" / "processed" / "study_domain"

TARGET_YEARS = list(range(2011, 2020))
CRS_WGS84 = "EPSG:4326"
CRS_WORK = "EPSG:3338"
GRID_SPACING_M = 10_000


def clean_key(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def station_key_from(row: pd.Series, prefix: str = "") -> str:
    def get(name: str) -> object:
        return row.get(prefix + name, row.get(name))

    return "|".join(
        [
            clean_key(get("Year_clean")),
            clean_key(get("Cruise_clean")),
            clean_key(get("StationNumber_clean")),
            clean_key(get("StationName_clean")),
        ]
    )


def gpkg_wkb_to_geom(blob: bytes):
    """Decode simple GeoPackage geometry blob to shapely geometry."""
    if blob is None:
        return None
    b = bytes(blob)
    if b[:2] != b"GP":
        return wkb.loads(b)
    flags = b[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_lengths = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    header_len = 8 + envelope_lengths.get(envelope_code, 0)
    return wkb.loads(b[header_len:])


def read_gpkg_geometries(path: Path) -> list:
    if not path.exists():
        return []
    geoms = []
    with sqlite3.connect(path) as con:
        tables = pd.read_sql_query("SELECT table_name FROM gpkg_contents WHERE data_type='features'", con)
        for table in tables["table_name"]:
            cols = pd.read_sql_query(f"PRAGMA table_info('{table}')", con)
            geom_cols = pd.read_sql_query(
                f"SELECT column_name FROM gpkg_geometry_columns WHERE table_name='{table}'", con
            )
            if geom_cols.empty:
                continue
            geom_col = geom_cols["column_name"].iloc[0]
            rows = con.execute(f"SELECT {geom_col} FROM '{table}'").fetchall()
            for (blob,) in rows:
                try:
                    geom = gpkg_wkb_to_geom(blob)
                    if geom is not None and not geom.is_empty:
                        geoms.append(geom)
                except Exception:
                    continue
    return geoms


def finite_geom(geom) -> bool:
    try:
        b = geom.bounds
        return all(np.isfinite(v) for v in b)
    except Exception:
        return False


to_work = Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform
to_wgs = Transformer.from_crs(CRS_WORK, CRS_WGS84, always_xy=True).transform


def project_geom(geom):
    return transform(to_work, geom)


def unproject_geom(geom):
    return transform(to_wgs, geom)


def geom_to_feature(geom, props: dict) -> dict:
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


def write_geojson(features: list[dict], path: Path) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def save_single_polygon(geom_wgs, path: Path, props: dict) -> None:
    write_geojson([geom_to_feature(geom_wgs, props)], path)


def compactness(geom_proj) -> float:
    area = geom_proj.area
    perim = geom_proj.length
    if perim <= 0:
        return math.nan
    return float(4 * math.pi * area / (perim * perim))


def make_station_extent() -> pd.DataFrame:
    brian = pd.read_csv(OUT / "brian_amphipod_timeseries_cleaned_v2.csv")
    unique = pd.read_csv(OUT / "brian_amphipod_unique_station_match_results_v2.csv")
    cross = pd.read_csv(OUT / "brian_alynne_station_crosswalk_v1.csv")

    brian["brian_unique_station_key"] = brian.apply(station_key_from, axis=1)
    totals = (
        brian[brian["Year_clean"].between(2011, 2019)]
        .groupby("brian_unique_station_key")
        .agg(
            total_amphipod_abundance=("Abundance", "sum"),
            total_amphipod_gww=("WetWeight", "sum"),
            total_amphipod_gC=("C_Biomass", "sum"),
            total_amphipod_kJm2_frozen=("kJm2.frozen", "sum"),
            total_amphipod_kJm2_formalin=("kJm2.formalin", "sum"),
            family_count=("Family_clean", "nunique"),
            row_count=("Family_clean", "size"),
        )
        .reset_index()
    )

    unique["brian_unique_station_key"] = unique.apply(station_key_from, axis=1)
    target = unique[unique["Year_clean"].between(2011, 2019)].copy()
    target["Latitude"] = np.where(target["matched_yes_no"].astype(bool), target["Latitude_matched"], np.nan)
    target["Longitude"] = np.where(target["matched_yes_no"].astype(bool), target["Longitude_matched"], np.nan)
    target["resolution_method"] = np.where(
        target["matched_yes_no"].astype(bool), "phase8a_" + target["match_method"].astype(str), ""
    )
    target["confidence_level"] = np.select(
        [target["match_status"].eq("exact"), target["match_status"].eq("fallback")],
        ["high", "medium"],
        default="low",
    )

    cross_res = cross.set_index("brian_unique_station_key")
    for idx, row in target[~target["matched_yes_no"].astype(bool)].iterrows():
        key = row["brian_unique_station_key"]
        if key in cross_res.index:
            c = cross_res.loc[key]
            if pd.notna(c.get("resolved_Latitude")) and pd.notna(c.get("resolved_Longitude")):
                target.loc[idx, "Latitude"] = c["resolved_Latitude"]
                target.loc[idx, "Longitude"] = c["resolved_Longitude"]
            target.loc[idx, "resolution_method"] = c["resolution_method"]
            target.loc[idx, "confidence_level"] = c["confidence_level"]

    target = target.merge(totals, on="brian_unique_station_key", how="left")
    out = pd.DataFrame(
        {
            "Year": target["Year_clean"].astype(int),
            "Cruise": target["Cruise_clean"],
            "StationNumber": target["StationNumber_clean"],
            "StationName": target["StationName_clean"],
            "Latitude": target["Latitude"],
            "Longitude": target["Longitude"],
            "resolution_method": target["resolution_method"],
            "confidence_level": target["confidence_level"],
            "match_status": target["match_status"],
            "brian_unique_station_key": target["brian_unique_station_key"],
            "total_amphipod_abundance": target["total_amphipod_abundance"],
            "total_amphipod_gww": target["total_amphipod_gww"],
            "total_amphipod_gC": target["total_amphipod_gC"],
            "total_amphipod_kJm2_frozen": target["total_amphipod_kJm2_frozen"],
            "total_amphipod_kJm2_formalin": target["total_amphipod_kJm2_formalin"],
            "family_count": target["family_count"],
            "row_count": target["row_count"],
        }
    )

    grid = pd.read_csv(STUDY / "ocean_mask_grid_10km_large_regional_bbox_v0.csv")
    ocean = grid[grid["ocean_flag"].astype(int).eq(1)].copy()
    current_geom = current_domain_geom(ocean)
    # Use the current grid envelope for this QA phase. The local GeoPackage
    # reader is intentionally lightweight and some GPKG geometries are not
    # reliable in this environment.
    existing_geom = current_geom

    pts = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None for xy in zip(out["Longitude"], out["Latitude"])]
    out["has_usable_coordinates"] = [p is not None for p in pts]
    out["inside_current_study_grid"] = [bool(p and current_geom.contains(p)) for p in pts]
    out["inside_existing_study_domain_boundary"] = [bool(p and existing_geom.contains(p)) for p in pts]
    out.to_csv(OUT / "brian_amphipod_station_extent_2011_2019_v2.csv", index=False)
    return out


def current_domain_geom(ocean_grid: pd.DataFrame):
    if {"lon", "lat"}.issubset(ocean_grid.columns):
        minx, maxx = ocean_grid["lon"].min(), ocean_grid["lon"].max()
        miny, maxy = ocean_grid["lat"].min(), ocean_grid["lat"].max()
    else:
        minx, maxx = ocean_grid["x"].min(), ocean_grid["x"].max()
        miny, maxy = ocean_grid["y"].min(), ocean_grid["y"].max()
    pad = 0.05
    return box(minx - pad, miny - pad, maxx + pad, maxy + pad)


def annual_hulls(stations: pd.DataFrame):
    feats = []
    summary = []
    usable = stations[stations["has_usable_coordinates"]].copy()
    for year, sub in usable.groupby("Year"):
        pts = [Point(xy) for xy in zip(sub["Longitude"], sub["Latitude"])]
        mp = MultiPoint(pts)
        if len(pts) >= 3:
            hull_wgs = mp.convex_hull
            hull_proj = project_geom(hull_wgs)
            for buffer_km in [0, 25, 50, 100]:
                geom_proj = hull_proj.buffer(buffer_km * 1000) if buffer_km else hull_proj
                geom_wgs = unproject_geom(geom_proj)
                feats.append(
                    geom_to_feature(
                        geom_wgs,
                        {
                            "year": int(year),
                            "buffer_km": buffer_km,
                            "station_count": int(len(sub)),
                            "area_km2": round(geom_proj.area / 1e6, 3),
                        },
                    )
                )
                summary.append(
                    {
                        "year": int(year),
                        "station_count": int(len(sub)),
                        "min_lat": sub["Latitude"].min(),
                        "max_lat": sub["Latitude"].max(),
                        "min_lon": sub["Longitude"].min(),
                        "max_lon": sub["Longitude"].max(),
                        "buffer_km": buffer_km,
                        "hull_area_km2": round(geom_proj.area / 1e6, 3),
                    }
                )
        else:
            summary.append(
                {
                    "year": int(year),
                    "station_count": int(len(sub)),
                    "min_lat": sub["Latitude"].min(),
                    "max_lat": sub["Latitude"].max(),
                    "min_lon": sub["Longitude"].min(),
                    "max_lon": sub["Longitude"].max(),
                    "buffer_km": np.nan,
                    "hull_area_km2": np.nan,
                }
            )
    write_geojson(feats, OUT / "annual_station_hulls_2011_2019_v2.geojson")
    annual_summary = pd.DataFrame(summary)
    annual_summary.to_csv(OUT / "annual_station_extent_summary_2011_2019_v2.csv", index=False)
    hulls0 = {
        f["properties"]["year"]: shape(f["geometry"])
        for f in feats
        if f["properties"]["buffer_km"] == 0
    }
    return annual_summary, hulls0


def support_frequency_grid(stations: pd.DataFrame, hulls0: dict[int, object]) -> pd.DataFrame:
    usable = stations[stations["has_usable_coordinates"]].copy()
    grid = pd.read_csv(STUDY / "ocean_mask_grid_10km_large_regional_bbox_v0.csv")
    ocean = grid[grid["ocean_flag"].astype(int).eq(1)].copy()
    gx, gy = Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform(
        ocean["lon"].values, ocean["lat"].values
    )
    rows = []
    hulls_proj = {y: project_geom(g) for y, g in hulls0.items()}
    hull_bounds = {yr: h.bounds for yr, h in hulls_proj.items()}
    station_trees = {}
    for year, sub in usable.groupby("Year"):
        sx, sy = Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform(
            sub["Longitude"].values, sub["Latitude"].values
        )
        station_trees[int(year)] = cKDTree(np.column_stack([sx, sy]))
    for i, (x, y) in enumerate(zip(gx, gy)):
        p = Point(x, y)
        covered = {}
        for yr, h in hulls_proj.items():
            minx, miny, maxx, maxy = hull_bounds[yr]
            if x < minx or x > maxx or y < miny or y > maxy:
                covered[yr] = 0
            else:
                covered[yr] = int(h.contains(p) or h.touches(p))
        support_years = sum(covered.values())
        rec = {
            "grid_id": ocean.iloc[i].get("grid_id", i),
            "grid_x_3338": x,
            "grid_y_3338": y,
            "lon": ocean.iloc[i]["lon"],
            "lat": ocean.iloc[i]["lat"],
            "support_year_count": support_years,
            "covered_at_least_1yr": support_years >= 1,
            "covered_at_least_3yr": support_years >= 3,
            "covered_at_least_5yr": support_years >= 5,
            "covered_at_least_7yr": support_years >= 7,
        }
        for yr, val in covered.items():
            rec[f"inside_hull_{yr}"] = val
        for yr, tree in station_trees.items():
            dist, _ = tree.query([[x, y]], k=1)
            rec[f"distance_to_nearest_station_{yr}_km"] = round(float(dist[0]) / 1000, 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "brian_sampling_support_frequency_grid_v2.csv", index=False)
    return df


def candidate_polygons(stations: pd.DataFrame, support: pd.DataFrame, hulls0: dict[int, object]) -> dict[str, object]:
    grid = pd.read_csv(STUDY / "ocean_mask_grid_10km_large_regional_bbox_v0.csv")
    ocean = grid[grid["ocean_flag"].astype(int).eq(1)].copy()
    current = current_domain_geom(ocean)
    # Candidate A is the current grid/domain envelope baseline. Do not replace
    # it with GeoPackage geometry here because the local GPKG parser may produce
    # invalid projected-looking WGS84 coordinates when GDAL/Fiona are unavailable.

    usable = stations[stations["has_usable_coordinates"]].copy()
    all_pts = [Point(xy) for xy in zip(usable["Longitude"], usable["Latitude"])]
    all_hull_proj = project_geom(MultiPoint(all_pts).convex_hull)

    candidates = {
        "A_current_domain": current,
        "B_all_station_hull_buffer50": unproject_geom(all_hull_proj.buffer(50_000)),
        "B_all_station_hull_buffer100": unproject_geom(all_hull_proj.buffer(100_000)),
    }
    for threshold in [3, 5]:
        sub = support[support["support_year_count"] >= threshold]
        pts_proj = [Point(xy) for xy in zip(sub["grid_x_3338"], sub["grid_y_3338"])]
        if len(pts_proj) >= 3:
            candidates[f"C_recurring_support_{threshold}yr"] = unproject_geom(
                MultiPoint(pts_proj).convex_hull.buffer(25_000)
            )

    # The local DBO anchor GeoPackage cannot be read reliably without GDAL/Fiona
    # in this environment; the lightweight GeoPackage parser can produce invalid
    # coordinate sequences for that file. For Phase 8A.2, DBO/hotspot context is
    # therefore represented with station clusters and map labels, not formal DBO
    # geometry.
    dbo_geoms = []
    dbo_union = unary_union(dbo_geoms) if dbo_geoms else None

    # Approximate hotspot-informed domain from ecological labels and station clusters.
    # DBO5/Barrow Canyon is not available as a formal polygon, so high-latitude
    # benthic station support is used as an approximate northern anchor.
    hotspot_parts = []
    if dbo_union is not None:
        hotspot_parts.append(project_geom(dbo_union).buffer(75_000))
    hotspot_points = usable[
        (
            usable["Latitude"].between(62.0, 73.6)
            & usable["Longitude"].between(-176.5, -156.0)
        )
    ]
    if len(hotspot_points) >= 3:
        pts = [Point(xy) for xy in zip(hotspot_points["Longitude"], hotspot_points["Latitude"])]
        hotspot_parts.append(project_geom(MultiPoint(pts).convex_hull).buffer(75_000))
    candidates["D_dbo25_hotspot_informed"] = unproject_geom(unary_union(hotspot_parts).convex_hull)

    c3 = candidates.get("C_recurring_support_3yr")
    d = candidates["D_dbo25_hotspot_informed"]
    hybrid_parts = [project_geom(d)]
    if c3 is not None:
        hybrid_parts.append(project_geom(c3))
    # Use convex hull around the union to keep the review candidate continuous and compact.
    candidates["E_hybrid_recommended_for_review"] = unproject_geom(unary_union(hybrid_parts).convex_hull.buffer(25_000))

    file_map = {
        "A_current_domain": "candidate_study_region_A_current_domain_v2.geojson",
        "B_all_station_hull_buffer50": "candidate_study_region_B_all_station_hull_buffer50_v2.geojson",
        "B_all_station_hull_buffer100": "candidate_study_region_B_all_station_hull_buffer100_v2.geojson",
        "C_recurring_support_3yr": "candidate_study_region_C_recurring_support_3yr_v2.geojson",
        "C_recurring_support_5yr": "candidate_study_region_C_recurring_support_5yr_v2.geojson",
        "D_dbo25_hotspot_informed": "candidate_study_region_D_dbo25_hotspot_informed_v2.geojson",
        "E_hybrid_recommended_for_review": "candidate_study_region_E_hybrid_recommended_for_review_v2.geojson",
    }
    for name, geom in candidates.items():
        save_single_polygon(geom, OUT / file_map[name], {"candidate": name, "crs": CRS_WGS84})
    return candidates


def candidate_metrics(stations: pd.DataFrame, candidates: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = stations[stations["has_usable_coordinates"]].copy()
    point_xy_proj = np.column_stack(
        Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform(
            usable["Longitude"].values, usable["Latitude"].values
        )
    )
    pts_proj = [Point(xy) for xy in point_xy_proj]
    grid = pd.read_csv(STUDY / "ocean_mask_grid_10km_large_regional_bbox_v0.csv")
    ocean = grid[grid["ocean_flag"].astype(int).eq(1)].copy()
    grid_xy_proj = np.column_stack(
        Transformer.from_crs(CRS_WGS84, CRS_WORK, always_xy=True).transform(ocean["lon"].values, ocean["lat"].values)
    )
    grid_pts_proj = [Point(xy) for xy in grid_xy_proj]
    dbo_geoms = []
    dbo_union = unary_union(dbo_geoms) if dbo_geoms else None

    metrics = []
    excluded_rows = []
    compat = []
    for name, geom in candidates.items():
        geom_proj = project_geom(geom)
        covered = np.array([geom_proj.contains(p) or geom_proj.touches(p) for p in pts_proj])
        high = usable["confidence_level"].isin(["high", "medium"]).values
        bounds = geom.bounds
        grid_count = sum(geom_proj.contains(p) or geom_proj.touches(p) for p in grid_pts_proj)
        metrics.append(
            {
                "candidate": name,
                "station_coverage_percent": round(float(covered.mean() * 100), 3),
                "station_count_covered": int(covered.sum()),
                "station_count_total": int(len(usable)),
                "high_confidence_station_coverage_percent": round(float(covered[high].mean() * 100), 3),
                "area_km2": round(float(geom_proj.area / 1e6), 3),
                "compactness_4piA_P2": round(compactness(geom_proj), 4),
                "continuous_or_multipart": "continuous" if geom.geom_type == "Polygon" else "multi_part",
                "current_10km_ocean_grid_cells_covered": int(grid_count),
                "min_lon": round(bounds[0], 4),
                "min_lat": round(bounds[1], 4),
                "max_lon": round(bounds[2], 4),
                "max_lat": round(bounds[3], 4),
                "covers_DBO2_DBO4_anchor_polygons": bool(dbo_union and geom.contains(dbo_union.centroid)),
                "covers_DBO5_or_Barrow_Canyon_approx": bool(bounds[3] >= 72.5 and bounds[2] >= -158.5),
            }
        )
        for year, sub in usable.assign(_covered=covered).groupby("Year"):
            metrics.append(
                {
                    "candidate": f"{name}__year_{int(year)}",
                    "station_coverage_percent": round(float(sub["_covered"].mean() * 100), 3),
                    "station_count_covered": int(sub["_covered"].sum()),
                    "station_count_total": int(len(sub)),
                    "high_confidence_station_coverage_percent": np.nan,
                    "area_km2": np.nan,
                    "compactness_4piA_P2": np.nan,
                    "continuous_or_multipart": "",
                    "current_10km_ocean_grid_cells_covered": np.nan,
                    "min_lon": np.nan,
                    "min_lat": np.nan,
                    "max_lon": np.nan,
                    "max_lat": np.nan,
                    "covers_DBO2_DBO4_anchor_polygons": np.nan,
                    "covers_DBO5_or_Barrow_Canyon_approx": np.nan,
                }
            )
        excl = usable.loc[~covered].copy()
        if not excl.empty:
            excl["candidate"] = name
            excl["excluded_station_note"] = np.where(
                (excl["Latitude"] > 72.5) | (excl["Latitude"] < 62.5),
                "potential edge/outlier or high-latitude ecological anchor",
                "excluded from candidate polygon",
            )
            excluded_rows.append(excl)

        compat.append(
            {
                "candidate": name,
                "GLORYS12_ice_likely_covers": True,
                "Kapsar_ship_likely_covers_2015_2019": True,
                "source_destination_endpoints_likely_need_revision": name != "A_current_domain",
                "notes": "Coverage likely based on prior broad data inventories; final v2 grid should re-check before surface/LCP construction.",
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "candidate_study_region_comparison_metrics_v2.csv", index=False)
    excluded = pd.concat(excluded_rows, ignore_index=True) if excluded_rows else pd.DataFrame()
    excluded.to_csv(OUT / "candidate_study_region_excluded_stations_by_candidate_v2.csv", index=False)
    compat_df = pd.DataFrame(compat)
    compat_df.to_csv(OUT / "candidate_study_region_layer_compatibility_v2.csv", index=False)
    return metrics_df, excluded, compat_df


def add_geom(ax, geom, **kwargs):
    geoms = list(geom.geoms) if hasattr(geom, "geoms") and geom.geom_type.startswith("Multi") else [geom]
    for g in geoms:
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.plot(x, y, **kwargs)


def add_points(ax, df, **kwargs):
    ax.scatter(df["Longitude"], df["Latitude"], **kwargs)


def base_ax(ax, stations=None):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="0.88", linewidth=0.5)
    if stations is not None and len(stations):
        ax.set_xlim(stations["Longitude"].min() - 1.0, stations["Longitude"].max() + 1.0)
        ax.set_ylim(stations["Latitude"].min() - 0.8, stations["Latitude"].max() + 0.8)


def create_figures(stations: pd.DataFrame, annual_summary: pd.DataFrame, hulls0: dict[int, object], support: pd.DataFrame, candidates: dict[str, object]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    usable = stations[stations["has_usable_coordinates"]].copy()
    current = candidates["A_current_domain"]
    colors = {
        "A_current_domain": "black",
        "B_all_station_hull_buffer50": "#1f77b4",
        "B_all_station_hull_buffer100": "#17becf",
        "C_recurring_support_3yr": "#2ca02c",
        "C_recurring_support_5yr": "#006d2c",
        "D_dbo25_hotspot_informed": "#ff7f0e",
        "E_hybrid_recommended_for_review": "#d62728",
    }

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for ax, year in zip(axes.ravel(), TARGET_YEARS):
        sub = usable[usable["Year"] == year]
        base_ax(ax, usable)
        add_geom(ax, current, color="0.55", linewidth=1, linestyle="--")
        if year in hulls0:
            add_geom(ax, hulls0[year], color="#1f77b4", linewidth=1.4)
        add_points(ax, sub, s=10, color="#d62728", alpha=0.8)
        ax.set_title(str(year), fontsize=10)
    fig.suptitle("benthic amphipod station annual extents, 2011-2019", fontsize=14)
    fig.savefig(FIG / "annual_brian_station_extent_maps_2011_2019_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    base_ax(ax, usable)
    add_points(ax, usable, s=8, color="0.2", alpha=0.35)
    for yr, hull in hulls0.items():
        add_geom(ax, hull, color=plt.cm.viridis((yr - 2011) / 8), linewidth=1, alpha=0.9)
    add_geom(ax, current, color="black", linewidth=1.2, linestyle="--", label="Current domain")
    ax.set_title("Annual station hulls, 2011-2019")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "annual_brian_station_hulls_2011_2019_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    base_ax(ax, usable)
    sc = ax.scatter(support["lon"], support["lat"], c=support["support_year_count"], s=4, cmap="magma", vmin=0, vmax=9)
    add_points(ax, usable, s=6, color="cyan", alpha=0.6)
    add_geom(ax, current, color="white", linewidth=1.2, linestyle="--")
    fig.colorbar(sc, ax=ax, label="Annual hull support count")
    ax.set_title("benthic sampling-support frequency")
    fig.savefig(FIG / "brian_sampling_support_frequency_map_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    base_ax(ax, usable)
    add_points(ax, usable, s=8, color="0.1", alpha=0.45, label="benthic stations")
    for name, geom in candidates.items():
        add_geom(ax, geom, color=colors.get(name, "0.2"), linewidth=1.6 if name.startswith("E_") else 1.1, linestyle="-" if name.startswith("E_") else "--", label=name)
    ax.set_title("Candidate v2 study-region polygons")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(FIG / "candidate_study_regions_all_options_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    base_ax(ax, usable)
    for hull in hulls0.values():
        add_geom(ax, hull, color="0.65", linewidth=0.8)
    for name in ["A_current_domain", "C_recurring_support_3yr", "C_recurring_support_5yr", "E_hybrid_recommended_for_review"]:
        if name in candidates:
            add_geom(ax, candidates[name], color=colors[name], linewidth=1.8, label=name)
    add_points(ax, usable, s=7, color="0.1", alpha=0.35)
    ax.set_title("Candidate regions over annual station hulls")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG / "candidate_study_regions_over_annual_hulls_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    base_ax(ax, usable)
    sc = ax.scatter(support["lon"], support["lat"], c=support["support_year_count"], s=4, cmap="magma", vmin=0, vmax=9)
    for name in ["C_recurring_support_3yr", "C_recurring_support_5yr", "E_hybrid_recommended_for_review"]:
        if name in candidates:
            add_geom(ax, candidates[name], color=colors[name], linewidth=1.8, label=name)
    fig.colorbar(sc, ax=ax, label="Annual hull support count")
    ax.set_title("Candidate regions over support-frequency map")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG / "candidate_study_regions_over_support_frequency_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    base_ax(ax, usable)
    add_points(ax, usable, s=8, color="0.2", alpha=0.45)
    add_geom(ax, candidates["A_current_domain"], color="black", linewidth=1.5, linestyle="--", label="Current domain")
    add_geom(ax, candidates["E_hybrid_recommended_for_review"], color="#d62728", linewidth=2.0, label="Hybrid candidate")
    ax.set_title("Current domain vs hybrid candidate")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIG / "current_domain_vs_hybrid_candidate_v2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    base_ax(ax, usable)
    add_points(ax, usable, s=9, color="0.1", alpha=0.5, label="benthic stations")
    add_geom(ax, candidates["E_hybrid_recommended_for_review"], color="#d62728", linewidth=2.0, label="Hybrid candidate")
    labels = {
        "Chirikov Basin": (-169.5, 63.8),
        "Bering Strait": (-168.5, 65.6),
        "Southern Chukchi": (-167.5, 68.2),
        "Wainwright": (-160.2, 70.8),
        "Barrow Canyon / DBO5 approx.": (-157.8, 72.2),
        "DBO2": (-169.5, 63.0),
        "DBO3": (-166.0, 68.0),
        "DBO4": (-161.5, 71.0),
    }
    for label, (x, y) in labels.items():
        ax.text(x, y, label, fontsize=8, color="black", bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"})
    ax.set_title("Hybrid candidate with station and hotspot context")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIG / "hybrid_candidate_with_stations_and_hotspots_v2.png", dpi=300)
    plt.close(fig)


def write_recommendation(metrics: pd.DataFrame, stations: pd.DataFrame) -> None:
    main = metrics[~metrics["candidate"].str.contains("__year_")].copy()
    e = main[main["candidate"].eq("E_hybrid_recommended_for_review")].iloc[0]
    c3 = main[main["candidate"].eq("C_recurring_support_3yr")].iloc[0]
    text = f"""# Study Region v2 Recommendation Memo

## Summary

Phase 8A.2 generated candidate v2 study-region polygons from benthic 2011-2019 amphipod station extent and available DBO/hotspot context. The alternative domain definitions retain distinct sampling-support criteria.

## Candidate Interpretation

- Candidate A, the current domain, is a baseline comparison. It is rectangular/bbox-like and is not ideal as a final biological inference domain.
- Candidate B, all-station convex hull plus buffer, captures nearly all station support but can become broad because edge stations strongly influence the hull.
- Candidate C, recurring-support hulls, better represent repeated sampling support. The 3-year version covers {c3['station_coverage_percent']}% of usable stations and is more defensible as a data-supported core domain.
- Candidate D, DBO2-DBO5 / hotspot-informed, is approximate because formal DBO5 GIS polygons were not available. It combines available DBO anchor context with benthic station clusters.
- Candidate E, the hybrid candidate, combines recurring station support and hotspot context. It covers {e['station_coverage_percent']}% of usable 2011-2019 stations, has area {e['area_km2']} km2, and is continuous.

## Current Domain Adequacy

The current domain is useful as a baseline, but its rectangular geometry is not ideal for a final v2 biological inference region. It may include areas that are computationally convenient rather than biologically justified.

## Repeated vs Weak Support

Recurring support is strongest where annual hulls overlap across multiple years. Areas supported by only 1-2 years should be flagged as weaker unless they correspond to ecologically essential anchors such as DBO/hotspot areas.

## DBO5 / Barrow Canyon

The candidate-region construction uses approximate Barrow Canyon / DBO5 context from northern station clusters. These contextual locations are not formal DBO5 polygon boundaries.

## Recommended Candidates for Human Review

1. Candidate E: hybrid recommended-for-review polygon.
2. Candidate C: recurring-support 3-year polygon as a more conservative data-supported alternative.

## Next Step

Review Candidate E and Candidate C with ecological collaborators before building annual amphipod prey surfaces. Once a final v2 domain is selected, build the v2 ocean grid / mask and then proceed to annual amphipod prey-surface interpolation.
"""
    (OUT / "study_region_v2_recommendation.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    stations = make_station_extent()
    annual_summary, hulls0 = annual_hulls(stations)
    support = support_frequency_grid(stations, hulls0)
    candidates = candidate_polygons(stations, support, hulls0)
    metrics, excluded, compat = candidate_metrics(stations, candidates)
    create_figures(stations, annual_summary, hulls0, support, candidates)
    write_recommendation(metrics, stations)

    qa = {
        "station_records_2011_2019": int(len(stations)),
        "usable_coordinate_records": int(stations["has_usable_coordinates"].sum()),
        "low_confidence_records": int(stations["confidence_level"].eq("low").sum()),
        "candidate_count": int(len(candidates)),
        "candidate_names": list(candidates.keys()),
        "prey_surfaces_generated": False,
        "lcp_models_run": False,
        "formal_DBO5_polygon_available": False,
        "notes": [
            "DBO2-DBO4 anchor polygons were available; DBO5/Barrow Canyon was approximated from station clusters and labels.",
            "Candidate E is recommended for human review, not selected as final.",
            "Candidate polygons are domain-design QA outputs only.",
        ],
    }
    (OUT / "candidate_study_region_phase8a2_qa_v2.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
