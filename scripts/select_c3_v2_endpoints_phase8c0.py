"""Phase 8C.0 C3 v2 endpoint selection and LCP-readiness QA.

This script selects fixed source/destination endpoint candidates for future C3
v2 LCP runs. It does not run weighted LCP or any connectivity model.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
from collections import deque
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from shapely.geometry import shape

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_ocean_mask_and_endpoints as old_endpoint_workflow  # noqa: E402


OUT = PROJECT_ROOT / "outputs" / "prey_v2"
FIG = PROJECT_ROOT / "outputs" / "figures" / "prey_v2"
STUDY = PROJECT_ROOT / "data" / "processed" / "study_domain"

GRID = STUDY / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
C3_DOMAIN = STUDY / "study_domain_C3_recurring_support_v2.geojson"
LAND_SHP = PROJECT_ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

ENDPOINTS_RAW = STUDY / "fixed_endpoint_candidates_main_south_to_north_C3_v2.csv"
ENDPOINTS_CLEAN = STUDY / "fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"

TARGET_YEARS = list(range(2011, 2020))
VARIABLES = ["total_kJm2_frozen", "total_gww", "total_gC"]
MAX_ENDPOINTS = 40
MIN_SPACING_KM = 25

SOURCE_RULE = "C3 ocean cells with latitude <=65N, ranked by pooled benthic amphipod full-trailing utility"
DESTINATION_RULE = "Old northern terminus bbox clipped to C3 largest connected component, ranked by pooled benthic amphipod full-trailing utility"

OLD_DESTINATION_ZONE = {
    "lat_min": 70.3,
    "lat_max": 73.0,
    "lon_min": -166.5,
    "lon_max": -157.0,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    STUDY.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(GRID)
    grid["grid_id"] = grid["grid_id"].astype(str)
    grid = grid[grid["ocean_flag"] == 1].copy()
    components, edge_pairs, adjacency = build_components_and_graph(grid)
    largest_component_size = int(components["component_size"].max())
    grid = grid.merge(components, on="grid_id", how="left")
    grid["in_largest_component"] = grid["component_size"].eq(largest_component_size)
    ranking = build_pooled_full_trailing_ranking()
    grid = grid.merge(ranking, on="grid_id", how="left")

    source_pool = grid[grid["lat"].le(65.0) & grid["in_largest_component"]].copy()
    dest_pool = grid[in_bbox(grid, OLD_DESTINATION_ZONE) & grid["in_largest_component"]].copy()
    source = select_endpoints(source_pool, "source")
    dest = select_endpoints(dest_pool, "destination")
    endpoints = pd.concat([source, dest], ignore_index=True)
    endpoints["route_pair"] = "main_south_to_north_C3_v2"
    endpoints["selection_version"] = "phase8c0_B_source_clean_north"
    endpoints["source_rule"] = SOURCE_RULE
    endpoints["destination_rule"] = DESTINATION_RULE
    endpoints = endpoints[
        [
            "grid_id",
            "grid_x",
            "grid_y",
            "x",
            "y",
            "lon",
            "lat",
            "endpoint_type",
            "pooled_brian_amphipod_utility",
            "rank_order_within_type",
            "connected_component_id",
            "component_size",
            "in_largest_component",
            "ocean_flag",
            "route_pair",
            "selection_version",
            "source_rule",
            "destination_rule",
        ]
    ]
    endpoints.to_csv(ENDPOINTS_RAW, index=False)
    endpoints.to_csv(ENDPOINTS_CLEAN, index=False)

    pool = pd.concat(
        [
            source_pool.assign(endpoint_type="source"),
            dest_pool.assign(endpoint_type="destination"),
        ],
        ignore_index=True,
    )
    pool.to_csv(OUT / "C3_v2_endpoint_selection_pool_cells_phase8c0.csv", index=False)

    endpoint_summary = summarize_endpoints(endpoints, source_pool, dest_pool)
    endpoint_summary.to_csv(OUT / "C3_v2_endpoint_selection_summary_phase8c0.csv", index=False)

    graph_summary = summarize_graph_readiness(grid, components, edge_pairs, adjacency, endpoints)
    graph_summary.to_csv(OUT / "C3_v2_graph_readiness_summary_phase8c0.csv", index=False)

    qa = build_qa(endpoint_summary, graph_summary, grid, components, edge_pairs, endpoints, source_pool, dest_pool)
    (OUT / "C3_v2_endpoint_selection_qa_phase8c0.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    c3_geom = read_c3_geometry()
    land_features = old_endpoint_workflow.read_polygon_shapefile(LAND_SHP)
    plot_endpoints(grid, c3_geom, land_features, source_pool, dest_pool, endpoints)

    print(
        json.dumps(
            {
                "phase": "Phase 8C.0",
                "raw_endpoints": str(ENDPOINTS_RAW),
                "clean_endpoints": str(ENDPOINTS_CLEAN),
                "source_endpoint_count": int((endpoints["endpoint_type"] == "source").sum()),
                "destination_endpoint_count": int((endpoints["endpoint_type"] == "destination").sum()),
                "attempted_future_path_count": int(
                    (endpoints["endpoint_type"] == "source").sum()
                    * (endpoints["endpoint_type"] == "destination").sum()
                ),
                "no_lcp_models_run": True,
            },
            indent=2,
        )
    )


def build_pooled_full_trailing_ranking() -> pd.DataFrame:
    frames = []
    for variable in VARIABLES:
        for year in TARGET_YEARS:
            path = OUT / f"amphipod_prey_surface_{variable}_target{year}_full_trailing_C3_v2.csv"
            df = pd.read_csv(path, usecols=["grid_id", "U_amphipod_prey_global_q05_q95"])
            frames.append(df.rename(columns={"U_amphipod_prey_global_q05_q95": f"{variable}_{year}"}))
    pooled = frames[0]
    for frame in frames[1:]:
        pooled = pooled.merge(frame, on="grid_id", how="outer")
    value_cols = [c for c in pooled.columns if c != "grid_id"]
    pooled["pooled_brian_amphipod_utility"] = pooled[value_cols].mean(axis=1, skipna=True)
    return pooled[["grid_id", "pooled_brian_amphipod_utility"]]


def in_bbox(df: pd.DataFrame, bbox: dict[str, float]) -> pd.Series:
    return df["lon"].between(bbox["lon_min"], bbox["lon_max"]) & df["lat"].between(
        bbox["lat_min"], bbox["lat_max"]
    )


def select_endpoints(pool: pd.DataFrame, endpoint_type: str) -> pd.DataFrame:
    ranked = pool.dropna(subset=["pooled_brian_amphipod_utility"]).sort_values(
        "pooled_brian_amphipod_utility", ascending=False
    )
    rows = []
    selected_xy: list[tuple[float, float]] = []
    for _, row in ranked.iterrows():
        xy = (float(row["x"]), float(row["y"]))
        if all(math.hypot(xy[0] - sx, xy[1] - sy) >= MIN_SPACING_KM * 1000.0 for sx, sy in selected_xy):
            rows.append(row)
            selected_xy.append(xy)
        if len(rows) >= MAX_ENDPOINTS:
            break
    selected = pd.DataFrame(rows).copy()
    selected["endpoint_type"] = endpoint_type
    selected["rank_order_within_type"] = range(1, len(selected) + 1)
    return selected


def build_components_and_graph(grid: pd.DataFrame):
    grid = grid.copy().reset_index(drop=True)
    node_by_xy = {(int(r.grid_x), int(r.grid_y)): i for i, r in grid.iterrows()}
    adjacency = [[] for _ in range(len(grid))]
    edges = set()
    for (gx, gy), idx in node_by_xy.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = node_by_xy.get((gx + dx, gy + dy))
                if n is None:
                    continue
                adjacency[idx].append(n)
                edges.add(tuple(sorted((idx, n))))
    component_ids = np.full(len(grid), -1, dtype=int)
    sizes = []
    for start in range(len(grid)):
        if component_ids[start] >= 0:
            continue
        cid = len(sizes)
        stack = [start]
        component_ids[start] = cid
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nb in adjacency[node]:
                if component_ids[nb] < 0:
                    component_ids[nb] = cid
                    stack.append(nb)
        sizes.append(size)
    components = grid[["grid_id"]].copy()
    components["connected_component_id"] = component_ids
    components["component_size"] = [sizes[cid] for cid in component_ids]
    return components, sorted(edges), adjacency


def summarize_endpoints(endpoints: pd.DataFrame, source_pool: pd.DataFrame, dest_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint_type, pool in [("source", source_pool), ("destination", dest_pool)]:
        sub = endpoints[endpoints["endpoint_type"] == endpoint_type]
        rows.append(
            {
                "endpoint_type": endpoint_type,
                "candidate_pool_cells": len(pool),
                "selected_endpoint_count": len(sub),
                "selected_clean_largest_component_count": int(sub["in_largest_component"].sum()),
                "lon_min": float(sub["lon"].min()),
                "lon_max": float(sub["lon"].max()),
                "lat_min": float(sub["lat"].min()),
                "lat_max": float(sub["lat"].max()),
                "mean_pooled_brian_amphipod_utility": float(sub["pooled_brian_amphipod_utility"].mean()),
                "median_pooled_brian_amphipod_utility": float(sub["pooled_brian_amphipod_utility"].median()),
                "min_pairwise_spacing_km": float(min_pairwise_spacing_km(sub)),
            }
        )
    return pd.DataFrame(rows)


def min_pairwise_spacing_km(df: pd.DataFrame) -> float:
    xy = df[["x", "y"]].to_numpy(dtype=float)
    best = math.inf
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            best = min(best, math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) / 1000.0)
    return best


def summarize_graph_readiness(grid, components, edge_pairs, adjacency, endpoints) -> pd.DataFrame:
    largest_component_size = int(components["component_size"].max())
    largest_component_id = int(components.loc[components["component_size"].idxmax(), "connected_component_id"])
    source_count = int((endpoints["endpoint_type"] == "source").sum())
    destination_count = int((endpoints["endpoint_type"] == "destination").sum())
    source_destination_pairs = source_count * destination_count
    reachable_pairs = count_unweighted_reachable_pairs(grid, adjacency, endpoints)
    return pd.DataFrame(
        [
            {
                "graph_node_count": len(grid),
                "graph_edge_count": len(edge_pairs),
                "connected_component_count": int(components["connected_component_id"].nunique()),
                "largest_component_id": largest_component_id,
                "largest_component_size": largest_component_size,
                "percent_nodes_in_largest_component": float(100 * largest_component_size / len(grid)),
                "source_endpoint_count": source_count,
                "destination_endpoint_count": destination_count,
                "source_destination_pair_count": source_destination_pairs,
                "unweighted_reachable_source_destination_pairs": reachable_pairs,
                "all_endpoint_pairs_unweighted_reachable": bool(reachable_pairs == source_destination_pairs),
            }
        ]
    )


def count_unweighted_reachable_pairs(grid: pd.DataFrame, adjacency, endpoints: pd.DataFrame) -> int:
    node_by_grid_id = {str(gid): i for i, gid in enumerate(grid["grid_id"])}
    sources = [node_by_grid_id[str(gid)] for gid in endpoints.loc[endpoints["endpoint_type"] == "source", "grid_id"]]
    destinations = {
        node_by_grid_id[str(gid)]
        for gid in endpoints.loc[endpoints["endpoint_type"] == "destination", "grid_id"]
    }
    reachable = 0
    for source in sources:
        seen = np.zeros(len(grid), dtype=bool)
        q = deque([source])
        seen[source] = True
        while q:
            node = q.popleft()
            for nb in adjacency[node]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        reachable += sum(1 for dest in destinations if seen[dest])
    return int(reachable)


def build_qa(endpoint_summary, graph_summary, grid, components, edge_pairs, endpoints, source_pool, dest_pool) -> dict:
    warnings = []
    if int((endpoints["endpoint_type"] == "source").sum()) != MAX_ENDPOINTS:
        warnings.append("source endpoint count is not 40.")
    if int((endpoints["endpoint_type"] == "destination").sum()) != MAX_ENDPOINTS:
        warnings.append("destination endpoint count is not 40.")
    if not endpoints["in_largest_component"].all():
        warnings.append("not all selected endpoints are in the C3 largest connected component.")
    source = endpoints[endpoints["endpoint_type"] == "source"]
    if not source["lat"].le(65.0).all():
        warnings.append("one or more source endpoints are north of 65N.")
    if not bool(graph_summary.loc[0, "all_endpoint_pairs_unweighted_reachable"]):
        warnings.append("not all source-destination pairs are reachable on the unweighted C3 ocean graph.")
    return {
        "phase": "Phase 8C.0",
        "purpose": "C3 v2 endpoint selection and graph-readiness QA",
        "no_lcp_models_run": True,
        "no_connectivity_models_run": True,
        "target_future_route_pair": "main_south_to_north_C3_v2",
        "selection_version": "phase8c0_B_source_clean_north",
        "source_rule": SOURCE_RULE,
        "destination_rule": DESTINATION_RULE,
        "ranking_surface": "pooled benthic full-trailing amphipod utility, averaged across kJm2.frozen, gWW, gC and 2011-2019",
        "max_endpoints_per_type": MAX_ENDPOINTS,
        "minimum_spacing_km": MIN_SPACING_KM,
        "source_pool_cells": int(len(source_pool)),
        "destination_pool_cells": int(len(dest_pool)),
        "endpoint_summary": endpoint_summary.to_dict(orient="records"),
        "graph_summary": graph_summary.to_dict(orient="records")[0],
        "component_size_distribution": components["component_size"].value_counts().sort_index().to_dict(),
        "warnings": warnings,
        "outputs": {
            "raw_endpoints": str(ENDPOINTS_RAW.relative_to(PROJECT_ROOT)),
            "clean_endpoints": str(ENDPOINTS_CLEAN.relative_to(PROJECT_ROOT)),
            "endpoint_summary": "outputs/prey_v2/C3_v2_endpoint_selection_summary_phase8c0.csv",
            "graph_summary": "outputs/prey_v2/C3_v2_graph_readiness_summary_phase8c0.csv",
            "figure": "outputs/figures/prey_v2/C3_v2_endpoint_selection_phase8c0.png",
        },
    }


def read_c3_geometry():
    data = json.loads(C3_DOMAIN.read_text())
    geoms = [shape(feature["geometry"]) for feature in data["features"]]
    geom = geoms[0]
    for other in geoms[1:]:
        geom = geom.union(other)
    return geom


def plot_endpoints(grid, c3_geom, land_features, source_pool, dest_pool, endpoints) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    setup_map(ax, land_features, c3_geom)
    ax.scatter(grid["lon"], grid["lat"], s=4, color="#c7d4df", alpha=0.23, linewidths=0, label="C3 grid")
    ax.scatter(source_pool["lon"], source_pool["lat"], s=8, color="#1b9e77", alpha=0.16, linewidths=0, label="source pool")
    ax.scatter(dest_pool["lon"], dest_pool["lat"], s=8, color="#6a3d9a", alpha=0.14, linewidths=0, label="destination pool")
    source = endpoints[endpoints["endpoint_type"] == "source"]
    dest = endpoints[endpoints["endpoint_type"] == "destination"]
    ax.scatter(source["lon"], source["lat"], s=38, marker="o", facecolor="#1b9e77", edgecolor="white", linewidth=0.5, label="source endpoints")
    ax.scatter(dest["lon"], dest["lat"], s=42, marker="^", facecolor="#6a3d9a", edgecolor="white", linewidth=0.5, label="destination endpoints")
    handles = [
        Line2D([0], [0], color="#263238", linewidth=1.6, label="C3 domain"),
        Line2D([0], [0], color="#cc4c02", linewidth=1.2, label="DBO boxes / DBO5 label"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b9e77", markeredgecolor="white", label="source endpoints"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#6a3d9a", markeredgecolor="white", label="destination endpoints"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.92)
    ax.set_title("C3 v2 source and destination endpoints for future LCP", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "Source: high-amphipod southern C3 entry cells <=65N. Destination: old northern terminus zone clipped to clean C3 component. No LCP models run.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG / "C3_v2_endpoint_selection_phase8c0.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup_map(ax, land_features, c3_geom) -> None:
    ax.set_facecolor("#eef6fb")
    plot_land(ax, land_features)
    plot_geom_outline(ax, c3_geom, color="#263238", linewidth=1.6)
    draw_dbo(ax)
    add_labels(ax)
    ax.set_xlim(-176.1, -156.4)
    ax.set_ylim(62.7, 73.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#ffffff", linewidth=0.6, alpha=0.65)


def plot_land(ax, land_features) -> None:
    bbox = {"lon_min": -178, "lon_max": -155, "lat_min": 61.5, "lat_max": 74.5}
    for feature in land_features:
        if not old_endpoint_workflow.bbox_intersects(feature.bbox, old_endpoint_workflow.bbox_tuple_with_padding(bbox, 1.0)):
            continue
        for ring in feature.rings:
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            if max(xs) < bbox["lon_min"] or min(xs) > bbox["lon_max"] or max(ys) < bbox["lat_min"] or min(ys) > bbox["lat_max"]:
                continue
            sampled = old_endpoint_workflow.downsample_ring(ring, 1200)
            ax.add_patch(MplPolygon(sampled, closed=True, facecolor="#d0d0d0", edgecolor="#8a8a8a", linewidth=0.35, zorder=0))


def plot_geom_outline(ax, geom, **kwargs) -> None:
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, **kwargs)


def draw_dbo(ax) -> None:
    for name, coords in old_endpoint_workflow.DBO_ANCHORS.items():
        lonlat = [(lon, lat) for lat, lon in coords]
        ax.add_patch(MplPolygon(lonlat, closed=True, facecolor="none", edgecolor="#cc4c02", linewidth=1.1))
        cx = sum(x for x, _ in lonlat) / len(lonlat)
        cy = sum(y for _, y in lonlat) / len(lonlat)
        ax.text(cx, cy, name, fontsize=7, color="#cc4c02", ha="center", va="center", fontweight="bold")
    ax.text(-158.7, 71.8, "DBO5 /\nBarrow Canyon", fontsize=7, color="#cc4c02", ha="center")


def add_labels(ax) -> None:
    labels = [
        ("St. Lawrence\nIsland", -170.5, 63.9),
        ("Bering\nStrait", -168.6, 65.6),
        ("Southern\nChukchi", -169.7, 68.2),
        ("Wainwright", -160.1, 70.8),
        ("Chirikov\nBasin", -170.9, 64.9),
    ]
    for text, lon, lat in labels:
        ax.text(lon, lat, text, fontsize=7, color="#455a64", ha="center", va="center")


if __name__ == "__main__":
    main()
