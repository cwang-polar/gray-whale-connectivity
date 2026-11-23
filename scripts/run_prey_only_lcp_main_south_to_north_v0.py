"""Run prey-only LCP ensemble v0 for the main south-to-north route.

This is a smoke-test corridor workflow. It uses global-scaled prey proxy
surfaces, fixed ocean-only endpoints, and an ocean-only 8-neighbor graph. It
does not include sea ice, vessel disturbance, ship strike, or fatality layers.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
from dataclasses import dataclass
import heapq
import json
import math
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd


ROUTE_PAIR = "main_south_to_north"
OCEAN_MASK_PATH = PROJECT_ROOT / "data/processed/study_domain/ocean_mask_grid_10km_large_regional_bbox_v0.csv"
ENDPOINTS_PATH = PROJECT_ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north.csv"
PREY_SURFACE_GLOB = "prey_proxy_grid_idw_global_rolling5_*_target*.csv"
LCP_DIR = PROJECT_ROOT / "outputs/lcp"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
UTILITY_COL_CANDIDATES = (
    "combined_prey_proxy_final_global",
    "combined_prey_proxy_global",
)
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


@dataclass(frozen=True)
class GraphTopology:
    grid: pd.DataFrame
    node_index_by_grid_id: dict[str, int]
    grid_id_by_node_index: list[str]
    edge_pairs: list[tuple[int, int, float]]
    source_indices: list[int]
    destination_indices: list[int]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prey-only LCP v0 for fixed main south-to-north endpoints."
    )
    parser.add_argument("--max-pairs", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LCP_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ocean_mask = read_required_csv(OCEAN_MASK_PATH)
    endpoints = read_required_csv(ENDPOINTS_PATH)
    surfaces = sorted((PROJECT_ROOT / "outputs").glob(PREY_SURFACE_GLOB))
    if not surfaces:
        raise FileNotFoundError(
            "No global rolling prey-proxy surfaces found under outputs/. "
            "Run build_dominant_macroinfauna_prey_proxy.py first."
        )

    topology = build_graph_topology(ocean_mask=ocean_mask, endpoints=endpoints)
    all_path_rows = []
    route_rows = []
    qa_years = []
    target_grids: dict[int, pd.DataFrame] = {}

    for surface_path in surfaces:
        target_year = extract_target_year(surface_path)
        surface = pd.read_csv(surface_path)
        lcp_grid, utility_warning = prepare_year_grid(surface, topology.grid, target_year)
        warnings = list(topology.warnings)
        if utility_warning:
            warnings.append(utility_warning)

        adjacency = build_weighted_adjacency(topology.edge_pairs, lcp_grid["R_log"].to_numpy())
        lcp_count = np.zeros(len(lcp_grid), dtype=int)
        path_rows, year_warnings = run_year_paths(
            target_year=target_year,
            lcp_grid=lcp_grid,
            adjacency=adjacency,
            topology=topology,
            lcp_count=lcp_count,
            max_pairs=args.max_pairs,
        )
        warnings.extend(year_warnings)
        successful_path_count = len(path_rows)
        attempted_path_count = min(
            len(topology.source_indices) * len(topology.destination_indices),
            args.max_pairs,
        )
        lcp_grid["lcp_count"] = lcp_count
        if successful_path_count > 0:
            lcp_grid["lcp_frequency_normalized"] = lcp_grid["lcp_count"] / successful_path_count
        else:
            lcp_grid["lcp_frequency_normalized"] = 0.0
            warnings.append(f"target {target_year}: no successful paths.")

        output_grid = format_lcp_grid_output(lcp_grid, target_year)
        grid_path = LCP_DIR / f"prey_only_lcp_main_south_to_north_target{target_year}.csv"
        output_grid.to_csv(grid_path, index=False)
        target_grids[target_year] = output_grid

        route_summary = summarize_route_year(
            target_year=target_year,
            path_rows=path_rows,
            lcp_grid=output_grid,
            source_count=len(topology.source_indices),
            destination_count=len(topology.destination_indices),
            attempted_path_count=attempted_path_count,
            warnings=warnings,
        )
        route_rows.append(route_summary)
        qa_years.append(
            {
                "target_year": target_year,
                "successful_path_count": route_summary["successful_path_count"],
                "failed_path_count": route_summary["failed_path_count"],
                "max_lcp_frequency": route_summary["top_bottleneck_frequency"],
                "corridor_area_cells_freq_gt_0": route_summary["corridor_area_cells_freq_gt_0"],
                "corridor_area_cells_freq_gt_0_05": route_summary["corridor_area_cells_freq_gt_0_05"],
                "corridor_area_cells_freq_gt_0_10": route_summary["corridor_area_cells_freq_gt_0_10"],
                "mean_path_cost": route_summary["mean_path_cost"],
                "median_path_cost": route_summary["median_path_cost"],
                "mean_path_length_km": route_summary["mean_path_length_km"],
                "median_path_length_km": route_summary["median_path_length_km"],
                "warning_flags": route_summary["warning_flags"],
            }
        )
        all_path_rows.extend(path_rows)
        plot_lcp_target(
            lcp_grid=output_grid,
            endpoints=endpoints,
            target_year=target_year,
            figure_path=FIGURES_DIR / f"lcp_prey_only_main_south_to_north_target{target_year}.png",
        )

    path_summary = pd.DataFrame(all_path_rows)
    path_summary_path = LCP_DIR / "prey_only_main_south_to_north_path_summary_v0.csv"
    path_summary.to_csv(path_summary_path, index=False)
    route_summary = pd.DataFrame(route_rows).sort_values("target_year")
    route_summary_path = LCP_DIR / "prey_only_main_south_to_north_route_summary_v0.csv"
    route_summary.to_csv(route_summary_path, index=False)
    plot_lcp_panel(
        target_grids=target_grids,
        endpoints=endpoints,
        figure_path=FIGURES_DIR / "lcp_prey_only_main_south_to_north_all_targets.png",
    )

    qa = {
        "route_pair": ROUTE_PAIR,
        "graph_node_count": len(topology.grid),
        "graph_edge_count": len(topology.edge_pairs),
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "years": qa_years,
        "warnings": sorted(set(sum((row["warning_flags"].split("; ") for row in route_rows if row["warning_flags"]), []))),
    }
    qa_path = LCP_DIR / "prey_only_main_south_to_north_lcp_qa_v0.json"
    with qa_path.open("w", encoding="utf-8") as file:
        json.dump(qa, file, indent=2)
        file.write("\n")

    print(json.dumps(qa, indent=2))
    print(f"Wrote route summary: {route_summary_path}")
    print(f"Wrote path summary: {path_summary_path}")
    print(f"Wrote QA JSON: {qa_path}")


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def build_graph_topology(ocean_mask: pd.DataFrame, endpoints: pd.DataFrame) -> GraphTopology:
    warnings = []
    grid = ocean_mask[ocean_mask["ocean_flag"] == 1].copy().reset_index(drop=True)
    for col in ["grid_x", "grid_y"]:
        if col not in grid.columns:
            grid[col] = grid["grid_id"].apply(lambda value: parse_grid_id(value)[0 if col == "grid_x" else 1])
    grid["grid_id"] = grid["grid_id"].astype(str)
    node_index_by_grid_id = {grid_id: idx for idx, grid_id in enumerate(grid["grid_id"])}
    grid_id_by_node_index = list(grid["grid_id"])
    node_by_xy = {
        (int(row.grid_x), int(row.grid_y)): idx
        for idx, row in grid.iterrows()
    }
    edge_pairs = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for (gx, gy), idx in node_by_xy.items():
        for dx, dy in neighbor_offsets:
            neighbor = (gx + dx, gy + dy)
            if neighbor not in node_by_xy:
                continue
            jdx = node_by_xy[neighbor]
            if idx >= jdx:
                continue
            distance_km = math.hypot(
                float(grid.at[idx, "x"]) - float(grid.at[jdx, "x"]),
                float(grid.at[idx, "y"]) - float(grid.at[jdx, "y"]),
            ) / 1000.0
            edge_pairs.append((idx, jdx, distance_km))

    endpoints = endpoints.copy()
    endpoints["grid_id"] = endpoints["grid_id"].astype(str)
    endpoints = endpoints.merge(
        grid[["grid_id", "ocean_flag"]],
        on="grid_id",
        how="left",
        suffixes=("", "_mask"),
    )
    valid_endpoints = endpoints[endpoints["ocean_flag_mask"] == 1].copy()
    missing = endpoints[endpoints["ocean_flag_mask"].isna()]
    if not missing.empty:
        warnings.append(f"{len(missing)} endpoint cells are missing from ocean mask.")
    source_ids = valid_endpoints.loc[valid_endpoints["endpoint_type"] == "source", "grid_id"].tolist()
    dest_ids = valid_endpoints.loc[valid_endpoints["endpoint_type"] == "destination", "grid_id"].tolist()
    source_indices = [node_index_by_grid_id[grid_id] for grid_id in source_ids if grid_id in node_index_by_grid_id]
    destination_indices = [node_index_by_grid_id[grid_id] for grid_id in dest_ids if grid_id in node_index_by_grid_id]
    if len(source_indices) != len(source_ids):
        warnings.append("Some source endpoints are not present in graph nodes.")
    if len(destination_indices) != len(dest_ids):
        warnings.append("Some destination endpoints are not present in graph nodes.")

    return GraphTopology(
        grid=grid,
        node_index_by_grid_id=node_index_by_grid_id,
        grid_id_by_node_index=grid_id_by_node_index,
        edge_pairs=edge_pairs,
        source_indices=source_indices,
        destination_indices=destination_indices,
        warnings=warnings,
    )


def prepare_year_grid(surface: pd.DataFrame, ocean_grid: pd.DataFrame, target_year: int) -> tuple[pd.DataFrame, str | None]:
    utility_col = next((col for col in UTILITY_COL_CANDIDATES if col in surface.columns), None)
    if utility_col is None:
        raise ValueError(
            f"No final global utility column found for target {target_year}. "
            f"Tried: {UTILITY_COL_CANDIDATES}"
        )
    merged = ocean_grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"))
    merged = merged[merged["ocean_flag"] == 1].copy().reset_index(drop=True)
    warning = None
    if merged[utility_col].isna().any():
        warning = f"target {target_year}: {int(merged[utility_col].isna().sum())} ocean utility cells filled from NaN to 0.001."
    merged["utility_original_nan_flag"] = merged[utility_col].isna().astype(int)
    merged["U_prey_proxy_global"] = pd.to_numeric(merged[utility_col], errors="coerce").fillna(0.001)
    merged["U_prey_proxy_global"] = merged["U_prey_proxy_global"].clip(lower=0.001, upper=1.0)
    merged["R_linear"] = 1.0 - merged["U_prey_proxy_global"] + 0.001
    merged["R_reciprocal"] = 1.0 / (merged["U_prey_proxy_global"] + 0.001)
    merged["R_log"] = -np.log(merged["U_prey_proxy_global"] + 0.001)
    if "observed_flag" not in merged.columns:
        merged["observed_flag"] = np.nan
    if "distance_to_nearest_observed_km" not in merged.columns:
        merged["distance_to_nearest_observed_km"] = np.nan
    return merged, warning


def build_weighted_adjacency(
    edge_pairs: list[tuple[int, int, float]],
    resistance: np.ndarray,
) -> list[list[tuple[int, float, float]]]:
    adjacency = [[] for _ in range(len(resistance))]
    for i, j, distance_km in edge_pairs:
        cost = float((resistance[i] + resistance[j]) / 2.0 * distance_km)
        adjacency[i].append((j, cost, distance_km))
        adjacency[j].append((i, cost, distance_km))
    return adjacency


def run_year_paths(
    target_year: int,
    lcp_grid: pd.DataFrame,
    adjacency: list[list[tuple[int, float, float]]],
    topology: GraphTopology,
    lcp_count: np.ndarray,
    max_pairs: int,
) -> tuple[list[dict], list[str]]:
    warnings = []
    all_pairs = [
        (source_idx, dest_idx)
        for source_idx in topology.source_indices
        for dest_idx in topology.destination_indices
    ]
    if len(all_pairs) > max_pairs:
        warnings.append(f"target {target_year}: capped path pairs from {len(all_pairs)} to {max_pairs}.")
        all_pairs = all_pairs[:max_pairs]
    destinations_by_source: dict[int, list[int]] = {}
    for source_idx, dest_idx in all_pairs:
        destinations_by_source.setdefault(source_idx, []).append(dest_idx)

    path_rows = []
    for source_idx, dest_indices in destinations_by_source.items():
        distances, previous = dijkstra(adjacency, source_idx)
        for dest_idx in dest_indices:
            if not np.isfinite(distances[dest_idx]):
                continue
            path = reconstruct_path(previous, source_idx, dest_idx)
            if not path:
                continue
            for node_idx in path:
                lcp_count[node_idx] += 1
            path_df = lcp_grid.iloc[path]
            path_rows.append(
                {
                    "target_year": target_year,
                    "source_grid_id": topology.grid_id_by_node_index[source_idx],
                    "destination_grid_id": topology.grid_id_by_node_index[dest_idx],
                    "path_cost": float(distances[dest_idx]),
                    "path_length_km": path_length_km(path, lcp_grid),
                    "path_cell_count": int(len(path)),
                    "path_nan_cell_count": int(path_df["utility_original_nan_flag"].sum()),
                    "path_mean_U": float(path_df["U_prey_proxy_global"].mean()),
                    "path_min_U": float(path_df["U_prey_proxy_global"].min()),
                }
            )
    failed = len(all_pairs) - len(path_rows)
    if failed:
        warnings.append(f"target {target_year}: {failed} failed/disconnected source-destination paths.")
    return path_rows, warnings


def dijkstra(
    adjacency: list[list[tuple[int, float, float]]],
    source: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(adjacency)
    distances = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    distances[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > distances[node]:
            continue
        for neighbor, edge_cost, _distance_km in adjacency[node]:
            new_dist = current_dist + edge_cost
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))
    return distances, previous


def reconstruct_path(previous: np.ndarray, source: int, destination: int) -> list[int]:
    path = []
    node = destination
    while node != -1:
        path.append(int(node))
        if node == source:
            return list(reversed(path))
        node = int(previous[node])
    return []


def path_length_km(path: list[int], lcp_grid: pd.DataFrame) -> float:
    if len(path) < 2:
        return 0.0
    x = lcp_grid.iloc[path]["x"].to_numpy(dtype=float)
    y = lcp_grid.iloc[path]["y"].to_numpy(dtype=float)
    return float(np.hypot(np.diff(x), np.diff(y)).sum() / 1000.0)


def format_lcp_grid_output(lcp_grid: pd.DataFrame, target_year: int) -> pd.DataFrame:
    output = lcp_grid.copy()
    output["target_year"] = target_year
    output["route_pair"] = ROUTE_PAIR
    keep = [
        "grid_id",
        "x",
        "y",
        "lon",
        "lat",
        "target_year",
        "route_pair",
        "U_prey_proxy_global",
        "R_log",
        "lcp_count",
        "lcp_frequency_normalized",
        "utility_original_nan_flag",
        "observed_flag",
        "distance_to_nearest_observed_km",
        "ocean_flag",
    ]
    return output[keep]


def summarize_route_year(
    target_year: int,
    path_rows: list[dict],
    lcp_grid: pd.DataFrame,
    source_count: int,
    destination_count: int,
    attempted_path_count: int,
    warnings: list[str],
) -> dict:
    path_df = pd.DataFrame(path_rows)
    freq = lcp_grid["lcp_frequency_normalized"]
    if path_df.empty:
        path_cost = pd.Series(dtype=float)
        path_length = pd.Series(dtype=float)
        path_nan = pd.Series(dtype=float)
    else:
        path_cost = path_df["path_cost"]
        path_length = path_df["path_length_km"]
        path_nan = path_df["path_nan_cell_count"]
    top_idx = freq.idxmax() if len(freq) else None
    return {
        "target_year": target_year,
        "source_endpoint_count": source_count,
        "destination_endpoint_count": destination_count,
        "attempted_path_count": attempted_path_count,
        "successful_path_count": int(len(path_rows)),
        "failed_path_count": int(attempted_path_count - len(path_rows)),
        "mean_path_cost": none_if_nan(path_cost.mean()),
        "median_path_cost": none_if_nan(path_cost.median()),
        "min_path_cost": none_if_nan(path_cost.min()),
        "max_path_cost": none_if_nan(path_cost.max()),
        "mean_path_length_km": none_if_nan(path_length.mean()),
        "median_path_length_km": none_if_nan(path_length.median()),
        "mean_path_nan_cell_count": none_if_nan(path_nan.mean()),
        "corridor_area_cells_freq_gt_0": int((freq > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int((freq > 0.05).sum()),
        "corridor_area_cells_freq_gt_0_10": int((freq > 0.10).sum()),
        "top_bottleneck_frequency": none_if_nan(freq.max()),
        "top_bottleneck_grid_id": None if top_idx is None else str(lcp_grid.at[top_idx, "grid_id"]),
        "warning_flags": "; ".join(sorted(set(warnings))),
    }


def plot_lcp_target(
    lcp_grid: pd.DataFrame,
    endpoints: pd.DataFrame,
    target_year: int,
    figure_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    add_lcp_scatter(ax, lcp_grid, show_colorbar=True)
    draw_context(ax)
    draw_endpoints(ax, endpoints)
    ax.set_title(f"Prey-only LCP frequency ({ROUTE_PAIR}, target {target_year})")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_lcp_panel(
    target_grids: dict[int, pd.DataFrame],
    endpoints: pd.DataFrame,
    figure_path: Path,
) -> None:
    if not target_grids:
        return
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    scatter = None
    for ax, target_year in zip(axes_flat, sorted(target_grids)):
        grid = target_grids[target_year]
        scatter = ax.scatter(
            grid["lon"],
            grid["lat"],
            c=grid["lcp_frequency_normalized"],
            s=5,
            cmap="magma",
            vmin=0,
            vmax=max(0.001, max(g["lcp_frequency_normalized"].max() for g in target_grids.values())),
            linewidth=0,
        )
        draw_context(ax, label=False)
        draw_endpoints(ax, endpoints, label=False)
        ax.set_title(f"target {target_year}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes_flat.tolist(), label="LCP frequency", shrink=0.76)
    fig.suptitle("Prey-only LCP ensemble frequency across rolling target years", y=0.995)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_lcp_scatter(ax: plt.Axes, lcp_grid: pd.DataFrame, show_colorbar: bool) -> None:
    sc = ax.scatter(
        lcp_grid["lon"],
        lcp_grid["lat"],
        c=lcp_grid["lcp_frequency_normalized"],
        s=8,
        cmap="magma",
        vmin=0,
        vmax=max(0.001, lcp_grid["lcp_frequency_normalized"].max()),
        linewidth=0,
    )
    if show_colorbar:
        ax.figure.colorbar(sc, ax=ax, label="LCP frequency")


def draw_endpoints(ax: plt.Axes, endpoints: pd.DataFrame, label: bool = True) -> None:
    source = endpoints[endpoints["endpoint_type"] == "source"]
    dest = endpoints[endpoints["endpoint_type"] == "destination"]
    ax.scatter(source["lon"], source["lat"], s=28, color="#2ca02c", edgecolor="black", label="source" if label else None)
    ax.scatter(dest["lon"], dest["lat"], s=28, color="#9467bd", edgecolor="black", label="destination" if label else None)


def draw_context(ax: plt.Axes, label: bool = True) -> None:
    add_bbox_patch(ax, {
        "lon_min": -176.5,
        "lon_max": -157.0,
        "lat_min": 63.0,
        "lat_max": 73.0,
    }, "study domain" if label else None, "#444444")
    for name, coords in DBO_ANCHORS.items():
        lonlat = [(lon, lat) for lat, lon in coords]
        patch = MplPolygon(lonlat, closed=True, facecolor="none", edgecolor="#333333", linewidth=0.8, label=name if label else None)
        ax.add_patch(patch)
    ax.set_xlim(-177.5, -156.0)
    ax.set_ylim(62.5, 73.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if label:
        ax.legend(loc="lower right", fontsize=7)


def add_bbox_patch(ax: plt.Axes, bbox: dict[str, float], label: str | None, color: str) -> None:
    coords = [
        (bbox["lon_min"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_min"]),
        (bbox["lon_max"], bbox["lat_max"]),
        (bbox["lon_min"], bbox["lat_max"]),
    ]
    ax.add_patch(MplPolygon(coords, closed=True, facecolor="none", edgecolor=color, linestyle="--", linewidth=1.0, label=label))


def extract_target_year(path: Path) -> int:
    match = re.search(r"target(\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not parse target year from {path}")
    return int(match.group(1))


def parse_grid_id(value: str) -> tuple[int, int]:
    nums = re.findall(r"-?\d+", str(value))
    if len(nums) < 2:
        raise ValueError(f"Could not parse grid_id: {value}")
    return int(nums[0]), int(nums[1])


def none_if_nan(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
