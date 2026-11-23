"""Phase 8C prey-only LCP with benthic amphipod full-trailing C3 surfaces.

Runs prey-only LCP ensembles for kJm2.frozen, gWW, and gC on the C3 v2 grid
using the Phase 8C.0 fixed endpoints. No ice or ship layers are used.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base  # noqa: E402
import build_ocean_mask_and_endpoints as old_context  # noqa: E402


GRID = PROJECT_ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = PROJECT_ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
LCP = PROJECT_ROOT / "outputs/lcp_v2"
FIG = PROJECT_ROOT / "outputs/figures/lcp_v2"
PREY = PROJECT_ROOT / "outputs/prey_v2"
OLD_LCP = PROJECT_ROOT / "outputs/lcp"

TARGET_YEARS = list(range(2011, 2020))
VARIABLES = {
    "total_kJm2_frozen": {
        "label": "kJm2_frozen",
        "pretty": "kJ m-2 frozen",
        "main": True,
    },
    "total_gww": {
        "label": "gww",
        "pretty": "gWW",
        "main": False,
    },
    "total_gC": {
        "label": "gC",
        "pretty": "gC",
        "main": False,
    },
}
ROUTE_PAIR = "main_south_to_north_C3_v2"
EPSILON = 0.001


def main() -> None:
    LCP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ocean = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    topology = lcp_base.build_graph_topology(ocean_mask=ocean, endpoints=endpoints)

    all_grids: dict[str, dict[int, pd.DataFrame]] = {}
    all_path_summaries = []
    route_rows = []

    for variable, meta in VARIABLES.items():
        yearly_grids: dict[int, pd.DataFrame] = {}
        variable_path_rows = []
        for year in TARGET_YEARS:
            surface = read_surface(variable, year)
            lcp_grid, warning = prepare_lcp_grid(surface, topology.grid, variable, year)
            warnings = list(topology.warnings)
            if warning:
                warnings.append(warning)
            adjacency = build_weighted_csr(topology.edge_pairs, lcp_grid["R_log"].to_numpy())
            lcp_count = np.zeros(len(lcp_grid), dtype=int)
            path_rows, path_warnings = run_year_paths(variable, year, lcp_grid, adjacency, topology, lcp_count)
            warnings.extend(path_warnings)
            successful = len(path_rows)
            attempted = len(topology.source_indices) * len(topology.destination_indices)
            lcp_grid["lcp_count"] = lcp_count
            lcp_grid["lcp_frequency_normalized"] = lcp_count / successful if successful else 0.0
            out_grid = format_grid_output(lcp_grid, variable, year)
            out_grid.to_csv(LCP / f"prey_only_brian_{meta['label']}_C3_lcp_target{year}_v2.csv", index=False)
            yearly_grids[year] = out_grid
            variable_path_rows.extend(path_rows)
            route_rows.append(summarize_route(variable, year, path_rows, out_grid, attempted, topology, warnings))

        path_summary = pd.DataFrame(variable_path_rows)
        path_summary.to_csv(LCP / f"prey_only_brian_{meta['label']}_C3_path_summary_v2.csv", index=False)
        all_path_summaries.append(path_summary)
        all_grids[variable] = yearly_grids
        plot_lcp_panel(yearly_grids, endpoints, FIG / f"prey_only_brian_{meta['label']}_C3_lcp_all_years_v2.png", meta["pretty"])

    route_summary = pd.DataFrame(route_rows).sort_values(["prey_variable", "target_year"])
    route_summary.to_csv(LCP / "prey_only_brian_C3_route_summary_v2.csv", index=False)

    variable_comparison = compare_variables(all_grids, route_summary)
    variable_comparison.to_csv(LCP / "prey_only_brian_C3_variable_comparison_v2.csv", index=False)

    old_comparison = compare_old_proxy(all_grids["total_kJm2_frozen"], route_summary)
    old_comparison.to_csv(LCP / "prey_only_old_proxy_vs_brian_C3_comparison_v2.csv", index=False)

    top, persistence = bottleneck_outputs(all_grids)
    top.to_csv(LCP / "prey_only_brian_C3_top_bottleneck_cells_v2.csv", index=False)
    persistence.to_csv(LCP / "prey_only_brian_C3_bottleneck_persistence_v2.csv", index=False)

    make_summary_figures(route_summary, variable_comparison, old_comparison, persistence)
    qa = build_qa(topology, route_summary)
    (LCP / "prey_only_brian_C3_lcp_qa_v2.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "phase": "Phase 8C",
        "variables": list(VARIABLES),
        "target_years": TARGET_YEARS,
        "route_summary": str(LCP / "prey_only_brian_C3_route_summary_v2.csv"),
        "qa": str(LCP / "prey_only_brian_C3_lcp_qa_v2.json"),
        "no_ice_layer_used": True,
        "no_ship_layer_used": True,
    }, indent=2))


def read_surface(variable: str, year: int) -> pd.DataFrame:
    path = PREY / f"amphipod_prey_surface_{variable}_target{year}_full_trailing_C3_v2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def prepare_lcp_grid(surface: pd.DataFrame, grid: pd.DataFrame, variable: str, year: int) -> tuple[pd.DataFrame, str | None]:
    utility_col = "U_amphipod_prey_global_q05_q95"
    merged = grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"))
    merged = merged[merged["ocean_flag"] == 1].copy().reset_index(drop=True)
    missing = int(merged[utility_col].isna().sum())
    warning = None
    if missing:
        warning = f"{variable} target {year}: {missing} missing utility cells filled to 0.001."
    u = pd.to_numeric(merged[utility_col], errors="coerce").fillna(EPSILON).clip(lower=EPSILON, upper=1.0)
    merged["U_amphipod_prey_global_q05_q95"] = u
    # The log transform gives a tiny negative value when U is exactly 1
    # because of epsilon. Resistance must be non-negative for Dijkstra.
    merged["R_log"] = np.maximum(-np.log(u + EPSILON), 0.0)
    merged["prey_variable"] = variable
    merged["target_year"] = year
    merged["utility_original_nan_flag"] = missing
    return merged, warning


def build_weighted_csr(edge_pairs, resistance: np.ndarray) -> csr_matrix:
    rows = []
    cols = []
    data = []
    for i, j, distance_km in edge_pairs:
        cost = float((resistance[i] + resistance[j]) / 2.0 * distance_km)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([cost, cost])
    return csr_matrix((data, (rows, cols)), shape=(len(resistance), len(resistance)))


def run_year_paths(variable: str, year: int, lcp_grid, adjacency: csr_matrix, topology, lcp_count):
    path_rows = []
    warnings = []
    distances, predecessors = sparse_dijkstra(
        adjacency,
        directed=False,
        indices=topology.source_indices,
        return_predecessors=True,
    )
    for source_pos, source_idx in enumerate(topology.source_indices):
        for dest_idx in topology.destination_indices:
            dist = distances[source_pos, dest_idx]
            if not np.isfinite(dist):
                continue
            path = reconstruct_sparse_path(predecessors[source_pos], source_idx, dest_idx)
            if not path:
                continue
            for node_idx in path:
                lcp_count[node_idx] += 1
            path_df = lcp_grid.iloc[path]
            path_rows.append({
                "prey_variable": variable,
                "target_year": year,
                "source_grid_id": topology.grid_id_by_node_index[source_idx],
                "destination_grid_id": topology.grid_id_by_node_index[dest_idx],
                "path_cost": float(dist),
                "path_length_km": lcp_base.path_length_km(path, lcp_grid),
                "path_cell_count": int(len(path)),
                "path_mean_U": float(path_df["U_amphipod_prey_global_q05_q95"].mean()),
                "path_min_U": float(path_df["U_amphipod_prey_global_q05_q95"].min()),
                "path_max_U": float(path_df["U_amphipod_prey_global_q05_q95"].max()),
            })
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    failed = attempted - len(path_rows)
    if failed:
        warnings.append(f"{variable} target {year}: {failed} failed/disconnected paths.")
    return path_rows, warnings


def reconstruct_sparse_path(predecessor_row: np.ndarray, source: int, destination: int) -> list[int]:
    path = [int(destination)]
    node = int(destination)
    guard = 0
    while node != source:
        node = int(predecessor_row[node])
        if node < 0:
            return []
        path.append(node)
        guard += 1
        if guard > len(predecessor_row):
            return []
    return list(reversed(path))


def format_grid_output(lcp_grid: pd.DataFrame, variable: str, year: int) -> pd.DataFrame:
    out = lcp_grid.copy()
    out["route_pair"] = ROUTE_PAIR
    keep = [
        "grid_id", "x", "y", "lon", "lat", "target_year", "route_pair", "prey_variable",
        "U_amphipod_prey_global_q05_q95", "R_log", "lcp_count", "lcp_frequency_normalized", "ocean_flag"
    ]
    return out[keep]


def summarize_route(variable, year, path_rows, grid, attempted, topology, warnings):
    paths = pd.DataFrame(path_rows)
    freq = grid["lcp_frequency_normalized"]
    top_idx = freq.idxmax()
    return {
        "prey_variable": variable,
        "target_year": year,
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "attempted_path_count": attempted,
        "successful_path_count": len(paths),
        "failed_path_count": attempted - len(paths),
        "mean_path_cost": none_if_nan(paths["path_cost"].mean()),
        "median_path_cost": none_if_nan(paths["path_cost"].median()),
        "min_path_cost": none_if_nan(paths["path_cost"].min()),
        "max_path_cost": none_if_nan(paths["path_cost"].max()),
        "mean_path_length_km": none_if_nan(paths["path_length_km"].mean()),
        "median_path_length_km": none_if_nan(paths["path_length_km"].median()),
        "corridor_area_cells_freq_gt_0": int((freq > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int((freq > 0.05).sum()),
        "corridor_area_cells_freq_gt_0_10": int((freq > 0.10).sum()),
        "top_bottleneck_frequency": none_if_nan(freq.max()),
        "top_bottleneck_grid_id": str(grid.loc[top_idx, "grid_id"]),
        "warning_flags": "; ".join(sorted(set(warnings))),
    }


def compare_variables(all_grids, route_summary) -> pd.DataFrame:
    rows = []
    main_var = "total_kJm2_frozen"
    for year in TARGET_YEARS:
        main_grid = all_grids[main_var][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(
            columns={"lcp_frequency_normalized": "freq_main"}
        )
        main_route = route_summary[(route_summary["prey_variable"] == main_var) & (route_summary["target_year"] == year)].iloc[0]
        for variable in VARIABLES:
            grid = all_grids[variable][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(
                columns={"lcp_frequency_normalized": "freq"}
            )
            route = route_summary[(route_summary["prey_variable"] == variable) & (route_summary["target_year"] == year)].iloc[0]
            merged = main_grid.merge(grid, on=["grid_id", "x", "y"], how="inner")
            rows.append({
                "target_year": year,
                "prey_variable": variable,
                "mean_path_cost": route["mean_path_cost"],
                "percent_cost_difference_vs_kJ": pct(route["mean_path_cost"], main_route["mean_path_cost"]),
                "corridor_area_freq_gt_0_05": route["corridor_area_cells_freq_gt_0_05"],
                "top_bottleneck_frequency": route["top_bottleneck_frequency"],
                "frequency_map_correlation_vs_kJ": corr(merged["freq_main"], merged["freq"]),
                "jaccard_vs_kJ_freq_gt_0": jaccard(merged["freq_main"] > 0, merged["freq"] > 0),
                "jaccard_vs_kJ_freq_gt_0_05": jaccard(merged["freq_main"] > 0.05, merged["freq"] > 0.05),
                "jaccard_vs_kJ_freq_gt_0_10": jaccard(merged["freq_main"] > 0.10, merged["freq"] > 0.10),
                "centroid_shift_vs_kJ_freq_gt_0_05_km": centroid_shift_km(merged, "freq_main", "freq", 0.05),
            })
    return pd.DataFrame(rows)


def compare_old_proxy(main_grids, route_summary) -> pd.DataFrame:
    rows = []
    for year in TARGET_YEARS:
        old_path = OLD_LCP / f"prey_only_lcp_main_south_to_north_clean_target{year}.csv"
        if not old_path.exists():
            continue
        old = pd.read_csv(old_path)
        brian = main_grids[year]
        m = old[["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(
            columns={"lcp_frequency_normalized": "old_freq"}
        ).merge(
            brian[["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(
                columns={"lcp_frequency_normalized": "brian_freq"}
            ),
            on="grid_id",
            how="inner",
            suffixes=("_old", "_brian"),
        )
        route = route_summary[
            (route_summary["prey_variable"] == "total_kJm2_frozen") & (route_summary["target_year"] == year)
        ].iloc[0]
        rows.append({
            "target_year": year,
            "comparison": "old_proxy_clean_vs_brian_kJ_C3",
            "shared_C3_cell_count": len(m),
            "brian_kJ_mean_path_cost": route["mean_path_cost"],
            "frequency_map_correlation": corr(m["old_freq"], m["brian_freq"]),
            "jaccard_freq_gt_0": jaccard(m["old_freq"] > 0, m["brian_freq"] > 0),
            "jaccard_freq_gt_0_05": jaccard(m["old_freq"] > 0.05, m["brian_freq"] > 0.05),
            "jaccard_freq_gt_0_10": jaccard(m["old_freq"] > 0.10, m["brian_freq"] > 0.10),
            "centroid_shift_freq_gt_0_05_km": centroid_shift_km(
                m.rename(columns={"x_brian": "x", "y_brian": "y"}), "old_freq", "brian_freq", 0.05
            ),
            "caveat": "Diagnostic only: old proxy and benthic v2 use different study domains and endpoints.",
        })
    return pd.DataFrame(rows)


def bottleneck_outputs(all_grids):
    top_rows = []
    persistence_rows = []
    for variable, yearly in all_grids.items():
        top5_members = []
        for year, grid in yearly.items():
            ranked = grid.sort_values("lcp_frequency_normalized", ascending=False).copy()
            ranked["rank"] = range(1, len(ranked) + 1)
            top_rows.extend(
                ranked.head(20)[
                    ["grid_id", "lon", "lat", "target_year", "prey_variable", "rank", "lcp_frequency_normalized"]
                ].to_dict(orient="records")
            )
            threshold = grid["lcp_frequency_normalized"].quantile(0.95)
            sub = grid[grid["lcp_frequency_normalized"] >= threshold].copy()
            sub["year_in_top5pct"] = year
            top5_members.append(sub[["grid_id", "lon", "lat", "year_in_top5pct", "lcp_frequency_normalized"]])
        all_top = pd.concat(top5_members, ignore_index=True)
        for grid_id, sub in all_top.groupby("grid_id"):
            persistence_rows.append({
                "prey_variable": variable,
                "grid_id": grid_id,
                "lon": float(sub["lon"].iloc[0]),
                "lat": float(sub["lat"].iloc[0]),
                "years_in_top5pct": int(sub["year_in_top5pct"].nunique()),
                "first_year": int(sub["year_in_top5pct"].min()),
                "last_year": int(sub["year_in_top5pct"].max()),
                "mean_lcp_frequency": float(sub["lcp_frequency_normalized"].mean()),
                "max_lcp_frequency": float(sub["lcp_frequency_normalized"].max()),
            })
    return pd.DataFrame(top_rows), pd.DataFrame(persistence_rows)


def plot_lcp_panel(yearly, endpoints, path: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    vmax = max(g["lcp_frequency_normalized"].max() for g in yearly.values())
    scatter = None
    for ax, year in zip(axes.ravel(), sorted(yearly)):
        grid = yearly[year]
        scatter = ax.scatter(grid["lon"], grid["lat"], c=grid["lcp_frequency_normalized"], s=5, cmap="magma", vmin=0, vmax=vmax, linewidth=0)
        draw_c3_context(ax, label=False)
        draw_endpoints(ax, endpoints, label=False)
        ax.set_title(str(year), fontsize=10)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="LCP frequency", shrink=0.76)
    fig.suptitle(f"benthic amphipod prey-only C3 LCP: {title_suffix}", y=0.995)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_summary_figures(route_summary, variable_comparison, old_comparison, persistence):
    labels = {k: v["pretty"] for k, v in VARIABLES.items()}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variable, sub in route_summary.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["mean_path_cost"], marker="o", label=labels[variable])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Mean LCP cost")
    ax.set_title("benthic C3 prey-only path cost by prey variable")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_only_brian_C3_variable_path_cost_comparison_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    comp = variable_comparison[variable_comparison["prey_variable"] != "total_kJm2_frozen"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variable, sub in comp.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["jaccard_vs_kJ_freq_gt_0_05"], marker="o", label=labels[variable])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Jaccard overlap vs kJ, freq > 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("Prey-variable corridor overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_only_brian_C3_variable_corridor_overlap_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variable, sub in persistence.groupby("prey_variable"):
        top = sub.sort_values("years_in_top5pct", ascending=False).head(25)
        ax.scatter(top["lon"], top["lat"], s=top["years_in_top5pct"] * 18, alpha=0.55, label=labels[variable])
    draw_c3_context(ax, label=False)
    ax.set_title("Persistent top-5% prey-only bottleneck cells")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "prey_only_brian_C3_bottleneck_persistence_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not old_comparison.empty:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(old_comparison["target_year"], old_comparison["frequency_map_correlation"], marker="o", label="frequency correlation")
        ax.plot(old_comparison["target_year"], old_comparison["jaccard_freq_gt_0_05"], marker="s", label="Jaccard freq > 0.05")
        ax.set_xlabel("Target year")
        ax.set_ylim(0, 1)
        ax.set_title("Old proxy vs benthic kJ C3 prey-only LCP diagnostic")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "prey_only_old_proxy_vs_brian_kJ_C3_comparison_v2.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def draw_c3_context(ax, label=True):
    ax.set_xlim(-176.1, -156.4)
    ax.set_ylim(62.7, 73.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    for name, coords in old_context.DBO_ANCHORS.items():
        lonlat = [(lon, lat) for lat, lon in coords]
        patch = MplPolygon(lonlat, closed=True, facecolor="none", edgecolor="#666666", linewidth=0.6, label=name if label else None)
        ax.add_patch(patch)
    ax.grid(True, color="#eeeeee", linewidth=0.5)


def draw_endpoints(ax, endpoints, label=True):
    src = endpoints[endpoints["endpoint_type"] == "source"]
    dst = endpoints[endpoints["endpoint_type"] == "destination"]
    ax.scatter(src["lon"], src["lat"], s=18, color="#1b9e77", edgecolor="white", linewidth=0.25, label="source" if label else None)
    ax.scatter(dst["lon"], dst["lat"], s=20, marker="^", color="#6a3d9a", edgecolor="white", linewidth=0.25, label="destination" if label else None)


def build_qa(topology, route_summary):
    rows = {}
    for _, row in route_summary.iterrows():
        key = f"{row['prey_variable']}_{int(row['target_year'])}"
        rows[key] = {
            "successful_path_count": int(row["successful_path_count"]),
            "failed_path_count": int(row["failed_path_count"]),
            "mean_path_cost": float(row["mean_path_cost"]),
            "corridor_area_freq_gt_0_05": int(row["corridor_area_cells_freq_gt_0_05"]),
        }
    warnings = sorted(set(w for flags in route_summary["warning_flags"].dropna() for w in str(flags).split("; ") if w))
    return {
        "phase": "Phase 8C",
        "target_years": TARGET_YEARS,
        "prey_variables": list(VARIABLES),
        "main_prey_variable": "total_kJm2_frozen",
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "graph_node_count": len(topology.grid),
        "graph_edge_count": len(topology.edge_pairs),
        "by_variable_year": rows,
        "warnings": warnings,
        "no_ice_layer_used": True,
        "no_ship_layer_used": True,
    }


def pct(value, baseline):
    if baseline in (0, None) or pd.isna(baseline):
        return math.nan
    return float(100 * (value - baseline) / baseline)


def corr(a, b):
    return float(pd.Series(a).corr(pd.Series(b))) if len(a) > 1 else math.nan


def jaccard(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(a, b).sum() / union)


def centroid_shift_km(df, col_a, col_b, threshold):
    a = df[df[col_a] > threshold]
    b = df[df[col_b] > threshold]
    if a.empty or b.empty:
        return math.nan
    ax = float(a["x"].mean())
    ay = float(a["y"].mean())
    bx = float(b["x"].mean())
    by = float(b["y"].mean())
    return math.hypot(ax - bx, ay - by) / 1000.0


def none_if_nan(value):
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
