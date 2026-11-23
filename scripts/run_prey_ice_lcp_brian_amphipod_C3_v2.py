"""Phase 8D.2 benthic amphipod C3 prey + ice LCP.

Runs prey+ice LCP ensembles on the C3 v2 grid using Phase 8D.1 utility
surfaces. The main prey variable is kJm2.frozen; gWW and gC are retained as
prey-variable sensitivity. No ship layer, ship-strike/fatality layer, or
observed validation is used.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import math
import os
import sys
from pathlib import Path

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
import build_ocean_mask_and_endpoints as old_context  # noqa: E402
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base  # noqa: E402


GRID = PROJECT_ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = PROJECT_ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
SURFACES = PROJECT_ROOT / "outputs/surfaces_v2"
LCP = PROJECT_ROOT / "outputs/lcp_v2"
FIG = PROJECT_ROOT / "outputs/figures/lcp_v2"
TARGET_YEARS = list(range(2011, 2020))
VARIABLES = {
    "total_kJm2_frozen": {"label": "kJm2_frozen", "pretty": "kJ m-2 frozen", "main": True},
    "total_gww": {"label": "gww", "pretty": "gWW", "main": False},
    "total_gC": {"label": "gC", "pretty": "gC", "main": False},
}
ROUTE_PAIR = "main_south_to_north_C3_v2"
PHASE_MEMO = PROJECT_ROOT / "docs/phase_memos/phase8d2_brian_C3_prey_ice_lcp.md"
PAPER_MEMO = PROJECT_ROOT / "docs/paper_use_decisions/phase8d2_brian_C3_prey_ice_lcp_paper_use.md"
CLAIMS = PROJECT_ROOT / "docs/paper_claims_evidence_matrix.md"
REGISTRY = PROJECT_ROOT / "docs/results_registry.md"
README = PROJECT_ROOT / "README.md"
MANIFEST = PROJECT_ROOT / "outputs/results_manifest.csv"


def main() -> None:
    LCP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ocean = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    topology = lcp_base.build_graph_topology(ocean_mask=ocean, endpoints=endpoints)

    all_grids: dict[str, dict[int, pd.DataFrame]] = {}
    route_rows: list[dict] = []

    for variable, meta in VARIABLES.items():
        yearly_grids: dict[int, pd.DataFrame] = {}
        path_rows: list[dict] = []
        for year in TARGET_YEARS:
            surface = read_surface(variable, year, meta)
            lcp_grid, warning = prepare_lcp_grid(surface, topology.grid, variable, year)
            warnings = list(topology.warnings)
            if warning:
                warnings.append(warning)
            adjacency = build_weighted_csr(topology.edge_pairs, lcp_grid["R_log_prey_ice_beta1"].to_numpy())
            lcp_count = np.zeros(len(lcp_grid), dtype=int)
            year_paths, path_warnings = run_year_paths(variable, year, lcp_grid, adjacency, topology, lcp_count)
            warnings.extend(path_warnings)
            successful = len(year_paths)
            attempted = len(topology.source_indices) * len(topology.destination_indices)
            lcp_grid["lcp_count"] = lcp_count
            lcp_grid["lcp_frequency_normalized"] = lcp_count / successful if successful else 0.0
            out_grid = format_grid_output(lcp_grid, variable, year)
            out_grid.to_csv(LCP / f"prey_ice_brian_{meta['label']}_C3_lcp_target{year}_v2.csv", index=False)
            yearly_grids[year] = out_grid
            path_rows.extend(year_paths)
            route_rows.append(summarize_route(variable, year, year_paths, out_grid, attempted, topology, warnings))

        pd.DataFrame(path_rows).to_csv(LCP / f"prey_ice_brian_{meta['label']}_C3_path_summary_v2.csv", index=False)
        all_grids[variable] = yearly_grids
        plot_lcp_panel(yearly_grids, endpoints, FIG / f"prey_ice_brian_{meta['label']}_C3_lcp_all_years_v2.png", meta["pretty"])

    route_summary = pd.DataFrame(route_rows).sort_values(["prey_variable", "target_year"])
    route_summary.to_csv(LCP / "prey_ice_brian_C3_route_summary_v2.csv", index=False)
    variable_comparison = compare_variables(all_grids, route_summary)
    variable_comparison.to_csv(LCP / "prey_ice_brian_C3_variable_comparison_v2.csv", index=False)
    prey_only_comparison = compare_with_prey_only(all_grids, route_summary)
    prey_only_comparison.to_csv(LCP / "prey_only_vs_prey_ice_brian_C3_comparison_v2.csv", index=False)
    overlap = corridor_overlap_metrics(all_grids, "total_kJm2_frozen")
    overlap.to_csv(LCP / "prey_only_vs_prey_ice_brian_C3_corridor_overlap_v2.csv", index=False)
    top, persistence = bottleneck_outputs(all_grids)
    top.to_csv(LCP / "prey_ice_brian_C3_top_bottleneck_cells_v2.csv", index=False)
    persistence.to_csv(LCP / "prey_ice_brian_C3_bottleneck_persistence_v2.csv", index=False)
    shift = bottleneck_shift_vs_prey_only(persistence)
    shift.to_csv(LCP / "prey_only_vs_prey_ice_brian_C3_bottleneck_shift_v2.csv", index=False)

    make_summary_figures(route_summary, variable_comparison, prey_only_comparison, overlap, persistence, shift, all_grids, endpoints)
    qa = build_qa(topology, route_summary, prey_only_comparison, overlap)
    (LCP / "prey_ice_brian_C3_lcp_qa_v2.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    write_docs(route_summary, prey_only_comparison, overlap, shift)
    update_registry_manifest_readme_claims()

    print(json.dumps({
        "phase": "Phase 8D.2",
        "variables": list(VARIABLES),
        "target_years": TARGET_YEARS,
        "route_summary": str(LCP / "prey_ice_brian_C3_route_summary_v2.csv"),
        "comparison": str(LCP / "prey_only_vs_prey_ice_brian_C3_comparison_v2.csv"),
        "qa": str(LCP / "prey_ice_brian_C3_lcp_qa_v2.json"),
        "no_ship_layer_used": True,
    }, indent=2))


def read_surface(variable: str, year: int, meta: dict) -> pd.DataFrame:
    path = SURFACES / f"prey_ice_brian_{meta['label']}_C3_utility_target{year}_v2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def prepare_lcp_grid(surface: pd.DataFrame, grid: pd.DataFrame, variable: str, year: int) -> tuple[pd.DataFrame, str | None]:
    merged = grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"))
    merged = merged[merged["ocean_flag"] == 1].copy().reset_index(drop=True)
    required = ["U_prey_global", "A_ice_v1_MayJul", "U_prey_ice_beta1", "R_log_prey_ice_beta1"]
    missing_cols = [c for c in required if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"{variable} target {year}: missing required columns {missing_cols}")
    missing = int(merged["U_prey_ice_beta1"].isna().sum())
    warning = f"{variable} target {year}: {missing} missing prey+ice utility cells filled to 0.001." if missing else None
    merged["U_prey_ice_beta1"] = pd.to_numeric(merged["U_prey_ice_beta1"], errors="coerce").fillna(0.001).clip(0.001, 1.0)
    merged["R_log_prey_ice_beta1"] = np.maximum(-np.log(merged["U_prey_ice_beta1"] + 0.001), 0.0)
    merged["U_prey_global"] = pd.to_numeric(merged["U_prey_global"], errors="coerce").fillna(0.001).clip(0.001, 1.0)
    merged["A_ice_v1_MayJul"] = pd.to_numeric(merged["A_ice_v1_MayJul"], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    merged["prey_variable"] = variable
    merged["target_year"] = year
    return merged, warning


def build_weighted_csr(edge_pairs, resistance: np.ndarray) -> csr_matrix:
    rows, cols, data = [], [], []
    for i, j, distance_km in edge_pairs:
        cost = float((resistance[i] + resistance[j]) / 2.0 * distance_km)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([cost, cost])
    return csr_matrix((data, (rows, cols)), shape=(len(resistance), len(resistance)))


def run_year_paths(variable: str, year: int, grid: pd.DataFrame, adjacency: csr_matrix, topology, lcp_count: np.ndarray):
    rows = []
    warnings = []
    distances, predecessors = sparse_dijkstra(adjacency, directed=False, indices=topology.source_indices, return_predecessors=True)
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
            p = grid.iloc[path]
            rows.append({
                "prey_variable": variable,
                "target_year": year,
                "source_grid_id": topology.grid_id_by_node_index[source_idx],
                "destination_grid_id": topology.grid_id_by_node_index[dest_idx],
                "path_cost": float(dist),
                "path_length_km": lcp_base.path_length_km(path, grid),
                "path_cell_count": int(len(path)),
                "path_mean_U_prey": float(p["U_prey_global"].mean()),
                "path_mean_A_ice": float(p["A_ice_v1_MayJul"].mean()),
                "path_mean_U_prey_ice": float(p["U_prey_ice_beta1"].mean()),
                "path_mean_delta_R_log_ice": float(p["delta_R_log_ice_beta1"].mean()) if "delta_R_log_ice_beta1" in p else math.nan,
                "path_min_A_ice": float(p["A_ice_v1_MayJul"].min()),
                "path_min_U_prey_ice": float(p["U_prey_ice_beta1"].min()),
                "path_Aice_lt_0_50_cell_count": int((p["A_ice_v1_MayJul"] < 0.50).sum()),
                "path_Aice_lt_0_25_cell_count": int((p["A_ice_v1_MayJul"] < 0.25).sum()),
            })
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    failed = attempted - len(rows)
    if failed:
        warnings.append(f"{variable} target {year}: {failed} failed/disconnected paths.")
    return rows, warnings


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


def format_grid_output(grid: pd.DataFrame, variable: str, year: int) -> pd.DataFrame:
    out = grid.copy()
    out["route_pair"] = ROUTE_PAIR
    keep = [
        "grid_id", "x", "y", "lon", "lat", "target_year", "route_pair", "prey_variable",
        "U_prey_global", "A_ice_v1_MayJul", "U_prey_ice_beta1", "R_log_prey_ice_beta1",
        "delta_R_log_ice_beta1", "lcp_count", "lcp_frequency_normalized", "ocean_flag",
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
        "mean_path_U_prey": none_if_nan(paths["path_mean_U_prey"].mean()),
        "mean_path_A_ice": none_if_nan(paths["path_mean_A_ice"].mean()),
        "mean_path_U_prey_ice": none_if_nan(paths["path_mean_U_prey_ice"].mean()),
        "mean_path_delta_R_log_ice": none_if_nan(paths["path_mean_delta_R_log_ice"].mean()),
        "mean_path_Aice_lt_0_50_cell_count": none_if_nan(paths["path_Aice_lt_0_50_cell_count"].mean()),
        "mean_path_Aice_lt_0_25_cell_count": none_if_nan(paths["path_Aice_lt_0_25_cell_count"].mean()),
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
        main_grid = all_grids[main_var][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "freq_main"})
        main_route = route_summary[(route_summary["prey_variable"] == main_var) & (route_summary["target_year"] == year)].iloc[0]
        for variable in VARIABLES:
            grid = all_grids[variable][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "freq"})
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


def compare_with_prey_only(all_grids, route_summary) -> pd.DataFrame:
    prey_only_route = pd.read_csv(LCP / "prey_only_brian_C3_route_summary_v2.csv")
    rows = []
    for _, ice in route_summary.iterrows():
        base = prey_only_route[
            (prey_only_route["prey_variable"] == ice["prey_variable"]) &
            (prey_only_route["target_year"] == ice["target_year"])
        ].iloc[0]
        rows.append({
            "target_year": ice["target_year"],
            "prey_variable": ice["prey_variable"],
            "prey_only_mean_path_cost": base["mean_path_cost"],
            "prey_ice_mean_path_cost": ice["mean_path_cost"],
            "delta_ice_vs_prey_only_mean_cost": ice["mean_path_cost"] - base["mean_path_cost"],
            "percent_ice_vs_prey_only_mean_cost": pct(ice["mean_path_cost"], base["mean_path_cost"]),
            "prey_only_median_path_cost": base["median_path_cost"],
            "prey_ice_median_path_cost": ice["median_path_cost"],
            "percent_ice_vs_prey_only_median_cost": pct(ice["median_path_cost"], base["median_path_cost"]),
            "prey_only_corridor_area_freq_gt_0_05": base["corridor_area_cells_freq_gt_0_05"],
            "prey_ice_corridor_area_freq_gt_0_05": ice["corridor_area_cells_freq_gt_0_05"],
            "delta_corridor_area_ice_vs_prey_only": ice["corridor_area_cells_freq_gt_0_05"] - base["corridor_area_cells_freq_gt_0_05"],
            "prey_only_top_bottleneck_frequency": base["top_bottleneck_frequency"],
            "prey_ice_top_bottleneck_frequency": ice["top_bottleneck_frequency"],
            "delta_top_bottleneck_frequency_ice_vs_prey_only": ice["top_bottleneck_frequency"] - base["top_bottleneck_frequency"],
        })
    return pd.DataFrame(rows)


def corridor_overlap_metrics(all_grids, variable: str) -> pd.DataFrame:
    rows = []
    for year in TARGET_YEARS:
        prey_only_path = LCP / f"prey_only_brian_{VARIABLES[variable]['label']}_C3_lcp_target{year}_v2.csv"
        prey_only = pd.read_csv(prey_only_path)[["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "prey_only_freq"})
        prey_ice = all_grids[variable][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "prey_ice_freq"})
        m = prey_only.merge(prey_ice, on=["grid_id", "x", "y"], how="inner")
        rows.append({
            "target_year": year,
            "prey_variable": variable,
            "comparison_pair": "prey_only_vs_prey_ice",
            "pearson_frequency_correlation": corr(m["prey_only_freq"], m["prey_ice_freq"]),
            "jaccard_freq_gt_0": jaccard(m["prey_only_freq"] > 0, m["prey_ice_freq"] > 0),
            "jaccard_freq_gt_0_05": jaccard(m["prey_only_freq"] > 0.05, m["prey_ice_freq"] > 0.05),
            "jaccard_freq_gt_0_10": jaccard(m["prey_only_freq"] > 0.10, m["prey_ice_freq"] > 0.10),
            "centroid_shift_freq_gt_0_05_km": centroid_shift_km(m, "prey_only_freq", "prey_ice_freq", 0.05),
        })
    return pd.DataFrame(rows)


def bottleneck_outputs(all_grids):
    top_rows, persistence_rows = [], []
    for variable, yearly in all_grids.items():
        top5_members = []
        for year, grid in yearly.items():
            ranked = grid.sort_values("lcp_frequency_normalized", ascending=False).copy()
            ranked["rank"] = range(1, len(ranked) + 1)
            top_rows.extend(ranked.head(20)[[
                "grid_id", "lon", "lat", "target_year", "prey_variable", "rank",
                "lcp_frequency_normalized", "U_prey_global", "A_ice_v1_MayJul", "U_prey_ice_beta1"
            ]].to_dict(orient="records"))
            threshold = grid["lcp_frequency_normalized"].quantile(0.95)
            sub = grid[grid["lcp_frequency_normalized"] >= threshold].copy()
            sub["year_in_top5pct"] = year
            top5_members.append(sub[["grid_id", "lon", "lat", "year_in_top5pct", "lcp_frequency_normalized", "A_ice_v1_MayJul", "U_prey_ice_beta1"]])
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
                "mean_A_ice": float(sub["A_ice_v1_MayJul"].mean()),
                "mean_U_prey_ice": float(sub["U_prey_ice_beta1"].mean()),
            })
    return pd.DataFrame(top_rows), pd.DataFrame(persistence_rows)


def bottleneck_shift_vs_prey_only(prey_ice_persistence: pd.DataFrame) -> pd.DataFrame:
    prey_only = pd.read_csv(LCP / "prey_only_brian_C3_bottleneck_persistence_v2.csv")
    rows = []
    for variable in VARIABLES:
        a = prey_only[prey_only["prey_variable"] == variable][["grid_id", "lon", "lat", "years_in_top5pct", "mean_lcp_frequency"]].rename(columns={
            "years_in_top5pct": "prey_only_years_in_top5pct",
            "mean_lcp_frequency": "prey_only_mean_lcp_frequency",
        })
        b = prey_ice_persistence[prey_ice_persistence["prey_variable"] == variable][["grid_id", "years_in_top5pct", "mean_lcp_frequency", "mean_A_ice", "mean_U_prey_ice"]].rename(columns={
            "years_in_top5pct": "prey_ice_years_in_top5pct",
            "mean_lcp_frequency": "prey_ice_mean_lcp_frequency",
        })
        m = a.merge(b, on="grid_id", how="outer")
        m["prey_variable"] = variable
        for col in ["prey_only_years_in_top5pct", "prey_ice_years_in_top5pct", "prey_only_mean_lcp_frequency", "prey_ice_mean_lcp_frequency"]:
            m[col] = m[col].fillna(0)
        m["delta_years_in_top5pct"] = m["prey_ice_years_in_top5pct"] - m["prey_only_years_in_top5pct"]
        m["delta_mean_lcp_frequency"] = m["prey_ice_mean_lcp_frequency"] - m["prey_only_mean_lcp_frequency"]
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def make_summary_figures(route_summary, variable_comparison, prey_only_comparison, overlap, persistence, shift, all_grids, endpoints):
    labels = {k: v["pretty"] for k, v in VARIABLES.items()}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variable, sub in route_summary.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["mean_path_cost"], marker="o", label=labels[variable])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Mean LCP cost")
    ax.set_title("benthic C3 prey+ice path cost by prey variable")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_ice_brian_C3_variable_path_cost_comparison_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    kj = prey_only_comparison[prey_only_comparison["prey_variable"] == "total_kJm2_frozen"].sort_values("target_year")
    ax.plot(kj["target_year"], kj["percent_ice_vs_prey_only_mean_cost"], marker="o", color="#d95f02")
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("Target year")
    ax.set_ylabel("% cost increase vs prey-only")
    ax.set_title("Ice-added cost, benthic kJ C3")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "prey_only_vs_prey_ice_brian_kJ_C3_cost_increase_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(kj["target_year"], kj["prey_only_corridor_area_freq_gt_0_05"], marker="o", label="prey-only")
    axes[0].plot(kj["target_year"], kj["prey_ice_corridor_area_freq_gt_0_05"], marker="o", label="prey+ice")
    axes[0].set_ylabel("Cells freq > 0.05")
    axes[0].set_title("Frequently used corridor area")
    axes[1].plot(overlap["target_year"], overlap["jaccard_freq_gt_0_05"], marker="o", label="Jaccard")
    axes[1].plot(overlap["target_year"], overlap["centroid_shift_freq_gt_0_05_km"] / 100, marker="s", label="Centroid shift /100 km")
    axes[1].set_title("Prey-only vs prey+ice overlap")
    for ax in axes:
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(FIG / "prey_only_vs_prey_ice_brian_kJ_C3_corridor_change_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    comp = variable_comparison[variable_comparison["prey_variable"] != "total_kJm2_frozen"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variable, sub in comp.groupby("prey_variable"):
        ax.plot(sub["target_year"], sub["jaccard_vs_kJ_freq_gt_0_05"], marker="o", label=labels[variable])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Jaccard overlap vs kJ, freq > 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("Prey+ice variable corridor overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_ice_brian_C3_variable_corridor_overlap_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = persistence[persistence["prey_variable"] == "total_kJm2_frozen"].sort_values("years_in_top5pct", ascending=False).head(35)
    ax.scatter(sub["lon"], sub["lat"], s=sub["years_in_top5pct"] * 20, color="#b2182b", alpha=0.65, label="prey+ice kJ")
    draw_c3_context(ax, label=False)
    ax.set_title("Persistent top-5% benthic prey+ice bottleneck cells")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_ice_brian_C3_bottleneck_persistence_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_key_year_maps(all_grids, endpoints)


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
    fig.suptitle(f"benthic amphipod prey+ice C3 LCP: {title_suffix}", y=0.995)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_key_year_maps(all_grids, endpoints):
    years = [2012, 2016, 2019]
    fig, axes = plt.subplots(len(years), 2, figsize=(9, 11), sharex=True, sharey=True)
    for r, year in enumerate(years):
        prey = pd.read_csv(LCP / f"prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv")
        ice = all_grids["total_kJm2_frozen"][year]
        for c, (df, title) in enumerate([(prey, f"{year} prey-only"), (ice, f"{year} prey+ice")]):
            ax = axes[r, c]
            ax.scatter(df["lon"], df["lat"], c=df["lcp_frequency_normalized"], s=5, cmap="magma", vmin=0, vmax=1, linewidth=0)
            draw_c3_context(ax, label=False)
            draw_endpoints(ax, endpoints, label=False)
            ax.set_title(title)
    fig.suptitle("benthic kJ C3 key-year LCP comparison", y=0.995)
    fig.savefig(FIG / "prey_only_vs_prey_ice_brian_kJ_C3_key_year_maps_v2.png", dpi=260, bbox_inches="tight")
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


def build_qa(topology, route_summary, prey_only_comparison, overlap):
    rows = {}
    for _, row in route_summary.iterrows():
        key = f"{row['prey_variable']}_{int(row['target_year'])}"
        rows[key] = {
            "successful_path_count": int(row["successful_path_count"]),
            "failed_path_count": int(row["failed_path_count"]),
            "mean_path_cost": float(row["mean_path_cost"]),
            "mean_path_A_ice": float(row["mean_path_A_ice"]),
            "corridor_area_freq_gt_0_05": int(row["corridor_area_cells_freq_gt_0_05"]),
        }
    warnings = sorted(set(w for flags in route_summary["warning_flags"].dropna() for w in str(flags).split("; ") if w))
    return {
        "phase": "Phase 8D.2",
        "target_years": TARGET_YEARS,
        "prey_variables": list(VARIABLES),
        "main_prey_variable": "total_kJm2_frozen",
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "graph_node_count": len(topology.grid),
        "graph_edge_count": len(topology.edge_pairs),
        "by_variable_year": rows,
        "comparison_with_prey_only": prey_only_comparison.to_dict(orient="records"),
        "corridor_overlap_kJ": overlap.to_dict(orient="records"),
        "warnings": warnings,
        "no_ship_layer_used": True,
        "no_validation_used": True,
    }


def write_docs(route_summary, comparison, overlap, shift):
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_registry_manifest_readme_claims():
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_once(path: Path, marker: str, text: str) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_manifest(paths: list[Path]) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


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
    return math.hypot(float(a["x"].mean()) - float(b["x"].mean()), float(a["y"].mean()) - float(b["y"].mean())) / 1000.0


def none_if_nan(value):
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
