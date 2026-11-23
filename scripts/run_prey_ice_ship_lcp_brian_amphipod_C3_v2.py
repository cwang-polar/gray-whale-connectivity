"""Phase 8E.2 benthic amphipod C3 prey + ice + ship LCP.

Runs 2015-2019 multi-stressor LCP ensembles using the Phase 8E.1 alpha0.5
all-vessel ship modifier. The ship layer is a disturbance / movement impedance
proxy only, not collision or fatality. No sensitivity modifiers or validation
metrics are run here.
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

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

sys.path.insert(0, str(ROOT / "scripts"))
import build_ocean_mask_and_endpoints as old_context  # noqa: E402
import run_prey_only_lcp_main_south_to_north_v0 as lcp_base  # noqa: E402


GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
SURFACES = ROOT / "outputs/surfaces_v2"
LCP = ROOT / "outputs/lcp_v2"
FIG = ROOT / "outputs/figures/lcp_v2"
TARGET_YEARS = list(range(2015, 2020))
VARIABLES = {
    "total_kJm2_frozen": {"label": "kJm2_frozen", "pretty": "kJ m-2 frozen", "main": True},
    "total_gww": {"label": "gww", "pretty": "gWW", "main": False},
    "total_gC": {"label": "gC", "pretty": "gC", "main": False},
}
ROUTE_PAIR = "main_south_to_north_C3_v2"
PHASE_MEMO = ROOT / "docs/phase_memos/phase8e2_brian_C3_prey_ice_ship_lcp.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase8e2_brian_C3_prey_ice_ship_lcp_paper_use.md"
CLAIMS = ROOT / "docs/paper_claims_evidence_matrix.md"
REGISTRY = ROOT / "docs/results_registry.md"
README = ROOT / "README.md"
MANIFEST = ROOT / "outputs/results_manifest.csv"


def main() -> None:
    LCP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ocean = pd.read_csv(GRID)
    endpoints = pd.read_csv(ENDPOINTS)
    topology = lcp_base.build_graph_topology(ocean_mask=ocean, endpoints=endpoints)

    all_grids: dict[str, dict[int, pd.DataFrame]] = {}
    route_rows = []
    for variable, meta in VARIABLES.items():
        yearly_grids = {}
        path_rows = []
        for year in TARGET_YEARS:
            surface = pd.read_csv(surface_path(variable, meta, year))
            lcp_grid, warning = prepare_grid(surface, topology.grid, variable, year)
            warnings = list(topology.warnings)
            if warning:
                warnings.append(warning)
            adjacency = build_weighted_csr(topology.edge_pairs, lcp_grid["R_log_prey_ice_ship_alpha0_5"].to_numpy())
            lcp_count = np.zeros(len(lcp_grid), dtype=int)
            year_paths, path_warnings = run_paths(variable, year, lcp_grid, adjacency, topology, lcp_count)
            warnings.extend(path_warnings)
            successful = len(year_paths)
            attempted = len(topology.source_indices) * len(topology.destination_indices)
            lcp_grid["lcp_count"] = lcp_count
            lcp_grid["lcp_frequency_normalized"] = lcp_count / successful if successful else 0.0
            out_grid = format_grid(lcp_grid, variable, year)
            out_grid.to_csv(LCP / f"prey_ice_ship_brian_{meta['label']}_C3_lcp_target{year}_v2.csv", index=False)
            yearly_grids[year] = out_grid
            path_rows.extend(year_paths)
            route_rows.append(summarize_route(variable, year, year_paths, out_grid, attempted, topology, warnings))
        pd.DataFrame(path_rows).to_csv(LCP / f"prey_ice_ship_brian_{meta['label']}_C3_path_summary_v2.csv", index=False)
        all_grids[variable] = yearly_grids
        plot_lcp_panel(yearly_grids, endpoints, FIG / f"prey_ice_ship_brian_{meta['label']}_C3_lcp_all_years_v2.png", meta["pretty"])

    route = pd.DataFrame(route_rows).sort_values(["prey_variable", "target_year"])
    route.to_csv(LCP / "prey_ice_ship_brian_C3_route_summary_v2.csv", index=False)
    three = build_three_layer_comparison(route)
    three.to_csv(LCP / "prey_preyice_preyiceship_brian_C3_comparison_v2.csv", index=False)
    overlap = build_overlap(all_grids, "total_kJm2_frozen")
    overlap.to_csv(LCP / "prey_ice_vs_prey_ice_ship_brian_C3_corridor_overlap_v2.csv", index=False)
    varcomp = compare_variables(all_grids, route)
    varcomp.to_csv(LCP / "prey_ice_ship_brian_C3_variable_comparison_v2.csv", index=False)
    top, persistence = bottleneck_outputs(all_grids)
    top.to_csv(LCP / "prey_ice_ship_brian_C3_top_bottleneck_cells_v2.csv", index=False)
    persistence.to_csv(LCP / "prey_ice_ship_brian_C3_bottleneck_persistence_v2.csv", index=False)
    shift = bottleneck_shift_vs_prey_ice(persistence)
    shift.to_csv(LCP / "prey_ice_vs_prey_ice_ship_brian_C3_bottleneck_shift_v2.csv", index=False)

    make_figures(route, three, overlap, varcomp, persistence, all_grids, endpoints)
    qa = build_qa(topology, route, three, overlap)
    (LCP / "prey_ice_ship_brian_C3_lcp_qa_v2.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    write_docs(route, three, overlap)
    update_docs_manifest()

    print(json.dumps({
        "phase": "Phase 8E.2",
        "target_years": TARGET_YEARS,
        "variables": list(VARIABLES),
        "route_summary": str(LCP / "prey_ice_ship_brian_C3_route_summary_v2.csv"),
        "comparison": str(LCP / "prey_preyice_preyiceship_brian_C3_comparison_v2.csv"),
        "qa": str(LCP / "prey_ice_ship_brian_C3_lcp_qa_v2.json"),
        "no_ship_strike_or_fatality": True,
    }, indent=2))


def surface_path(variable: str, meta: dict, year: int) -> Path:
    return SURFACES / f"prey_ice_ship_brian_{meta['label']}_C3_utility_target{year}_v2.csv"


def prepare_grid(surface: pd.DataFrame, grid: pd.DataFrame, variable: str, year: int) -> tuple[pd.DataFrame, str | None]:
    merged = grid.merge(surface, on="grid_id", how="left", suffixes=("", "_surface"))
    merged = merged[merged["ocean_flag"] == 1].copy().reset_index(drop=True)
    req = ["U_prey_ice_ship_alpha0_5", "R_log_prey_ice_ship_alpha0_5", "U_prey_ice_beta1", "D_ship_alpha0_5_MayJul_mean"]
    missing_cols = [c for c in req if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"{variable} {year}: missing columns {missing_cols}")
    missing = int(merged["U_prey_ice_ship_alpha0_5"].isna().sum())
    warning = f"{variable} {year}: {missing} missing multi utility cells filled to 0.001." if missing else None
    merged["U_prey_ice_ship_alpha0_5"] = pd.to_numeric(merged["U_prey_ice_ship_alpha0_5"], errors="coerce").fillna(0.001).clip(0.001, 1.0)
    merged["R_log_prey_ice_ship_alpha0_5"] = np.maximum(-np.log(merged["U_prey_ice_ship_alpha0_5"] + 0.001), 0.0)
    for col in ["U_prey_global", "A_ice_v1_MayJul", "U_prey_ice_beta1", "D_ship_alpha0_5_MayJul_mean", "S_scaled_ship_primary_MayJul_mean", "delta_R_log_ship_alpha0_5"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
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


def run_paths(variable, year, grid, adjacency, topology, lcp_count):
    rows, warnings = [], []
    distances, predecessors = sparse_dijkstra(adjacency, directed=False, indices=topology.source_indices, return_predecessors=True)
    for source_pos, source_idx in enumerate(topology.source_indices):
        for dest_idx in topology.destination_indices:
            dist = distances[source_pos, dest_idx]
            if not np.isfinite(dist):
                continue
            path = reconstruct(predecessors[source_pos], source_idx, dest_idx)
            if not path:
                continue
            for node in path:
                lcp_count[node] += 1
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
                "path_mean_D_ship": float(p["D_ship_alpha0_5_MayJul_mean"].mean()),
                "path_mean_S_scaled_ship": float(p["S_scaled_ship_primary_MayJul_mean"].mean()),
                "path_mean_U_multi": float(p["U_prey_ice_ship_alpha0_5"].mean()),
                "path_mean_delta_R_ship": float(p["delta_R_log_ship_alpha0_5"].mean()),
                "path_min_D_ship": float(p["D_ship_alpha0_5_MayJul_mean"].min()),
                "path_min_U_multi": float(p["U_prey_ice_ship_alpha0_5"].min()),
                "path_S_scaled_gt_0_50_cell_count": int((p["S_scaled_ship_primary_MayJul_mean"] > 0.50).sum()),
            })
    failed = len(topology.source_indices) * len(topology.destination_indices) - len(rows)
    if failed:
        warnings.append(f"{variable} {year}: {failed} failed/disconnected paths.")
    return rows, warnings


def reconstruct(pred: np.ndarray, source: int, dest: int) -> list[int]:
    path, node, guard = [int(dest)], int(dest), 0
    while node != source:
        node = int(pred[node])
        if node < 0:
            return []
        path.append(node)
        guard += 1
        if guard > len(pred):
            return []
    return list(reversed(path))


def format_grid(grid: pd.DataFrame, variable: str, year: int) -> pd.DataFrame:
    out = grid.copy()
    out["route_pair"] = "main_south_to_north_C3_v2"
    keep = [
        "grid_id", "x", "y", "lon", "lat", "target_year", "route_pair", "prey_variable",
        "U_prey_global", "A_ice_v1_MayJul", "U_prey_ice_beta1", "D_ship_alpha0_5_MayJul_mean",
        "S_scaled_ship_primary_MayJul_mean", "U_prey_ice_ship_alpha0_5",
        "R_log_prey_ice_ship_alpha0_5", "delta_R_log_ship_alpha0_5",
        "lcp_count", "lcp_frequency_normalized", "ocean_flag",
    ]
    return out[keep]


def summarize_route(variable, year, paths, grid, attempted, topology, warnings):
    p = pd.DataFrame(paths)
    f = grid["lcp_frequency_normalized"]
    top_idx = f.idxmax()
    return {
        "prey_variable": variable,
        "target_year": year,
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "attempted_path_count": attempted,
        "successful_path_count": len(p),
        "failed_path_count": attempted - len(p),
        "mean_path_cost": none_if_nan(p["path_cost"].mean()),
        "median_path_cost": none_if_nan(p["path_cost"].median()),
        "mean_path_length_km": none_if_nan(p["path_length_km"].mean()),
        "mean_path_A_ice": none_if_nan(p["path_mean_A_ice"].mean()),
        "mean_path_D_ship": none_if_nan(p["path_mean_D_ship"].mean()),
        "mean_path_S_scaled_ship": none_if_nan(p["path_mean_S_scaled_ship"].mean()),
        "mean_path_U_multi": none_if_nan(p["path_mean_U_multi"].mean()),
        "mean_path_delta_R_ship": none_if_nan(p["path_mean_delta_R_ship"].mean()),
        "mean_path_S_scaled_gt_0_50_cell_count": none_if_nan(p["path_S_scaled_gt_0_50_cell_count"].mean()),
        "corridor_area_cells_freq_gt_0": int((f > 0).sum()),
        "corridor_area_cells_freq_gt_0_05": int((f > 0.05).sum()),
        "corridor_area_cells_freq_gt_0_10": int((f > 0.10).sum()),
        "top_bottleneck_frequency": none_if_nan(f.max()),
        "top_bottleneck_grid_id": str(grid.loc[top_idx, "grid_id"]),
        "warning_flags": "; ".join(sorted(set(warnings))),
    }


def build_three_layer_comparison(route: pd.DataFrame) -> pd.DataFrame:
    prey = pd.read_csv(LCP / "prey_only_brian_C3_route_summary_v2.csv")
    ice = pd.read_csv(LCP / "prey_ice_brian_C3_route_summary_v2.csv")
    rows = []
    for _, multi in route.iterrows():
        v, y = multi["prey_variable"], multi["target_year"]
        p = prey[(prey.prey_variable == v) & (prey.target_year == y)].iloc[0]
        i = ice[(ice.prey_variable == v) & (ice.target_year == y)].iloc[0]
        rows.append({
            "target_year": y,
            "prey_variable": v,
            "prey_only_mean_path_cost": p["mean_path_cost"],
            "prey_ice_mean_path_cost": i["mean_path_cost"],
            "prey_ice_ship_mean_path_cost": multi["mean_path_cost"],
            "percent_ice_vs_prey_only_mean_cost": pct(i["mean_path_cost"], p["mean_path_cost"]),
            "percent_ship_vs_prey_ice_mean_cost": pct(multi["mean_path_cost"], i["mean_path_cost"]),
            "percent_multi_vs_prey_only_mean_cost": pct(multi["mean_path_cost"], p["mean_path_cost"]),
            "prey_only_corridor_area_freq_gt_0_05": p["corridor_area_cells_freq_gt_0_05"],
            "prey_ice_corridor_area_freq_gt_0_05": i["corridor_area_cells_freq_gt_0_05"],
            "prey_ice_ship_corridor_area_freq_gt_0_05": multi["corridor_area_cells_freq_gt_0_05"],
            "delta_corridor_area_ship_vs_prey_ice": multi["corridor_area_cells_freq_gt_0_05"] - i["corridor_area_cells_freq_gt_0_05"],
            "prey_ice_top_bottleneck_frequency": i["top_bottleneck_frequency"],
            "prey_ice_ship_top_bottleneck_frequency": multi["top_bottleneck_frequency"],
            "delta_top_bottleneck_ship_vs_prey_ice": multi["top_bottleneck_frequency"] - i["top_bottleneck_frequency"],
        })
    return pd.DataFrame(rows)


def build_overlap(all_grids, variable: str) -> pd.DataFrame:
    rows = []
    meta = VARIABLES[variable]
    for year in TARGET_YEARS:
        ice = pd.read_csv(LCP / f"prey_ice_brian_{meta['label']}_C3_lcp_target{year}_v2.csv")[["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "ice_freq"})
        multi = all_grids[variable][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "multi_freq"})
        m = ice.merge(multi, on=["grid_id", "x", "y"], how="inner")
        rows.append({
            "target_year": year,
            "prey_variable": variable,
            "pearson_frequency_correlation": corr(m.ice_freq, m.multi_freq),
            "jaccard_freq_gt_0": jaccard(m.ice_freq > 0, m.multi_freq > 0),
            "jaccard_freq_gt_0_05": jaccard(m.ice_freq > 0.05, m.multi_freq > 0.05),
            "jaccard_freq_gt_0_10": jaccard(m.ice_freq > 0.10, m.multi_freq > 0.10),
            "centroid_shift_freq_gt_0_05_km": centroid_shift_km(m, "ice_freq", "multi_freq", 0.05),
        })
    return pd.DataFrame(rows)


def compare_variables(all_grids, route):
    rows, main_var = [], "total_kJm2_frozen"
    for year in TARGET_YEARS:
        main = all_grids[main_var][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "main_freq"})
        main_route = route[(route.prey_variable == main_var) & (route.target_year == year)].iloc[0]
        for v in VARIABLES:
            grid = all_grids[v][year][["grid_id", "x", "y", "lcp_frequency_normalized"]].rename(columns={"lcp_frequency_normalized": "freq"})
            r = route[(route.prey_variable == v) & (route.target_year == year)].iloc[0]
            m = main.merge(grid, on=["grid_id", "x", "y"], how="inner")
            rows.append({
                "target_year": year,
                "prey_variable": v,
                "mean_path_cost": r["mean_path_cost"],
                "percent_cost_difference_vs_kJ": pct(r["mean_path_cost"], main_route["mean_path_cost"]),
                "jaccard_vs_kJ_freq_gt_0_05": jaccard(m.main_freq > 0.05, m.freq > 0.05),
                "frequency_map_correlation_vs_kJ": corr(m.main_freq, m.freq),
            })
    return pd.DataFrame(rows)


def bottleneck_outputs(all_grids):
    top_rows, persistence_rows = [], []
    for v, yearly in all_grids.items():
        pieces = []
        for year, grid in yearly.items():
            ranked = grid.sort_values("lcp_frequency_normalized", ascending=False).copy()
            ranked["rank"] = range(1, len(ranked) + 1)
            top_rows.extend(ranked.head(20)[[
                "grid_id", "lon", "lat", "target_year", "prey_variable", "rank",
                "lcp_frequency_normalized", "A_ice_v1_MayJul", "D_ship_alpha0_5_MayJul_mean",
                "S_scaled_ship_primary_MayJul_mean", "U_prey_ice_ship_alpha0_5"
            ]].to_dict(orient="records"))
            threshold = grid.lcp_frequency_normalized.quantile(0.95)
            sub = grid[grid.lcp_frequency_normalized >= threshold].copy()
            sub["year_in_top5pct"] = year
            pieces.append(sub[["grid_id", "lon", "lat", "year_in_top5pct", "lcp_frequency_normalized", "A_ice_v1_MayJul", "D_ship_alpha0_5_MayJul_mean", "S_scaled_ship_primary_MayJul_mean"]])
        all_top = pd.concat(pieces, ignore_index=True)
        for grid_id, sub in all_top.groupby("grid_id"):
            persistence_rows.append({
                "prey_variable": v,
                "grid_id": grid_id,
                "lon": float(sub.lon.iloc[0]),
                "lat": float(sub.lat.iloc[0]),
                "years_in_top5pct": int(sub.year_in_top5pct.nunique()),
                "first_year": int(sub.year_in_top5pct.min()),
                "last_year": int(sub.year_in_top5pct.max()),
                "mean_lcp_frequency": float(sub.lcp_frequency_normalized.mean()),
                "max_lcp_frequency": float(sub.lcp_frequency_normalized.max()),
                "mean_A_ice": float(sub.A_ice_v1_MayJul.mean()),
                "mean_D_ship": float(sub.D_ship_alpha0_5_MayJul_mean.mean()),
                "mean_S_scaled_ship": float(sub.S_scaled_ship_primary_MayJul_mean.mean()),
            })
    return pd.DataFrame(top_rows), pd.DataFrame(persistence_rows)


def bottleneck_shift_vs_prey_ice(multi_persistence):
    ice = pd.read_csv(LCP / "prey_ice_brian_C3_bottleneck_persistence_v2.csv")
    rows = []
    for v in VARIABLES:
        a = ice[ice.prey_variable == v][["grid_id", "lon", "lat", "years_in_top5pct", "mean_lcp_frequency"]].rename(columns={"years_in_top5pct": "prey_ice_years_in_top5pct", "mean_lcp_frequency": "prey_ice_mean_lcp_frequency"})
        b = multi_persistence[multi_persistence.prey_variable == v][["grid_id", "years_in_top5pct", "mean_lcp_frequency", "mean_D_ship", "mean_S_scaled_ship"]].rename(columns={"years_in_top5pct": "multi_years_in_top5pct", "mean_lcp_frequency": "multi_mean_lcp_frequency"})
        m = a.merge(b, on="grid_id", how="outer")
        m["prey_variable"] = v
        for col in ["prey_ice_years_in_top5pct", "multi_years_in_top5pct", "prey_ice_mean_lcp_frequency", "multi_mean_lcp_frequency"]:
            m[col] = m[col].fillna(0)
        m["delta_years_in_top5pct"] = m.multi_years_in_top5pct - m.prey_ice_years_in_top5pct
        m["delta_mean_lcp_frequency"] = m.multi_mean_lcp_frequency - m.prey_ice_mean_lcp_frequency
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def make_figures(route, three, overlap, varcomp, persistence, all_grids, endpoints):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    k = three[three.prey_variable == "total_kJm2_frozen"].sort_values("target_year")
    ax.plot(k.target_year, k.percent_ship_vs_prey_ice_mean_cost, marker="o", label="ship vs prey+ice")
    ax.plot(k.target_year, k.percent_multi_vs_prey_only_mean_cost, marker="o", label="multi vs prey-only")
    ax.set_xlabel("Target year")
    ax.set_ylabel("% mean path-cost increase")
    ax.set_title("benthic kJ C3 multi-stressor cost increase")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_preyice_preyiceship_brian_kJ_C3_cost_comparison_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(k.target_year, k.prey_ice_corridor_area_freq_gt_0_05, marker="o", label="prey+ice")
    axes[0].plot(k.target_year, k.prey_ice_ship_corridor_area_freq_gt_0_05, marker="o", label="prey+ice+ship")
    axes[0].set_title("Freq > 0.05 corridor area")
    axes[0].set_ylabel("Cell count")
    axes[1].plot(overlap.target_year, overlap.jaccard_freq_gt_0_05, marker="o", label="Jaccard")
    axes[1].plot(overlap.target_year, overlap.centroid_shift_freq_gt_0_05_km / 100, marker="s", label="Centroid shift /100 km")
    axes[1].set_title("Prey+ice vs multi overlap")
    for ax in axes:
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(FIG / "prey_ice_vs_prey_ice_ship_brian_kJ_C3_corridor_change_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_lcp_panel(all_grids["total_kJm2_frozen"], endpoints, FIG / "prey_ice_ship_brian_kJm2_frozen_C3_lcp_all_years_v2.png", "kJ m-2 frozen")

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = persistence[persistence.prey_variable == "total_kJm2_frozen"].sort_values("years_in_top5pct", ascending=False).head(35)
    ax.scatter(sub.lon, sub.lat, s=sub.years_in_top5pct * 22, color="#7b3294", alpha=0.65, label="multi kJ")
    draw_c3_context(ax)
    ax.set_title("Persistent top-5% benthic prey+ice+ship bottleneck cells")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "prey_ice_ship_brian_C3_bottleneck_persistence_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, meta in VARIABLES.items():
        if v == "total_kJm2_frozen":
            continue
        s = varcomp[varcomp.prey_variable == v]
        ax.plot(s.target_year, s.jaccard_vs_kJ_freq_gt_0_05, marker="o", label=meta["pretty"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target year")
    ax.set_ylabel("Jaccard vs kJ, freq > 0.05")
    ax.set_title("Multi-stressor prey-variable corridor overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG / "prey_ice_ship_brian_C3_variable_corridor_overlap_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lcp_panel(yearly, endpoints, path: Path, title_suffix: str):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    vmax = max(g.lcp_frequency_normalized.max() for g in yearly.values())
    sc = None
    for ax, year in zip(axes.ravel(), TARGET_YEARS):
        g = yearly[year]
        sc = ax.scatter(g.lon, g.lat, c=g.lcp_frequency_normalized, s=5, cmap="magma", vmin=0, vmax=vmax, linewidth=0)
        draw_c3_context(ax)
        draw_endpoints(ax, endpoints)
        ax.set_title(str(year), fontsize=10)
    axes.ravel()[-1].axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="LCP frequency", shrink=0.76)
    fig.suptitle(f"benthic amphipod prey+ice+ship C3 LCP: {title_suffix}", y=0.995)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_c3_context(ax):
    ax.set_xlim(-176.1, -156.4)
    ax.set_ylim(62.7, 73.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    for coords in old_context.DBO_ANCHORS.values():
        lonlat = [(lon, lat) for lat, lon in coords]
        ax.add_patch(MplPolygon(lonlat, closed=True, facecolor="none", edgecolor="#777777", linewidth=0.6))
    ax.grid(True, color="#eeeeee", linewidth=0.5)


def draw_endpoints(ax, endpoints):
    src = endpoints[endpoints.endpoint_type == "source"]
    dst = endpoints[endpoints.endpoint_type == "destination"]
    ax.scatter(src.lon, src.lat, s=18, color="#1b9e77", edgecolor="white", linewidth=0.25)
    ax.scatter(dst.lon, dst.lat, s=20, marker="^", color="#6a3d9a", edgecolor="white", linewidth=0.25)


def build_qa(topology, route, three, overlap):
    rows = {}
    for _, r in route.iterrows():
        rows[f"{r.prey_variable}_{int(r.target_year)}"] = {
            "successful_path_count": int(r.successful_path_count),
            "failed_path_count": int(r.failed_path_count),
            "mean_path_cost": float(r.mean_path_cost),
            "mean_path_D_ship": float(r.mean_path_D_ship),
            "corridor_area_freq_gt_0_05": int(r.corridor_area_cells_freq_gt_0_05),
        }
    warnings = sorted(set(w for flags in route.warning_flags.dropna() for w in str(flags).split("; ") if w))
    return {
        "phase": "Phase 8E.2",
        "target_years": TARGET_YEARS,
        "ship_modifier": "D_ship_alpha0_5_MayJul_mean",
        "source_endpoint_count": len(topology.source_indices),
        "destination_endpoint_count": len(topology.destination_indices),
        "graph_node_count": len(topology.grid),
        "graph_edge_count": len(topology.edge_pairs),
        "by_variable_year": rows,
        "three_layer_comparison": three.to_dict(orient="records"),
        "prey_ice_vs_multi_overlap_kJ": overlap.to_dict(orient="records"),
        "warnings": warnings,
        "no_ship_strike_or_fatality": True,
    }


def write_docs(route, three, overlap):
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_docs_manifest():
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_once(path: Path, marker: str, text: str):
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def append_manifest(paths: list[Path]):
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def pct(value, baseline):
    if baseline in (0, None) or pd.isna(baseline):
        return math.nan
    return float(100 * (value - baseline) / baseline)


def corr(a, b):
    return float(pd.Series(a).corr(pd.Series(b))) if len(a) > 1 else math.nan


def jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    return math.nan if union == 0 else float(np.logical_and(a, b).sum() / union)


def centroid_shift_km(df, a, b, threshold):
    da, db = df[df[a] > threshold], df[df[b] > threshold]
    if da.empty or db.empty:
        return math.nan
    return math.hypot(float(da.x.mean()) - float(db.x.mean()), float(da.y.mean()) - float(db.y.mean())) / 1000


def none_if_nan(value):
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
