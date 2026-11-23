
# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts" / "manuscript_figures" / "build_figure3_cee_v0.py"
SPEC = spec_from_file_location("figure3_base", BASE_SCRIPT)
BASE = module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

OUT_DIR = ROOT / "outputs" / "manuscript_figures_v3" / "figure_3"
CANDIDATE_DIR = OUT_DIR / "candidates"
PANEL_DIR = OUT_DIR / "panels_candidate_v1"
SOURCE_DIR = OUT_DIR / "source_data"
QA_DIR = OUT_DIR / "qa"


def _prepare_legacy_export_dirs():
    for directory in (CANDIDATE_DIR, PANEL_DIR, SOURCE_DIR, QA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def plot_north_south_displacement(ax, centroid):
    rows = centroid.sort_values("from_year").copy()
    years = rows["from_year"].astype(int).to_numpy()
    shifts = rows["delta_y_km_north_positive"].to_numpy()

    ax.axvline(0, color="#666666", linewidth=0.75, zorder=1)
    for year, shift in zip(years, shifts):
        color = BASE.PALETTE["ice"] if shift >= 0 else BASE.PALETTE["prey"]
        ax.hlines(year, min(0, shift), max(0, shift), color=color, linewidth=1.45, zorder=2)
        ax.scatter(
            shift,
            year,
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_xlim(-25, 75)
    ax.set_ylim(2019.65, 2010.25)
    ax.set_xticks([-20, 0, 20, 40, 60])
    ax.set_yticks(years)
    ax.set_yticklabels(years, fontsize=5.7)
    ax.tick_params(axis="x", labelsize=5.7, length=2)
    ax.tick_params(axis="y", length=2, pad=1.5)
    ax.set_xlabel("North-south displacement (km)", fontsize=6.3, labelpad=2.2)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.45, zorder=0)
    ax.text(
        0.02,
        1.015,
        "South",
        transform=ax.transAxes,
        fontsize=5.6,
        color=BASE.PALETTE["prey"],
        ha="left",
        va="bottom",
    )
    ax.text(
        0.98,
        1.015,
        "North",
        transform=ax.transAxes,
        fontsize=5.6,
        color=BASE.PALETTE["ice"],
        ha="right",
        va="bottom",
    )


def save_panel(fig, name):
    fig.savefig(PANEL_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PANEL_DIR / f"{name}.png", dpi=600, bbox_inches="tight")


def build_figure():
    BASE.apply_style()
    ocean_grid = BASE.load_ocean_grid()
    land = BASE.gpd.read_file(BASE.LAND).to_crs(4326)
    extent = BASE.extent_from_ocean_grid(ocean_grid)

    comp = pd.read_csv(BASE.COMPARISON)
    comp = comp[comp["prey_variable"] == "total_kJm2_frozen"].sort_values("target_year")
    overlap = pd.read_csv(BASE.OVERLAP).sort_values("target_year")
    centroid = pd.read_csv(BASE.CENTROID)
    centroid = centroid[
        (centroid["prey_variable"] == "total_kJm2_frozen")
        & (centroid["comparison_type"] == "same_year_prey_only_to_prey_ice")
    ].sort_values("from_year")

    grids = {
        ("prey_only", year): BASE.load_grid("prey_only", year)
        for year in BASE.YEARS
    }
    grids.update(
        {("prey_ice", year): BASE.load_grid("prey_ice", year) for year in BASE.YEARS}
    )

    fig = plt.figure(figsize=(10.3, 8.6), constrained_layout=False)
    ax_cost = fig.add_axes([0.075, 0.735, 0.33, 0.22])
    ax_metrics = fig.add_axes([0.425, 0.735, 0.36, 0.22])
    ax_direction = fig.add_axes([0.855, 0.735, 0.12, 0.22])
    map_axes_top, map_axes_bottom = BASE.make_map_axes_grid(
        fig,
        extent,
        left=0.080,
        right=0.925,
        bottom=0.065,
        row_gap=0.026,
        col_gap=0.010,
    )

    BASE.plot_cost_increase(ax_cost, comp)
    BASE.plot_metrics(ax_metrics, comp, overlap)

    sm = None
    for i, year in enumerate(BASE.YEARS):
        sm = BASE.plot_corridor_surface(
            map_axes_top[i],
            grids[("prey_only", year)],
            ocean_grid,
            land,
            extent,
            title=str(year),
        )
        sm = BASE.plot_corridor_surface(
            map_axes_bottom[i],
            grids[("prey_ice", year)],
            ocean_grid,
            land,
            extent,
            title=None,
        )

    BASE.add_colorbar_for_axes(
        fig,
        sm,
        map_axes_top + map_axes_bottom,
        "Corridor frequency",
        shrink=1.0,
    )
    plot_north_south_displacement(ax_direction, centroid)

    BASE.add_panel_label_inside(ax_cost, "A")
    BASE.add_panel_label_inside(map_axes_top[0], "B")
    BASE.add_panel_label_inside(ax_metrics, "C")
    BASE.add_panel_label_inside(ax_direction, "D")
    BASE.add_vertical_map_row_labels(fig, map_axes_top, map_axes_bottom)

    stem = CANDIDATE_DIR / "gray_whale_figure3_centroid_direction_candidate_v1"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    panel_fig, panel_ax = plt.subplots(figsize=(2.2, 3.0))
    plot_north_south_displacement(panel_ax, centroid)
    BASE.add_panel_label_inside(panel_ax, "D")
    save_panel(panel_fig, "fig3d_north_south_centroid_displacement_candidate_v1")
    plt.close(panel_fig)

    source = centroid[
        [
            "from_year",
            "from_centroid_lon",
            "from_centroid_lat",
            "to_centroid_lon",
            "to_centroid_lat",
            "delta_x_km_east_positive",
            "delta_y_km_north_positive",
            "centroid_shift_km",
        ]
    ].rename(
        columns={
            "from_year": "year",
            "from_centroid_lon": "prey_only_centroid_lon",
            "from_centroid_lat": "prey_only_centroid_lat",
            "to_centroid_lon": "prey_ice_centroid_lon",
            "to_centroid_lat": "prey_ice_centroid_lat",
            "delta_x_km_east_positive": "east_west_displacement_km_east_positive",
            "delta_y_km_north_positive": "north_south_displacement_km_north_positive",
            "centroid_shift_km": "total_centroid_displacement_km",
        }
    )
    source.to_csv(
        SOURCE_DIR / "figure3_panel_d_centroid_direction_source_data_v1.csv",
        index=False,
    )


if __name__ == "__main__":
    build_figure()
