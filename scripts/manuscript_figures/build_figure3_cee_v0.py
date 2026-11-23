
# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale")

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "manuscript_figures" / "figure_3"
PANEL_DIR = OUT_DIR / "panels_v0"
SOURCE_DIR = OUT_DIR / "source_assets"

DOMAIN = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
LAND = ROOT / "data/raw/prey/ne_10m_land/ne_10m_land.shp"
COMPARISON = ROOT / "outputs/lcp_v2/prey_only_vs_prey_ice_brian_C3_comparison_v2.csv"
OVERLAP = ROOT / "outputs/lcp_v2/prey_only_vs_prey_ice_brian_C3_corridor_overlap_v2.csv"
CENTROID = ROOT / "outputs/lcp_v2/corridor_centroid_direction_summary_v2.csv"

YEARS = [2011, 2013, 2015, 2017, 2019]

PALETTE = {
    "ocean": "#E9F2F5",
    "land": "#D8D2C7",
    "domain": "#0B4F71",
    "prey": "#B65F2A",
    "ice": "#0F4D92",
    "shift": "#6A51A3",
    "area": "#767676",
    "neutral": "#4D4D4D",
}


def _prepare_legacy_export_dirs():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def load_ocean_grid():
    grid = pd.read_csv(GRID)
    in_domain = grid["inside_C3_domain"].astype(str).str.lower().eq("true")
    return grid[(grid["ocean_flag"] == 1) & in_domain].copy()


def extent_from_ocean_grid(ocean_grid, pad=1.25):
    minx, maxx = ocean_grid["lon"].min(), ocean_grid["lon"].max()
    miny, maxy = ocean_grid["lat"].min(), ocean_grid["lat"].max()
    return (minx - pad, maxx + pad, miny - pad * 0.45, maxy + pad * 0.25)


def set_geographic_aspect(ax, extent):
    _, _, ymin, ymax = extent
    mean_lat = (ymin + ymax) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")


def map_box_ratio(extent):
    xmin, xmax, ymin, ymax = extent
    mean_lat = (ymin + ymax) / 2
    return (1.0 / np.cos(np.deg2rad(mean_lat))) * ((ymax - ymin) / (xmax - xmin))


def make_map_axes_grid(fig, extent, *, left, right, bottom, row_gap, col_gap, ncols=5):
    fig_w, fig_h = fig.get_size_inches()
    width = (right - left - (ncols - 1) * col_gap) / ncols
    height = width * (fig_w / fig_h) * map_box_ratio(extent)
    top_bottom = bottom + height + row_gap
    top_axes = []
    bottom_axes = []
    for i in range(ncols):
        x = left + i * (width + col_gap)
        top_axes.append(fig.add_axes([x, top_bottom, width, height]))
        bottom_axes.append(fig.add_axes([x, bottom, width, height]))
    return top_axes, bottom_axes


def draw_base_map(ax, ocean_grid, land, extent, show_ticks=False):
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.22, zorder=1
    )
    ax.scatter(
        ocean_grid["lon"],
        ocean_grid["lat"],
        s=1.3,
        marker="s",
        color="#BFD4DD",
        edgecolor="none",
        alpha=0.22,
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    set_geographic_aspect(ax, extent)
    ax.grid(color="#FFFFFF", linewidth=0.32, zorder=0)
    if show_ticks:
        ax.tick_params(labelsize=6.2, length=2)
        ax.set_xlabel("Longitude", fontsize=6.8)
        ax.set_ylabel("Latitude", fontsize=6.8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def load_grid(layer, year):
    if layer == "prey_only":
        p = ROOT / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    elif layer == "prey_ice":
        p = ROOT / f"outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    else:
        raise ValueError(layer)
    return pd.read_csv(p)


def add_colorbar_for_axes(fig, mappable, axes, label, width=0.012, pad=0.012, shrink=0.84):
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    height = (y1 - y0) * shrink
    cax = fig.add_axes([x1 + pad, y0 + (y1 - y0 - height) / 2, width, height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    return cb


def add_panel_label_inside(ax, label, x=0.025, y=0.965):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color="#222222",
        clip_on=False,
        zorder=30,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.7),
    )


def add_panel_label_for_axes(fig, axes, label, dx=0.052, dy=0.012):
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    fig.text(
        x0 - dx,
        y1 + dy,
        label,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#222222",
    )


def add_vertical_map_row_labels(
    fig, top_axes, bottom_axes, x_pad=0.031, labels=("Prey-only", "Prey+ice")
):
    fig.canvas.draw()
    for axes, label in zip((top_axes, bottom_axes), labels):
        boxes = [ax.get_position() for ax in axes]
        x0 = min(box.x0 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        fig.text(
            x0 - x_pad,
            (y0 + y1) / 2,
            label,
            rotation=90,
            fontsize=8.6,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["neutral"],
        )


def plot_corridor_surface(ax, df, ocean_grid, land, extent, title=None):
    draw_base_map(ax, ocean_grid, land, extent, show_ticks=False)
    bg = df[df["ocean_flag"] == 1].copy()
    ax.scatter(bg["lon"], bg["lat"], color="#D6E2E7", s=1.0, alpha=0.08, linewidths=0, zorder=2)
    norm = colors.PowerNorm(gamma=0.65, vmin=0, vmax=1)
    levels = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.65, 1.0]
    ax.tricontourf(
        bg["lon"],
        bg["lat"],
        bg["lcp_frequency_normalized"],
        levels=levels,
        cmap="YlOrBr",
        norm=norm,
        alpha=0.92,
        zorder=5,
    )
    ax.tricontour(
        bg["lon"],
        bg["lat"],
        bg["lcp_frequency_normalized"],
        levels=[0.05],
        colors=["#6A3A12"],
        linewidths=0.42,
        alpha=0.72,
        zorder=6,
    )
    xmin, xmax, ymin, ymax = extent
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.22, zorder=7
    )
    set_geographic_aspect(ax, extent)
    if title:
        ax.set_title(title, fontsize=7.5, pad=1.5)
    return ScalarMappable(norm=norm, cmap="YlOrBr")


def plot_cost_increase(ax, comp):
    years = comp["target_year"].to_numpy()
    vals = comp["percent_ice_vs_prey_only_mean_cost"].to_numpy()
    ax.bar(years, vals, color="#75B7C8", edgecolor="#2B6F80", linewidth=0.5, width=0.72)
    ax.plot(years, vals, color="#2B6F80", marker="o", markersize=3.2, linewidth=1.0)
    ax.set_ylabel("Mean cost increase (%)", fontsize=7.3)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    max_i = comp["percent_ice_vs_prey_only_mean_cost"].idxmax()
    row = comp.loc[max_i]
    ax.text(row["target_year"], row["percent_ice_vs_prey_only_mean_cost"] + 2.0, f"{row['percent_ice_vs_prey_only_mean_cost']:.0f}%", ha="center", fontsize=6.4)
    ax.set_ylim(0, max(vals) + 9)


def plot_metrics(ax, comp, overlap):
    years = comp["target_year"].to_numpy()
    jaccard = overlap["jaccard_freq_gt_0_05"]
    ax.plot(
        years,
        overlap["centroid_shift_freq_gt_0_05_km"],
        color=PALETTE["shift"],
        marker="o",
        linewidth=1.25,
        markersize=3.3,
        label="Centroid shift",
    )
    ax.set_ylabel("Centroid shift (km)", color=PALETTE["shift"], fontsize=7.2)
    ax.tick_params(axis="y", labelcolor=PALETTE["shift"], labelsize=6.2)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=6.2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax2 = ax.twinx()
    ax2.plot(
        years,
        jaccard,
        color=PALETTE["area"],
        marker="s",
        linewidth=1.1,
        markersize=3.2,
        label="Jaccard overlap",
    )
    ax2.set_ylabel("Jaccard overlap", color=PALETTE["area"], fontsize=7.2)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=PALETTE["area"], labelsize=6.2)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + [ax2.get_lines()[-1]]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=0.9)


def plot_shift_arrows(ax, centroid, ocean_grid, land, extent):
    rows = centroid[centroid["from_year"].isin(YEARS)].copy()
    xmin = min(rows["from_centroid_lon"].min(), rows["to_centroid_lon"].min()) - 1.1
    xmax = max(rows["from_centroid_lon"].max(), rows["to_centroid_lon"].max()) + 1.1
    ymin = min(rows["from_centroid_lat"].min(), rows["to_centroid_lat"].min()) - 0.75
    ymax = max(rows["from_centroid_lat"].max(), rows["to_centroid_lat"].max()) + 0.75
    zoom_extent = (xmin, xmax, ymin, ymax)
    draw_base_map(ax, ocean_grid, land, zoom_extent, show_ticks=True)
    label_offsets = {
        2011: (0.14, -0.02),
        2013: (0.18, 0.16),
        2015: (-0.66, 0.13),
        2017: (-0.56, 0.02),
        2019: (-0.48, -0.12),
    }
    for _, row in rows.iterrows():
        year = int(row["from_year"])
        ax.annotate(
            "",
            xy=(row["to_centroid_lon"], row["to_centroid_lat"]),
            xytext=(row["from_centroid_lon"], row["from_centroid_lat"]),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["shift"], lw=1.1, alpha=0.82),
            zorder=9,
        )
        ax.scatter(row["from_centroid_lon"], row["from_centroid_lat"], s=18, color=PALETTE["prey"], edgecolors="white", linewidths=0.35, zorder=10)
        ax.scatter(row["to_centroid_lon"], row["to_centroid_lat"], s=20, color=PALETTE["ice"], edgecolors="white", linewidths=0.35, zorder=10)
        dx, dy = label_offsets.get(year, (0.05, 0.05))
        ax.text(
            row["to_centroid_lon"] + dx,
            row["to_centroid_lat"] + dy,
            str(year),
            fontsize=6.1,
            color=PALETTE["neutral"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.4),
            zorder=11,
        )
    ax.scatter([], [], s=18, color=PALETTE["prey"], label="Prey-only")
    ax.scatter([], [], s=20, color=PALETTE["ice"], label="Prey+ice")
    ax.legend(loc="lower left", fontsize=5.5, handlelength=0.9, borderaxespad=0.2)
    inset = ax.inset_axes([0.70, 0.08, 0.26, 0.34])
    draw_base_map(inset, ocean_grid, land, extent, show_ticks=False)
    inset.scatter(rows["from_centroid_lon"], rows["from_centroid_lat"], s=6, color=PALETTE["prey"], zorder=9)
    inset.scatter(rows["to_centroid_lon"], rows["to_centroid_lat"], s=6, color=PALETTE["ice"], zorder=9)
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=PALETTE["shift"], linewidth=0.7, zorder=10)
    inset.add_patch(rect)


def save_panel(fig, name):
    fig.savefig(PANEL_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PANEL_DIR / f"{name}.png", dpi=450, bbox_inches="tight")


def build_figure():
    apply_style()
    ocean_grid = load_ocean_grid()
    land = gpd.read_file(LAND).to_crs(4326)
    extent = extent_from_ocean_grid(ocean_grid)
    comp = pd.read_csv(COMPARISON)
    comp = comp[comp["prey_variable"] == "total_kJm2_frozen"].sort_values("target_year")
    overlap = pd.read_csv(OVERLAP).sort_values("target_year")
    centroid = pd.read_csv(CENTROID)
    centroid = centroid[
        (centroid["prey_variable"] == "total_kJm2_frozen")
        & (centroid["comparison_type"] == "same_year_prey_only_to_prey_ice")
    ].sort_values("from_year")

    grids = {
        ("prey_only", year): load_grid("prey_only", year)
        for year in YEARS
    }
    grids.update({("prey_ice", year): load_grid("prey_ice", year) for year in YEARS})

    fig = plt.figure(figsize=(10.3, 8.6), constrained_layout=False)
    ax_cost = fig.add_axes([0.075, 0.735, 0.33, 0.22])
    ax_metrics = fig.add_axes([0.425, 0.735, 0.36, 0.22])
    ax_arrows = fig.add_axes([0.855, 0.735, 0.12, 0.22])
    map_axes_top, map_axes_bottom = make_map_axes_grid(
        fig,
        extent,
        left=0.080,
        right=0.925,
        bottom=0.065,
        row_gap=0.026,
        col_gap=0.010,
    )

    plot_cost_increase(ax_cost, comp)
    plot_metrics(ax_metrics, comp, overlap)

    sm = None
    for i, year in enumerate(YEARS):
        sm = plot_corridor_surface(map_axes_top[i], grids[("prey_only", year)], ocean_grid, land, extent, title=str(year))
        sm = plot_corridor_surface(map_axes_bottom[i], grids[("prey_ice", year)], ocean_grid, land, extent, title=None)

    add_colorbar_for_axes(fig, sm, map_axes_top + map_axes_bottom, "Corridor frequency", shrink=1.0)

    plot_shift_arrows(ax_arrows, centroid, ocean_grid, land, extent)

    add_panel_label_inside(ax_cost, "A")
    add_panel_label_inside(map_axes_top[0], "B")
    add_panel_label_inside(ax_metrics, "C")
    add_panel_label_inside(ax_arrows, "D")
    add_vertical_map_row_labels(fig, map_axes_top, map_axes_bottom)

    out = OUT_DIR / "gray_whale_cee_figure3_jaccard_overlap_compact_v3"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure 3.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    pfig, pax = plt.subplots(figsize=(3.6, 1.9))
    plot_cost_increase(pax, comp)
    add_panel_label_inside(pax, "A")
    save_panel(pfig, "fig3a_ice_added_cost_increase_v3")
    plt.close(pfig)

    pfig = plt.figure(figsize=(9.2, 4.9))
    axes_top, axes_bottom = make_map_axes_grid(
        pfig,
        extent,
        left=0.070,
        right=0.910,
        bottom=0.075,
        row_gap=0.026,
        col_gap=0.022,
    )
    sm = None
    for i, year in enumerate(YEARS):
        sm = plot_corridor_surface(axes_top[i], grids[("prey_only", year)], ocean_grid, land, extent, title=str(year))
        sm = plot_corridor_surface(axes_bottom[i], grids[("prey_ice", year)], ocean_grid, land, extent, title=None)
    add_panel_label_inside(axes_top[0], "B")
    add_vertical_map_row_labels(pfig, axes_top, axes_bottom)
    add_colorbar_for_axes(pfig, sm, axes_top + axes_bottom, "Corridor frequency", shrink=1.0)
    save_panel(pfig, "fig3b_prey_only_vs_prey_ice_selected_years_v3")
    plt.close(pfig)

    pfig, pax = plt.subplots(figsize=(3.8, 1.95))
    plot_metrics(pax, comp, overlap)
    add_panel_label_inside(pax, "C")
    save_panel(pfig, "fig3c_ice_jaccard_overlap_centroid_metrics_v3")
    plt.close(pfig)

    pfig, pax = plt.subplots(figsize=(4.0, 2.3))
    plot_shift_arrows(pax, centroid, ocean_grid, land, extent)
    add_panel_label_inside(pax, "D")
    save_panel(pfig, "fig3d_same_year_centroid_shift_arrows_v3")
    plt.close(pfig)


if __name__ == "__main__":
    build_figure()
