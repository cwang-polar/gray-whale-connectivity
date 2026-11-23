
# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale")

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "manuscript_figures" / "figure_2"
PANEL_DIR = OUT_DIR / "panels_v0"
SOURCE_DIR = OUT_DIR / "source_assets"

DOMAIN = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
LAND = ROOT / "data/raw/prey/ne_10m_land/ne_10m_land.shp"
ROUTE_SUMMARY = ROOT / "outputs/lcp_v2/prey_only_brian_C3_route_summary_v2.csv"
CENTROID_SUMMARY = ROOT / "outputs/lcp_v2/corridor_centroid_direction_summary_v2.csv"

YEARS = [2011, 2013, 2015, 2017, 2019]


PALETTE = {
    "ocean": "#E9F2F5",
    "land": "#D8D2C7",
    "domain": "#0B4F71",
    "neutral": "#4D4D4D",
    "cost": "#0F4D92",
    "area": "#B65F2A",
    "shift": "#6A51A3",
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
    return (minx - pad, maxx + pad * 1.0, miny - pad * 0.45, maxy + pad * 0.25)


def set_geographic_aspect(ax, extent):
    _, _, ymin, ymax = extent
    mean_lat = (ymin + ymax) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")


def add_geographic_labels(ax):
    labels = [
        ("St. Lawrence\nIsland", -170.4, 63.75, "center"),
        ("Bering\nStrait", -168.35, 65.9, "center"),
        ("Wainwright", -160.05, 70.62, "center"),
        ("Barrow\nCanyon", -156.95, 71.65, "center"),
    ]
    for text, x, y, ha in labels:
        ax.text(
            x,
            y,
            text,
            fontsize=5.6,
            color="#5F5F5F",
            ha=ha,
            va="center",
            zorder=8,
        )


def draw_base_map(ax, ocean_grid, land, extent, show_ticks=False):
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.25, zorder=1
    )
    ax.scatter(
        ocean_grid["lon"],
        ocean_grid["lat"],
        s=1.8,
        marker="s",
        color="#BFD4DD",
        edgecolor="none",
        alpha=0.24,
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    set_geographic_aspect(ax, extent)
    ax.grid(color="#FFFFFF", linewidth=0.35, zorder=0)
    if show_ticks:
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def load_year_grid(year):
    p = ROOT / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv"
    return pd.read_csv(p)


def add_colorbar_for_axes(fig, mappable, axes, label, width=0.012, pad=0.012, shrink=0.82):
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


def plot_prey_map(ax, df, ocean_grid, land, extent, year, show_labels=False):
    draw_base_map(ax, ocean_grid, land, extent)
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["U_amphipod_prey_global_q05_q95"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=7,
        marker="s",
        linewidths=0,
        alpha=0.92,
        zorder=3,
    )
    if show_labels:
        add_geographic_labels(ax)
    ax.set_title(f"{year}", fontsize=8.5, pad=2)
    return sc


def plot_lcp_map(ax, df, ocean_grid, land, extent, year, show_labels=False):
    draw_base_map(ax, ocean_grid, land, extent)
    bg = df[df["ocean_flag"] == 1].copy()
    ax.scatter(bg["lon"], bg["lat"], color="#D6E2E7", s=1.5, alpha=0.10, linewidths=0, zorder=2)
    norm = colors.PowerNorm(gamma=0.65, vmin=0, vmax=1)
    levels = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.65, 1.0]
    if len(bg) > 4:
        sc = ax.tricontourf(
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
            linewidths=0.45,
            alpha=0.75,
            zorder=6,
        )
    else:
        sc = ax.scatter(
            freq["lon"],
            freq["lat"],
            c=freq["lcp_frequency_normalized"],
            cmap="YlOrBr",
            norm=norm,
            s=8,
            linewidths=0,
            alpha=0.95,
            zorder=5,
        )
    if show_labels:
        add_geographic_labels(ax)
    xmin, xmax, ymin, ymax = extent
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.25, zorder=7
    )
    set_geographic_aspect(ax, extent)
    ax.set_title(f"{year}", fontsize=8.5, pad=2)
    return ScalarMappable(norm=norm, cmap="YlOrBr")


def add_panel_label(ax, label, x=0.01, y=0.98):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1F1F1F",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.1),
        zorder=20,
    )


def add_outer_panel_label(ax, label, x=-0.075, y=1.045):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#1F1F1F",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.1),
        zorder=20,
        clip_on=False,
    )


def plot_cost_area(ax, route):
    years = route["target_year"].to_numpy()
    ax.plot(
        years,
        route["mean_path_cost"],
        color=PALETTE["cost"],
        marker="o",
        markersize=3.6,
        linewidth=1.4,
        label="Mean cost",
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelcolor=PALETTE["cost"], labelsize=6.5, pad=1.5)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax2 = ax.twinx()
    ax._fig2_twin_axis = ax2
    ax2.plot(
        years,
        route["corridor_area_cells_freq_gt_0_05"],
        color=PALETTE["area"],
        marker="s",
        markersize=3.4,
        linewidth=1.2,
        label="Frequent-corridor cells",
    )
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", labelcolor=PALETTE["area"], labelsize=6.5, pad=1.5)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 1.025),
        ncol=2,
        fontsize=5.8,
        handlelength=1.25,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.set_xlim(years.min() - 0.4, years.max() + 0.4)
    return ax


def align_outer_left_to_reference(fig, ax, reference_ax, min_width=0.26):
    """Shift an axis right so its tick-label bbox aligns with a reference panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    ref_left = reference_ax.get_position().x0
    shift = ref_left - tight.x0
    if shift <= 0:
        return 0.0
    pos = ax.get_position()
    new_width = max(pos.width - shift, min_width)
    new_x0 = pos.x1 - new_width
    ax.set_position([new_x0, pos.y0, new_width, pos.height])
    twin = getattr(ax, "_fig2_twin_axis", None)
    if twin is not None:
        twin.set_position([new_x0, pos.y0, new_width, pos.height])
    return shift


def centroid_points(centroid):
    rows = []
    first = centroid.iloc[0]
    rows.append(
        {
            "year": int(first["from_year"]),
            "lon": first["from_centroid_lon"],
            "lat": first["from_centroid_lat"],
        }
    )
    for _, row in centroid.iterrows():
        rows.append(
            {
                "year": int(row["to_year"]),
                "lon": row["to_centroid_lon"],
                "lat": row["to_centroid_lat"],
            }
        )
    return pd.DataFrame(rows)


def plot_centroid_trajectory(ax, centroid, land, ocean_grid=None, full_extent=None):
    pts = centroid_points(centroid)
    xmin, xmax = pts["lon"].min() - 1.2, pts["lon"].max() + 1.2
    ymin, ymax = pts["lat"].min() - 0.75, pts["lat"].max() + 0.75
    ax.set_facecolor(PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.25, zorder=1
    )
    ax.plot(pts["lon"], pts["lat"], color=PALETTE["shift"], linewidth=1.3, alpha=0.85, zorder=4)
    sc = ax.scatter(
        pts["lon"],
        pts["lat"],
        c=pts["year"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )
    for i in range(len(pts) - 1):
        ax.annotate(
            "",
            xy=(pts.loc[i + 1, "lon"], pts.loc[i + 1, "lat"]),
            xytext=(pts.loc[i, "lon"], pts.loc[i, "lat"]),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["shift"], lw=1.0, alpha=0.65),
            zorder=4,
        )
    for year in [2011, 2015, 2019]:
        row = pts[pts["year"] == year].iloc[0]
        ax.text(row["lon"] + 0.07, row["lat"] + 0.06, str(year), fontsize=6.6, color=PALETTE["neutral"], zorder=6)
    max_row = centroid.loc[centroid["centroid_shift_km"].idxmax()]
    ax.text(
        0.02,
        0.88,
        f"Largest step: {max_row['centroid_shift_km']:.0f} km ({int(max_row['from_year'])}-{str(int(max_row['to_year']))[-2:]})",
        transform=ax.transAxes,
        fontsize=6.6,
        ha="left",
        va="top",
        color=PALETTE["neutral"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.2),
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6.4, length=2)
    ax.grid(color="#FFFFFF", linewidth=0.45)
    if ocean_grid is not None and full_extent is not None:
        inset = ax.inset_axes([0.64, 0.08, 0.32, 0.36])
        draw_base_map(inset, ocean_grid, land, full_extent, show_ticks=False)
        inset.scatter(pts["lon"], pts["lat"], s=8, color=PALETTE["shift"], edgecolors="white", linewidths=0.25, zorder=8)
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=PALETTE["shift"],
            linewidth=0.8,
            linestyle="-",
            zorder=9,
        )
        inset.add_patch(rect)
        inset.set_title("location", fontsize=5.8, pad=1)
    return ax


def save_panel(fig, name):
    fig.savefig(PANEL_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PANEL_DIR / f"{name}.png", dpi=450, bbox_inches="tight")


def build_figure():
    apply_style()
    ocean_grid = load_ocean_grid()
    land = gpd.read_file(LAND).to_crs(4326)
    extent = extent_from_ocean_grid(ocean_grid)
    grids = {year: load_year_grid(year) for year in YEARS}

    route = pd.read_csv(ROUTE_SUMMARY)
    route = route[route["prey_variable"] == "total_kJm2_frozen"].sort_values("target_year")
    centroid = pd.read_csv(CENTROID_SUMMARY)
    centroid = centroid[
        (centroid["prey_variable"] == "total_kJm2_frozen")
        & (centroid["comparison_type"] == "consecutive_year_prey_only")
    ].sort_values(["from_year", "to_year"])

    fig = plt.figure(figsize=(9.0, 7.4), constrained_layout=False)
    gs = GridSpec(
        3,
        10,
        figure=fig,
        height_ratios=[1.25, 1.25, 1.05],
        hspace=0.30,
        wspace=0.18,
    )

    prey_axes = [fig.add_subplot(gs[0, i * 2 : i * 2 + 2]) for i in range(5)]
    lcp_axes = [fig.add_subplot(gs[1, i * 2 : i * 2 + 2]) for i in range(5)]
    ax_cost = fig.add_subplot(gs[2, 0:4])
    ax_shift = fig.add_subplot(gs[2, 5:10])
    fig.subplots_adjust(left=0.062, right=0.945, top=0.965, bottom=0.072)

    prey_sc = None
    lcp_sc = None
    for i, year in enumerate(YEARS):
        prey_sc = plot_prey_map(prey_axes[i], grids[year], ocean_grid, land, extent, year, show_labels=False)
        lcp_sc = plot_lcp_map(lcp_axes[i], grids[year], ocean_grid, land, extent, year, show_labels=False)
    add_panel_label(prey_axes[0], "A")
    add_panel_label(lcp_axes[0], "B")
    add_outer_panel_label(ax_cost, "C")
    add_outer_panel_label(ax_shift, "D")
    add_colorbar_for_axes(fig, prey_sc, prey_axes, "Scaled prey support")
    add_colorbar_for_axes(fig, lcp_sc, lcp_axes, "Corridor frequency")

    plot_cost_area(ax_cost, route)
    plot_centroid_trajectory(ax_shift, centroid, land, ocean_grid=ocean_grid, full_extent=extent)
    align_outer_left_to_reference(fig, ax_cost, prey_axes[0])

    out = OUT_DIR / "gray_whale_cee_figure2_draft_v1"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure 2.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Panel exports for flexible manuscript assembly.
    pfig = plt.figure(figsize=(8.2, 2.45))
    pgs = GridSpec(1, 5, figure=pfig, wspace=0.12)
    axes = [pfig.add_subplot(pgs[0, i]) for i in range(5)]
    sc = None
    for i, year in enumerate(YEARS):
        sc = plot_prey_map(axes[i], grids[year], ocean_grid, land, extent, year, show_labels=False)
    add_panel_label(axes[0], "A")
    add_colorbar_for_axes(pfig, sc, axes, "Scaled prey support")
    save_panel(pfig, "fig2a_annual_prey_support_selected_years_v1")
    plt.close(pfig)

    pfig = plt.figure(figsize=(8.2, 2.45))
    pgs = GridSpec(1, 5, figure=pfig, wspace=0.12)
    axes = [pfig.add_subplot(pgs[0, i]) for i in range(5)]
    sc = None
    for i, year in enumerate(YEARS):
        sc = plot_lcp_map(axes[i], grids[year], ocean_grid, land, extent, year, show_labels=False)
    add_panel_label(axes[0], "B")
    add_colorbar_for_axes(pfig, sc, axes, "Corridor frequency")
    save_panel(pfig, "fig2b_prey_only_corridor_frequency_selected_years_v1")
    plt.close(pfig)

    pfig, pax = plt.subplots(figsize=(3.5, 2.1))
    plot_cost_area(pax, route)
    add_outer_panel_label(pax, "C")
    save_panel(pfig, "fig2c_prey_only_cost_area_timeseries_v1")
    plt.close(pfig)

    pfig, pax = plt.subplots(figsize=(3.5, 2.1))
    plot_centroid_trajectory(pax, centroid, land, ocean_grid=ocean_grid, full_extent=extent)
    add_outer_panel_label(pax, "D")
    save_panel(pfig, "fig2d_prey_only_centroid_trajectory_v1")
    plt.close(pfig)

    source_route = route.copy()
    source_route.to_csv(OUT_DIR / "gray_whale_cee_figure2_route_source_data_v1.csv", index=False)
    source_centroid = centroid.drop(columns=["source_files"], errors="ignore").copy()
    source_centroid.to_csv(OUT_DIR / "gray_whale_cee_figure2_centroid_source_data_v1.csv", index=False)
    selected_maps = []
    for year, df in grids.items():
        cols = [
            "grid_id",
            "lon",
            "lat",
            "target_year",
            "U_amphipod_prey_global_q05_q95",
            "lcp_frequency_normalized",
            "lcp_count",
            "ocean_flag",
        ]
        selected_maps.append(df[[c for c in cols if c in df.columns]].copy())
    pd.concat(selected_maps, ignore_index=True).to_csv(
        OUT_DIR / "gray_whale_cee_figure2_selected_map_source_data_v1.csv",
        index=False,
    )


if __name__ == "__main__":
    build_figure()
