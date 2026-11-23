
# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale")

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "manuscript_figures" / "figure_1"
PANEL_DIR = OUT_DIR / "panels_v1"

DOMAIN = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
ENDPOINTS = ROOT / "data/processed/study_domain/fixed_endpoint_candidates_main_south_to_north_C3_v2_clean_largest_component.csv"
DBO = ROOT / "data/processed/study_domain/dbo_anchor_polygons.gpkg"
LAND = ROOT / "data/raw/prey/ne_10m_land/ne_10m_land.shp"
PREY = ROOT / "data/processed/prey/dominant_macroinfauna_clean_1970_2019.csv"


PALETTE = {
    "ocean": "#E9F2F5",
    "land": "#D8D2C7",
    "domain": "#0B4F71",
    "grid": "#B7C7CE",
    "prey": "#4D4D4D",
    "source": "#1B9E77",
    "sink": "#6A51A3",
    "path": "#C75D2C",
    "ice": "#84C5D8",
    "ship": "#7F7F7F",
    "dbo": "#315B70",
    "neutral": "#3F3F3F",
    "soft": "#F6F6F3",
}


def _prepare_legacy_export_dirs():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)


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


def add_panel_label(ax, label, x=0.015, y=0.985):
    # Keep labels inside each panel so they do not expand the saved figure box.
    x = max(0.012, x)
    y = min(0.985, y)
    ax.text(
        x,
        y,
        label.upper(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#1F1F1F",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.0),
        zorder=30,
    )


def make_dbo_reference_polygons():
    coords = {
        "DBO1": {
            "lat": [63.769, 61.946, 61.847, 63.663],
            "lon": [-172.312, -172.187, -175.791, -176.147],
        },
        "DBO2": {
            "lat": [65.111, 65.097, 64.482, 64.496],
            "lon": [-170.492, -167.860, -167.908, -170.481],
        },
        "DBO3": {
            "lat": [66.786, 68.609, 68.572, 66.752],
            "lon": [-171.330, -171.419, -166.481, -166.756],
        },
        "DBO4": {
            "lat": [71.867, 71.721, 70.682, 70.820],
            "lon": [-164.255, -160.507, -160.994, -164.553],
        },
        "DBO5": {
            "lat": [71.808, 71.636, 71.111, 71.278],
            "lon": [-158.538, -155.931, -156.307, -158.848],
        },
    }
    rows = []
    for name, xy in coords.items():
        rows.append(
            {
                "name": name,
                "geometry": Polygon(zip(xy["lon"], xy["lat"])),
            }
        )
    return gpd.GeoDataFrame(rows, crs=4326)


def load_ocean_grid():
    grid = pd.read_csv(GRID)
    in_domain = grid["inside_C3_domain"].astype(str).str.lower().eq("true")
    return grid[(grid["ocean_flag"] == 1) & in_domain].copy()


def load_layers():
    domain = gpd.read_file(DOMAIN).to_crs(4326)
    grid = load_ocean_grid()
    dbo = make_dbo_reference_polygons()
    land = gpd.read_file(LAND).to_crs(4326)
    endpoints = pd.read_csv(ENDPOINTS)
    prey = pd.read_csv(PREY)
    prey = prey[(prey["DataYear"] >= 2007) & (prey["DataYear"] <= 2019)].copy()
    prey = prey.dropna(subset=["Longitude", "Latitude"])
    prey_gdf = gpd.GeoDataFrame(
        prey,
        geometry=gpd.points_from_xy(prey["Longitude"], prey["Latitude"]),
        crs=4326,
    )
    analysis_geom = domain.geometry.iloc[0]
    prey_gdf = prey_gdf[prey_gdf.within(analysis_geom)].copy()
    return domain, grid, dbo, land, endpoints, prey_gdf


def extent_from_ocean_grid(ocean_grid, pad=1.65):
    minx, maxx = ocean_grid["lon"].min(), ocean_grid["lon"].max()
    miny, maxy = ocean_grid["lat"].min(), ocean_grid["lat"].max()
    return (minx - pad, maxx + pad * 1.35, miny - pad * 0.55, maxy + pad * 0.35)


def draw_base_map(ax, ocean_grid, land, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax, color=PALETTE["land"], edgecolor="white", linewidth=0.35, zorder=1
    )
    ax.scatter(
        ocean_grid["lon"],
        ocean_grid["lat"],
        s=3.0,
        marker="s",
        color=PALETTE["grid"],
        edgecolor="none",
        alpha=0.28,
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mean_lat = (ymin + ymax) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=7, length=2.5)
    ax.grid(color="#FFFFFF", linewidth=0.45, zorder=0)


def add_geographic_labels(ax):
    labels = [
        ("St. Lawrence\nIsland", -170.4, 63.75, "center"),
        ("Russia", -174.8, 65.6, "left"),
        ("Alaska\nUSA", -159.2, 69.2, "center"),
        ("Bering\nStrait", -168.4, 65.9, "center"),
        ("Wainwright", -160.05, 70.62, "center"),
        ("Barrow\nCanyon", -156.85, 71.65, "center"),
    ]
    for text, x, y, ha in labels:
        ax.text(
            x,
            y,
            text,
            fontsize=6.5,
            color="#5F5F5F",
            ha=ha,
            va="center",
            zorder=8,
        )


def add_chirikov_basin_label(ax):
    ax.annotate(
        "Chirikov\nBasin",
        xy=(-167.95, 64.66),
        xytext=(-165.55, 64.72),
        textcoords="data",
        fontsize=6.0,
        color="#5F5F5F",
        fontweight="normal",
        ha="center",
        va="center",
        zorder=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.5),
        arrowprops=dict(arrowstyle="-", color="#5F5F5F", linewidth=0.65),
    )


def add_dbo_labels(ax):
    positions = {
        "DBO1": (-174.6, 62.8),
        "DBO2": (-169.2, 64.7),
        "DBO3": (-168.9, 67.7),
        "DBO4": (-162.5, 71.25),
        "DBO5": (-157.35, 71.45),
    }
    for name, (x, y) in positions.items():
        ax.text(
            x,
            y,
            name,
            fontsize=7.0,
            color=PALETTE["dbo"],
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.5),
        )


def panel_a(ax):
    ax.set_axis_off()
    add_panel_label(ax, "A", x=0.004, y=0.96)

    boxes = [
        (0.02, 0.24, 0.20, 0.34, "Traditional prey\nbase weakens", PALETTE["path"]),
        (0.29, 0.24, 0.20, 0.34, "Northward access\nmust be tested", PALETTE["source"]),
        (0.56, 0.24, 0.20, 0.34, "Annual usable\ncorridors shift", PALETTE["ice"]),
        (0.80, 0.24, 0.18, 0.34, "Vessel exposure\nintersects cells", PALETTE["ship"]),
    ]
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=color,
            edgecolor="none",
            alpha=0.88,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white" if color != PALETTE["ice"] else "#1F3F48",
            fontsize=8,
            fontweight="bold",
        )
    for x0, x1 in [(0.225, 0.285), (0.495, 0.555), (0.765, 0.795)]:
        ax.annotate(
            "",
            xy=(x1, 0.41),
            xytext=(x0, 0.41),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=PALETTE["neutral"]),
        )


def panel_b(ax, ocean_grid, dbo, land, prey_gdf, extent):
    add_panel_label(ax, "B", x=0.012, y=0.985)
    draw_base_map(ax, ocean_grid, land, extent)
    xmin, xmax, ymin, ymax = extent
    prey_clip = prey_gdf.cx[xmin:xmax, ymin:ymax]
    if len(prey_clip):
        prey_clip.plot(
            ax=ax,
            color=PALETTE["prey"],
            markersize=8,
            alpha=0.42,
            linewidth=0,
            zorder=3,
        )
    dbo.boundary.plot(
        ax=ax,
        color=PALETTE["dbo"],
        linewidth=1.25,
        linestyle=(0, (4.0, 2.2)),
        zorder=7,
    )
    add_dbo_labels(ax)
    add_geographic_labels(ax)
    add_chirikov_basin_label(ax)
    ax.scatter([], [], s=12, color=PALETTE["prey"], alpha=0.45, label="Benthic prey stations")
    ax.scatter([], [], s=10, marker="s", color=PALETTE["grid"], alpha=0.55, label="Regional ocean grid")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.6)


def panel_c(ax, ocean_grid, land, endpoints, extent):
    add_panel_label(ax, "C", x=0.012, y=0.985)
    draw_base_map(ax, ocean_grid, land, extent)
    add_geographic_labels(ax)
    src = endpoints[endpoints["endpoint_type"] == "source"]
    dst = endpoints[endpoints["endpoint_type"] != "source"]
    ax.scatter(
        src["lon"],
        src["lat"],
        s=18,
        color=PALETTE["source"],
        edgecolor="white",
        linewidth=0.25,
        zorder=6,
        label="40 source endpoints",
    )
    ax.scatter(
        dst["lon"],
        dst["lat"],
        s=22,
        marker="^",
        color=PALETTE["sink"],
        edgecolor="white",
        linewidth=0.25,
        zorder=6,
        label="40 destination endpoints",
    )
    ax.scatter(
        [],
        [],
        s=8,
        color=PALETTE["grid"],
        alpha=0.55,
        label="Discretized ocean grid",
    )
    ax.legend(loc="lower right", fontsize=6.5)


def panel_d(ax):
    ax.set_axis_off()
    add_panel_label(ax, "D", x=0.004, y=0.965)

    # Mini source and destination endpoint columns.
    yvals = [0.22, 0.36, 0.50, 0.64, 0.78]
    for y in yvals:
        ax.scatter(0.10, y, s=34, color=PALETTE["source"], transform=ax.transAxes, zorder=3)
        ax.scatter(0.38, y, s=40, marker="^", color=PALETTE["sink"], transform=ax.transAxes, zorder=3)
    for y0 in yvals:
        for y1 in yvals[::2]:
            ax.plot(
                [0.11, 0.37],
                [y0, y1],
                color=PALETTE["path"],
                alpha=0.10,
                linewidth=0.8,
                transform=ax.transAxes,
                zorder=1,
            )
    ax.text(0.02, 0.08, "40 sources", transform=ax.transAxes, fontsize=6.3, ha="left", color=PALETTE["source"])
    ax.text(0.27, 0.08, "40 destinations", transform=ax.transAxes, fontsize=6.3, ha="left", color=PALETTE["sink"])
    ax.text(0.12, 0.92, "1,600 LCPs per year", transform=ax.transAxes, fontsize=7.4, ha="left", color=PALETTE["neutral"])
    ax.annotate(
        "",
        xy=(0.55, 0.50),
        xytext=(0.44, 0.50),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=PALETTE["neutral"]),
    )

    # Frequency grid.
    x0, y0, w, h = 0.62, 0.20, 0.28, 0.60
    n = 7
    vals = [
        [0.04, 0.06, 0.08, 0.12, 0.09, 0.06, 0.04],
        [0.06, 0.10, 0.18, 0.28, 0.20, 0.10, 0.05],
        [0.05, 0.12, 0.28, 0.54, 0.36, 0.14, 0.06],
        [0.04, 0.10, 0.22, 0.44, 0.62, 0.30, 0.10],
        [0.03, 0.06, 0.12, 0.24, 0.42, 0.55, 0.20],
        [0.02, 0.04, 0.08, 0.14, 0.22, 0.30, 0.18],
        [0.01, 0.02, 0.04, 0.08, 0.10, 0.14, 0.08],
    ]
    for i in range(n):
        for j in range(n):
            v = vals[n - 1 - i][j]
            color = plt.cm.YlOrBr(0.18 + 0.78 * v / 0.62)
            ax.add_patch(
                patches.Rectangle(
                    (x0 + j * w / n, y0 + i * h / n),
                    w / n,
                    h / n,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    ax.text(0.66, 0.08, "frequency map", transform=ax.transAxes, fontsize=6.3, color=PALETTE["neutral"])
    ax.text(
        0.60,
        0.88,
        "cell value = proportion\nof paths crossing cell",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#666666",
        ha="left",
    )


def save_panel(fig, name):
    fig.savefig(PANEL_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PANEL_DIR / f"{name}.png", dpi=450, bbox_inches="tight")


def build_figure():
    apply_style()
    domain, ocean_grid, dbo, land, endpoints, prey_gdf = load_layers()
    extent = extent_from_ocean_grid(ocean_grid)

    fig = plt.figure(figsize=(7.2, 8.9), constrained_layout=False)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.58, 2.95, 1.95],
        hspace=0.16,
        wspace=0.16,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2:])
    ax_d = fig.add_subplot(gs[2, :])

    panel_a(ax_a)
    panel_b(ax_b, ocean_grid, dbo, land, prey_gdf, extent)
    panel_c(ax_c, ocean_grid, land, endpoints, extent)
    panel_d(ax_d)

    out = OUT_DIR / "gray_whale_cee_figure1_draft_v1"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure 1_updated.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure 1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Individual panel exports for manuscript workbench.
    for name, drawer in [
        ("fig1a_conceptual_mechanism_v1", lambda ax: panel_a(ax)),
        ("fig1d_lcp_ensemble_schematic_v1", lambda ax: panel_d(ax)),
    ]:
        pfig, pax = plt.subplots(figsize=(7.0, 1.8 if "conceptual" in name else 3.1))
        drawer(pax)
        save_panel(pfig, name)
        plt.close(pfig)

    for name, drawer in [
        ("fig1b_c3_domain_station_context_v1", lambda ax: panel_b(ax, ocean_grid, dbo, land, prey_gdf, extent)),
        ("fig1c_endpoint_design_v1", lambda ax: panel_c(ax, ocean_grid, land, endpoints, extent)),
    ]:
        pfig, pax = plt.subplots(figsize=(3.0, 4.4))
        drawer(pax)
        save_panel(pfig, name)
        plt.close(pfig)


if __name__ == "__main__":
    build_figure()
