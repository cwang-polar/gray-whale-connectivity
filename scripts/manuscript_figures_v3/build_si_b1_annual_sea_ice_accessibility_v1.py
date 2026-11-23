#!/usr/bin/env python3
"""Build the isolated all-year May-July sea-ice accessibility SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-si-b1")

import geopandas as gpd
import matplotlib as mpl
from matplotlib import colors, patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "b1_annual_sea_ice_accessibility_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

GRID_PATH = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
LAND_PATH = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"
ICE_TEMPLATE = ROOT / "outputs" / "ice" / "ice_accessibility_v1_MayJul_target{year}.csv"
FROZEN_SUMMARY = (
    ROOT
    / "outputs"
    / "manuscript_figures"
    / "supplementary_figures"
    / "figure_s2_sea_ice_accessibility"
    / "gray_whale_supplementary_figure_s2_sea_ice_accessibility_summary_source_data_v3.csv"
)

YEARS = tuple(range(2011, 2020))
OCEAN_COLOR = "#E9F2F5"
LAND_COLOR = "#D8D2C7"
TEXT_COLOR = "#2B2B2B"
LOW_ACCESS_COLOR = "#4F8398"
LINE_COLOR = "#1D4056"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_ocean_grid() -> pd.DataFrame:
    grid = pd.read_csv(GRID_PATH)
    in_domain = grid["inside_C3_domain"].astype(str).str.lower().eq("true")
    grid = grid.loc[grid["ocean_flag"].astype(int).eq(1) & in_domain].copy()
    grid["grid_id"] = grid["grid_id"].astype(str)
    if len(grid) != 2839 or grid["grid_id"].duplicated().any():
        raise ValueError("Unexpected study-domain ocean grid")
    return grid


def load_annual_ice(
    grid: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], list[Path]]:
    expected_ids = set(grid["grid_id"])
    annual: dict[int, pd.DataFrame] = {}
    inputs: list[Path] = []
    columns = [
        "grid_id",
        "lon",
        "lat",
        "target_year",
        "SIC_MayJul_mean_filled",
        "A_ice_MayJul_mean_from_monthly_A_filled",
        "moderate_access_Aice_lt_0_50",
        "high_SIC_gt_0_55",
    ]
    for year in YEARS:
        path = Path(str(ICE_TEMPLATE).format(year=year))
        frame = pd.read_csv(path, usecols=columns)
        frame["grid_id"] = frame["grid_id"].astype(str)
        frame = frame.loc[frame["grid_id"].isin(expected_ids)].copy()
        if len(frame) != 2839 or set(frame["grid_id"]) != expected_ids:
            raise ValueError(f"Grid mismatch in {path}")
        if frame["grid_id"].duplicated().any() or frame[columns].isna().any().any():
            raise ValueError(f"Missing or duplicate values in {path}")
        if set(frame["target_year"].astype(int)) != {year}:
            raise ValueError(f"Unexpected target year in {path}")
        accessibility = frame["A_ice_MayJul_mean_from_monthly_A_filled"]
        if ((accessibility < 0.05) | (accessibility > 1.0)).any():
            raise ValueError(f"Accessibility outside [0.05, 1] in {path}")
        low_flag = frame["moderate_access_Aice_lt_0_50"].astype(bool)
        high_sic_flag = frame["high_SIC_gt_0_55"].astype(bool)
        if not low_flag.equals(accessibility.lt(0.5)):
            raise ValueError(f"A_ice < 0.5 flag mismatch in {path}")
        if not high_sic_flag.equals(frame["SIC_MayJul_mean_filled"].gt(0.55)):
            raise ValueError(f"SIC > 0.55 flag mismatch in {path}")
        annual[year] = frame.sort_values("grid_id").reset_index(drop=True)
        inputs.append(path)
    return annual, inputs


def summarize_annual_ice(annual: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        frame = annual[year]
        rows.append(
            {
                "target_year": year,
                "ocean_cell_count": len(frame),
                "mean_A_ice": frame[
                    "A_ice_MayJul_mean_from_monthly_A_filled"
                ].mean(),
                "median_A_ice": frame[
                    "A_ice_MayJul_mean_from_monthly_A_filled"
                ].median(),
                "mean_SIC": frame["SIC_MayJul_mean_filled"].mean(),
                "percent_cells_Aice_lt_0_50": frame[
                    "moderate_access_Aice_lt_0_50"
                ].astype(bool).mean()
                * 100.0,
                "percent_cells_SIC_gt_0_55": frame["high_SIC_gt_0_55"].astype(
                    bool
                ).mean()
                * 100.0,
            }
        )
    summary = pd.DataFrame(rows)
    frozen = pd.read_csv(FROZEN_SUMMARY).sort_values("target_year").reset_index(drop=True)
    summary = summary.sort_values("target_year").reset_index(drop=True)
    if list(summary.columns) != list(frozen.columns):
        raise ValueError("Frozen summary columns changed")
    for column in summary.columns:
        if column == "target_year" or column == "ocean_cell_count":
            if not summary[column].equals(frozen[column]):
                raise ValueError(f"Frozen summary mismatch: {column}")
        elif not np.allclose(summary[column], frozen[column], rtol=0, atol=1e-12):
            raise ValueError(f"Frozen summary mismatch: {column}")
    return summary


def map_extent(grid: pd.DataFrame) -> tuple[float, float, float, float]:
    pad = 1.25
    return (
        float(grid["lon"].min() - pad),
        float(grid["lon"].max() + pad),
        float(grid["lat"].min() - pad * 0.45),
        float(grid["lat"].max() + pad * 0.25),
    )


def set_map_aspect(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    _, _, ymin, ymax = extent
    mean_lat = (ymin + ymax) / 2.0
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box", anchor="C")


def add_geographic_labels(ax: plt.Axes) -> None:
    labels = [
        ("Russia", -173.0, 65.2, 4.7, "normal"),
        ("Alaska", -158.5, 66.4, 4.7, "normal"),
        ("Chukchi Sea", -168.1, 69.0, 4.3, "italic"),
        ("Bering\nStrait", -168.9, 65.8, 4.1, "italic"),
        ("Chirikov\nBasin", -170.1, 64.3, 4.1, "italic"),
        ("St. Lawrence\nIsland", -170.2, 63.45, 3.9, "normal"),
    ]
    for label, lon, lat, size, style in labels:
        item = ax.text(
            lon,
            lat,
            label,
            fontsize=size,
            color="#4A4640",
            fontstyle=style,
            ha="center",
            va="center",
            zorder=9,
        )
        item.set_path_effects(
            [patheffects.withStroke(linewidth=1.25, foreground="white")]
        )


def draw_map(
    ax: plt.Axes,
    year: int,
    frame: pd.DataFrame,
    land: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    cmap: colors.Colormap,
    norm: colors.Normalize,
    show_labels: bool,
) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(OCEAN_COLOR)
    ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame["A_ice_MayJul_mean_from_monthly_A_filled"],
        cmap=cmap,
        norm=norm,
        s=4.5,
        marker="s",
        linewidths=0,
        alpha=0.98,
        zorder=3,
        rasterized=True,
    )
    high_sic = frame.loc[frame["high_SIC_gt_0_55"].astype(bool)]
    ax.scatter(
        high_sic["lon"],
        high_sic["lat"],
        facecolors="none",
        edgecolors="#31576B",
        linewidths=0.18,
        marker="s",
        s=5.0,
        alpha=0.36,
        zorder=5,
        rasterized=True,
    )
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax,
        color=LAND_COLOR,
        edgecolor="white",
        linewidth=0.20,
        zorder=7,
        rasterized=True,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    set_map_aspect(ax, extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(year), fontsize=6.8, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#555555")
    if show_labels:
        add_geographic_labels(ax)


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        0.045,
        box.y1 + 0.010,
        label,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#111111",
    )


def build_figure(
    annual: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    grid: pd.DataFrame,
) -> list[Path]:
    configure_style()
    land = gpd.read_file(LAND_PATH).to_crs(4326)
    extent = map_extent(grid)
    cmap = colors.LinearSegmentedColormap.from_list(
        "ice_accessibility",
        ["#1D4056", "#4F8398", "#A9CEDB", "#F8FBFC"],
    )
    norm = colors.Normalize(vmin=0.05, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 9.35), facecolor="white")
    grid_spec = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.0, 1.0, 1.0, 0.14, 0.58],
        left=0.075,
        right=0.925,
        bottom=0.070,
        top=0.975,
        wspace=0.055,
        hspace=0.13,
    )

    map_axes: list[plt.Axes] = []
    for index, year in enumerate(YEARS):
        ax = fig.add_subplot(grid_spec[index // 3, index % 3])
        draw_map(
            ax,
            year,
            annual[year],
            land,
            extent,
            cmap,
            norm,
            show_labels=index == 0,
        )
        map_axes.append(ax)

    info_ax = fig.add_subplot(grid_spec[3, :])
    info_ax.axis("off")
    info_box = info_ax.get_position()
    cax = fig.add_axes(
        [
            info_box.x0 + 0.02,
            info_box.y0 + info_box.height * 0.47,
            info_box.width * 0.43,
            0.010,
        ]
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0.05, 0.25, 0.50, 0.75, 1.00])
    colorbar.ax.tick_params(labelsize=5.7, length=2, width=0.5, pad=1.5)
    colorbar.set_label(
        r"May-July accessibility, $A_{\mathrm{ice}}$", fontsize=6.2, labelpad=1.5
    )
    colorbar.outline.set_linewidth(0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            marker="o",
            linewidth=1.25,
            markersize=3.4,
            label=r"Mean $A_{\mathrm{ice}}$",
        ),
        mpl.patches.Patch(
            facecolor=LOW_ACCESS_COLOR,
            edgecolor=LOW_ACCESS_COLOR,
            alpha=0.24,
            label=r"Cells with $A_{\mathrm{ice}}<0.5$",
        ),
    ]
    info_ax.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.99, 0.48),
        ncol=2,
        fontsize=6.0,
        handlelength=1.6,
        columnspacing=1.1,
    )

    trend_ax = fig.add_subplot(grid_spec[4, :])
    trend_ax.plot(
        summary["target_year"],
        summary["mean_A_ice"],
        color=LINE_COLOR,
        marker="o",
        linewidth=1.35,
        markersize=3.7,
        zorder=4,
    )
    trend_ax.set_xlim(2010.55, 2019.45)
    trend_ax.set_ylim(0.40, 1.02)
    trend_ax.set_xticks(YEARS)
    trend_ax.set_xlabel("Target year", labelpad=3)
    trend_ax.set_ylabel(r"Mean $A_{\mathrm{ice}}$", color=LINE_COLOR, labelpad=4)
    trend_ax.tick_params(
        axis="y", labelcolor=LINE_COLOR, direction="out", length=2.5, width=0.6
    )
    trend_ax.tick_params(axis="x", direction="out", length=2.5, width=0.6)
    trend_ax.grid(axis="y", color="#E3E6E8", linewidth=0.55, zorder=0)

    bar_ax = trend_ax.twinx()
    bar_ax.bar(
        summary["target_year"],
        summary["percent_cells_Aice_lt_0_50"],
        width=0.62,
        color=LOW_ACCESS_COLOR,
        edgecolor=LOW_ACCESS_COLOR,
        linewidth=0.45,
        alpha=0.24,
        zorder=1,
    )
    bar_ax.set_ylim(0, 70)
    bar_ax.set_yticks([0, 20, 40, 60])
    bar_ax.set_ylabel(
        r"Ocean cells with $A_{\mathrm{ice}}<0.5$ (%)",
        color=LOW_ACCESS_COLOR,
        labelpad=4,
    )
    bar_ax.tick_params(
        axis="y",
        labelcolor=LOW_ACCESS_COLOR,
        direction="out",
        length=2.5,
        width=0.6,
    )
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(True)
    bar_ax.spines["right"].set_color(LOW_ACCESS_COLOR)
    bar_ax.spines["right"].set_linewidth(0.7)

    fig.canvas.draw()
    add_panel_label(fig, map_axes[0], "A")
    add_panel_label(fig, trend_ax, "B")

    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    stem = CANDIDATE_OUT / "gray_whale_si_b1_annual_sea_ice_accessibility_candidate_v1"
    outputs = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=400, facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    QA_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        QA_OUT / "gray_whale_si_b1_annual_sea_ice_accessibility_review_v1.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)
    return outputs


def write_supporting_files(
    annual: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    inputs: list[Path],
    figure_paths: list[Path],
) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_OUT / "annual_sea_ice_accessibility_summary_v1.csv"
    cells_path = SOURCE_OUT / "annual_sea_ice_accessibility_cells_v1.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    pd.concat([annual[year] for year in YEARS], ignore_index=True).to_csv(
        cells_path,
        index=False,
        float_format="%.10f",
    )

    raster_qa: dict[str, dict[str, float | int | bool]] = {}
    for path in figure_paths:
        if path.suffix.lower() not in {".png", ".tiff"}:
            continue
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"))
            raster_qa[path.suffix.lower()] = {
                "width_px": image.width,
                "height_px": image.height,
                "nonblank": bool(gray.std() > 0.5),
                "mean_luminance": float(gray.mean()),
            }

    qa = {
        "backend": "Python",
        "environment": "mypy10",
        "years": list(YEARS),
        "annual_map_count": len(YEARS),
        "ocean_cells_per_year": 2839,
        "frozen_summary_reproduction_pass": True,
        "missing_cell_values": int(
            sum(frame.isna().sum().sum() for frame in annual.values())
        ),
        "raster_exports": raster_qa,
        "manuscript_files_modified": False,
        "existing_figure_files_modified": False,
        "input_files": [str(path.relative_to(ROOT)) for path in inputs],
        "figure_files": [str(path.relative_to(ROOT)) for path in figure_paths],
    }
    (QA_OUT / "b1_annual_sea_ice_accessibility_qa_v1.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )


def main() -> None:
    grid = load_ocean_grid()
    annual, inputs = load_annual_ice(grid)
    summary = summarize_annual_ice(annual)
    figure_paths = build_figure(annual, summary, grid)
    write_supporting_files(annual, summary, inputs, figure_paths)
    for path in figure_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
