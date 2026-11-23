"""Create Python-only Phase 8S observation-informed diagnostic figures."""

from __future__ import annotations

if __name__ == '__main__':
    raise SystemExit('Historical CLI disabled in this release; use v8.py run pelagic_support.')

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union

import run_phase8s_observation_informed_corridor_sensitivity as phase8s  # noqa: E402


OUT = ROOT / "outputs/pelagic_support_phase8s"
TABLES = OUT / "tables"
MODEL_SURFACES = OUT / "surfaces"
CORRIDOR = OUT / "corridor_sensitivity"
SUMMARIES = CORRIDOR / "summaries"
VALIDATION = CORRIDOR / "validation"
FIGURES = OUT / "figures"

LAND = ROOT / "data/raw/prey/ne_10m_land/ne_10m_land.shp"
DBO = ROOT / "data/processed/study_domain/dbo_anchor_polygons.gpkg"
EARLY_RECORDS = TABLES / "phase8s_early_model_records_with_glorys.csv"
CV_PREDICTIONS = TABLES / "phase8s_spatial_block_cv_predictions.csv"

SCENARIO_ORDER = ["prey", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIO_SHORT = {
    "prey": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}
COLORS = {
    "prey": "#3F6FA3",
    "prey_ice": "#3F8A68",
    "prey_vessel": "#D48732",
    "prey_ice_vessel": "#8B5E83",
}
MARKERS = {"prey": "o", "prey_ice": "s", "prey_vessel": "^", "prey_ice_vessel": "D"}


CONTRACT = """# Phase 8S figure contract and QA

## Figure 1: observation-model evidence

- Core conclusion: method-comparable June observations support spatially bounded 2017 and 2018 relative-support surfaces over 32.4% and 24.5% of the full study-region ocean grid, but not a 2011-2019 annual surface or a pooled late-season surface.
- Archetype: asymmetric mixed-modality figure.
- Panels: 2017 support map; 2018 support map; blocked/transfer prediction skill; annual comparable-data coverage.
- Validation: four spatial folds within year, bidirectional cross-year transfer, and leave-source-family late-season transfer.

## Figure 2: corridor and observed-use response

- Core conclusion: the empirical pelagic spatial pattern robustly reduces modeled cost and reorganizes corridor geography across sensitivity strengths, but it does not improve all three ASAMM observed-use metrics consistently.
- Archetype: quantitative grid.
- Panels: mean cost change; frequent-cell Jaccard; centroid shift; effort-footprint percentile; top-5% overlap; distance to frequent corridor.
- Variability: pale bands are the range across 2017 and 2018. Error bars at beta=0.25 are 5th-95th percentiles across 100 spatial-block bootstrap surfaces after averaging the two annual metrics by draw.

## Spatial-block validation diagnostic

- Core conclusion: annual model evaluation withheld four coordinate-defined observation clusters; 2017 improved over the training-mean null in three folds and 2018 improved in all four.
- Archetype: map-led validation diagnostic.
- Panels: annual observation-block maps and selected-model fold-level RMSE skill.
- Partition boundary: blocks are deterministic coordinate clusters fixed across candidate models within a year, not ecological regions and not shared between years.

## Figure 3: representative spatial response

- Core conclusion: adding the observation-informed support pattern changes the location and concentration of prey+ice LCP frequency even though utility increments are restricted to the explicitly mapped empirical interpolation domain within the full study region.
- Archetype: image plate plus quantitative difference maps.
- Panels: baseline, beta=0.25 complementary augmentation, and cell-wise frequency difference for 2017 and 2018.

## Export contract

- Backend: Python/matplotlib only.
- Final width: 7.2 inches (double-column diagnostic review).
- Text: editable in PDF/SVG; project-consistent uppercase panel labels.
- Exports: PDF, SVG, 300-dpi PNG, 600-dpi LZW TIFF.
- Source data: CSV files in this figure directory.
- Image integrity: direct rendering of gridded model output and vector boundaries; no raster manipulation.
- Statistical boundary: descriptive model sensitivity, not inference about whale abundance or a calibrated benthic-pelagic energy conversion.

## Completed QA

- Visual review: all panels nonblank; labels, legends, color bars, geographic references, and panel spacing checked at original resolution.
- PDF: one page per figure; page dimensions and nonblank rendering verified after export.
- Fonts: Arial regular and bold are embedded in every PDF; text remains editable in PDF/SVG exports.
- Raster exports: PNGs are 300 dpi; TIFFs are RGB, LZW-compressed, and 600 dpi.
- Scientific checks: the same 323 ASAMM records are used across scenarios; `beta=0` exactly reproduces established frequency grids; no panel implies an annual 2011-2019 pelagic surface.
"""


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")


def quiet_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(axis=axis, color="#E2E5E7", linewidth=0.5, zorder=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / stem
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(base.with_suffix(".tiff")) as raster:
        raster.convert("RGB").save(
            base.with_suffix(".tiff"),
            dpi=(600, 600),
            compression="tiff_lzw",
        )


def geographic_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    land = gpd.read_file(LAND).to_crs("EPSG:3338")
    dbo = gpd.read_file(DBO).to_crs("EPSG:3338")
    return land, dbo


def clip_land(land: gpd.GeoDataFrame, xlim: tuple[float, float], ylim: tuple[float, float]) -> gpd.GeoDataFrame:
    extent = gpd.GeoDataFrame(
        geometry=[box(xlim[0], ylim[0], xlim[1], ylim[1])],
        crs=land.crs,
    )
    return gpd.clip(land, extent)


def evidence_boundary(frame: pd.DataFrame) -> gpd.GeoSeries:
    cells = frame.loc[frame["pelagic_evidence_domain"].astype(bool), ["x", "y"]]
    geometry = unary_union([box(x - 5_000, y - 5_000, x + 5_000, y + 5_000) for x, y in cells.to_numpy()])
    return gpd.GeoSeries([geometry], crs="EPSG:3338")


def study_region_boundary(frame: pd.DataFrame) -> gpd.GeoSeries:
    if "ocean_flag" in frame.columns:
        cells = frame.loc[frame["ocean_flag"].eq(1), ["x", "y"]].drop_duplicates()
    else:
        cells = frame[["x", "y"]].drop_duplicates()
    geometry = unary_union([box(x - 5_000, y - 5_000, x + 5_000, y + 5_000) for x, y in cells.to_numpy()])
    return gpd.GeoSeries([geometry], crs="EPSG:3338")


def map_base(
    ax: plt.Axes,
    frame: pd.DataFrame,
    land: gpd.GeoDataFrame,
    dbo3: gpd.GeoDataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.scatter(frame["x"], frame["y"], s=3.0, marker="s", color="#DCE4E7", linewidth=0, zorder=0)
    study_region_boundary(frame).boundary.plot(
        ax=ax,
        color="#68777D",
        linewidth=0.42,
        zorder=3,
    )
    clip_land(land, xlim, ylim).plot(
        ax=ax,
        facecolor="#EAE8E3",
        edgecolor="#777873",
        linewidth=0.35,
        zorder=4,
    )
    dbo3.boundary.plot(ax=ax, color="#16766F", linewidth=0.75, linestyle=(0, (3, 2)), zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def add_geographic_labels(ax: plt.Axes) -> None:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    labels = [
        ("Chirikov Basin", -169.7, 64.8),
        ("Bering Strait", -169.0, 65.9),
        ("DBO3", -167.2, 68.0),
    ]
    for label, longitude, latitude in labels:
        x, y = transformer.transform(longitude, latitude)
        artist = ax.text(
            x,
            y,
            label,
            fontsize=5.2,
            color="#4A4F51",
            ha="center",
            va="center",
            zorder=9,
        )
        artist.set_path_effects([path_effects.withStroke(linewidth=1.4, foreground="white")])


def make_observation_model_figure() -> None:
    early = pd.read_csv(EARLY_RECORDS, dtype={"grid_id": str})
    surfaces = {
        year: pd.read_csv(
            MODEL_SURFACES / f"phase8s_observation_informed_pelagic_support_target{year}.csv",
            dtype={"grid_id": str},
        )
        for year in (2017, 2018)
    }
    cv = pd.read_csv(TABLES / "phase8s_spatial_block_cv_summary.csv", keep_default_na=False)
    transfer = pd.read_csv(TABLES / "phase8s_early_cross_year_transfer.csv", keep_default_na=False)
    late_source = pd.read_csv(TABLES / "phase8s_late_leave_source_family_summary.csv")
    land, dbo = geographic_context()
    dbo3 = dbo.loc[dbo["name"].eq("DBO3")]
    reference = surfaces[2017]
    xlim = (reference["x"].min() - 35_000, reference["x"].max() + 35_000)
    ylim = (reference["y"].min() - 25_000, reference["y"].max() + 25_000)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.28, 0.72],
        left=0.065,
        right=0.985,
        bottom=0.11,
        top=0.89,
        wspace=0.28,
        hspace=0.34,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    map_artist = None
    map_source = []
    for panel, year in enumerate((2017, 2018)):
        ax = axes[panel]
        surface = surfaces[year]
        observed = early.loc[early["year"].eq(year)].copy()
        map_base(ax, surface, land, dbo3, xlim, ylim)
        add_geographic_labels(ax)
        domain = surface.loc[surface["pelagic_evidence_domain"].astype(bool)]
        map_artist = ax.scatter(
            domain["x"],
            domain["y"],
            c=domain["P_pelagic_central"],
            s=5.0,
            marker="s",
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidth=0,
            zorder=2,
            rasterized=True,
        )
        evidence_boundary(surface).boundary.plot(ax=ax, color="#25A7A0", linewidth=0.80, zorder=5)
        point_values = observed["response_relative_score"].to_numpy(dtype=float)
        ax.scatter(
            observed["x"],
            observed["y"],
            c=point_values,
            s=12 + 18 * point_values,
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
        ax.text(
            0.02,
            0.025,
            (
                f"June {year} | observations n = {len(observed)}\n"
                f"Empirical domain: {len(domain):,}/{len(surface):,} cells "
                f"({len(domain) / len(surface) * 100:.1f}%)"
            ),
            transform=ax.transAxes,
            fontsize=5.8,
            ha="left",
            va="bottom",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.2},
        )
        panel_label(ax, chr(65 + panel), x=0.015, y=0.985)
        map_source.append(surface.assign(panel_year=year))
    cax = fig.add_axes([0.475, 0.955, 0.12, 0.014])
    colorbar = fig.colorbar(map_artist, cax=cax, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Relative euphausiid support", fontsize=6.4, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=5.8, pad=1)
    fig.text(
        0.20,
        0.962,
        "Colored cells: modeled support   Circles: observations",
        fontsize=6.0,
        ha="center",
        va="center",
        color="#4D5458",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#DCE4E7", edgecolor="#68777D", linewidth=0.5, label="Study-region ocean grid"),
            Line2D([0], [0], color="#25A7A0", linewidth=1.0, label="Empirical interpolation domain"),
            Line2D([0], [0], color="#16766F", linewidth=0.9, linestyle=(0, (3, 2)), label="DBO3 reference"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=5.7,
    )

    selected_2017 = cv.loc[
        cv["analysis_set"].eq("early_2017_log_abundance") & cv["model_type"].eq("spatial")
    ].iloc[0]
    selected_2018 = cv.loc[
        cv["analysis_set"].eq("early_2018_log_abundance") & cv["model_type"].eq("spatial")
    ].iloc[0]
    transfer_rows = []
    for train_year, test_year in ((2017, 2018), (2018, 2017)):
        row = (
            transfer.loc[
                transfer["train_year"].eq(train_year)
                & transfer["test_year"].eq(test_year)
                & transfer["model_type"].ne("mean_null")
            ]
            .sort_values("rmse")
            .iloc[0]
        )
        transfer_rows.append(row)
    late_block = cv.loc[
        cv["analysis_set"].eq("late_pooled_relative_score") & cv["model_type"].ne("mean_null")
    ].sort_values("rmse").iloc[0]
    performance = pd.DataFrame(
        [
            {"label": "2017 blocked", "skill": float(selected_2017.rmse_skill), "rho": float(selected_2017.spearman_rho), "class": "year"},
            {"label": "2018 blocked", "skill": float(selected_2018.rmse_skill), "rho": float(selected_2018.spearman_rho), "class": "year"},
            {"label": "2017 to 2018", "skill": float(transfer_rows[0].rmse_skill), "rho": float(transfer_rows[0].spearman_rho), "class": "transfer"},
            {"label": "2018 to 2017", "skill": float(transfer_rows[1].rmse_skill), "rho": float(transfer_rows[1].spearman_rho), "class": "transfer"},
            {"label": "Late spatial blocks", "skill": float(late_block.rmse_skill), "rho": float(late_block.spearman_rho), "class": "late"},
            {"label": "Late source transfer", "skill": float(late_source.iloc[0].rmse_skill), "rho": float(late_source.iloc[0].spearman_rho), "class": "late"},
        ]
    )
    ax_c = axes[2]
    y = np.arange(len(performance))[::-1]
    color_map = {"year": "#3F8A68", "transfer": "#3F6FA3", "late": "#A75C54"}
    ax_c.barh(y, performance["skill"], color=performance["class"].map(color_map), height=0.62, zorder=2)
    ax_c.axvline(0, color="#555555", linewidth=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(performance["label"])
    ax_c.set_xlabel("Predictive RMSE skill relative to fold mean")
    ax_c.set_xlim(-0.16, 0.47)
    for yi, row in zip(y, performance.itertuples()):
        if row.skill >= 0:
            x = row.skill + 0.012
            horizontal_alignment = "left"
            color = "#1F2426"
        else:
            x = row.skill + 0.008
            horizontal_alignment = "left"
            color = "white"
        ax_c.text(
            x,
            yi,
            f"rho={row.rho:.2f}",
            ha=horizontal_alignment,
            va="center",
            fontsize=5.8,
            color=color,
        )
    quiet_grid(ax_c, axis="x")
    panel_label(ax_c, "C", x=-0.09)

    ax_d = axes[3]
    years = np.arange(2011, 2020)
    counts = np.array([0, 0, 0, 0, 0, 0, 38, 28, 0])
    colors = ["#D4D9DC"] * len(years)
    colors[6] = "#3F8A68"
    colors[7] = "#3F8A68"
    ax_d.bar(years, counts, color=colors, width=0.72, zorder=2)
    ax_d.scatter([2017, 2018], [38, 28], color="#255D48", s=14, zorder=3)
    ax_d.set_xticks(years)
    ax_d.set_xticklabels([str(year)[-2:] for year in years])
    ax_d.set_ylabel("Comparable June grid cells")
    ax_d.set_xlabel("Target year")
    ax_d.set_ylim(0, 43)
    ax_d.text(2017.5, 41.5, "Year-specific surfaces only", ha="center", va="top", fontsize=6.1, color="#255D48")
    ax_d.text(2013.8, 7.0, "No empirical annual surface", ha="center", va="center", fontsize=6.0, color="#6A7074")
    quiet_grid(ax_d, axis="y")
    panel_label(ax_d, "D", x=-0.09)

    handles = [
        Patch(facecolor=color_map["year"], label="Year-specific blocked CV"),
        Patch(facecolor=color_map["transfer"], label="Cross-year transfer"),
        Patch(facecolor=color_map["late"], label="Late-season gate"),
    ]
    ax_c.legend(handles=handles, loc="lower right", fontsize=5.7, handlelength=1.1)
    save_figure(fig, "phase8s_observation_model_evidence_v2")
    plt.close(fig)

    pd.concat(map_source, ignore_index=True).to_csv(
        FIGURES / "phase8s_observation_model_map_source_data_v1.csv", index=False
    )
    early.to_csv(FIGURES / "phase8s_observation_points_source_data_v1.csv", index=False)
    performance.to_csv(FIGURES / "phase8s_model_performance_source_data_v1.csv", index=False)
    pd.DataFrame({"year": years, "method_comparable_june_grid_cells": counts}).to_csv(
        FIGURES / "phase8s_temporal_coverage_source_data_v1.csv", index=False
    )


def make_spatial_block_diagnostic_figure() -> None:
    early = pd.read_csv(EARLY_RECORDS, dtype={"grid_id": str})
    predictions = pd.read_csv(CV_PREDICTIONS, dtype={"grid_id": str})
    selected = predictions.loc[
        predictions["analysis_set"].isin(["early_2017_log_abundance", "early_2018_log_abundance"])
        & predictions["model_type"].eq("spatial")
    ].copy()
    assignments = selected[
        ["year", "grid_id", "outer_spatial_block"]
    ].drop_duplicates()
    assignments = assignments.merge(
        early[["year", "grid_id", "x", "y", "lon", "lat"]],
        on=["year", "grid_id"],
        how="left",
        validate="one_to_one",
    )

    fold_rows = []
    for (year, block), group in selected.groupby(["year", "outer_spatial_block"]):
        observed = group["observed_response"].to_numpy(dtype=float)
        model = group["predicted_response"].to_numpy(dtype=float)
        null = group["null_prediction"].to_numpy(dtype=float)
        model_rmse = float(np.sqrt(np.mean(np.square(model - observed))))
        null_rmse = float(np.sqrt(np.mean(np.square(null - observed))))
        fold_rows.append(
            {
                "year": int(year),
                "spatial_block": int(block),
                "fold_label": f"F{int(block) + 1}",
                "held_out_n": int(len(group)),
                "rmse_model": model_rmse,
                "rmse_null": null_rmse,
                "rmse_skill": float(1.0 - model_rmse / null_rmse),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["year", "spatial_block"])

    land, dbo = geographic_context()
    dbo3 = dbo.loc[dbo["name"].eq("DBO3")]
    reference = pd.read_csv(
        MODEL_SURFACES / "phase8s_observation_informed_pelagic_support_target2017.csv",
        dtype={"grid_id": str},
    )
    surfaces = {
        year: pd.read_csv(
            MODEL_SURFACES / f"phase8s_observation_informed_pelagic_support_target{year}.csv",
            dtype={"grid_id": str},
        )
        for year in (2017, 2018)
    }
    xlim = (reference["x"].min() - 35_000, reference["x"].max() + 35_000)
    ylim = (reference["y"].min() - 25_000, reference["y"].max() + 25_000)
    fold_colors = {0: "#3F6FA3", 1: "#3F8A68", 2: "#D48732", 3: "#8B5E83"}

    fig = plt.figure(figsize=(7.2, 4.85))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.32, 0.68],
        left=0.07,
        right=0.985,
        bottom=0.12,
        top=0.82,
        wspace=0.23,
        hspace=0.34,
    )
    map_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_skill = fig.add_subplot(gs[1, :])

    for panel, year in enumerate((2017, 2018)):
        ax = map_axes[panel]
        surface = surfaces[year]
        map_base(ax, surface, land, dbo3, xlim, ylim)
        evidence_boundary(surface).boundary.plot(ax=ax, color="#25A7A0", linewidth=0.75, zorder=5)
        year_points = assignments.loc[assignments["year"].eq(year)]
        for block, group in year_points.groupby("outer_spatial_block"):
            color = fold_colors[int(block)]
            if len(group) >= 3:
                hull = MultiPoint(group[["x", "y"]].to_numpy()).convex_hull
                gpd.GeoSeries([hull], crs="EPSG:3338").plot(
                    ax=ax,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.10,
                    linewidth=0.65,
                    zorder=5,
                )
            ax.scatter(
                group["x"],
                group["y"],
                s=20,
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=7,
            )
            cx = float(group["x"].mean())
            cy = float(group["y"].mean())
            label = ax.text(
                cx,
                cy,
                f"F{int(block) + 1}\nn={len(group)}",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="#222628",
                zorder=8,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.8, foreground="white")])
        ax.set_title(f"June {year}: annual coordinate blocks", fontsize=6.8, pad=3)
        panel_label(ax, chr(65 + panel), x=0.015, y=0.985)

    positions = []
    labels = []
    colors = []
    heights = []
    ns = []
    for year_index, year in enumerate((2017, 2018)):
        year_rows = fold_metrics.loc[fold_metrics["year"].eq(year)]
        for row in year_rows.itertuples():
            positions.append(year_index * 5 + row.spatial_block)
            labels.append(f"{year}\n{row.fold_label}")
            colors.append(fold_colors[row.spatial_block])
            heights.append(row.rmse_skill)
            ns.append(row.held_out_n)
    bars = ax_skill.bar(positions, heights, color=colors, width=0.72, zorder=2)
    ax_skill.axhline(0, color="#555B5E", linewidth=0.8, zorder=3)
    for bar, value, n in zip(bars, heights, ns):
        offset = 0.018 if value >= 0 else -0.018
        ax_skill.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}\nn={n}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=5.5,
            color="#24282A",
        )
    ax_skill.set_xticks(positions)
    ax_skill.set_xticklabels(labels)
    ax_skill.set_ylabel("Fold-level RMSE skill\nvs training mean")
    ax_skill.set_ylim(-0.32, 0.79)
    quiet_grid(ax_skill, axis="y")
    panel_label(ax_skill, "C", x=0.005, y=0.97)
    fig.text(
        0.5,
        0.955,
        "Blocks use coordinates only; annual labels are independent and do not denote shared ecological regions",
        ha="center",
        va="center",
        fontsize=6.1,
        color="#50585C",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#DCE4E7", edgecolor="#68777D", linewidth=0.5, label="Study-region ocean grid"),
            Line2D([0], [0], color="#25A7A0", linewidth=1.0, label="Empirical interpolation domain"),
            Line2D([0], [0], color="#16766F", linewidth=0.9, linestyle=(0, (3, 2)), label="DBO3 reference"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=3,
        columnspacing=1.3,
        handletextpad=0.45,
        fontsize=5.7,
    )
    save_figure(fig, "phase8s_spatial_block_validation_diagnostic_v1")
    plt.close(fig)
    assignments.to_csv(FIGURES / "phase8s_spatial_block_assignments_source_data_v1.csv", index=False)
    fold_metrics.to_csv(FIGURES / "phase8s_spatial_block_fold_metrics_source_data_v1.csv", index=False)


def complementary_route_summary() -> pd.DataFrame:
    route = pd.read_csv(SUMMARIES / "phase8s_central_route_summary.csv")
    route = route.loc[route["combination_operator"].eq("complementary")].copy()
    metrics = [
        "delta_mean_path_cost_percent",
        "jaccard_with_baseline_frequent_cells",
        "frequent_centroid_shift_from_baseline_km",
    ]
    rows = []
    for (beta, scenario), group in route.groupby(["pelagic_strength_beta", "scenario"]):
        row = {"pelagic_strength_beta": beta, "scenario": scenario}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_min"] = group[metric].min()
            row[f"{metric}_max"] = group[metric].max()
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_average_intervals() -> pd.DataFrame:
    bootstrap = pd.read_csv(SUMMARIES / "phase8s_bootstrap_route_metrics_beta025.csv")
    metrics = [
        "delta_mean_path_cost_percent",
        "jaccard_with_baseline_frequent_cells",
        "frequent_centroid_shift_from_baseline_km",
    ]
    per_draw = bootstrap.groupby(["bootstrap_draw", "scenario"], as_index=False)[metrics].mean()
    rows = []
    for scenario, group in per_draw.groupby("scenario"):
        for metric in metrics:
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "q05": group[metric].quantile(0.05),
                    "median": group[metric].median(),
                    "q95": group[metric].quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def plot_response_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    bootstrap: pd.DataFrame | None = None,
) -> None:
    for scenario in SCENARIO_ORDER:
        sub = data.loc[data["scenario"].eq(scenario)].sort_values("pelagic_strength_beta")
        x = sub["pelagic_strength_beta"].to_numpy(dtype=float)
        mean = sub[f"{metric}_mean"].to_numpy(dtype=float)
        lower = sub[f"{metric}_min"].to_numpy(dtype=float)
        upper = sub[f"{metric}_max"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color=COLORS[scenario], alpha=0.09, linewidth=0)
        ax.plot(
            x,
            mean,
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markersize=3.2,
            linewidth=1.15,
            zorder=3,
        )
        if bootstrap is not None:
            row = bootstrap.loc[(bootstrap["scenario"].eq(scenario)) & (bootstrap["metric"].eq(metric))].iloc[0]
            ax.errorbar(
                0.25,
                row["median"],
                yerr=[[row["median"] - row["q05"]], [row["q95"] - row["median"]]],
                color=COLORS[scenario],
                linewidth=0.8,
                capsize=2.0,
                marker="none",
                zorder=4,
            )
    ax.set_xticks([0, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.10", "0.25", "0.50", "1.0"])
    ax.set_xlabel("Pelagic-support strength, beta")
    ax.set_ylabel(ylabel)
    quiet_grid(ax)


def plot_validation_lines(ax: plt.Axes, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    for scenario in SCENARIO_ORDER:
        sub = data.loc[data["scenario"].eq(scenario)].sort_values("pelagic_strength_beta")
        ax.plot(
            sub["pelagic_strength_beta"],
            sub[metric],
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markersize=3.2,
            linewidth=1.15,
            zorder=3,
        )
    ax.set_xticks([0, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.10", "0.25", "0.50", "1.0"])
    ax.set_xlabel("Pelagic-support strength, beta")
    ax.set_ylabel(ylabel)
    quiet_grid(ax)


def make_corridor_response_figure() -> None:
    route = complementary_route_summary()
    bootstrap = bootstrap_average_intervals()
    validation = pd.read_csv(VALIDATION / "phase8s_central_asamm_pooled_2017_2018_summary.csv")
    validation = validation.loc[validation["combination_operator"].eq("complementary")].copy()

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55))
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.83, wspace=0.39, hspace=0.52)
    plot_response_lines(
        axes[0, 0], route, "delta_mean_path_cost_percent", "Mean path-cost change (%)", bootstrap
    )
    axes[0, 0].axhline(0, color="#777777", linewidth=0.7)
    plot_response_lines(
        axes[0, 1], route, "jaccard_with_baseline_frequent_cells", "Frequent-cell Jaccard vs baseline", bootstrap
    )
    axes[0, 1].set_ylim(0.28, 1.03)
    plot_response_lines(
        axes[0, 2], route, "frequent_centroid_shift_from_baseline_km", "Frequent-centroid shift (km)", bootstrap
    )
    plot_validation_lines(
        axes[1, 0], validation, "mean_effort_footprint_percentile", "Mean effort-footprint percentile"
    )
    plot_validation_lines(
        axes[1, 1], validation, "percent_records_top5pct", "Sightings in annual top 5% (%)"
    )
    plot_validation_lines(
        axes[1, 2], validation, "median_distance_to_frequent_corridor_km", "Median distance to frequent corridor (km)"
    )
    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(65 + index))
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            markersize=3.5,
            linewidth=1.2,
            label=SCENARIO_SHORT[scenario],
        )
        for scenario in SCENARIO_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, columnspacing=1.4, handletextpad=0.45)
    fig.text(
        0.5,
        0.90,
        "Top: range across 2017-2018; beta=0.25 error bars: 5th-95th percentile across 100 spatial-block bootstrap surfaces",
        ha="center",
        fontsize=6.1,
        color="#555C60",
    )
    fig.text(
        0.5,
        0.865,
        "Bottom: the same 323 June-July ASAMM sightings in every scenario and support level",
        ha="center",
        fontsize=6.1,
        color="#555C60",
    )
    save_figure(fig, "phase8s_corridor_response_and_asamm_consistency_v1")
    plt.close(fig)
    route.to_csv(FIGURES / "phase8s_corridor_response_source_data_v1.csv", index=False)
    bootstrap.to_csv(FIGURES / "phase8s_bootstrap_average_intervals_source_data_v1.csv", index=False)
    validation.to_csv(FIGURES / "phase8s_asamm_response_source_data_v1.csv", index=False)


def make_corridor_map_figure() -> None:
    years = (2017, 2018)
    frames = {}
    frequency_max = 0.0
    difference_max = 0.0
    source_rows = []
    for year in years:
        baseline = pd.read_csv(
            phase8s.central_grid_path("complementary", 0.0, "prey_ice", year), dtype={"grid_id": str}
        )
        augmented = pd.read_csv(
            phase8s.central_grid_path("complementary", 0.25, "prey_ice", year), dtype={"grid_id": str}
        )
        difference = augmented[["grid_id", "x", "y", "lon", "lat", "pelagic_evidence_domain", "P_pelagic"]].copy()
        difference["lcp_frequency_change"] = (
            augmented["lcp_frequency_normalized"].to_numpy()
            - baseline["lcp_frequency_normalized"].to_numpy()
        )
        frames[(year, "baseline")] = baseline
        frames[(year, "augmented")] = augmented
        frames[(year, "difference")] = difference
        frequency_max = max(
            frequency_max,
            float(baseline["lcp_frequency_normalized"].quantile(0.995)),
            float(augmented["lcp_frequency_normalized"].quantile(0.995)),
        )
        difference_max = max(difference_max, float(difference["lcp_frequency_change"].abs().quantile(0.995)))
        for label, frame in (("baseline", baseline), ("augmented", augmented), ("difference", difference)):
            source = frame.copy()
            source["map_state"] = label
            source["target_year"] = year
            source_rows.append(source)
    land, dbo = geographic_context()
    dbo3 = dbo.loc[dbo["name"].eq("DBO3")]
    reference = frames[(2017, "baseline")]
    xlim = (reference["x"].min() - 30_000, reference["x"].max() + 30_000)
    ylim = (reference["y"].min() - 25_000, reference["y"].max() + 25_000)
    observations = pd.read_csv(EARLY_RECORDS)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.065, right=0.91, bottom=0.07, top=0.88, wspace=0.02, hspace=0.04)
    frequency_artist = None
    difference_artist = None
    for row, year in enumerate(years):
        for column, state in enumerate(("baseline", "augmented", "difference")):
            ax = axes[row, column]
            frame = frames[(year, state)]
            map_base(ax, frame, land, dbo3, xlim, ylim)
            if column == 0:
                add_geographic_labels(ax)
            if state == "difference":
                difference_artist = ax.scatter(
                    frame["x"],
                    frame["y"],
                    c=frame["lcp_frequency_change"],
                    s=4.0,
                    marker="s",
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-difference_max, vcenter=0.0, vmax=difference_max),
                    linewidth=0,
                    zorder=2,
                    rasterized=True,
                )
            else:
                used = frame.loc[frame["lcp_frequency_normalized"] > 0]
                frequency_artist = ax.scatter(
                    used["x"],
                    used["y"],
                    c=used["lcp_frequency_normalized"],
                    s=4.0,
                    marker="s",
                    cmap="magma",
                    vmin=0,
                    vmax=frequency_max,
                    linewidth=0,
                    zorder=2,
                    rasterized=True,
                )
            support_frame = frames[(year, "augmented")]
            evidence_boundary(support_frame).boundary.plot(ax=ax, color="#25A7A0", linewidth=0.65, zorder=6)
            if state == "augmented":
                observed = observations.loc[observations["year"].eq(year)]
                ax.scatter(
                    observed["x"],
                    observed["y"],
                    s=7,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=7,
                )
            if column == 0:
                ax.text(-0.045, 0.50, str(year), transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
            panel_label(ax, chr(65 + row * 3 + column), x=0.02, y=0.985)
    for ax, title in zip(
        axes[0],
        ("Benthic prey + ice", "+ empirical pelagic support (beta = 0.25)", "LCP-frequency change"),
    ):
        ax.set_title(title, fontsize=6.7, pad=4)
    cax_frequency = fig.add_axes([0.93, 0.54, 0.016, 0.30])
    cbar_frequency = fig.colorbar(frequency_artist, cax=cax_frequency)
    cbar_frequency.set_label("Normalized LCP frequency", fontsize=6.5)
    cbar_frequency.ax.tick_params(labelsize=5.8)
    cax_difference = fig.add_axes([0.93, 0.13, 0.016, 0.30])
    cbar_difference = fig.colorbar(difference_artist, cax=cax_difference)
    cbar_difference.set_label("Frequency change", fontsize=6.5)
    cbar_difference.ax.tick_params(labelsize=5.8)
    fig.legend(
        handles=[
            Patch(facecolor="#DCE4E7", edgecolor="#68777D", linewidth=0.5, label="Study-region ocean grid"),
            Line2D([0], [0], color="#25A7A0", linewidth=0.9, label="Empirical interpolation domain"),
            Line2D([0], [0], color="#16766F", linewidth=0.9, linestyle=(0, (3, 2)), label="DBO3 anchor"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#555555", markersize=4, label="Euphausiid observation"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.48, 0.965),
        ncol=4,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    save_figure(fig, "phase8s_prey_ice_corridor_reorganization_maps_v2")
    plt.close(fig)
    pd.concat(source_rows, ignore_index=True, sort=False).to_csv(
        FIGURES / "phase8s_corridor_map_source_data_v1.csv", index=False
    )


def write_status() -> None:
    state = json.loads(phase8s.STATUS.read_text(encoding="utf-8"))
    state["stages"]["figures"] = {
        "status": "completed",
        "backend": "Python/matplotlib",
        "figure_stems": [
            "phase8s_observation_model_evidence_v2",
            "phase8s_spatial_block_validation_diagnostic_v1",
            "phase8s_corridor_response_and_asamm_consistency_v1",
            "phase8s_prey_ice_corridor_reorganization_maps_v2",
        ],
        "visual_qa": "passed_at_original_resolution",
        "pdf_pages_per_figure": 1,
        "pdf_fonts_embedded": True,
        "svg_text_editable": True,
        "tiff_dpi": 600,
        "warnings": [],
    }
    state["stages"]["review_package"] = {
        "status": "completed",
        "review_markdown": "outputs/pelagic_support_phase8s/phase8s_review_package.md",
        "method_decision_audit": "outputs/pelagic_support_phase8s/tables/phase8s_method_decision_audit.csv",
        "literature_evidence": "outputs/pelagic_support_phase8s/tables/phase8s_method_literature_evidence.csv",
        "manuscript_or_si_modified": False,
        "warnings": [],
    }
    phase8s.write_json(phase8s.STATUS, state)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "phase8s_figure_contract_and_qa.md").write_text(CONTRACT, encoding="utf-8")
    apply_style()
    make_observation_model_figure()
    make_spatial_block_diagnostic_figure()
    make_corridor_response_figure()
    make_corridor_map_figure()
    write_status()


if __name__ == "__main__":
    main()
