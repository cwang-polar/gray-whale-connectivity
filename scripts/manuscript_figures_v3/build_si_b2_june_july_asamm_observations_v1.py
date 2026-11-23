#!/usr/bin/env python3
"""Build the B2 June-July ASAMM observations and effort SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-b2")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import box


OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "b2_june_july_asamm_observations_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SIGHTINGS = (
    ROOT
    / "outputs"
    / "pelagic_prey_phase8m"
    / "figures"
    / "phase8m_asamm_feeding_evidence_maps_source_data_v1.csv"
)
ANNUAL_SUMMARY = (
    ROOT
    / "outputs"
    / "pelagic_prey_phase8m"
    / "figures"
    / "phase8m_asamm_behavior_summary_source_data_v1.csv"
)
EFFORT = (
    ROOT
    / "data"
    / "processed"
    / "validation"
    / "asamm_phase8i2_effort_covered_C3_cells_by_window.csv"
)
GRID = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
DBO = ROOT / "data" / "processed" / "study_domain" / "dbo_anchor_polygons.gpkg"
LAND = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"
MARITIME_BOUNDARY = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "figure_5"
    / "source_data"
    / "figure5_us_russia_maritime_boundary_candidate_v2.csv"
)

TARGET_YEARS = list(range(2011, 2020))
EXPECTED_CATEGORY_COUNTS = {
    "mud-plume evidence": 313,
    "feeding recorded; no mud cue": 354,
    "other/unknown behavior": 512,
}
BOUNDARY_LON_NORTH_OF_65_5 = -168.97694444
BOUNDARY_SOURCE = (
    "https://www.un.org/Depts/los/LEGISLATIONANDTREATIES/PDFFILES/"
    "TREATIES/USA-RUS1990MB.PDF"
)

PALETTE = {
    "ocean": "#EDF4F6",
    "land": "#D8D3CA",
    "land_edge": "#FFFFFF",
    "effort": "#B9C5CB",
    "mud": "#2166AC",
    "feeding": "#D9822B",
    "other": "#68727A",
    "pelagic": "#6B3D8C",
    "dbo": "#167C72",
    "boundary": "#4F5B62",
    "text": "#252A2E",
    "grid": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "figure.dpi": 150,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.1,
            "ytick.major.size": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    pd.DataFrame,
]:
    sightings = pd.read_csv(SIGHTINGS)
    summary = pd.read_csv(ANNUAL_SUMMARY)
    effort = pd.read_csv(EFFORT, dtype={"grid_id": str})
    effort = effort.loc[
        effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    effort = effort.merge(grid[["grid_id", "lon", "lat"]], on="grid_id", how="left")
    effort = effort.drop_duplicates(["year", "grid_id"]).copy()

    land = gpd.read_file(LAND).to_crs("EPSG:4326")
    map_clip = box(-173.2, 66.25, -156.4, 72.55)
    land = land.loc[land.geometry.intersects(map_clip)].copy()
    land["geometry"] = land.geometry.intersection(map_clip).simplify(
        0.005, preserve_topology=True
    )
    land = land.loc[~land.geometry.is_empty].copy()
    dbo = gpd.read_file(DBO).to_crs("EPSG:4326")
    dbo3 = dbo.loc[dbo["name"].eq("DBO3")].copy()
    boundary = pd.read_csv(MARITIME_BOUNDARY).sort_values("vertex_order")

    if len(sightings) != 1179:
        raise ValueError(f"Expected 1,179 sightings, found {len(sightings):,}")
    observed_counts = sightings["feeding_evidence_class"].value_counts().to_dict()
    if observed_counts != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"Unexpected behavior counts: {observed_counts}")
    if sorted(sightings["year"].unique().tolist()) != TARGET_YEARS:
        raise ValueError("Sighting years do not match 2011-2019")
    if int(sightings["explicit_possible_pelagic_cue"].sum()) != 1:
        raise ValueError("Expected exactly one observer note describing possible skim feeding")
    if len(dbo3) != 1:
        raise ValueError("Expected exactly one DBO3 anchor polygon")
    if effort[["lon", "lat"]].isna().any().any():
        raise ValueError("Effort cells failed to join the analysis grid")
    if not (sightings["lon"] > BOUNDARY_LON_NORTH_OF_65_5).all():
        raise ValueError("At least one raw sighting lies west of the treaty boundary")

    annual_counts = sightings.groupby("year").size()
    expected_annual = summary.set_index("year")["record_count"].astype(int)
    if not annual_counts.equals(expected_annual):
        raise ValueError("Annual sighting counts differ from the Phase 8M summary")
    effort_counts = effort.groupby("year")["grid_id"].nunique()
    expected_effort = summary.set_index("year")["strict_effort_cell_count"].astype(int)
    if not effort_counts.equals(expected_effort):
        raise ValueError("Annual effort-cell counts differ from the Phase 8M summary")

    return sightings, summary, effort, land, dbo3, boundary


def degree_formatter(value: float, _: int) -> str:
    suffix = "W" if value < 0 else "E"
    return f"{abs(value):.0f}\N{DEGREE SIGN}{suffix}"


def latitude_formatter(value: float, _: int) -> str:
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.0f}\N{DEGREE SIGN}{suffix}"


def outlined_text(ax: plt.Axes, x: float, y: float, label: str, **kwargs: object) -> None:
    item = ax.text(x, y, label, zorder=9, **kwargs)
    item.set_path_effects([patheffects.withStroke(linewidth=1.3, foreground="white")])


def draw_context_labels(ax: plt.Axes) -> None:
    outlined_text(
        ax,
        -171.2,
        71.65,
        "Russia",
        fontsize=5.1,
        color="#504C47",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -159.3,
        69.25,
        "Alaska",
        fontsize=5.1,
        color="#504C47",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -167.45,
        68.50,
        "Point Hope",
        fontsize=4.7,
        color="#504C47",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -171.55,
        66.74,
        "Bering Strait",
        fontsize=4.6,
        color="#504C47",
        fontstyle="italic",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -169.60,
        68.25,
        "DBO3",
        fontsize=4.9,
        fontweight="bold",
        color=PALETTE["dbo"],
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -169.26,
        70.05,
        "U.S.-Russia maritime boundary",
        fontsize=4.2,
        color=PALETTE["boundary"],
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )


def draw_panel(
    ax: plt.Axes,
    year: int,
    sightings: pd.DataFrame,
    summary: pd.DataFrame,
    effort: pd.DataFrame,
    land: gpd.GeoDataFrame,
    dbo3: gpd.GeoDataFrame,
    boundary: pd.DataFrame,
) -> None:
    xmin, xmax, ymin, ymax = -173.1, -156.5, 66.35, 72.45
    ax.set_facecolor(PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax,
        color=PALETTE["land"],
        edgecolor=PALETTE["land_edge"],
        linewidth=0.22,
        zorder=1,
    )

    annual_effort = effort.loc[effort["year"].eq(year)]
    ax.scatter(
        annual_effort["lon"],
        annual_effort["lat"],
        s=3.1,
        marker="s",
        facecolor=PALETTE["effort"],
        edgecolor="none",
        alpha=0.46,
        zorder=2,
        rasterized=True,
    )
    dbo3.boundary.plot(
        ax=ax,
        color=PALETTE["dbo"],
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=3,
    )
    visible_boundary = boundary.loc[
        boundary["lat"].between(ymin, ymax) & boundary["lon"].between(xmin, xmax)
    ]
    ax.plot(
        visible_boundary["lon"],
        visible_boundary["lat"],
        color=PALETTE["boundary"],
        linewidth=0.75,
        linestyle=(0, (4, 2)),
        zorder=4,
    )

    annual = sightings.loc[sightings["year"].eq(year)]
    layers = [
        ("other/unknown behavior", PALETTE["other"], "o", 8.5, 0.72),
        ("feeding recorded; no mud cue", PALETTE["feeding"], "^", 14.0, 0.90),
        ("mud-plume evidence", PALETTE["mud"], "o", 13.0, 0.90),
    ]
    for category, color, marker, size, alpha in layers:
        subset = annual.loc[annual["feeding_evidence_class"].eq(category)]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.28,
            alpha=alpha,
            zorder=6,
        )

    cue = annual.loc[annual["explicit_possible_pelagic_cue"]]
    if len(cue):
        ax.scatter(
            cue["lon"],
            cue["lat"],
            s=34,
            marker="*",
            facecolor=PALETTE["pelagic"],
            edgecolor="white",
            linewidth=0.42,
            zorder=8,
        )

    n = int(summary.loc[summary["year"].eq(year), "record_count"].iloc[0])
    ax.set_title(f"{year} (n = {n})", loc="left", fontsize=7.0, fontweight="bold", pad=2.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / np.cos(np.deg2rad((ymin + ymax) / 2)), adjustable="box")
    ax.set_xticks([-172, -168, -164, -160])
    ax.set_yticks([67, 69, 71])
    ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_formatter))
    ax.grid(color=PALETTE["grid"], linewidth=0.32, alpha=0.90, zorder=0)
    ax.tick_params(labelsize=5.4, pad=1.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#70777B")


def save_source_data(
    sightings: pd.DataFrame,
    summary: pd.DataFrame,
    effort: pd.DataFrame,
    boundary: pd.DataFrame,
) -> None:
    sighting_columns = [
        "record_id",
        "year",
        "month",
        "lat",
        "lon",
        "feeding_evidence_class",
        "explicit_possible_pelagic_cue",
        "inside_DBO3",
    ]
    sightings[sighting_columns].to_csv(
        SOURCE_OUT / "b2_june_july_asamm_sightings_source_data_v1.csv", index=False
    )
    effort[["year", "grid_id", "lon", "lat"]].to_csv(
        SOURCE_OUT / "b2_june_july_asamm_effort_cells_source_data_v1.csv", index=False
    )
    summary.to_csv(SOURCE_OUT / "b2_june_july_asamm_annual_summary_v1.csv", index=False)
    boundary[["vertex_order", "lon", "lat", "source"]].to_csv(
        SOURCE_OUT / "b2_us_russia_maritime_boundary_source_data_v1.csv", index=False
    )

    inputs = [SIGHTINGS, ANNUAL_SUMMARY, EFFORT, GRID, DBO, LAND, MARITIME_BOUNDARY]
    rows = [
        {
            "relative_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in inputs
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "input_provenance_manifest_v1.csv", index=False)


def build_figure(*, prepared=None) -> list[Path]:
    configure_style()
    for directory in (CANDIDATE_OUT, SOURCE_OUT, QA_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    if prepared is None:
        sightings, summary, effort, land, dbo3, boundary = load_inputs()
        save_source_data(sightings, summary, effort, boundary)
    else:
        sightings, summary, effort, land, dbo3, boundary = prepared

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.05), facecolor="white")
    fig.subplots_adjust(left=0.068, right=0.992, top=0.965, bottom=0.108, wspace=0.075, hspace=0.115)

    for index, (ax, year) in enumerate(zip(axes.flat, TARGET_YEARS)):
        draw_panel(ax, year, sightings, summary, effort, land, dbo3, boundary)
        row, col = divmod(index, 3)
        if row < 2:
            ax.tick_params(labelbottom=False)
        if col > 0:
            ax.tick_params(labelleft=False)
        if index == 0:
            draw_context_labels(ax)

    effort_handle = Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=PALETTE["effort"],
            markeredgecolor="none",
            markersize=5.1,
            alpha=0.70,
            label="Effort-covered 10-km cell",
        )
    mud_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PALETTE["mud"],
            markeredgecolor="white",
            markersize=5.4,
            label="Mud-plume evidence",
        )
    feeding_handle = Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor=PALETTE["feeding"],
            markeredgecolor="white",
            markersize=5.5,
            label="Feeding recorded; no mud cue",
        )
    other_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PALETTE["other"],
            markeredgecolor="white",
            markersize=5.0,
            label="Other or unknown behavior",
        )
    pelagic_handle = Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=PALETTE["pelagic"],
            markeredgecolor="white",
            markersize=6.8,
            label="Observer note: possible skim feeding (1 record)",
        )
    # Matplotlib fills multi-row legends by column. This ordering produces a
    # first row of effort, mud plume, and feeding-without-mud categories.
    handles = [
        effort_handle,
        other_handle,
        mud_handle,
        pelagic_handle,
        feeding_handle,
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.022),
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handletextpad=0.45,
        columnspacing=1.25,
        labelspacing=0.65,
    )

    stem = CANDIDATE_OUT / "gray_whale_si_b2_june_july_asamm_observations_candidate_v1"
    outputs = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], format="pdf")
    fig.savefig(outputs[1], format="svg")
    fig.savefig(outputs[2], format="png", dpi=300)
    fig.savefig(outputs[3], format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    with Image.open(outputs[2]) as image:
        extrema = image.convert("L").getextrema()
        png_size = list(image.size)
    with Image.open(outputs[3]) as image:
        tiff_size = list(image.size)

    qa = {
        "backend": "Python/matplotlib in mypy10",
        "sighting_records": int(len(sightings)),
        "annual_record_counts": {
            str(int(row.year)): int(row.record_count) for row in summary.itertuples(index=False)
        },
        "behavior_category_counts": EXPECTED_CATEGORY_COUNTS,
        "observer_note_possible_skim_feeding_records": 1,
        "raw_sightings_west_of_treaty_boundary": int(
            (sightings["lon"] <= BOUNDARY_LON_NORTH_OF_65_5).sum()
        ),
        "effort_cells": int(len(effort)),
        "annual_effort_cell_counts": {
            str(int(year)): int(count)
            for year, count in effort.groupby("year")["grid_id"].nunique().items()
        },
        "maritime_boundary_source": BOUNDARY_SOURCE,
        "png_dimensions_px": png_size,
        "tiff_dimensions_px": tiff_size,
        "png_grayscale_extrema": list(extrema),
        "nonblank": extrema[0] < extrema[1],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "source_figure_preserved": True,
        "manuscript_or_word_modified": False,
    }
    (QA_OUT / "b2_june_july_asamm_qa_v1.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    for output in build_figure():
        print(output)
