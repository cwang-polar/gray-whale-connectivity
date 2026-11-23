#!/usr/bin/env python3
"""Build the B3 August-October ASAMM observations and effort SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-b3")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
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
    / "b3_august_october_asamm_observations_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SIGHTINGS = (
    ROOT
    / "outputs"
    / "southward_connectivity_phase8q"
    / "validation"
    / "asamm_aug_oct_sightings_study_region.csv"
)
EFFORT = (
    ROOT
    / "outputs"
    / "southward_connectivity_phase8q"
    / "validation"
    / "asamm_aug_oct_strict_effort_cells.csv"
)
GRID = (
    ROOT
    / "data"
    / "processed"
    / "study_domain"
    / "ocean_mask_grid_10km_C3_recurring_support_v2.csv"
)
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
EXPECTED_ANNUAL_COUNTS = {
    2011: 111,
    2012: 135,
    2013: 99,
    2014: 367,
    2015: 184,
    2016: 288,
    2017: 243,
    2018: 202,
    2019: 214,
}
EXPECTED_EFFORT_COUNTS = {
    2011: 663,
    2012: 739,
    2013: 739,
    2014: 789,
    2015: 833,
    2016: 783,
    2017: 790,
    2018: 770,
    2019: 644,
}
BOUNDARY_LON_NORTH_OF_65_5 = -168.97694444
BOUNDARY_SOURCE = (
    "https://www.un.org/Depts/los/LEGISLATIONANDTREATIES/PDFFILES/"
    "TREATIES/USA-RUS1990MB.PDF"
)

MAP_EXTENT = (-173.1, -156.5, 64.60, 73.05)

PALETTE = {
    "ocean": "#EDF4F6",
    "land": "#D8D3CA",
    "land_edge": "#FFFFFF",
    "effort": "#B9C5CB",
    "august": "#2166AC",
    "september": "#D9822B",
    "october": "#B65C86",
    "boundary": "#4F5B62",
    "text": "#252A2E",
    "label": "#504C47",
    "gridline": "#FFFFFF",
}

MONTH_STYLE = {
    8: {"label": "August", "color": PALETTE["august"], "marker": "o", "size": 10.5},
    9: {"label": "September", "color": PALETTE["september"], "marker": "^", "size": 13.5},
    10: {"label": "October", "color": PALETTE["october"], "marker": "D", "size": 10.5},
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
    pd.DataFrame,
]:
    sightings = pd.read_csv(SIGHTINGS)
    effort = pd.read_csv(EFFORT, dtype={"grid_id": str})
    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    effort = effort.merge(
        grid[["grid_id", "lon", "lat"]],
        on="grid_id",
        how="left",
        validate="many_to_one",
    )
    effort = effort.drop_duplicates(["year", "grid_id"]).copy()

    xmin, xmax, ymin, ymax = MAP_EXTENT
    land = gpd.read_file(LAND).to_crs("EPSG:4326")
    map_clip = box(xmin, ymin, xmax, ymax)
    land = land.loc[land.geometry.intersects(map_clip)].copy()
    land["geometry"] = land.geometry.intersection(map_clip).simplify(
        0.005, preserve_topology=True
    )
    land = land.loc[~land.geometry.is_empty].copy()
    boundary = pd.read_csv(MARITIME_BOUNDARY).sort_values("vertex_order")

    if len(sightings) != 1843:
        raise ValueError(f"Expected 1,843 sightings, found {len(sightings):,}")
    if sorted(sightings["year"].unique().tolist()) != TARGET_YEARS:
        raise ValueError("Sighting years do not match 2011-2019")
    if set(sightings["month"].unique()) != {8, 9, 10}:
        raise ValueError("Sighting months do not match August-October")
    if effort[["lon", "lat"]].isna().any().any():
        raise ValueError("Effort cells failed to join the analysis grid")

    annual_counts = sightings.groupby("year").size().to_dict()
    if annual_counts != EXPECTED_ANNUAL_COUNTS:
        raise ValueError(f"Unexpected annual sighting counts: {annual_counts}")
    effort_counts = effort.groupby("year")["grid_id"].nunique().to_dict()
    if effort_counts != EXPECTED_EFFORT_COUNTS:
        raise ValueError(f"Unexpected annual effort-cell counts: {effort_counts}")

    xmin, xmax, ymin, ymax = MAP_EXTENT
    outside_map = ~(
        sightings["lon"].between(xmin, xmax)
        & sightings["lat"].between(ymin, ymax)
    )
    if outside_map.any():
        raise ValueError(f"{int(outside_map.sum())} sightings fall outside the map extent")
    effort_outside = ~(
        effort["lon"].between(xmin, xmax) & effort["lat"].between(ymin, ymax)
    )
    if effort_outside.any():
        raise ValueError(f"{int(effort_outside.sum())} effort cells fall outside the map extent")
    if not (sightings["lon"] > BOUNDARY_LON_NORTH_OF_65_5).all():
        raise ValueError("At least one raw sighting lies west of the treaty boundary")

    annual_summary = (
        sightings.groupby("year")
        .agg(retained_sighting_records=("record_id", "size"))
        .reset_index()
    )
    for month, label in ((8, "august"), (9, "september"), (10, "october")):
        counts = sightings.loc[sightings["month"].eq(month)].groupby("year").size()
        annual_summary[f"{label}_records"] = (
            annual_summary["year"].map(counts).fillna(0).astype(int)
        )
    annual_summary["strict_effort_cells"] = annual_summary["year"].map(
        effort.groupby("year")["grid_id"].nunique()
    )
    return sightings, effort, annual_summary, land, boundary


def degree_formatter(value: float, _: int) -> str:
    suffix = "W" if value < 0 else "E"
    return f"{abs(value):.0f}\N{DEGREE SIGN}{suffix}"


def latitude_formatter(value: float, _: int) -> str:
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.0f}\N{DEGREE SIGN}{suffix}"


def outlined_text(ax: plt.Axes, x: float, y: float, label: str, **kwargs: object) -> None:
    item = ax.text(x, y, label, zorder=9, **kwargs)
    item.set_path_effects([patheffects.withStroke(linewidth=1.35, foreground="white")])


def draw_context_labels(ax: plt.Axes) -> None:
    outlined_text(
        ax,
        -171.65,
        65.25,
        "Russia",
        fontsize=4.9,
        color=PALETTE["label"],
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -158.85,
        67.45,
        "Alaska",
        fontsize=5.0,
        color=PALETTE["label"],
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -164.20,
        72.15,
        "Chukchi Sea",
        fontsize=4.8,
        color=PALETTE["label"],
        fontstyle="italic",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -171.35,
        66.05,
        "Bering Strait",
        fontsize=4.5,
        color=PALETTE["label"],
        fontstyle="italic",
        rotation=72,
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -170.25,
        64.86,
        "Chirikov Basin",
        fontsize=4.5,
        color=PALETTE["label"],
        fontstyle="italic",
        ha="center",
        va="center",
    )
    outlined_text(
        ax,
        -169.28,
        69.35,
        "U.S.-Russia maritime boundary",
        fontsize=4.1,
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
    effort: pd.DataFrame,
    land: gpd.GeoDataFrame,
    boundary: pd.DataFrame,
) -> None:
    xmin, xmax, ymin, ymax = MAP_EXTENT
    ax.set_facecolor(PALETTE["ocean"])
    land.plot(
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
        s=3.0,
        marker="s",
        facecolor=PALETTE["effort"],
        edgecolor="none",
        alpha=0.50,
        zorder=2,
        rasterized=True,
    )

    visible_boundary = boundary.loc[
        boundary["lat"].between(ymin, ymax) & boundary["lon"].between(xmin, xmax)
    ]
    ax.plot(
        visible_boundary["lon"],
        visible_boundary["lat"],
        color=PALETTE["boundary"],
        linewidth=0.78,
        linestyle=(0, (4, 2)),
        zorder=4,
    )

    annual = sightings.loc[sightings["year"].eq(year)]
    for month, style in MONTH_STYLE.items():
        subset = annual.loc[annual["month"].eq(month)]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=style["size"],
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.28,
            alpha=0.90,
            zorder=6,
        )

    ax.set_title(
        f"{year} (n = {len(annual):,})",
        loc="left",
        fontsize=7.0,
        fontweight="bold",
        pad=2.0,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / np.cos(np.deg2rad((ymin + ymax) / 2)), adjustable="box")
    ax.set_xticks([-172, -168, -164, -160])
    ax.set_yticks([65, 67, 69, 71, 73])
    ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_formatter))
    ax.grid(color=PALETTE["gridline"], linewidth=0.32, alpha=0.90, zorder=0)
    ax.tick_params(labelsize=5.2, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#70777B")


def save_source_data(
    sightings: pd.DataFrame,
    effort: pd.DataFrame,
    annual_summary: pd.DataFrame,
    boundary: pd.DataFrame,
) -> None:
    sighting_columns = [
        "record_id",
        "year",
        "month",
        "lat",
        "lon",
        "nearest_grid_id",
        "nearest_grid_lon",
        "nearest_grid_lat",
        "distance_to_nearest_grid_cell_km",
        "count_or_best_count",
    ]
    sightings[sighting_columns].to_csv(
        SOURCE_OUT / "b3_aug_oct_asamm_sightings_source_data_v1.csv", index=False
    )
    effort_columns = [
        "year",
        "grid_id",
        "lon",
        "lat",
        "strict_effort_event_count",
        "strict_effort_unique_flights",
        "months_covered",
    ]
    effort[effort_columns].to_csv(
        SOURCE_OUT / "b3_aug_oct_asamm_effort_cells_source_data_v1.csv", index=False
    )
    annual_summary.to_csv(
        SOURCE_OUT / "b3_aug_oct_asamm_annual_summary_v1.csv", index=False
    )
    boundary[["vertex_order", "lon", "lat", "source"]].to_csv(
        SOURCE_OUT / "b3_us_russia_maritime_boundary_source_data_v1.csv", index=False
    )

    rows = [
        {
            "relative_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (SIGHTINGS, EFFORT, GRID, LAND, MARITIME_BOUNDARY)
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "input_provenance_manifest_v1.csv", index=False)


def write_figure_contract() -> None:
    contract = """# Figure contract: B3 annual August-October ASAMM observations

## Core conclusion

The late-season corridor comparison is supported by 1,843 retained ASAMM sighting records collected within year-specific August-October effort footprints; the spatial footprint and monthly composition vary among years, and all retained raw coordinates lie east of the U.S.-Russia maritime boundary.

## Evidence chain

- Nine annual maps show every retained raw sighting coordinate from 2011-2019.
- Pale 10-km cells show the eligible annual strict-effort background used for model-observation comparisons.
- Month-specific colors and symbols distinguish August, September and October without implying movement direction.
- A treaty-based boundary resolves whether plotted records occur in U.S. or Russian waters.

## Archetype and role

- Archetype: quantitative small-multiple map grid.
- Manuscript role: transparent input and survey-footprint audit for the late-season sensitivity analysis.

## Interpretation

- Annual n is defined as retained sighting records, not whales or abundance.
- Unflown cells are not interpreted as absences.
- Observation month is not interpreted as travel direction.
- Raw sightings and grid-based effort are visually and verbally distinguished.

## Export contract

- Python/matplotlib only, using the mypy10 environment.
- Fixed 7.2 x 8.95 inch canvas.
- PDF and SVG with editable text; 300-dpi PNG and 600-dpi LZW-compressed TIFF.
- Candidate outputs remain isolated from the Phase 8Q source figure and manuscript files.
"""
    (OUT_ROOT / "figure_contract.md").write_text(contract, encoding="utf-8")


def build_figure(*, prepared=None) -> list[Path]:
    configure_style()
    for directory in (CANDIDATE_OUT, SOURCE_OUT, QA_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    if prepared is None:
        sightings, effort, annual_summary, land, boundary = load_inputs()
        save_source_data(sightings, effort, annual_summary, boundary)
        write_figure_contract()
    else:
        sightings, effort, annual_summary, land, boundary = prepared

    fig = plt.figure(figsize=(7.2, 8.95), facecolor="white")
    panel_width_inches = 1.90
    panel_height_inches = 2.61
    column_gap_inches = 0.28
    row_gap_inches = 0.14
    left_margin_inches = 0.47
    bottom_margin_inches = 0.58

    axes: list[plt.Axes] = []
    for row in range(3):
        for col in range(3):
            left = (left_margin_inches + col * (panel_width_inches + column_gap_inches)) / 7.2
            bottom = (
                bottom_margin_inches
                + (2 - row) * (panel_height_inches + row_gap_inches)
            ) / 8.95
            axes.append(
                fig.add_axes(
                    [
                        left,
                        bottom,
                        panel_width_inches / 7.2,
                        panel_height_inches / 8.95,
                    ]
                )
            )

    for index, (ax, year) in enumerate(zip(axes, TARGET_YEARS)):
        draw_panel(ax, year, sightings, effort, land, boundary)
        row, col = divmod(index, 3)
        if row < 2:
            ax.tick_params(labelbottom=False)
        if col > 0:
            ax.tick_params(labelleft=False)
        if index == 0:
            draw_context_labels(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=PALETTE["effort"],
            markeredgecolor="none",
            markersize=5.1,
            alpha=0.75,
            label="Effort-covered 10-km cell",
        )
    ]
    for style in MONTH_STYLE.values():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=5.3,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        handletextpad=0.42,
        columnspacing=1.35,
    )

    stem = CANDIDATE_OUT / "gray_whale_si_b3_aug_oct_asamm_observations_candidate_v1"
    outputs = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], format="pdf")
    fig.savefig(outputs[1], format="svg")
    fig.savefig(outputs[2], format="png", dpi=300)
    fig.savefig(
        outputs[3],
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
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
            str(year): int(count) for year, count in EXPECTED_ANNUAL_COUNTS.items()
        },
        "monthly_record_counts": {
            str(int(row.year)): {
                "August": int(row.august_records),
                "September": int(row.september_records),
                "October": int(row.october_records),
            }
            for row in annual_summary.itertuples(index=False)
        },
        "annual_effort_cell_counts": {
            str(year): int(count) for year, count in EXPECTED_EFFORT_COUNTS.items()
        },
        "raw_sightings_west_of_treaty_boundary": int(
            (sightings["lon"] <= BOUNDARY_LON_NORTH_OF_65_5).sum()
        ),
        "maritime_boundary_source": BOUNDARY_SOURCE,
        "map_extent_lon_lat": list(MAP_EXTENT),
        "png_dimensions_px": png_size,
        "tiff_dimensions_px": tiff_size,
        "png_grayscale_extrema": list(extrema),
        "nonblank": extrema[0] < extrema[1],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "source_figure_preserved": True,
        "manuscript_or_word_modified": False,
    }
    (QA_OUT / "b3_aug_oct_asamm_qa_v1.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    for output in build_figure():
        print(output)
