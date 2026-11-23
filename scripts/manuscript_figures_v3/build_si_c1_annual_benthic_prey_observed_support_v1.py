#!/usr/bin/env python3
"""Build the C1 annual benthic-prey surfaces and observed-support SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-c1")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "manuscript_figures"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
import build_phase8b_brian_amphipod_prey_surfaces_c3_v2 as prey_plot
import build_figure1_cee_v0 as figure1_base


OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "c1_annual_benthic_prey_observed_support_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

PHASE8L_SOURCE = (
    ROOT
    / "outputs"
    / "prey_reconciliation_phase8l"
    / "tables"
    / "phase8l_annual_surface_map_source_data.csv"
)
LAND_SHP = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

TARGET_YEARS = list(range(2011, 2020))
UTILITY_COLUMN = "U_amphipod_prey_global_q05_q95"
OBSERVED_COLUMN = "observed_window_flag"
INTERPOLATED_COLUMN = "interpolated_flag"

DBO_LABEL_POSITIONS = {
    "DBO1": (-174.6, 62.8),
    "DBO2": (-169.2, 64.7),
    "DBO3": (-168.9, 67.7),
    "DBO4": (-162.5, 71.25),
    "DBO5": (-157.35, 71.45),
}
DBO_COLOR = figure1_base.PALETTE["dbo"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_source(source_path=None) -> pd.DataFrame:
    frame = pd.read_csv(PHASE8L_SOURCE if source_path is None else source_path, dtype={"grid_id": str})
    required = {
        "grid_id",
        "lon",
        "lat",
        "target_year",
        UTILITY_COLUMN,
        OBSERVED_COLUMN,
        INTERPOLATED_COLUMN,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Phase 8L source is missing columns: {sorted(missing)}")

    frame[OBSERVED_COLUMN] = parse_bool(frame[OBSERVED_COLUMN])
    frame[INTERPOLATED_COLUMN] = parse_bool(frame[INTERPOLATED_COLUMN])
    if sorted(frame["target_year"].unique().tolist()) != TARGET_YEARS:
        raise ValueError("Phase 8L source does not contain exactly 2011-2019")
    counts = frame.groupby("target_year")["grid_id"].nunique()
    if not counts.eq(2839).all():
        raise ValueError(f"Unexpected annual grid-cell counts: {counts.to_dict()}")
    if frame.duplicated(["target_year", "grid_id"]).any():
        raise ValueError("Duplicate target-year/grid-cell rows in Phase 8L source")
    if not (frame[OBSERVED_COLUMN] ^ frame[INTERPOLATED_COLUMN]).all():
        raise ValueError("Observed and interpolated flags are not mutually exclusive")
    utility = pd.to_numeric(frame[UTILITY_COLUMN], errors="coerce")
    if utility.isna().any() or not utility.between(0, 1).all():
        raise ValueError("Scaled benthic-prey utility is missing or outside [0, 1]")

    x, y = prey_plot.project_points(frame["lon"], frame["lat"])
    frame["x"] = x
    frame["y"] = y
    return frame


def draw_dbo_polygons(ax: plt.Axes, dbo, add_labels: bool) -> None:
    for row in dbo.itertuples(index=False):
        label = str(row.name)
        projected = prey_plot.to_work_geom(row.geometry)
        x, y = projected.exterior.xy
        ax.plot(
            x,
            y,
            color=DBO_COLOR,
            linewidth=0.72,
            linestyle=(0, (4.0, 2.2)),
            zorder=6,
        )
        if add_labels:
            label_lon, label_lat = DBO_LABEL_POSITIONS[label]
            label_x, label_y = prey_plot.project_points([label_lon], [label_lat])
            text = ax.text(
                float(label_x[0]),
                float(label_y[0]),
                label,
                color=DBO_COLOR,
                fontsize=5.5,
                ha="center",
                va="center",
                zorder=8,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.0, foreground="white")]
            )


def style_map_axis(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    year: int,
    panel: str,
) -> None:
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_facecolor("#EAF3F5")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(str(year), pad=2.0)
    ax.text(
        0.018,
        0.982,
        panel,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.4},
    )


def write_supporting_records(frame: pd.DataFrame, dbo) -> None:
    source_copy = SOURCE_OUT / "c1_annual_benthic_prey_observed_support_source_data_v1.csv"
    frame.drop(columns=["x", "y"]).to_csv(source_copy, index=False)

    summary = (
        frame.groupby("target_year", as_index=False)
        .agg(
            ocean_grid_cells=("grid_id", "nunique"),
            directly_observed_cells=(OBSERVED_COLUMN, "sum"),
            interpolated_cells=(INTERPOLATED_COLUMN, "sum"),
            minimum_scaled_utility=(UTILITY_COLUMN, "min"),
            maximum_scaled_utility=(UTILITY_COLUMN, "max"),
        )
    )
    summary.to_csv(SOURCE_OUT / "c1_annual_support_counts_v1.csv", index=False)

    dbo_rows = []
    for row in dbo.itertuples(index=False):
        for vertex, (lon, lat) in enumerate(row.geometry.exterior.coords[:-1], start=1):
            dbo_rows.append(
                {
                    "reference_area": row.name,
                    "vertex": vertex,
                    "longitude": lon,
                    "latitude": lat,
                }
            )
    pd.DataFrame(dbo_rows).to_csv(
        SOURCE_OUT / "c1_dbo_reference_polygons_figure1_matched_v1.csv", index=False
    )

    provenance = pd.DataFrame(
        [
            {
                "relative_path": str(PHASE8L_SOURCE.relative_to(ROOT)),
                "size_bytes": PHASE8L_SOURCE.stat().st_size,
                "sha256": sha256(PHASE8L_SOURCE),
                "role": "audited annual prey-surface and observation-support source",
            }
        ]
    )
    provenance.to_csv(OUT_ROOT / "input_provenance_manifest_v1.csv", index=False)


def write_figure_contract() -> None:
    text = """# Figure contract: C1 annual benthic-prey surfaces and observed support

## Core conclusion

The nine primary annual prey landscapes are spatially explicit modeled surfaces whose directly observed support varies among trailing five-year windows and is distinguishable from interpolated cells.

## Evidence chain

- Panels A-I show all target years from 2011 through 2019 on a common 0-1 utility scale.
- Open circles identify 10-km grid cells with direct observations in the corresponding trailing five-year window.
- Unmarked colored cells are interpolated values in the established study-region surface.
- DBO1-DBO5 reference areas provide geographic context and match the geometry shown in Figure 1.

## Archetype and role

- Archetype: quantitative map grid.
- Manuscript role: input-evidence transparency for the primary benthic-prey landscapes.

## Interpretation

- Target-year surfaces use observations from t-4 through t, not observations from future years.
- Observed-cell symbols indicate data support and do not encode prey magnitude or sampling effort.
- Colors show globally scaled modeled utility, not raw energy density in kJ m-2.
- DBO outlines are fixed geographic reference areas and do not define model boundaries.

## Export contract

- Python/matplotlib only, using the mypy10 environment.
- PDF and SVG with editable text; 300-dpi PNG and 600-dpi LZW-compressed TIFF.
- New outputs remain isolated from Phase 8L figures and all manuscript files.
"""
    (OUT_ROOT / "figure_contract.md").write_text(text, encoding="utf-8")


def build_figure(prepared=None) -> list[Path]:
    configure_style()
    for directory in (CANDIDATE_OUT, SOURCE_OUT, QA_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    if prepared is None:
        frame = load_source()
        dbo = figure1_base.make_dbo_reference_polygons()
        write_supporting_records(frame, dbo)
        write_figure_contract()
    else:
        frame, dbo = prepared

    projected_dbo = [prey_plot.to_work_geom(geometry) for geometry in dbo.geometry]
    dbo_min_x = min(geometry.bounds[0] for geometry in projected_dbo)
    dbo_min_y = min(geometry.bounds[1] for geometry in projected_dbo)
    dbo_max_x = max(geometry.bounds[2] for geometry in projected_dbo)
    dbo_max_y = max(geometry.bounds[3] for geometry in projected_dbo)
    minimum_x = min(float(frame["x"].min()), dbo_min_x)
    minimum_y = min(float(frame["y"].min()), dbo_min_y)
    maximum_x = max(float(frame["x"].max()), dbo_max_x)
    maximum_y = max(float(frame["y"].max()), dbo_max_y)
    x_span = maximum_x - minimum_x
    y_span = maximum_y - minimum_y
    bounds = (
        minimum_x - 0.018 * x_span,
        minimum_y - 0.018 * y_span,
        maximum_x + 0.018 * x_span,
        maximum_y + 0.018 * y_span,
    )
    land = prey_plot.read_polygon_shapefile(LAND_SHP)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 8.65), facecolor="white")
    fig.subplots_adjust(
        left=0.025,
        right=0.895,
        bottom=0.082,
        top=0.975,
        wspace=0.055,
        hspace=0.075,
    )

    image = None
    for index, (ax, year) in enumerate(zip(axes.flat, TARGET_YEARS)):
        annual = frame.loc[frame["target_year"].eq(year)].copy()
        style_map_axis(ax, bounds, year, chr(ord("A") + index))
        image = ax.scatter(
            annual["x"],
            annual["y"],
            c=annual[UTILITY_COLUMN],
            cmap="viridis",
            vmin=0,
            vmax=1,
            marker="s",
            s=4.8,
            linewidth=0,
            rasterized=False,
            zorder=2,
        )
        prey_plot.draw_land(ax, land)

        observed = annual.loc[annual[OBSERVED_COLUMN]]
        ax.scatter(
            observed["x"],
            observed["y"],
            s=8.2,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=1.05,
            zorder=4,
        )
        ax.scatter(
            observed["x"],
            observed["y"],
            s=8.2,
            marker="o",
            facecolors="none",
            edgecolors="#151719",
            linewidths=0.38,
            zorder=5,
        )
        draw_dbo_polygons(ax, dbo, add_labels=index == 0)

    if image is None:
        raise RuntimeError("No annual map layer was drawn")
    colorbar_axis = fig.add_axes([0.918, 0.345, 0.017, 0.34])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_ticks([0, 0.25, 0.50, 0.75, 1.00])
    colorbar.set_label("Scaled benthic-prey utility", labelpad=5)
    colorbar.outline.set_linewidth(0.65)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#151719",
            markeredgewidth=0.7,
            markersize=4.5,
            label="Directly observed grid cell in trailing window",
        ),
        Line2D(
            [0],
            [0],
            color=DBO_COLOR,
            linewidth=0.7,
            linestyle=(0, (4.0, 2.2)),
            label="DBO reference area",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.46, 0.024),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.5,
        fontsize=6.2,
        frameon=False,
    )

    stem = CANDIDATE_OUT / "gray_whale_si_c1_annual_benthic_prey_observed_support_candidate_v1"
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

    with Image.open(outputs[2]) as png:
        png_size = list(png.size)
        extrema = png.convert("L").getextrema()
    with Image.open(outputs[3]) as tiff:
        tiff_size = list(tiff.size)

    annual_counts = (
        frame.groupby("target_year")[OBSERVED_COLUMN].sum().astype(int).to_dict()
    )
    qa = {
        "backend": "Python/matplotlib in mypy10",
        "target_years": TARGET_YEARS,
        "annual_ocean_grid_cell_count": 2839,
        "annual_directly_observed_grid_cell_counts": annual_counts,
        "dbo_reference_areas": dbo["name"].tolist(),
        "dbo_geometry_source": "same DBO1-DBO5 reference polygons used in Figure 1",
        "trailing_window_rule": "target year t uses observations from t-4 through t",
        "utility_scale": [0.0, 1.0],
        "png_dimensions_px": png_size,
        "tiff_dimensions_px": tiff_size,
        "png_grayscale_extrema": list(extrema),
        "nonblank": extrema[0] < extrema[1],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "source_phase8l_figure_preserved": True,
        "prey_surfaces_rerun": False,
        "corridor_models_rerun": False,
        "manuscript_or_word_modified": False,
    }
    (QA_OUT / "c1_annual_benthic_prey_observed_support_qa_v1.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    for output in build_figure():
        print(output)
