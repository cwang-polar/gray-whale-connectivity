#!/usr/bin/env python3
"""Build Figure 5 candidate V2 with explicit spatial semantics in panels C-D."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Geod

import build_figure5_dual_period_candidate_v1 as base


ROOT = base.ROOT
DBO = ROOT / "data" / "processed" / "study_domain" / "dbo_anchor_polygons.gpkg"

OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_5"
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

BOUNDARY_SOURCE_URL = (
    "https://www.un.org/Depts/los/LEGISLATIONANDTREATIES/PDFFILES/"
    "TREATIES/USA-RUS1990MB.PDF"
)

# WGS84 positions from the annex to the 1990 U.S.-U.S.S.R. Maritime Boundary
# Agreement. Points 1-16 cover the part of the agreement line visible here.
MARITIME_BOUNDARY_DMS = [
    ((65, 30, 0), (168, 58, 37)),
    ((65, 19, 58), (169, 21, 38)),
    ((65, 9, 51), (169, 44, 34)),
    ((64, 59, 41), (170, 7, 23)),
    ((64, 49, 26), (170, 30, 6)),
    ((64, 39, 8), (170, 52, 43)),
    ((64, 28, 46), (171, 15, 14)),
    ((64, 18, 20), (171, 37, 40)),
    ((64, 7, 50), (172, 0, 0)),
    ((63, 59, 27), (172, 18, 39)),
    ((63, 51, 1), (172, 37, 13)),
    ((63, 42, 33), (172, 55, 42)),
    ((63, 34, 1), (173, 14, 7)),
    ((63, 25, 27), (173, 32, 27)),
    ((63, 16, 50), (173, 50, 42)),
    ((63, 8, 11), (174, 8, 52)),
]

DBO_COLOR = "#187A73"
BOUNDARY_COLOR = "#525E66"


def dms_to_decimal(parts: tuple[int, int, int], negative: bool = False) -> float:
    degrees, minutes, seconds = parts
    value = degrees + minutes / 60.0 + seconds / 3600.0
    return -value if negative else value


def maritime_boundary_coordinates(ymax: float) -> pd.DataFrame:
    treaty = [
        (
            dms_to_decimal(lon, negative=True),
            dms_to_decimal(lat),
        )
        for lat, lon in MARITIME_BOUNDARY_DMS
    ]
    treaty.insert(0, (treaty[0][0], max(ymax, treaty[0][1])))

    geod = Geod(ellps="WGS84")
    dense: list[tuple[float, float]] = []
    for start, end in zip(treaty[:-1], treaty[1:]):
        lon1, lat1 = start
        lon2, lat2 = end
        _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)
        intermediate_count = max(0, int(distance_m // 10_000) - 1)
        segment = [(lon1, lat1)]
        if intermediate_count:
            segment.extend(geod.npts(lon1, lat1, lon2, lat2, intermediate_count))
        dense.extend(segment)
    dense.append(treaty[-1])
    return pd.DataFrame(dense, columns=["lon", "lat"])


def load_map_context_v2() -> tuple[
    gpd.GeoDataFrame,
    pd.DataFrame,
    tuple[float, float, float, float],
    gpd.GeoDataFrame,
    pd.DataFrame,
]:
    land, grid, extent = base.load_map_context()
    dbo = gpd.read_file(DBO).to_crs("EPSG:4326")
    dbo3 = dbo.loc[dbo["name"].eq("DBO3")].copy()
    if len(dbo3) != 1:
        raise ValueError("Expected exactly one DBO3 anchor polygon")
    boundary = maritime_boundary_coordinates(extent[3])
    return land, grid, extent, dbo3, boundary


def add_geographic_context(
    ax: plt.Axes,
    dbo3: gpd.GeoDataFrame,
    boundary: pd.DataFrame,
) -> None:
    dbo3.boundary.plot(
        ax=ax,
        color=DBO_COLOR,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    ax.plot(
        boundary["lon"],
        boundary["lat"],
        color=BOUNDARY_COLOR,
        linewidth=0.85,
        linestyle=(0, (4, 2)),
        zorder=4,
    )


def add_geographic_labels(ax: plt.Axes) -> None:
    labels = [
        ("Russia", -173.2, 66.0, 5.2, "normal", 0),
        ("Alaska", -158.4, 68.5, 5.2, "normal", 0),
        ("St. Lawrence\nIsland", -170.1, 63.35, 4.6, "normal", 0),
        ("Chirikov\nBasin", -168.25, 64.40, 4.7, "italic", 0),
        ("Bering\nStrait", -167.20, 65.62, 4.7, "italic", 0),
        ("DBO3 / SE Chukchi Sea", -168.75, 67.55, 4.6, "italic", 0),
        ("U.S.-Russia maritime boundary", -168.62, 71.12, 4.25, "normal", 90),
    ]
    for label, lon, lat, size, style, rotation in labels:
        text = ax.text(
            lon,
            lat,
            label,
            fontsize=size,
            color=BOUNDARY_COLOR if "boundary" in label else "#4A4640",
            fontstyle=style,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            zorder=8,
        )
        text.set_path_effects(
            [patheffects.withStroke(linewidth=1.35, foreground="white")]
        )


def draw_recurrence_map(
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
    dbo3: gpd.GeoDataFrame,
    boundary: pd.DataFrame,
    union: pd.DataFrame,
    *,
    exclusive_labels: bool = False,
) -> None:
    base.draw_base_map(ax, land, grid, extent)
    add_geographic_context(ax, dbo3, boundary)
    add_geographic_labels(ax)

    specs = [
        ("Prey only", "#B7BCC2", "#80858A", 22),
        ("Prey + ice only", "#85B6D7", "#547E9B", 22),
        ("Both", "#168778", "#075E54", 39),
    ]
    for category, face, edge, size in specs:
        subset = union.loc[union["recurrence_class"].eq(category)]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=size,
            marker="s",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.45,
            alpha=0.96,
            zorder=6,
        )

    counts = union["recurrence_class"].value_counts()
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.45,
            markersize=5.3,
        )
        for _, face, edge, _ in specs
    ]
    if exclusive_labels:
        labels = [
            f"Prey-exclusive (n={int(counts.get('Prey only', 0))})",
            f"Prey + ice-exclusive (n={int(counts.get('Prey + ice only', 0))})",
            f"Shared (n={int(counts.get('Both', 0))})",
        ]
    else:
        labels = [
            f"Prey only (n={int(counts.get('Prey only', 0))})",
            f"Prey + ice only (n={int(counts.get('Prey + ice only', 0))})",
            f"Both landscapes (n={int(counts.get('Both', 0))})",
        ]
    legend = ax.legend(
        handles,
        labels,
        title="Modeled recurrence",
        loc="lower right",
        borderpad=0.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=5.1,
        title_fontsize=5.2,
    )
    legend._legend_box.align = "left"
    ax.text(
        0.98,
        0.98,
        "2011-2019",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        fontweight="bold",
        color="#4B4B4B",
    )


def draw_exposure_map(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    land: gpd.GeoDataFrame,
    grid: pd.DataFrame,
    extent: tuple[float, float, float, float],
    dbo3: gpd.GeoDataFrame,
    boundary: pd.DataFrame,
    exposure: pd.DataFrame,
) -> None:
    base.draw_base_map(ax, land, grid, extent)
    add_geographic_context(ax, dbo3, boundary)
    add_geographic_labels(ax)

    norm = colors.Normalize(vmin=0.0, vmax=0.5)
    cmap = colors.LinearSegmentedColormap.from_list(
        "exposure_magma", plt.cm.magma(np.linspace(0.12, 0.94, 256))
    )
    points = ax.scatter(
        exposure["lon"],
        exposure["lat"],
        c=exposure["mean_scaled_vessel_exposure"],
        cmap=cmap,
        norm=norm,
        s=43,
        marker="s",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.98,
        zorder=6,
    )
    candidates = exposure.loc[exposure["exposure_priority"]]
    ax.scatter(
        candidates["lon"],
        candidates["lat"],
        s=68,
        marker="s",
        facecolor="none",
        edgecolor="#C13E32",
        linewidth=1.0,
        zorder=7,
    )

    colorbar = fig.colorbar(points, cax=cax)
    colorbar.set_label("Mean scaled vessel exposure", fontsize=5.8, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.4, length=2, width=0.55)
    colorbar.outline.set_linewidth(0.55)

    stable_handle = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor="#4C1D73",
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=5.2,
    )
    priority_handle = Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor="none",
        markeredgecolor="#C13E32",
        markeredgewidth=1.0,
        markersize=5.5,
    )
    legend = ax.legend(
        [stable_handle, priority_handle],
        [
            f"Recurrent in both (n={len(exposure)})",
            f"Exposure priority (n={len(candidates)}; >0.25)",
        ],
        title="Model-derived cells",
        loc="lower right",
        borderpad=0.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=5.0,
        title_fontsize=5.2,
    )
    legend._legend_box.align = "left"
    ax.text(
        0.98,
        0.98,
        "2015-2019",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        fontweight="bold",
        color="#4B4B4B",
    )


def export_source_data(
    full: pd.DataFrame,
    common: pd.DataFrame,
    union: pd.DataFrame,
    exposure: pd.DataFrame,
    boundary: pd.DataFrame,
) -> list[Path]:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    panel_a = full.copy()
    panel_a["scenario"] = panel_a["scenario"].astype(str)
    panel_a["scenario_label"] = panel_a["scenario"].map(base.SCENARIO_LABELS)
    panel_a["observation_window"] = "June-July"
    panel_a["corridor_window"] = "May-July"
    panel_a["panel_semantics"] = "observed-use metric; no sighting coordinates shown"
    panel_a_path = SOURCE_OUT / "figure5_panel_a_full_period_validation_candidate_v2.csv"
    panel_a.to_csv(panel_a_path, index=False, float_format="%.6f")

    panel_b = common.copy()
    panel_b["scenario"] = panel_b["scenario"].astype(str)
    panel_b["scenario_label"] = panel_b["scenario"].map(base.SCENARIO_LABELS)
    panel_b["observation_window"] = "June-July"
    panel_b["corridor_window"] = "May-July"
    panel_b["panel_semantics"] = "observed-use metric; no sighting coordinates shown"
    panel_b_path = SOURCE_OUT / "figure5_panel_b_common_period_validation_candidate_v2.csv"
    panel_b.to_csv(panel_b_path, index=False, float_format="%.6f")

    panel_c = union[
        [
            "grid_id",
            "lon",
            "lat",
            "recurrence_class",
            "prey_years_in_top5pct",
            "ice_years_in_top5pct",
        ]
    ].copy()
    panel_c["analysis_period"] = "2011-2019"
    panel_c["annual_membership_definition"] = "annual top 5% of corridor frequency"
    panel_c["recurrence_threshold_years"] = base.RECURRENCE_THRESHOLD
    panel_c["cell_semantics"] = "model-derived corridor cell; not a sighting"
    panel_c_path = SOURCE_OUT / "figure5_panel_c_recurrent_cells_candidate_v2.csv"
    panel_c.to_csv(panel_c_path, index=False, float_format="%.6f")

    panel_d = exposure.copy()
    panel_d["analysis_period"] = "2015-2019"
    panel_d["exposure_priority_threshold"] = base.EXPOSURE_THRESHOLD
    panel_d["cell_semantics"] = "model-derived recurrent corridor cell; not a sighting"
    panel_d_path = SOURCE_OUT / "figure5_panel_d_exposure_priority_candidate_v2.csv"
    panel_d.to_csv(panel_d_path, index=False, float_format="%.6f")

    boundary_source = boundary.copy()
    boundary_source.insert(0, "vertex_order", np.arange(1, len(boundary_source) + 1))
    boundary_source["source"] = BOUNDARY_SOURCE_URL
    boundary_path = SOURCE_OUT / "figure5_us_russia_maritime_boundary_candidate_v2.csv"
    boundary_source.to_csv(boundary_path, index=False, float_format="%.8f")
    return [panel_a_path, panel_b_path, panel_c_path, panel_d_path, boundary_path]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(source_paths: list[Path]) -> Path:
    inputs = [
        (base.FULL_VALIDATION, "full_period_validation"),
        (base.COMMON_VALIDATION, "common_period_validation"),
        (base.GRID, "ocean_grid"),
        (base.LAND, "land_basemap"),
        (DBO, "DBO_anchor_polygons"),
        (base.PREY_ONLY_PERSISTENCE, "prey_recurrence"),
        (base.PREY_ICE_PERSISTENCE, "prey_ice_recurrence"),
        (
            ROOT
            / "scripts"
            / "manuscript_figures_v3"
            / "build_figure5_dual_period_candidate_v1.py",
            "preserved_candidate_v1_implementation",
        ),
    ]
    inputs.extend(
        (base.vessel_surface_path(year), f"vessel_surface_{year}")
        for year in base.VESSEL_YEARS
    )
    inputs.extend((path, "plot_ready_source_data") for path in source_paths)
    records = [
        {
            "role": role,
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path, role in inputs
    ]
    output = OUT_ROOT / "provenance_manifest_v2.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    return output


def write_contract_and_qa(
    full: pd.DataFrame,
    common: pd.DataFrame,
    union: pd.DataFrame,
    exposure: pd.DataFrame,
) -> tuple[Path, Path]:
    QA_OUT.mkdir(parents=True, exist_ok=True)
    contract_path = OUT_ROOT / "figure5_revision_contract_v2.md"
    contract_path.write_text(
        """# Figure 5 candidate V2 figure contract

- Core conclusion: observed sightings evaluate scenario consistency, whereas recurrent and exposure-priority cells are separate model-derived spatial summaries.
- Evidence chain: panels A-B retain the frozen observed-use metrics; panel C retains recurrent-cell membership; panel D retains vessel exposure across cells recurrent in both biological landscapes.
- Revision purpose: prevent model-derived cells in panels C-D from being mistaken for ASAMM sighting locations.
- Archetype: mixed quantitative and spatial evidence.
- Geographic context: DBO3 anchor polygon and the 1990 U.S.-U.S.S.R. maritime boundary are shown without changing modeled cells.
- Export contract: 7.2 x 8.65 inches; editable PDF/SVG; 600-dpi PNG/TIFF.
- Display checks: sighting-versus-model semantics, U.S.-Russia spatial context, legend crowding, unchanged counts and metrics.
""",
        encoding="utf-8",
    )

    counts = union["recurrence_class"].value_counts()
    qa_path = QA_OUT / "figure5_dual_period_candidate_v2_qa.md"
    qa_path.write_text(
        f"""# Figure 5 dual-period candidate V2 QA

- Backend: Python in the mypy10 environment.
- Figure size: 7.2 x 8.65 inches.
- Panel A: 2011-2019, 1,179 identical sighting records per scenario.
- Panel B: 2015-2019, 666 identical sighting records per scenario.
- Panels A-B: all values, metric definitions and axis limits are unchanged from candidate V1.
- Panel C: {int(counts.get('Prey only', 0))} Prey-only cells, {int(counts.get('Prey + ice only', 0))} Prey-plus-ice-only cells and {int(counts.get('Both', 0))} cells recurrent in both.
- Panel D: {len(exposure)} model-derived recurrent cells, including {int(exposure['exposure_priority'].sum())} exposure-priority cells.
- Panels C-D contain no ASAMM sighting coordinates.
- DBO3 and the U.S.-Russia maritime boundary are geographic context only and do not enter the recurrence or exposure calculations.
- Maritime boundary source: {BOUNDARY_SOURCE_URL}
""",
        encoding="utf-8",
    )
    return contract_path, qa_path


def build_figure() -> tuple[list[Path], list[Path]]:
    base.configure_style()
    full, common = base.load_validation_data()
    union, stable = base.load_recurrent_cells()
    exposure = base.load_exposure(stable)
    land, grid, extent, dbo3, boundary = load_map_context_v2()
    source_paths = export_source_data(full, common, union, exposure, boundary)

    fig = plt.figure(figsize=(7.2, 8.65), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.72, 1.12, 2.2],
        left=0.105,
        right=0.925,
        bottom=0.045,
        top=0.955,
        hspace=0.43,
    )
    base.draw_metric_panel(fig, outer[0, 0], full, "A", "2011-2019 | 1,179 sightings")
    base.draw_metric_panel(fig, outer[1, 0], common, "B", "2015-2019 | 666 sightings")

    maps = outer[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.17)
    c_spec = maps[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    d_spec = maps[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    ax_c = fig.add_subplot(c_spec[0, 0])
    ax_c_dummy = fig.add_subplot(c_spec[0, 1])
    ax_c_dummy.set_axis_off()
    ax_d = fig.add_subplot(d_spec[0, 0])
    cax_d = fig.add_subplot(d_spec[0, 1])

    draw_recurrence_map(ax_c, land, grid, extent, dbo3, boundary, union)
    draw_exposure_map(fig, ax_d, cax_d, land, grid, extent, dbo3, boundary, exposure)
    base.add_panel_label(ax_c, "C")
    base.add_panel_label(ax_d, "D")

    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    stem = CANDIDATE_OUT / "gray_whale_figure5_dual_period_validation_candidate_v2"
    outputs = base.export_figure(fig, stem)
    review_path = QA_OUT / "gray_whale_figure5_dual_period_validation_candidate_v2_review.png"
    QA_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(review_path, dpi=180, facecolor="white")
    plt.close(fig)

    write_contract_and_qa(full, common, union, exposure)
    write_provenance(source_paths)
    return outputs + [review_path], source_paths


def main() -> None:
    outputs, source_paths = build_figure()
    for path in outputs + source_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
