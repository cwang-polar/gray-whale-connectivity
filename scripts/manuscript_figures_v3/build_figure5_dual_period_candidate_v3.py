#!/usr/bin/env python3
"""Build Figure 5 candidate V3 with DBO2-DBO4 as neutral map context."""

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
from matplotlib import patheffects

import build_figure5_dual_period_candidate_v1 as base
import build_figure5_dual_period_candidate_v2 as prior


ROOT = base.ROOT
OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_5"
CANDIDATE_OUT = OUT_ROOT / "candidates"
QA_OUT = OUT_ROOT / "qa"


def load_map_context_v3(*, paths=None):
    land, grid, extent = base.load_map_context(paths=paths)
    dbo = gpd.read_file(paths.input("dbo") if paths is not None else prior.DBO).to_crs("EPSG:4326")
    expected = {"DBO2", "DBO3", "DBO4"}
    if set(dbo["name"]) != expected:
        raise ValueError(f"Expected DBO2-DBO4 polygons, found {set(dbo['name'])}")
    boundary = prior.maritime_boundary_coordinates(extent[3])
    return land, grid, extent, dbo, boundary


def add_geographic_labels(ax: plt.Axes) -> None:
    neutral_labels = [
        ("Russia", -173.2, 66.0, 5.2, "normal", 0),
        ("Alaska", -158.4, 68.5, 5.2, "normal", 0),
        ("St. Lawrence\nIsland", -170.1, 63.35, 4.6, "normal", 0),
        ("Chirikov\nBasin", -168.05, 64.23, 4.7, "italic", 0),
        ("Bering\nStrait", -167.20, 65.62, 4.7, "italic", 0),
        ("U.S.-Russia maritime boundary", -168.62, 71.12, 4.25, "normal", 90),
    ]
    for label, lon, lat, size, style, rotation in neutral_labels:
        text = ax.text(
            lon,
            lat,
            label,
            fontsize=size,
            color=prior.BOUNDARY_COLOR if "boundary" in label else "#4A4640",
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

    dbo_labels = [
        ("DBO2", -170.25, 64.80),
        ("DBO3", -168.95, 67.70),
        ("DBO4", -163.75, 71.72),
    ]
    for label, lon, lat in dbo_labels:
        text = ax.text(
            lon,
            lat,
            label,
            fontsize=4.8,
            fontweight="bold",
            color=prior.DBO_COLOR,
            ha="center",
            va="center",
            zorder=8,
        )
        text.set_path_effects(
            [patheffects.withStroke(linewidth=1.4, foreground="white")]
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_qa() -> Path:
    output = QA_OUT / "figure5_dual_period_candidate_v3_qa.md"
    output.write_text(
        f"""# Figure 5 dual-period candidate V3 QA

- Backend: Python in the mypy10 environment.
- Revision scope: map context only in panels C-D.
- DBO2, DBO3 and DBO4 anchor areas are shown with identical line and label styling.
- The U.S.-Russia maritime boundary remains visible as geographic context.
- Panels A-B and all modeled-cell values are unchanged from candidate V2.
- Panel C retains 18 Prey-only cells, 23 Prey-plus-ice-only cells and 20 cells recurrent in both landscapes.
- Panel D retains 20 recurrent cells and nine exposure-priority cells.
- Panels C-D contain no ASAMM sighting coordinates.
- Maritime boundary source: {prior.BOUNDARY_SOURCE_URL}
""",
        encoding="utf-8",
    )
    return output


def write_provenance() -> Path:
    inputs = [
        (base.FULL_VALIDATION, "full_period_validation"),
        (base.COMMON_VALIDATION, "common_period_validation"),
        (base.GRID, "ocean_grid"),
        (base.LAND, "land_basemap"),
        (prior.DBO, "DBO2-DBO4_anchor_polygons"),
        (base.PREY_ONLY_PERSISTENCE, "prey_recurrence"),
        (base.PREY_ICE_PERSISTENCE, "prey_ice_recurrence"),
        (
            ROOT
            / "scripts"
            / "manuscript_figures_v3"
            / "build_figure5_dual_period_candidate_v2.py",
            "preserved_candidate_v2_implementation",
        ),
    ]
    inputs.extend(
        (base.vessel_surface_path(year), f"vessel_surface_{year}")
        for year in base.VESSEL_YEARS
    )
    records = [
        {
            "role": role,
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path, role in inputs
    ]
    output = OUT_ROOT / "provenance_manifest_v3.csv"
    import pandas as pd

    pd.DataFrame(records).to_csv(output, index=False)
    return output


def build_figure() -> list[Path]:
    base.configure_style()
    full, common = base.load_validation_data()
    union, stable = base.load_recurrent_cells()
    exposure = base.load_exposure(stable)
    land, grid, extent, dbo, boundary = load_map_context_v3()

    # The V2 map functions resolve this module-level label helper at draw time.
    prior.add_geographic_labels = add_geographic_labels

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

    prior.draw_recurrence_map(ax_c, land, grid, extent, dbo, boundary, union)
    prior.draw_exposure_map(fig, ax_d, cax_d, land, grid, extent, dbo, boundary, exposure)
    base.add_panel_label(ax_c, "C")
    base.add_panel_label(ax_d, "D")

    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    stem = CANDIDATE_OUT / "gray_whale_figure5_dual_period_validation_candidate_v3"
    outputs = base.export_figure(fig, stem)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    review_path = QA_OUT / "gray_whale_figure5_dual_period_validation_candidate_v3_review.png"
    fig.savefig(review_path, dpi=180, facecolor="white")
    plt.close(fig)

    write_qa()
    write_provenance()
    return outputs + [review_path]


def main() -> None:
    for path in build_figure():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
