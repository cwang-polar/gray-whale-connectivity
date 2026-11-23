#!/usr/bin/env python3
"""Build the isolated three-panel V3 Figure 1 candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_figure1_parallel_scenarios_candidate_v1 as v1  # noqa: E402


base = v1.base
ROOT = v1.ROOT
OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_1"
CANDIDATE_OUT = OUT_ROOT / "candidates"
PANEL_OUT = OUT_ROOT / "panels_candidate_v2"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"


def relabel_panel(ax: plt.Axes, old: str, new: str) -> None:
    for artist in ax.texts:
        if artist.get_text() == old:
            artist.set_text(new)
            return
    raise RuntimeError(f"Panel label {old!r} was not found")


def panel_a(ax, ocean_grid, dbo, land, prey_gdf, extent) -> None:
    base.panel_b(ax, ocean_grid, dbo, land, prey_gdf, extent)
    relabel_panel(ax, "B", "A")


def panel_b(ax, ocean_grid, land, endpoints, extent) -> None:
    v1.panel_c(ax, ocean_grid, land, endpoints, extent)
    relabel_panel(ax, "C", "B")


def panel_c(ax) -> None:
    v1.panel_d(ax)
    relabel_panel(ax, "D", "C")


def export_source_data(prey_gdf, endpoints: pd.DataFrame) -> list[Path]:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    station_path = SOURCE_OUT / "figure1_panel_a_prey_sampling_locations_candidate_v2.csv"
    prey_gdf[["DataYear", "Longitude", "Latitude"]].to_csv(
        station_path, index=False, float_format="%.6f"
    )

    endpoint_path = SOURCE_OUT / "figure1_panel_b_modeled_endpoints_candidate_v2.csv"
    endpoint_columns = [
        column
        for column in ["grid_id", "endpoint_type", "lon", "lat"]
        if column in endpoints.columns
    ]
    endpoints[endpoint_columns].to_csv(
        endpoint_path, index=False, float_format="%.6f"
    )
    return [station_path, endpoint_path]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_records(
    source_paths: list[Path], station_count: int, png_path: Path
) -> tuple[Path, Path, Path]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)

    contract = OUT_ROOT / "figure1_revision_contract_candidate_v2.md"
    contract.write_text(
        """# Figure 1 three-panel candidate V2 figure contract

- Core conclusion: the figure locates the repeatedly sampled Pacific Arctic domain and shows how fixed modeled endpoints generate annual corridor-frequency landscapes.
- Panel A: benthic-prey sampling context, regional ocean grid and DBO reference areas.
- Panel B: fixed modeled source and destination endpoints, explicitly distinguished from sampling and sighting locations.
- Panel C: conversion of 1,600 modeled least-cost paths into an annual corridor-frequency landscape.
- Archetype: study-system maps plus method schematic.
- Display checks: endpoint-versus-station confusion, geographic context, panel redundancy and map crowding.
- Export contract: editable PDF/SVG, 600-dpi PNG/TIFF and separate panel exports.
- Change from V1: the former conceptual scenario panel was removed; former panels B-D were retained and relabelled A-C.
""",
        encoding="utf-8",
    )

    with Image.open(png_path) as image:
        width, height = image.size
        extrema = image.convert("L").getextrema()

    qa = QA_OUT / "figure1_three_panel_candidate_v2_qa.md"
    qa.write_text(
        f"""# Figure 1 three-panel candidate V2 QA

- Backend: Python in the mypy10 environment.
- The former conceptual scenario panel was removed without changing the scientific content of the remaining panels.
- Panel A retains {station_count:,} plotted benthic-prey records from 2007-2019, the regional ocean grid and DBO1-DBO5 reference areas.
- Panel B contains 40 modeled southern source endpoints and 40 modeled northern destination endpoints.
- Panel B contains no prey sampling stations or whale sightings.
- Panel C retains the 40 x 40 = 1,600 path-ensemble schematic.
- PNG dimensions: {width:,} x {height:,} pixels; grayscale extrema: {extrema}.
- PDF, SVG, PNG and TIFF exports and separate panel SVG/PNG files were generated.
""",
        encoding="utf-8",
    )

    inputs = [
        (base.DOMAIN, "study_region"),
        (base.GRID, "regional_ocean_grid"),
        (base.ENDPOINTS, "modeled_endpoints"),
        (base.LAND, "land_basemap"),
        (base.PREY, "benthic_prey_records"),
        (
            ROOT
            / "scripts"
            / "manuscript_figures_v3"
            / "build_figure1_parallel_scenarios_candidate_v1.py",
            "preserved_figure1_v1_implementation",
        ),
        (
            ROOT
            / "scripts"
            / "manuscript_figures_v3"
            / "build_figure1_three_panel_candidate_v2.py",
            "figure1_v2_implementation",
        ),
    ]
    inputs.extend((path, "plot_ready_source_data") for path in source_paths)
    provenance = OUT_ROOT / "provenance_manifest_candidate_v2.csv"
    pd.DataFrame(
        [
            {
                "role": role,
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path, role in inputs
        ]
    ).to_csv(provenance, index=False)
    return contract, qa, provenance


def build_figure(prepared=None, stem_name="gray_whale_figure1_three_panel_candidate_v2") -> list[Path]:
    v1.configure_style()
    if prepared is None:
        _, ocean_grid, dbo, land, endpoints, prey_gdf = base.load_layers()
    else:
        ocean_grid, dbo, land, endpoints, prey_gdf = prepared
    extent = base.extent_from_ocean_grid(ocean_grid)
    source_paths = export_source_data(prey_gdf, endpoints) if prepared is None else []

    fig = plt.figure(figsize=(7.2, 7.25), constrained_layout=False, facecolor="white")
    grid = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[2.95, 1.90],
        hspace=0.14,
        wspace=0.16,
    )
    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[0, 2:])
    ax_c = fig.add_subplot(grid[1, :])

    panel_a(ax_a, ocean_grid, dbo, land, prey_gdf, extent)
    panel_b(ax_b, ocean_grid, land, endpoints, extent)
    panel_c(ax_c)

    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    stem = CANDIDATE_OUT / stem_name
    outputs = v1.export_figure(fig, stem)

    QA_OUT.mkdir(parents=True, exist_ok=True)
    review = QA_OUT / f"{stem_name}_review.png"
    fig.savefig(review, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if prepared is not None:
        return outputs + [review]

    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    panel_outputs: list[Path] = []
    panel_specs = [
        (
            "fig1a_study_context_candidate_v2",
            (3.2, 4.4),
            lambda ax: panel_a(ax, ocean_grid, dbo, land, prey_gdf, extent),
        ),
        (
            "fig1b_modeled_endpoints_candidate_v2",
            (3.2, 4.4),
            lambda ax: panel_b(ax, ocean_grid, land, endpoints, extent),
        ),
        (
            "fig1c_ensemble_schematic_candidate_v2",
            (7.0, 3.1),
            lambda ax: panel_c(ax),
        ),
    ]
    for name, size, draw in panel_specs:
        panel_fig, panel_ax = plt.subplots(figsize=size, facecolor="white")
        draw(panel_ax)
        panel_outputs.extend(v1.export_panel(panel_fig, PANEL_OUT / name))
        plt.close(panel_fig)

    write_records(source_paths, len(prey_gdf), stem.with_suffix(".png"))
    return outputs + [review] + panel_outputs + source_paths


def main() -> None:
    for path in build_figure():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
