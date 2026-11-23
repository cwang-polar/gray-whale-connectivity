#!/usr/bin/env python3
"""Build the isolated V3 Figure 1 parallel-scenario candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "manuscript_figures"))
import build_figure1_cee_v0 as base  # noqa: E402


ROOT = base.ROOT
OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_1"
CANDIDATE_OUT = OUT_ROOT / "candidates"
PANEL_OUT = OUT_ROOT / "panels_candidate_v1"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SCENARIO_COLORS = {
    "Prey": "#3B6EA8",
    "Prey + ice": "#3A8D6D",
    "Prey + vessel": "#D9822B",
    "Prey + ice + vessel": "#A85A91",
}


def configure_style() -> None:
    base.apply_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    textcolor: str = "white",
    fontsize: float = 7.0,
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor="none",
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.08,
        zorder=3,
    )


def panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    base.add_panel_label(ax, "A", x=0.004, y=0.975)

    rounded_box(
        ax,
        0.055,
        0.61,
        0.31,
        0.25,
        "Traditional benthic prey\nbase weakens",
        "#C86F42",
        fontsize=7.7,
    )
    rounded_box(
        ax,
        0.475,
        0.61,
        0.32,
        0.25,
        "Evaluate May-July\nnorthward connectivity",
        "#247A72",
        fontsize=7.7,
    )
    ax.annotate(
        "",
        xy=(0.47, 0.735),
        xytext=(0.37, 0.735),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#41464A"),
    )

    branch_y = 0.49
    ax.plot(
        [0.635, 0.635],
        [0.60, branch_y],
        transform=ax.transAxes,
        color="#41464A",
        linewidth=0.9,
        clip_on=False,
    )
    ax.plot(
        [0.125, 0.875],
        [branch_y, branch_y],
        transform=ax.transAxes,
        color="#41464A",
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        0.50,
        0.525,
        "Parallel annual corridor scenarios",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="#555B60",
    )

    scenarios = [
        ("Prey", "2011-2019"),
        ("Prey + ice", "2011-2019"),
        ("Prey + vessel", "2015-2019"),
        ("Prey + ice + vessel", "2015-2019"),
    ]
    x_positions = [0.045, 0.285, 0.525, 0.765]
    width = 0.19
    for (scenario, years), x in zip(scenarios, x_positions):
        center = x + width / 2
        ax.annotate(
            "",
            xy=(center, 0.425),
            xytext=(center, branch_y),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#41464A"),
        )
        rounded_box(
            ax,
            x,
            0.19,
            width,
            0.22,
            f"{scenario}\n{years}",
            SCENARIO_COLORS[scenario],
            fontsize=6.45 if scenario == "Prey + ice + vessel" else 6.8,
        )

    ax.text(
        0.50,
        0.075,
        "All four scenarios were compared over their shared 2015-2019 period",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        color="#555B60",
    )


def panel_c(
    ax: plt.Axes,
    ocean_grid: pd.DataFrame,
    land,
    endpoints: pd.DataFrame,
    extent,
) -> None:
    base.add_panel_label(ax, "C", x=0.012, y=0.985)
    base.draw_base_map(ax, ocean_grid, land, extent)
    base.add_geographic_labels(ax)

    source = endpoints.loc[endpoints["endpoint_type"].eq("source")]
    destination = endpoints.loc[~endpoints["endpoint_type"].eq("source")]
    ax.scatter(
        source["lon"],
        source["lat"],
        s=18,
        color=base.PALETTE["source"],
        edgecolor="white",
        linewidth=0.25,
        zorder=6,
        label=f"Southern source (n={len(source)})",
    )
    ax.scatter(
        destination["lon"],
        destination["lat"],
        s=22,
        marker="^",
        color=base.PALETTE["sink"],
        edgecolor="white",
        linewidth=0.25,
        zorder=6,
        label=f"Northern destination (n={len(destination)})",
    )
    ax.scatter(
        [],
        [],
        s=8,
        color=base.PALETTE["grid"],
        alpha=0.55,
        label="Regional ocean grid",
    )
    legend = ax.legend(
        title="Modeled endpoints",
        loc="lower right",
        fontsize=5.7,
        title_fontsize=5.9,
        borderpad=0.25,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    legend._legend_box.align = "left"


def panel_d(ax: plt.Axes) -> None:
    base.panel_d(ax)
    for text in ax.texts:
        if text.get_text() == "40 sources":
            text.set_text("40 modeled sources")
        elif text.get_text() == "40 destinations":
            text.set_text("40 modeled destinations")


def export_source_data(
    prey_gdf,
    endpoints: pd.DataFrame,
) -> list[Path]:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    scenario_path = SOURCE_OUT / "figure1_panel_a_scenario_design_candidate_v1.csv"
    pd.DataFrame(
        [
            ["Prey", 2011, 2019, True],
            ["Prey + ice", 2011, 2019, True],
            ["Prey + vessel", 2015, 2019, True],
            ["Prey + ice + vessel", 2015, 2019, True],
        ],
        columns=["scenario", "first_year", "last_year", "included_in_2015_2019_common_comparison"],
    ).to_csv(scenario_path, index=False)

    station_path = SOURCE_OUT / "figure1_panel_b_prey_sampling_locations_candidate_v1.csv"
    prey_gdf[["DataYear", "Longitude", "Latitude"]].to_csv(
        station_path, index=False, float_format="%.6f"
    )

    endpoint_path = SOURCE_OUT / "figure1_panel_c_modeled_endpoints_candidate_v1.csv"
    endpoint_columns = [
        column
        for column in ["grid_id", "endpoint_type", "lon", "lat"]
        if column in endpoints.columns
    ]
    endpoints[endpoint_columns].to_csv(endpoint_path, index=False, float_format="%.6f")
    return [scenario_path, station_path, endpoint_path]


def export_panel(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=450, bbox_inches="tight", facecolor="white")
    return outputs


def export_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    CANDIDATE_OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_records(source_paths: list[Path], station_count: int) -> tuple[Path, Path, Path]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    contract = OUT_ROOT / "figure1_revision_contract_candidate_v1.md"
    contract.write_text(
        """# Figure 1 candidate V1 figure contract

- Core conclusion: one May-July northward-connectivity question is evaluated under four parallel corridor scenarios rather than a sequential stressor narrative.
- Panel A: motivation, seasonal connectivity question and scenario design.
- Panel B: observed benthic-prey sampling context, regional ocean grid and DBO reference areas.
- Panel C: fixed modeled source and destination endpoints, explicitly distinguished from sampling and sighting locations.
- Panel D: conversion of 1,600 modeled least-cost paths into an annual corridor-frequency landscape.
- Archetype: schematic-led composite.
- Display checks: sequential-scenario framing, endpoint-versus-station confusion, period comparability and map crowding.
- Export contract: editable PDF/SVG, 600-dpi PNG/TIFF and separate panel exports.
""",
        encoding="utf-8",
    )

    qa = QA_OUT / "figure1_parallel_scenarios_candidate_v1_qa.md"
    qa.write_text(
        f"""# Figure 1 parallel-scenario candidate V1 QA

- Backend: Python in the mypy10 environment.
- Panel A shows four parallel scenarios and their available analysis periods.
- The 2015-2019 common-period comparison includes all four scenarios.
- Panel B retains {station_count:,} plotted benthic-prey records from 2007-2019, the regional ocean grid and DBO1-DBO5 reference areas.
- Panel C contains 40 modeled southern source endpoints and 40 modeled northern destination endpoints.
- Panel C contains no prey sampling stations or whale sightings.
- Panel D retains the 40 x 40 = 1,600 path-ensemble schematic.
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
            ROOT / "scripts" / "manuscript_figures" / "build_figure1_cee_v0.py",
            "preserved_figure1_v1_implementation",
        ),
    ]
    inputs.extend((path, "plot_ready_source_data") for path in source_paths)
    provenance = OUT_ROOT / "provenance_manifest_candidate_v1.csv"
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


def build_figure() -> tuple[list[Path], list[Path]]:
    configure_style()
    _, ocean_grid, dbo, land, endpoints, prey_gdf = base.load_layers()
    extent = base.extent_from_ocean_grid(ocean_grid)
    source_paths = export_source_data(prey_gdf, endpoints)

    fig = plt.figure(figsize=(7.2, 9.05), constrained_layout=False, facecolor="white")
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.02, 2.95, 1.90],
        hspace=0.16,
        wspace=0.16,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :2])
    ax_c = fig.add_subplot(grid[1, 2:])
    ax_d = fig.add_subplot(grid[2, :])

    panel_a(ax_a)
    base.panel_b(ax_b, ocean_grid, dbo, land, prey_gdf, extent)
    panel_c(ax_c, ocean_grid, land, endpoints, extent)
    panel_d(ax_d)

    stem = CANDIDATE_OUT / "gray_whale_figure1_parallel_scenarios_candidate_v1"
    outputs = export_figure(fig, stem)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    review = QA_OUT / "gray_whale_figure1_parallel_scenarios_candidate_v1_review.png"
    fig.savefig(review, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    panel_outputs: list[Path] = []
    panel_specs = [
        ("fig1a_parallel_scenarios_candidate_v1", (7.0, 2.25), lambda ax: panel_a(ax)),
        ("fig1b_study_context_candidate_v1", (3.2, 4.4), lambda ax: base.panel_b(ax, ocean_grid, dbo, land, prey_gdf, extent)),
        ("fig1c_modeled_endpoints_candidate_v1", (3.2, 4.4), lambda ax: panel_c(ax, ocean_grid, land, endpoints, extent)),
        ("fig1d_ensemble_schematic_candidate_v1", (7.0, 3.1), lambda ax: panel_d(ax)),
    ]
    for name, size, draw in panel_specs:
        panel_fig, panel_ax = plt.subplots(figsize=size, facecolor="white")
        draw(panel_ax)
        panel_outputs.extend(export_panel(panel_fig, PANEL_OUT / name))
        plt.close(panel_fig)

    write_records(source_paths, len(prey_gdf))
    return outputs + [review] + panel_outputs, source_paths


def main() -> None:
    outputs, source_paths = build_figure()
    for path in outputs + source_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
