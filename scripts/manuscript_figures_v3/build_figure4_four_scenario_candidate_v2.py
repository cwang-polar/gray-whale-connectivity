#!/usr/bin/env python3
"""Build Figure 4 candidate V2 with unified frequent-corridor terminology."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANNUAL_INPUT = (
    ROOT
    / "outputs"
    / "prey_vessel_phase8k"
    / "comparisons"
    / "four_scenario_annual_summary_2015_2019_v2.csv"
)
SPATIAL_INPUT = (
    ROOT
    / "outputs"
    / "prey_vessel_phase8k"
    / "comparisons"
    / "four_scenario_corridor_overlap_centroid_v2.csv"
)
OUT_ROOT = ROOT / "outputs" / "manuscript_figures_v3" / "figure_4"
OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SCENARIOS = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIO_LABELS = {
    "prey_only": "Prey",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}
SCENARIO_COLORS = {
    "prey_only": "#3B6EA8",
    "prey_ice": "#3A8D6D",
    "prey_vessel": "#D9822B",
    "prey_ice_vessel": "#A85A91",
}
SCENARIO_MARKERS = {
    "prey_only": "o",
    "prey_ice": "s",
    "prey_vessel": "^",
    "prey_ice_vessel": "D",
}
SCENARIO_LINESTYLES = {
    "prey_only": "-",
    "prey_ice": "-",
    "prey_vessel": (0, (4, 2)),
    "prey_ice_vessel": (0, (4, 2)),
}

COMPARISONS = ["vessel_effect_without_ice", "vessel_effect_with_ice"]
COMPARISON_LABELS = {
    "vessel_effect_without_ice": "Prey vs prey + vessel",
    "vessel_effect_with_ice": "Prey + ice vs prey + ice + vessel",
}
COMPARISON_COLORS = {
    "vessel_effect_without_ice": SCENARIO_COLORS["prey_vessel"],
    "vessel_effect_with_ice": SCENARIO_COLORS["prey_ice_vessel"],
}
COMPARISON_MARKERS = {
    "vessel_effect_without_ice": SCENARIO_MARKERS["prey_vessel"],
    "vessel_effect_with_ice": SCENARIO_MARKERS["prey_ice_vessel"],
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.1,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(ANNUAL_INPUT)
    annual_required = {
        "target_year",
        "scenario",
        "mean_path_cost",
        "corridor_area_cells_freq_gt_0_05",
        "successful_path_count",
    }
    missing = annual_required.difference(annual.columns)
    if missing:
        raise ValueError(f"Annual input missing columns: {sorted(missing)}")
    annual = annual.loc[
        annual["scenario"].isin(SCENARIOS)
        & annual["target_year"].between(2015, 2019),
        sorted(annual_required),
    ].copy()
    if len(annual) != 20 or annual.duplicated(["target_year", "scenario"]).any():
        raise ValueError("Expected one annual row for each of four scenarios in 2015-2019")
    if annual[list(annual_required)].isna().any().any():
        raise ValueError("Missing values in annual Figure 4 input")
    if set(annual["successful_path_count"]) != {1600}:
        raise ValueError("Figure 4 requires 1,600 successful paths per scenario-year")

    spatial = pd.read_csv(SPATIAL_INPUT)
    spatial_required = {
        "target_year",
        "comparison",
        "scenario_a",
        "scenario_b",
        "centroid_shift_freq_gt_0_05_km",
        "jaccard_freq_gt_0_05",
    }
    missing = spatial_required.difference(spatial.columns)
    if missing:
        raise ValueError(f"Spatial input missing columns: {sorted(missing)}")
    spatial = spatial.loc[
        spatial["comparison"].isin(COMPARISONS)
        & spatial["target_year"].between(2015, 2019),
        sorted(spatial_required),
    ].copy()
    if len(spatial) != 10 or spatial.duplicated(["target_year", "comparison"]).any():
        raise ValueError("Expected one row for each vessel contrast in 2015-2019")
    if spatial[list(spatial_required)].isna().any().any():
        raise ValueError("Missing values in spatial Figure 4 input")
    if not spatial["jaccard_freq_gt_0_05"].between(0, 1).all():
        raise ValueError("Jaccard overlap must be bounded between 0 and 1")
    if (spatial["centroid_shift_freq_gt_0_05_km"] < 0).any():
        raise ValueError("Centroid displacement cannot be negative")

    expected_years = set(range(2015, 2020))
    if set(annual["target_year"]) != expected_years:
        raise ValueError("Annual input does not cover the common 2015-2019 period")
    if set(spatial["target_year"]) != expected_years:
        raise ValueError("Spatial input does not cover the common 2015-2019 period")
    return annual, spatial


def export_source_data(annual: pd.DataFrame, spatial: pd.DataFrame) -> Path:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    blocks: list[pd.DataFrame] = []

    for panel, metric_col, metric_label, unit in [
        ("A", "mean_path_cost", "annual mean modeled path cost", "modeled cost"),
        (
            "B",
            "corridor_area_cells_freq_gt_0_05",
            "number of frequent-corridor cells",
            "10-km ocean cells",
        ),
    ]:
        block = annual[
            ["target_year", "scenario", metric_col, "successful_path_count"]
        ].rename(columns={"scenario": "series_id", metric_col: "value"})
        block["panel"] = panel
        block["series_label"] = block["series_id"].map(SCENARIO_LABELS)
        block["metric"] = metric_label
        block["unit"] = unit
        block["frequency_threshold"] = "" if panel == "A" else "F_s > 0.05"
        blocks.append(block)

    for panel, metric_col, metric_label, unit in [
        (
            "C",
            "centroid_shift_freq_gt_0_05_km",
            "centroid displacement between paired corridor footprints",
            "km",
        ),
        (
            "D",
            "jaccard_freq_gt_0_05",
            "Jaccard overlap between paired corridor footprints",
            "0-1",
        ),
    ]:
        block = spatial[["target_year", "comparison", metric_col]].rename(
            columns={"comparison": "series_id", metric_col: "value"}
        )
        block["panel"] = panel
        block["series_label"] = block["series_id"].map(COMPARISON_LABELS)
        block["metric"] = metric_label
        block["unit"] = unit
        block["frequency_threshold"] = "F_s > 0.05"
        block["successful_path_count"] = 1600
        blocks.append(block)

    source = pd.concat(blocks, ignore_index=True)
    source["corridor_window"] = "May-July"
    source = source[
        [
            "panel",
            "target_year",
            "series_id",
            "series_label",
            "metric",
            "value",
            "unit",
            "frequency_threshold",
            "successful_path_count",
            "corridor_window",
        ]
    ].sort_values(["panel", "target_year", "series_id"])
    path = SOURCE_OUT / "gray_whale_figure4_four_scenario_source_data_candidate_v2.csv"
    source.to_csv(path, index=False, float_format="%.6f")
    return path


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(2014.75, 2019.25)
    ax.set_xticks(range(2015, 2020))


def plot_scenario_lines(ax: plt.Axes, annual: pd.DataFrame, metric: str) -> None:
    for scenario in SCENARIOS:
        subset = annual.loc[annual["scenario"] == scenario].sort_values("target_year")
        ax.plot(
            subset["target_year"],
            subset[metric],
            color=SCENARIO_COLORS[scenario],
            marker=SCENARIO_MARKERS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario],
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=SCENARIO_LABELS[scenario],
            zorder=3,
        )


def plot_comparison_lines(ax: plt.Axes, spatial: pd.DataFrame, metric: str) -> None:
    for comparison in COMPARISONS:
        subset = spatial.loc[spatial["comparison"] == comparison].sort_values(
            "target_year"
        )
        ax.plot(
            subset["target_year"],
            subset[metric],
            color=COMPARISON_COLORS[comparison],
            marker=COMPARISON_MARKERS[comparison],
            linestyle="-",
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=COMPARISON_LABELS[comparison],
            zorder=3,
        )


def add_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("ABCD", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.045,
            box.y1 + 0.006,
            label,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )


def build_figure(annual: pd.DataFrame, spatial: pd.DataFrame) -> list[Path]:
    expected_root = ROOT / "outputs" / "manuscript_figures_v3"
    if expected_root.resolve() not in OUT.resolve().parents:
        raise RuntimeError("Refusing to write outside the isolated V3 figure tree")

    OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig, grid = plt.subplots(2, 2, figsize=(7.08, 5.0))
    ax_a, ax_b, ax_c, ax_d = grid.ravel()
    axes = [ax_a, ax_b, ax_c, ax_d]
    for ax in axes:
        style_axis(ax)

    plot_scenario_lines(ax_a, annual, "mean_path_cost")
    ax_a.set_ylabel("Mean modeled path cost")
    ax_a.set_ylim(1080, 1990)
    ax_a.set_yticks([1100, 1300, 1500, 1700, 1900])
    ax_a.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.85,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.25,
    )

    plot_scenario_lines(ax_b, annual, "corridor_area_cells_freq_gt_0_05")
    ax_b.set_ylabel("Frequent-corridor cells (n)")
    ax_b.set_ylim(160, 275)
    ax_b.set_yticks([160, 180, 200, 220, 240, 260])

    plot_comparison_lines(ax_c, spatial, "centroid_shift_freq_gt_0_05_km")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Corridor-centroid displacement (km)")
    ax_c.set_ylim(0, 110)
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.legend(
        loc="lower right",
        frameon=False,
        handlelength=2.1,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.35,
    )

    plot_comparison_lines(ax_d, spatial, "jaccard_freq_gt_0_05")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Frequent-corridor overlap (Jaccard)")
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.98, wspace=0.30, hspace=0.34)
    add_panel_labels(fig, axes)

    stem = OUT / "gray_whale_figure4_four_scenario_comparison_candidate_v2"
    outputs = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".png"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=600, facecolor="white")
    fig.savefig(
        outputs[3],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        QA_OUT / "gray_whale_figure4_four_scenario_candidate_v2_manuscript_scale.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    return outputs


def main() -> None:
    annual, spatial = validate_inputs()
    source_path = export_source_data(annual, spatial)
    outputs = build_figure(annual, spatial)
    print(f"Source data: {source_path}")
    for path in outputs:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
