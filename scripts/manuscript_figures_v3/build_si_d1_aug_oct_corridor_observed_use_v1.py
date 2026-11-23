#!/usr/bin/env python3
"""Build the D1 August-October corridor and observed-use SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-d1-v1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


PHASE_ROOT = ROOT / "outputs" / "southward_connectivity_phase8q"
COMPARISON_PATH = PHASE_ROOT / "summaries" / "phase8q_aug_oct_scenario_comparisons.csv"
ROUTE_PATH = PHASE_ROOT / "summaries" / "phase8q_aug_oct_route_summary.csv"
ANNUAL_OBSERVED_PATH = PHASE_ROOT / "validation" / "asamm_aug_oct_four_scenario_by_year.csv"
AVAILABLE_OBSERVED_PATH = PHASE_ROOT / "validation" / "asamm_aug_oct_available_period_summary.csv"
COMMON_OBSERVED_PATH = PHASE_ROOT / "validation" / "asamm_aug_oct_common_2015_2019_summary.csv"

OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "d1_august_october_corridor_observed_use_v1"
)
CANDIDATE_DIR = OUT_ROOT / "candidates"
SOURCE_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"
STEM = "gray_whale_si_d1_aug_oct_corridor_observed_use_candidate_v1"

SCENARIO_ORDER = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
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

CONTRAST_ORDER = [
    "ice_effect_without_vessel",
    "ice_effect_with_vessel",
    "vessel_effect_without_ice",
    "vessel_effect_with_ice",
]
CONTRASTS = {
    "ice_effect_without_vessel": {
        "label": "Ice added to Prey",
        "factor": "ice",
        "color": SCENARIO_COLORS["prey_ice"],
        "marker": "s",
        "linestyle": "-",
        "filled": True,
        "offset": -0.055,
    },
    "ice_effect_with_vessel": {
        "label": "Ice added to Prey + vessel",
        "factor": "ice",
        "color": SCENARIO_COLORS["prey_ice"],
        "marker": "s",
        "linestyle": (0, (3, 2)),
        "filled": False,
        "offset": 0.055,
    },
    "vessel_effect_without_ice": {
        "label": "Vessel added to Prey",
        "factor": "vessel",
        "color": SCENARIO_COLORS["prey_vessel"],
        "marker": "^",
        "linestyle": "-",
        "filled": True,
        "offset": -0.055,
    },
    "vessel_effect_with_ice": {
        "label": "Vessel added to Prey + ice",
        "factor": "vessel",
        "color": SCENARIO_COLORS["prey_vessel"],
        "marker": "^",
        "linestyle": (0, (3, 2)),
        "filled": False,
        "offset": 0.055,
    },
}

CORRIDOR_METRICS = [
    (
        "mean_cost_percent_change",
        "Mean path cost\nchange (%)",
        (-1.5, 24.0),
        [0, 10, 20],
        0.0,
    ),
    (
        "frequent_area_b_minus_a",
        "Frequent-corridor area\nchange (cells)",
        (-86.0, 7.0),
        [-80, -40, 0],
        0.0,
    ),
    (
        "jaccard_freq_gt_0_05",
        "Jaccard overlap",
        (0.0, 1.06),
        [0.0, 0.5, 1.0],
        1.0,
    ),
    (
        "centroid_shift_freq_gt_0_05_km",
        "Centroid displacement\n(km)",
        (-5.0, 135.0),
        [0, 60, 120],
        0.0,
    ),
]

OBSERVED_METRICS = [
    (
        "mean_effort_footprint_percentile",
        "Mean effort-footprint\npercentile",
        (45.0, 67.5),
        [45, 50, 55, 60, 65],
        50.0,
    ),
    (
        "percent_records_top5pct",
        "Sightings in annual\ntop 5% cells (%)",
        (-1.0, 26.5),
        [0, 10, 20],
        None,
    ),
    (
        "median_distance_to_frequent_corridor_km",
        "Median distance to\nfrequent corridor (km)",
        (-3.0, 91.0),
        [0, 30, 60, 90],
        None,
    ),
]


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
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_directories() -> None:
    for path in (OUT_ROOT, CANDIDATE_DIR, SOURCE_DIR, QA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = pd.read_csv(COMPARISON_PATH)
    routes = pd.read_csv(ROUTE_PATH)
    annual = pd.read_csv(ANNUAL_OBSERVED_PATH)
    available = pd.read_csv(AVAILABLE_OBSERVED_PATH)
    common = pd.read_csv(COMMON_OBSERVED_PATH)

    required_comparison = {
        "target_year",
        "comparison",
        "scenario_a",
        "scenario_b",
        "mean_cost_percent_change",
        "frequent_area_b_minus_a",
        "jaccard_freq_gt_0_05",
        "centroid_shift_freq_gt_0_05_km",
    }
    missing = required_comparison.difference(comparisons.columns)
    if missing:
        raise ValueError(f"Missing comparison fields: {sorted(missing)}")
    comparisons = comparisons.loc[comparisons["comparison"].isin(CONTRAST_ORDER)].copy()
    expected_counts = {
        "ice_effect_without_vessel": 9,
        "ice_effect_with_vessel": 5,
        "vessel_effect_without_ice": 5,
        "vessel_effect_with_ice": 5,
    }
    if comparisons.groupby("comparison").size().to_dict() != expected_counts:
        raise ValueError("Unexpected August-October contrast inventory")
    if comparisons[list(required_comparison)].isna().any().any():
        raise ValueError("Missing values in August-October contrasts")

    if len(routes) != 28 or set(routes["successful_path_count"]) != {1600}:
        raise ValueError("Expected 28 complete August-October scenario-year ensembles")
    if routes["failed_path_count"].sum() != 0:
        raise ValueError("A Phase 8Q route ensemble contains failed paths")

    required_observed = {
        "scenario",
        "scenario_label",
        "record_count",
        "mean_effort_footprint_percentile",
        "percent_records_top5pct",
        "median_distance_to_frequent_corridor_km",
    }
    for label, frame in [("annual", annual), ("available", available), ("common", common)]:
        missing = required_observed.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} observed-use fields: {sorted(missing)}")

    if set(common["scenario"]) != set(SCENARIO_ORDER) or set(common["record_count"]) != {1131}:
        raise ValueError("Common-period comparison must contain 1,131 records for all four scenarios")
    full = available.loc[available["scenario"].isin(["prey_only", "prey_ice"])].copy()
    if set(full["record_count"]) != {1843} or set(full["first_year"]) != {2011} or set(full["last_year"]) != {2019}:
        raise ValueError("Full-period comparison must contain 1,843 records from 2011-2019")
    return comparisons, routes, annual, full, common


def export_corridor_source(comparisons: pd.DataFrame) -> Path:
    source = comparisons[
        [
            "target_year",
            "comparison",
            "scenario_a",
            "scenario_b",
            "mean_cost_percent_change",
            "frequent_area_b_minus_a",
            "jaccard_freq_gt_0_05",
            "centroid_shift_freq_gt_0_05_km",
        ]
    ].copy()
    source.insert(2, "contrast_label", source["comparison"].map(lambda key: CONTRASTS[key]["label"]))
    source.insert(3, "factor_added", source["comparison"].map(lambda key: CONTRASTS[key]["factor"]))
    source["analysis_window"] = "August-October"
    source["frequent_corridor_definition"] = "F_s > 0.05"
    source["successful_paths_per_scenario_year"] = 1600
    source = source.sort_values(["comparison", "target_year"], kind="mergesort")
    path = SOURCE_DIR / "gray_whale_si_d1_corridor_contrasts_source_data_v1.csv"
    source.to_csv(path, index=False, float_format="%.6f")
    return path


def build_observed_source(
    annual: pd.DataFrame,
    full: pd.DataFrame,
    common: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scenario",
        "scenario_label",
        "record_count",
        "mean_effort_footprint_percentile",
        "percent_records_top5pct",
        "median_distance_to_frequent_corridor_km",
    ]
    pieces: list[pd.DataFrame] = []

    full_annual = annual.loc[
        annual["target_year"].between(2011, 2019)
        & annual["scenario"].isin(["prey_only", "prey_ice"])
    ].copy()
    full_annual["display_period"] = "2011-2019"
    full_annual["summary_type"] = "annual"
    pieces.append(full_annual)

    full_pooled = full[columns].copy()
    full_pooled["target_year"] = np.nan
    full_pooled["strict_effort_background_cell_count"] = np.nan
    full_pooled["display_period"] = "2011-2019"
    full_pooled["summary_type"] = "pooled"
    pieces.append(full_pooled)

    common_annual = annual.loc[
        annual["target_year"].between(2015, 2019)
        & annual["scenario"].isin(SCENARIO_ORDER)
    ].copy()
    common_annual["display_period"] = "2015-2019"
    common_annual["summary_type"] = "annual"
    pieces.append(common_annual)

    common_pooled = common[columns].copy()
    common_pooled["target_year"] = np.nan
    common_pooled["strict_effort_background_cell_count"] = np.nan
    common_pooled["display_period"] = "2015-2019"
    common_pooled["summary_type"] = "pooled"
    pieces.append(common_pooled)

    source = pd.concat(pieces, ignore_index=True, sort=False)
    source["analysis_window"] = "August-October"
    source["unit_of_observation"] = "retained ASAMM sighting record"
    source = source[
        [
            "display_period",
            "summary_type",
            "target_year",
            "scenario",
            "scenario_label",
            "record_count",
            "strict_effort_background_cell_count",
            "mean_effort_footprint_percentile",
            "percent_records_top5pct",
            "median_distance_to_frequent_corridor_km",
            "analysis_window",
            "unit_of_observation",
        ]
    ].sort_values(["display_period", "summary_type", "scenario", "target_year"], na_position="last")
    path = SOURCE_DIR / "gray_whale_si_d1_observed_use_source_data_v1.csv"
    source.to_csv(path, index=False, float_format="%.6f")
    return source


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#202427",
        clip_on=False,
    )


def format_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E1E5E7", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", width=0.6, length=2.3, pad=1.8)
    ax.spines["left"].set_color("#676D70")
    ax.spines["bottom"].set_color("#676D70")


def plot_corridor_panel(
    ax: plt.Axes,
    comparisons: pd.DataFrame,
    column: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    reference: float,
    panel: str,
) -> None:
    for contrast in CONTRAST_ORDER:
        style = CONTRASTS[contrast]
        selected = comparisons.loc[comparisons["comparison"].eq(contrast)].sort_values("target_year")
        x = selected["target_year"].to_numpy(dtype=float) + float(style["offset"])
        y = selected[column].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=str(style["color"]),
            linestyle=style["linestyle"],
            marker=str(style["marker"]),
            markersize=3.7,
            markerfacecolor=str(style["color"]) if bool(style["filled"]) else "white",
            markeredgecolor=str(style["color"]),
            markeredgewidth=0.8,
            zorder=4,
        )
    ax.axhline(reference, color="#9DA3A6", linewidth=0.65, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(2010.55, 2019.45)
    ax.set_xticks([2011, 2013, 2015, 2017, 2019])
    ax.set_xticklabels(["2011", "2013", "2015", "2017", "2019"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    format_axis(ax)
    panel_label(ax, panel)


def plot_observed_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    column: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    reference: float | None,
    panel: str,
) -> None:
    period_specs = [
        ("2011-2019", 0.0, ["prey_only", "prey_ice"], {"prey_only": -0.085, "prey_ice": 0.085}),
        (
            "2015-2019",
            1.0,
            SCENARIO_ORDER,
            {"prey_only": -0.18, "prey_ice": -0.06, "prey_vessel": 0.06, "prey_ice_vessel": 0.18},
        ),
    ]
    for period, center, scenarios, offsets in period_specs:
        period_data = source.loc[source["display_period"].eq(period)]
        for scenario in scenarios:
            annual = period_data.loc[
                period_data["summary_type"].eq("annual") & period_data["scenario"].eq(scenario),
                column,
            ].dropna()
            pooled = period_data.loc[
                period_data["summary_type"].eq("pooled") & period_data["scenario"].eq(scenario),
                column,
            ]
            if annual.empty or len(pooled) != 1:
                raise ValueError(f"Incomplete observed-use display data for {period}, {scenario}, {column}")
            y = float(pooled.iloc[0])
            low = float(annual.min())
            high = float(annual.max())
            x = center + offsets[scenario]
            ax.errorbar(
                x,
                y,
                yerr=np.array([[y - low], [high - y]]),
                fmt=SCENARIO_MARKERS[scenario],
                markersize=5.0,
                markerfacecolor=SCENARIO_COLORS[scenario],
                markeredgecolor="white",
                markeredgewidth=0.55,
                color=SCENARIO_COLORS[scenario],
                ecolor=SCENARIO_COLORS[scenario],
                elinewidth=1.0,
                capsize=2.3,
                capthick=0.8,
                zorder=5,
            )
    if reference is not None:
        ax.axhline(reference, color="#9DA3A6", linewidth=0.65, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2011-2019\nn = 1,843", "2015-2019\nn = 1,131"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    format_axis(ax)
    panel_label(ax, panel)


def save_figure(fig: plt.Figure) -> dict[str, Path]:
    paths = {extension: CANDIDATE_DIR / f"{STEM}.{extension}" for extension in ["pdf", "png", "svg", "tiff"]}
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths["png"], dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def build_figure(comparisons: pd.DataFrame, observed_source: pd.DataFrame) -> dict[str, Path]:
    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        12,
        left=0.075,
        right=0.995,
        top=0.835,
        bottom=0.18,
        wspace=2.65,
        hspace=0.68,
        height_ratios=[1.0, 1.0],
    )
    top_axes = [fig.add_subplot(grid[0, start : start + 3]) for start in [0, 3, 6, 9]]
    bottom_axes = [fig.add_subplot(grid[1, start : start + 4]) for start in [0, 4, 8]]

    for index, (ax, metric) in enumerate(zip(top_axes, CORRIDOR_METRICS)):
        plot_corridor_panel(ax, comparisons, *metric, chr(ord("A") + index))
    for index, (ax, metric) in enumerate(zip(bottom_axes, OBSERVED_METRICS)):
        plot_observed_panel(ax, observed_source, *metric, chr(ord("E") + index))

    contrast_handles = []
    for contrast in CONTRAST_ORDER:
        style = CONTRASTS[contrast]
        contrast_handles.append(
            Line2D(
                [0],
                [0],
                color=str(style["color"]),
                linestyle=style["linestyle"],
                marker=str(style["marker"]),
                markersize=4.0,
                markerfacecolor=str(style["color"]) if bool(style["filled"]) else "white",
                markeredgecolor=str(style["color"]),
                markeredgewidth=0.8,
                label=str(style["label"]),
            )
        )
    fig.legend(
        handles=contrast_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.45,
        columnspacing=1.10,
        fontsize=5.9,
    )

    scenario_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=SCENARIO_MARKERS[scenario],
            markersize=5.0,
            markerfacecolor=SCENARIO_COLORS[scenario],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIO_ORDER
    ]
    fig.legend(
        handles=scenario_handles,
        loc="lower center",
        bbox_to_anchor=(0.535, 0.025),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.50,
        fontsize=6.2,
    )
    return save_figure(fig)


def image_qa(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        nonwhite = np.any(rgb < 248, axis=2)
        return {
            "width_px": int(image.width),
            "height_px": int(image.height),
            "nonwhite_fraction": float(nonwhite.mean()),
            "channel_standard_deviation": float(rgb.std()),
            "nonblank": bool(nonwhite.mean() > 0.01 and rgb.std() > 2.0),
        }


def write_qa(
    comparisons: pd.DataFrame,
    observed_source: pd.DataFrame,
    paths: dict[str, Path],
    source_paths: list[Path],
) -> Path:
    ice = comparisons.loc[comparisons["comparison"].str.startswith("ice_effect")]
    vessel = comparisons.loc[comparisons["comparison"].str.startswith("vessel_effect")]
    png_qa = image_qa(paths["png"])
    tiff_qa = image_qa(paths["tiff"])
    if not png_qa["nonblank"] or not tiff_qa["nonblank"]:
        raise RuntimeError("D1 raster output failed nonblank QA")

    common_pooled = observed_source.loc[
        observed_source["display_period"].eq("2015-2019")
        & observed_source["summary_type"].eq("pooled")
    ]
    qa = {
        "figure": "D1 August-October corridor and observed-use response candidate v1",
        "status": "rendered",
        "manuscript_modified": False,
        "approved_assets_modified": False,
        "phase8q_outputs_modified": False,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                COMPARISON_PATH,
                ROUTE_PATH,
                ANNUAL_OBSERVED_PATH,
                AVAILABLE_OBSERVED_PATH,
                COMMON_OBSERVED_PATH,
            ]
        },
        "source_data": [str(path.relative_to(ROOT)) for path in source_paths],
        "outputs": {extension: str(path.relative_to(ROOT)) for extension, path in paths.items()},
        "output_sha256": {extension: sha256(path) for extension, path in paths.items()},
        "render_qa": {"png": png_qa, "tiff": tiff_qa},
        "data_checks": {
            "scenario_year_ensembles": 28,
            "successful_paths": 44800,
            "failed_paths": 0,
            "max_ice_cost_change_percent": float(ice["mean_cost_percent_change"].max()),
            "min_ice_jaccard": float(ice["jaccard_freq_gt_0_05"].min()),
            "max_ice_centroid_displacement_km": float(ice["centroid_shift_freq_gt_0_05_km"].max()),
            "vessel_cost_change_percent_range": [
                float(vessel["mean_cost_percent_change"].min()),
                float(vessel["mean_cost_percent_change"].max()),
            ],
            "vessel_area_change_cells_range": [
                int(vessel["frequent_area_b_minus_a"].min()),
                int(vessel["frequent_area_b_minus_a"].max()),
            ],
            "vessel_jaccard_range": [
                float(vessel["jaccard_freq_gt_0_05"].min()),
                float(vessel["jaccard_freq_gt_0_05"].max()),
            ],
            "vessel_centroid_displacement_km_range": [
                float(vessel["centroid_shift_freq_gt_0_05_km"].min()),
                float(vessel["centroid_shift_freq_gt_0_05_km"].max()),
            ],
            "full_period_sightings": 1843,
            "common_period_sightings_each_scenario": 1131,
            "common_period_pooled_metrics": common_pooled[
                [
                    "scenario",
                    "mean_effort_footprint_percentile",
                    "percent_records_top5pct",
                    "median_distance_to_frequent_corridor_km",
                ]
            ].to_dict(orient="records"),
        },
        "display_notes": {
            "panels_A_to_D": "Annual layer-added contrasts; small horizontal offsets reveal otherwise coincident paired traces.",
            "panels_E_to_G": "Symbols are pooled estimates; vertical intervals span annual values and are not confidence intervals.",
            "interpretation": "Observed-use metrics are effort-conditioned correspondence diagnostics, not abundance or movement-direction estimates.",
        },
    }
    path = QA_DIR / "d1_aug_oct_corridor_observed_use_qa_v1.json"
    path.write_text(json.dumps(qa, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_metadata(paths: dict[str, Path], source_paths: list[Path], qa_path: Path) -> None:
    text = f"""# D1 August-October corridor and observed-use response

- Output: rendered figure and numerical sources
- Scientific role: pair the annual August-October ASAMM map with a quantitative summary of late-season corridor response and observed-use correspondence.
- Core result: late-season sea ice produced negligible corridor change, whereas vessel conditioning strongly reorganized modeled corridors without a consistent observed-use winner.
- Scope: existing Phase 8Q results only; no corridor experiment was rerun.
- Candidate assets: `{paths['pdf'].relative_to(ROOT)}`, `{paths['png'].relative_to(ROOT)}`, `{paths['svg'].relative_to(ROOT)}`, `{paths['tiff'].relative_to(ROOT)}`
- Source data: `{source_paths[0].relative_to(ROOT)}`, `{source_paths[1].relative_to(ROOT)}`
- QA record: `{qa_path.relative_to(ROOT)}`
- Main-text anchor: Results subsection on August-October late-season connectivity.
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    configure_style()
    comparisons, _, annual, full, common = load_inputs()
    corridor_source_path = export_corridor_source(comparisons)
    observed_source = build_observed_source(annual, full, common)
    observed_source_path = SOURCE_DIR / "gray_whale_si_d1_observed_use_source_data_v1.csv"
    paths = build_figure(comparisons, observed_source)
    plt.close("all")
    qa_path = write_qa(
        comparisons,
        observed_source,
        paths,
        [corridor_source_path, observed_source_path],
    )
    write_metadata(paths, [corridor_source_path, observed_source_path], qa_path)
    print(f"Wrote D1 candidate to {OUT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
