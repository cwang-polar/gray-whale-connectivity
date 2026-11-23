#!/usr/bin/env python3
"""Build the four-scenario consecutive-year corridor-stability figure."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-v3")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[2]
LCP_DIR = ROOT / "outputs" / "lcp_v2"
PREY_VESSEL_DIR = ROOT / "outputs" / "prey_vessel_phase8k" / "lcp"
OLD_CENTROID = (
    ROOT
    / "outputs"
    / "manuscript_figures"
    / "supplementary_figures"
    / "figure_s5_centroid_shift"
    / "gray_whale_supplementary_figure_s5_centroid_shift_source_data_v1.csv"
)
OLD_JACCARD = (
    ROOT
    / "outputs"
    / "manuscript_figures"
    / "supplementary_figures"
    / "figure_s6_interannual_jaccard"
    / "gray_whale_supplementary_figure_s6_interannual_jaccard_source_data_v1.csv"
)
OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "c3_consecutive_year_stability_v1"
)
CANDIDATE_DIR = OUT_ROOT / "candidates"
SOURCE_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"

FREQUENCY_THRESHOLD = 0.05
EXPECTED_GRID_CELLS = 2_839
STEM = "gray_whale_si_c3_consecutive_year_stability_candidate_v1"

SCENARIO_ORDER = ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIOS = {
    "prey_only": {
        "label": "Prey",
        "years": range(2011, 2020),
        "directory": LCP_DIR,
        "pattern": "prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "color": "#3B6EA8",
        "marker": "o",
        "linestyle": "-",
    },
    "prey_ice": {
        "label": "Prey + ice",
        "years": range(2011, 2020),
        "directory": LCP_DIR,
        "pattern": "prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "color": "#3A8D6D",
        "marker": "s",
        "linestyle": "-",
    },
    "prey_vessel": {
        "label": "Prey + vessel",
        "years": range(2015, 2020),
        "directory": PREY_VESSEL_DIR,
        "pattern": "prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "color": "#D9822B",
        "marker": "^",
        "linestyle": (0, (4, 2)),
    },
    "prey_ice_vessel": {
        "label": "Prey + ice + vessel",
        "years": range(2015, 2020),
        "directory": LCP_DIR,
        "pattern": "prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "color": "#A85A91",
        "marker": "D",
        "linestyle": (0, (4, 2)),
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_annual_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    required = ["grid_id", "x", "y", "lcp_frequency_normalized"]
    grid = pd.read_csv(path, usecols=required)
    if len(grid) != EXPECTED_GRID_CELLS:
        raise ValueError(f"Unexpected grid size in {path.name}: {len(grid)}")
    if grid[required].isna().any().any():
        raise ValueError(f"Missing required values in {path.name}")
    if grid["grid_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate grid identifiers in {path.name}")
    if not grid["lcp_frequency_normalized"].between(0, 1).all():
        raise ValueError(f"Corridor frequencies outside [0, 1] in {path.name}")
    grid["grid_id"] = grid["grid_id"].astype(str)
    return grid


def summarize_consecutive_years() -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for scenario in SCENARIO_ORDER:
        cfg = SCENARIOS[scenario]
        years = list(cfg["years"])
        annual: dict[int, pd.DataFrame] = {}
        for year in years:
            path = cfg["directory"] / cfg["pattern"].format(year=year)
            grid = load_annual_grid(path)
            frequent = grid.loc[
                grid["lcp_frequency_normalized"] > FREQUENCY_THRESHOLD,
                ["grid_id", "x", "y"],
            ].copy()
            if frequent.empty:
                raise ValueError(f"No frequent cells for {scenario}, {year}")
            annual[year] = frequent

        for from_year, to_year in zip(years[:-1], years[1:]):
            previous = annual[from_year]
            current = annual[to_year]
            previous_ids = set(previous["grid_id"])
            current_ids = set(current["grid_id"])
            shared = previous_ids & current_ids
            union = previous_ids | current_ids
            previous_x = float(previous["x"].mean())
            previous_y = float(previous["y"].mean())
            current_x = float(current["x"].mean())
            current_y = float(current["y"].mean())
            delta_x_km = (current_x - previous_x) / 1_000
            delta_y_km = (current_y - previous_y) / 1_000
            records.append(
                {
                    "scenario": scenario,
                    "scenario_label": cfg["label"],
                    "from_year": from_year,
                    "to_year": to_year,
                    "year_pair": f"{from_year}-{to_year}",
                    "previous_frequent_cells": len(previous_ids),
                    "current_frequent_cells": len(current_ids),
                    "shared_frequent_cells": len(shared),
                    "union_frequent_cells": len(union),
                    "centroid_delta_x_km": delta_x_km,
                    "centroid_delta_y_km": delta_y_km,
                    "centroid_displacement_km": float(np.hypot(delta_x_km, delta_y_km)),
                    "consecutive_year_jaccard": len(shared) / len(union),
                    "frequency_threshold": FREQUENCY_THRESHOLD,
                    "prey_representation": "amphipod energy density",
                }
            )
    result = pd.DataFrame(records)
    if len(result) != 24:
        raise ValueError(f"Expected 24 consecutive-year comparisons, found {len(result)}")
    return result


def verify_legacy_reproduction(result: pd.DataFrame) -> dict[str, float]:
    old_centroid = pd.read_csv(OLD_CENTROID)
    old_jaccard = pd.read_csv(OLD_JACCARD)
    label_to_key = {
        "prey-only": "prey_only",
        "prey+ice": "prey_ice",
        "prey+ice+vessel": "prey_ice_vessel",
    }
    old_centroid["scenario"] = old_centroid["scenario"].map(label_to_key)
    old_jaccard["scenario"] = old_jaccard["scenario"].map(label_to_key)

    centroid_check = result.merge(
        old_centroid[["scenario", "from_year", "to_year", "centroid_shift_km"]],
        on=["scenario", "from_year", "to_year"],
        how="inner",
        validate="one_to_one",
    )
    jaccard_check = result.merge(
        old_jaccard[["scenario", "from_year", "to_year", "jaccard_overlap"]],
        on=["scenario", "from_year", "to_year"],
        how="inner",
        validate="one_to_one",
    )
    if len(centroid_check) != 20 or len(jaccard_check) != 20:
        raise ValueError("Legacy comparison did not recover all 20 original rows")

    centroid_error = float(
        np.max(
            np.abs(
                centroid_check["centroid_displacement_km"]
                - centroid_check["centroid_shift_km"]
            )
        )
    )
    jaccard_error = float(
        np.max(
            np.abs(
                jaccard_check["consecutive_year_jaccard"]
                - jaccard_check["jaccard_overlap"]
            )
        )
    )
    if centroid_error > 1e-5 or jaccard_error > 1e-12:
        raise ValueError(
            "Recalculated metrics differ from legacy summaries: "
            f"centroid={centroid_error}, jaccard={jaccard_error}"
        )
    return {
        "legacy_rows_checked": 20,
        "max_centroid_error_km": centroid_error,
        "max_jaccard_error": jaccard_error,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.5, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def draw_figure(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharex=True)
    x_ticks = list(range(2012, 2020))
    x_labels = [f"{year - 1}\N{EN DASH}{str(year)[-2:]}" for year in x_ticks]

    for scenario in SCENARIO_ORDER:
        cfg = SCENARIOS[scenario]
        subset = data.loc[data["scenario"] == scenario].sort_values("to_year")
        common = {
            "color": cfg["color"],
            "marker": cfg["marker"],
            "linestyle": cfg["linestyle"],
            "markerfacecolor": cfg["color"],
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
            "label": cfg["label"],
            "zorder": 3,
        }
        axes[0].plot(subset["to_year"], subset["centroid_displacement_km"], **common)
        axes[1].plot(subset["to_year"], subset["consecutive_year_jaccard"], **common)

    axes[0].set_ylabel("Centroid displacement (km)")
    axes[1].set_ylabel("Jaccard overlap")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 0.60)
    axes[1].set_yticks(np.arange(0, 0.61, 0.1))

    for label, ax in zip(["A", "B"], axes):
        style_axis(ax)
        ax.set_xlim(2011.65, 2019.35)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Consecutive years")
        ax.text(
            -0.135,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.083, right=0.985, bottom=0.23, top=0.82, wspace=0.29)
    return fig


def save_figure(fig: plt.Figure) -> dict[str, Path]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": CANDIDATE_DIR / f"{STEM}.pdf",
        "png": CANDIDATE_DIR / f"{STEM}.png",
        "svg": CANDIDATE_DIR / f"{STEM}.svg",
        "tiff": CANDIDATE_DIR / f"{STEM}.tiff",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def image_nonblank(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(ImageStat.Stat(image.convert("L")).stddev[0] > 1.0)


def write_qa(paths: dict[str, Path], data: pd.DataFrame, legacy: dict[str, float]) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for file_type, path in paths.items():
        row: dict[str, object] = {
            "check": "export",
            "file_type": file_type,
            "file": path.name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        if file_type in {"png", "tiff"} and path.exists():
            with Image.open(path) as image:
                row.update(
                    {
                        "width_px": image.width,
                        "height_px": image.height,
                        "nonblank": image_nonblank(path),
                    }
                )
        rows.append(row)

    rows.extend(
        [
            {
                "check": "data_rows",
                "value": len(data),
                "expected": 24,
                "passed": len(data) == 24,
            },
            {
                "check": "jaccard_range",
                "value": f"{data['consecutive_year_jaccard'].min():.6f}-{data['consecutive_year_jaccard'].max():.6f}",
                "expected": "within 0-1",
                "passed": data["consecutive_year_jaccard"].between(0, 1).all(),
            },
            {
                "check": "centroid_nonnegative",
                "value": float(data["centroid_displacement_km"].min()),
                "expected": ">=0",
                "passed": bool((data["centroid_displacement_km"] >= 0).all()),
            },
            {
                "check": "legacy_rows_reproduced",
                "value": legacy["legacy_rows_checked"],
                "expected": 20,
                "passed": legacy["legacy_rows_checked"] == 20,
            },
            {
                "check": "legacy_centroid_max_error_km",
                "value": legacy["max_centroid_error_km"],
                "expected": "<=1e-5",
                "passed": legacy["max_centroid_error_km"] <= 1e-5,
            },
            {
                "check": "legacy_jaccard_max_error",
                "value": legacy["max_jaccard_error"],
                "expected": "<=1e-12",
                "passed": legacy["max_jaccard_error"] <= 1e-12,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(
        QA_DIR / "gray_whale_si_c3_consecutive_year_stability_qa_v1.csv", index=False
    )


def format_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def write_metadata(data: pd.DataFrame, legacy: dict[str, float]) -> None:
    summary = (
        data.groupby(["scenario", "scenario_label"], sort=False)
        .agg(
            comparisons=("to_year", "size"),
            centroid_min_km=("centroid_displacement_km", "min"),
            centroid_max_km=("centroid_displacement_km", "max"),
            jaccard_min=("consecutive_year_jaccard", "min"),
            jaccard_max=("consecutive_year_jaccard", "max"),
            jaccard_mean=("consecutive_year_jaccard", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Consecutive-year corridor stability candidate v1",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: year-to-year corridor stability differs among the four modeled landscapes and is not captured by centroid movement alone.",
        "- Evidence chain: panel A measures movement of the frequent-corridor center; panel B measures retention of the full frequent-cell set.",
        "- Archetype: two-panel quantitative comparison.",
        "- Backend: Python/matplotlib only.",
        "- Export: double-column PDF, SVG, 600-dpi PNG and LZW-compressed TIFF with editable vector text.",
        "",
        "## Computation",
        "",
        "Both metrics were recalculated directly from the current primary energy-density grid outputs. Frequent cells satisfy F_s > 0.05. Centroids are unweighted means of projected cell-center coordinates; displacement is Euclidean distance between consecutive-year centroids. Jaccard overlap is the intersection divided by the union of consecutive-year frequent-cell sets.",
        "",
        "## Numerical summary",
        "",
        format_markdown_table(summary),
        "",
        "## Reproduction check",
        "",
        f"The recalculation reproduced all {legacy['legacy_rows_checked']} rows from the three legacy scenarios. Maximum centroid-displacement error was {legacy['max_centroid_error_km']:.3g} km and maximum Jaccard error was {legacy['max_jaccard_error']:.3g}.",
        "",
        "## Interpretation",
        "",
        "- Centroid displacement reports movement magnitude, not movement direction.",
        "- Jaccard overlap and centroid displacement describe different spatial properties and need not covary.",
        "- Vessel scenarios begin with the 2015-2016 comparison because vessel-modified landscapes start in 2015.",
    ]
    (OUT_ROOT / "gray_whale_si_c3_consecutive_year_stability_metadata_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    data = summarize_consecutive_years()
    legacy = verify_legacy_reproduction(data)
    data.to_csv(
        SOURCE_DIR / "gray_whale_si_c3_consecutive_year_stability_source_data_v1.csv",
        index=False,
    )
    figure = draw_figure(data)
    paths = save_figure(figure)
    plt.close(figure)
    write_qa(paths, data, legacy)
    write_metadata(data, legacy)

    summary = (
        data.groupby("scenario_label", sort=False)
        .agg(
            n=("to_year", "size"),
            displacement_min=("centroid_displacement_km", "min"),
            displacement_max=("centroid_displacement_km", "max"),
            jaccard_min=("consecutive_year_jaccard", "min"),
            jaccard_max=("consecutive_year_jaccard", "max"),
            jaccard_mean=("consecutive_year_jaccard", "mean"),
        )
        .round(4)
    )
    print(summary.to_string())
    print(f"\nCandidate: {paths['pdf'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
