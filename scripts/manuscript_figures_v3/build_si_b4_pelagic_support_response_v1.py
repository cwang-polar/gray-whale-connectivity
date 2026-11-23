#!/usr/bin/env python3
"""Build the B4 observation-informed pelagic-support response SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-b4")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "b4_pelagic_support_response_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

PHASE8S_FIGURES = ROOT / "outputs" / "pelagic_support_phase8s" / "figures"
ROUTE_SOURCE = PHASE8S_FIGURES / "phase8s_corridor_response_source_data_v1.csv"
BOOTSTRAP_SOURCE = (
    PHASE8S_FIGURES / "phase8s_bootstrap_average_intervals_source_data_v1.csv"
)
ASAMM_SOURCE = PHASE8S_FIGURES / "phase8s_asamm_response_source_data_v1.csv"

SCENARIO_ORDER = ["prey", "prey_ice", "prey_vessel", "prey_ice_vessel"]
SCENARIO_LABELS = {
    "prey": "Prey-only",
    "prey_ice": "Prey + ice",
    "prey_vessel": "Prey + vessel",
    "prey_ice_vessel": "Prey + ice + vessel",
}
COLORS = {
    "prey": "#3F6FA3",
    "prey_ice": "#3F8A68",
    "prey_vessel": "#D48732",
    "prey_ice_vessel": "#8B5E83",
}
MARKERS = {"prey": "o", "prey_ice": "s", "prey_vessel": "^", "prey_ice_vessel": "D"}
LINESTYLES = {
    "prey": "-",
    "prey_ice": "-",
    "prey_vessel": (0, (4, 2)),
    "prey_ice_vessel": (0, (4, 2)),
}

BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
PRIMARY_BETA = 0.25
PRIMARY_INTERVAL_X_OFFSETS = {
    "prey": -0.045,
    "prey_ice": -0.015,
    "prey_vessel": 0.015,
    "prey_ice_vessel": 0.045,
}

TOP_METRICS = [
    {
        "column": "delta_mean_path_cost_percent",
        "ylabel": "Mean path-cost change\nfrom β = 0 (%)",
        "ylim": (-32, 2),
        "yticks": [-30, -20, -10, 0],
        "reference": 0,
    },
    {
        "column": "jaccard_with_baseline_frequent_cells",
        "ylabel": "Jaccard overlap with\nβ = 0 frequent corridors",
        "ylim": (0, 1.03),
        "yticks": [0, 0.25, 0.50, 0.75, 1.00],
        "reference": 1,
    },
    {
        "column": "frequent_centroid_shift_from_baseline_km",
        "ylabel": "Corridor-centroid displacement\nfrom β = 0 (km)",
        "ylim": (0, 52),
        "yticks": [0, 10, 20, 30, 40, 50],
        "reference": 0,
    },
]

BOTTOM_METRICS = [
    {
        "column": "mean_effort_footprint_percentile",
        "ylabel": "Mean effort-footprint\npercentile",
        "ylim": (50, 60),
        "yticks": [50, 55, 60],
        "reference": 50,
    },
    {
        "column": "percent_records_top5pct",
        "ylabel": "Sightings in annual\ntop 5% (%)",
        "ylim": (0, 20),
        "yticks": [0, 5, 10, 15, 20],
        "reference": None,
    },
    {
        "column": "median_distance_to_frequent_corridor_km",
        "ylabel": "Median distance to\nfrequent corridor (km)",
        "ylim": (0, 32),
        "yticks": [0, 10, 20, 30],
        "reference": None,
    },
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route = pd.read_csv(ROUTE_SOURCE)
    bootstrap = pd.read_csv(BOOTSTRAP_SOURCE)
    asamm = pd.read_csv(ASAMM_SOURCE)
    asamm = asamm.loc[asamm["combination_operator"].eq("complementary")].copy()

    expected_pairs = {(beta, scenario) for beta in BETAS for scenario in SCENARIO_ORDER}
    route_pairs = set(zip(route["pelagic_strength_beta"], route["scenario"]))
    asamm_pairs = set(zip(asamm["pelagic_strength_beta"], asamm["scenario"]))
    if route_pairs != expected_pairs:
        raise ValueError("Route-response source lacks the expected beta-scenario grid")
    if asamm_pairs != expected_pairs:
        raise ValueError("ASAMM-response source lacks the expected beta-scenario grid")
    expected_bootstrap = {
        (scenario, metric["column"])
        for scenario in SCENARIO_ORDER
        for metric in TOP_METRICS
    }
    bootstrap_pairs = set(zip(bootstrap["scenario"], bootstrap["metric"]))
    if bootstrap_pairs != expected_bootstrap:
        raise ValueError("Bootstrap source lacks an expected scenario-metric interval")

    baseline = route.loc[route["pelagic_strength_beta"].eq(0)]
    if not np.allclose(baseline["delta_mean_path_cost_percent_mean"], 0, atol=1e-10):
        raise ValueError("beta=0 path-cost changes do not reproduce zero")
    if not np.allclose(baseline["jaccard_with_baseline_frequent_cells_mean"], 1):
        raise ValueError("beta=0 Jaccard values do not reproduce one")
    if not np.allclose(baseline["frequent_centroid_shift_from_baseline_km_mean"], 0):
        raise ValueError("beta=0 centroid shifts do not reproduce zero")
    if set(asamm["sighting_record_count"].unique()) != {323}:
        raise ValueError("ASAMM comparisons do not all use 323 records")
    if set(asamm["year_count"].unique()) != {2}:
        raise ValueError("ASAMM comparisons do not all use two years")

    for metric in TOP_METRICS:
        column = metric["column"]
        central = route.loc[route["pelagic_strength_beta"].eq(PRIMARY_BETA), ["scenario", f"{column}_mean"]]
        central = central.rename(columns={f"{column}_mean": "central"})
        intervals = bootstrap.loc[bootstrap["metric"].eq(column), ["scenario", "q05", "q95"]]
        audit = central.merge(intervals, on="scenario", validate="one_to_one")
        if not ((audit["central"] >= audit["q05"]) & (audit["central"] <= audit["q95"])).all():
            raise ValueError(f"Central {column} value falls outside its bootstrap interval")
    return route, bootstrap, asamm


def style_axis(ax: plt.Axes, metric: dict[str, object], show_xlabel: bool) -> None:
    ax.set_ylim(*metric["ylim"])
    ax.set_yticks(metric["yticks"])
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(BETAS)
    ax.set_xticklabels(["0", "0.10", "0.25", "0.50", "1.0"])
    if show_xlabel:
        ax.set_xlabel("Pelagic-support strength, β")
    else:
        ax.tick_params(labelbottom=False)
    ax.grid(axis="y", color="#E1E5E7", linewidth=0.50, zorder=0)
    ax.axvline(
        PRIMARY_BETA,
        color="#9BA2A6",
        linewidth=0.70,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    reference = metric["reference"]
    if reference is not None:
        ax.axhline(
            float(reference),
            color="#7D8386",
            linewidth=0.65,
            linestyle=(0, (3, 2)),
            zorder=1,
        )


def plot_top_panel(
    ax: plt.Axes,
    route: pd.DataFrame,
    bootstrap: pd.DataFrame,
    metric: dict[str, object],
) -> None:
    column = str(metric["column"])
    for scenario in SCENARIO_ORDER:
        subset = route.loc[route["scenario"].eq(scenario)].sort_values(
            "pelagic_strength_beta"
        )
        x = subset["pelagic_strength_beta"].to_numpy(dtype=float)
        mean = subset[f"{column}_mean"].to_numpy(dtype=float)
        lower = subset[f"{column}_min"].to_numpy(dtype=float)
        upper = subset[f"{column}_max"].to_numpy(dtype=float)
        mean[np.isclose(mean, 0, atol=1e-10)] = 0
        lower[np.isclose(lower, 0, atol=1e-10)] = 0
        upper[np.isclose(upper, 0, atol=1e-10)] = 0

        ax.fill_between(
            x,
            lower,
            upper,
            color=COLORS[scenario],
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            marker=MARKERS[scenario],
            markerfacecolor="white",
            markeredgecolor=COLORS[scenario],
            markeredgewidth=0.8,
            markersize=4.0,
            linewidth=1.15,
            zorder=3,
        )

        central = float(
            subset.loc[
                subset["pelagic_strength_beta"].eq(PRIMARY_BETA), f"{column}_mean"
            ].iloc[0]
        )
        interval = bootstrap.loc[
            bootstrap["scenario"].eq(scenario) & bootstrap["metric"].eq(column)
        ].iloc[0]
        ax.errorbar(
            PRIMARY_BETA + PRIMARY_INTERVAL_X_OFFSETS[scenario],
            central,
            yerr=[[central - float(interval["q05"])], [float(interval["q95"]) - central]],
            color=COLORS[scenario],
            linewidth=0.85,
            capsize=2.1,
            capthick=0.85,
            fmt="none",
            zorder=4,
        )
    style_axis(ax, metric, show_xlabel=False)


def plot_bottom_panel(
    ax: plt.Axes,
    asamm: pd.DataFrame,
    metric: dict[str, object],
) -> None:
    column = str(metric["column"])
    for scenario in SCENARIO_ORDER:
        subset = asamm.loc[asamm["scenario"].eq(scenario)].sort_values(
            "pelagic_strength_beta"
        )
        ax.plot(
            subset["pelagic_strength_beta"],
            subset[column],
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            marker=MARKERS[scenario],
            markerfacecolor="white",
            markeredgecolor=COLORS[scenario],
            markeredgewidth=0.8,
            markersize=4.0,
            linewidth=1.15,
            zorder=3,
        )
    style_axis(ax, metric, show_xlabel=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.20,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_source_data(
    route: pd.DataFrame,
    bootstrap: pd.DataFrame,
    asamm: pd.DataFrame,
) -> None:
    route.to_csv(SOURCE_OUT / "b4_pelagic_corridor_response_source_data_v1.csv", index=False)
    bootstrap.to_csv(
        SOURCE_OUT / "b4_pelagic_bootstrap_intervals_source_data_v1.csv", index=False
    )
    asamm.to_csv(SOURCE_OUT / "b4_pelagic_asamm_response_source_data_v1.csv", index=False)
    rows = [
        {
            "relative_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (ROUTE_SOURCE, BOOTSTRAP_SOURCE, ASAMM_SOURCE)
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "input_provenance_manifest_v1.csv", index=False)


def write_figure_contract() -> None:
    contract = """# Figure contract: B4 pelagic-support corridor and ASAMM response

## Core conclusion

Observation-informed relative pelagic support reduced modeled path cost and reorganized frequent-corridor geography in 2017 and 2018, but increasing support did not consistently improve all three measures of correspondence with ASAMM sightings.

## Evidence chain

- Panels A-C quantify cost and geographic change relative to the same scenario at beta = 0.
- Pale bands show the two-year range, and beta = 0.25 intervals propagate spatial-block uncertainty.
- Panels D-F use the same 323 June-July sightings to test whether geometric changes improve three complementary observed-use metrics.
- Alternative utility-combination operators remain tabular sensitivity evidence and are not added as competing visual layers.

## Archetype and role

- Archetype: quantitative response grid.
- Manuscript role: response evidence for the bounded 2017-2018 alternative-prey sensitivity.

## Interpretation

- Beta is a prespecified sensitivity strength, not an estimated energetic conversion.
- Scenario labels identify the baseline landscape to which relative pelagic support is added.
- Jaccard and centroid displacement are calculated against beta = 0 within scenario.
- The ASAMM panels measure observed-use correspondence, not pelagic feeding, abundance or behavior.

## Export contract

- Python/matplotlib only, using the mypy10 environment.
- Fixed 7.2 x 4.55 inch double-column canvas.
- PDF and SVG with editable text; 300-dpi PNG and 600-dpi LZW-compressed TIFF.
- Candidate outputs remain isolated from Phase 8S source figures and manuscript files.
"""
    (OUT_ROOT / "figure_contract.md").write_text(contract, encoding="utf-8")


def build_figure(*, prepared=None) -> list[Path]:
    configure_style()
    for directory in (CANDIDATE_OUT, SOURCE_OUT, QA_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    if prepared is None:
        route, bootstrap, asamm = load_inputs()
        save_source_data(route, bootstrap, asamm)
        write_figure_contract()
    else:
        route, bootstrap, asamm = prepared

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), facecolor="white")
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.145,
        top=0.875,
        wspace=0.43,
        hspace=0.29,
    )

    for column, metric in enumerate(TOP_METRICS):
        plot_top_panel(axes[0, column], route, bootstrap, metric)
    for column, metric in enumerate(BOTTOM_METRICS):
        plot_bottom_panel(axes[1, column], asamm, metric)
    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(65 + index))

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            marker=MARKERS[scenario],
            markerfacecolor="white",
            markeredgecolor=COLORS[scenario],
            markeredgewidth=0.8,
            markersize=4.2,
            linewidth=1.2,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIO_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=4,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=6.4,
    )

    stem = CANDIDATE_OUT / "gray_whale_si_b4_pelagic_support_response_candidate_v1"
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
        png_size = list(image.size)
        extrema = image.convert("L").getextrema()
    with Image.open(outputs[3]) as image:
        tiff_size = list(image.size)

    qa = {
        "backend": "Python/matplotlib in mypy10",
        "years": [2017, 2018],
        "central_sighting_records": 323,
        "scenarios": SCENARIO_ORDER,
        "pelagic_strength_beta_values": BETAS,
        "primary_beta": PRIMARY_BETA,
        "combination_operator": "complementary",
        "top_panel_central_statistic": "arithmetic mean across 2017 and 2018",
        "top_panel_band": "minimum-maximum range across 2017 and 2018",
        "primary_beta_interval": "5th-95th percentile across 100 spatial-block bootstrap draws after averaging annual metrics within draw",
        "primary_beta_interval_display": "scenario intervals are horizontally offset around beta=0.25 for legibility; curve coordinates are unchanged",
        "png_dimensions_px": png_size,
        "tiff_dimensions_px": tiff_size,
        "png_grayscale_extrema": list(extrema),
        "nonblank": extrema[0] < extrema[1],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "source_figure_preserved": True,
        "corridor_models_rerun": False,
        "manuscript_or_word_modified": False,
    }
    (QA_OUT / "b4_pelagic_support_response_qa_v1.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    for output in build_figure():
        print(output)
