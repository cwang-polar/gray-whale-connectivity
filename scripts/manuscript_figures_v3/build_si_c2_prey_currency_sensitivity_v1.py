#!/usr/bin/env python3
"""Build the C2 four-scenario benthic-prey currency sensitivity figure."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-c2")

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
    / "c2_prey_currency_sensitivity_v1"
)
CANDIDATE_OUT = OUT_ROOT / "candidates"
SOURCE_OUT = OUT_ROOT / "source_data"
QA_OUT = OUT_ROOT / "qa"

SCENARIOS = {
    "prey": {
        "label": "Prey",
        "path": ROOT / "outputs/lcp_v2/prey_only_brian_C3_route_summary_v2.csv",
        "years": list(range(2011, 2020)),
    },
    "prey_ice": {
        "label": "Prey + ice",
        "path": ROOT / "outputs/lcp_v2/prey_ice_brian_C3_route_summary_v2.csv",
        "years": list(range(2011, 2020)),
    },
    "prey_vessel": {
        "label": "Prey + vessel",
        "path": ROOT
        / "outputs/prey_vessel_phase8k/currency_sensitivity_v1/lcp/"
        "prey_vessel_brian_C3_route_summary_currency_sensitivity_v1.csv",
        "years": list(range(2015, 2020)),
    },
    "prey_ice_vessel": {
        "label": "Prey + ice + vessel",
        "path": ROOT / "outputs/lcp_v2/prey_ice_ship_brian_C3_route_summary_v2.csv",
        "years": list(range(2015, 2020)),
    },
}

VARIABLES = {
    "total_kJm2_frozen": {
        "label": "Energy density (primary)",
        "color": "#183A52",
        "marker": "^",
    },
    "total_gww": {
        "label": "Wet-weight biomass",
        "color": "#2F7F8F",
        "marker": "o",
    },
    "total_gC": {
        "label": "Carbon biomass",
        "color": "#B36A2E",
        "marker": "s",
    },
}

METRICS = {
    "mean_path_cost": "Mean modeled path cost",
    "corridor_area_cells_freq_gt_0_05": "Frequent-corridor area\n(number of cells)",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7.3,
            "axes.labelsize": 7,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
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


def load_data() -> pd.DataFrame:
    frames = []
    for scenario, specification in SCENARIOS.items():
        path = specification["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "prey_variable",
            "target_year",
            "successful_path_count",
            "failed_path_count",
            *METRICS,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[
            frame["prey_variable"].isin(VARIABLES)
            & frame["target_year"].isin(specification["years"])
        ].copy()
        expected = pd.MultiIndex.from_product(
            [list(VARIABLES), specification["years"]],
            names=["prey_variable", "target_year"],
        )
        observed = pd.MultiIndex.from_frame(frame[["prey_variable", "target_year"]])
        if len(frame) != len(expected) or set(observed) != set(expected):
            missing_pairs = sorted(set(expected).difference(observed))
            raise ValueError(f"{scenario} has incomplete variable-year coverage: {missing_pairs}")
        if not frame["successful_path_count"].eq(1600).all() or not frame["failed_path_count"].eq(0).all():
            raise ValueError(f"{scenario} contains incomplete least-cost-path ensembles")
        frame["scenario"] = scenario
        frame["scenario_label"] = specification["label"]
        frame["prey_currency_label"] = frame["prey_variable"].map(
            {key: value["label"] for key, value in VARIABLES.items()}
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[
        [
            "scenario",
            "scenario_label",
            "target_year",
            "prey_variable",
            "prey_currency_label",
            "mean_path_cost",
            "corridor_area_cells_freq_gt_0_05",
            "successful_path_count",
            "failed_path_count",
        ]
    ].sort_values(["scenario", "target_year", "prey_variable"])
    if len(data) != 84:
        raise ValueError(f"Expected 84 source-data rows, found {len(data)}")
    return data


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.34,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def save_bundle(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    return outputs


def write_figure_contract() -> None:
    contract = """# C2 figure contract

- Core conclusion: the temporal pattern and broad magnitude of modeled corridor summaries can be checked against two biomass-based alternatives to the primary amphipod energy-density surface across all four scenario families.
- Evidence chain: Panel A shows mean modeled path cost; Panel B shows the number of frequent-corridor cells; columns facet the four scenario families; line identity is prey currency.
- Archetype: quantitative grid.
- Backend: Python/matplotlib in the mypy10 environment.
- Export contract: 7.2-inch-wide PDF and SVG with editable text, plus 600-dpi PNG and TIFF; source data and input provenance retained.
- Interpretation: vessel scenarios have a shorter 2015-2019 period; biomass currencies are sensitivity representations rather than additional prey taxa; the figure must not imply that the three currencies are independent datasets.
"""
    (OUT_ROOT / "figure_contract.md").write_text(contract, encoding="utf-8")


def build_figure(*, prepared=None) -> tuple[list[Path], pd.DataFrame]:
    configure_style()
    for directory in (CANDIDATE_OUT, SOURCE_OUT, QA_OUT):
        directory.mkdir(parents=True, exist_ok=True)
    if prepared is None:
        write_figure_contract()
        data = load_data()
        data.to_csv(SOURCE_OUT / "c2_prey_currency_sensitivity_source_data_v1.csv", index=False)
        provenance = pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    "input_file": str(specification["path"].relative_to(ROOT)),
                    "sha256": sha256(specification["path"]),
                }
                for scenario, specification in SCENARIOS.items()
            ]
        )
        provenance.to_csv(OUT_ROOT / "input_provenance_manifest_v1.csv", index=False)
    else:
        data = prepared

    scenario_order = list(SCENARIOS)
    metric_order = list(METRICS)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.55), sharey="row")
    fig.subplots_adjust(
        left=0.082,
        right=0.992,
        bottom=0.195,
        top=0.905,
        wspace=0.15,
        hspace=0.25,
    )

    path_min = float(data["mean_path_cost"].min())
    path_max = float(data["mean_path_cost"].max())
    area_min = float(data["corridor_area_cells_freq_gt_0_05"].min())
    area_max = float(data["corridor_area_cells_freq_gt_0_05"].max())
    limits = {
        "mean_path_cost": (
            np.floor(path_min / 100.0) * 100.0 - 50.0,
            np.ceil(path_max / 100.0) * 100.0 + 50.0,
        ),
        "corridor_area_cells_freq_gt_0_05": (
            np.floor(area_min / 25.0) * 25.0 - 12.5,
            np.ceil(area_max / 25.0) * 25.0 + 12.5,
        ),
    }

    for row_index, metric in enumerate(metric_order):
        for column_index, scenario in enumerate(scenario_order):
            ax = axes[row_index, column_index]
            subset = data.loc[data["scenario"].eq(scenario)]
            for variable, style in VARIABLES.items():
                line = subset.loc[subset["prey_variable"].eq(variable)]
                ax.plot(
                    line["target_year"],
                    line[metric],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.3,
                    markeredgewidth=0.35,
                    linewidth=1.05,
                    solid_capstyle="round",
                    zorder=3,
                )

            years = SCENARIOS[scenario]["years"]
            ax.set_xlim(min(years) - 0.4, max(years) + 0.4)
            if len(years) == 9:
                ticks = [2011, 2013, 2015, 2017, 2019]
            else:
                ticks = [2015, 2017, 2019]
            ax.set_xticks(ticks)
            if row_index == 0:
                ax.set_xticklabels([])
                ax.set_title(SCENARIOS[scenario]["label"], pad=4.5)
            else:
                ax.set_xticklabels([str(year) for year in ticks], rotation=45, ha="right")
            ax.set_ylim(*limits[metric])
            ax.grid(axis="y", color="#E4E6E7", linewidth=0.45, zorder=0)
            ax.tick_params(length=2.2, pad=1.8)
            if column_index == 0:
                ax.set_ylabel(METRICS[metric], labelpad=6)
                panel_label(ax, "A" if row_index == 0 else "B")
            else:
                ax.tick_params(labelleft=False)

    fig.supxlabel("Target year", x=0.535, y=0.095, fontsize=7)
    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            markersize=3.7,
            markeredgewidth=0.35,
            linewidth=1.15,
            label=style["label"],
        )
        for style in VARIABLES.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.535, 0.012),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    stem = CANDIDATE_OUT / "gray_whale_si_c2_prey_currency_sensitivity_candidate_v1"
    outputs = save_bundle(fig, stem)
    plt.close(fig)
    return outputs, data


def write_qa(outputs: list[Path], data: pd.DataFrame) -> None:
    png = next(path for path in outputs if path.suffix == ".png")
    tiff = next(path for path in outputs if path.suffix == ".tiff")
    with Image.open(png) as image:
        grayscale = np.asarray(image.convert("L"))
        png_size = [int(image.width), int(image.height)]
        nonblank = bool(grayscale.std() > 0.5)
    with Image.open(tiff) as image:
        tiff_size = [int(image.width), int(image.height)]

    scenario_rows = {
        scenario: int(data["scenario"].eq(scenario).sum()) for scenario in SCENARIOS
    }
    payload = {
        "backend": "Python/matplotlib in mypy10",
        "source_rows": int(len(data)),
        "rows_by_scenario": scenario_rows,
        "prey_variables": list(VARIABLES),
        "successful_path_count_unique": sorted(
            data["successful_path_count"].astype(int).unique().tolist()
        ),
        "failed_path_count_total": int(data["failed_path_count"].sum()),
        "png_dimensions_px": png_size,
        "tiff_dimensions_px": tiff_size,
        "nonblank": nonblank,
        "existing_scientific_outputs_modified": False,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    (QA_OUT / "c2_prey_currency_sensitivity_qa_v1.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    outputs, data = build_figure()
    write_qa(outputs, data)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
