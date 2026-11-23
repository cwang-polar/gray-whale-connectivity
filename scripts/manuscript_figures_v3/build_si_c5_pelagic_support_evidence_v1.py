#!/usr/bin/env python3
"""Build the C5 pelagic-support observation and model-gate SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-c5")
sys.path.insert(0, str(ROOT / "scripts" / "manuscript_figures"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageStat
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

import build_figure1_cee_v0 as figure1_base


OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "c5_pelagic_support_evidence_v1"
)
CANDIDATE_DIR = OUT_ROOT / "candidates"
SOURCE_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"
STEM = "gray_whale_si_c5_pelagic_support_evidence_candidate_v1"

PHASE8S = ROOT / "outputs" / "pelagic_support_phase8s"
SURFACE_TEMPLATE = PHASE8S / "surfaces" / "phase8s_observation_informed_pelagic_support_target{year}.csv"
OBSERVATIONS_PATH = PHASE8S / "figures" / "phase8s_observation_points_source_data_v1.csv"
CV_PATH = PHASE8S / "tables" / "phase8s_spatial_block_cv_summary.csv"
COVERAGE_PATH = PHASE8S / "figures" / "phase8s_temporal_coverage_source_data_v1.csv"
SURFACE_QA_PATH = PHASE8S / "tables" / "phase8s_early_surface_qa.csv"
LAND_PATH = ROOT / "data" / "raw" / "prey" / "ne_10m_land" / "ne_10m_land.shp"

YEARS = [2017, 2018]
CONTEXT_YEARS = list(range(2011, 2020))
EXPECTED_OBSERVATIONS = {2017: 38, 2018: 28}
EXPECTED_DOMAIN_CELLS = {2017: 921, 2018: 695}
EXPECTED_DOMAIN_PERCENT = {2017: 32.441000352236706, 2018: 24.48045086297992}
EXPECTED_MODEL = {2017: "spatial", 2018: "spatial"}
MODEL_COLORS = {2017: "#3B6EA8", 2018: "#3A8D6D"}
DOMAIN_COLOR = "#148B83"
DBO_COLOR = figure1_base.PALETTE["dbo"]
DBO_LABEL_POSITIONS = {
    "DBO1": (-174.6, 62.8),
    "DBO2": (-169.2, 64.7),
    "DBO3": (-168.9, 67.7),
    "DBO4": (-162.5, 71.25),
    "DBO5": (-157.35, 71.45),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_observed_support(surfaces: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Read and scale observed points without drawing or fitting models."""
    observations = pd.read_csv(OBSERVATIONS_PATH, dtype={"grid_id": str})
    observations = observations.loc[observations["year"].isin(YEARS)].copy()
    scaled_pieces = []
    for year in YEARS:
        subset = observations.loc[observations["year"].eq(year)].copy()
        if len(subset) != EXPECTED_OBSERVATIONS[year] or subset["grid_id"].nunique() != len(subset):
            raise ValueError(f"Unexpected {year} observation inventory")
        lower_values = surfaces[year]["support_scale_lower_log_abundance_q05"].dropna().unique()
        upper_values = surfaces[year]["support_scale_upper_log_abundance_q95"].dropna().unique()
        if len(lower_values) != 1 or len(upper_values) != 1:
            raise ValueError(f"Nonunique support-scaling anchors for {year}")
        lower, upper = float(lower_values[0]), float(upper_values[0])
        subset["observed_relative_support"] = np.clip(
            (subset["response_log_abundance"] - lower) / (upper - lower), 0, 1
        )
        scaled_pieces.append(subset)
    observations = pd.concat(scaled_pieces, ignore_index=True)
    return observations


def load_inputs() -> tuple[
    dict[int, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    surfaces: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        surface = pd.read_csv(str(SURFACE_TEMPLATE).format(year=year), dtype={"grid_id": str})
        surface["pelagic_evidence_domain"] = parse_bool(surface["pelagic_evidence_domain"])
        if len(surface) != 2_839 or surface["grid_id"].duplicated().any():
            raise ValueError(f"Unexpected {year} surface inventory")
        if int(surface["pelagic_evidence_domain"].sum()) != EXPECTED_DOMAIN_CELLS[year]:
            raise ValueError(f"Unexpected {year} empirical-domain size")
        outside = surface.loc[~surface["pelagic_evidence_domain"], "P_pelagic_central"]
        if not np.allclose(outside, 0):
            raise ValueError(f"Nonzero pelagic support outside the {year} empirical domain")
        model_types = surface["observation_model_type"].dropna().unique().tolist()
        if model_types != [EXPECTED_MODEL[year]]:
            raise ValueError(f"Unexpected selected model for {year}: {model_types}")
        surfaces[year] = surface

    observations = load_observed_support(surfaces)

    cv = pd.read_csv(CV_PATH)
    gate_rows = []
    for year in YEARS:
        analysis_set = f"early_{year}_log_abundance"
        row = cv.loc[
            cv["analysis_set"].eq(analysis_set)
            & cv["model_type"].eq(EXPECTED_MODEL[year])
        ]
        if len(row) != 1:
            raise ValueError(f"Could not identify one retained model for {year}")
        row = row.iloc[0]
        positive_folds = int(round(float(row["positive_skill_fold_fraction"]) * 4))
        passes = bool(
            float(row["rmse_skill"]) > 0
            and float(row["spearman_rho"]) >= 0.30
            and float(row["positive_skill_fold_fraction"]) >= 0.75
        )
        if not passes:
            raise ValueError(f"Retained {year} model does not reproduce the model gate")
        gate_rows.append(
            {
                "year": year,
                "model_type": str(row["model_type"]),
                "observation_count": int(row["record_count"]),
                "spatial_block_count": int(row["spatial_block_count"]),
                "rmse_model": float(row["rmse"]),
                "rmse_null": float(row["null_rmse"]),
                "rmse_skill": float(row["rmse_skill"]),
                "spearman_rho": float(row["spearman_rho"]),
                "positive_skill_fold_fraction": float(row["positive_skill_fold_fraction"]),
                "positive_skill_folds": positive_folds,
                "gate_passed": passes,
            }
        )
    gates = pd.DataFrame(gate_rows)

    coverage = pd.read_csv(COVERAGE_PATH)
    coverage = coverage.rename(
        columns={"method_comparable_june_grid_cells": "observed_june_grid_cells"}
    )
    if coverage["year"].tolist() != CONTEXT_YEARS:
        raise ValueError("Temporal-coverage years do not reproduce 2011-2019")
    expected_counts = {year: EXPECTED_OBSERVATIONS.get(year, 0) for year in CONTEXT_YEARS}
    actual_counts = coverage.set_index("year")["observed_june_grid_cells"].to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected temporal coverage: {actual_counts}")
    coverage["comparable_surface_available"] = coverage["year"].isin(YEARS)

    surface_qa = pd.read_csv(SURFACE_QA_PATH)
    for year in YEARS:
        row = surface_qa.loc[surface_qa["target_year"].eq(year)]
        if len(row) != 1:
            raise ValueError(f"Missing surface QA row for {year}")
        row = row.iloc[0]
        if int(row["prediction_domain_cell_count"]) != EXPECTED_DOMAIN_CELLS[year]:
            raise ValueError(f"Surface-QA domain count differs for {year}")
        if not np.isclose(
            float(row["prediction_domain_percent_of_grid"]), EXPECTED_DOMAIN_PERCENT[year]
        ):
            raise ValueError(f"Surface-QA domain percentage differs for {year}")
        if float(row["shared_interpolation_limit_km"]) != 60:
            raise ValueError(f"Unexpected interpolation limit for {year}")

    land = gpd.read_file(LAND_PATH).to_crs("EPSG:3338")
    dbo = figure1_base.make_dbo_reference_polygons()
    if set(dbo["name"]) != set(DBO_LABEL_POSITIONS):
        raise ValueError("DBO references do not reproduce Figure 1")
    return surfaces, observations, gates, coverage, land, dbo


def cell_boundary(frame: pd.DataFrame, mask: pd.Series | None = None) -> gpd.GeoSeries:
    cells = frame if mask is None else frame.loc[mask]
    geometry = unary_union(
        [box(x - 5_000, y - 5_000, x + 5_000, y + 5_000) for x, y in cells[["x", "y"]].to_numpy()]
    )
    return gpd.GeoSeries([geometry], crs="EPSG:3338")


def project_point(transformer: Transformer, lon: float, lat: float) -> tuple[float, float]:
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def add_panel_label(ax: plt.Axes, label: str, inside: bool = False) -> None:
    x, y = (0.018, 0.985) if inside else (0.0, 1.045)
    artist = ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=30,
    )
    if inside:
        artist.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])


def map_extent(surfaces: dict[int, pd.DataFrame], dbo: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    reference = surfaces[2017]
    dbo_projected = dbo.to_crs("EPSG:3338")
    dbo_xmin, dbo_ymin, dbo_xmax, dbo_ymax = dbo_projected.total_bounds
    xmin = min(float(reference["x"].min()), float(dbo_xmin)) - 18_000
    xmax = max(float(reference["x"].max()), float(dbo_xmax)) + 18_000
    ymin = min(float(reference["y"].min()), float(dbo_ymin)) - 18_000
    ymax = max(float(reference["y"].max()), float(dbo_ymax)) + 18_000
    return xmin, xmax, ymin, ymax


def plot_support_map(
    ax: plt.Axes,
    year: int,
    surface: pd.DataFrame,
    observations: pd.DataFrame,
    land: gpd.GeoDataFrame,
    dbo: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    add_labels: bool,
) -> mpl.collections.PathCollection:
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor("#EDF4F6")
    ax.scatter(
        surface["x"],
        surface["y"],
        s=2.5,
        marker="s",
        color="#D7E0E3",
        linewidth=0,
        rasterized=True,
        zorder=1,
    )
    domain = surface.loc[surface["pelagic_evidence_domain"]]
    map_artist = ax.scatter(
        domain["x"],
        domain["y"],
        c=domain["P_pelagic_central"],
        s=4.2,
        marker="s",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    land_clip = land.clip(box(xmin, ymin, xmax, ymax))
    land_clip.plot(
        ax=ax,
        facecolor="#D8D3CA",
        edgecolor="white",
        linewidth=0.25,
        zorder=4,
    )
    cell_boundary(surface).boundary.plot(
        ax=ax,
        color="#7C8B91",
        linewidth=0.42,
        zorder=5,
    )
    cell_boundary(surface, surface["pelagic_evidence_domain"]).boundary.plot(
        ax=ax,
        color=DOMAIN_COLOR,
        linewidth=0.90,
        zorder=7,
    )
    dbo_projected = dbo.to_crs("EPSG:3338")
    dbo_projected.boundary.plot(
        ax=ax,
        color=DBO_COLOR,
        linewidth=0.68,
        linestyle=(0, (4.0, 2.2)),
        zorder=8,
    )

    observed = observations.loc[observations["year"].eq(year)]
    ax.scatter(
        observed["x"],
        observed["y"],
        c=observed["observed_relative_support"],
        s=18,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.55,
        zorder=10,
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    if add_labels:
        for label, (lon, lat) in DBO_LABEL_POSITIONS.items():
            x, y = project_point(transformer, lon, lat)
            text = ax.text(
                x,
                y,
                label,
                fontsize=4.8,
                color=DBO_COLOR,
                ha="center",
                va="center",
                zorder=12,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
        for label, lon, lat in [
            ("Russia", -173.8, 65.2),
            ("Alaska", -159.8, 68.8),
            ("Bering Strait", -168.8, 65.9),
        ]:
            x, y = project_point(transformer, lon, lat)
            text = ax.text(
                x,
                y,
                label,
                fontsize=5.0,
                color="#5E6266",
                ha="center",
                va="center",
                zorder=11,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    ax.text(
        0.02,
        0.018,
        f"n = {len(observed)}; domain = {len(domain):,} cells ({len(domain) / len(surface) * 100:.1f}%)",
        transform=ax.transAxes,
        fontsize=5.3,
        ha="left",
        va="bottom",
        color="#33383A",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
        zorder=20,
    )
    ax.set_title(str(year), pad=2.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "A" if year == 2017 else "B", inside=True)
    return map_artist


def style_quantitative_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color="#DCE1E4", linewidth=0.5, zorder=0)
    ax.tick_params(length=2.5, pad=2)
    ax.set_axisbelow(True)


def plot_model_gate(ax: plt.Axes, gates: pd.DataFrame) -> None:
    metric_columns = ["rmse_skill", "spearman_rho", "positive_skill_fold_fraction"]
    metric_labels = ["RMSE\nskill", "Spearman\n$\\rho$", "Positive\nfolds"]
    thresholds = [0.0, 0.30, 0.75]
    x = np.arange(len(metric_columns), dtype=float)
    width = 0.28
    for index, year in enumerate(YEARS):
        row = gates.loc[gates["year"].eq(year)].iloc[0]
        values = [float(row[column]) for column in metric_columns]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            color=MODEL_COLORS[year],
            edgecolor="white",
            linewidth=0.35,
            label=str(year),
            zorder=3,
        )
        for metric_index, (bar, value) in enumerate(zip(bars, values)):
            label = (
                f"{int(row['positive_skill_folds'])}/4"
                if metric_index == 2
                else f"{value:.2f}"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="#34383A",
            )
    for position, threshold in zip(x, thresholds):
        ax.hlines(
            threshold,
            position - 0.37,
            position + 0.37,
            color="#222629",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.52, 2.52)
    style_quantitative_axis(ax)
    threshold_handle = Line2D(
        [0], [0], color="#222629", linewidth=0.75, linestyle=(0, (3, 2)), label="Retention threshold"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [threshold_handle],
        labels + ["Retention threshold"],
        loc="upper left",
        frameon=False,
        ncol=3,
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.35,
        fontsize=5.6,
    )
    add_panel_label(ax, "C")


def plot_temporal_coverage(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    available = coverage.loc[coverage["comparable_surface_available"]]
    bars = ax.bar(
        available["year"],
        available["observed_june_grid_cells"],
        color=[MODEL_COLORS[int(year)] for year in available["year"]],
        width=0.68,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for bar, value in zip(bars, available["observed_june_grid_cells"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 1.0,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=5.9,
            color="#34383A",
        )
    ax.set_xlim(2010.55, 2019.45)
    ax.set_ylim(0, 43)
    ax.set_xticks(CONTEXT_YEARS)
    ax.set_xticklabels(CONTEXT_YEARS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Observed June grid cells")
    ax.set_xlabel("Year")
    style_quantitative_axis(ax)
    add_panel_label(ax, "D")


def draw_figure(
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    gates: pd.DataFrame,
    coverage: pd.DataFrame,
    land: gpd.GeoDataFrame,
    dbo: gpd.GeoDataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 5.05), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.34, 0.66],
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.83,
        hspace=0.34,
        wspace=0.29,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    extent = map_extent(surfaces, dbo)
    map_artist = plot_support_map(
        ax_a, 2017, surfaces[2017], observations, land, dbo, extent, add_labels=True
    )
    plot_support_map(
        ax_b, 2018, surfaces[2018], observations, land, dbo, extent, add_labels=False
    )
    plot_model_gate(ax_c, gates)
    plot_temporal_coverage(ax_d, coverage)

    cax = fig.add_axes([0.42, 0.94, 0.17, 0.014])
    colorbar = fig.colorbar(map_artist, cax=cax, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Relative euphausiid support", fontsize=6.5, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=5.8, pad=1)

    fig.legend(
        handles=[
            Patch(
                facecolor="#D7E0E3",
                edgecolor="#7C8B91",
                linewidth=0.45,
                label="Study-region ocean grid",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="", markersize=4.5,
                markerfacecolor="#3C8D77", markeredgecolor="white",
                label="June observation",
            ),
            Line2D([0], [0], color=DOMAIN_COLOR, linewidth=1.0, label="Prediction domain"),
            Line2D(
                [0], [0], color=DBO_COLOR, linewidth=0.8,
                linestyle=(0, (4.0, 2.2)), label="DBO reference area",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=4,
        frameon=False,
        fontsize=5.8,
        columnspacing=1.05,
        handlelength=1.7,
        handletextpad=0.4,
    )
    return fig


def save_figure(fig: plt.Figure) -> dict[str, Path]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        extension: CANDIDATE_DIR / f"{STEM}.{extension}"
        for extension in ["pdf", "png", "svg", "tiff"]
    }
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.03)
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


def write_source_data(
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    gates: pd.DataFrame,
    coverage: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(surfaces.values(), ignore_index=True).to_csv(
        SOURCE_DIR / "c5_pelagic_support_map_source_data_v1.csv", index=False
    )
    observations.to_csv(
        SOURCE_DIR / "c5_pelagic_observations_source_data_v1.csv", index=False
    )
    gates.to_csv(SOURCE_DIR / "c5_model_gate_source_data_v1.csv", index=False)
    coverage.to_csv(SOURCE_DIR / "c5_temporal_availability_source_data_v1.csv", index=False)
    dbo_source = dbo.copy()
    dbo_source["geometry_wkt"] = dbo_source.geometry.to_wkt()
    dbo_source.drop(columns="geometry").to_csv(
        SOURCE_DIR / "c5_dbo_reference_areas_figure1_matched_v1.csv", index=False
    )


def write_qa(
    paths: dict[str, Path],
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    gates: pd.DataFrame,
    coverage: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
) -> None:
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
    for year in YEARS:
        surface = surfaces[year]
        observed = observations.loc[observations["year"].eq(year)]
        gate = gates.loc[gates["year"].eq(year)].iloc[0]
        rows.extend(
            [
                {
                    "check": f"surface_cells_{year}",
                    "value": len(surface),
                    "expected": 2_839,
                    "passed": len(surface) == 2_839,
                },
                {
                    "check": f"domain_cells_{year}",
                    "value": int(surface["pelagic_evidence_domain"].sum()),
                    "expected": EXPECTED_DOMAIN_CELLS[year],
                    "passed": int(surface["pelagic_evidence_domain"].sum()) == EXPECTED_DOMAIN_CELLS[year],
                },
                {
                    "check": f"observations_{year}",
                    "value": len(observed),
                    "expected": EXPECTED_OBSERVATIONS[year],
                    "passed": len(observed) == EXPECTED_OBSERVATIONS[year],
                },
                {
                    "check": f"model_gate_{year}",
                    "value": bool(gate["gate_passed"]),
                    "expected": True,
                    "passed": bool(gate["gate_passed"]),
                },
                {
                    "check": f"outside_domain_nonzero_{year}",
                    "value": int((surface.loc[~surface["pelagic_evidence_domain"], "P_pelagic_central"] != 0).sum()),
                    "expected": 0,
                    "passed": bool(np.allclose(surface.loc[~surface["pelagic_evidence_domain"], "P_pelagic_central"], 0)),
                },
            ]
        )
    rows.extend(
        [
            {
                "check": "available_surface_years",
                "value": ",".join(map(str, coverage.loc[coverage["comparable_surface_available"], "year"])),
                "expected": "2017,2018",
                "passed": coverage.loc[coverage["comparable_surface_available"], "year"].tolist() == YEARS,
            },
            {
                "check": "dbo_reference_areas",
                "value": len(dbo),
                "expected": 5,
                "passed": len(dbo) == 5,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(
        QA_DIR / "gray_whale_si_c5_pelagic_support_evidence_qa_v1.csv", index=False
    )


def write_metadata(gates: pd.DataFrame) -> None:
    lines = [
        "# Pelagic-support observation and model-gate candidate v1",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: observation-informed pelagic-support surfaces are supported only for June 2017 and June 2018 and only within spatial domains bounded by the observations.",
        "- Evidence chain: panels A and B show observations, relative predictions and prediction domains; panel C shows the three prespecified model-retention gates; panel D shows temporal data availability.",
        "- Cross-year transfer and rejected late-season tests are intentionally excluded because they do not define the retained corridor sensitivity.",
        "- DBO1-DBO5 outlines reproduce Figure 1 and are geographic references only.",
        "- Backend: Python/matplotlib only.",
        "",
        "## Retained models",
        "",
    ]
    for row in gates.itertuples(index=False):
        lines.append(
            f"- {row.year}: {row.model_type}; RMSE skill={row.rmse_skill:.3f}; "
            f"Spearman rho={row.spearman_rho:.3f}; positive folds={row.positive_skill_folds}/4."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Colored predictions are displayed only inside the empirical domain; pelagic support is zero outside that domain in the sensitivity analysis.",
            "- Blank years in panel D indicate no method-comparable June abundance surface, not zero euphausiid abundance.",
            "- The two annual surfaces support a bounded sensitivity analysis and are not extrapolated to 2011-2016 or 2019.",
        ]
    )
    (OUT_ROOT / "gray_whale_si_c5_pelagic_support_evidence_metadata_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    for directory in [CANDIDATE_DIR, SOURCE_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    surfaces, observations, gates, coverage, land, dbo = load_inputs()
    write_source_data(surfaces, observations, gates, coverage, dbo)
    figure = draw_figure(surfaces, observations, gates, coverage, land, dbo)
    paths = save_figure(figure)
    plt.close(figure)
    write_qa(paths, surfaces, observations, gates, coverage, dbo)
    write_metadata(gates)
    print(gates[["year", "model_type", "rmse_skill", "spearman_rho", "positive_skill_folds"]].to_string(index=False))
    print(f"\nCandidate: {paths['pdf'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
