#!/usr/bin/env python3
"""Build the revised C5 pelagic-support evidence SI candidate."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gray-whale-c5-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageStat

import build_si_c5_pelagic_support_evidence_v1 as v1


OUT_ROOT = (
    ROOT
    / "outputs"
    / "manuscript_figures_v3"
    / "supplementary_figures_working"
    / "c5_pelagic_support_evidence_v2"
)
CANDIDATE_DIR = OUT_ROOT / "candidates"
SOURCE_DIR = OUT_ROOT / "source_data"
QA_DIR = OUT_ROOT / "qa"
STEM = "gray_whale_si_c5_pelagic_support_evidence_candidate_v2"

MODEL_TYPES = ["hydrographic", "spatial", "spatial_hydrographic"]
MODEL_LABELS = {
    "hydrographic": "Hydrographic",
    "spatial": "Spatial",
    "spatial_hydrographic": "Spatial + hydro.",
}
SELECTED_MODEL = {2017: "spatial", 2018: "spatial"}
PASS_COLOR = "#DCEFE5"
FAIL_COLOR = "#F2E5E1"
TEXT_COLOR = "#303538"


def configure_style() -> None:
    v1.configure_style()
    mpl.rcParams.update(
        {
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 5.8,
        }
    )


def candidate_model_results(cv: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in v1.YEARS:
        analysis_set = f"early_{year}_log_abundance"
        annual = cv.loc[
            cv["analysis_set"].eq(analysis_set)
            & cv["model_type"].isin(MODEL_TYPES)
        ].copy()
        if set(annual["model_type"]) != set(MODEL_TYPES):
            raise ValueError(f"Incomplete candidate-model inventory for {year}")
        annual["positive_skill_folds"] = np.rint(
            annual["positive_skill_fold_fraction"] * 4
        ).astype(int)
        annual["passes_gate"] = (
            annual["rmse_skill"].gt(0)
            & annual["spearman_rho"].ge(0.30)
            & annual["positive_skill_fold_fraction"].ge(0.75)
        )
        passing = annual.loc[annual["passes_gate"]].sort_values(
            ["rmse", "model_type"], kind="mergesort"
        )
        if passing.empty:
            raise ValueError(f"No candidate model passed all gates in {year}")
        selected = str(passing.iloc[0]["model_type"])
        if selected != SELECTED_MODEL[year]:
            raise ValueError(f"Unexpected selected model for {year}: {selected}")
        annual["selected"] = annual["model_type"].eq(selected)
        annual["year"] = year
        pieces.append(annual)
    result = pd.concat(pieces, ignore_index=True)
    order = {model: index for index, model in enumerate(MODEL_TYPES)}
    result["model_order"] = result["model_type"].map(order)
    return result.sort_values(["year", "model_order"]).reset_index(drop=True)


def load_inputs() -> tuple[
    dict[int, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    surfaces, observations, _, coverage, _, dbo = v1.load_inputs()
    # Read directly in geographic coordinates, matching Figure 1 and avoiding
    # antimeridian artifacts introduced by a projected round trip.
    land = gpd.read_file(v1.LAND_PATH).to_crs(4326)
    candidates = candidate_model_results(pd.read_csv(v1.CV_PATH))
    return surfaces, observations, candidates, coverage, land, dbo.to_crs(4326)


def map_extent(surface: pd.DataFrame) -> tuple[float, float, float, float]:
    # Match the longitude-latitude framing and latitude correction used in Fig. 1.
    return v1.figure1_base.extent_from_ocean_grid(surface, pad=1.65)


def add_text_with_halo(
    ax: plt.Axes,
    text: str,
    lon: float,
    lat: float,
    *,
    fontsize: float = 5.2,
    color: str = "#5E6266",
    ha: str = "center",
    zorder: int = 20,
) -> None:
    artist = ax.text(
        lon,
        lat,
        text,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va="center",
        zorder=zorder,
    )
    artist.set_path_effects(
        [path_effects.withStroke(linewidth=1.7, foreground="white")]
    )


def add_reference_labels(ax: plt.Axes) -> None:
    # Labels likely to cover the support field are placed on land with leaders.
    add_text_with_halo(ax, "DBO1", -174.15, 62.75, color=v1.DBO_COLOR, fontsize=5.0)
    add_text_with_halo(ax, "DBO4", -162.55, 71.30, color=v1.DBO_COLOR, fontsize=5.0)
    add_text_with_halo(ax, "DBO5", -157.35, 71.48, color=v1.DBO_COLOR, fontsize=5.0)
    for label, text_xy, target_xy in [
        ("DBO2", (-165.55, 64.65), (-169.15, 64.78)),
        ("DBO3", (-164.45, 67.70), (-168.90, 67.70)),
    ]:
        ax.annotate(
            label,
            xy=target_xy,
            xytext=text_xy,
            textcoords="data",
            fontsize=5.0,
            color=v1.DBO_COLOR,
            ha="center",
            va="center",
            zorder=21,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.45},
            arrowprops={"arrowstyle": "-", "color": v1.DBO_COLOR, "linewidth": 0.55},
        )

    add_text_with_halo(ax, "Russia", -174.75, 65.55, ha="left")
    add_text_with_halo(ax, "Alaska\nUSA", -159.15, 69.20)
    ax.annotate(
        "St. Lawrence\nIsland",
        xy=(-170.25, 63.78),
        xytext=(-165.20, 63.35),
        textcoords="data",
        fontsize=4.9,
        color="#5E6266",
        ha="center",
        va="center",
        zorder=21,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.4},
        arrowprops={"arrowstyle": "-", "color": "#6A6E71", "linewidth": 0.50},
    )
    ax.annotate(
        "Bering\nStrait",
        xy=(-168.35, 65.90),
        xytext=(-164.55, 65.95),
        textcoords="data",
        fontsize=4.9,
        color="#5E6266",
        ha="center",
        va="center",
        zorder=21,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.4},
        arrowprops={"arrowstyle": "-", "color": "#6A6E71", "linewidth": 0.50},
    )


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
    ax.set_facecolor(v1.figure1_base.PALETTE["ocean"])
    land.cx[xmin:xmax, ymin:ymax].plot(
        ax=ax,
        color=v1.figure1_base.PALETTE["land"],
        edgecolor="white",
        linewidth=0.35,
        zorder=1,
    )
    ax.scatter(
        surface["lon"],
        surface["lat"],
        s=3.0,
        marker="s",
        color="#D7E0E3",
        edgecolor="none",
        alpha=0.80,
        rasterized=True,
        zorder=2,
    )
    domain = surface.loc[surface["pelagic_evidence_domain"]]
    map_artist = ax.scatter(
        domain["lon"],
        domain["lat"],
        c=domain["P_pelagic_central"],
        s=5.0,
        marker="s",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidth=0,
        rasterized=True,
        zorder=4,
    )
    v1.cell_boundary(surface).to_crs(4326).boundary.plot(
        ax=ax, color="#7C8B91", linewidth=0.45, zorder=5
    )
    v1.cell_boundary(surface, surface["pelagic_evidence_domain"]).to_crs(4326).boundary.plot(
        ax=ax, color=v1.DOMAIN_COLOR, linewidth=0.95, zorder=7
    )
    dbo.boundary.plot(
        ax=ax,
        color=v1.DBO_COLOR,
        linewidth=0.85,
        linestyle=(0, (4.0, 2.2)),
        zorder=8,
    )

    observed = observations.loc[observations["year"].eq(year)]
    ax.scatter(
        observed["lon"],
        observed["lat"],
        c=observed["observed_relative_support"],
        s=19,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.55,
        zorder=10,
    )

    if add_labels:
        add_reference_labels(ax)

    ax.text(
        0.975,
        0.018,
        f"n = {len(observed)}; domain = {len(domain):,} cells ({len(domain) / len(surface) * 100:.1f}%)",
        transform=ax.transAxes,
        fontsize=5.25,
        ha="right",
        va="bottom",
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.0},
        zorder=22,
    )
    ax.set_title(str(year), pad=2.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    mean_lat = (ymin + ymax) / 2
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks([-175, -170, -165, -160, -155])
    ax.set_yticks([64, 66, 68, 70, 72])
    ax.grid(color="white", linewidth=0.45, zorder=0)
    ax.tick_params(length=2.4, pad=1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    v1.add_panel_label(ax, "A" if year == 2017 else "B", inside=True)
    return map_artist


def metric_passes(model: pd.Series, column: str) -> bool:
    if column == "rmse_skill":
        return float(model[column]) > 0
    if column == "spearman_rho":
        return float(model[column]) >= 0.30
    if column == "positive_skill_fold_fraction":
        return float(model[column]) >= 0.75
    raise ValueError(column)


def plot_model_comparison(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    metric_columns = ["rmse_skill", "spearman_rho", "positive_skill_fold_fraction"]
    headers = ["RMSE skill\n> 0", "Spearman $\\rho$\n$\\geq$ 0.30", "Positive folds\n$\\geq$ 3/4"]
    rows = []
    for year in v1.YEARS:
        annual = candidates.loc[candidates["year"].eq(year)]
        for model_type in MODEL_TYPES:
            rows.append(annual.loc[annual["model_type"].eq(model_type)].iloc[0])

    ax.set_xlim(-1.62, 3.02)
    ax.set_ylim(-0.98, len(rows) + 0.72)
    ax.axis("off")

    for column, header in enumerate(headers):
        ax.text(
            column + 0.5,
            len(rows) + 0.32,
            header,
            ha="center",
            va="center",
            fontsize=5.45,
            color=TEXT_COLOR,
            linespacing=1.05,
        )

    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        model_type = str(row["model_type"])
        selected = bool(row["selected"])
        label = MODEL_LABELS[model_type]
        ax.text(
            -0.06,
            y + 0.5,
            label,
            ha="right",
            va="center",
            fontsize=5.55,
            fontweight="bold" if selected else "normal",
            color=TEXT_COLOR,
        )
        values = [
            f"{float(row['rmse_skill']):.2f}",
            f"{float(row['spearman_rho']):.2f}",
            f"{int(row['positive_skill_folds'])}/4",
        ]
        for column, (metric, value) in enumerate(zip(metric_columns, values)):
            facecolor = PASS_COLOR if metric_passes(row, metric) else FAIL_COLOR
            ax.add_patch(
                Rectangle(
                    (column + 0.04, y + 0.06),
                    0.92,
                    0.88,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            ax.text(
                column + 0.5,
                y + 0.5,
                value,
                ha="center",
                va="center",
                fontsize=5.75,
                color=TEXT_COLOR,
                zorder=2,
            )
        if selected:
            ax.add_patch(
                Rectangle(
                    (-0.98, y + 0.025),
                    3.96,
                    0.95,
                    fill=False,
                    edgecolor=v1.MODEL_COLORS[int(row["year"])],
                    linewidth=0.9,
                    zorder=4,
                )
            )

    ax.text(-1.50, 4.50, "2017", ha="left", va="center", fontsize=6.0, fontweight="bold", color=v1.MODEL_COLORS[2017])
    ax.text(-1.50, 1.50, "2018", ha="left", va="center", fontsize=6.0, fontweight="bold", color=v1.MODEL_COLORS[2018])
    ax.plot([-1.53, 2.98], [3.0, 3.0], color="#AEB7BB", linewidth=0.65, zorder=5)
    v1.add_panel_label(ax, "C")


def plot_temporal_coverage(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    v1.plot_temporal_coverage(ax, coverage)
    for artist in list(ax.texts):
        if artist.get_text() == "D":
            artist.remove()
    v1.add_panel_label(ax, "D", inside=True)


def draw_figure(
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    land: gpd.GeoDataFrame,
    dbo: gpd.GeoDataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 5.55), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.46, 0.74],
        width_ratios=[1.18, 0.82],
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.84,
        hspace=0.35,
        wspace=0.27,
    )
    map_grid = gs[0, :].subgridspec(1, 2, wspace=0.19)
    ax_a = fig.add_subplot(map_grid[0, 0])
    ax_b = fig.add_subplot(map_grid[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_c.set_position([0.160, 0.105, 0.330, 0.195])
    ax_d.set_position([0.600, 0.105, 0.250, 0.195])

    extent = map_extent(surfaces[2017])
    map_artist = plot_support_map(
        ax_a, 2017, surfaces[2017], observations, land, dbo, extent, add_labels=True
    )
    plot_support_map(
        ax_b, 2018, surfaces[2018], observations, land, dbo, extent, add_labels=False
    )
    ax_a.set_anchor("E")
    ax_b.set_anchor("W")
    plot_model_comparison(ax_c, candidates)
    plot_temporal_coverage(ax_d, coverage)

    cax = fig.add_axes([0.42, 0.944, 0.17, 0.014])
    colorbar = fig.colorbar(map_artist, cax=cax, orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Relative euphausiid support", fontsize=6.5, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=5.8, pad=1)

    fig.legend(
        handles=[
            Patch(facecolor="#D7E0E3", edgecolor="#7C8B91", linewidth=0.45, label="Study-region ocean grid"),
            Line2D([0], [0], marker="o", linestyle="", markersize=4.5, markerfacecolor="#3C8D77", markeredgecolor="white", label="June observation"),
            Line2D([0], [0], color=v1.DOMAIN_COLOR, linewidth=1.0, label="Prediction domain"),
            Line2D([0], [0], color=v1.DBO_COLOR, linewidth=0.8, linestyle=(0, (4.0, 2.2)), label="DBO reference area"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.875),
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
    paths = {extension: CANDIDATE_DIR / f"{STEM}.{extension}" for extension in ["pdf", "png", "svg", "tiff"]}
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    return paths


def image_nonblank(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(ImageStat.Stat(image.convert("L")).stddev[0] > 1.0)


def write_source_data(
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    dbo: gpd.GeoDataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(surfaces.values(), ignore_index=True).to_csv(SOURCE_DIR / "c5_pelagic_support_map_source_data_v2.csv", index=False)
    observations.to_csv(SOURCE_DIR / "c5_pelagic_observations_source_data_v2.csv", index=False)
    candidates.to_csv(SOURCE_DIR / "c5_candidate_model_comparison_source_data_v2.csv", index=False)
    coverage.to_csv(SOURCE_DIR / "c5_temporal_availability_source_data_v2.csv", index=False)
    dbo_source = dbo.copy()
    dbo_source["geometry_wkt"] = dbo_source.geometry.to_wkt()
    dbo_source.drop(columns="geometry").to_csv(SOURCE_DIR / "c5_dbo_reference_areas_figure1_matched_v2.csv", index=False)


def write_qa(
    paths: dict[str, Path],
    surfaces: dict[int, pd.DataFrame],
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
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
                row.update({"width_px": image.width, "height_px": image.height, "nonblank": image_nonblank(path)})
        rows.append(row)
    for year in v1.YEARS:
        annual = candidates.loc[candidates["year"].eq(year)]
        selected = annual.loc[annual["selected"]]
        rows.extend(
            [
                {"check": f"candidate_models_{year}", "value": len(annual), "expected": 3, "passed": len(annual) == 3},
                {"check": f"selected_model_{year}", "value": selected.iloc[0]["model_type"] if len(selected) == 1 else "invalid", "expected": "spatial", "passed": len(selected) == 1 and selected.iloc[0]["model_type"] == "spatial"},
                {"check": f"observations_{year}", "value": int(observations["year"].eq(year).sum()), "expected": v1.EXPECTED_OBSERVATIONS[year], "passed": int(observations["year"].eq(year).sum()) == v1.EXPECTED_OBSERVATIONS[year]},
                {"check": f"domain_cells_{year}", "value": int(surfaces[year]["pelagic_evidence_domain"].sum()), "expected": v1.EXPECTED_DOMAIN_CELLS[year], "passed": int(surfaces[year]["pelagic_evidence_domain"].sum()) == v1.EXPECTED_DOMAIN_CELLS[year]},
            ]
        )
    rows.append(
        {
            "check": "available_surface_years",
            "value": ",".join(map(str, coverage.loc[coverage["comparable_surface_available"], "year"])),
            "expected": "2017,2018",
            "passed": coverage.loc[coverage["comparable_surface_available"], "year"].tolist() == v1.YEARS,
        }
    )
    pd.DataFrame(rows).to_csv(QA_DIR / "gray_whale_si_c5_pelagic_support_evidence_qa_v2.csv", index=False)


def write_metadata(candidates: pd.DataFrame) -> None:
    lines = [
        "# Pelagic-support observation and model-gate candidate v2",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: empirical pelagic-support surfaces are defensible only for June 2017 and June 2018, within observation-bounded domains, and after candidate-model comparison.",
        "- Panels A and B: observations, relative predictions and bounded prediction domains in the longitude-latitude framing used by Figure 1.",
        "- Panel C: all annual non-null candidate models, all three retention metrics and the prespecified selection decision.",
        "- Panel D: temporal availability of method-comparable June observations.",
        "- Archetype: map-led evidence and validation composite.",
        "- Backend: Python/matplotlib in the mypy10 environment.",
        "",
        "## Candidate-model decisions",
        "",
    ]
    for row in candidates.itertuples(index=False):
        decision = "selected" if row.selected else ("pass" if row.passes_gate else "fail")
        lines.append(
            f"- {row.year} {row.model_type}: RMSE skill={row.rmse_skill:.3f}; Spearman rho={row.spearman_rho:.3f}; positive folds={row.positive_skill_folds}/4; {decision}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The maps show relative pelagic support, not benthic-plus-pelagic utility and not prey energy density.",
            "- Labels were moved away from colored support cells where possible; DBO reference outlines do not define the prediction domain.",
            "- Blank years in panel D denote unavailable comparable observations, not zero euphausiid abundance.",
        ]
    )
    (OUT_ROOT / "gray_whale_si_c5_pelagic_support_evidence_metadata_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    for directory in [CANDIDATE_DIR, SOURCE_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    surfaces, observations, candidates, coverage, land, dbo = load_inputs()
    write_source_data(surfaces, observations, candidates, coverage, dbo)
    figure = draw_figure(surfaces, observations, candidates, coverage, land, dbo)
    paths = save_figure(figure)
    plt.close(figure)
    write_qa(paths, surfaces, observations, candidates, coverage)
    write_metadata(candidates)
    print(candidates[["year", "model_type", "rmse_skill", "spearman_rho", "positive_skill_folds", "passes_gate", "selected"]].to_string(index=False))
    print(f"\nCandidate: {paths['pdf'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
