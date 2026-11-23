"""Phase 8I.2 ASAMM sighting-to-C3/LCP diagnostic extraction.

This phase matches ASAMM gray whale records and ASAMM survey-effort event
points to the study-domain grid, extracts existing LCP frequency values, and
builds descriptive validation summaries. It does not rerun connectivity models.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import unary_union


ASAMM_ROOT = ROOT / "data/raw/validation/asamm_1979_2021/csv_export/ASAMM to csv"
GRID_PATH = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
C3_GEOJSON = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
LCP_DIR = ROOT / "outputs/lcp_v2"
OUT_DIR = ROOT / "data/processed/validation"
FIG_DIR = ROOT / "outputs/figures/validation"
PHASE_MEMO = ROOT / "docs/phase_memos/phase8i2_asamm_sighting_to_C3_lcp_extraction.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase8i2_asamm_lcp_extraction_paper_use.md"
MANIFEST = ROOT / "outputs/results_manifest.csv"
REGISTRY = ROOT / "docs/results_registry.md"
CLAIMS = ROOT / "docs/paper_claims_evidence_matrix.md"
README = ROOT / "README.md"
FINAL_CLAIMS = ROOT / "docs/final_synthesis_v2/paper_claims_evidence_matrix_final_v2.md"
FIG_PLAN = ROOT / "docs/final_synthesis_v2/main_figure_plan_v2.md"
RESULTS_OUTLINE = ROOT / "docs/final_synthesis_v2/results_section_outline_v2.md"
KEY_BULLETS = ROOT / "docs/final_synthesis_v2/key_result_bullets_v2.md"
GAPS = ROOT / "docs/final_synthesis_v2/gaps_before_manuscript_v2.md"

PHASE = "Phase 8I.2 ASAMM sighting-to-C3 LCP diagnostic extraction"
SCRIPT = "scripts/extract_asamm_to_brian_C3_lcp_phase8i2_v2.py"
TARGET_YEARS = list(range(2011, 2020))
SHIP_YEARS = list(range(2015, 2020))
PREY_VARIABLES = {
    "total_kJm2_frozen": "kJm2_frozen",
    "total_gww": "gww",
    "total_gC": "gC",
}
MODEL_LAYERS = {
    "prey_only": {
        "pattern": "prey_only_brian_{label}_C3_lcp_target{year}_v2.csv",
        "years": TARGET_YEARS,
    },
    "prey_ice": {
        "pattern": "prey_ice_brian_{label}_C3_lcp_target{year}_v2.csv",
        "years": TARGET_YEARS,
    },
    "prey_ice_ship": {
        "pattern": "prey_ice_ship_brian_{label}_C3_lcp_target{year}_v2.csv",
        "years": SHIP_YEARS,
    },
}
VALIDATION_WINDOWS = {
    "July_September": [7, 8, 9],
    "July_only": [7],
    "June_July": [6, 7],
    "August_September": [8, 9],
}
OUTPUT_COLS = [
    "ID",
    "Event",
    "Flt_No",
    "GMT_Minus8_DateTime",
    "Lat",
    "Long",
    "ArcLat",
    "ArcLong",
    "LatTemp",
    "LongTemp",
    "Entry",
    "Species",
    "Totalno",
    "Final_group",
    "Low_estimate",
    "High_estimate",
    "FltType",
    "Transect_id",
    "Block",
    "Aerial_Survey",
    "Icepercent",
    "Seastate",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_c3_polygon():
    data = json.loads(C3_GEOJSON.read_text())
    geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
    if data.get("type") in {"Polygon", "MultiPolygon"}:
        geoms.append(shape(data))
    return unary_union(geoms)


def normalize_lon(values: pd.Series) -> pd.Series:
    lon = pd.to_numeric(values, errors="coerce")
    return ((lon + 180) % 360) - 180


def best_lat_lon(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lat = pd.to_numeric(df.get("ArcLat"), errors="coerce")
    lon = normalize_lon(df.get("ArcLong"))
    for lat_col, lon_col in [("LatTemp", "LongTemp"), ("Lat", "Long")]:
        lat_alt = pd.to_numeric(df.get(lat_col), errors="coerce")
        lon_alt = normalize_lon(df.get(lon_col))
        lat = lat.where(lat.notna(), lat_alt)
        lon = lon.where(lon.notna(), lon_alt)
    return lat, lon


def is_effort_entry(entry: object) -> bool:
    text = str(entry).strip().lower()
    if text in {"", ".", "nan", "none"}:
        return False
    if "deadhead" in text:
        return False
    keywords = [
        "transect",
        "search",
        "circling",
        "resume",
        "start",
        "end",
        "fov",
        "caps",
        "fgf",
        "divert",
    ]
    return any(k in text for k in keywords)


def read_asamm_events() -> pd.DataFrame:
    frames = []
    for path in sorted(ASAMM_ROOT.glob("*/_Output__*.csv")):
        usecols = lambda c: c in OUTPUT_COLS
        for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=150_000, low_memory=False):
            dt = pd.to_datetime(chunk["GMT_Minus8_DateTime"], errors="coerce")
            sub = chunk.loc[dt.dt.year.between(2011, 2019) & dt.dt.month.between(6, 9)].copy()
            if sub.empty:
                continue
            lat, lon = best_lat_lon(sub)
            sub["lat"] = lat
            sub["lon"] = lon
            sub = sub.loc[sub["lat"].between(50, 75) & sub["lon"].between(-180, -120)].copy()
            if sub.empty:
                continue
            sub["source_file"] = str(path.relative_to(ROOT))
            sub["period_folder"] = path.parent.name
            sub["datetime"] = dt.loc[sub.index]
            sub["year"] = sub["datetime"].dt.year.astype("Int64")
            sub["month"] = sub["datetime"].dt.month.astype("Int64")
            sub["day"] = sub["datetime"].dt.day.astype("Int64")
            sub["species_clean"] = sub["Species"].fillna("").astype(str).str.strip().str.lower()
            sub["is_gray_whale"] = sub["species_clean"].eq("gray whale")
            sub["is_effort_entry"] = sub["Entry"].map(is_effort_entry)
            sub["final_group_num"] = pd.to_numeric(sub.get("Final_group"), errors="coerce")
            sub["totalno_num"] = pd.to_numeric(sub.get("Totalno"), errors="coerce")
            sub["count_or_best_count"] = sub["final_group_num"].where(sub["final_group_num"].gt(0), sub["totalno_num"])
            sub["count_or_best_count"] = sub["count_or_best_count"].where(sub["count_or_best_count"].gt(0), np.nan)
            frames.append(sub)
    if not frames:
        raise RuntimeError("No ASAMM output events were found for 2011-2019 months 6-9.")
    events = pd.concat(frames, ignore_index=True)
    events["asamm_event_id"] = np.arange(1, len(events) + 1)
    return events


def assign_to_grid(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    gx, gy = transformer.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    px, py = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    tree = cKDTree(np.column_stack([gx, gy]))
    dist, idx = tree.query(np.column_stack([px, py]), k=1)
    out = df.copy()
    matched = grid.iloc[idx].reset_index(drop=True)
    out["nearest_grid_id"] = matched["grid_id"].to_numpy()
    out["nearest_grid_lon"] = matched["lon"].to_numpy()
    out["nearest_grid_lat"] = matched["lat"].to_numpy()
    out["nearest_grid_x"] = matched["x"].to_numpy()
    out["nearest_grid_y"] = matched["y"].to_numpy()
    out["distance_to_nearest_grid_cell_km"] = dist / 1000.0
    return out


def add_domain_flags(df: pd.DataFrame, c3_poly) -> pd.DataFrame:
    out = df.copy()
    out["inside_C3_domain"] = [c3_poly.contains(Point(lon, lat)) or c3_poly.touches(Point(lon, lat)) for lon, lat in zip(out["lon"], out["lat"])]
    out["use_for_C3_validation"] = out["inside_C3_domain"] & out["distance_to_nearest_grid_cell_km"].le(15)
    out["season_group"] = np.where(out["month"].isin([6, 7]), "June_July", "August_September")
    out.loc[out["month"].eq(7), "season_group"] = "July"
    return out


def build_effort_footprint(events_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_events = events_grid.loc[events_grid["inside_C3_domain"]].copy()
    strict = all_events.loc[all_events["is_effort_entry"]].copy()
    rows = []
    for source_name, frame in [("strict_effort", strict), ("all_valid_events", all_events)]:
        if frame.empty:
            continue
        grouped = (
            frame.groupby(["year", "month", "nearest_grid_id"], dropna=False)
            .agg(
                event_count=("asamm_event_id", "count"),
                unique_flights=("Flt_No", pd.Series.nunique),
                mean_distance_to_grid_km=("distance_to_nearest_grid_cell_km", "mean"),
            )
            .reset_index()
        )
        grouped["effort_background_type"] = source_name
        rows.append(grouped)
    cell_month = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    window_rows = []
    for source_name, frame in [("strict_effort", strict), ("all_valid_events", all_events)]:
        for window, months in VALIDATION_WINDOWS.items():
            sub = frame.loc[frame["month"].isin(months)]
            if sub.empty:
                continue
            grouped = (
                sub.groupby(["year", "nearest_grid_id"], dropna=False)
                .agg(
                    event_count=("asamm_event_id", "count"),
                    unique_flights=("Flt_No", pd.Series.nunique),
                    months_covered=("month", lambda x: ";".join(map(str, sorted(set(map(int, x)))))),
                )
                .reset_index()
            )
            grouped["validation_window"] = window
            grouped["effort_background_type"] = source_name
            window_rows.append(grouped)
    cell_window = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    cell_month.rename(columns={"nearest_grid_id": "grid_id"}, inplace=True)
    cell_window.rename(columns={"nearest_grid_id": "grid_id"}, inplace=True)
    cell_month.to_csv(OUT_DIR / "asamm_phase8i2_effort_covered_C3_cells_by_month.csv", index=False)
    cell_window.to_csv(OUT_DIR / "asamm_phase8i2_effort_covered_C3_cells_by_window.csv", index=False)
    return cell_month, cell_window


def lcp_path(model_layer: str, prey_variable: str, year: int) -> Path | None:
    label = PREY_VARIABLES[prey_variable]
    spec = MODEL_LAYERS[model_layer]
    if year not in spec["years"]:
        return None
    path = LCP_DIR / spec["pattern"].format(label=label, year=year)
    return path if path.exists() else None


def load_lcp(model_layer: str, prey_variable: str, year: int) -> pd.DataFrame | None:
    path = lcp_path(model_layer, prey_variable, year)
    if path is None:
        return None
    df = pd.read_csv(path)
    df["model_layer"] = model_layer
    df["prey_variable"] = prey_variable
    df["target_year"] = year
    freq = df["lcp_frequency_normalized"].fillna(0)
    df["lcp_frequency_percentile_within_C3"] = freq.rank(method="average", pct=True) * 100
    q95 = freq.quantile(0.95)
    df["is_top5pct_lcp"] = freq >= q95
    return df


def load_lcp_cache() -> dict[tuple[str, str, int], pd.DataFrame]:
    cache = {}
    missing = []
    for layer in MODEL_LAYERS:
        for variable in PREY_VARIABLES:
            for year in MODEL_LAYERS[layer]["years"]:
                df = load_lcp(layer, variable, year)
                if df is None:
                    missing.append({"model_layer": layer, "prey_variable": variable, "year": year})
                else:
                    cache[(layer, variable, year)] = df
    pd.DataFrame(missing).to_csv(OUT_DIR / "asamm_phase8i2_missing_model_files.csv", index=False)
    return cache


def extract_lcp_values(sightings: pd.DataFrame, lcp_cache: dict[tuple[str, str, int], pd.DataFrame], effort_cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    effort_lookup = {}
    for (year, window, bgtype), sub in effort_cells.groupby(["year", "validation_window", "effort_background_type"], dropna=False):
        effort_lookup[(int(year), window, bgtype)] = set(sub["grid_id"].astype(str))
    for _, rec in sightings.iterrows():
        if not bool(rec["use_for_C3_validation"]):
            continue
        year = int(rec["year"])
        grid_id = str(rec["nearest_grid_id"])
        rec_windows = [w for w, months in VALIDATION_WINDOWS.items() if int(rec["month"]) in months]
        for layer in MODEL_LAYERS:
            for variable in PREY_VARIABLES:
                df = lcp_cache.get((layer, variable, year))
                if df is None:
                    continue
                row = df.loc[df["grid_id"].astype(str).eq(grid_id)]
                if row.empty:
                    continue
                l = row.iloc[0]
                for window in rec_windows:
                    strict_cells = effort_lookup.get((year, window, "strict_effort"), set())
                    all_cells = effort_lookup.get((year, window, "all_valid_events"), set())
                    if grid_id in strict_cells:
                        effort_flag = "strict_effort_covered"
                    elif grid_id in all_cells:
                        effort_flag = "all_event_covered_only"
                    else:
                        effort_flag = "not_in_effort_footprint"
                    pct_effort = np.nan
                    if strict_cells:
                        bg = df.loc[df["grid_id"].astype(str).isin(strict_cells), "lcp_frequency_normalized"].fillna(0)
                        pct_effort = ((bg < l["lcp_frequency_normalized"]).sum() + 0.5 * (bg == l["lcp_frequency_normalized"]).sum()) / len(bg) * 100
                    rows.append(
                        {
                            "record_id": rec["record_id"],
                            "asamm_event_id": rec["asamm_event_id"],
                            "year": year,
                            "month": int(rec["month"]),
                            "validation_window": window,
                            "model_layer": layer,
                            "prey_variable": variable,
                            "target_year": year,
                            "nearest_grid_id": grid_id,
                            "lcp_frequency_normalized_at_sighting": l["lcp_frequency_normalized"],
                            "lcp_frequency_percentile_within_C3": l["lcp_frequency_percentile_within_C3"],
                            "lcp_frequency_percentile_within_strict_effort_footprint": pct_effort,
                            "is_corridor_cell_freq_gt_0": l["lcp_frequency_normalized"] > 0,
                            "is_core_corridor_cell_freq_gt_0_05": l["lcp_frequency_normalized"] > 0.05,
                            "is_high_use_corridor_cell_freq_gt_0_10": l["lcp_frequency_normalized"] > 0.10,
                            "is_top5pct_lcp_cell": bool(l["is_top5pct_lcp"]),
                            "effort_footprint_status": effort_flag,
                            "count_or_best_count": rec["count_or_best_count"],
                            "notes": "diagnostic_not_effort_corrected; footprint-aware background available",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "asamm_phase8i2_lcp_value_extraction.csv", index=False)
    return out


def load_persistent_cells() -> dict[tuple[str, str], pd.DataFrame]:
    paths = {
        "prey_only": LCP_DIR / "prey_only_brian_C3_bottleneck_persistence_v2.csv",
        "prey_ice": LCP_DIR / "prey_ice_brian_C3_bottleneck_persistence_v2.csv",
        "prey_ice_ship": LCP_DIR / "prey_ice_ship_brian_C3_bottleneck_persistence_v2.csv",
    }
    out = {}
    for layer, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for variable in PREY_VARIABLES:
            sub = df.loc[df["prey_variable"].eq(variable)].copy()
            if sub.empty:
                continue
            keep = sub.loc[sub["years_in_top5pct"].ge(2)]
            if keep.empty:
                keep = sub.loc[sub["years_in_top5pct"].ge(1)]
            out[(layer, variable)] = keep
    return out


def nearest_distance_km(x: float, y: float, points: np.ndarray) -> float:
    if len(points) == 0:
        return math.nan
    tree = cKDTree(points)
    dist, _ = tree.query(np.array([[x, y]]), k=1)
    return float(dist[0] / 1000.0)


def compute_distances(sightings: pd.DataFrame, lcp_cache: dict[tuple[str, str, int], pd.DataFrame]) -> pd.DataFrame:
    persistent = load_persistent_cells()
    rows = []
    for _, rec in sightings.iterrows():
        if not bool(rec["use_for_C3_validation"]):
            continue
        year = int(rec["year"])
        x, y = float(rec["nearest_grid_x"]), float(rec["nearest_grid_y"])
        rec_windows = [w for w, months in VALIDATION_WINDOWS.items() if int(rec["month"]) in months]
        for layer in MODEL_LAYERS:
            for variable in PREY_VARIABLES:
                df = lcp_cache.get((layer, variable, year))
                if df is None:
                    continue
                freq = df["lcp_frequency_normalized"].fillna(0)
                top5 = df.loc[freq >= freq.quantile(0.95), ["x", "y"]].to_numpy()
                core = df.loc[freq > 0.05, ["x", "y"]].to_numpy()
                high = df.loc[freq > 0.10, ["x", "y"]].to_numpy()
                pers = persistent.get((layer, variable), pd.DataFrame())
                if not pers.empty:
                    pxy = df.loc[df["grid_id"].astype(str).isin(pers["grid_id"].astype(str)), ["x", "y"]].to_numpy()
                else:
                    pxy = np.empty((0, 2))
                for window in rec_windows:
                    d_core = nearest_distance_km(x, y, core)
                    d_top5 = nearest_distance_km(x, y, top5)
                    rows.append(
                        {
                            "record_id": rec["record_id"],
                            "year": year,
                            "month": int(rec["month"]),
                            "validation_window": window,
                            "model_layer": layer,
                            "prey_variable": variable,
                            "distance_to_core_corridor_km": d_core,
                            "distance_to_high_use_corridor_km": nearest_distance_km(x, y, high),
                            "distance_to_top5pct_corridor_km": d_top5,
                            "distance_to_persistent_bottleneck_km": nearest_distance_km(x, y, pxy),
                            "inside_or_near_core_corridor_10km": d_core <= 10 if not math.isnan(d_core) else False,
                            "inside_or_near_core_corridor_25km": d_core <= 25 if not math.isnan(d_core) else False,
                            "inside_or_near_top5pct_25km": d_top5 <= 25 if not math.isnan(d_top5) else False,
                            "notes": "distance_to_existing_LCP_features; diagnostic_only",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "asamm_phase8i2_distance_to_lcp_features.csv", index=False)
    return out


def summarize_background(
    extraction: pd.DataFrame,
    lcp_cache: dict[tuple[str, str, int], pd.DataFrame],
    effort_cells: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (year, window, layer, variable), sub in extraction.groupby(["year", "validation_window", "model_layer", "prey_variable"], dropna=False):
        df = lcp_cache.get((layer, variable, int(year)))
        if df is None or sub.empty:
            continue
        sight_freq = sub["lcp_frequency_normalized_at_sighting"].fillna(0)
        all_bg = df["lcp_frequency_normalized"].fillna(0)
        effort_ids = set(
            effort_cells.loc[
                effort_cells["year"].eq(year)
                & effort_cells["validation_window"].eq(window)
                & effort_cells["effort_background_type"].eq("strict_effort"),
                "grid_id",
            ].astype(str)
        )
        effort_bg = df.loc[df["grid_id"].astype(str).isin(effort_ids), "lcp_frequency_normalized"].fillna(0)
        def pct_top(vals: pd.Series, bg: pd.Series, q: float) -> float:
            if len(vals) == 0 or len(bg) == 0:
                return math.nan
            thr = bg.quantile(q)
            return float((vals >= thr).mean() * 100)
        rows.append(
            {
                "target_year": int(year),
                "validation_window": window,
                "model_layer": layer,
                "prey_variable": variable,
                "record_count": int(len(sub)),
                "total_whales_seen_if_available": float(sub["count_or_best_count"].fillna(0).sum()),
                "effort_covered_background_cell_count": int(len(effort_bg)),
                "all_C3_background_cell_count": int(len(all_bg)),
                "mean_lcp_frequency_at_sightings": float(sight_freq.mean()),
                "median_lcp_frequency_at_sightings": float(sight_freq.median()),
                "mean_effort_background_lcp_frequency": float(effort_bg.mean()) if len(effort_bg) else math.nan,
                "median_effort_background_lcp_frequency": float(effort_bg.median()) if len(effort_bg) else math.nan,
                "mean_all_C3_background_lcp_frequency": float(all_bg.mean()),
                "median_all_C3_background_lcp_frequency": float(all_bg.median()),
                "sighting_mean_minus_effort_background_mean": float(sight_freq.mean() - effort_bg.mean()) if len(effort_bg) else math.nan,
                "sighting_median_minus_effort_background_median": float(sight_freq.median() - effort_bg.median()) if len(effort_bg) else math.nan,
                "percent_sightings_in_effort_top25pct_lcp_cells": pct_top(sight_freq, effort_bg, 0.75),
                "percent_sightings_in_effort_top10pct_lcp_cells": pct_top(sight_freq, effort_bg, 0.90),
                "percent_sightings_in_effort_top5pct_lcp_cells": pct_top(sight_freq, effort_bg, 0.95),
                "percent_sightings_in_all_C3_top25pct_lcp_cells": pct_top(sight_freq, all_bg, 0.75),
                "percent_sightings_in_all_C3_top10pct_lcp_cells": pct_top(sight_freq, all_bg, 0.90),
                "percent_sightings_in_all_C3_top5pct_lcp_cells": pct_top(sight_freq, all_bg, 0.95),
                "notes": "strict_effort_background_is_primary; all_C3_background_is_secondary_QA",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "asamm_phase8i2_background_contrast.csv", index=False)
    return out


def summarize_validation(extraction: pd.DataFrame, distances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_cols = [
        "distance_to_core_corridor_km",
        "distance_to_top5pct_corridor_km",
        "distance_to_persistent_bottleneck_km",
    ]
    merged = extraction.merge(
        distances[["record_id", "year", "validation_window", "model_layer", "prey_variable", *dist_cols]],
        on=["record_id", "year", "validation_window", "model_layer", "prey_variable"],
        how="left",
    )
    rows = []
    for keys, sub in merged.groupby(["model_layer", "prey_variable", "target_year", "validation_window"], dropna=False):
        layer, variable, year, window = keys
        rows.append(
            {
                "model_layer": layer,
                "prey_variable": variable,
                "target_year": int(year),
                "validation_window": window,
                "record_count": int(len(sub)),
                "total_whales_seen_if_available": float(sub["count_or_best_count"].fillna(0).sum()),
                "mean_lcp_frequency_at_sightings": float(sub["lcp_frequency_normalized_at_sighting"].mean()),
                "median_lcp_frequency_at_sightings": float(sub["lcp_frequency_normalized_at_sighting"].median()),
                "mean_lcp_percentile_within_C3_at_sightings": float(sub["lcp_frequency_percentile_within_C3"].mean()),
                "median_lcp_percentile_within_C3_at_sightings": float(sub["lcp_frequency_percentile_within_C3"].median()),
                "mean_lcp_percentile_within_effort_footprint_at_sightings": float(sub["lcp_frequency_percentile_within_strict_effort_footprint"].mean()),
                "median_lcp_percentile_within_effort_footprint_at_sightings": float(sub["lcp_frequency_percentile_within_strict_effort_footprint"].median()),
                "percent_records_on_corridor_freq_gt_0": float(sub["is_corridor_cell_freq_gt_0"].mean() * 100),
                "percent_records_on_core_corridor_freq_gt_0_05": float(sub["is_core_corridor_cell_freq_gt_0_05"].mean() * 100),
                "percent_records_on_high_use_corridor_freq_gt_0_10": float(sub["is_high_use_corridor_cell_freq_gt_0_10"].mean() * 100),
                "percent_records_top5pct_lcp": float(sub["is_top5pct_lcp_cell"].mean() * 100),
                "median_distance_to_core_corridor_km": float(sub["distance_to_core_corridor_km"].median()),
                "median_distance_to_top5pct_corridor_km": float(sub["distance_to_top5pct_corridor_km"].median()),
                "median_distance_to_persistent_bottleneck_km": float(sub["distance_to_persistent_bottleneck_km"].median()),
            }
        )
    annual = pd.DataFrame(rows)
    annual.to_csv(OUT_DIR / "asamm_phase8i2_diagnostic_summary_by_year_season.csv", index=False)

    overall_rows = []
    for keys, sub in merged.groupby(["model_layer", "prey_variable", "validation_window"], dropna=False):
        layer, variable, window = keys
        interpretation = "diagnostic consistency only"
        if sub["lcp_frequency_percentile_within_strict_effort_footprint"].mean() >= 60:
            interpretation = "sightings fall above the effort-footprint median LCP frequency"
        elif sub["lcp_frequency_percentile_within_strict_effort_footprint"].mean() < 50:
            interpretation = "sightings do not exceed the effort-footprint median LCP frequency"
        overall_rows.append(
            {
                "model_layer": layer,
                "prey_variable": variable,
                "validation_window": window,
                "record_count": int(len(sub)),
                "mean_lcp_percentile_within_C3": float(sub["lcp_frequency_percentile_within_C3"].mean()),
                "mean_lcp_percentile_within_effort_footprint": float(sub["lcp_frequency_percentile_within_strict_effort_footprint"].mean()),
                "percent_records_top5pct_lcp": float(sub["is_top5pct_lcp_cell"].mean() * 100),
                "percent_records_on_core_corridor_freq_gt_0_05": float(sub["is_core_corridor_cell_freq_gt_0_05"].mean() * 100),
                "median_distance_to_core_corridor_km": float(sub["distance_to_core_corridor_km"].median()),
                "median_distance_to_top5pct_corridor_km": float(sub["distance_to_top5pct_corridor_km"].median()),
                "interpretation": interpretation,
                "caveat": "ASAMM validation is diagnostic and footprint-aware, not abundance/density/causal response.",
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT_DIR / "asamm_phase8i2_diagnostic_overall_summary.csv", index=False)

    compare = overall.loc[overall["prey_variable"].eq("total_kJm2_frozen")].copy()
    compare.to_csv(OUT_DIR / "asamm_phase8i2_model_layer_alignment_comparison.csv", index=False)
    return annual, overall, compare


def create_bottleneck_extraction(extraction: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    out = extraction.merge(
        distances[
            [
                "record_id",
                "year",
                "validation_window",
                "model_layer",
                "prey_variable",
                "distance_to_persistent_bottleneck_km",
                "inside_or_near_top5pct_25km",
            ]
        ],
        on=["record_id", "year", "validation_window", "model_layer", "prey_variable"],
        how="left",
    )
    out["near_persistent_bottleneck_25km"] = out["distance_to_persistent_bottleneck_km"].le(25)
    out = out[
        [
            "record_id",
            "year",
            "month",
            "validation_window",
            "model_layer",
            "prey_variable",
            "nearest_grid_id",
            "lcp_frequency_normalized_at_sighting",
            "is_top5pct_lcp_cell",
            "inside_or_near_top5pct_25km",
            "distance_to_persistent_bottleneck_km",
            "near_persistent_bottleneck_25km",
            "notes",
        ]
    ]
    out.to_csv(OUT_DIR / "asamm_phase8i2_bottleneck_extraction.csv", index=False)
    return out


def plot_figures(sightings: pd.DataFrame, effort_cells: pd.DataFrame, grid: pd.DataFrame, overall: pd.DataFrame, distances: pd.DataFrame, compare: pd.DataFrame) -> None:
    main = overall.loc[(overall["prey_variable"] == "total_kJm2_frozen") & (overall["validation_window"].isin(["July_September", "July_only", "June_July"]))].copy()
    layer_order = ["prey_only", "prey_ice", "prey_ice_ship"]
    labels = {"prey_only": "Prey-only", "prey_ice": "Prey + ice", "prey_ice_ship": "Prey + ice + ship"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(layer_order))
    width = 0.25
    for i, window in enumerate(["July_September", "July_only", "June_July"]):
        vals = [
            main.loc[(main["model_layer"] == layer) & (main["validation_window"] == window), "mean_lcp_percentile_within_effort_footprint"].mean()
            for layer in layer_order
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=window.replace("_", "-"))
    ax.axhline(50, color="0.5", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[l] for l in layer_order])
    ax.set_ylabel("Mean LCP percentile at ASAMM sightings\nwithin ASAMM effort footprint")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("ASAMM diagnostic alignment with study-domain corridors")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "asamm_phase8i2_lcp_percentile_summary.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for window in ["July_September", "July_only", "June_July"]:
        sub = main.loc[main["validation_window"] == window]
        ax.plot(
            [labels[l] for l in layer_order],
            [sub.loc[sub["model_layer"] == l, "percent_records_top5pct_lcp"].mean() for l in layer_order],
            marker="o",
            label=window.replace("_", "-"),
        )
    ax.set_ylabel("ASAMM records in top 5% LCP cells (%)")
    ax.set_title("Top-5% LCP overlap, diagnostic only")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "asamm_phase8i2_top5pct_overlap_summary.png", dpi=300)
    plt.close(fig)

    dist_main = distances.loc[(distances["prey_variable"] == "total_kJm2_frozen") & (distances["validation_window"].isin(["July_September", "July_only", "June_July"]))]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for window in ["July_September", "July_only", "June_July"]:
        sub = dist_main.loc[dist_main["validation_window"] == window]
        ax.plot(
            [labels[l] for l in layer_order],
            [sub.loc[sub["model_layer"] == l, "distance_to_core_corridor_km"].median() for l in layer_order],
            marker="o",
            label=window.replace("_", "-"),
        )
    ax.set_ylabel("Median distance to frequent corridor cells (km)")
    ax.set_title("Distance from ASAMM sightings to modeled corridors")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "asamm_phase8i2_distance_to_corridor_summary.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(grid["lon"], grid["lat"], s=2, c="lightgray", alpha=0.45, label="C3 grid")
    strict_julsep = effort_cells.loc[
        effort_cells["effort_background_type"].eq("strict_effort")
        & effort_cells["validation_window"].eq("July_September")
    ]
    egrid = grid.loc[grid["grid_id"].astype(str).isin(set(strict_julsep["grid_id"].astype(str)))]
    ax.scatter(egrid["lon"], egrid["lat"], s=5, c="#8ecae6", alpha=0.7, label="ASAMM effort-covered C3 cells")
    s = sightings.loc[sightings["use_for_C3_validation"] & sightings["month"].isin([7, 8, 9])]
    ax.scatter(s["lon"], s["lat"], s=10, c="#bc4749", alpha=0.75, label="ASAMM gray whale sightings")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ASAMM July-Sept gray whale sightings and effort footprint")
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "asamm_phase8i2_sightings_on_C3_grid_preview.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if not compare.empty:
        for layer in layer_order:
            sub = compare.loc[(compare["model_layer"] == layer) & (compare["validation_window"] == "July_September")]
            if not sub.empty:
                ax.bar(labels[layer], sub["mean_lcp_percentile_within_effort_footprint"].mean())
        ax.axhline(50, color="0.5", lw=1, ls="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean effort-footprint percentile")
    ax.set_title("Model-layer alignment comparison, kJ main prey variable")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "asamm_phase8i2_model_layer_alignment_comparison.png", dpi=300)
    plt.close(fig)


def write_docs(qa: dict, overall: pd.DataFrame, background: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_text_once(path: Path, marker: str, text: str) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_registry_manifest() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def main() -> None:
    ensure_dirs()
    grid = pd.read_csv(GRID_PATH)
    c3_poly = load_c3_polygon()
    events = read_asamm_events()
    events_grid = add_domain_flags(assign_to_grid(events, grid), c3_poly)

    gray = events_grid.loc[events_grid["is_gray_whale"]].copy()
    gray = gray.sort_values(["year", "month", "source_file", "ID"]).reset_index(drop=True)
    gray["record_id"] = ["ASAMM_GW_{:05d}".format(i + 1) for i in range(len(gray))]
    keep_cols = [
        "record_id",
        "asamm_event_id",
        "source_file",
        "ID",
        "Event",
        "Flt_No",
        "datetime",
        "year",
        "month",
        "day",
        "lat",
        "lon",
        "Entry",
        "Species",
        "count_or_best_count",
        "Aerial_Survey",
        "FltType",
        "Transect_id",
        "Block",
        "nearest_grid_id",
        "nearest_grid_lon",
        "nearest_grid_lat",
        "distance_to_nearest_grid_cell_km",
        "inside_C3_domain",
        "use_for_C3_validation",
        "season_group",
    ]
    gray[keep_cols].to_csv(OUT_DIR / "asamm_phase8i2_gray_whale_records_on_C3_grid.csv", index=False)

    effort_month, effort_window = build_effort_footprint(events_grid)
    lcp_cache = load_lcp_cache()
    extraction = extract_lcp_values(gray, lcp_cache, effort_window)
    distances = compute_distances(gray, lcp_cache)
    bottleneck = create_bottleneck_extraction(extraction, distances)
    background = summarize_background(extraction, lcp_cache, effort_window)
    annual, overall, compare = summarize_validation(extraction, distances)
    plot_figures(gray, effort_window, grid, overall, distances, compare)

    qa = {
        "phase": PHASE,
        "input_asamm_events_months_6_9": int(len(events)),
        "gray_whale_records_months_6_9": int(len(gray)),
        "gray_whale_records_inside_C3": int(gray["inside_C3_domain"].sum()),
        "gray_whale_records_used_for_C3_validation": int(gray["use_for_C3_validation"].sum()),
        "gray_whale_records_july_september_used": int((gray["use_for_C3_validation"] & gray["month"].isin([7, 8, 9])).sum()),
        "gray_whale_records_july_only_used": int((gray["use_for_C3_validation"] & gray["month"].eq(7)).sum()),
        "strict_effort_cells_july_september": int(
            effort_window.loc[
                effort_window["effort_background_type"].eq("strict_effort")
                & effort_window["validation_window"].eq("July_September"),
                "grid_id",
            ].nunique()
        ),
        "model_layers_used": list(MODEL_LAYERS),
        "prey_variables_used": list(PREY_VARIABLES),
        "validation_windows": VALIDATION_WINDOWS,
        "warnings": [
            "ASAMM validation is diagnostic only and not abundance or density.",
            "Strict effort footprint is reconstructed from event entries, not full trackline geometry.",
            "July-September observed window is seasonally later than May-July model layers.",
            "Whole-C3 background is secondary QA; effort-covered background is primary.",
        ],
    }
    (OUT_DIR / "asamm_phase8i2_lcp_extraction_qa.json").write_text(json.dumps(qa, indent=2))
    write_docs(qa, overall, background)
    update_registry_manifest()


if __name__ == "__main__":
    main()
