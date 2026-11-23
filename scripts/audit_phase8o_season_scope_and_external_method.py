"""Phase 8O seasonal-scope and external-method comparison audit.

This isolated audit tests whether August-October can be appended to the
existing May-July northbound corridor analysis without redesigning the
scientific question. It also compares the current gray-whale workflow with
Szesciorka et al. (2026). It does not rerun corridor models or edit manuscript,
Supplementary Information, or figure files.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import unary_union


PHASE = "Phase 8O Seasonal-scope and external-method comparison audit"
SCRIPT = "scripts/audit_phase8o_season_scope_and_external_method.py"
TARGET_YEARS = list(range(2011, 2020))
AUDIT_MONTHS = list(range(5, 11))

OUT = ROOT / "outputs/season_scope_phase8o"
TABLE_OUT = OUT / "tables"
STATUS_PATH = OUT / "phase8o_run_status.json"
MANIFEST_PATH = OUT / "phase8o_analysis_manifest.json"
SUMMARY_PATH = OUT / "phase8o_season_scope_summary.md"

ASAMM_ROOT = ROOT / "data/raw/validation/asamm_1979_2021/csv_export/ASAMM to csv"
ASAMM_OUTPUTS = sorted(ASAMM_ROOT.glob("*/_Output__*.csv"))
GRID_PATH = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
DOMAIN_PATH = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
ICE_INVENTORY = ROOT / "data/processed/ice/ice_data_inventory_v0.csv"
SHIP_INVENTORY = ROOT / "data/processed/ship/ship_data_inventory_v0.csv"
PREY_SURFACE_PATTERN = "outputs/prey_v2/amphipod_prey_surface_total_kJm2_frozen_target*_full_trailing_C3_v2.csv"
SZESCIOKA_PDF = ROOT / "docs/coauthor reviews rounds/Szesciorka.etal.2026.BH&vesseltraffic-FMARS.pdf"

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
    "Aerial_Survey",
    "Block",
]

WINDOWS = {
    "model_season_May_July": [5, 6, 7],
    "current_validation_June_July": [6, 7],
    "existing_open_water_context_July_September": [7, 8, 9],
    "late_summer_August": [8],
    "fall_transition_September_October": [9, 10],
    "candidate_late_season_August_October": [8, 9, 10],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_stage(stage: str, status: str, **details: object) -> None:
    if STATUS_PATH.exists():
        state = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "phase": PHASE,
            "started_utc": utc_now(),
            "status": "running",
            "stages": {},
            "manuscript_or_si_modified": False,
            "existing_figure_modified": False,
            "corridor_model_rerun": False,
        }
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    if status == "failed":
        state["status"] = "failed"
    write_json(STATUS_PATH, state)


def normalize_lon(values: pd.Series) -> pd.Series:
    lon = pd.to_numeric(values, errors="coerce")
    return ((lon + 180) % 360) - 180


def best_lat_lon(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lat = pd.to_numeric(frame.get("ArcLat"), errors="coerce")
    lon = normalize_lon(frame.get("ArcLong"))
    for lat_name, lon_name in [("LatTemp", "LongTemp"), ("Lat", "Long")]:
        alt_lat = pd.to_numeric(frame.get(lat_name), errors="coerce")
        alt_lon = normalize_lon(frame.get(lon_name))
        lat = lat.where(lat.notna(), alt_lat)
        lon = lon.where(lon.notna(), alt_lon)
    return lat, lon


def is_effort_entry(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"", ".", "nan", "none"} or "deadhead" in text:
        return False
    return any(
        token in text
        for token in ["transect", "search", "circling", "resume", "start", "end", "fov", "caps", "fgf", "divert"]
    )


def load_domain():
    data = json.loads(DOMAIN_PATH.read_text(encoding="utf-8"))
    if data.get("type") in {"Polygon", "MultiPolygon"}:
        return shape(data)
    geoms = [shape(feature["geometry"]) for feature in data.get("features", []) if feature.get("geometry")]
    return unary_union(geoms)


def read_asamm_events() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in ASAMM_OUTPUTS:
        for chunk in pd.read_csv(
            path,
            usecols=lambda column: column in OUTPUT_COLS,
            dtype=str,
            chunksize=150_000,
            low_memory=False,
        ):
            dt = pd.to_datetime(chunk["GMT_Minus8_DateTime"], errors="coerce")
            keep = dt.dt.year.between(min(TARGET_YEARS), max(TARGET_YEARS)) & dt.dt.month.isin(AUDIT_MONTHS)
            if not keep.any():
                continue
            sub = chunk.loc[keep].copy()
            sub["datetime"] = dt.loc[keep]
            sub["year"] = sub["datetime"].dt.year.astype(int)
            sub["month"] = sub["datetime"].dt.month.astype(int)
            sub["lat"], sub["lon"] = best_lat_lon(sub)
            sub = sub.loc[sub["lat"].between(50, 75) & sub["lon"].between(-180, -120)].copy()
            if sub.empty:
                continue
            sub["source_file"] = str(path.relative_to(ROOT))
            sub["source_row"] = sub.index.astype(int)
            sub["event_uid"] = path.parent.name + ":" + sub["source_row"].astype(str)
            sub["species_clean"] = sub["Species"].fillna("").str.strip().str.lower()
            sub["is_gray_whale"] = sub["species_clean"].eq("gray whale")
            sub["is_effort_entry"] = sub["Entry"].map(is_effort_entry)
            final_group = pd.to_numeric(sub.get("Final_group"), errors="coerce")
            total_number = pd.to_numeric(sub.get("Totalno"), errors="coerce")
            sub["count_or_best_count"] = final_group.where(final_group.gt(0), total_number)
            sub["count_or_best_count"] = sub["count_or_best_count"].where(sub["count_or_best_count"].gt(0))
            frames.append(sub)
    if not frames:
        raise RuntimeError("No ASAMM events found for 2011-2019 and May-October")
    events = pd.concat(frames, ignore_index=True)
    if events["event_uid"].duplicated().any():
        raise RuntimeError("ASAMM event_uid is not unique")
    return events


def assign_spatial_context(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(GRID_PATH, dtype={"grid_id": str})
    domain = load_domain()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    grid_x, grid_y = transformer.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    event_x, event_y = transformer.transform(events["lon"].to_numpy(), events["lat"].to_numpy())
    tree = cKDTree(np.column_stack([grid_x, grid_y]))
    distance, index = tree.query(np.column_stack([event_x, event_y]), k=1)
    matched = grid.iloc[index].reset_index(drop=True)

    out = events.reset_index(drop=True).copy()
    out["nearest_grid_id"] = matched["grid_id"].to_numpy()
    out["nearest_grid_lon"] = matched["lon"].to_numpy()
    out["nearest_grid_lat"] = matched["lat"].to_numpy()
    out["distance_to_nearest_grid_cell_km"] = distance / 1000.0
    out["inside_study_region"] = [
        domain.covers(Point(lon, lat)) for lon, lat in zip(out["lon"], out["lat"])
    ]
    out["use_for_study_region_sighting_summary"] = (
        out["inside_study_region"] & out["distance_to_nearest_grid_cell_km"].le(15)
    )
    return out, grid


def monthly_scope_table(events: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in TARGET_YEARS:
        for month in AUDIT_MONTHS:
            annual_month = events.loc[(events["year"] == year) & (events["month"] == month)]
            gray_all = annual_month.loc[annual_month["is_gray_whale"]]
            gray_region = gray_all.loc[gray_all["use_for_study_region_sighting_summary"]]
            effort = annual_month.loc[annual_month["inside_study_region"] & annual_month["is_effort_entry"]]
            effort_cells = effort["nearest_grid_id"].dropna().astype(str).nunique()
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "month_name": pd.Timestamp(2000, month, 1).month_name(),
                    "all_gray_whale_record_count": len(gray_all),
                    "study_region_gray_whale_record_count": len(gray_region),
                    "descriptive_observed_individual_sum": gray_region["count_or_best_count"].sum(min_count=1),
                    "gray_whale_unique_flights": gray_region["Flt_No"].nunique(dropna=True),
                    "strict_effort_event_count": len(effort),
                    "strict_effort_cell_count": effort_cells,
                    "strict_effort_unique_flights": effort["Flt_No"].nunique(dropna=True),
                    "effort_grid_fraction": effort_cells / len(grid),
                    "gray_records_per_100_effort_cells": (100 * len(gray_region) / effort_cells) if effort_cells else np.nan,
                    "record_count_is_abundance": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "phase8o_asamm_monthly_scope_audit_2011_2019.csv", index=False)
    return out


def event_sets(events: pd.DataFrame, year: int, months: list[int]) -> tuple[set[str], set[str]]:
    subset = events.loc[(events["year"] == year) & events["month"].isin(months)]
    gray = subset.loc[subset["is_gray_whale"] & subset["use_for_study_region_sighting_summary"]]
    effort = subset.loc[subset["inside_study_region"] & subset["is_effort_entry"]]
    return set(gray["event_uid"].astype(str)), set(effort["nearest_grid_id"].dropna().astype(str))


def window_scope_tables(events: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    footprint_sets: dict[tuple[str, int], set[str]] = {}
    for window, months in WINDOWS.items():
        for year in TARGET_YEARS:
            subset = events.loc[(events["year"] == year) & events["month"].isin(months)]
            gray = subset.loc[subset["is_gray_whale"] & subset["use_for_study_region_sighting_summary"]]
            effort = subset.loc[subset["inside_study_region"] & subset["is_effort_entry"]]
            cells = set(effort["nearest_grid_id"].dropna().astype(str))
            footprint_sets[(window, year)] = cells
            rows.append(
                {
                    "window": window,
                    "months": ";".join(map(str, months)),
                    "year": year,
                    "study_region_gray_whale_record_count": len(gray),
                    "descriptive_observed_individual_sum": gray["count_or_best_count"].sum(min_count=1),
                    "strict_effort_event_count": len(effort),
                    "strict_effort_cell_count": len(cells),
                    "effort_grid_fraction": len(cells) / len(grid),
                    "gray_records_per_100_effort_cells": (100 * len(gray) / len(cells)) if cells else np.nan,
                }
            )
    annual = pd.DataFrame(rows)
    overall_rows: list[dict[str, object]] = []
    for window, months in WINDOWS.items():
        subset = events.loc[events["month"].isin(months)]
        gray = subset.loc[subset["is_gray_whale"] & subset["use_for_study_region_sighting_summary"]]
        effort = subset.loc[subset["inside_study_region"] & subset["is_effort_entry"]]
        annual_sub = annual.loc[annual["window"] == window]
        overall_rows.append(
            {
                "window": window,
                "months": ";".join(map(str, months)),
                "years_with_sightings": int((annual_sub["study_region_gray_whale_record_count"] > 0).sum()),
                "years_with_effort": int((annual_sub["strict_effort_event_count"] > 0).sum()),
                "study_region_gray_whale_record_count": len(gray),
                "descriptive_observed_individual_sum": gray["count_or_best_count"].sum(min_count=1),
                "strict_effort_event_count": len(effort),
                "union_effort_cell_count": effort["nearest_grid_id"].dropna().astype(str).nunique(),
                "median_annual_effort_cell_count": annual_sub["strict_effort_cell_count"].median(),
                "median_annual_gray_records_per_100_effort_cells": annual_sub["gray_records_per_100_effort_cells"].median(),
            }
        )
    overall = pd.DataFrame(overall_rows)
    annual.to_csv(TABLE_OUT / "phase8o_asamm_window_scope_by_year.csv", index=False)
    overall.to_csv(TABLE_OUT / "phase8o_asamm_window_scope_summary.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    pairs = [
        ("current_validation_June_July", "candidate_late_season_August_October"),
        ("model_season_May_July", "candidate_late_season_August_October"),
        ("current_validation_June_July", "existing_open_water_context_July_September"),
    ]
    for year in TARGET_YEARS:
        for first, second in pairs:
            a = footprint_sets[(first, year)]
            b = footprint_sets[(second, year)]
            union = a | b
            comparison_rows.append(
                {
                    "year": year,
                    "first_window": first,
                    "second_window": second,
                    "first_effort_cells": len(a),
                    "second_effort_cells": len(b),
                    "shared_effort_cells": len(a & b),
                    "effort_footprint_jaccard": len(a & b) / len(union) if union else np.nan,
                    "first_cells_retained_in_second_percent": 100 * len(a & b) / len(a) if a else np.nan,
                    "second_cells_also_in_first_percent": 100 * len(a & b) / len(b) if b else np.nan,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLE_OUT / "phase8o_asamm_window_footprint_overlap.csv", index=False)
    return overall, comparison


def effort_stability(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in AUDIT_MONTHS:
        sets: dict[int, set[str]] = {}
        for year in TARGET_YEARS:
            subset = events.loc[
                (events["year"] == year)
                & (events["month"] == month)
                & events["inside_study_region"]
                & events["is_effort_entry"]
            ]
            cells = set(subset["nearest_grid_id"].dropna().astype(str))
            if cells:
                sets[year] = cells
        if sets:
            union = set().union(*sets.values())
            intersection = set.intersection(*sets.values())
            counts: dict[str, int] = {}
            for cells in sets.values():
                for cell in cells:
                    counts[cell] = counts.get(cell, 0) + 1
            threshold = math.ceil(len(sets) / 2)
            recurrent = {cell for cell, count in counts.items() if count >= threshold}
            consecutive = []
            for first, second in zip(TARGET_YEARS[:-1], TARGET_YEARS[1:]):
                if first in sets and second in sets:
                    denominator = sets[first] | sets[second]
                    consecutive.append(len(sets[first] & sets[second]) / len(denominator) if denominator else np.nan)
        else:
            union, intersection, recurrent, consecutive = set(), set(), set(), []
            threshold = 0
        rows.append(
            {
                "month": month,
                "month_name": pd.Timestamp(2000, month, 1).month_name(),
                "years_with_effort": len(sets),
                "effort_years": ";".join(map(str, sets)),
                "union_effort_cells": len(union),
                "cells_covered_in_every_available_year": len(intersection),
                "at_least_half_year_threshold": threshold,
                "cells_covered_in_at_least_half_of_available_years": len(recurrent),
                "median_consecutive_year_footprint_jaccard": float(np.nanmedian(consecutive)) if consecutive else np.nan,
                "min_consecutive_year_footprint_jaccard": float(np.nanmin(consecutive)) if consecutive else np.nan,
                "max_consecutive_year_footprint_jaccard": float(np.nanmax(consecutive)) if consecutive else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "phase8o_asamm_effort_stability_by_month.csv", index=False)
    return out


def data_layer_transferability(monthly: pd.DataFrame) -> pd.DataFrame:
    prey_files = sorted(ROOT.glob(PREY_SURFACE_PATTERN))
    ice = pd.read_csv(ICE_INVENTORY)
    ship = pd.read_csv(SHIP_INVENTORY)
    asamm_months = sorted(monthly.loc[monthly["strict_effort_event_count"] > 0, "month"].unique())
    rows = [
        {
            "data_layer": "benthic amphipod prey",
            "available_years": "2011-2019 target surfaces; 2007-2019 trailing support",
            "available_months": "annual target-year surface, not month-specific",
            "existing_model_use": "May-July biological support",
            "august_october_data_available": len(prey_files) == 9,
            "can_reuse_without_redefinition": "conditional",
            "reason": "The nine annual surfaces exist, but they do not resolve within-year prey change or late-season prey switching.",
        },
        {
            "data_layer": "sea-ice concentration",
            "available_years": str(ice.iloc[0]["year_range"]) if "year_range" in ice.columns else "2010-2021",
            "available_months": str(ice.iloc[0]["months_present"]) if "months_present" in ice.columns else "1-12",
            "existing_model_use": "May-July accessibility modifier",
            "august_october_data_available": True,
            "can_reuse_without_redefinition": "no",
            "reason": "Monthly SIC is available, but the ecological role and expected leverage of ice differ in late open water and fall freeze-up.",
        },
        {
            "data_layer": "AIS-derived vessel track length",
            "available_years": f"{int(ship['year'].min())}-{int(ship['year'].max())}",
            "available_months": ";".join(map(str, sorted(ship["month"].unique()))),
            "existing_model_use": "May-July exposure modifier, 2015-2019",
            "august_october_data_available": set([8, 9, 10]).issubset(set(ship["month"].unique())),
            "can_reuse_without_redefinition": "conditional",
            "reason": "Monthly rasters are available, but a late-season analysis must define whether track length represents exposure, impedance, or strike-risk screening.",
        },
        {
            "data_layer": "ASAMM sightings and effort",
            "available_years": "2011-2019 in current common target period",
            "available_months": ";".join(map(str, asamm_months)),
            "existing_model_use": "June-July effort-conditioned observed-use comparison",
            "august_october_data_available": set([8, 9, 10]).issubset(set(asamm_months)),
            "can_reuse_without_redefinition": "conditional",
            "reason": "Late-season records exist, but survey footprints vary and sightings represent foraging/residency plus migration transition rather than a single northbound process.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "phase8o_data_layer_season_transferability.csv", index=False)
    return out


def season_scope_decision_matrix() -> pd.DataFrame:
    rows = [
        ["seasonal biological question", "May-July northward access among prey-supported areas", "August-October summer/fall use and eventual southward departure", "fails direct transfer", "Define a distinct late-season question."],
        ["movement direction", "south-to-north", "residency/foraging dominates much of August-September; southward transition begins in fall", "fails direct transfer", "Do not retain northbound interpretation by default."],
        ["source and destination endpoints", "fixed southern sources and northern destinations", "late-season origins and destinations are not represented by the current endpoint pools", "fails direct transfer", "Select and justify new directional or occupancy-based design."],
        ["prey representation", "annual benthic amphipod support", "benthic and pelagic prey use may vary within the late season", "conditional", "Use annual benthic support only with an explicit seasonal limitation."],
        ["sea-ice role", "accessibility during early northward entry", "largely open-water context in August and variable fall freeze-up in September-October", "requires redefinition", "Rebuild the ice hypothesis rather than reusing the May-July modifier unchanged."],
        ["vessel data", "May-July AIS track length, 2015-2019", "August-October rasters exist for 2015-2019", "data pass; inference conditional", "Choose exposure, impedance, or strike-risk metrics before modeling."],
        ["whale observations", "June-July ASAMM observed-use consistency", "August-October ASAMM records and effort exist but mix behavioral states", "data pass; inference conditional", "Use effort-aware late-season behavior or distribution analysis."],
        ["same-paper necessity", "tests the stated early-season northbound hypothesis", "answers a second seasonal movement question", "not required for completeness", "Treat as a future/separate analysis unless the paper's scope is deliberately expanded."],
    ]
    out = pd.DataFrame(rows, columns=["criterion", "current_May_July_design", "August_October_context", "transferability_result", "required_action"])
    out.to_csv(TABLE_OUT / "phase8o_season_scope_decision_matrix.csv", index=False)
    return out


def method_comparison_table() -> pd.DataFrame:
    rows = [
        ["species and ice ecology", "Eastern North Pacific gray whale; open-water migrant and flexible forager", "Bering-Chukchi-Beaufort bowhead whale; ice-associated Arctic cetacean", "not directly interchangeable"],
        ["season", "May-July; early northward access", "September-March; fall-spring distribution around a low-winter-ice event", "different seasonal processes"],
        ["movement representation", "1,600 annual least-cost paths between fixed south-to-north endpoint pools", "monthly 95% home-range and 50% core-use utilization distributions from satellite telemetry", "modeled potential connectivity versus empirical space use"],
        ["prey", "annual benthic amphipod energy-density support", "no prey layer", "our prey component is an added biological mechanism"],
        ["sea ice", "GLORYS12 monthly SIC converted to a May-July prey-accessibility modifier", "NSIDC sea-ice extent/SIC summarized by month and event period, with a 15% ice-edge contour", "modifier versus observed environmental context"],
        ["whale observations", "ASAMM aerial sightings used for effort-conditioned observed-use comparison", "satellite-tag locations used to construct the response distribution itself", "independent validation versus model input"],
        ["vessel source", "Kapsar et al. 10-km monthly AIS-derived track-length rasters, 2015-2020", "PAME Level-3 AIS positions and reconstructed vessel transits, 2013-2019", "different source products and processing"],
        ["vessel metrics", "log-scaled all-vessel track-length exposure and bounded modifier", "transit density, mean/SD speed, and total transit time within whale UDs", "relative impedance/exposure versus co-occurrence and strike-risk factors"],
        ["spatial grain", "10-km ocean grid", "10 x 10 km risk-factor grid plus telemetry UDs", "similar grid grain, different analytical unit"],
        ["behavioral inference", "scenario comparison; ASAMM alignment does not estimate avoidance", "observed distribution shifts relative to ice; overlap is not direct strike risk", "both require restraint, for different reasons"],
        ["management output", "recurrent biological cells and exposure-priority intersections", "UD overlap with transit density/speed/time and candidate strike-risk areas", "complementary screening outputs"],
    ]
    out = pd.DataFrame(rows, columns=["dimension", "current_gray_whale_analysis", "Szesciorka_et_al_2026", "comparison_implication"])
    out.to_csv(TABLE_OUT / "phase8o_current_vs_szesciorka2026_method_comparison.csv", index=False)
    return out


def literature_evidence_table() -> pd.DataFrame:
    rows = [
        ["8O-LIT-01", "Szesciorka et al. 2026", "Changes in sea ice influence bowhead whale distribution and overlap with vessel transits in the Pacific Arctic", "10.3389/fmars.2026.1818623", "Methods and Discussion", "The analysis used September-March bowhead telemetry UDs, NSIDC ice, and PAME AIS transit density, speed, and transit time; it did not include prey or least-cost paths.", "Defines the external-method comparison."],
        ["8O-LIT-02", "NOAA Fisheries ASAMM", "Aerial Surveys of Arctic Marine Mammals", "https://www.fisheries.noaa.gov/alaska/marine-mammal-protection/aerial-surveys-arctic-marine-mammals", "History and purpose", "ASAMM/COMIDA surveys monitored the open-water season from mid-June or early July through October, with consistent protocols through 2019.", "Supports the late-season data-availability audit."],
        ["8O-LIT-03", "NOAA Fisheries", "1979-2021 Aerial Surveys of Arctic Marine Mammals Historical Database", "https://www.fisheries.noaa.gov/resource/data/1979-2021-aerial-surveys-arctic-marine-mammals-historical-database", "Database documentation", "The historical database contains public ASAMM observations and is periodically updated; version-aware attribution is required.", "Supports source provenance and local raw-data use."],
        ["8O-LIT-04", "Clarke, Kennedy and Ferguson 2016", "Bowhead and Gray Whale Distributions, Sighting Rates, and Habitat Associations in the Eastern Chukchi Sea, Summer and Fall 2009-15", "10.14430/arctic4597", "Abstract", "Gray whales were observed in July-October and used the eastern Chukchi Sea primarily for feeding in both summer and fall.", "Shows that late-season observations concern feeding-area use, not simply continuation of a northbound corridor."],
        ["8O-LIT-05", "Urbán et al. 2021", "Migratory Behavior of an Eastern North Pacific Gray Whale From Baja California Sur to Chirikov Basin, Alaska", "10.3389/fmars.2021.619290", "Introduction and results", "The final northbound phase reaches northern feeding areas by about June, while whales remain on northern feeding grounds through early fall before southward migration.", "Separates early northbound access from late-season residency and departure."],
        ["8O-LIT-06", "Moore et al. 2022", "Changes in gray whale phenology and distribution related to prey variability and ocean biophysics in the northern Bering and eastern Chukchi seas", "10.1371/journal.pone.0265934", "Methods and results", "The study analyzed July-September ASAMM observations and treated whale distributions as responses to prey and oceanographic variability.", "Supports using late-season data as a distinct distribution/foraging analysis."],
    ]
    out = pd.DataFrame(rows, columns=["evidence_id", "source", "title", "doi_or_url", "source_location", "evidence_paraphrase", "supports"])
    out.to_csv(TABLE_OUT / "phase8o_literature_evidence.csv", index=False)
    return out




def integrity_table() -> pd.DataFrame:
    paths = ASAMM_OUTPUTS + [GRID_PATH, DOMAIN_PATH, ICE_INVENTORY, SHIP_INVENTORY, SZESCIOKA_PDF]
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path) if path.stat().st_size < 25_000_000 else "not_computed_for_large_read_only_input",
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    monthly: pd.DataFrame,
    windows: pd.DataFrame,
    footprint: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    june_july = windows.loc[windows["window"] == "current_validation_June_July"].iloc[0]
    aug_oct = windows.loc[windows["window"] == "candidate_late_season_August_October"].iloc[0]
    pair = footprint.loc[
        (footprint["first_window"] == "current_validation_June_July")
        & (footprint["second_window"] == "candidate_late_season_August_October")
    ]
    month_totals = monthly.groupby(["month", "month_name"], as_index=False).agg(
        records=("study_region_gray_whale_record_count", "sum"),
        effort_events=("strict_effort_event_count", "sum"),
    )
    month_lines = "\n".join(
        f"- {row.month_name}: {int(row.records)} study-region gray-whale records and {int(row.effort_events)} strict-effort events."
        for row in month_totals.itertuples(index=False)
    )
    text = f"""# Phase 8O Seasonal-scope Summary

## Scope

The seasonal-scope analysis summarizes August-October observation and environmental coverage and compares methods with Szesciorka et al. (2026).

## ASAMM Coverage

{month_lines}

The current June-July validation window contains {int(june_july.study_region_gray_whale_record_count)} study-region sighting records. The candidate August-October window contains {int(aug_oct.study_region_gray_whale_record_count)} records. These are sighting-record totals and descriptive group-count sums, not abundance estimates.

The median annual Jaccard overlap between the June-July and August-October strict effort footprints is {pair.effort_footprint_jaccard.median():.3f} (range {pair.effort_footprint_jaccard.min():.3f}-{pair.effort_footprint_jaccard.max():.3f}). The survey background therefore changes enough that late-season observations cannot be treated as a drop-in extension of the current validation window.

## Decision

August-October data are available, but the current northbound model is not directly transferable. Late summer is dominated by feeding-area use and residency, while fall includes changing residence and southward departure. A defensible late-season analysis requires a new movement hypothesis, endpoint design, seasonal ice interpretation, and observation model. It is not required to make the present early-season northbound paper complete.

Szesciorka et al. (2026) is complementary rather than a direct template. That study uses bowhead telemetry utilization distributions, September-March NSIDC ice, and reconstructed PAME AIS transit density, speed, and transit time. The gray-whale study uses prey-supported least-cost connectivity, GLORYS SIC as an accessibility modifier, Kapsar AIS track-length exposure, and independent ASAMM observations. The comparison supports clearer separation of vessel exposure, modeled impedance, observed displacement, and strike-risk metrics; it does not identify an error in the present model.

## Output Entry Points

- `tables/phase8o_asamm_monthly_scope_audit_2011_2019.csv`
- `tables/phase8o_asamm_window_scope_summary.csv`
- `tables/phase8o_asamm_window_footprint_overlap.csv`
- `tables/phase8o_asamm_effort_stability_by_month.csv`
- `tables/phase8o_data_layer_season_transferability.csv`
- `tables/phase8o_season_scope_decision_matrix.csv`
- `tables/phase8o_current_vs_szesciorka2026_method_comparison.csv`
- `tables/phase8o_literature_evidence.csv`
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def finalize_manifest(input_before: pd.DataFrame, input_after: pd.DataFrame) -> None:
    input_before.to_csv(OUT / "input_integrity_before.csv", index=False)
    input_after.to_csv(OUT / "input_integrity_after.csv", index=False)
    comparison = input_before.merge(input_after, on="file_path", suffixes=("_before", "_after"), how="outer")
    comparison["unchanged"] = (
        comparison["size_bytes_before"].eq(comparison["size_bytes_after"])
        & comparison["mtime_ns_before"].eq(comparison["mtime_ns_after"])
        & comparison["sha256_before"].eq(comparison["sha256_after"])
    )
    comparison.to_csv(OUT / "input_integrity_comparison.csv", index=False)
    if not comparison["unchanged"].all():
        raise RuntimeError("At least one read-only Phase 8O input changed during the audit")

    outputs = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != MANIFEST_PATH:
            outputs.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        MANIFEST_PATH,
        {
            "phase": PHASE,
            "script": SCRIPT,
            "completed_utc": utc_now(),
            "status": "complete_season_extension_requires_redesign",
            "decision": "retain_May_July_northbound_scope; treat_August_October_as_distinct_future_analysis",
            "manuscript_or_si_modified": False,
            "existing_figure_modified": False,
            "corridor_model_rerun": False,
            "outputs": outputs,
        },
    )


def main() -> None:
    ensure_dirs()
    set_stage("8O.0_input_freeze", "running")
    before = integrity_table()
    set_stage("8O.0_input_freeze", "complete", input_count=len(before))

    set_stage("8O.1_asamm_monthly_scope", "running")
    events = read_asamm_events()
    events, grid = assign_spatial_context(events)
    monthly = monthly_scope_table(events, grid)
    windows, footprint = window_scope_tables(events, grid)
    stability = effort_stability(events)
    set_stage(
        "8O.1_asamm_monthly_scope",
        "complete",
        event_rows=len(events),
        study_region_gray_records=int(monthly["study_region_gray_whale_record_count"].sum()),
    )

    set_stage("8O.2_transferability_gate", "running")
    data_layer_transferability(monthly)
    season_scope_decision_matrix()
    set_stage("8O.2_transferability_gate", "complete", decision="requires_new_late_season_design")

    set_stage("8O.3_external_method_comparison", "running")
    method_comparison_table()
    literature_evidence_table()
    set_stage("8O.3_external_method_comparison", "complete", source="Szesciorka_et_al_2026")

    write_summary(monthly, windows, footprint, stability)

    set_stage("8O.5_integrity_and_manifest", "running")
    after = integrity_table()
    finalize_manifest(before, after)
    state = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    state["status"] = "complete"
    state["completed_utc"] = utc_now()
    state["decision"] = "retain_May_July_northbound_scope; late_season_requires_separate_design"
    state["stages"]["8O.5_integrity_and_manifest"] = {"status": "complete", "updated_utc": utc_now()}
    write_json(STATUS_PATH, state)


if __name__ == "__main__":
    main()
