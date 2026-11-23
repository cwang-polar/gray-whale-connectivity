"""Phase 8N comprehensive euphausiid data discovery and integration audit.

This isolated diagnostic harmonizes public station observations without
interpolating a pelagic-prey surface or rerunning least-cost paths. Late-season
abundance, target-window prey quality, and broad-group observations remain
separate because they answer different ecological questions.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


PHASE = "Phase 8N comprehensive euphausiid data discovery and integration audit"
SCRIPT = "scripts/audit_phase8n_comprehensive_euphausiid_integration.py"
TARGET_YEARS = list(range(2011, 2020))
TARGET_MONTHS = {5, 6, 7}
GRID_MATCH_LIMIT_KM = 15.0

OUT = ROOT / "outputs/pelagic_prey_phase8n"
RAW = OUT / "raw_external"
TABLE_OUT = OUT / "tables"
FIG_OUT = OUT / "figures"
STATUS_PATH = OUT / "phase8n_run_status.json"
MANIFEST_PATH = OUT / "phase8n_analysis_manifest.json"

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
DOMAIN = ROOT / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson"
DBO = ROOT / "data/processed/study_domain/dbo_anchor_polygons.gpkg"
LAND = ROOT / "data/raw/prey/ne_10m_land/ne_10m_land.shp"

PALETTE = {
    "text": "#283036",
    "grid": "#D4DADD",
    "land": "#D9D3C7",
    "ocean": "#EFF5F7",
    "domain": "#315E6B",
    "CSESP": "#3B7890",
    "Point Barrow": "#7B6BA7",
    "AMBON": "#2B8A6E",
    "Arctic IERP": "#CE7C34",
    "ASGARD abundance": "#8E4A72",
    "ASGARD quality": "#B83C63",
    "IERP synthesis": "#6B5A95",
    "late": "#83939C",
    "target_abundance": "#8E4A72",
    "target_quality": "#B83C63",
    "regional_index": "#4C7C84",
    "partial_target_broad": "#D6A43B",
    "published_only": "#7B6BA7",
    "pass": "#2B8A6E",
    "partial": "#D6A43B",
    "fail": "#C75B54",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def set_stage(stage: str, status: str, **details: object) -> None:
    if STATUS_PATH.exists():
        state = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "phase": PHASE,
            "started_utc": utc_now(),
            "status": "running",
            "stages": {},
        }
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    if status == "failed":
        state["status"] = "failed"
    write_json(STATUS_PATH, state)


def source_inventory() -> pd.DataFrame:
    rows = [
        {
            "source_id": "Iwahara2025",
            "source_title": "Relationship Between the Distribution of Gray Whales and Two Types of Prey in the Northern Bering and Chukchi Seas",
            "years_in_or_near_target": "2012;2013",
            "sampling_months": "2012 Sep-Oct; 2013 Jul",
            "geographic_role": "localized DBO3 and Pacific Arctic stations",
            "euphausiid_variables": "abundance; localized whale-prey association",
            "raw_station_data_downloaded": False,
            "target_window_fit": "partial",
            "spatial_surface_fit": "no",
            "phase8n_role": "published mechanistic evidence; not an annual layer",
            "doi_or_url": "https://doi.org/10.14430/arctic83529",
        },
        {
            "source_id": "CSESP2008_2014",
            "source_title": "Chukchi Sea Environmental Studies Program zooplankton",
            "years_in_or_near_target": "2011-2014",
            "sampling_months": "Aug-Oct",
            "geographic_role": "northeastern Chukchi shelf",
            "euphausiid_variables": "abundance per m3; dry biomass mg per m3; taxon and life stage",
            "raw_station_data_downloaded": True,
            "target_window_fit": "no",
            "spatial_surface_fit": "quantitative late-season observations only",
            "phase8n_role": "quantitative late-season spatial evidence",
            "doi_or_url": "https://doi.org/10.25921/04ty-v962",
        },
        {
            "source_id": "PointBarrow2011_2015",
            "source_title": "Point Barrow and Barrow Canyon mesozooplankton abundance",
            "years_in_or_near_target": "2011-2015",
            "sampling_months": "Aug-Sep",
            "geographic_role": "Barrow Canyon and adjacent Chukchi-Beaufort shelves",
            "euphausiid_variables": "Thysanoessa species-stage abundance per m3",
            "raw_station_data_downloaded": True,
            "target_window_fit": "no",
            "spatial_surface_fit": "local late-season time series",
            "phase8n_role": "quantitative DBO5/northern-alternative evidence",
            "doi_or_url": "https://doi.org/10.1371/journal.pone.0254418",
        },
        {
            "source_id": "AMBON2015",
            "source_title": "Arctic Marine Biodiversity Observing Network zooplankton 2015",
            "years_in_or_near_target": "2015",
            "sampling_months": "Aug-Sep",
            "geographic_role": "US Chukchi shelf",
            "euphausiid_variables": "abundance per m3; dry biomass mg per m3",
            "raw_station_data_downloaded": True,
            "target_window_fit": "no",
            "spatial_surface_fit": "no quantitative use from downloaded derivative; original CSV retrieval pending",
            "phase8n_role": "occurrence footprint only; downloaded occurrence-to-measurement identifiers are not reliably linked",
            "doi_or_url": "https://doi.org/10.25921/qxcs-0x08",
        },
        {
            "source_id": "AMBON2017",
            "source_title": "Arctic Marine Biodiversity Observing Network zooplankton 2017",
            "years_in_or_near_target": "2017",
            "sampling_months": "Aug",
            "geographic_role": "US Chukchi shelf",
            "euphausiid_variables": "abundance per m3; dry biomass mg per m3",
            "raw_station_data_downloaded": True,
            "target_window_fit": "no",
            "spatial_surface_fit": "late-season observations only",
            "phase8n_role": "quantitative late-season spatial evidence",
            "doi_or_url": "https://api.obis.org/v3/dataset/bc01451e-d990-4ad1-8315-e3fb6e9cf461",
        },
        {
            "source_id": "ArcticIERP2017_2019",
            "source_title": "Arctic Integrated Ecosystem Research Program zooplankton",
            "years_in_or_near_target": "2017;2019",
            "sampling_months": "Aug-Sep",
            "geographic_role": "northern Bering and Chukchi seas",
            "euphausiid_variables": "laboratory and rapid abundance per m3; lipid content",
            "raw_station_data_downloaded": True,
            "target_window_fit": "no for abundance",
            "spatial_surface_fit": "late-season observations only",
            "phase8n_role": "quantitative abundance and prey-quality evidence",
            "doi_or_url": "https://doi.org/10.1093/plankt/fbaf034",
        },
        {
            "source_id": "ASGARD2017_2018",
            "source_title": "ASGARD spring zooplankton abundance and lipid observations",
            "years_in_or_near_target": "2017;2018",
            "sampling_months": "Jun",
            "geographic_role": "northern Bering and southern Chukchi seas",
            "euphausiid_variables": "station abundance per m3; lipid per wet mass; specimen wet mass",
            "raw_station_data_downloaded": True,
            "target_window_fit": "yes for 2017-2018 only",
            "spatial_surface_fit": "two-year quantitative station evidence; not an annual series",
            "phase8n_role": "target-season abundance diagnostic and prey-quality context",
            "doi_or_url": "https://doi.org/10.24431/rw1kai8",
        },
        {
            "source_id": "ArcticIERPSynthesis2002_2022",
            "source_title": "Arctic IERP pelagic-benthic synthesis zooplankton inputs and regional indices",
            "years_in_or_near_target": "2011-2019",
            "sampling_months": "mostly Aug-Oct in Chukchi Sea; sparse Apr-Jul in northern Bering Sea",
            "geographic_role": "northern Bering and Chukchi seas",
            "euphausiid_variables": "harmonized abundance and carbon biomass; bias-corrected regional index",
            "raw_station_data_downloaded": True,
            "target_window_fit": "sparse context only outside ASGARD 2017-2018",
            "spatial_surface_fit": "regional annual index is not a spatial surface",
            "phase8n_role": "late-season annual context and source harmonization audit",
            "doi_or_url": "https://doi.org/10.24431/rw1kai8",
        },
        {
            "source_id": "ARA10_2019",
            "source_title": "ARA10A/B Bering and Chukchi Sea oceanographic observations",
            "years_in_or_near_target": "2019",
            "sampling_months": "Jul-Aug",
            "geographic_role": "Bering and Chukchi seas",
            "euphausiid_variables": "mesozooplankton broad-group abundance only",
            "raw_station_data_downloaded": True,
            "target_window_fit": "partial dates; taxon resolution fails",
            "spatial_surface_fit": "no",
            "phase8n_role": "screened out because euphausiids are not separately quantified",
            "doi_or_url": "https://doi.org/10.25921/3pp6-v545",
        },
        {
            "source_id": "OBIS2011_2019",
            "source_title": "OBIS Euphausiacea occurrence query",
            "years_in_or_near_target": "2011-2019",
            "sampling_months": "multiple",
            "geographic_role": "Pacific Arctic bounding box",
            "euphausiid_variables": "occurrence records",
            "raw_station_data_downloaded": True,
            "target_window_fit": "mixed",
            "spatial_surface_fit": "no quantitative layer in occurrence export",
            "phase8n_role": "source discovery and duplicate audit",
            "doi_or_url": "https://obis.org/data/access/",
        },
        {
            "source_id": "COPEPOD",
            "source_title": "NOAA COPEPOD Bering and Chukchi zooplankton database",
            "years_in_or_near_target": "multiple",
            "sampling_months": "multiple",
            "geographic_role": "Bering and Chukchi seas",
            "euphausiid_variables": "aggregated plankton holdings",
            "raw_station_data_downloaded": False,
            "target_window_fit": "not resolved",
            "spatial_surface_fit": "not independently added",
            "phase8n_role": "aggregator screened for source discovery",
            "doi_or_url": "https://apps-st.fisheries.noaa.gov/copepod/database/index.html",
        },
        {
            "source_id": "CHAOZ2011",
            "source_title": "Chukchi Sea Cruise MB11-01 zooplankton",
            "years_in_or_near_target": "2011",
            "sampling_months": "Aug-Sep",
            "geographic_role": "Bering Strait to Barrow Canyon",
            "euphausiid_variables": "zooplankton abundance; distribution restricted",
            "raw_station_data_downloaded": False,
            "target_window_fit": "no",
            "spatial_surface_fit": "not publicly downloadable from catalog",
            "phase8n_role": "catalog evidence; likely overlaps published late-season holdings",
            "doi_or_url": "https://www.fisheries.noaa.gov/inport/item/17091",
        },
        {
            "source_id": "RussianFarEast1984_2013",
            "source_title": "Russian Far East zooplankton database",
            "years_in_or_near_target": "2011-2013 subset possible",
            "sampling_months": "seasonal aggregates",
            "geographic_role": "Russian Far East seas",
            "euphausiid_variables": "region-season-period aggregates",
            "raw_station_data_downloaded": False,
            "target_window_fit": "no annual target-window resolution",
            "spatial_surface_fit": "no",
            "phase8n_role": "screened out because data are aggregated and access-limited",
            "doi_or_url": "https://doi.org/10.1594/PANGAEA.937751",
        },
        {
            "source_id": "NOAA_EBS_Index",
            "source_title": "Eastern Bering Sea euphausiid biomass index",
            "years_in_or_near_target": "2011-2019",
            "sampling_months": "survey-season annual index",
            "geographic_role": "eastern Bering Sea",
            "euphausiid_variables": "annual acoustic-trawl biomass index",
            "raw_station_data_downloaded": True,
            "target_window_fit": "regional context only",
            "spatial_surface_fit": "no",
            "phase8n_role": "upstream annual context; outside study-grid surface role",
            "doi_or_url": "https://apps-afsc.fisheries.noaa.gov/refm/reem/ecoweb/index.php?ID=1",
        },
        {
            "source_id": "Spear_CHAOZ_2011_2015",
            "source_title": "Advection and in situ processes as drivers of large-zooplankton abundance in the Chukchi Sea",
            "years_in_or_near_target": "2011-2015",
            "sampling_months": "primarily late summer",
            "geographic_role": "Chukchi Sea",
            "euphausiid_variables": "Thysanoessa abundance and life stages",
            "raw_station_data_downloaded": False,
            "target_window_fit": "no",
            "spatial_surface_fit": "publication-level synthesis; underlying holdings overlap screened programs",
            "phase8n_role": "mechanistic advection evidence and source cross-check",
            "doi_or_url": "https://repository.library.noaa.gov/view/noaa/29055/noaa_29055_DS1.pdf",
        },
        {
            "source_id": "ARAON_2014_2016",
            "source_title": "Western Arctic zooplankton observations from ARAON cruises",
            "years_in_or_near_target": "2014-2016",
            "sampling_months": "Aug",
            "geographic_role": "northern Chukchi and western Arctic",
            "euphausiid_variables": "net abundance in publication analyses",
            "raw_station_data_downloaded": False,
            "target_window_fit": "no",
            "spatial_surface_fit": "no reusable station package identified",
            "phase8n_role": "published spatial and interannual context",
            "doi_or_url": "https://doi.org/10.3389/fmars.2020.00612",
        },
        {
            "source_id": "Kuznetsova_Chukchi_2019",
            "source_title": "Species composition, structure and biomass of zooplankton in the Chukchi Sea in August-September 2019",
            "years_in_or_near_target": "2019",
            "sampling_months": "Aug-Sep",
            "geographic_role": "western and eastern Chukchi Sea",
            "euphausiid_variables": "regional zooplankton biomass and composition",
            "raw_station_data_downloaded": False,
            "target_window_fit": "no",
            "spatial_surface_fit": "regional publication summaries only",
            "phase8n_role": "late-season Russian and US sector context",
            "doi_or_url": "https://izvestiya.tinro-center.ru/jour/article/view/707",
        },
        {
            "source_id": "CPR_OBIS",
            "source_title": "Continuous Plankton Recorder Survey occurrence records",
            "years_in_or_near_target": "selected 2011-2019 records",
            "sampling_months": "route-dependent",
            "geographic_role": "sparse Pacific Arctic transects",
            "euphausiid_variables": "occurrence in OBIS export",
            "raw_station_data_downloaded": True,
            "target_window_fit": "mixed",
            "spatial_surface_fit": "no quantitative station layer in queried export",
            "phase8n_role": "occurrence-only source within OBIS duplicate audit",
            "doi_or_url": "https://api.obis.org/v3/dataset/5c4667b8-8d93-4768-bc7e-3c31cde1cf18",
        },
        {
            "source_id": "AMBON2020",
            "source_title": "Arctic Marine Biodiversity Observing Network zooplankton 2020",
            "years_in_or_near_target": "2020",
            "sampling_months": "Oct",
            "geographic_role": "northern Bering and Chukchi seas",
            "euphausiid_variables": "abundance and biomass",
            "raw_station_data_downloaded": True,
            "target_window_fit": "outside model years and season",
            "spatial_surface_fit": "no for current analysis",
            "phase8n_role": "future post-2019 context only",
            "doi_or_url": "https://doi.org/10.15468/r3fatd",
        },
        {
            "source_id": "SHELFZ2013",
            "source_title": "2013 Shelf Habitat and Ecology of Fish and Zooplankton survey",
            "years_in_or_near_target": "2013",
            "sampling_months": "summer survey",
            "geographic_role": "northeastern Chukchi Sea",
            "euphausiid_variables": "catalog mentions zooplankton; downloadable tables are survey/haul oriented",
            "raw_station_data_downloaded": False,
            "target_window_fit": "not demonstrated",
            "spatial_surface_fit": "no direct euphausiid layer identified",
            "phase8n_role": "catalog screened; not promoted to model input",
            "doi_or_url": "https://www.fisheries.noaa.gov/inport/item/24160",
        },
    ]
    return pd.DataFrame(rows)


def empty_observations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_id",
            "source_family",
            "year",
            "month",
            "timing_class",
            "method_group",
            "event_id",
            "lat",
            "lon",
            "euphausiid_abundance_m3",
            "euphausiid_biomass_mg_m3",
            "lipid_ug_mg_wet",
            "event_taxon_records",
            "data_role",
        ]
    )


def parse_cses() -> pd.DataFrame:
    path = RAW / "CSESP_ZOOPLANKTON_2008_14_Ed5.csv"
    data = pd.read_csv(path, low_memory=False)
    data["date"] = pd.to_datetime(data["DATE_TIME_LOC"], errors="coerce")
    data = data[data["date"].dt.year.isin(TARGET_YEARS)].copy()
    euphausiid = data["SCIENTIFIC_NAME"].astype(str).str.contains(
        "Euphausi|Thysanoessa", case=False, na=False
    )
    abundance = pd.to_numeric(data["ABUNDANCE_M3"], errors="coerce").fillna(0.0)
    biomass = pd.to_numeric(data["BIOMASS_MG_M3"], errors="coerce").fillna(0.0)
    data["euph_abundance"] = np.where(euphausiid, abundance, 0.0)
    data["euph_biomass"] = np.where(euphausiid, biomass, 0.0)
    data["euph_record"] = euphausiid.astype(int)
    keys = ["CRUISE", "STATION", "DATE_TIME_LOC", "GEAR_TYPE", "LATITUDE", "LONGITUDE"]
    out = (
        data.groupby(keys, dropna=False, as_index=False)
        .agg(
            euphausiid_abundance_m3=("euph_abundance", "sum"),
            euphausiid_biomass_mg_m3=("euph_biomass", "sum"),
            event_taxon_records=("euph_record", "sum"),
            date=("date", "first"),
        )
    )
    out["source_id"] = "CSESP2008_2014"
    out["source_family"] = "CSESP"
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["timing_class"] = "late_season_abundance"
    out["method_group"] = out["GEAR_TYPE"].astype(str).str.replace(" ", "_", regex=False)
    out["event_id"] = (
        out["CRUISE"].astype(str)
        + ":"
        + out["STATION"].astype(str)
        + ":"
        + out["DATE_TIME_LOC"].astype(str)
        + ":"
        + out["GEAR_TYPE"].astype(str)
        + ":"
        + out["LATITUDE"].round(6).astype(str)
        + ":"
        + out["LONGITUDE"].round(6).astype(str)
    )
    out["lat"] = pd.to_numeric(out["LATITUDE"], errors="coerce")
    out["lon"] = pd.to_numeric(out["LONGITUDE"], errors="coerce")
    out["lipid_ug_mg_wet"] = np.nan
    out["data_role"] = "quantitative_late_season_abundance_biomass"
    return out[empty_observations().columns]


def parse_ambon(year: int) -> pd.DataFrame:
    folder = RAW / f"AMBON_zooplankton_{year}_dwca"
    events = pd.read_csv(folder / "event.txt", sep="\t", low_memory=False)
    occurrence = pd.read_csv(folder / "occurrence.txt", sep="\t", low_memory=False)
    measurements = pd.read_csv(
        folder / "extendedmeasurementorfact.txt", sep="\t", low_memory=False
    )
    measurements["metric"] = np.where(
        measurements["measurementType"].astype(str).str.contains("abundance", case=False),
        "abundance",
        np.where(
            measurements["measurementType"].astype(str).str.contains("biomass", case=False),
            "biomass",
            "other",
        ),
    )
    measurements["measurementValue"] = pd.to_numeric(
        measurements["measurementValue"], errors="coerce"
    )
    wide = (
        measurements[measurements["metric"].isin(["abundance", "biomass"])]
        .pivot_table(
            index="occurrenceID",
            columns="metric",
            values="measurementValue",
            aggfunc="sum",
        )
        .reset_index()
    )
    linkage_fraction = occurrence["occurrenceID"].isin(wide["occurrenceID"]).mean()
    occurrence = occurrence.merge(wide, on="occurrenceID", how="left")
    event_cols = ["eventID", "eventDate", "decimalLatitude", "decimalLongitude"]
    occurrence = occurrence.merge(
        events[event_cols].rename(
            columns={
                "eventDate": "eventDate_event",
                "decimalLatitude": "decimalLatitude_event",
                "decimalLongitude": "decimalLongitude_event",
            }
        ),
        on="eventID",
        how="left",
    )
    occurrence["event_date"] = pd.to_datetime(
        occurrence.get("eventDate", occurrence["eventDate_event"]), errors="coerce", utc=True
    )
    if occurrence["event_date"].isna().all():
        occurrence["event_date"] = pd.to_datetime(
            occurrence["eventDate_event"], errors="coerce", utc=True
        )
    occurrence["lat_resolved"] = pd.to_numeric(
        occurrence.get("decimalLatitude", occurrence["decimalLatitude_event"]), errors="coerce"
    )
    occurrence["lat_resolved"] = occurrence["lat_resolved"].fillna(
        pd.to_numeric(occurrence["decimalLatitude_event"], errors="coerce")
    )
    occurrence["lon_resolved"] = pd.to_numeric(
        occurrence.get("decimalLongitude", occurrence["decimalLongitude_event"]), errors="coerce"
    )
    occurrence["lon_resolved"] = occurrence["lon_resolved"].fillna(
        pd.to_numeric(occurrence["decimalLongitude_event"], errors="coerce")
    )
    euphausiid = occurrence["scientificName"].astype(str).str.contains(
        "Euphausi|Thysanoessa", case=False, na=False
    )
    occurrence["euph_abundance"] = np.where(
        euphausiid, pd.to_numeric(occurrence.get("abundance"), errors="coerce").fillna(0.0), 0.0
    )
    occurrence["euph_biomass"] = np.where(
        euphausiid, pd.to_numeric(occurrence.get("biomass"), errors="coerce").fillna(0.0), 0.0
    )
    occurrence["euph_record"] = euphausiid.astype(int)
    keys = ["eventID", "samplingProtocol", "lat_resolved", "lon_resolved"]
    out = (
        occurrence.groupby(keys, dropna=False, as_index=False)
        .agg(
            euphausiid_abundance_m3=("euph_abundance", "sum"),
            euphausiid_biomass_mg_m3=("euph_biomass", "sum"),
            event_taxon_records=("euph_record", "sum"),
            event_date=("event_date", "first"),
        )
    )
    if linkage_fraction < 0.9:
        out["euphausiid_abundance_m3"] = np.nan
        out["euphausiid_biomass_mg_m3"] = np.nan
    out["source_id"] = f"AMBON{year}"
    out["source_family"] = "AMBON"
    out["year"] = year
    out["month"] = out["event_date"].dt.month.astype("Int64")
    out["timing_class"] = "late_season_abundance"
    out["method_group"] = out["samplingProtocol"].astype(str).str.replace(" ", "_", regex=False)
    out["event_id"] = out["eventID"].astype(str) + ":" + out["samplingProtocol"].astype(str)
    out["lat"] = out["lat_resolved"]
    out["lon"] = out["lon_resolved"]
    out["lipid_ug_mg_wet"] = np.nan
    out["data_role"] = (
        "quantitative_late_season_abundance_biomass"
        if linkage_fraction >= 0.9
        else "occurrence_footprint_only_unreliable_measurement_linkage"
    )
    return out[empty_observations().columns]


def parse_ierp_lab(year: int) -> pd.DataFrame:
    tag = str(year)[-2:]
    data = pd.read_csv(RAW / f"OS{tag}01_EcoFOCI_ZooplanktonNet.csv", low_memory=False)
    euphausiid = data["TAXON_NAME"].astype(str).str.contains(
        "Euphausi|Thysanoessa", case=False, na=False
    )
    abundance = pd.to_numeric(data["EST_NUM_PERM3"], errors="coerce").fillna(0.0)
    data["euph_abundance"] = np.where(euphausiid, abundance, 0.0)
    data["euph_record"] = euphausiid.astype(int)
    keys = [
        "CRUISE",
        "STATION_NAME",
        "HAUL_NAME",
        "GEAR_NAME",
        "NET",
        "MONTH",
        "DAY",
        "YEAR",
        "LAT",
        "LON",
        "MESH",
    ]
    out = (
        data.groupby(keys, dropna=False, as_index=False)
        .agg(
            euphausiid_abundance_m3=("euph_abundance", "sum"),
            event_taxon_records=("euph_record", "sum"),
        )
    )
    out["source_id"] = f"ArcticIERP_lab_{year}"
    out["source_family"] = "Arctic IERP"
    out["year"] = year
    out["month"] = pd.to_numeric(out["MONTH"], errors="coerce").astype("Int64")
    out["timing_class"] = "late_season_abundance"
    out["method_group"] = (
        "lab_" + out["GEAR_NAME"].astype(str) + "_" + out["MESH"].astype(str) + "um"
    )
    out["event_id"] = (
        out["CRUISE"].astype(str)
        + ":"
        + out["STATION_NAME"].astype(str)
        + ":"
        + out["HAUL_NAME"].astype(str)
        + ":"
        + out["NET"].astype(str)
    )
    out["lat"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["lon"] = pd.to_numeric(out["LON"], errors="coerce")
    out["euphausiid_biomass_mg_m3"] = np.nan
    out["lipid_ug_mg_wet"] = np.nan
    out["data_role"] = "quantitative_late_season_abundance"
    return out[empty_observations().columns]


def parse_ierp_rza(year: int) -> pd.DataFrame:
    tag = str(year)[-2:]
    data = pd.read_csv(RAW / f"OS{tag}01_RZA.csv", low_memory=False)
    euphausiid = data["RZA_TAXA"].astype(str).str.contains("Euphausi", case=False, na=False)
    abundance = pd.to_numeric(data["EST_NUM_PERM3"], errors="coerce").fillna(0.0)
    data["euph_abundance"] = np.where(euphausiid, abundance, 0.0)
    data["euph_record"] = euphausiid.astype(int)
    keys = ["CRUISE", "STATION_NAME", "DATE", "GEAR_NAME", "NET", "LAT", "LON"]
    out = (
        data.groupby(keys, dropna=False, as_index=False)
        .agg(
            euphausiid_abundance_m3=("euph_abundance", "sum"),
            event_taxon_records=("euph_record", "sum"),
        )
    )
    dates = pd.to_datetime(out["DATE"], errors="coerce")
    out["source_id"] = f"ArcticIERP_RZA_{year}"
    out["source_family"] = "Arctic IERP"
    out["year"] = year
    out["month"] = dates.dt.month.astype("Int64")
    out["timing_class"] = "late_season_abundance"
    out["method_group"] = "RZA_" + out["GEAR_NAME"].astype(str)
    out["event_id"] = (
        out["CRUISE"].astype(str)
        + ":"
        + out["STATION_NAME"].astype(str)
        + ":"
        + out["DATE"].astype(str)
        + ":"
        + out["NET"].astype(str)
    )
    out["lat"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["lon"] = pd.to_numeric(out["LON"], errors="coerce")
    out["euphausiid_biomass_mg_m3"] = np.nan
    out["lipid_ug_mg_wet"] = np.nan
    out["data_role"] = "rapid_late_season_abundance_method_check"
    return out[empty_observations().columns]


def normalize_west_longitude(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if value < 0:
        return value
    if value <= 180:
        return -value
    return 360.0 - value


def parse_point_barrow() -> pd.DataFrame:
    root = RAW / "point_barrow_2011_2015"
    frames = []
    for year in range(2011, 2016):
        candidates = [
            path
            for path in (root / str(year)).glob("*ZooplanktonAbundfromTuckerTrawl.txt")
            if "Variable" not in path.name
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one Tucker-trawl file for {year}, found {candidates}")
        data = pd.read_csv(candidates[0], sep="\t", low_memory=False)
        data = data[pd.to_numeric(data["LatDDeg"], errors="coerce").notna()].copy()
        total_columns = [column for column in ["Trashall", "Tinerall", "Tlongall"] if column in data]
        for column in total_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
        out = pd.DataFrame(
            {
                "source_id": f"PointBarrow{year}",
                "source_family": "Point Barrow",
                "year": year,
                "month": pd.to_numeric(data["Mo"], errors="coerce").astype("Int64"),
                "timing_class": "late_season_abundance",
                "method_group": "Tucker_sled_335_500um",
                "event_id": (
                    "PB:"
                    + str(year)
                    + ":"
                    + data["Stn"].astype(str)
                    + ":"
                    + data["Hr"].astype(str)
                    + ":"
                    + data["Mn"].astype(str)
                    + ":"
                    + data["MinTowDep"].astype(str)
                    + ":"
                    + data["MaxTowDep"].astype(str)
                ),
                "lat": pd.to_numeric(data["LatDDeg"], errors="coerce"),
                "lon": pd.to_numeric(data["LonDDeg"], errors="coerce").map(
                    normalize_west_longitude
                ),
                "euphausiid_abundance_m3": data[total_columns].sum(axis=1),
                "euphausiid_biomass_mg_m3": np.nan,
                "lipid_ug_mg_wet": np.nan,
                "event_taxon_records": (data[total_columns] > 0).sum(axis=1),
                "data_role": "quantitative_local_late_season_abundance",
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def parse_asgard_quality() -> pd.DataFrame:
    frames = []
    for year in [2017, 2018]:
        path = RAW / f"Zooplankton_Lipid_ASGARD_{year}_Copeman.csv"
        data = pd.read_csv(path, low_memory=False)
        data = data[data["Taxa"].astype(str).str.contains("Euphausi", case=False, na=False)].copy()
        data["lipid"] = pd.to_numeric(
            data["Total lipid per wet mass (micrograms/mg)"], errors="coerce"
        )
        out = (
            data.groupby(["Latitude", "Longitude"], dropna=False, as_index=False)
            .agg(
                lipid_ug_mg_wet=("lipid", "mean"),
                event_taxon_records=("Taxa", "size"),
            )
        )
        out["source_id"] = f"ASGARD_quality_{year}"
        out["source_family"] = "ASGARD quality"
        out["year"] = year
        out["month"] = 6
        out["timing_class"] = "target_window_quality_only"
        out["method_group"] = "targeted_euphausiid_lipid_sample"
        out["event_id"] = (
            "ASGARD:"
            + str(year)
            + ":"
            + out["Latitude"].round(5).astype(str)
            + ":"
            + out["Longitude"].round(5).astype(str)
        )
        out["lat"] = pd.to_numeric(out["Latitude"], errors="coerce")
        out["lon"] = pd.to_numeric(out["Longitude"], errors="coerce")
        out["euphausiid_abundance_m3"] = np.nan
        out["euphausiid_biomass_mg_m3"] = np.nan
        out["data_role"] = "target_window_prey_quality_without_abundance"
        frames.append(out[empty_observations().columns])
    return pd.concat(frames, ignore_index=True)


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text.where(~compact), errors="coerce")
    compact_parsed = pd.to_datetime(text.where(compact), format="%Y%m%d", errors="coerce")
    return parsed.fillna(compact_parsed)


def parse_asgard_abundance() -> pd.DataFrame:
    path = RAW / "arctic_ierp_synthesis_2026/Gonzalez_INPUT_env_phyto_zoo_taxa.csv"
    data = pd.read_csv(path, low_memory=False)
    data["date"] = parse_mixed_dates(data["Date"])
    data = data[
        data["date"].dt.year.isin([2017, 2018])
        & (data["date"].dt.month == 6)
    ].copy()
    broad = pd.to_numeric(data["Euphausiacea"], errors="coerce").fillna(0.0)
    thysanoessa = pd.to_numeric(data["Thysanoessa.spp."], errors="coerce").fillna(0.0)
    data["euphausiid_abundance_m3"] = broad + thysanoessa
    data["event_taxon_records"] = (broad > 0).astype(int) + (thysanoessa > 0).astype(int)
    out = pd.DataFrame(
        {
            "source_id": "ASGARD_abundance_" + data["date"].dt.year.astype(str),
            "source_family": "ASGARD abundance",
            "year": data["date"].dt.year,
            "month": data["date"].dt.month,
            "timing_class": "target_window_abundance",
            "method_group": "ASGARD_station_composite_150_505um",
            "event_id": (
                data["Cruise"].astype(str)
                + ":"
                + data["Cast"].astype(str)
                + ":"
                + data["Station"].astype(str)
                + ":"
                + data["date"].dt.strftime("%Y%m%d")
            ),
            "lat": pd.to_numeric(data["Latitude"], errors="coerce"),
            "lon": pd.to_numeric(data["Longitude"], errors="coerce"),
            "euphausiid_abundance_m3": data["euphausiid_abundance_m3"],
            "euphausiid_biomass_mg_m3": np.nan,
            "lipid_ug_mg_wet": np.nan,
            "event_taxon_records": data["event_taxon_records"],
            "data_role": "quantitative_target_window_abundance",
        }
    )
    return out[empty_observations().columns]


def parse_synthesis_target_context() -> pd.DataFrame:
    path = RAW / "arctic_ierp_synthesis_2026/Spear_Kimmel_INPUT_zoop_process_biomass.csv"
    data = pd.read_csv(path, low_memory=False)
    data["date"] = pd.to_datetime(data["GMT_DATE_TIME_TXT"], format="mixed", errors="coerce")
    data = data[
        data["date"].dt.year.isin(TARGET_YEARS)
        & data["date"].dt.month.isin(TARGET_MONTHS)
    ].copy()
    euphausiid = data["TAXA_COARSE"].astype(str).str.contains(
        "Euphausi|Thysanoessa", case=False, na=False
    ) | data["TAXON_NAME"].astype(str).str.contains(
        "Euphausi|Thysanoessa", case=False, na=False
    )
    data = data[euphausiid].copy()
    data["euph_abundance"] = pd.to_numeric(data["EST_NUM_PERM3"], errors="coerce").fillna(0.0)
    data["euph_biomass"] = pd.to_numeric(
        data["BIOMASS_C_MG_M3_MEAN"], errors="coerce"
    ).fillna(0.0)
    keys = [
        "DATA_SOURCE",
        "CRUISE",
        "STATION_NAME",
        "GMT_DATE_TIME_TXT",
        "GEAR_NAME",
        "MESH",
        "LAT",
        "LON",
    ]
    out = (
        data.groupby(keys, dropna=False, as_index=False)
        .agg(
            euphausiid_abundance_m3=("euph_abundance", "sum"),
            euphausiid_biomass_mg_m3=("euph_biomass", "sum"),
            event_taxon_records=("TAXON_NAME", "size"),
            date=("date", "first"),
        )
    )
    out["source_id"] = (
        "IERP_synthesis_" + out["DATA_SOURCE"].astype(str) + "_" + out["date"].dt.year.astype(str)
    )
    out["source_family"] = "IERP synthesis"
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["timing_class"] = "target_window_abundance_context"
    out["method_group"] = (
        out["DATA_SOURCE"].astype(str)
        + ":"
        + out["GEAR_NAME"].astype(str)
        + ":"
        + out["MESH"].astype(str)
        + "um"
    )
    out["event_id"] = (
        out["DATA_SOURCE"].astype(str)
        + ":"
        + out["CRUISE"].astype(str)
        + ":"
        + out["STATION_NAME"].astype(str)
        + ":"
        + out["GMT_DATE_TIME_TXT"].astype(str)
        + ":"
        + out["GEAR_NAME"].astype(str)
    )
    out["lat"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["lon"] = pd.to_numeric(out["LON"], errors="coerce")
    out["lipid_ug_mg_wet"] = np.nan
    out["data_role"] = "sparse_target_window_synthesis_context"
    return out[empty_observations().columns]


def load_observations() -> pd.DataFrame:
    frames = [
        parse_cses(),
        parse_ambon(2015),
        parse_ambon(2017),
        parse_ierp_lab(2017),
        parse_ierp_lab(2019),
        parse_ierp_rza(2017),
        parse_ierp_rza(2019),
        parse_point_barrow(),
        parse_asgard_abundance(),
        parse_asgard_quality(),
        parse_synthesis_target_context(),
    ]
    data = pd.concat(frames, ignore_index=True)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["month"] = pd.to_numeric(data["month"], errors="coerce").astype("Int64")
    for column in [
        "lat",
        "lon",
        "euphausiid_abundance_m3",
        "euphausiid_biomass_mg_m3",
        "lipid_ug_mg_wet",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def obis_duplication_audit() -> pd.DataFrame:
    path = RAW / "OBIS_euphausiacea_2011_2019_pacific_arctic_bbox.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("results", payload)
    data = pd.DataFrame(records)
    if data.empty:
        return pd.DataFrame(columns=["dataset_id", "dataset_name", "records", "years"])
    dataset_col = "dataset_id" if "dataset_id" in data else "datasetID"
    name_col = "dataset" if "dataset" in data else "datasetName"
    year_col = "year" if "year" in data else None
    rows = []
    for dataset_id, group in data.groupby(dataset_col, dropna=False):
        years = ""
        if year_col:
            years = ";".join(
                str(int(value))
                for value in sorted(pd.to_numeric(group[year_col], errors="coerce").dropna().unique())
            )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": group[name_col].dropna().iloc[0] if name_col in group and group[name_col].notna().any() else "",
                "records": len(group),
                "years": years,
                "quantitative_fields_present_in_export": False,
                "phase8n_use": "source duplication audit only",
            }
        )
    return pd.DataFrame(rows).sort_values("records", ascending=False)


def match_to_grid(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(GRID)
    grid["grid_id"] = grid["grid_id"].astype(str)
    tree = cKDTree(grid[["x", "y"]].to_numpy(float))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    valid = observations["lat"].notna() & observations["lon"].notna()
    observations = observations.copy()
    observations["obs_x"] = np.nan
    observations["obs_y"] = np.nan
    observations["nearest_grid_distance_km"] = np.nan
    observations["grid_id"] = pd.NA
    observations["grid_match_within_15km"] = False
    if valid.any():
        x, y = transformer.transform(
            observations.loc[valid, "lon"].to_numpy(float),
            observations.loc[valid, "lat"].to_numpy(float),
        )
        distance, index = tree.query(np.column_stack([x, y]), k=1)
        observations.loc[valid, "obs_x"] = x
        observations.loc[valid, "obs_y"] = y
        observations.loc[valid, "nearest_grid_distance_km"] = distance / 1000.0
        observations.loc[valid, "grid_id"] = grid.iloc[index]["grid_id"].to_numpy()
        observations.loc[valid, "grid_match_within_15km"] = distance <= GRID_MATCH_LIMIT_KM * 1000
    observations.loc[~observations["grid_match_within_15km"], "grid_id"] = pd.NA

    matched = observations[observations["grid_match_within_15km"]].copy()
    matched["detected"] = matched["euphausiid_abundance_m3"].fillna(0) > 0
    group_cols = [
        "source_id",
        "source_family",
        "year",
        "month",
        "timing_class",
        "method_group",
        "grid_id",
        "data_role",
    ]
    gridded = (
        matched.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_sampling_events=("event_id", "nunique"),
            mean_euphausiid_abundance_m3=("euphausiid_abundance_m3", "mean"),
            mean_euphausiid_biomass_mg_m3=("euphausiid_biomass_mg_m3", "mean"),
            mean_lipid_ug_mg_wet=("lipid_ug_mg_wet", "mean"),
            euphausiid_detected=("detected", "max"),
            grid_match_distance_km=("nearest_grid_distance_km", "mean"),
        )
    )
    gridded = gridded.merge(grid[["grid_id", "x", "y", "lon", "lat"]], on="grid_id", how="left")
    abundance_mask = gridded["mean_euphausiid_abundance_m3"].notna()
    gridded["abundance_percentile_within_source_year_method"] = np.nan
    gridded.loc[abundance_mask, "abundance_percentile_within_source_year_method"] = (
        gridded.loc[abundance_mask]
        .groupby(["source_id", "year", "method_group"])["mean_euphausiid_abundance_m3"]
        .rank(method="average", pct=True)
        * 100.0
    )
    return observations, gridded


def coverage_summary(observations: pd.DataFrame) -> pd.DataFrame:
    data = observations.copy()
    data["detected"] = data["euphausiid_abundance_m3"].fillna(0) > 0
    data["euphausiid_record_present"] = data["event_taxon_records"].fillna(0) > 0
    return (
        data.groupby(
            ["source_id", "source_family", "year", "timing_class", "method_group", "data_role"],
            dropna=False,
            as_index=False,
        )
        .agg(
            month_min=("month", "min"),
            month_max=("month", "max"),
            sampling_events=("event_id", "size"),
            events_with_euphausiid_records=("euphausiid_record_present", "sum"),
            events_with_positive_quantitative_abundance=("detected", "sum"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            abundance_available=("euphausiid_abundance_m3", lambda series: bool(series.notna().any())),
            biomass_available=("euphausiid_biomass_mg_m3", lambda series: bool(series.notna().any())),
            lipid_available=("lipid_ug_mg_wet", lambda series: bool(series.notna().any())),
            matched_events=("grid_match_within_15km", "sum")
            if "grid_match_within_15km" in data
            else ("event_id", "size"),
        )
        .sort_values(["year", "source_id", "method_group"])
    )


def synthesis_regional_index() -> pd.DataFrame:
    import pyreadr

    index_path = (
        RAW
        / "arctic_ierp_synthesis_2026/Thorson_OUTPUT_All_Eco_Component_2002_2022_index.csv"
    )
    input_path = (
        RAW
        / "arctic_ierp_synthesis_2026/Thorson_INPUT_All_Eco_Component_2002_2022.rds"
    )
    index = pd.read_csv(index_path)
    index = index[
        (index["taxa"] == "Euphausiacea") & index["year"].isin(TARGET_YEARS)
    ].copy()
    r_objects = pyreadr.read_r(input_path)
    samples = next(iter(r_objects.values()))
    samples = samples[
        (samples["category"].astype(str) == "Euphausiacea")
        & pd.to_numeric(samples["year"], errors="coerce").isin(TARGET_YEARS)
    ].copy()
    samples["year"] = pd.to_numeric(samples["year"], errors="coerce").astype(int)
    samples["month"] = pd.to_numeric(samples["month"], errors="coerce").astype("Int64")
    samples["positive"] = pd.to_numeric(samples["SumWeight"], errors="coerce").fillna(0) > 0
    season = (
        samples.groupby(["Region", "year"], as_index=False)
        .agg(
            source_sampling_events=("SampleID", "size"),
            source_positive_events=("positive", "sum"),
            month_min=("month", "min"),
            month_max=("month", "max"),
            target_window_samples=("month", lambda x: int(x.isin(TARGET_MONTHS).sum())),
            late_season_samples=("month", lambda x: int(x.isin({8, 9, 10}).sum())),
            input_units=("SumWeightUNITS", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        )
    )
    month_labels = (
        samples.groupby(["Region", "year"])["month"]
        .apply(lambda x: ";".join(str(int(value)) for value in sorted(x.dropna().unique())))
        .rename("sampling_months")
        .reset_index()
    )
    season = season.merge(month_labels, on=["Region", "year"], how="left")
    out = index.merge(
        season,
        left_on=["region", "year"],
        right_on=["Region", "year"],
        how="left",
    )
    out = out.rename(
        columns={
            "est": "plugin_index",
            "se": "asymptotic_se",
            "bc": "bias_corrected_regional_index",
            "samples": "model_sample_count",
            "encounter_rate": "model_encounter_rate",
        }
    )
    out["seasonal_interpretation"] = np.where(
        (out["target_window_samples"].fillna(0) > 0)
        & (out["late_season_samples"].fillna(0) > out["target_window_samples"].fillna(0)),
        "mixed but predominantly late season",
        np.where(
            out["target_window_samples"].fillna(0) > 0,
            "target-window samples present",
            "late-season context only",
        ),
    )
    columns = [
        "region",
        "year",
        "yearcruise",
        "plugin_index",
        "asymptotic_se",
        "bias_corrected_regional_index",
        "model_sample_count",
        "model_encounter_rate",
        "sampling_months",
        "target_window_samples",
        "late_season_samples",
        "source_sampling_events",
        "source_positive_events",
        "input_units",
        "seasonal_interpretation",
    ]
    return out[columns].sort_values(["region", "year"])


def asgard_abundance_quality_pairing(
    gridded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abundance = gridded[
        (gridded["source_family"] == "ASGARD abundance")
        & gridded["mean_euphausiid_abundance_m3"].notna()
    ][["year", "grid_id", "mean_euphausiid_abundance_m3", "x", "y", "lon", "lat"]]
    quality = gridded[
        (gridded["source_family"] == "ASGARD quality")
        & gridded["mean_lipid_ug_mg_wet"].notna()
    ][["year", "grid_id", "mean_lipid_ug_mg_wet"]]
    paired = abundance.merge(quality, on=["year", "grid_id"], how="inner")
    rows = []
    for year, group in paired.groupby("year"):
        rho, p_value = safe_spearman(
            group["mean_euphausiid_abundance_m3"], group["mean_lipid_ug_mg_wet"]
        )
        rows.append(
            {
                "year": int(year),
                "paired_grid_cells": len(group),
                "spearman_abundance_vs_lipid": rho,
                "spearman_p_value": p_value,
                "interpretation": "paired target-season quantity-quality context; not an energy surface",
            }
        )
    return paired, pd.DataFrame(rows)


def asgard_dbo_summary(observations: pd.DataFrame) -> pd.DataFrame:
    data = observations[observations["source_family"] == "ASGARD abundance"].copy()
    points = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["lon"], data["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3338")
    anchors = gpd.read_file(DBO).to_crs("EPSG:3338")
    joined = gpd.sjoin(
        points,
        anchors[["name", "geometry"]],
        predicate="within",
        how="left",
    )
    joined["anchor"] = joined["name"].fillna("Outside DBO anchor polygons")
    return (
        joined.groupby(["year", "anchor"], as_index=False)
        .agg(
            sampling_stations=("event_id", "nunique"),
            mean_euphausiid_abundance_m3=("euphausiid_abundance_m3", "mean"),
            median_euphausiid_abundance_m3=("euphausiid_abundance_m3", "median"),
            maximum_euphausiid_abundance_m3=("euphausiid_abundance_m3", "max"),
        )
        .sort_values(["year", "anchor"])
    )


def corridor_path(scenario: str, year: int) -> Path | None:
    names = {
        "prey-only": ROOT
        / f"outputs/lcp_v2/prey_only_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "prey+ice": ROOT
        / f"outputs/lcp_v2/prey_ice_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "prey+vessel": ROOT
        / f"outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
        "prey+ice+vessel": ROOT
        / f"outputs/lcp_v2/prey_ice_ship_brian_kJm2_frozen_C3_lcp_target{year}_v2.csv",
    }
    path = names[scenario]
    return path if path.exists() else None


def load_corridor(scenario: str, year: int) -> pd.DataFrame | None:
    path = corridor_path(scenario, year)
    if path is None:
        return None
    data = pd.read_csv(path, usecols=["grid_id", "lcp_frequency_normalized"])
    data["grid_id"] = data["grid_id"].astype(str)
    return data.rename(columns={"lcp_frequency_normalized": "corridor_frequency"})


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 5 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def corridor_overlap(
    gridded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abundance = gridded[
        (gridded["timing_class"] == "late_season_abundance")
        & gridded["mean_euphausiid_abundance_m3"].notna()
    ].copy()
    overlap_rows = []
    for (source_id, source_family, year, method), group in abundance.groupby(
        ["source_id", "source_family", "year", "method_group"], dropna=False
    ):
        year = int(year)
        for scenario in ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]:
            corridor = load_corridor(scenario, year)
            if corridor is None:
                continue
            merged = group.merge(corridor, on="grid_id", how="left")
            positive = merged[merged["euphausiid_detected"]]
            rho, p_value = safe_spearman(
                merged["mean_euphausiid_abundance_m3"], merged["corridor_frequency"]
            )
            sampled_frequent = 100.0 * (merged["corridor_frequency"] > 0.05).mean()
            positive_frequent = (
                100.0 * (positive["corridor_frequency"] > 0.05).mean()
                if len(positive)
                else np.nan
            )
            overlap_rows.append(
                {
                    "source_id": source_id,
                    "source_family": source_family,
                    "year": year,
                    "method_group": method,
                    "scenario": scenario,
                    "sampled_grid_cells": len(merged),
                    "euphausiid_positive_grid_cells": len(positive),
                    "sampled_cells_in_frequent_corridor_pct": sampled_frequent,
                    "positive_cells_in_frequent_corridor_pct": positive_frequent,
                    "positive_minus_sampled_overlap_percentage_points": positive_frequent
                    - sampled_frequent,
                    "mean_frequency_all_sampled_cells": merged["corridor_frequency"].mean(),
                    "mean_frequency_positive_cells": positive["corridor_frequency"].mean(),
                    "spearman_abundance_vs_frequency": rho,
                    "spearman_p_value": p_value,
                    "timing_warning": "late-season euphausiid sample compared with May-July modeled corridor",
                    "inference_role": "diagnostic spatial compatibility only",
                }
            )

    target = gridded[
        gridded["timing_class"].isin(
            ["target_window_abundance", "target_window_abundance_context"]
        )
        & gridded["mean_euphausiid_abundance_m3"].notna()
    ].copy()
    target_rows = []
    for (source_id, source_family, year, method), group in target.groupby(
        ["source_id", "source_family", "year", "method_group"], dropna=False
    ):
        year = int(year)
        for scenario in ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]:
            corridor = load_corridor(scenario, year)
            if corridor is None:
                continue
            merged = group.merge(corridor, on="grid_id", how="left")
            abundance_values = merged["mean_euphausiid_abundance_m3"].fillna(0.0)
            q75 = float(abundance_values.quantile(0.75))
            high = merged[abundance_values >= q75]
            rho, p_value = safe_spearman(
                merged["mean_euphausiid_abundance_m3"], merged["corridor_frequency"]
            )
            weighted_frequency = np.nan
            if abundance_values.sum() > 0:
                weighted_frequency = float(
                    np.average(
                        merged["corridor_frequency"].fillna(0.0),
                        weights=abundance_values,
                    )
                )
            all_frequent = 100.0 * (merged["corridor_frequency"] > 0.05).mean()
            high_frequent = 100.0 * (high["corridor_frequency"] > 0.05).mean()
            target_rows.append(
                {
                    "source_id": source_id,
                    "source_family": source_family,
                    "year": year,
                    "method_group": method,
                    "scenario": scenario,
                    "sampled_grid_cells": len(merged),
                    "high_abundance_grid_cells": len(high),
                    "top_quartile_threshold_abundance_m3": q75,
                    "sampled_cells_in_frequent_corridor_pct": all_frequent,
                    "high_abundance_cells_in_frequent_corridor_pct": high_frequent,
                    "high_minus_sampled_overlap_percentage_points": high_frequent
                    - all_frequent,
                    "mean_frequency_all_sampled_cells": merged["corridor_frequency"].mean(),
                    "mean_frequency_high_abundance_cells": high["corridor_frequency"].mean(),
                    "abundance_weighted_mean_frequency": weighted_frequency,
                    "spearman_abundance_vs_frequency": rho,
                    "spearman_p_value": p_value,
                    "timing_warning": "none; observations and modeled corridors are both within May-July",
                    "inference_role": "target-season spatial compatibility diagnostic; not pelagic-corridor validation",
                }
            )

    quality = gridded[gridded["timing_class"] == "target_window_quality_only"].copy()
    quality_rows = []
    for (source_id, source_family, year, method), group in quality.groupby(
        ["source_id", "source_family", "year", "method_group"], dropna=False
    ):
        year = int(year)
        for scenario in ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]:
            corridor = load_corridor(scenario, year)
            if corridor is None:
                continue
            merged = group.merge(corridor, on="grid_id", how="left")
            rho, p_value = safe_spearman(
                merged["mean_lipid_ug_mg_wet"], merged["corridor_frequency"]
            )
            quality_rows.append(
                {
                    "source_id": source_id,
                    "source_family": source_family,
                    "year": year,
                    "method_group": method,
                    "scenario": scenario,
                    "quality_sample_grid_cells": len(merged),
                    "quality_cells_in_frequent_corridor_pct": 100.0
                    * (merged["corridor_frequency"] > 0.05).mean(),
                    "mean_corridor_frequency": merged["corridor_frequency"].mean(),
                    "spearman_lipid_vs_frequency": rho,
                    "spearman_p_value": p_value,
                    "sampling_warning": "targeted specimen-quality samples provide no abundance background",
                    "inference_role": "target-season prey-quality context only",
                }
            )
    return (
        pd.DataFrame(overlap_rows),
        pd.DataFrame(target_rows),
        pd.DataFrame(quality_rows),
    )


def source_year_matrix() -> pd.DataFrame:
    rows = []
    definitions = {
        "CSESP abundance/biomass": ({2011, 2012, 2013, 2014}, "late_season_abundance"),
        "Point Barrow abundance": ({2011, 2012, 2013, 2014, 2015}, "late_season_abundance"),
        "AMBON abundance/biomass": ({2015, 2017}, "late_season_abundance"),
        "Arctic IERP abundance": ({2017, 2019}, "late_season_abundance"),
        "ASGARD abundance": ({2017, 2018}, "target_window_abundance"),
        "ASGARD prey quality": ({2017, 2018}, "target_window_quality_only"),
        "IERP regional index": (set(TARGET_YEARS), "late_season_regional_index"),
        "Iwahara published survey": ({2013}, "published_target_window_abundance"),
        "ARA10 broad mesozooplankton": ({2019}, "partial_target_broad_group"),
    }
    for source, (years, evidence_class) in definitions.items():
        for year in TARGET_YEARS:
            assigned_class = evidence_class if year in years else "none"
            if source == "AMBON abundance/biomass" and year == 2015:
                assigned_class = "archived_quantitative_linkage_unusable"
            rows.append(
                {
                    "source": source,
                    "year": year,
                    "evidence_class": assigned_class,
                }
            )
    return pd.DataFrame(rows)


def integration_gate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "criterion": "May-July quantitative euphausiid abundance",
                "status": "partial",
                "evidence": "Station-level June abundance is available for ASGARD 2017-2018, with sparse additional target-window records in other years",
                "model_consequence": "supports a two-year spatial diagnostic but not annual pelagic prey utility",
            },
            {
                "criterion": "2011-2019 annual temporal coverage",
                "status": "fail",
                "evidence": "A regional index spans all target years, but Chukchi inputs are August-October and target-season station coverage is limited to selected years",
                "model_consequence": "cannot form comparable annual surfaces",
            },
            {
                "criterion": "Comparable gear and taxonomic resolution",
                "status": "fail",
                "evidence": "150-um, 335/500/505-um, rapid assessment, and targeted specimens differ",
                "model_consequence": "raw magnitudes cannot be pooled",
            },
            {
                "criterion": "Abundance paired with prey quality/energy",
                "status": "partial",
                "evidence": "June 2017-2018 abundance and targeted lipid observations can be paired in shared grid cells, but not across the full annual series",
                "model_consequence": "supports limited quantity-quality context, not an annual energy surface",
            },
            {
                "criterion": "Broad Pacific Arctic spatial coverage",
                "status": "partial",
                "evidence": "Combined surveys cover DBO3, northern Chukchi, and Barrow Canyon, but footprints vary",
                "model_consequence": "supports observational evidence mapping only",
            },
            {
                "criterion": "Current/advection information paired to prey",
                "status": "fail",
                "evidence": "Hydrography exists in some cruises but no harmonized prey-transport field",
                "model_consequence": "cannot build a transport-conditioned krill surface",
            },
        ]
    )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["text"],
    )


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def plot_coverage_figure(
    matrix: pd.DataFrame, gridded: pd.DataFrame, gate: pd.DataFrame
) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    grid_spec = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 1.0],
        height_ratios=[0.88, 1.12],
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.96,
        wspace=0.28,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(grid_spec[0, :])
    source_order = matrix["source"].drop_duplicates().tolist()
    class_codes = {
        "none": 0,
        "late_season_abundance": 1,
        "late_season_regional_index": 2,
        "target_window_quality_only": 3,
        "target_window_abundance": 4,
        "published_target_window_abundance": 5,
        "partial_target_broad_group": 6,
        "archived_quantitative_linkage_unusable": 7,
    }
    image = np.zeros((len(source_order), len(TARGET_YEARS)))
    for row_i, source in enumerate(source_order):
        for col_i, year in enumerate(TARGET_YEARS):
            value = matrix.loc[
                (matrix["source"] == source) & (matrix["year"] == year), "evidence_class"
            ].iloc[0]
            image[row_i, col_i] = class_codes[value]
    cmap = ListedColormap(
        [
            "#F4F5F5",
            PALETTE["late"],
            PALETTE["regional_index"],
            PALETTE["target_quality"],
            PALETTE["target_abundance"],
            "#6B5A95",
            PALETTE["partial_target_broad"],
            "#9A6A4F",
        ]
    )
    ax_a.imshow(image, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=7.5)
    ax_a.set_xticks(range(len(TARGET_YEARS)), TARGET_YEARS)
    ax_a.set_yticks(range(len(source_order)), source_order)
    ax_a.set_xlabel("Year")
    for x in np.arange(-0.5, len(TARGET_YEARS), 1):
        ax_a.axvline(x, color="white", lw=1)
    for y in np.arange(-0.5, len(source_order), 1):
        ax_a.axhline(y, color="white", lw=1)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["late"], markeredgecolor="none", label="Aug-Oct abundance/biomass"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["regional_index"], markeredgecolor="none", label="Annual regional index; mainly late season"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["target_quality"], markeredgecolor="none", label="May-Jul prey quality only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["target_abundance"], markeredgecolor="none", label="May-Jul station abundance"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#6B5A95", markeredgecolor="none", label="May-Jul published snapshot"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["partial_target_broad"], markeredgecolor="none", label="Partial-window broad group only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#9A6A4F", markeredgecolor="none", label="Quantitative archive; linkage unusable"),
    ]
    ax_a.legend(handles=legend, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(grid_spec[1, 0])
    domain = gpd.read_file(DOMAIN).to_crs("EPSG:3338")
    land = gpd.read_file(LAND).to_crs("EPSG:3338")
    dbo = gpd.read_file(DBO).to_crs("EPSG:3338")
    xmin, ymin, xmax, ymax = domain.total_bounds
    pad_x = 0.035 * (xmax - xmin)
    pad_y = 0.035 * (ymax - ymin)
    land_local = land.cx[xmin - pad_x : xmax + pad_x, ymin - pad_y : ymax + pad_y]
    domain.plot(ax=ax_b, color=PALETTE["ocean"], edgecolor=PALETTE["domain"], linewidth=0.8)
    land_local.plot(ax=ax_b, color=PALETTE["land"], edgecolor="#8B8B82", linewidth=0.35)
    dbo.boundary.plot(ax=ax_b, color="#4D7774", linewidth=0.7, linestyle=(0, (4, 2)))
    abundance = gridded[
        (gridded["timing_class"] == "late_season_abundance")
        & gridded["mean_euphausiid_abundance_m3"].notna()
    ].copy()
    for family, group in abundance.groupby("source_family"):
        ax_b.scatter(
            group["x"],
            group["y"],
            s=5 + 15 * group["abundance_percentile_within_source_year_method"].fillna(0) / 100,
            color=PALETTE.get(family, "#66757C"),
            alpha=0.48,
            linewidth=0,
            label=family,
        )
    quality = gridded[gridded["timing_class"] == "target_window_quality_only"]
    target_abundance = gridded[gridded["timing_class"] == "target_window_abundance"]
    ax_b.scatter(
        target_abundance["x"],
        target_abundance["y"],
        s=7
        + 19
        * target_abundance["abundance_percentile_within_source_year_method"].fillna(0)
        / 100,
        marker="o",
        color=PALETTE["ASGARD abundance"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
        label="ASGARD abundance",
        zorder=4,
    )
    ax_b.scatter(
        quality["x"],
        quality["y"],
        marker="^",
        s=27,
        color=PALETTE["ASGARD quality"],
        edgecolor="white",
        linewidth=0.4,
        label="ASGARD quality",
        zorder=5,
    )
    for _, row in dbo.iterrows():
        point = row.geometry.centroid
        ax_b.text(point.x, point.y, row["name"], ha="center", va="center", fontsize=6, color="#325D5A")
    ax_b.text(-0.02, 0.02, "Marker size: within-source-year abundance rank", transform=ax_b.transAxes, fontsize=5.5, color="#586269")
    ax_b.set_axis_off()
    ax_b.set_aspect("equal")
    ax_b.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_b.set_ylim(ymin - pad_y, ymax + pad_y)
    handles, labels = ax_b.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_b.legend(unique.values(), unique.keys(), loc="upper left", ncol=2, handletextpad=0.4, columnspacing=0.8)
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(grid_spec[1, 1])
    status_code = {"fail": 0, "partial": 1, "pass": 2}
    status_colors = ListedColormap([PALETTE["fail"], PALETTE["partial"], PALETTE["pass"]])
    values = np.array([[status_code[value]] for value in gate["status"]])
    ax_c.imshow(values, aspect="auto", cmap=status_colors, vmin=-0.5, vmax=2.5)
    ax_c.set_xticks([0], ["Integration gate"])
    short_gate_labels = [
        "May-Jul abundance",
        "Annual coverage",
        "Comparable gear/taxonomy",
        "Abundance + prey quality",
        "Broad spatial coverage",
        "Current/advection pairing",
    ]
    ax_c.set_yticks(range(len(gate)), short_gate_labels)
    ax_c.tick_params(axis="y", labelsize=6)
    for index, status in enumerate(gate["status"]):
        ax_c.text(0, index, status.upper(), ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    for y in np.arange(-0.5, len(gate), 1):
        ax_c.axhline(y, color="white", lw=1)
    ax_c.text(
        0.5,
        -0.15,
        "Decision: observational evidence layer only; no annual pelagic LCP surface",
        transform=ax_c.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=PALETTE["text"],
        wrap=True,
    )
    panel_label(ax_c, "C")

    stem = FIG_OUT / "phase8n_euphausiid_data_coverage_and_integration_gate_v2"
    export_figure(fig, stem)
    plt.close(fig)


def plot_overlap_figure(overlap: pd.DataFrame) -> None:
    configure_matplotlib()
    plot_data = overlap[overlap["sampled_grid_cells"] >= 5].copy()
    scenarios = ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]
    families = plot_data["source_family"].drop_duplicates().tolist()
    x_positions = {scenario: index for index, scenario in enumerate(scenarios)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.29, top=0.91)

    ax = axes[0]
    for family_i, family in enumerate(families):
        group = plot_data[plot_data["source_family"] == family]
        jitter = (family_i - (len(families) - 1) / 2) * 0.045
        ax.scatter(
            [x_positions[value] + jitter for value in group["scenario"]],
            group["positive_minus_sampled_overlap_percentage_points"],
            s=22,
            color=PALETTE.get(family, "#65747B"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
            label=family,
        )
    ax.axhline(0, color="#6D7478", lw=0.8, ls="--")
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=18, ha="right")
    ax.set_ylabel("Positive-cell overlap minus sampled-background overlap\n(percentage points)")
    ax.set_xlim(-0.45, len(scenarios) - 0.55)
    ax.legend(loc="best")
    panel_label(ax, "A")

    ax = axes[1]
    for family_i, family in enumerate(families):
        group = plot_data[plot_data["source_family"] == family]
        jitter = (family_i - (len(families) - 1) / 2) * 0.045
        ax.scatter(
            [x_positions[value] + jitter for value in group["scenario"]],
            group["spearman_abundance_vs_frequency"],
            s=22,
            color=PALETTE.get(family, "#65747B"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.axhline(0, color="#6D7478", lw=0.8, ls="--")
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=18, ha="right")
    ax.set_ylabel("Spearman correlation: euphausiid abundance vs. $F_s$")
    ax.set_xlim(-0.45, len(scenarios) - 0.55)
    ax.set_ylim(-1.05, 1.05)
    panel_label(ax, "B")
    fig.text(
        0.5,
        0.025,
        "Each point is one source-year-method group. Euphausiids were sampled mainly in Aug-Oct; corridors represent May-Jul.",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#4B555B",
    )
    stem = FIG_OUT / "phase8n_late_season_euphausiid_corridor_overlap_diagnostic_v1"
    export_figure(fig, stem)
    plt.close(fig)


def plot_target_window_diagnostic(
    target_overlap: pd.DataFrame,
    gridded: pd.DataFrame,
    regional_index: pd.DataFrame,
) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=False)
    grid_spec = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.12,
        top=0.95,
        wspace=0.27,
        hspace=0.30,
    )
    domain = gpd.read_file(DOMAIN).to_crs("EPSG:3338")
    land = gpd.read_file(LAND).to_crs("EPSG:3338")
    dbo = gpd.read_file(DBO).to_crs("EPSG:3338")
    xmin, ymin, xmax, ymax = domain.total_bounds
    pad_x = 0.035 * (xmax - xmin)
    pad_y = 0.035 * (ymax - ymin)
    land_local = land.cx[xmin - pad_x : xmax + pad_x, ymin - pad_y : ymax + pad_y]
    asgard = gridded[
        (gridded["source_family"] == "ASGARD abundance")
        & gridded["mean_euphausiid_abundance_m3"].notna()
    ].copy()
    color_values = np.log10(1 + asgard["mean_euphausiid_abundance_m3"].clip(lower=0))
    norm = mpl.colors.Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    scatter_for_bar = None
    for panel_index, year in enumerate([2017, 2018]):
        ax = fig.add_subplot(grid_spec[0, panel_index])
        domain.plot(ax=ax, color=PALETTE["ocean"], edgecolor=PALETTE["domain"], linewidth=0.7)
        land_local.plot(ax=ax, color=PALETTE["land"], edgecolor="#8B8B82", linewidth=0.3)
        dbo.boundary.plot(ax=ax, color="#4D7774", linewidth=0.65, linestyle=(0, (4, 2)))
        group = asgard[asgard["year"] == year]
        scatter_for_bar = ax.scatter(
            group["x"],
            group["y"],
            c=np.log10(1 + group["mean_euphausiid_abundance_m3"].clip(lower=0)),
            cmap="viridis",
            norm=norm,
            s=24,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        for _, row in dbo.iterrows():
            point = row.geometry.centroid
            ax.text(
                point.x,
                point.y,
                row["name"],
                ha="center",
                va="center",
                fontsize=5.5,
                color="#325D5A",
            )
        ax.text(0.98, 0.98, str(year), transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        panel_label(ax, "A" if panel_index == 0 else "B")
    if scatter_for_bar is not None:
        colorbar = fig.colorbar(
            scatter_for_bar,
            ax=[fig.axes[0], fig.axes[1]],
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            aspect=35,
        )
        colorbar.set_label(r"$\log_{10}(1 + \mathrm{euphausiids\ [ind.\ m^{-3}]})$")

    ax_c = fig.add_subplot(grid_spec[1, 0])
    diagnostic = target_overlap[
        (target_overlap["source_family"] == "ASGARD abundance")
        & (target_overlap["sampled_grid_cells"] >= 5)
    ].copy()
    scenarios = ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]
    x = np.arange(len(scenarios))
    for year, marker, color in [(2017, "o", "#3B7890"), (2018, "s", "#B25A45")]:
        group = diagnostic[diagnostic["year"] == year].set_index("scenario").reindex(scenarios)
        ax_c.plot(
            x,
            group["abundance_weighted_mean_frequency"],
            marker=marker,
            color=color,
            lw=1.2,
            ms=4.5,
            label=str(year),
        )
    ax_c.set_xticks(x, scenarios, rotation=18, ha="right")
    ax_c.set_ylabel("Abundance-weighted mean $F_s$")
    ax_c.legend(frameon=False, loc="upper right")
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(grid_spec[1, 1])
    for region, label, color, marker in [
        ("CS", "Chukchi Sea", "#3B7890", "o"),
        ("NBS", "Northern Bering Sea", "#B25A45", "s"),
    ]:
        group = regional_index[regional_index["region"] == region].sort_values("year")
        ax_d.plot(
            group["year"],
            group["bias_corrected_regional_index"],
            color=color,
            marker=marker,
            lw=1.2,
            ms=4,
            label=label,
        )
    ax_d.set_yscale("log")
    ax_d.set_xticks(TARGET_YEARS)
    ax_d.tick_params(axis="x", rotation=45)
    ax_d.set_ylabel("Bias-corrected regional euphausiid index")
    ax_d.legend(frameon=False, loc="upper left")
    panel_label(ax_d, "D")

    fig.text(
        0.5,
        0.025,
        "A-B, June station abundance. C, compatibility with existing May-July corridors. D, regional index based mainly on Aug-Oct samples; not used as a corridor input.",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#4B555B",
    )
    stem = FIG_OUT / "phase8n_target_window_euphausiid_integration_diagnostic_v1"
    export_figure(fig, stem)
    plt.close(fig)


def write_summary(
    inventory: pd.DataFrame,
    observations: pd.DataFrame,
    gridded: pd.DataFrame,
    overlap: pd.DataFrame,
    target_overlap: pd.DataFrame,
    regional_index: pd.DataFrame,
    pairing_summary: pd.DataFrame,
    dbo_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    late_years = sorted(
        int(value)
        for value in observations.loc[
            observations["timing_class"] == "late_season_abundance", "year"
        ]
        .dropna()
        .unique()
    )
    quality_years = sorted(
        int(value)
        for value in observations.loc[
            observations["timing_class"] == "target_window_quality_only", "year"
        ]
        .dropna()
        .unique()
    )
    target_abundance_years = sorted(
        int(value)
        for value in observations.loc[
            observations["timing_class"] == "target_window_abundance", "year"
        ]
        .dropna()
        .unique()
    )
    asgard_target = observations[
        observations["timing_class"] == "target_window_abundance"
    ]
    asgard_matched = int(asgard_target["grid_match_within_15km"].sum())
    chukchi_index_years = sorted(
        int(value)
        for value in regional_index.loc[regional_index["region"] == "CS", "year"].unique()
    )
    target_diagnostic_rows = target_overlap[
        target_overlap["source_family"] == "ASGARD abundance"
    ]
    dbo3 = dbo_summary[dbo_summary["anchor"] == "DBO3"]
    dbo3_year_counts = ", ".join(
        f"{int(row.year)}: {int(row.sampling_stations)}"
        for row in dbo3.itertuples(index=False)
    )
    lines = [
        "# Phase 8N Euphausiid Data Integration Summary",
        "",
        "## Decision",
        "",
        "The expanded search recovered quantitative June euphausiid abundance for 2017-2018 and a continuous 2011-2019 Chukchi regional index. It still does not support a comparable annual May-July euphausiid utility surface for 2011-2019. No pelagic-prey LCP scenario was constructed or run.",
        "",
        "## What the search added",
        "",
        f"- Screened {len(inventory)} source families or catalogs and downloaded the strongest available station-level packages.",
        f"- Harmonized {len(observations):,} sampling events or prey-quality locations from public station files.",
        f"- Matched {gridded['grid_id'].nunique():,} unique study-region ocean cells within {GRID_MATCH_LIMIT_KM:g} km of a 10-km grid center.",
        f"- Quantitative euphausiid abundance is available in late-season surveys for {late_years}.",
        f"- Quantitative target-window station abundance is available for {target_abundance_years}: {len(asgard_target):,} ASGARD stations, of which {asgard_matched:,} matched the study-region grid.",
        f"- The ASGARD data include {int(dbo3['sampling_stations'].sum())} June stations within the DBO3 anchor ({dbo3_year_counts}), directly documenting target-window euphausiid occurrence and abundance in the southeastern Chukchi Sea. This is a two-year station record, not an annual prey surface.",
        f"- Target-window euphausiid lipid quality is available for {quality_years}, but those samples do not provide abundance or a sampled-background surface.",
        f"- The bias-corrected Chukchi regional index covers {chukchi_index_years}, but its input samples are primarily August-October.",
        "- The 2019 ARA10A/B archive overlaps late July, but its public mesozooplankton table reports broad taxonomic groups and does not isolate euphausiids.",
        "",
        "## Framework integration test",
        "",
        f"- Generated {len(overlap):,} source-year-method-scenario diagnostic comparisons against existing corridor-frequency grids.",
        f"- Generated {len(target_diagnostic_rows):,} ASGARD year-scenario comparisons using June station abundance, including abundance-weighted mean corridor frequency and rank association.",
        f"- Paired June abundance and lipid-quality observations in {int(pairing_summary['paired_grid_cells'].sum()) if not pairing_summary.empty else 0} year-specific grid cells.",
        "- Late-season station values were not interpolated and were not used to alter resistance, endpoints, or least-cost paths.",
        "- Within-dataset ranks are retained only for visualization; raw magnitudes from different gears were not pooled.",
        "- Late-season overlap diagnostics are season-mismatched spatial checks. The June ASGARD diagnostics match the target season but cover only 2017-2018; neither comparison validates an annual pelagic corridor surface.",
        "",
        "## Scientifically defensible use",
        "",
        "Use the June 2017-2018 ASGARD observations as a target-season pelagic-prey evidence layer and the regional index as late-season temporal context. These data can test whether existing benthic-prey corridors intersect observed euphausiid concentrations in two years, but they cannot calibrate annual pelagic utility, a low-medium-high krill scenario, or a 2011-2019 pelagic-prey LCP ensemble. The present corridors should therefore remain explicitly benthic-prey corridors.",
        "",
        "## Gate details",
        "",
    ]
    for row in gate.itertuples(index=False):
        lines.append(f"- **{row.criterion} ({row.status})**: {row.evidence}. Consequence: {row.model_consequence}.")
    (OUT / "phase8n_euphausiid_integration_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_manifest(paths: list[Path], corridor_inputs: list[Path]) -> dict[str, object]:
    records = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    corridor_records = []
    for path in sorted(set(corridor_inputs)):
        corridor_records.append(
            {
                "file": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "phase": PHASE,
        "script": SCRIPT,
        "completed_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "decision": "observational_evidence_layer_only_no_pelagic_lcp_surface",
        "grid_match_limit_km": GRID_MATCH_LIMIT_KM,
        "outputs": records,
        "read_only_corridor_inputs": corridor_records,
        "primary_outputs_modified": False,
        "manuscript_or_si_modified": False,
    }


def main() -> None:
    ensure_dirs()
    write_json(
        STATUS_PATH,
        {
            "phase": PHASE,
            "started_utc": utc_now(),
            "status": "running",
            "stages": {},
        },
    )
    try:
        set_stage("8N.1_source_inventory", "running")
        inventory = source_inventory()
        inventory_path = TABLE_OUT / "phase8n_euphausiid_source_inventory.csv"
        inventory.to_csv(inventory_path, index=False)
        obis = obis_duplication_audit()
        obis_path = TABLE_OUT / "phase8n_obis_source_duplication_audit.csv"
        obis.to_csv(obis_path, index=False)
        set_stage(
            "8N.1_source_inventory",
            "complete",
            source_records=len(inventory),
            obis_dataset_records=len(obis),
        )

        set_stage("8N.2_station_harmonization", "running")
        observations = load_observations()
        raw_observation_path = TABLE_OUT / "phase8n_harmonized_euphausiid_observations.csv"
        observations.to_csv(raw_observation_path, index=False)
        set_stage(
            "8N.2_station_harmonization",
            "complete",
            observation_rows=len(observations),
            source_ids=int(observations["source_id"].nunique()),
        )

        set_stage("8N.3_grid_compatibility", "running")
        matched_observations, gridded = match_to_grid(observations)
        matched_path = TABLE_OUT / "phase8n_euphausiid_observations_grid_match.csv"
        gridded_path = TABLE_OUT / "phase8n_euphausiid_grid_evidence.csv"
        matched_observations.to_csv(matched_path, index=False)
        gridded.to_csv(gridded_path, index=False)
        coverage = coverage_summary(matched_observations)
        coverage_path = TABLE_OUT / "phase8n_temporal_spatial_coverage.csv"
        coverage.to_csv(coverage_path, index=False)
        regional_index = synthesis_regional_index()
        regional_index_path = (
            TABLE_OUT / "phase8n_arctic_ierp_regional_euphausiid_index_2011_2019.csv"
        )
        regional_index.to_csv(regional_index_path, index=False)
        pairing, pairing_summary = asgard_abundance_quality_pairing(gridded)
        pairing_path = TABLE_OUT / "phase8n_asgard_abundance_quality_grid_pairing.csv"
        pairing_summary_path = (
            TABLE_OUT / "phase8n_asgard_abundance_quality_pairing_summary.csv"
        )
        pairing.to_csv(pairing_path, index=False)
        pairing_summary.to_csv(pairing_summary_path, index=False)
        dbo_summary = asgard_dbo_summary(matched_observations)
        dbo_summary_path = (
            TABLE_OUT / "phase8n_asgard_euphausiid_abundance_by_dbo_anchor.csv"
        )
        dbo_summary.to_csv(dbo_summary_path, index=False)
        set_stage(
            "8N.3_grid_compatibility",
            "complete",
            matched_observation_rows=int(matched_observations["grid_match_within_15km"].sum()),
            gridded_evidence_rows=len(gridded),
            unique_grid_cells=int(gridded["grid_id"].nunique()),
            asgard_dbo3_station_rows=int(
                dbo_summary.loc[
                    dbo_summary["anchor"] == "DBO3", "sampling_stations"
                ].sum()
            ),
        )

        set_stage(
            "8N.7_arctic_ierp_synthesis_resolution",
            "complete",
            asgard_target_window_station_rows=int(
                (matched_observations["source_family"] == "ASGARD abundance").sum()
            ),
            asgard_target_window_matched_rows=int(
                (
                    (matched_observations["source_family"] == "ASGARD abundance")
                    & matched_observations["grid_match_within_15km"]
                ).sum()
            ),
            regional_index_rows=len(regional_index),
            paired_abundance_quality_grid_rows=len(pairing),
        )

        set_stage("8N.4_existing_framework_diagnostic", "running")
        overlap, target_overlap, quality_overlap = corridor_overlap(gridded)
        overlap_path = TABLE_OUT / "phase8n_late_season_euphausiid_corridor_overlap_diagnostic.csv"
        target_overlap_path = (
            TABLE_OUT / "phase8n_target_window_euphausiid_corridor_diagnostic.csv"
        )
        quality_path = TABLE_OUT / "phase8n_target_window_quality_corridor_context.csv"
        overlap.to_csv(overlap_path, index=False)
        target_overlap.to_csv(target_overlap_path, index=False)
        quality_overlap.to_csv(quality_path, index=False)
        set_stage(
            "8N.4_existing_framework_diagnostic",
            "complete",
            late_season_comparisons=len(overlap),
            target_window_abundance_comparisons=len(target_overlap),
            target_window_quality_context_rows=len(quality_overlap),
        )

        set_stage("8N.5_integration_gate", "running")
        matrix = source_year_matrix()
        matrix_path = TABLE_OUT / "phase8n_source_year_evidence_matrix.csv"
        matrix.to_csv(matrix_path, index=False)
        gate = integration_gate()
        gate_path = TABLE_OUT / "phase8n_pelagic_surface_integration_gate.csv"
        gate.to_csv(gate_path, index=False)
        write_summary(
            inventory,
            matched_observations,
            gridded,
            overlap,
            target_overlap,
            regional_index,
            pairing_summary,
            dbo_summary,
            gate,
        )
        set_stage(
            "8N.5_integration_gate",
            "complete",
            decision="observational_evidence_layer_only_no_pelagic_lcp_surface",
        )

        set_stage("8N.6_diagnostic_figures", "running")
        plot_coverage_figure(matrix, gridded, gate)
        plot_overlap_figure(overlap)
        plot_target_window_diagnostic(target_overlap, gridded, regional_index)
        set_stage("8N.6_diagnostic_figures", "complete", figure_stems=3)

        output_paths = [
            RAW
            / "arctic_ierp_synthesis_2026/arctic_ierp_synthesis_dataone_download_manifest.json",
            inventory_path,
            obis_path,
            raw_observation_path,
            matched_path,
            gridded_path,
            coverage_path,
            regional_index_path,
            pairing_path,
            pairing_summary_path,
            dbo_summary_path,
            overlap_path,
            target_overlap_path,
            quality_path,
            matrix_path,
            gate_path,
            OUT / "phase8n_euphausiid_integration_summary.md",
        ]
        for stem in [
            FIG_OUT / "phase8n_euphausiid_data_coverage_and_integration_gate_v2",
            FIG_OUT / "phase8n_late_season_euphausiid_corridor_overlap_diagnostic_v1",
            FIG_OUT / "phase8n_target_window_euphausiid_integration_diagnostic_v1",
        ]:
            output_paths.extend(stem.with_suffix(suffix) for suffix in [".pdf", ".svg", ".png", ".tiff"])
        corridor_inputs = [
            path
            for year in TARGET_YEARS
            for scenario in ["prey-only", "prey+ice", "prey+vessel", "prey+ice+vessel"]
            if (path := corridor_path(scenario, year)) is not None
        ]
        write_json(MANIFEST_PATH, build_manifest(output_paths, corridor_inputs))
        state = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        state["status"] = "complete"
        state["completed_utc"] = utc_now()
        write_json(STATUS_PATH, state)
        print(f"Phase 8N complete: {len(observations):,} observations, {len(gridded):,} gridded rows")
    except Exception as error:
        set_stage("failure", "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
