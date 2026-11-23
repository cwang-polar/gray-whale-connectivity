#!/usr/bin/env python3
"""Isolated ASAMM distance/time-normalized observation versus frozen path costs."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, shape
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gray_whale_corridor.metrics.survey_effort import (
    RATE_FORMULAS, aggregate_flights, clipped_fractions, flight_bootstrap,
    point_membership, reconstruct_segments, resolve_modes, summarize_rates,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_inputs(config, paths):
    event_files = paths.input_glob(config["raw_root"], "*/_Output__*.csv")
    summary_files = paths.input_glob(config["raw_root"], "*/_SummaryInfo__*.csv")
    if len(event_files) != 4 or len(summary_files) != 4:
        raise ValueError("Expected four primary event tables and four flight-summary tables")
    cols = ["ID", "Event", "Flt_No", "GMT_Minus8_DateTime", "ArcLat", "ArcLong", "LatTemp", "LongTemp",
            "Entry", "Species", "Totalno", "Final_group", "Repeat", "FltType", "Aerial_Survey", "PrimObs", "Seastate"]
    event_parts = []
    for path in event_files:
        df = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
        df["datetime"] = pd.to_datetime(df["GMT_Minus8_DateTime"], errors="coerce", format="mixed")
        df = df.loc[df["datetime"].dt.year.isin(config["years"]) & df["datetime"].dt.month.isin([6, 7])].copy()
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["source_file"] = str(path.relative_to(paths.data_root))
        df["source_id"] = path.parent.name + ":" + df["ID"]
        df["flight_key"] = path.parent.name + ":" + df["year"].astype(str) + ":" + df["Flt_No"]
        df["event_number"] = pd.to_numeric(df["Event"], errors="coerce")
        lat = pd.to_numeric(df["ArcLat"], errors="coerce")
        lon = pd.to_numeric(df["ArcLong"], errors="coerce")
        lat_alt = pd.to_numeric(df["LatTemp"], errors="coerce")
        lon_alt = pd.to_numeric(df["LongTemp"], errors="coerce")
        df["lat"] = lat.where(lat.between(50, 80), lat_alt)
        df["lon"] = lon.where(lon.between(-180, 360), lon_alt)
        df["lon"] = (df["lon"] + 180) % 360 - 180
        df["entry_clean"] = df["Entry"].fillna("").str.strip().str.lower()
        df["repeat_clean"] = df["Repeat"].fillna("").str.strip().str.lower()
        final = pd.to_numeric(df["Final_group"], errors="coerce")
        total = pd.to_numeric(df["Totalno"], errors="coerce")
        df["whale_count"] = final.where(final.gt(0), total.where(total.gt(0)))
        event_parts.append(df)
    events = pd.concat(event_parts, ignore_index=True)
    if events["source_id"].duplicated().any():
        raise ValueError("Duplicate source event identifiers")
    summaries = []
    for path in summary_files:
        df = pd.read_csv(path, dtype=str)
        df["datetime"] = pd.to_datetime(df["StartTime"], errors="coerce", format="mixed")
        df = df.loc[df["datetime"].dt.year.isin(config["years"]) & df["datetime"].dt.month.isin([6, 7])].copy()
        df["year"] = df["datetime"].dt.year
        df["flight_key"] = path.parent.name + ":" + df["year"].astype(str) + ":" + df["Flt_No"]
        for col in ["TransectKilo", "SearchKilo", "TransectMin", "SearchMin", "TotalKilo", "TotalMin"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        summaries.append(df)
    summary = pd.concat(summaries, ignore_index=True)
    if summary["flight_key"].duplicated().any():
        raise ValueError("Flight-summary keys are not unique")
    return events, summary, event_files + summary_files


def make_footprints(config, paths, *, regional_only=False):
    grid = pd.read_csv(paths.input_path(config["grid"]))
    if len(grid) != 2839:
        raise ValueError("Regional grid inventory changed")
    points = shapely.points(grid[["x", "y"]].to_numpy())
    size = config["max_grid_distance_m"]
    extent = box(grid.x.min()-size*3, grid.y.min()-size*3, grid.x.max()+size*3, grid.y.max()+size*3)
    voronoi = shapely.get_parts(shapely.voronoi_polygons(shapely.MultiPoint(points), extend_to=extent))
    assignment = shapely.STRtree(voronoi).query(points, predicate="intersects")
    if len(assignment[0]) != len(grid) or len(np.unique(assignment[0])) != len(grid):
        raise ValueError("Cannot assign one Voronoi polygon to each frozen ocean-grid centre")
    mapping = np.empty(len(grid), dtype=int)
    mapping[assignment[0]] = assignment[1]
    # Reproduce the frozen nearest-ocean-centre <=15 km selection, including
    # coastal cells; square cells would discard previously eligible sightings.
    cells = shapely.intersection(voronoi[mapping], shapely.buffer(points, size, quad_segs=128))
    geojson = json.loads((paths.input_path(config["domain"])).read_text())
    from pyproj import Transformer
    from shapely.ops import transform
    geoms = [shape(f["geometry"]) for f in geojson.get("features", [])]
    if geojson.get("type") in ["Polygon", "MultiPolygon"]:
        geoms.append(shape(geojson))
    def densify_ring(coords):
        dense = []
        a = np.asarray(coords)[:, :2]
        for start, end in zip(a[:-1], a[1:]):
            count = max(1, int(np.ceil(np.abs(end-start).max()/0.01)))
            dense.extend(start + (end-start)*k/count for k in range(count))
        dense.append(a[-1])
        return dense
    from shapely.geometry import Polygon, MultiPolygon
    def densify(geom):
        if geom.geom_type == "Polygon":
            return Polygon(densify_ring(geom.exterior.coords), [densify_ring(r.coords) for r in geom.interiors])
        return MultiPolygon([densify(g) for g in geom.geoms])
    domain = transform(Transformer.from_crs("EPSG:4326", config["projected_crs"], always_xy=True).transform,
                       densify(shapely.union_all(geoms)))
    if regional_only:
        return grid, {"regional": shapely.intersection(shapely.union_all(cells), domain)}
    effort = pd.read_csv(paths.input_path(config["existing_effort_cells"]))
    effort = effort.loc[effort["validation_window"].eq("June_July")
                        & effort["effort_background_type"].eq("strict_effort")
                        & effort["year"].isin(config["years"])]
    coverage = effort.groupby("grid_id")["year"].nunique()
    grid["previous_survey_years"] = grid["grid_id"].map(coverage).fillna(0).astype(int)
    footprints = {"regional": shapely.intersection(shapely.union_all(cells), domain)}
    for threshold, expected in [(7, 105), (8, 52), (9, 19)]:
        selected = grid["previous_survey_years"].ge(threshold)
        if selected.sum() != expected:
            raise ValueError("Previously frozen fixed-mask membership changed")
        footprints[f"stable_{threshold}"] = shapely.intersection(shapely.union_all(cells[selected]), domain)
    return grid, footprints


def reconstruction_qa(segments, summary, out):
    audit_parts = []
    for mode, official in [(2, "Transect"), (4, "Search")]:
        # The 2018-2021 summary schema folds circling into its associated
        # transect/search totals. Compare like-for-like while retaining mode-2
        # only as the temporally consistent primary denominator.
        associated_circle = 5 if mode == 2 else 6
        mode_match = segments["start_mode"].eq(mode) | (segments["year"].ge(2018) & segments["start_mode"].eq(associated_circle))
        subset = segments.loc[mode_match & segments["accepted"]].copy()
        byflight = subset.groupby("flight_key").agg(
            gps_km=("raw_km", "sum"), gps_seconds=("seconds", "sum"), segments=("source_id", "size"))
        qa = summary[["flight_key", "year", official + "Kilo", official + "Min"]].merge(
            byflight, on="flight_key", how="left", validate="one_to_one")
        qa[["gps_km", "gps_seconds", "segments"]] = qa[["gps_km", "gps_seconds", "segments"]].fillna(0)
        qa["mode"] = official.lower()
        qa["summary_comparison_modes"] = np.where(qa.year.ge(2018), f"{mode}+{associated_circle}", str(mode))
        qa["official_km"] = qa[official + "Kilo"]
        qa["official_minutes"] = qa[official + "Min"]
        qa["km_percent_difference"] = (100 * (qa["gps_km"] - qa["official_km"]) / qa["official_km"]).where(qa["official_km"].gt(0))
        qa["absolute_km_difference"] = (qa["gps_km"] - qa["official_km"]).abs()
        qa["flag_over_5pct_and_1km"] = qa["km_percent_difference"].abs().gt(5) & qa["absolute_km_difference"].gt(1)
        audit_parts.append(qa)
    audit = pd.concat(audit_parts, ignore_index=True)
    audit.to_csv(out / "qa" / "flight_summary_reconstruction.csv", index=False)
    return audit


def make_correlations(annual, costs):
    rows = []
    for (window, footprint, mode), d in annual.groupby(["window", "footprint", "effort_definition"]):
        for scenario, c in costs.groupby("scenario"):
            merged = d.merge(c[["year", "mean_path_cost"]], on="year", validate="one_to_one")
            subsets = {
                "2011_2019": merged,
                "2011_2017": merged.loc[merged["year"].le(2017)],
                "omit_2018": merged.loc[merged["year"].ne(2018)],
                "omit_2019": merged.loc[merged["year"].ne(2019)],
            }
            for metric in RATE_FORMULAS:
                for period, sub in subsets.items():
                    a = sub[["mean_path_cost", metric]].dropna()
                    varying = len(a) >= 3 and a.nunique().min() > 1
                    rows.append(dict(window=window, footprint=footprint, effort_definition=mode,
                        scenario=scenario, metric=metric, comparison=period, n_years=len(a),
                        pearson_r=pearsonr(a.iloc[:, 0], a.iloc[:, 1]).statistic if varying else np.nan,
                        spearman_rho=spearmanr(a.iloc[:, 0], a.iloc[:, 1]).statistic if varying else np.nan))
                delta = merged.sort_values("year").set_index("year")[["mean_path_cost", metric]].diff().dropna()
                varying = len(delta) >= 3 and delta.nunique().min() > 1
                rows.append(dict(window=window, footprint=footprint, effort_definition=mode,
                    scenario=scenario, metric=metric, comparison="first_differences", n_years=len(delta),
                    pearson_r=pearsonr(delta.iloc[:, 0], delta.iloc[:, 1]).statistic if varying else np.nan,
                    spearman_rho=spearmanr(delta.iloc[:, 0], delta.iloc[:, 1]).statistic if varying else np.nan))
    return pd.DataFrame(rows)


def markdown_table(frame, digits=3):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(out, annual, costs, correlations, intervals, diagnostics):
    main = annual.loc[annual.window.eq("June_July") & annual.footprint.eq("regional") & annual.effort_definition.eq("transect")].sort_values("year")
    rates = main.set_index("year")
    stats = correlations.loc[correlations.window.eq("June_July") & correlations.footprint.eq("regional") & correlations.effort_definition.eq("transect") & correlations.comparison.eq("2011_2019")]
    corr_rows = []
    for metric in RATE_FORMULAS:
        row = {"Metric": metric}
        for scenario in ["prey_only", "prey_ice"]:
            s = stats.loc[stats.scenario.eq(scenario) & stats.metric.eq(metric)].iloc[0]
            row[scenario + " Pearson r"] = s.pearson_r
            row[scenario + " Spearman rho"] = s.spearman_rho
        corr_rows.append(row)
    fixed = annual.loc[annual.window.eq("June_July") & annual.effort_definition.eq("transect") & annual.year.ge(2017)]
    fixed_table = fixed.pivot(index="footprint",columns="year",values="whales_per_100km").reset_index()
    mode_table = annual.loc[annual.window.eq("June_July") & annual.footprint.eq("regional") & annual.year.ge(2017)].pivot(index="effort_definition", columns="year",values="whales_per_100km").reset_index()
    change18 = 100*(rates.loc[2018,"whales_per_100km"]/rates.loc[2017,"whales_per_100km"]-1)
    change19 = 100*(rates.loc[2019,"whales_per_100km"]/rates.loc[2017,"whales_per_100km"]-1)
    encounter19 = 100*(rates.loc[2019,"sightings_per_100km"]/rates.loc[2017,"sightings_per_100km"]-1)
    group19 = 100*(rates.loc[2019,"mean_group_size"]/rates.loc[2017,"mean_group_size"]-1)
    differenced = correlations.loc[correlations.window.eq("June_July") & correlations.footprint.eq("regional") & correlations.effort_definition.eq("transect") & correlations.comparison.eq("first_differences") & correlations.metric.eq("whales_per_100km")].set_index("scenario")
    qa = pd.read_csv(out/"qa/flight_summary_reconstruction.csv")
    qa_annual = qa.loc[qa["mode"].eq("transect")].groupby("year")[["gps_km","official_km"]].sum()
    max_qa_difference = (100*(qa_annual.gps_km/qa_annual.official_km-1)).abs().max()
    search_flags = int((qa.flag_over_5pct_and_1km & qa["mode"].eq("search")).sum())
    table = main[["year","flight_count","survey_km","survey_hours","sighting_count","reported_whales","whales_per_100km","sightings_per_100km","whales_per_hour","mean_group_size"]].copy()
    table["year"] = table.year.astype(str)
    report = f"""# ASAMM survey-effort-normalized observations

## Annual observations

The primary series uses regional transect effort and gray whale observations recorded in the same survey mode, excluding confirmed repeated sightings. Rates in 2017, 2018 and 2019 were {rates.loc[2017,'whales_per_100km']:.3f}, {rates.loc[2018,'whales_per_100km']:.3f} and {rates.loc[2019,'whales_per_100km']:.3f} reported whales per 100 km. Relative to 2017, changes were {change18:+.1f}% in 2018 and {change19:+.1f}% in 2019.

The 2019 changes in encounter rate and reported whales per sighting were {encounter19:+.1f}% and {group19:+.1f}%, respectively. Whale-count rate is the product of encounter rate and mean reported group size; this identity does not identify ecological or detection mechanisms.

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| Reported whales/100 km | 100 x matched whale count / survey km | Recorded whale-count rate per survey distance |
| Sightings/100 km | 100 x matched sighting count / survey km | Encounter rate |
| Reported whales/hour | Matched whale count / survey hours | Time-standardized recorded count |
| Reported whales/sighting | Whale count / sighting count | Mean recorded group size |

Positive Final_group values take precedence over positive Totalno values. Flights with valid effort but no sightings are retained. Rates without a valid effort denominator are missing, not zero. Cumulative observations are not counts of unique individuals or estimates of abundance.

{markdown_table(table)}

## Annual rate-cost comparisons

The following descriptive correlations use nine annual observations and archived path costs. They are distinct from spatial sighting-to-corridor metrics and do not establish causation. Overlapping five-year prey windows and observation error limit interpretation of temporal associations.

{markdown_table(pd.DataFrame(corr_rows))}

For first differences, Pearson r was {differenced.loc['prey_only','pearson_r']:.3f} for prey costs and {differenced.loc['prey_ice','pearson_r']:.3f} for prey+ice costs.

## Survey-footprint sensitivity

The fixed footprints contain 105, 52 and 19 cells. Counts and effort use the same spatial boundary. Nearest-grid partitions retain the study-domain membership rule and the 15-km observation-to-ocean-cell gate.

{markdown_table(fixed_table)}

## Survey-mode sensitivity

{markdown_table(mode_table)}

Search and circling add opportunistic or targeted activity to the systematic transect design. Group counts are missing in the transect-plus-circling series in 2014 and 2015; whale rates and their bootstrap intervals remain missing, while sighting rates can still be calculated.

## Computation and checks

- Global Positioning System (GPS) events are sorted within flights. Automatic GPS records inherit the active survey mode. A segment uses its starting mode; flights are not connected.
- WGS84 geodesic distances are clipped using projected EPSG:3338 geometry. Time is apportioned by the fraction of segment length inside the boundary, assuming constant speed within each segment.
- The primary calculation excludes gaps longer than 300 seconds, nonpositive durations, invalid coordinates and speeds above 600 km/h. Alternative gap thresholds are 60 and 600 seconds.
- Whole flights are resampled 2,000 times within year-month strata, retaining zero-sighting flights. The 2.5th and 97.5th percentiles represent sampled-flight variation, not detection-corrected uncertainty in population abundance.
- The maximum annual transect-distance discrepancy relative to flight summaries was {max_qa_difference:.3f}%. There were {diagnostics['whole_flight_summary_flags']} flight/mode discrepancies exceeding both 5% and 1 km, including {search_flags} search cases.
- Additional summaries cover July-only observations, primary observers and exclusion of flagged flight summaries.

Source: [Aerial Surveys of Arctic Marine Mammals historical database](https://www.fisheries.noaa.gov/resource/data/1979-2021-aerial-surveys-arctic-marine-mammals-historical-database). Related transect-rate analysis: [Moore et al. (2022)](https://doi.org/10.1371/journal.pone.0265934).
"""
    (out/"analysis_report.md").write_text(report)
    (out/"README.md").write_text("""# ASAMM effort-normalized observation diagnostic

Annual effort-normalized rates and descriptive comparisons with archived corridor costs.

Start with [the survey-rate methods and results](analysis_report.md).

- [Annual costs, effort, encounters and group counts](figures/figure_1_annual_effort_normalized_observations.pdf)
- [Frozen costs versus distance-normalized rates](figures/figure_2_cost_vs_effort_normalized_rates.pdf)
- [Survey-window, spatial-mask, mode and observer sensitivities](figures/figure_3_effort_rate_sensitivities.pdf)
- [Primary annual numeric results](source_data/primary_annual_comparison.csv)
- [All rate-cost comparisons](source_data/cost_rate_correlations.csv)
- [Figure captions](figure_captions.md)
- [Output and QA index](source_data_index.md)

The primary regional transect-only point estimate increases from 2017 to 2018 and decreases in 2019. Spatial footprint and inclusion of targeted circling affect the annual pattern. Flight-bootstrap intervals are broad. Correlations describe nine annual observations and do not establish causation or independently rank connectivity scenarios.
""")
    (out/"figure_captions.md").write_text("""# Survey diagnostic figure captions

**Figure 1 | Annual modeled costs and survey-effort-normalized gray whale observations.** **a**, Frozen primary benthic-energy path costs for prey-only and prey+ice scenarios, scaled to thousands of resistance-weighted kilometers. **b, c**, Reported whales and sighting records per 100 km of transect effort, respectively. **d**, Matched regional transect kilometers by month. **e**, Reported whales per sighting. Observations and effort are restricted to the same regional footprint and June–July during 2011–2019. The spatial boundary reproduces the frozen C3 and nearest-ocean-grid eligibility rule. Confirmed repeat sightings, non-transect modes and deadhead effort are excluded from b–e. Final group counts take precedence over initial counts. Ribbons in b, c and e show the 2.5th–97.5th percentiles from 2,000 whole-flight resamples within year-month strata, retaining zero-sighting flights. Gray vertical backgrounds mark 2018–2019. Annual samples contain 11–24 flights with regional transect effort. Rates describe recorded encounters, not detection-corrected abundance.

**Figure 2 | Frozen path costs compared with distance-normalized observation rates.** **a, b**, Prey-only costs; **c, d**, prey+ice costs. Left panels show reported whales per 100 transect km; right panels show sighting records per 100 transect km. Each point is one year (n=9, 2011–2019), with selected years labeled. Pearson r and Spearman rho describe annual-level association; no fitted relationship or causal interpretation is imposed. The same observation data are compared with both frozen cost series. Year-to-year differences and alternative year subsets are provided in the source-data tables.

**Figure 3 | Sensitivity of distance-normalized whale counts to survey definitions.** **a**, June–July versus July-only. **b**, Regional coverage versus the frozen sets of 105, 52 and 19 grid cells surveyed in at least seven, at least eight and all nine years. Both track kilometers and sighting numerators are restricted to the same nearest-ocean-grid partitions. **c**, Transect-only, transect plus search, and transect plus its associated circling, with matched numerator and denominator modes. Search and circling are supplementary survey-definition sensitivities because they include opportunistic or targeted activity. Gaps in the transect-plus-circling series indicate missing group counts in 2014–2015. **d**, All recorded observers versus primary observers only, retaining the same transect denominator. Lines show annual point estimates; the full bootstrap outputs are supplied separately. Gray backgrounds mark 2018–2019.
""")
    files = [
        ("source_data/primary_annual_comparison.csv", "Primary nine-year transect-only regional results"),
        ("source_data/annual_effort_normalized_rates.csv", "All windows, spatial masks and survey-mode definitions"),
        ("source_data/flight_effort_normalized_counts.csv", "Matched flight-level denominators and counts, including zero-sighting flights"),
        ("source_data/flight_bootstrap_intervals.csv", "Year-month stratified flight-bootstrap intervals"),
        ("source_data/frozen_path_costs.csv", "Unmodified prey-only and prey+ice annual costs"),
        ("source_data/cost_rate_correlations.csv", "Annual-level, 2011–2017, omitted-year and first-difference comparisons"),
        ("source_data/gps_segment_audit.csv.gz", "Consecutive GPS segments, acceptance flags, clipping fractions and modes"),
        ("source_data/gray_whale_event_audit.csv", "Source-event counts, modes, repeat flags and spatial membership"),
        ("qa/flight_summary_reconstruction.csv", "Whole-flight GPS versus archived effort-summary checks"),
        ("qa/frozen_spatial_membership_reproduction.csv", "Exact recovery of frozen regional and fixed-mask sighting membership"),
        ("qa/gps_gap_sensitivity.csv", "60- and 600-second GPS-gap thresholds versus the 300-second primary setting"),
        ("qa/primary_observer_sensitivity.csv", "Counts from primary observers with the same effort denominator"),
        ("qa/primary_observer_correlations.csv", "Primary-observer rates versus both frozen cost scenarios"),
        ("qa/flagged_flight_exclusion_sensitivity.csv", "Primary rate after omitting the flagged 2017 transect flight"),
        ("qa/metadata_definitions.json", "Original data definitions for modes, repeat flags, coordinates and group counts"),
        ("qa/final_integrity_checks.json", "Final numeric, spatial, input-integrity and unit-test checks"),
        ("qa/figure_export_checks.json", "Editable SVG text, embedded PDF fonts and PNG export checks"),
        ("input_sha256_before.json", "Protected inputs before analysis"),
        ("input_sha256_after.json", "Protected inputs after analysis"),
        ("configuration_snapshot.json", "Complete analysis settings"),
        ("run_manifest.json", "Versions, run time, integrity and QA counts"),
    ]
    (out/"source_data_index.md").write_text("# Source data and QA index\n\n"+markdown_table(pd.DataFrame(files,columns=["File","Purpose"]))+"\n")


def main(paths):
    paths.require_task("annual_rates")
    config_path = paths.package("config/ncc_v8/annual_effort.json")
    config = json.loads(config_path.read_text())
    paths.reserve_output()
    out = paths.output("results")
    for sub in ["source_data", "qa", "figures", "cache"]:
        (out / sub).mkdir(parents=True, exist_ok=False)
    protect = paths.input_glob(config["raw_root"], "*/_Output__*.csv")
    protect += paths.input_glob(config["raw_root"], "*/_SummaryInfo__*.csv")
    protect += [paths.input_path(config[k]) for k in ["metadata", "grid", "domain", "existing_effort_cells", "existing_sightings", "existing_annual_summary"]]
    protect += [paths.input_path(p) for p in config["cost_sources"].values()]
    # Manuscripts are not computational inputs. Numeric source checks remain above.
    before = {str(p.relative_to(paths.data_root)): sha256(p) for p in protect}
    (out / "input_sha256_before.json").write_text(json.dumps(before, indent=2) + "\n")
    (out / "configuration_snapshot.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Reading original flight events and summary effort", flush=True)
    events, summary, _ = read_inputs(config, paths)
    resolved = resolve_modes(events)
    print(f"Reconstructing {len(events):,} events from {events.flight_key.nunique()} flights", flush=True)
    segments = reconstruct_segments(events, max_gap_seconds=config["max_gap_seconds"],
        max_speed_kmh=config["max_speed_kmh"], projected_crs=config["projected_crs"])
    grid, footprints = make_footprints(config, paths)
    grid.to_csv(out / "source_data" / "frozen_mask_membership.csv", index=False)
    gray = resolved.loc[resolved["Species"].fillna("").str.strip().str.lower().eq("gray whale")].copy()
    for footprint, polygon in footprints.items():
        print(f"Clipping tracks and observations to {footprint}", flush=True)
        segments[f"fraction_{footprint}"] = clipped_fractions(segments, polygon)
        gray[f"in_{footprint}"] = point_membership(gray, polygon, config["projected_crs"])
    qa = reconstruction_qa(segments, summary, out)
    segments.to_csv(out / "source_data" / "gps_segment_audit.csv.gz", index=False, compression="gzip")
    gray.to_csv(out / "source_data" / "gray_whale_event_audit.csv", index=False)
    resolved.loc[resolved["mode_text_conflict"]].to_csv(out / "qa" / "mode_text_conflicts.csv", index=False)
    gray.groupby(["year", "entry_clean", "mode", "repeat_clean"], dropna=False).agg(
        sighting_records=("source_id", "size"), reported_whales=("whale_count", "sum")
    ).reset_index().to_csv(out / "qa" / "gray_whale_modes_by_year.csv", index=False)
    frozen = pd.read_csv(paths.input_path(config["existing_sightings"]))
    frozen = frozen.loc[frozen["use_for_C3_validation"].eq(True) & frozen["month"].isin([6, 7]) & frozen["year"].isin(config["years"])]
    overlap = frozen[["source_file", "ID", "year", "count_or_best_count"]].merge(
        gray[["source_file", "ID", "in_regional", "mode", "entry_clean", "repeat_clean"]].assign(ID=lambda x: pd.to_numeric(x.ID)),
        on=["source_file", "ID"], how="left", validate="one_to_one")
    overlap.to_csv(out / "qa" / "previous_sighting_selection_comparison.csv", index=False)
    if len(frozen) != 1179 or not np.isclose(frozen["count_or_best_count"].sum(), 2226):
        raise ValueError("Existing manuscript sighting selection did not reproduce")
    if not overlap["in_regional"].eq(True).all() or int(gray["in_regional"].sum()) != 1179:
        raise ValueError("Exact analysis footprint does not reproduce all 1,179 frozen sightings")
    spatial_checks = []
    for footprint in footprints:
        if footprint == "regional":
            expected = np.ones(len(frozen), dtype=bool)
        else:
            threshold = int(footprint.split("_")[-1])
            ids = set(grid.loc[grid.previous_survey_years.ge(threshold), "grid_id"])
            expected = frozen["nearest_grid_id"].isin(ids).to_numpy()
        matched = frozen[["source_file", "ID"]].merge(
            gray[["source_file", "ID", f"in_{footprint}"]].assign(ID=lambda x: pd.to_numeric(x.ID)),
            on=["source_file", "ID"], validate="one_to_one", how="left")
        differences = int((matched[f"in_{footprint}"].to_numpy() != expected).sum())
        spatial_checks.append(dict(footprint=footprint, frozen_sightings_in_footprint=int(expected.sum()), geometry_membership_mismatches=differences))
        if differences:
            raise ValueError("A fixed footprint does not reproduce the frozen sighting membership")
    pd.DataFrame(spatial_checks).to_csv(out/"qa/frozen_spatial_membership_reproduction.csv",index=False)
    annual_parts, flight_parts, interval_parts, gap_parts, observer_parts = [], [], [], [], []
    for footprint in footprints:
        for definition, modes in config["mode_sets"].items():
            flights = aggregate_flights(segments, gray, fraction_column=f"fraction_{footprint}",
                member_column=f"in_{footprint}", modes=modes, max_gap_seconds=config["max_gap_seconds"])
            if flights["unmatched_numerator"].any():
                flights.loc[flights["unmatched_numerator"]].to_csv(out / "qa" / f"unmatched_{footprint}_{definition}.csv", index=False)
                raise ValueError("Sighting numerator has no matching flight effort; inspect QA")
            for window, months in config["windows"].items():
                sub = flights.loc[flights["month"].isin(months)].copy()
                ids = dict(window=window, footprint=footprint, effort_definition=definition)
                annual_parts.append(summarize_rates(sub, config["years"]).assign(**ids))
                flight_parts.append(sub.assign(**ids))
                interval_parts.append(flight_bootstrap(sub, replicates=config["bootstrap_replicates"], seed=config["random_seed"]).assign(**ids))
            if footprint == "regional" and definition == "transect":
                for gap in config["gap_sensitivities_seconds"]:
                    g = aggregate_flights(segments, gray, fraction_column="fraction_regional", member_column="in_regional", modes=modes, max_gap_seconds=gap)
                    gap_parts.append(summarize_rates(g, config["years"]).assign(max_gap_seconds=gap))
                gray["in_regional_primary_observer"] = gray["in_regional"] & pd.to_numeric(gray["PrimObs"], errors="coerce").eq(1)
                observed = aggregate_flights(segments, gray, fraction_column="fraction_regional", member_column="in_regional_primary_observer", modes=modes, max_gap_seconds=config["max_gap_seconds"])
                for window, months in config["windows"].items():
                    observer_parts.append(summarize_rates(observed.loc[observed.month.isin(months)], config["years"]).assign(window=window, footprint="regional", effort_definition="transect", observer_subset="primary_only"))
    annual = pd.concat(annual_parts, ignore_index=True)
    flights = pd.concat(flight_parts, ignore_index=True)
    intervals = pd.concat(interval_parts, ignore_index=True)
    annual.to_csv(out / "source_data" / "annual_effort_normalized_rates.csv", index=False)
    flights.to_csv(out / "source_data" / "flight_effort_normalized_counts.csv", index=False)
    intervals.to_csv(out / "source_data" / "flight_bootstrap_intervals.csv", index=False)
    pd.concat(gap_parts, ignore_index=True).to_csv(out / "qa" / "gps_gap_sensitivity.csv", index=False)
    observers = pd.concat(observer_parts, ignore_index=True)
    observers.to_csv(out / "qa" / "primary_observer_sensitivity.csv", index=False)
    primary_flights = flights.loc[flights.window.eq("June_July") & flights.footprint.eq("regional") & flights.effort_definition.eq("transect")]
    flagged = set(qa.loc[qa["mode"].eq("transect") & qa.flag_over_5pct_and_1km, "flight_key"])
    unflagged = primary_flights.loc[~primary_flights.flight_key.isin(flagged)]
    summarize_rates(unflagged, config["years"]).to_csv(out/"qa/flagged_flight_exclusion_sensitivity.csv",index=False)
    costs = []
    for scenario, path in config["cost_sources"].items():
        c = pd.read_csv(paths.input_path(path))
        c = c.loc[c["prey_variable"].eq(config["prey_variable"]) & c["target_year"].isin(config["years"])]
        if len(c) != 9:
            raise ValueError("Frozen path-cost rows do not match the nine-year comparison")
        costs.append(c[["target_year", "mean_path_cost"]].rename(columns={"target_year": "year"}).assign(scenario=scenario))
    costs = pd.concat(costs, ignore_index=True)
    costs.to_csv(out / "source_data" / "frozen_path_costs.csv", index=False)
    correlations = make_correlations(annual, costs)
    correlations.to_csv(out / "source_data" / "cost_rate_correlations.csv", index=False)
    make_correlations(observers, costs).assign(observer_subset="primary_only").to_csv(out / "qa" / "primary_observer_correlations.csv", index=False)
    old = pd.read_csv(paths.input_path(config["existing_annual_summary"]))
    old = old.loc[old["footprint"].eq("full") & old["year"].isin(config["years"])]
    old.to_csv(out / "source_data" / "previous_count_per_cell_comparator.csv", index=False)
    root = ET.parse(paths.input_path(config["metadata"])).getroot()
    excerpts = {}
    for attr in root.iter("attr"):
        label = attr.findtext("attrlabl", "").strip()
        if label in ["Entry", "FltType", "Final_group", "Repeat", "ArcLat", "ArcLong", "PrimObs"]:
            excerpts[label] = {"definition": attr.findtext("attrdef", "").strip(),
                "values": {e.findtext("edomv", "").strip(): e.findtext("edomvd", "").strip() for e in attr.iter("edom")}}
    (out / "qa" / "metadata_definitions.json").write_text(json.dumps(excerpts, indent=2) + "\n")
    after = {str(p.relative_to(paths.data_root)): sha256(p) for p in protect}
    (out / "input_sha256_after.json").write_text(json.dumps(after, indent=2) + "\n")
    if before != after:
        raise ValueError("Protected input checksum changed")
    primary = annual.loc[annual.window.eq("June_July") & annual.footprint.eq("regional") & annual.effort_definition.eq("transect")]
    primary.to_csv(out / "source_data" / "primary_annual_comparison.csv", index=False)
    diagnostics = dict(
        analysis_id=config["analysis_id"], created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={"pandas": pd.__version__, "numpy": np.__version__, "shapely": shapely.__version__},
        protected_files=len(protect), protected_files_unchanged=before == after,
        raw_events=len(events), flights_with_events=events.flight_key.nunique(),
        gray_whale_records=len(gray), mode_text_conflicts=int(resolved.mode_text_conflict.sum()),
        invalid_base_segments=int((~segments.valid_base).sum()),
        long_gap_segments=int((segments.valid_base & segments.seconds.gt(config["max_gap_seconds"])).sum()),
        whole_flight_summary_flags=int(qa.flag_over_5pct_and_1km.sum()),
        frozen_1179_records_outside_exact_footprint=int((overlap.in_regional != True).sum()),
        missing_counts_in_primary=int(primary.missing_group_counts.sum()),
    )
    (out / "run_manifest.json").write_text(json.dumps(diagnostics, indent=2, default=int) + "\n")
    write_reports(out, annual, costs, correlations, intervals, diagnostics)
    print(primary[["year", "survey_km", "sighting_count", "reported_whales", "whales_per_100km", "sightings_per_100km", "whales_per_hour"]].to_string(index=False), flush=True)
    print("QA", json.dumps(diagnostics, default=int), flush=True)
    print("ASAMM survey-rate calculation complete.", flush=True)


if __name__ == "__main__":
    main()
