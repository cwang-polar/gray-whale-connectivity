"""Prepared study spatial matching and survey-year entries; no scientific imports here.

The original Phase8T metric and omission functions are reused unchanged.
Only an explicit configured run command invokes these functions.
"""

from contextlib import contextmanager
import importlib
import json

from .archived_figures import bound_attributes
from .control import CONFIG, digest
from .figure_adapters import plotting_imports


LEGACY_MODULE = "run_phase8t_primary_robustness_smoke_test"
METRICS = ("effort_footprint_percentile", "top5_overlap_percent", "distance_to_frequent_corridor_km")
RECORD_FILE = "p4_asamm_record_metrics.csv.gz"
SUMMARY_FILE = "p4_asamm_pooled_and_leave_one_year_out.csv"
RANGE_FILE = "p4_asamm_year_leverage_ranges.csv"


def spatial_contract():
    scope = json.loads((CONFIG / "scope.json").read_text())
    spatial = scope["spatial_metrics"]
    assert spatial["early_observation_months"] == [6, 7]
    assert spatial["early_sightings_full_period"] == 1179
    assert spatial["early_sightings_common_period"] == 666
    assert spatial["percentile_background"] == "same-year strict effort cells"
    assert spatial["core_target"] == "regional annual Fs>=q95"
    assert spatial["distance_target"] == "nearest regional Fs>0.05 cell"
    assert spatial["mapping_gate_km"] == 15
    assert scope["primary"]["prey_variable"] == "total_kJm2_frozen"
    return spatial


def definitions(legacy):
    spatial_contract()
    assert legacy.YEARS == list(range(2011, 2020))
    assert legacy.VESSEL_YEARS == list(range(2015, 2020))
    assert legacy.PREY_VARIABLE == "total_kJm2_frozen"
    assert legacy.FREQUENT_THRESHOLD == 0.05
    return [
        ("full_2011_2019", legacy.YEARS, ["prey_only", "prey_ice"]),
        ("common_2015_2019", legacy.VESSEL_YEARS, ["prey_only", "prey_ice", "prey_vessel", "prey_ice_vessel"]),
    ]


@contextmanager
def spatial_bindings(paths, legacy):
    """Replace only I/O dependencies of the selected, unchanged legacy functions."""
    paths.require_reserved()
    allowed = {(scenario, year) for _, years, scenarios in definitions(legacy)
               for scenario in scenarios for year in years}

    def load_grid(scenario, year):
        if (scenario, year) not in allowed:
            raise ValueError(f"Not a V8 spatial scenario-year: {scenario}, {year}")
        frame = legacy.pd.read_csv(paths.input(f"grid_{scenario}_{year}"))
        frame["grid_id"] = frame["grid_id"].astype(str)
        return frame

    def record_stage(stage, state, **details):
        if stage != "p4":
            raise ValueError("Only the adopted spatial survey-year branch may report status")
        paths.output("qa/spatial_stage.json").write_text(json.dumps(
            {"task": paths.task, "stage": stage, "state": state, **details}, indent=2), encoding="utf-8")

    with bound_attributes(legacy, {
        "ASAMM_SIGHTINGS": paths.input("sightings"),
        "ASAMM_EFFORT": paths.input("effort"),
        "VALIDATION_OUT": paths.output("source_data"),
        "load_primary_grid": load_grid,
        "set_stage": record_stage,
    }):
        yield


def load_observations(legacy):
    """Same selection statements as the beginning of legacy p4_year_leverage."""
    pd, YEARS = legacy.pd, legacy.YEARS
    ASAMM_SIGHTINGS, ASAMM_EFFORT = legacy.ASAMM_SIGHTINGS, legacy.ASAMM_EFFORT
    sightings = pd.read_csv(ASAMM_SIGHTINGS)
    sightings = sightings.loc[
        sightings["use_for_C3_validation"].astype(bool)
        & sightings["year"].isin(YEARS)
        & sightings["month"].isin([6, 7])
    ].copy()
    sightings["nearest_grid_id"] = sightings["nearest_grid_id"].astype(str)
    effort = pd.read_csv(ASAMM_EFFORT)
    effort = effort.loc[
        effort["year"].isin(YEARS)
        & effort["validation_window"].eq("June_July")
        & effort["effort_background_type"].eq("strict_effort")
    ].copy()
    effort["grid_id"] = effort["grid_id"].astype(str)
    return sightings, effort


def check_records(paths, records, legacy):
    """Future row/key and numerical verification against the archived computation."""
    expected = legacy.pd.read_csv(paths.input("record_reference"))
    keys = ["analysis_period", "scenario", "record_id"]
    if records.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate spatial observation keys")
    if len(records) != 5022:
        raise ValueError("Expected the unchanged six pooled sample blocks (5022 rows)")
    legacy.pd.testing.assert_frame_equal(records.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-12, atol=1e-12)
    for period, _, scenarios in definitions(legacy):
        sub = records.loc[records.analysis_period.eq(period)]
        expected_count = 1179 if period == "full_2011_2019" else 666
        ids = [set(sub.loc[sub.scenario.eq(scenario), "record_id"]) for scenario in scenarios]
        if any(len(values) != expected_count or values != ids[0] for values in ids):
            raise ValueError("Scenario sample identity differs from V8")


def check_primary(paths, summary, legacy):
    expected = legacy.pd.read_csv(paths.input("summary_reference"))
    expected = expected.loc[expected.summary_type.eq("primary_all_years")].reset_index(drop=True)
    if len(summary) != 6:
        raise ValueError("Expected six primary period/scenario summaries")
    legacy.pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected,
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    figure = legacy.pd.read_csv(paths.input("figure_reference"))
    for row in summary.itertuples(index=False):
        reference = figure.loc[figure.analysis_period.eq(row.analysis_period) & figure.scenario.eq(row.scenario)]
        if len(reference) != 1 or int(reference.record_count.iloc[0]) != row.record_count:
            raise ValueError("Frozen Fig5 period/scenario identity differs")
        for metric in METRICS:
            legacy.np.testing.assert_allclose(getattr(row, metric), reference[f"{metric}_primary"].iloc[0],
                                             rtol=1e-12, atol=1e-12)


def prepare(paths, task):
    paths.require_task(task)
    spatial_contract()
    before = paths.snapshot()
    paths.reserve_output()
    for folder in ("source_data", "qa"):
        paths.output(folder).mkdir(exist_ok=False)
    return before


def finish(paths, before, legacy, checks):
    if paths.snapshot() != before:
        raise RuntimeError("A frozen spatial input changed")
    module = paths.package("scripts/" + LEGACY_MODULE + ".py")
    paths.output("qa/run_manifest.json").write_text(json.dumps({
        "task": paths.task, "inputs": before, "checks": checks,
        "implementation": paths.label(module), "implementation_sha256": digest(module),
        "contract": spatial_contract(),
        "periods": [{"id": p, "years": y, "scenarios": s} for p, y, s in definitions(legacy)],
        "layer": "Processed ASAMM and archived primary corridor surfaces; not raw-data reproduction",
        "scope": "Unweighted spatial sighting metrics, not annual whale-per-distance correlation",
        "corridor_models_rerun": False, "raw_sightings_remapped": False,
        "parameters_changed": False, "random_sampling_used": False,
        "backgrounds": "same-year strict effort percentile; regional top-5% and regional Fs>0.05 distance",
        "output_names": [p.name for p in sorted(paths.output("source_data").glob("*"))],
    }, indent=2), encoding="utf-8")


def run_spatial_matching(paths):
    """Extract and pool the three spatial correspondence metrics."""
    before = prepare(paths, "spatial_matching")
    with plotting_imports(paths):
        legacy = importlib.import_module(LEGACY_MODULE)
        with spatial_bindings(paths, legacy):
            sightings, effort = load_observations(legacy)
            frames, summaries = [], []
            for period, years, scenarios in definitions(legacy):
                for scenario in scenarios:
                    records = legacy.scenario_record_metrics(scenario, years, sightings, effort)
                    records.insert(0, "analysis_period", period)
                    frames.append(records)
                    summaries.append({"analysis_period": period, "scenario": scenario, "omitted_year": "none",
                                      "summary_type": "primary_all_years", **legacy.summarize_records(records)})
            records = legacy.pd.concat(frames, ignore_index=True)
            summary = legacy.pd.DataFrame(summaries)
            check_records(paths, records, legacy)
            check_primary(paths, summary, legacy)
            records.to_csv(paths.output("source_data/" + RECORD_FILE), index=False, compression="gzip")
            summary.to_csv(paths.output("source_data/spatial_primary_summary.csv"), index=False)
        finish(paths, before, legacy, {"records_match_archive": True, "primary_matches_archive_and_fig5": True,
                                      "record_rows": len(records), "primary_rows": len(summary)})


def run_survey_year_sensitivity(paths):
    """Calculate survey-year omission sensitivity using the original metric functions."""
    before = prepare(paths, "sensitivity_survey_year")
    with plotting_imports(paths):
        legacy = importlib.import_module(LEGACY_MODULE)
        with spatial_bindings(paths, legacy):
            legacy.p4_year_leverage()
            records = legacy.pd.read_csv(paths.output("source_data/" + RECORD_FILE))
            summary = legacy.pd.read_csv(paths.output("source_data/" + SUMMARY_FILE))
            ranges = legacy.pd.read_csv(paths.output("source_data/" + RANGE_FILE))
            check_records(paths, records, legacy)
            check_primary(paths, summary.loc[summary.summary_type.eq("primary_all_years")], legacy)
            for actual, alias in ((summary, "summary_reference"), (ranges, "range_reference"), (ranges, "figure_reference")):
                expected = legacy.pd.read_csv(paths.input(alias))
                legacy.pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                    check_exact=False, rtol=1e-12, atol=1e-12)
            if len(summary) != 44 or len(ranges) != 6:
                raise ValueError("Expected six primary summaries, 38 omissions and six ranges")
        finish(paths, before, legacy, {"records_match_archive": True, "primary_matches_archive_and_fig5": True,
            "omissions_match_archive": True, "ranges_match_archive_and_fig5": True,
            "record_rows": len(records), "summary_rows": len(summary), "range_rows": len(ranges)})
