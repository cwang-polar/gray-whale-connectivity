"""study-only annual ASAMM orchestration for explicitly authorized execution."""

import importlib
import json

from .control import CONFIG
from .figure_adapters import plotting_imports


def annual_contract():
    config = json.loads((CONFIG / "annual_effort.json").read_text())
    scope = json.loads((CONFIG / "scope.json").read_text())
    assert config["years"] == list(range(2011, 2020))
    assert config["primary_window"] == "June_July" and config["windows"]["June_July"] == [6, 7]
    assert config["primary_footprint"] == "regional" and config["primary_effort"] == "transect"
    assert config["mode_sets"]["transect"] == [scope["annual_metrics"]["primary_mode"]]
    for name in ("max_gap_seconds", "max_speed_kmh"):
        assert config[name] == scope["annual_metrics"][name]
    assert config["max_grid_distance_m"] == 1000 * scope["spatial_metrics"]["mapping_gate_km"]
    return config


def available_period_correlations(main, costs, temporal):
    """Same Pearson and year-omission operations, restricted to the adopted four rows."""
    import numpy as np
    import pandas as pd
    rate_by_year = main.set_index("year").whales_per_100km
    rows, omissions = [], []
    for scenario in temporal.SCENARIOS:
        years = list(range(2011 if scenario in ("prey_only", "prey_ice") else 2015, 2020))
        c = costs.loc[costs.scenario.eq(scenario)].set_index("year").mean_path_cost.loc[years].to_numpy()
        y = rate_by_year.loc[years].to_numpy()
        r = temporal.pearson(c, y)
        lo_values = []
        period = f"{years[0]}\u20132019"
        for index, omitted_year in enumerate(years):
            keep = np.arange(len(years)) != index
            value = temporal.pearson(c[keep], y[keep])
            omissions.append(dict(period=period, scenario=scenario, omitted_year=omitted_year,
                                  n_years=len(years)-1, pearson_r=value))
            lo_values.append(value)
        rows.append(dict(scenario=scenario, period=period, n_years=len(years), pearson_r=r,
                         omit_year_min=min(lo_values), omit_year_max=max(lo_values)))
    return pd.DataFrame(rows), pd.DataFrame(omissions)


def write_annual_correlations(paths, main):
    import numpy as np
    import pandas as pd
    temporal = importlib.import_module("build_phase8v_temporal_metric_candidates_20261117")
    cost_frames = []
    for scenario, (_, _, _, relative) in temporal.SCENARIOS.items():
        d = pd.read_csv(paths.input_path(relative))
        d = d.loc[d.prey_variable.eq("total_kJm2_frozen")].copy()
        if not d.successful_path_count.eq(1600).all() or not d.failed_path_count.eq(0).all():
            raise ValueError(f"Unexpected ensemble sizes: {scenario}")
        cost_frames.append(d[["target_year", "mean_path_cost"]].rename(columns={"target_year": "year"}).assign(scenario=scenario))
    costs = pd.concat(cost_frames, ignore_index=True)
    if costs.duplicated(["scenario", "year"]).any() or len(costs) != 28:
        raise ValueError("Expected 28 unchanged primary scenario-year costs")
    band = pd.read_csv(paths.input("cost_band")).rename(columns={"target_year": "year"})
    data = main.merge(costs.loc[costs.scenario.eq("prey_ice"), ["year", "mean_path_cost"]],
                      on="year", validate="one_to_one").merge(band, on="year", validate="one_to_one")
    np.testing.assert_allclose(data.mean_path_cost, data.mean_path_cost_b_primary, rtol=0, atol=1e-9)
    assert (data.mean_path_cost_b_minimum <= data.mean_path_cost).all()
    assert (data.mean_path_cost_b_maximum >= data.mean_path_cost).all()
    summary, omissions = available_period_correlations(main, costs, temporal)
    expected = pd.read_csv(paths.input("correlations"))
    assert summary.scenario.tolist() == expected.scenario.tolist()
    assert summary.period.tolist() == expected.period.tolist()
    np.testing.assert_allclose(summary[["pearson_r", "omit_year_min", "omit_year_max"]],
                               expected[["pearson_r", "omit_year_min", "omit_year_max"]], rtol=0, atol=1e-12)
    for name, frame in (("available_period_correlations", summary), ("leave_one_year_out_correlations", omissions),
                        ("frozen_all_scenario_costs", costs), ("fig5_panel_C_annual_data", data)):
        frame.to_csv(paths.output(f"source_data/{name}.csv"), index=False)
    return {"scenario_year_cost_rows": len(costs), "available_period_rows": len(summary),
            "omission_rows": len(omissions), "correlations_match_frozen": True}


def _prepare(paths):
    before = paths.snapshot()
    paths.reserve_output()
    for folder in ("source_data", "qa"):
        paths.output(folder).mkdir(exist_ok=False)
    return before


def _finish(paths, before, checks):
    if paths.snapshot() != before:
        raise RuntimeError("Annual-analysis inputs changed")
    manifest = {"task": paths.task, "inputs": before, "checks": checks,
                "configuration": annual_contract(),
                "scope": "June-July/regional/transect; available-period Pearson and year omission only",
                "excluded": ["flight bootstrap", "stable-footprint rate variants", "July-only",
                             "alternate survey modes", "observer restriction", "gap sweeps",
                             "flagged-flight exclusion", "first differences", "Spearman correlations",
                             "common-period cross-scenario temporal diagnostic", "spatial matching"],
                "random_sampling_used": False, "corridor_models_rerun": False}
    paths.output("qa/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_annual_rates(paths):
    """Calculate study annual rates and correlations from survey-event inputs."""
    paths.require_task("annual_rates")
    config = annual_contract()
    before = _prepare(paths)
    with plotting_imports(paths):
        import numpy as np
        import pandas as pd
        legacy = importlib.import_module("run_phase8v_asamm_effort_rates")
        events, summary, _ = legacy.read_inputs(config, paths)
        resolved = legacy.resolve_modes(events)
        segments = legacy.reconstruct_segments(events, max_gap_seconds=config["max_gap_seconds"],
            max_speed_kmh=config["max_speed_kmh"], projected_crs=config["projected_crs"])
        _, footprints = legacy.make_footprints(config, paths, regional_only=True)
        gray = resolved.loc[resolved["Species"].fillna("").str.strip().str.lower().eq("gray whale")].copy()
        segments["fraction_regional"] = legacy.clipped_fractions(segments, footprints["regional"])
        gray["in_regional"] = legacy.point_membership(gray, footprints["regional"], config["projected_crs"])
        frozen = pd.read_csv(paths.input_path(config["existing_sightings"]))
        frozen = frozen.loc[frozen["use_for_C3_validation"].eq(True) & frozen["month"].isin([6, 7]) & frozen["year"].isin(config["years"])]
        overlap = frozen[["source_file", "ID", "year", "count_or_best_count"]].merge(
            gray[["source_file", "ID", "in_regional", "mode", "entry_clean", "repeat_clean"]].assign(ID=lambda x: pd.to_numeric(x.ID)),
            on=["source_file", "ID"], how="left", validate="one_to_one")
        if len(frozen) != 1179 or not np.isclose(frozen["count_or_best_count"].sum(), 2226):
            raise ValueError("Frozen regional sighting selection differs")
        if not overlap.in_regional.eq(True).all() or int(gray.in_regional.sum()) != 1179:
            raise ValueError("Regional geometry no longer reproduces frozen membership")
        flights = legacy.aggregate_flights(segments, gray, fraction_column="fraction_regional", member_column="in_regional",
                                            modes=config["mode_sets"]["transect"], max_gap_seconds=config["max_gap_seconds"])
        if flights.unmatched_numerator.any():
            raise ValueError("A sighting numerator lacks matching flight effort")
        sub = flights.loc[flights.month.isin(config["windows"]["June_July"])].copy()
        ids = dict(window="June_July", footprint="regional", effort_definition="transect")
        primary = legacy.summarize_rates(sub, config["years"], metrics=("whales_per_100km",)).assign(**ids)
        reference = pd.read_csv(paths.input("annual_reference")).sort_values("year")
        columns = ["survey_km", "sighting_count", "reported_whales", "whales_per_100km"]
        np.testing.assert_allclose(primary[columns], reference[columns], rtol=1e-12, equal_nan=True)
        primary.to_csv(paths.output("source_data/primary_annual_comparison.csv"), index=False)
        sub.assign(**ids).to_csv(paths.output("source_data/flight_effort_normalized_counts.csv"), index=False)
        segments.to_csv(paths.output("qa/gps_segment_audit.csv.gz"), index=False, compression="gzip")
        gray.to_csv(paths.output("qa/gray_whale_event_audit.csv"), index=False)
        overlap.to_csv(paths.output("qa/regional_membership_check.csv"), index=False)
        checks = write_annual_correlations(paths, primary)
        checks.update(annual_rate_rows=len(primary), rates_match_frozen=True, regional_membership_matches=True)
    _finish(paths, before, checks)


def run_annual_correlation(paths):
    """Calculate study annual correlations from archived rates and costs."""
    paths.require_task("annual_correlation")
    before = _prepare(paths)
    with plotting_imports(paths):
        import pandas as pd
        main = pd.read_csv(paths.input("annual_reference")).sort_values("year")
        checks = write_annual_correlations(paths, main)
    _finish(paths, before, checks)
