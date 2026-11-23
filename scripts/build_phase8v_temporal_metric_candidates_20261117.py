#!/usr/bin/env python3
"""Generate isolated temporal diagnostic candidates from frozen model outputs."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gray_whale_temporal_candidates_20261117")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/gray_whale_temporal_candidates_20261117")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from plot_phase8v_asamm_effort_rates import save, style

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/asamm_effort_rates_phase8v/v1/source_data"
DEFAULT_OUT = ROOT / "outputs/asamm_effort_rates_phase8v/figure_candidates_20261117_v1"
SCENARIOS = {
    "prey_only": ("Prey", "#3B6EA8", "o", "outputs/lcp_v2/prey_only_brian_C3_route_summary_v2.csv"),
    "prey_ice": ("Prey + ice", "#3A8D6D", "s", "outputs/lcp_v2/prey_ice_brian_C3_route_summary_v2.csv"),
    "prey_vessel": ("Prey + vessel", "#D9822B", "^", "outputs/prey_vessel_phase8k/lcp/prey_vessel_brian_C3_route_summary_v2.csv"),
    "prey_ice_vessel": ("Prey + ice + vessel", "#A85A91", "D", "outputs/lcp_v2/prey_ice_ship_brian_C3_route_summary_v2.csv"),
}
PERIODS = {"2011–2019": (list(range(2011, 2020)), list(SCENARIOS)[:2]),
           "2015–2019": (list(range(2015, 2020)), list(SCENARIOS))}
BAND = ROOT / "outputs/manuscript_figures_v3/figure_3/source_data/gray_whale_figure3_ncc_seasonality_candidate_v3_panel_A_total_cost.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_inputs():
    paths = [BASE / x for x in ["primary_annual_comparison.csv", "annual_effort_normalized_rates.csv",
                               "flight_effort_normalized_counts.csv", "frozen_path_costs.csv",
                               "cost_rate_correlations.csv"]]
    paths += [ROOT / x[3] for x in SCENARIOS.values()] + [BAND]
    paths += list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py"))
    paths.append(ROOT / "config/ncc_v8/frozen_figure_references.json")
    for folder in [ROOT / "outputs/asamm_effort_rates_phase8v",
                   ROOT / "outputs/manuscript_figures_v3/figure_5/candidates"]:
        paths += [p for p in folder.rglob("*") if p.is_file() and
                  p.suffix in {".png", ".pdf", ".svg"}]
    return sorted(p for p in set(paths) if p.resolve() != Path(__file__).resolve())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def replicate_correlations(x, histories):
    """Compute one correlation per matched annual encounter-rate history."""
    x = np.asarray(x, float)
    xx = x - x.mean()
    yy = histories - histories.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(xx**2) * np.sum(yy**2, axis=1))
    return np.divide(yy @ xx, denominator, out=np.full(len(histories), np.nan),
                     where=denominator > 0)


def flight_histories(flights, years, replicates, seed):
    rng = np.random.default_rng(seed)
    histories = np.full((replicates, len(years)), np.nan)
    for j, year in enumerate(years):
        annual = flights.loc[flights.year.eq(year)]
        if annual.reported_whales.isna().any():
            raise ValueError("A missing group count prevents resampling this annual whale total")
        sums = np.zeros((replicates, 2))
        for _, month in annual.groupby("month", sort=True):
            values = month[["reported_whales", "survey_km"]].to_numpy(float)
            indices = rng.integers(0, len(values), (replicates, len(values)))
            sums += values[indices].sum(axis=1)
        histories[:, j] = np.divide(100 * sums[:, 0], sums[:, 1],
                                    out=np.full(replicates, np.nan), where=sums[:, 1] > 0)
    return histories


def load_sources():
    main = pd.read_csv(BASE / "primary_annual_comparison.csv").sort_values("year")
    annual = pd.read_csv(BASE / "annual_effort_normalized_rates.csv")
    flights = pd.read_csv(BASE / "flight_effort_normalized_counts.csv")
    flights = flights.loc[flights.window.eq("June_July") & flights.footprint.eq("regional") &
                          flights.effort_definition.eq("transect")].copy()
    cost_frames = []
    for scenario, (_, _, _, path) in SCENARIOS.items():
        d = pd.read_csv(ROOT / path)
        d = d.loc[d.prey_variable.eq("total_kJm2_frozen")].copy()
        if not d.successful_path_count.eq(1600).all() or not d.failed_path_count.eq(0).all():
            raise ValueError(f"Unexpected model ensemble sizes for {scenario}")
        cost_frames.append(d[["target_year", "mean_path_cost"]].rename(columns={"target_year": "year"}).assign(scenario=scenario))
    costs = pd.concat(cost_frames, ignore_index=True)
    if costs.duplicated(["scenario", "year"]).any() or len(costs) != 28:
        raise ValueError("Expected 28 unique primary scenario-year costs")
    band = pd.read_csv(BAND).rename(columns={"target_year": "year"})
    data = main.merge(costs.loc[costs.scenario.eq("prey_ice"), ["year", "mean_path_cost"]],
                      on="year", validate="one_to_one").merge(band, on="year", validate="one_to_one")
    np.testing.assert_allclose(data.mean_path_cost, data.mean_path_cost_b_primary, rtol=0, atol=1e-9)
    assert (data.mean_path_cost_b_minimum <= data.mean_path_cost).all()
    assert (data.mean_path_cost_b_maximum >= data.mean_path_cost).all()
    reproduced = flights.groupby("year")[["reported_whales", "survey_km"]].sum()
    np.testing.assert_allclose(100 * reproduced.reported_whales / reproduced.survey_km,
                               main.set_index("year").whales_per_100km, rtol=1e-12)
    return main, annual, flights, costs, data


def analyze(main, annual, costs, histories, out):
    rows, omissions, bootstrap_rows, footprint_rows, difference_rows = [], [], [], [], []
    rate_by_year = main.set_index("year").whales_per_100km
    for period, (years, scenarios) in PERIODS.items():
        y = rate_by_year.loc[years].to_numpy()
        indexes = [int(year - 2011) for year in years]
        shared_histories = histories[:, indexes]
        replicate_values = {}
        full_values = {}
        for scenario in scenarios:
            c = costs.loc[costs.scenario.eq(scenario)].set_index("year").mean_path_cost.loc[years].to_numpy()
            r = pearson(c, y)
            full_values[scenario] = r
            lo_values = []
            for index, omitted_year in enumerate(years):
                keep = np.arange(len(years)) != index
                value = pearson(c[keep], y[keep])
                omissions.append(dict(period=period, scenario=scenario, omitted_year=omitted_year,
                                      n_years=len(years)-1, pearson_r=value))
                lo_values.append(value)
            values = replicate_correlations(c, shared_histories)
            replicate_values[scenario] = values
            valid = values[np.isfinite(values)]
            low, high = np.quantile(valid, [.025, .975])
            rows.append(dict(period=period, scenario=scenario, n_years=len(years), pearson_r=r,
                             omit_year_min=min(lo_values), omit_year_max=max(lo_values),
                             omit_year_mean=np.mean(lo_values), flight_bootstrap_mean=np.mean(valid),
                             flight_bootstrap_lower=low, flight_bootstrap_upper=high,
                             valid_flight_bootstrap_replicates=len(valid),
                             first_difference_r=pearson(np.diff(c), np.diff(y))))
            bootstrap_rows.append(pd.DataFrame({"period": period, "scenario": scenario,
                                                "replicate": np.arange(len(values)), "pearson_r": values}))
            for footprint in ["regional", "stable_7", "stable_8", "stable_9"]:
                rates = annual.loc[annual.window.eq("June_July") & annual.effort_definition.eq("transect") &
                                   annual.footprint.eq(footprint)].set_index("year").whales_per_100km.loc[years]
                footprint_rows.append(dict(period=period, scenario=scenario, footprint=footprint,
                                            n_years=len(years), pearson_r=pearson(c, rates)))
        for first, second in itertools.combinations(scenarios, 2):
            delta = replicate_values[first] - replicate_values[second]
            valid = delta[np.isfinite(delta)]
            low, high = np.quantile(valid, [.025, .975])
            difference_rows.append(dict(period=period, first=first, second=second,
                                        delta_r_first_minus_second=full_values[first]-full_values[second],
                                        paired_flight_bootstrap_lower=low, paired_flight_bootstrap_upper=high,
                                        interval_includes_zero=bool(low <= 0 <= high)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "source_data/scenario_correlation_summary.csv", index=False)
    pd.DataFrame(omissions).to_csv(out / "source_data/leave_one_year_out_correlations.csv", index=False)
    pd.concat(bootstrap_rows).to_csv(out / "source_data/flight_bootstrap_correlations.csv.gz", index=False)
    pd.DataFrame(footprint_rows).to_csv(out / "source_data/footprint_correlations.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(out / "source_data/paired_scenario_differences.csv", index=False)
    return summary, pd.DataFrame(footprint_rows), pd.DataFrame(difference_rows)


def line_candidate(data, out):
    fig, left = plt.subplots(figsize=(7.2, 3.35), layout="constrained")
    right = left.twinx()
    cost_color, whale_color = SCENARIOS["prey_ice"][1], SCENARIOS["prey_only"][1]
    left.fill_between(data.year, data.mean_path_cost_b_minimum, data.mean_path_cost_b_maximum,
                       color=cost_color, alpha=.18, linewidth=0)
    first, = left.plot(data.year, data.mean_path_cost, marker="s", color=cost_color,
                       linewidth=1.8, label="Prey + ice path cost")
    second, = right.plot(data.year, data.whales_per_100km, marker="o", color=whale_color,
                         linewidth=1.8, label="Reported whales / 100 transect km")
    left.set_ylim(0, 3200)
    left.set_yticks([0, 800, 1600, 2400, 3200])
    right.set_ylim(0, 4.5)
    left.set_xlim(2010.7, 2019.3)
    left.set_xticks(data.year)
    left.ticklabel_format(axis="x", useOffset=False)
    left.set_xlabel("Year")
    left.set_ylabel("Mean modeled path cost\n(resistance-weighted km)", color=cost_color)
    right.set_ylabel("Reported whales / 100 transect km", color=whale_color)
    left.tick_params(axis="y", colors=cost_color)
    right.tick_params(axis="y", colors=whale_color)
    left.spines["left"].set_color(cost_color)
    right.spines["left"].set_visible(False)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(whale_color)
    left.grid(axis="y", color="#E4E7E9", linewidth=.6)
    left.set_axisbelow(True)
    r = pearson(data.mean_path_cost, data.whales_per_100km)
    left.text(.025, .97, f"Pearson r = {r:.2f} | 9 annual estimates", transform=left.transAxes,
               fontsize=7.5, va="top")
    left.legend(handles=[first, second], loc="upper center", bbox_to_anchor=(.5, 1.20),
                ncol=2, fontsize=7.5)
    assert left.get_ylim()[1] > data.mean_path_cost_b_maximum.max()
    save(fig, out, "candidate_1_prey_ice_lines_with_sensitivity_and_r")


def correlation_panel(ax, data, title, letter, low_column, high_column):
    scenarios = data.scenario.tolist()
    positions = np.arange(len(scenarios))[::-1]
    for (_, row), y in zip(data.iterrows(), positions):
        label, color, marker, _ = SCENARIOS[row.scenario]
        low, high = row[low_column], row[high_column]
        ax.hlines(y, low, high, color=color, linewidth=1.4)
        ax.vlines([low, high], y-.07, y+.07, color=color, linewidth=1)
        ax.plot(row.pearson_r, y, marker=marker, color=color, markersize=5,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        ax.annotate(f"{row.pearson_r:.3f}", (row.pearson_r, y), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_yticks(positions, [SCENARIOS[x][0] for x in scenarios])
    ax.set_ylim(-.65, len(scenarios)-.35)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.axvline(0, color="#AAB1B5", linestyle="--", linewidth=.7)
    ax.grid(axis="x", color="#EDF0F1", linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8.5, pad=11)
    ax.text(-.05, 1.11, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.set_xlabel("Pearson r\nMore negative = stronger inverse correspondence", fontsize=7)


def scenario_candidate(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
    for ax, (period, (years, _)), letter in zip(axes, PERIODS.items(), "ab"):
        d = summary.loc[summary.period.eq(period)]
        correlation_panel(ax, d, f"{period} | {len(years)} annual estimates", letter,
                           "omit_year_min", "omit_year_max")
    save(fig, out, "candidate_2_scenarios_with_year_omission_ranges")


def sensitivity_candidate(summary, footprints, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("correlation", ["#356EA0", "#F7F7F5", "#C4833D"])
    for row, (period, (years, scenarios)) in enumerate(PERIODS.items()):
        d = summary.loc[summary.period.eq(period)]
        correlation_panel(axes[row, 0], d, f"{period} | flight bootstrap", "ac"[row],
                           "flight_bootstrap_lower", "flight_bootstrap_upper")
        matrix = footprints.loc[footprints.period.eq(period)].pivot(index="scenario", columns="footprint", values="pearson_r")
        matrix = matrix.loc[scenarios, ["regional", "stable_7", "stable_8", "stable_9"]]
        ax = axes[row, 1]
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(matrix)):
            for j in range(4):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(value) > .7 else "#26343B")
        ax.set_xticks(range(4), ["Regional", "105 cells", "52 cells", "19 cells"], fontsize=6.5)
        ax.set_yticks(range(len(scenarios)), [SCENARIOS[x][0] for x in scenarios], fontsize=7)
        ax.set_title(f"{period} | survey-footprint sensitivity", loc="left", fontsize=8, pad=11)
        ax.text(-.05, 1.11, "bd"[row], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=axes[:, 1], label="Pearson r", shrink=.65, fraction=.025)
    save(fig, out, "candidate_3_sampling_and_footprint_sensitivity")


def write_notes(out, summary, differences, replicates, seed):
    captions = f"""# Candidate captions

## Candidate 1: two annual lines
Annual mean primary May–July prey+ice path cost (green squares, left axis) and
June–July reported whales per 100 km of matched pure-transect effort (blue
circles, right axis), 2011–2019. Each cost averages the fixed 1,600-path regional
ensemble. The green band exactly reproduces the Fig. 3A envelope across paired
sea-ice concentration thresholds 10%/45%, 15%/55% (primary), and 20%/65%. This
is a sensitivity range, not a confidence interval. Pearson r is calculated
from the nine primary annual cost–encounter pairs. The crossing of lines on
different axes has no quantitative meaning. Encounter rates are not population
abundance or detection-corrected density. Existing whale sampling intervals
remain available in the earlier effort-rate figures and source tables.

## Candidate 2: proposed temporal diagnostic
Primary annual model-cost correlations with the same regional pure-transect
whale encounter index for each scenario. A: Prey and prey+ice, 2011–2019,
n=9 annual estimates. B: all four scenarios, 2015–2019, n=5. Symbols show
full-period Pearson r. Horizontal ranges span correlations obtained after
omitting one year at a time (n=8 and n=4, respectively); these are sensitivity
ranges, not confidence intervals. Primary amphipod-energy surfaces, sea-ice
settings and vessel penalty strength alpha=0.50 are retained. Negative r is
the hypothesized lower-cost/higher-encounter direction. Single-year values
cannot be averaged to obtain a correlation. Omission and bootstrap means are
provided in source tables only; symbols do not represent those means.

## Candidate 3: additional diagnostics
A/C: same full-period symbols with 95% flight-bootstrap percentile intervals
from {replicates:,} whole-flight resamples within each year-month (seed {seed}).
Years, modeled costs, footprint and mode are fixed. All scenarios use the same
resampled annual histories, allowing paired differences. These intervals reflect
survey-flight sampling variation conditional on the observed years; they do
not account for all detection, ecological or model uncertainty. B/D: Pearson r
using the existing regional and 105-, 52-, and 19-cell survey masks. Fixed masks
are defined by coverage in at least seven, eight or all nine years of 2011–2019;
they are not newly selected for the five-year comparison and do not enforce equal
effort within cells. Distinct uncertainty and sensitivity sources are not pooled.
"""
    (out / "figure_captions.md").write_text(captions)
    result_table = summary[["period", "scenario", "pearson_r", "omit_year_min", "omit_year_max",
                            "flight_bootstrap_mean", "flight_bootstrap_lower", "flight_bootstrap_upper",
                            "first_difference_r"]].round(4).to_csv(index=False)
    notes = """# Read this first

Annual rate-cost correlations describe temporal association, separately from spatial observed-use metrics.

The comparison asks whether lower annual modeled cost corresponds to a higher
regional survey encounter index. It complements spatial observed-use metrics
but uses the same underlying observations, not new independent data. No model
or survey reconstruction was rerun. The nine- and five-year coefficients must
not be ranked against each other as if their time windows were identical.

The full-year symbol is the appropriate primary estimate. Bootstrap means are
reported as resampling summaries, not substitutes for the original estimate.
Correlations are scale invariant, so this diagnostic measures temporal shape,
not absolute cost calibration or the accuracy of predicting whale numbers.

The common five-year comparison has very little annual replication. Broad
year-omission and sampling ranges, differences in fixed-footprint results, and
first-difference correlations limit interpretation as a formal fourth score.
Paired bootstrap differences are in source_data/paired_scenario_differences.csv;
overlap of two marginal intervals is not itself a test of their difference.

## Results

```csv
""" + result_table + "```\n"
    notes += "\n## Reproduction\n\nNew script: `scripts/build_phase8v_temporal_metric_candidates_20261117.py`.\n"
    notes += "Use `--output-dir` with a new destination; overwriting is blocked.\n"
    notes += "The input-hash manifest covers prior scripts, source modules, review Word documents, reused data and prior figures.\n"
    (out / "README.md").write_text(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs/asamm_effort_rates_phase8v"):
        raise ValueError("Use an isolated Phase 8V output directory")
    if out.exists():
        raise FileExistsError("Existing outputs are protected; choose a new directory")
    paths = protected_inputs()
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    main_data, annual, flights, costs, data = load_sources()
    years = main_data.year.tolist()
    assert years == list(range(2011, 2020))
    histories = flight_histories(flights, years, args.replicates, args.seed)
    # Independent numerical checks catch axis mistakes in the vectorized estimator.
    x = np.arange(5.)
    np.testing.assert_allclose(replicate_correlations(x, np.array([x, -x, x*3+7])), [1, -1, 1], atol=1e-12)
    assert np.isnan(pearson(x, np.ones(5)))
    (out / "source_data").mkdir(parents=True)
    (out / "figures").mkdir()
    data.to_csv(out / "source_data/line_candidate_annual_data.csv", index=False)
    costs.to_csv(out / "source_data/frozen_all_scenario_costs.csv", index=False)
    pd.DataFrame(histories, columns=years).to_csv(out / "source_data/annual_flight_bootstrap_histories.csv.gz", index=False)
    summary, footprints, differences = analyze(main_data, annual, costs, histories, out)
    frozen = pd.read_csv(BASE / "cost_rate_correlations.csv")
    old = frozen.loc[frozen.window.eq("June_July") & frozen.footprint.eq("regional") &
                     frozen.effort_definition.eq("transect") & frozen.metric.eq("whales_per_100km") &
                     frozen.comparison.eq("2011_2019")].set_index("scenario").pearson_r
    np.testing.assert_allclose(summary.loc[summary.period.eq("2011–2019")].set_index("scenario").pearson_r.loc[old.index], old, atol=1e-12)
    check = replicate_correlations(costs.loc[costs.scenario.eq("prey_ice")].sort_values("year").mean_path_cost, histories[:20])
    direct = [pearson(data.mean_path_cost, y) for y in histories[:20]]
    np.testing.assert_allclose(check, direct, atol=1e-12)
    style()
    line_candidate(data, out)
    scenario_candidate(summary, out)
    sensitivity_candidate(summary, footprints, out)
    write_notes(out, summary, differences, args.replicates, args.seed)
    after = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    checks = {"protected_inputs_unchanged": before == after,
              "frozen_primary_correlations_reproduced": True,
              "flight_totals_reproduce_annual_index": True,
              "vectorized_correlations_match_direct_estimator": True,
              "fig3_primary_costs_and_band_reproduced": True,
              "all_primary_bootstrap_replicates_valid": bool(summary.valid_flight_bootstrap_replicates.eq(args.replicates).all()),
              "editable_svg_text": all("<text" in p.read_text() for p in (out / "figures").glob("*.svg")),
              "embedded_pdf_fonts": all(b"/FontFile2" in p.read_bytes() for p in (out / "figures").glob("*.pdf"))}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "backend": "Python/matplotlib",
                "replicates": args.replicates, "seed": args.seed,
                "new_candidate_script_sha256": sha(Path(__file__).resolve()),
                "protected_input_count": len(before), "protected_input_sha256_before": before,
                "protected_input_sha256_after": after, "checks": checks}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(checks.values()):
        raise RuntimeError(f"QA failure: {checks}")
    print(summary[["period", "scenario", "pearson_r", "omit_year_min", "omit_year_max",
                   "flight_bootstrap_lower", "flight_bootstrap_upper"]].round(4).to_string(index=False))
    print(json.dumps({"out": str(out), "protected_input_count": len(before), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
