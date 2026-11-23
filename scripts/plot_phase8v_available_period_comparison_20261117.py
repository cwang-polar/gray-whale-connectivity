#!/usr/bin/env python3
"""Plot available-period summaries beside matched-period temporal diagnostics."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import build_phase8v_temporal_metric_candidates_20261117 as previous
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

ROOT = previous.ROOT
SOURCE = ROOT / "outputs/asamm_effort_rates_phase8v/figure_candidates_20261117_v1"
DEFAULT_OUT = ROOT / "outputs/asamm_effort_rates_phase8v/available_period_comparison_20261117_v1"


def available_view(selected, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.65), layout="constrained")
    previous.correlation_panel(ax, selected, "Annual correlations over each scenario's available years", "",
                               "omit_year_min", "omit_year_max")
    labels = [f"{previous.SCENARIOS[row.scenario][0]}\n{row.period} (n = {row.n_years})"
              for row in selected.itertuples()]
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(1.5, color="#D5DBDF", linewidth=.7, linestyle=(0, (3, 3)))
    previous.save(fig, out, "available_period_scenario_correlations")


def paired_view(selected, common, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    positions = np.arange(len(selected))[::-1]
    for y, row in zip(positions, selected.itertuples()):
        base = common.loc[common.scenario.eq(row.scenario)].iloc[0]
        _, color, marker, _ = previous.SCENARIOS[row.scenario]
        for offset, value, low, high, ink, filled in [
            (.14, base.pearson_r, base.omit_year_min, base.omit_year_max, "#9CA6AD", False),
            (-.14, row.pearson_r, row.omit_year_min, row.omit_year_max, color, True),
        ]:
            ax.hlines(y+offset, low, high, color=ink, linewidth=1.3)
            ax.vlines([low, high], y+offset-.045, y+offset+.045, color=ink, linewidth=.8)
            ax.plot(value, y+offset, marker=marker, markersize=4.6, markeredgewidth=1.2,
                    color=ink, markerfacecolor=ink if filled else "white", zorder=3)
    ax.set_yticks(positions, [f"{previous.SCENARIOS[row.scenario][0]}\navailable: {row.period}"
                             for row in selected.itertuples()], fontsize=7)
    ax.set_ylim(-.55, 3.55)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.axvline(0, color="#AAB1B5", linestyle="--", linewidth=.7)
    ax.grid(axis="x", color="#EDF0F1", linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Pearson r\nMore negative = stronger inverse correspondence", fontsize=7)
    handles = [Line2D([], [], marker="o", markerfacecolor="white", color="#9CA6AD", linewidth=1.2,
                      label="Common 2015–2019 for all scenarios"),
               Line2D([], [], marker="o", markerfacecolor="#445761", color="#445761", linewidth=1.2,
                      label="Each scenario's available period")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.15), fontsize=7, ncol=1)
    previous.save(fig, out, "common_vs_available_period_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs/asamm_effort_rates_phase8v") or out.exists():
        raise ValueError("Choose a new isolated Phase 8V output directory")
    paths = set(previous.protected_inputs())
    paths.add(Path(previous.__file__).resolve())
    paths.discard(Path(__file__).resolve())
    paths.update([SOURCE / "source_data/scenario_correlation_summary.csv",
                  SOURCE / "source_data/frozen_all_scenario_costs.csv"])
    before = {str(p.relative_to(ROOT)): previous.sha(p) for p in sorted(paths)}
    summary = pd.read_csv(SOURCE / "source_data/scenario_correlation_summary.csv")
    selected = pd.concat([summary.loc[summary.period.eq("2011–2019")],
                          summary.loc[summary.period.eq("2015–2019") &
                                      summary.scenario.isin(["prey_vessel", "prey_ice_vessel"])]], ignore_index=True)
    common = summary.loc[summary.period.eq("2015–2019")].copy()
    costs = pd.read_csv(SOURCE / "source_data/frozen_all_scenario_costs.csv")
    rates = pd.read_csv(previous.BASE / "primary_annual_comparison.csv").set_index("year").whales_per_100km
    for row in pd.concat([selected, common]).itertuples():
        years = list(range(2011 if row.n_years == 9 else 2015, 2020))
        x = costs.loc[costs.scenario.eq(row.scenario)].set_index("year").mean_path_cost.loc[years].to_numpy()
        y = rates.loc[years].to_numpy()
        r = previous.pearson(x, y)
        omitted = [previous.pearson(np.delete(x, j), np.delete(y, j)) for j in range(len(years))]
        np.testing.assert_allclose([r, min(omitted), max(omitted)],
                                   [row.pearson_r, row.omit_year_min, row.omit_year_max], atol=1e-12)
    comparison = common[["scenario", "pearson_r", "omit_year_min", "omit_year_max"]].merge(
        selected[["scenario", "period", "n_years", "pearson_r", "omit_year_min", "omit_year_max"]],
        on="scenario", suffixes=("_common", "_available"), validate="one_to_one")
    (out / "figures").mkdir(parents=True)
    (out / "source_data").mkdir()
    selected.to_csv(out / "source_data/available_period_summary.csv", index=False)
    comparison.to_csv(out / "source_data/common_vs_available_periods.csv", index=False)
    previous.style()
    available_view(selected, out)
    paired_view(selected, common, out)
    captions = """# Available-period candidate captions

## available_period_scenario_correlations
Pearson correlations between annual primary modeled path cost and June–July
reported whales per 100 matched pure-transect kilometers. Prey and prey+ice use
2011–2019 (nine annual pairs); prey+vessel and prey+ice+vessel use 2015–2019
(five annual pairs). Symbols show full-period r. Horizontal ranges show the
minimum and maximum after omitting one year at a time: eight retained pairs
for the nine-year series and four retained pairs for the five-year series.
They are sensitivity ranges, not confidence intervals. The separator distinguishes
the two observation periods. These are available-period summaries, not controlled
same-year scenario comparisons. All inputs and previous figures are unchanged.

## common_vs_available_period_comparison
Hollow gray symbols and ranges use 2015–2019 for every scenario. Filled colored
symbols use the scenario-specific periods indicated at left. Their ranges use
the same leave-one-year-out definition. Vessel-scenario values are identical
between the two displays because their periods have not changed. Changes in
Prey and prey+ice values result solely from the time window; no model was rerun.
"""
    (out / "figure_captions.md").write_text(captions)
    (out / "README.md").write_text("""# Available-period sensitivity preview

The proposed view preserves each scenario's full available period. It is valid
for describing each scenario over that period, but differences between rows can
reflect both the model scenario and the sampled years. Matching the years is
necessary to attribute a score difference to adding vessel exposure, because
the environmental state, survey coverage and whale observations vary over time.
The existing Fig. 5A already retains the complete nine-year Prey/prey+ice analysis;
Fig. 5B provides the four-scenario comparison on the same five years and 666
observations. This retains longer coverage while preserving fair paired contrasts.

Original Phase 8K rationale: docs/phase_memos/phase8k_prey_vessel_factorial_diagnostic.md.
The paired plot illustrates why correlations from different periods should not
be treated as a single four-scenario ranking. Model parameters and observations
are not resampled or altered in this visualization. Original files remain read-only.

New runner: scripts/plot_phase8v_available_period_comparison_20261117.py.
Use --output-dir with a new destination; existing destinations are protected.
""")
    after = {str(p.relative_to(ROOT)): previous.sha(p) for p in sorted(paths)}
    checks = {"protected_inputs_unchanged": before == after,
              "coefficients_and_omission_ranges_reproduced": True,
              "four_unique_available_period_rows": len(selected) == 4 and selected.scenario.nunique() == 4,
              "vessel_rows_unchanged": bool(np.allclose(comparison.iloc[2:].pearson_r_common,
                                                         comparison.iloc[2:].pearson_r_available)),
              "editable_svg_text": all("<text" in p.read_text() for p in (out / "figures").glob("*.svg")),
              "embedded_pdf_fonts": all(b"/FontFile2" in p.read_bytes() for p in (out / "figures").glob("*.pdf"))}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "backend": "Python/matplotlib",
                "new_script_sha256": previous.sha(Path(__file__).resolve()),
                "protected_input_count": len(before), "protected_input_sha256_before": before,
                "protected_input_sha256_after": after, "checks": checks}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(checks.values()):
        raise RuntimeError(checks)
    print(comparison.round(4).to_string(index=False))
    print(json.dumps({"out": str(out), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
