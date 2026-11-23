#!/usr/bin/env python3
"""Plot the isolated Phase 8V effort-rate analysis using matplotlib only."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/asamm_effort_rates_phase8v/v1"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUT / "cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUT / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"prey_only": "#3B6EA8", "prey_ice": "#3A8D6D"}
LABELS = {"prey_only": "Prey only", "prey_ice": "Prey + ice"}
YEARS = np.arange(2011, 2020)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
    })


def panel(ax, letter, title):
    ax.text(-.12, 1.045, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="y", color="#E4E7E9", lw=.6)
    ax.set_axisbelow(True)


def years_axis(ax, *, shade=True):
    ax.set_xlim(2010.6, 2019.4)
    ax.set_xticks(YEARS[::2])
    ax.set_xticks(YEARS, minor=True)
    ax.ticklabel_format(axis="x", useOffset=False)
    if shade:
        ax.axvspan(2017.5, 2019.4, color="#F1F2F3", zorder=0)


def save(fig, out, stem):
    fig.canvas.draw()
    for ax in fig.axes:
        ymin, ymax = ax.get_ylim()
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if len(y) and (y.min() < ymin-1e-8 or y.max() > ymax+1e-8):
                raise ValueError(f"Data outside visible y range in {stem}")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out / "figures" / f"{stem}.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_interval(ax, data, intervals, metric, color="#4F5C64"):
    d = data.sort_values("year").merge(intervals.loc[intervals.metric.eq(metric), ["year", "lower", "upper"]], on="year", validate="one_to_one")
    ax.fill_between(d.year, d.lower, d.upper, color=color, alpha=.16, lw=0)
    ax.plot(d.year, d[metric], color=color, marker="o")
    ax.set_ylim(bottom=0)


def overview(out, annual, costs, intervals, flights):
    selected = annual.loc[annual.window.eq("June_July") & annual.footprint.eq("regional") & annual.effort_definition.eq("transect")]
    ci = intervals.loc[intervals.window.eq("June_July") & intervals.footprint.eq("regional") & intervals.effort_definition.eq("transect")]
    fig = plt.figure(figsize=(7.2, 7.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[.9, 1, 1], hspace=.10, wspace=.08)
    a = fig.add_subplot(gs[0, :])
    for scenario in ["prey_only", "prey_ice"]:
        d = costs.loc[costs.scenario.eq(scenario)].sort_values("year")
        a.plot(d.year, d.mean_path_cost/1000, color=COLORS[scenario], marker="o" if scenario=="prey_only" else "s", label=LABELS[scenario])
    a.set_ylabel("Mean path cost\n(thousand resistance-weighted km)")
    a.legend(loc="upper right", ncol=2)
    panel(a, "a", "Frozen annual connectivity costs")
    axes = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]
    for ax, letter, title, metric, ylabel in [
        (axes[0], "b", "Whales per survey distance", "whales_per_100km", "Reported whales / 100 km"),
        (axes[1], "c", "Encounters per survey distance", "sightings_per_100km", "Sightings / 100 km"),
        (axes[3], "e", "Whales per sighting", "mean_group_size", "Mean reported group size"),
    ]:
        add_interval(ax, selected, ci, metric)
        ax.set_ylabel(ylabel)
        panel(ax, letter, title)
    d = flights.loc[flights.window.eq("June_July") & flights.footprint.eq("regional") & flights.effort_definition.eq("transect")]
    monthly = d.groupby(["year","month"]).survey_km.sum().unstack(fill_value=0).reindex(YEARS, fill_value=0)
    june = monthly.get(6, pd.Series(0,index=YEARS))/1000
    july = monthly.get(7, pd.Series(0,index=YEARS))/1000
    axes[2].bar(YEARS, july, width=.65, color="#84939C", label="July")
    axes[2].bar(YEARS, june, bottom=july, width=.65, color="#CAD2D7", label="June")
    axes[2].set_ylabel("Transect effort (thousand km)")
    axes[2].legend(loc="upper right")
    panel(axes[2], "d", "Matched survey distance")
    for ax in [a]+axes:
        years_axis(ax)
    for ax in axes[2:]:
        ax.set_xlabel("Year")
    save(fig, out, "figure_1_annual_effort_normalized_observations")


def cost_scatter(out, annual, costs, correlations):
    selected = annual.loc[annual.window.eq("June_July") & annual.footprint.eq("regional") & annual.effort_definition.eq("transect")]
    stats = correlations.loc[correlations.window.eq("June_July") & correlations.footprint.eq("regional") & correlations.effort_definition.eq("transect") & correlations.comparison.eq("2011_2019")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharey="col", layout="constrained")
    letter = iter("abcd")
    for i, scenario in enumerate(["prey_only", "prey_ice"]):
        data = selected.merge(costs.loc[costs.scenario.eq(scenario)], on="year")
        for j, (metric, label) in enumerate([("whales_per_100km", "Reported whales / 100 km"), ("sightings_per_100km", "Sightings / 100 km")]):
            ax = axes[i,j]
            ax.scatter(data.mean_path_cost/1000, data[metric], color=COLORS[scenario], s=28, edgecolor="white", linewidth=.4, zorder=3)
            for _, row in data.loc[data.year.isin([2011,2017,2018,2019])].iterrows():
                dx = -24 if row.year == 2019 else 4
                dy = -13 if row.year in [2017,2019] else 5
                ax.annotate(str(int(row.year)), (row.mean_path_cost/1000,row[metric]), xytext=(dx,dy), textcoords="offset points", fontsize=6.5)
            s = stats.loc[stats.scenario.eq(scenario)&stats.metric.eq(metric)].iloc[0]
            ax.text(.97,.97,f"n = 9 years\nr = {s.pearson_r:.2f}\nSpearman ρ = {s.spearman_rho:.2f}",ha="right",va="top",transform=ax.transAxes,fontsize=7)
            ax.set_ylabel(label)
            ax.set_xlabel("Mean path cost\n(thousand resistance-weighted km)")
            ax.margins(x=.18, y=.22)
            ax.set_ylim(bottom=0)
            panel(ax, next(letter), LABELS[scenario])
    save(fig, out, "figure_2_cost_vs_effort_normalized_rates")


def sensitivity(out, annual):
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.4),layout="constrained")
    for ax in axes.flat:
        years_axis(ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Reported whales / 100 km")
        ax.set_ylim(bottom=0)
    base = annual.loc[annual.footprint.eq("regional") & annual.effort_definition.eq("transect")]
    for window, color, marker, label in [("June_July","#4F5C64","o","June–July"),("July_only","#A78B59","s","July only")]:
        d=base.loc[base.window.eq(window)]
        axes[0,0].plot(d.year,d.whales_per_100km,color=color,marker=marker,label=label)
    panel(axes[0,0],"a","Seasonal sampling window")
    for fp, color, marker, label in [("regional","#4F5C64","o","Regional"),("stable_7","#3A8D6D","s","≥7 years: 105 cells"),("stable_8","#3B6EA8","^","≥8 years: 52 cells"),("stable_9","#A85A91","D","All 9 years: 19 cells")]:
        d=annual.loc[annual.window.eq("June_July")&annual.effort_definition.eq("transect")&annual.footprint.eq(fp)]
        axes[0,1].plot(d.year,d.whales_per_100km,color=color,marker=marker,label=label)
    panel(axes[0,1],"b","Fixed survey-area sensitivity")
    for mode, color, marker, label in [("transect","#4F5C64","o","Transect only"),("transect_search","#A78B59","s","Transect + search"),("transect_circling","#6F7FA1","^","Transect + its circling")]:
        d=annual.loc[annual.window.eq("June_July")&annual.effort_definition.eq(mode)&annual.footprint.eq("regional")]
        axes[1,0].plot(d.year,d.whales_per_100km,color=color,marker=marker,label=label)
    panel(axes[1,0],"c","Matched survey-mode sensitivity")
    main=base.loc[base.window.eq("June_July")]
    obs=pd.read_csv(out/"qa/primary_observer_sensitivity.csv")
    obs=obs.loc[obs.window.eq("June_July")]
    axes[1,1].plot(main.year,main.whales_per_100km,color="#4F5C64",marker="o",label="All recorded observers")
    axes[1,1].plot(obs.year,obs.whales_per_100km,color="#A78B59",marker="s",label="Primary observers only")
    panel(axes[1,1],"d","Observer sensitivity")
    for ax in axes.flat:
        values=np.concatenate([np.asarray(line.get_ydata(),dtype=float) for line in ax.lines])
        finite=values[np.isfinite(values)]
        ax.set_ylim(0,finite.max()*1.3 if len(finite) and finite.max()>0 else 1)
        ax.legend(loc="upper left",fontsize=6.3)
    save(fig,out,"figure_3_effort_rate_sensitivities")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=parser.parse_args()
    out=Path(args.output_dir).resolve()
    if not out.is_relative_to((ROOT/"outputs/asamm_effort_rates_phase8v").resolve()):
        raise ValueError("Figures must remain in the isolated experiment directory")
    style()
    annual=pd.read_csv(out/"source_data/annual_effort_normalized_rates.csv")
    costs=pd.read_csv(out/"source_data/frozen_path_costs.csv")
    intervals=pd.read_csv(out/"source_data/flight_bootstrap_intervals.csv")
    correlations=pd.read_csv(out/"source_data/cost_rate_correlations.csv")
    flights=pd.read_csv(out/"source_data/flight_effort_normalized_counts.csv")
    overview(out,annual,costs,intervals,flights)
    cost_scatter(out,annual,costs,correlations)
    sensitivity(out,annual)
    print("Saved three independent figure sets (PNG, PDF, editable SVG).")


if __name__=="__main__":
    main()
