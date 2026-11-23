#!/usr/bin/env python3
"""Render S8 from the available-period correlation table."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))

import build_phase8v_temporal_metric_candidates_20261117 as previous
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.text import Text

SOURCE = ROOT / "outputs/asamm_effort_rates_phase8v/available_period_comparison_20261117_v1"
FIGURES = ROOT / "outputs/manuscript_figures_v3"
OUT = FIGURES / "supplementary_figures_working/available_period_S8_20261117_v2"
STEM = "gray_whale_si_S8_available_period_single_candidate_v2"
CAPTION = """**Supplementary Figure S8 | Annual cost–encounter correlations over each scenario's available period.** Pearson correlations compare annual mean modeled May–July path costs with June–July reported gray whales per 100 km of matched transect survey effort. Prey and prey+ice use 2011–2019 (nine annual pairs); prey+vessel and prey+ice+vessel use 2015–2019 (five annual pairs). Symbols and adjacent values show full-period Pearson r. Horizontal ranges span the minimum and maximum r after omitting one year at a time, retaining eight annual pairs for the nine-year series and four for the five-year series. The horizontal dashed line separates the two periods. Negative r indicates lower modeled costs coinciding with higher survey encounter rates. All scenarios use the primary amphipod-energy surface and applicable primary ice and vessel settings (α=0.50 for vessel scenarios).
"""


def relative(path):
    return str(path.relative_to(ROOT))


def main(paths):
    paths.require_task("S8")
    paths.require_reserved()
    OUT = paths.output("figures/S8")
    protected = [paths.input("correlations"), Path(__file__).resolve(), Path(previous.__file__).resolve()]
    before = {paths.label(p): previous.sha(p) for p in protected}
    source = paths.input("correlations")
    data = pd.read_csv(source)
    assert data.scenario.tolist() == list(previous.SCENARIOS)
    assert data.n_years.tolist() == [9, 9, 5, 5]
    assert data.period.tolist() == ["2011–2019", "2011–2019", "2015–2019", "2015–2019"]
    for child in ["candidates", "source_data", "qa", "baselines"]:
        (OUT / child).mkdir(parents=True, exist_ok=False)
    (OUT / "baselines/v8_frozen_reference.json").write_text(json.dumps(paths.figure_reference("S8"), indent=2))
    columns = ["scenario", "period", "n_years", "pearson_r", "omit_year_min", "omit_year_max"]
    data[columns].to_csv(OUT / "source_data/available_period_correlations.csv", index=False)
    previous.style()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 86 / 25.4), layout="constrained")
    previous.correlation_panel(ax, data, "", "", "omit_year_min", "omit_year_max")
    labels = [f"{previous.SCENARIOS[row.scenario][0]}\n{row.period} (n = {row.n_years})"
              for row in data.itertuples()]
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(1.5, color="#D5DBDF", linewidth=.7, linestyle=(0, (3, 3)))
    ax.set_xlabel("Pearson $r$", fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(item.get_text())
    for ext in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(OUT / "candidates" / f"{STEM}.{ext}", dpi=600, facecolor="white")
    fig.savefig(OUT / "qa" / f"{STEM}_review.png", dpi=160, facecolor="white")
    text_items = [t.get_text() for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip()]
    plt.close(fig)
    (OUT / "caption.md").write_text(paths.figure_reference("S8")["caption"])
    (OUT / "README.md").write_text("""# Single-chart Supplementary Figure S8

Pearson correlations between annual corridor cost and survey-effort-standardized
whale counts, with ranges from omitting each survey year in turn.

Prey and prey+ice: 2011–2019; vessel scenarios: 2015–2019.
The ranges describe survey-year sensitivity, not confidence intervals.
See caption.md for metric definitions and interpretation.
""")
    after = {paths.label(p): previous.sha(p) for p in protected}
    checks = {
        "protected_inputs_unchanged": before == after,
        "single_axes": True,
        "source_values_unchanged": data[columns].equals(pd.read_csv(source)[columns]),
        "no_text_outside_canvas": not outside,
        "no_title_or_explanatory_subtitle": not any(
            "More negative" in t or "Annual correlations" in t for t in text_items),
        "editable_svg_text": "<text" in (OUT / "candidates" / f"{STEM}.svg").read_text(),
        "embedded_pdf_fonts": b"/FontFile2" in (OUT / "candidates" / f"{STEM}.pdf").read_bytes(),
    }
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "script": paths.label(Path(__file__).resolve()), "source": paths.label(source),
                "source_sha256": previous.sha(source), "protected_count": len(before),
                "protected_before": before, "protected_after": after,
                "dimensions_mm": [183, 86], "checks": checks, "text_outside": outside}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(checks.values()):
        raise RuntimeError(checks)
    print(json.dumps({"out": str(OUT), "protected_count": len(before), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
