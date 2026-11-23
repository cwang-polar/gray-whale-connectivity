"""Phase 8I.2B ASAMM validation-window refinement.

This script re-summarizes existing Phase 8I.2 outputs. It does not rerun
models or change extraction tables. The main manuscript validation window is
set to June-July, with July-only as sensitivity and July-September as
supplementary open-water context.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = ROOT / "data/processed/validation"
FIG = ROOT / "outputs/figures/validation"
DOCS = ROOT / "docs"
MANIFEST = ROOT / "outputs/results_manifest.csv"
REGISTRY = DOCS / "results_registry.md"
CLAIMS = DOCS / "paper_claims_evidence_matrix.md"
README = ROOT / "README.md"
PHASE_MEMO = DOCS / "phase_memos/phase8i2b_asamm_validation_window_refinement.md"
PAPER_MEMO = DOCS / "paper_use_decisions/phase8i2b_asamm_validation_window_refinement_paper_use.md"
FINAL_CLAIMS = DOCS / "final_synthesis_v2/paper_claims_evidence_matrix_final_v2.md"
FIG_PLAN = DOCS / "final_synthesis_v2/main_figure_plan_v2.md"
RESULTS_OUTLINE = DOCS / "final_synthesis_v2/results_section_outline_v2.md"
KEY_BULLETS = DOCS / "final_synthesis_v2/key_result_bullets_v2.md"
GAPS = DOCS / "final_synthesis_v2/gaps_before_manuscript_v2.md"
PAPER_USE_MAP = DOCS / "final_synthesis_v2/outputs_to_paper_use_map_v2.csv"

PHASE = "Phase 8I.2B ASAMM validation-window refinement"
SCRIPT = "scripts/refine_asamm_validation_window_phase8i2b_v2.py"
WINDOW_ROLE = {
    "June_July": "main_validation",
    "July_only": "sensitivity_validation",
    "July_September": "supplement_open_water_context",
    "August_September": "diagnostic_late_season_context",
}
MODEL_LABELS = {
    "prey_only": "Prey-only",
    "prey_ice": "Prey + ice",
    "prey_ice_ship": "Prey + ice + ship",
}
MODEL_ORDER = ["prey_only", "prey_ice", "prey_ice_ship"]
WINDOW_ORDER = ["June_July", "July_only", "July_September"]


def update_text_once(path: Path, marker: str, text: str) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def replace_if_present(path: Path, old: str, new: str) -> None:
    if not path.exists():
        return
    txt = path.read_text()
    if old in txt:
        path.write_text(txt.replace(old, new))


def month_availability() -> pd.DataFrame:
    records = []
    for path in sorted((ROOT / "data/raw/validation/asamm_1979_2021/csv_export/ASAMM to csv").glob("*/_Output__*.csv")):
        usecols = lambda c: c in {"GMT_Minus8_DateTime", "Species", "Entry"}
        df = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
        dt = pd.to_datetime(df["GMT_Minus8_DateTime"], errors="coerce")
        sub = df.loc[dt.dt.year.between(2011, 2019)].copy()
        sub["month"] = dt.loc[sub.index].dt.month
        sub["is_gray_whale"] = sub["Species"].fillna("").str.strip().str.lower().eq("gray whale")
        def effort_entry(entry: object) -> bool:
            text = str(entry).strip().lower()
            if text in {"", ".", "nan", "none"} or "deadhead" in text:
                return False
            return any(k in text for k in ["transect", "search", "circling", "resume", "start", "end", "fov", "caps", "fgf", "divert"])
        sub["is_strict_effort_entry"] = sub["Entry"].map(effort_entry)
        records.append(sub[["month", "is_gray_whale", "is_strict_effort_entry"]])
    all_events = pd.concat(records, ignore_index=True)
    months = list(range(1, 13))
    rows = []
    for m in months:
        s = all_events.loc[all_events["month"].eq(m)]
        rows.append(
            {
                "month": m,
                "gray_whale_record_count_2011_2019": int(s["is_gray_whale"].sum()),
                "strict_effort_event_count_2011_2019": int(s["is_strict_effort_entry"].sum()),
                "validation_role": "not_available" if m == 5 and s["is_gray_whale"].sum() == 0 and s["is_strict_effort_entry"].sum() == 0 else "",
            }
        )
    out = pd.DataFrame(rows)
    out.loc[out["month"].isin([6, 7]), "validation_role"] = "main_window_month"
    out.loc[out["month"].eq(7), "validation_role"] = "main_window_month_and_july_sensitivity"
    out.loc[out["month"].isin([8, 9]), "validation_role"] = "supplement_open_water_context_only"
    out.to_csv(OUT / "asamm_phase8i2b_month_availability_and_window_decision.csv", index=False)
    return out


def build_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.read_csv(OUT / "asamm_phase8i2_diagnostic_overall_summary.csv")
    bg = pd.read_csv(OUT / "asamm_phase8i2_background_contrast.csv")
    overall["window_role"] = overall["validation_window"].map(WINDOW_ROLE)
    bg["window_role"] = bg["validation_window"].map(WINDOW_ROLE)

    priority = overall.loc[overall["validation_window"].isin(WINDOW_ORDER)].copy()
    priority["prey_variable_role"] = np.where(priority["prey_variable"].eq("total_kJm2_frozen"), "main_prey_variable", "prey_variable_sensitivity")
    priority["recommended_paper_use"] = priority["validation_window"].map(
        {
            "June_July": "main_text_validation_candidate",
            "July_only": "sensitivity_or_supplement",
            "July_September": "supplement_open_water_context",
        }
    )
    priority["interpretation_guardrail"] = np.where(
        priority["validation_window"].eq("July_September"),
        "Do not use as main ice-effect validation; late-season ASAMM records can overemphasize northern residence/open-water context.",
        "Interpret as effort-footprint-aware diagnostic consistency with May-July model season.",
    )
    priority.to_csv(OUT / "asamm_phase8i2b_validation_window_priority_summary.csv", index=False)

    main = priority.loc[(priority["validation_window"].eq("June_July")) & (priority["prey_variable"].eq("total_kJm2_frozen"))].copy()
    main.to_csv(OUT / "asamm_phase8i2b_main_june_july_validation_summary.csv", index=False)

    kJ_compare = priority.loc[priority["prey_variable"].eq("total_kJm2_frozen")].copy()
    kJ_compare.to_csv(OUT / "asamm_phase8i2b_kJ_window_model_comparison.csv", index=False)

    bg_priority = bg.loc[(bg["validation_window"].isin(WINDOW_ORDER)) & (bg["prey_variable"].eq("total_kJm2_frozen"))].copy()
    bg_priority.to_csv(OUT / "asamm_phase8i2b_background_contrast_window_priority.csv", index=False)
    return overall, priority, main, kJ_compare


def plot_window_refinement(kj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True)
    metrics = [
        ("mean_lcp_percentile_within_effort_footprint", "Mean LCP percentile\nwithin ASAMM effort footprint", (0, 100)),
        ("percent_records_top5pct_lcp", "Records in top 5% LCP cells (%)", (0, max(25, kj["percent_records_top5pct_lcp"].max() * 1.25))),
        ("median_distance_to_core_corridor_km", "Median distance to frequent\ncorridor cells (km)", (0, max(25, kj["median_distance_to_core_corridor_km"].max() * 1.25))),
    ]
    colors = {"June_July": "#2ca25f", "July_only": "#3182bd", "July_September": "#9e9ac8"}
    hatches = {"June_July": "", "July_only": "//", "July_September": ".."}
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for i, window in enumerate(WINDOW_ORDER):
            vals = [
                kj.loc[(kj["model_layer"].eq(layer)) & (kj["validation_window"].eq(window)), metric].mean()
                for layer in MODEL_ORDER
            ]
            ax.bar(x + (i - 1) * width, vals, width=width, color=colors[window], hatch=hatches[window], label=window.replace("_", "-"))
        if "percentile" in metric:
            ax.axhline(50, color="0.45", ls="--", lw=1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right")
    axes[0].set_title("A")
    axes[1].set_title("B")
    axes[2].set_title("C")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    fig.suptitle("ASAMM validation-window refinement", y=1.08, fontsize=13)
    fig.text(0.5, 1.01, "June-July is the main comparison; July-September is supplementary open-water context.", ha="center", fontsize=9)
    fig.text(0.5, -0.02, "All metrics are diagnostic and use the ASAMM effort-covered C3 footprint where applicable.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    fig.savefig(FIG / "asamm_phase8i2b_validation_window_refinement_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    main = kj.loc[kj["validation_window"].eq("June_July")].copy()
    for metric, marker, label in [
        ("mean_lcp_percentile_within_effort_footprint", "o", "Mean percentile"),
        ("percent_records_top5pct_lcp", "s", "Top-5% overlap (%)"),
    ]:
        ax.plot([MODEL_LABELS[m] for m in MODEL_ORDER], [main.loc[main["model_layer"].eq(m), metric].mean() for m in MODEL_ORDER], marker=marker, label=label)
    ax.axhline(50, color="0.6", ls="--", lw=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentile or percent")
    ax.set_title("Main ASAMM validation window: June-July")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "asamm_phase8i2b_main_june_july_validation_summary.png", dpi=300)
    plt.close(fig)


def write_docs(months: pd.DataFrame, main: pd.DataFrame, kj: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def update_registry_manifest() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    months = month_availability()
    _, _, main_summary, kj = build_summaries()
    plot_window_refinement(kj)
    write_docs(months, main_summary, kj)
    update_registry_manifest()
    qa = {
        "phase": PHASE,
        "main_validation_window": "June_July",
        "sensitivity_window": "July_only",
        "supplement_window": "July_September",
        "may_gray_whale_records_2011_2019": int(months.loc[months["month"].eq(5), "gray_whale_record_count_2011_2019"].iloc[0]),
        "may_strict_effort_events_2011_2019": int(months.loc[months["month"].eq(5), "strict_effort_event_count_2011_2019"].iloc[0]),
        "warnings": [
            "June-July is mostly July-driven because ASAMM June gray whale records are sparse.",
            "July-September should not be used as main May-July ice-effect validation.",
            "ASAMM effort does not cover southern C3; unflown cells are unvalidated, not whale absence.",
            "No models or extraction tables were rerun in this phase.",
        ],
    }
    (OUT / "asamm_phase8i2b_validation_window_refinement_qa.json").write_text(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
