"""Future table formatting from archived results, never a model/metric runner."""

import csv
import json
from decimal import Decimal, ROUND_HALF_UP

from .control import CONFIG


SCENARIOS = {"prey_only": "Prey", "prey_ice": "Prey+ice", "prey_vessel": "Prey+vessel",
             "prey_ice_vessel": "Prey+ice+vessel"}
ORDER = {name: index for index, name in enumerate(SCENARIOS)}


def read_rows(paths, alias):
    with paths.input(alias).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integer(value, *, signed=False):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Expected an integer, got {value}")
    return format(int(number), "+d" if signed else "d")


def decimal(value, digits=1, *, signed=False, thousands=False):
    return format(float(value), ("+" if signed else "") + ("," if thousands else "") + f".{digits}f")


def decimal_half_up(value, digits):
    """Round the archived decimal string for display, without editing its value."""
    return format(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), f".{digits}f")


def table_s1(paths):
    annual = []
    for scenario in SCENARIOS:
        for row in read_rows(paths, scenario):
            if row["prey_variable"] != "total_kJm2_frozen":
                continue
            annual.append((int(row["target_year"]), ORDER[scenario], [row["target_year"], SCENARIOS[scenario],
                decimal(row["mean_path_cost"], thousands=True), integer(row["corridor_area_cells_freq_gt_0_05"])]))
    contrasts = []
    for alias, setting in (("ice_contrasts", "primary_lower_0_15_upper_0_55"),
                           ("vessel_contrasts", "primary_alpha_0_50")):
        for row in read_rows(paths, alias):
            if row["setting"] != setting:
                continue
            a, b = row["scenario_a"], row["scenario_b"]
            contrasts.append((ORDER[b], int(row["target_year"]), [row["target_year"], f"{SCENARIOS[b]} versus {SCENARIOS[a]}",
                decimal(row["percent_cost_change_b_vs_a"], signed=True),
                integer(row["frequent_cell_change_b_minus_a"], signed=True), decimal(row["jaccard_freq_gt_0_05"], 3),
                decimal(row["centroid_displacement_km"]), decimal(row["centroid_north_south_displacement_km"], signed=True)]))
    return [[r[2] for r in sorted(annual)], [r[2] for r in sorted(contrasts)]]


def table_s2(paths):
    early = read_rows(paths, "early_full") + read_rows(paths, "early_common")
    late = [r for r in read_rows(paths, "late") if r["summary_type"] == "pooled"]
    parts = []
    for rows, period, percentile, top5, distance in (
        (early, "analysis_period", "effort_footprint_percentile", "top5_overlap_percent", "distance_to_frequent_corridor_km"),
        (late, "display_period", "mean_effort_footprint_percentile", "percent_records_top5pct", "median_distance_to_frequent_corridor_km"),
    ):
        parts.append([[r[period], SCENARIOS[r["scenario"]], f'{int(r["record_count"]):,}',
                       decimal(r[percentile]), decimal(r[top5]), decimal(r[distance])]
                      for r in sorted(rows, key=lambda r: (r[period], ORDER[r["scenario"]]))])
    return parts


def table_s3(paths):
    rows = sorted(read_rows(paths, "exposure"), key=lambda r: float(r["lat"]))
    if any(r["exposure_priority"] not in ("True", "False") for r in rows):
        raise ValueError("Unrecognized archived exposure classification")
    return [[[str(index), decimal(r["lon"], 3), decimal(r["lat"], 3), integer(r["prey_years_in_top5pct"]),
              integer(r["ice_years_in_top5pct"]), decimal(r["mean_scaled_vessel_exposure"], 3),
              "Yes" if r["exposure_priority"] == "True" else "No"] for index, r in enumerate(rows, 1)]]


def table_s4(paths):
    rows = read_rows(paths, "changes")
    scenario = lambda r: "prey_only" if r["scenario"] == "prey" else r["scenario"]
    primary = [r for r in rows if r["combination_operator"] == "complementary" and float(r["pelagic_strength_beta"]) == 0.25]
    part_a = [[r["target_year"], SCENARIOS[scenario(r)], decimal(r["delta_mean_path_cost_percent"]),
               integer(r["delta_frequent_corridor_cell_count"], signed=True), decimal_half_up(r["jaccard_with_baseline_frequent_cells"], 3),
               decimal(r["frequent_centroid_shift_from_baseline_km"])]
              for r in sorted(primary, key=lambda r: (int(r["target_year"]), ORDER[scenario(r)]))]
    part_b = []
    for operator, label in (("maximum", "Maximum"), ("additive_capped", "Capped additive")):
        for beta in (0.25, 0.50):
            group = [r for r in rows if r["combination_operator"] == operator and float(r["pelagic_strength_beta"]) == beta]
            if {(int(r["target_year"]), scenario(r)) for r in group} != {(y, s) for y in (2017, 2018) for s in SCENARIOS} or len(group) != 8:
                raise ValueError("Expected the eight frozen scenario-year results for the displayed range")
            values = [label, f"{beta:.2f}", str(len(group))]
            for field, digits, separator in (("delta_mean_path_cost_percent", 1, " to "),
                    ("jaccard_with_baseline_frequent_cells", 3, "-"), ("frequent_centroid_shift_from_baseline_km", 1, "-")):
                numbers = [float(r[field]) for r in group]
                values.append(decimal(min(numbers), digits) + separator + decimal(max(numbers), digits))
            part_b.append(values)
    return [part_a, part_b]


def table_s5(paths):
    rows = read_rows(paths, "coefficients")
    families = ("Ampeliscidae", "Isaeidae", "Melitidae", "Pontoporeiidae")
    by_family = {r["amphipod_family"]: r for r in rows}
    if len(rows) != 4 or set(by_family) != set(families):
        raise ValueError("Expected the four archived amphipod-family coefficient rows")
    fields = (("dry_weight_to_wet_weight_ratio", 3), ("formalin_preserved_energy_kJ_g_DW", 2),
              ("frozen_energy_kJ_g_DW", 2), ("frozen_to_formalin_energy_ratio", 3))
    return [[[family] + [decimal(by_family[family][field], digits) for field, digits in fields]
             for family in families]]


BUILDERS = {"TableS1": table_s1, "TableS2": table_s2, "TableS3": table_s3, "TableS4": table_s4,
            "TableS5": table_s5}


def export_table(paths):
    """Run only after authorization; compare every displayed cell with frozen study."""
    paths.require_task(*BUILDERS)
    before = paths.snapshot()
    reference_name = "frozen_table_s5_reference.json" if paths.task == "TableS5" else "frozen_table_references.json"
    references = json.loads((CONFIG / reference_name).read_text())["tables"][paths.task]
    parts = BUILDERS[paths.task](paths)
    expected = [part["rows"] for part in references["parts"]]
    if parts != expected:
        raise ValueError("Formatted archive values do not match frozen V8; do not silently substitute")
    paths.reserve_output()
    paths.output("tables").mkdir()
    paths.output("qa").mkdir()
    markdown = ["# " + references["title"], ""]
    for index, (part, rows) in enumerate(zip(references["parts"], parts), 1):
        with paths.output(f"tables/{paths.task}_part_{index}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(part["headers"])
            writer.writerows(rows)
        markdown += ["## " + part["title"], "", "| " + " | ".join(part["headers"]) + " |",
                     "| " + " | ".join(["---"] * len(part["headers"])) + " |"]
        markdown.extend("| " + " | ".join(row) + " |" for row in rows)
        markdown.append("")
    markdown += [references["note"], ""]
    paths.output(f"tables/{paths.task}.md").write_text("\n".join(markdown), encoding="utf-8")
    if paths.snapshot() != before:
        raise RuntimeError("An archived table input changed")
    paths.output("qa/table_manifest.json").write_text(json.dumps({
        "task": paths.task, "inputs": before, "frozen_display_cells_match": True,
        "part_row_counts": [len(part) for part in parts], "metrics_recomputed": False,
        "layer": "Formatting stored metrics; S4B ranges aggregate the eight stored results only",
        "reference": "config/ncc_v8/" + reference_name, "exporter": __name__,
    }, indent=2), encoding="utf-8")
