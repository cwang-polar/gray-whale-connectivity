"""Resolve amphipod-observation/station-inventory coordinate crosswalk ambiguities for Phase 8A.1.

This script does not build prey surfaces. It expands candidate coordinate matches
for unresolved 2011-2019 benthic station records and applies conservative,
documented resolution rules.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "prey_v2"

BRIAN_CLEAN = OUT / "brian_amphipod_timeseries_cleaned_v2.csv"
ALYNNE_CLEAN = OUT / "alynne_station_coordinates_cleaned_v2.csv"
UNIQUE_MATCHES = OUT / "brian_amphipod_unique_station_match_results_v2.csv"

NEAR_COORD_TOL_DEG = 0.01
TARGET_MIN_YEAR = 2011
TARGET_MAX_YEAR = 2019


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _clean_key(value: object) -> str:
    if _is_blank(value):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _candidate_key(row: pd.Series) -> tuple:
    return (
        _clean_key(row.get("CruiseID_clean")),
        _clean_key(row.get("StationNum_clean")),
        _clean_key(row.get("StationName_clean")),
        str(row.get("DataDate", "")),
        round(_safe_float(row.get("Latitude")), 6),
        round(_safe_float(row.get("Longitude")), 6),
    )


def _unique_station_key(row: pd.Series) -> str:
    return "|".join(
        [
            _clean_key(row.get("Year_clean")),
            _clean_key(row.get("Cruise_clean")),
            _clean_key(row.get("StationNumber_clean")),
            _clean_key(row.get("StationName_clean")),
        ]
    )


def _candidate_rules(brian_row: pd.Series, coords: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    year = _safe_float(brian_row["Year_clean"])
    cruise = _clean_key(brian_row["Cruise_clean"])
    station_num = _clean_key(brian_row["StationNumber_clean"])
    station_name = _clean_key(brian_row["StationName_clean"])

    base = coords[coords["DataYear_clean"].astype(float) == year].copy()
    rules: list[tuple[str, pd.DataFrame]] = []

    if station_name:
        rules.append(
            (
                "year_cruise_stationnumber_stationname",
                base[
                    (base["CruiseID_clean"].map(_clean_key) == cruise)
                    & (base["StationNum_clean"].map(_clean_key) == station_num)
                    & (base["StationName_clean"].map(_clean_key) == station_name)
                ],
            )
        )
    rules.append(
        (
            "year_cruise_stationnumber",
            base[
                (base["CruiseID_clean"].map(_clean_key) == cruise)
                & (base["StationNum_clean"].map(_clean_key) == station_num)
            ],
        )
    )
    if station_name:
        rules.append(
            (
                "year_stationnumber_stationname",
                base[
                    (base["StationNum_clean"].map(_clean_key) == station_num)
                    & (base["StationName_clean"].map(_clean_key) == station_name)
                ],
            )
        )
        rules.append(
            (
                "year_stationname",
                base[base["StationName_clean"].map(_clean_key) == station_name],
            )
        )
    rules.append(
        (
            "year_stationnumber",
            base[base["StationNum_clean"].map(_clean_key) == station_num],
        )
    )
    return rules


def _collect_candidates(problem: pd.DataFrame, coords: pd.DataFrame, family_counts: pd.DataFrame) -> pd.DataFrame:
    rows: dict[tuple, dict] = {}
    for _, brian_row in problem.iterrows():
        key = _unique_station_key(brian_row)
        fam = family_counts.get(key, {})
        for rule, candidates in _candidate_rules(brian_row, coords):
            for _, cand in candidates.iterrows():
                ckey = (key, _candidate_key(cand))
                if ckey not in rows:
                    rows[ckey] = {
                        "brian_unique_station_key": key,
                        "Brian_Year": int(brian_row["Year_clean"]),
                        "Brian_Cruise": brian_row["Cruise_clean"],
                        "Brian_StationNumber": brian_row["StationNumber_clean"],
                        "Brian_StationName": brian_row["StationName_clean"],
                        "Brian_family_count": fam.get("family_count", 0),
                        "Brian_row_count": fam.get("row_count", 0),
                        "candidate_CruiseID": cand.get("CruiseID"),
                        "candidate_StationNum": cand.get("StationNum"),
                        "candidate_StationName": cand.get("StationName"),
                        "candidate_DataYear": cand.get("DataYear"),
                        "candidate_DataDate": cand.get("DataDate"),
                        "candidate_Latitude": cand.get("Latitude"),
                        "candidate_Longitude": cand.get("Longitude"),
                        "match_rule_source": set(),
                    }
                rows[ckey]["match_rule_source"].add(rule)
    out = pd.DataFrame(rows.values())
    if out.empty:
        return pd.DataFrame(
            columns=[
                "brian_unique_station_key",
                "Brian_Year",
                "Brian_Cruise",
                "Brian_StationNumber",
                "Brian_StationName",
                "Brian_family_count",
                "Brian_row_count",
                "candidate_CruiseID",
                "candidate_StationNum",
                "candidate_StationName",
                "candidate_DataYear",
                "candidate_DataDate",
                "candidate_Latitude",
                "candidate_Longitude",
                "match_rule_source",
            ]
        )
    out["match_rule_source"] = out["match_rule_source"].map(lambda x: ";".join(sorted(x)))
    return out.sort_values(["Brian_Year", "Brian_Cruise", "Brian_StationNumber", "candidate_CruiseID"])


def _build_family_counts(brian: pd.DataFrame) -> pd.DataFrame:
    tmp = brian.copy()
    tmp["brian_unique_station_key"] = tmp.apply(
        lambda r: "|".join(
            [
                _clean_key(r.get("Year_clean")),
                _clean_key(r.get("Cruise_clean")),
                _clean_key(r.get("StationNumber_clean")),
                _clean_key(r.get("StationName_clean")),
            ]
        ),
        axis=1,
    )
    grouped = (
        tmp.groupby("brian_unique_station_key")
        .agg(family_count=("Family_clean", "nunique"), row_count=("Family_clean", "size"))
        .reset_index()
    )
    return grouped.set_index("brian_unique_station_key").to_dict("index")


def _classification(problem: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, b in problem.iterrows():
        key = _unique_station_key(b)
        cand = candidates[candidates["brian_unique_station_key"] == key]
        if cand.empty:
            ambiguity_class = "no_candidate_found"
            spread_lat = math.nan
            spread_lon = math.nan
            candidate_count = 0
            cruises = ""
        else:
            spread_lat = cand["candidate_Latitude"].astype(float).max() - cand["candidate_Latitude"].astype(float).min()
            spread_lon = cand["candidate_Longitude"].astype(float).max() - cand["candidate_Longitude"].astype(float).min()
            candidate_count = len(cand)
            cruises = ";".join(sorted(cand["candidate_CruiseID"].map(_clean_key).unique()))
            station_name = _clean_key(b.get("StationName_clean"))
            same_name = (
                bool(station_name)
                and cand["candidate_StationName"].map(_clean_key).eq(station_name).all()
                and cand["candidate_CruiseID"].map(_clean_key).nunique() > 1
            )
            if candidate_count > 1 and spread_lat <= NEAR_COORD_TOL_DEG and spread_lon <= NEAR_COORD_TOL_DEG:
                ambiguity_class = "multiple_candidates_identical_or_near_identical_coordinates"
            elif same_name:
                ambiguity_class = "multiple_candidates_same_station_name_different_cruise_code"
            elif candidate_count > 1:
                ambiguity_class = "multiple_candidates_conflicting_coordinates"
            else:
                ambiguity_class = "single_candidate_found_after_expanded_search"
        rows.append(
            {
                "brian_unique_station_key": key,
                "Brian_Year": int(b["Year_clean"]),
                "Brian_Cruise": b["Cruise_clean"],
                "Brian_StationNumber": b["StationNumber_clean"],
                "Brian_StationName": b["StationName_clean"],
                "candidate_count": candidate_count,
                "candidate_cruises": cruises,
                "coordinate_spread_lat": spread_lat,
                "coordinate_spread_lon": spread_lon,
                "ambiguity_class": ambiguity_class,
            }
        )
    return pd.DataFrame(rows).sort_values(["Brian_Year", "Brian_Cruise", "Brian_StationNumber"])


def _cruise_crosswalk_candidates(unique_matches: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    target = unique_matches[unique_matches["Year_clean"].between(TARGET_MIN_YEAR, TARGET_MAX_YEAR)].copy()
    matched = target[target["matched_yes_no"].astype(bool)].copy()
    support_rows = []
    for cruise, sub in matched.groupby("Cruise_clean"):
        tokens = []
        for val in sub["candidate_cruises"].dropna():
            parts = [p for p in str(val).split(";") if p]
            if len(parts) == 1:
                tokens.append(parts[0])
        if not tokens:
            continue
        counts = pd.Series(tokens).value_counts()
        total = int(counts.sum())
        for cand_cruise, count in counts.items():
            support_rows.append(
                {
                    "Brian_Cruise": cruise,
                    "candidate_Alynne_CruiseID": cand_cruise,
                    "supporting_matched_unique_station_count": int(count),
                    "matched_single_cruise_station_count": total,
                    "support_fraction": round(float(count / total), 4) if total else 0,
                    "candidate_status": "observed_from_existing_matches",
                }
            )

    # Add conservative known-pattern recommendations supported by repeated LAU/SWL matches.
    crosswalk = pd.DataFrame(support_rows)
    known_rows = []
    lau_support = crosswalk[
        crosswalk["Brian_Cruise"].astype(str).str.match(r"LAU1[3-9]$")
        & crosswalk["candidate_Alynne_CruiseID"].astype(str).str.match(r"SWL20(13|14|15|16|17|18|19)$")
    ]
    lau_support_count = int(lau_support["supporting_matched_unique_station_count"].sum())
    for year in [2011, 2012, 2013, 2015]:
        brian = f"LAU{str(year)[-2:]}"
        alynne = f"SWL{year}"
        unresolved_has_candidate = not candidates[
            (candidates["Brian_Cruise"].map(_clean_key) == brian)
            & (candidates["candidate_CruiseID"].map(_clean_key) == alynne)
        ].empty
        if unresolved_has_candidate:
            known_rows.append(
                {
                    "Brian_Cruise": brian,
                    "candidate_Alynne_CruiseID": alynne,
                    "supporting_matched_unique_station_count": lau_support_count,
                    "matched_single_cruise_station_count": lau_support_count,
                    "support_fraction": 1.0,
                    "candidate_status": "proposed_known_pattern_LAU_to_SWL_supported_by_LAU2013_2019_matches",
                }
            )
    if known_rows:
        crosswalk = pd.concat([crosswalk, pd.DataFrame(known_rows)], ignore_index=True)
    return crosswalk.sort_values(["Brian_Cruise", "candidate_status", "candidate_Alynne_CruiseID"])


def _choose_resolution(row: pd.Series, candidates: pd.DataFrame, classification: pd.DataFrame) -> dict:
    key = _unique_station_key(row)
    cand = candidates[candidates["brian_unique_station_key"] == key].copy()
    cls = classification[classification["brian_unique_station_key"] == key].iloc[0]
    base = {
        "brian_unique_station_key": key,
        "Brian_Year": int(row["Year_clean"]),
        "Brian_Cruise": row["Cruise_clean"],
        "Brian_StationNumber": row["StationNumber_clean"],
        "Brian_StationName": row["StationName_clean"],
        "resolved_Latitude": math.nan,
        "resolved_Longitude": math.nan,
        "resolution_method": "unresolved",
        "confidence_level": "low",
        "candidate_count": int(cls["candidate_count"]),
        "notes": "",
    }
    if cand.empty:
        base["resolution_method"] = "unresolved_no_candidate_found"
        base["notes"] = "No coordinate-inventory candidate found under conservative matching rules."
        return base

    lat_spread = cand["candidate_Latitude"].astype(float).max() - cand["candidate_Latitude"].astype(float).min()
    lon_spread = cand["candidate_Longitude"].astype(float).max() - cand["candidate_Longitude"].astype(float).min()
    if len(cand) > 1 and lat_spread <= NEAR_COORD_TOL_DEG and lon_spread <= NEAR_COORD_TOL_DEG:
        base["resolved_Latitude"] = cand["candidate_Latitude"].astype(float).mean()
        base["resolved_Longitude"] = cand["candidate_Longitude"].astype(float).mean()
        base["resolution_method"] = "resolved_by_identical_coordinates"
        base["confidence_level"] = "high"
        base["notes"] = f"Candidate coordinate spread <= {NEAR_COORD_TOL_DEG} degrees."
        return base

    # Conservative supported crosswalk: observation cruise LAUYY -> station-inventory cruise SWL20YY.
    cruise = _clean_key(row["Cruise_clean"])
    year = int(row["Year_clean"])
    expected_swl = f"SWL{year}"
    if re.match(r"LAU(11|12|13|15)$", cruise):
        swl = cand[cand["candidate_CruiseID"].map(_clean_key) == expected_swl]
        if len(swl) == 1:
            base["resolved_Latitude"] = float(swl.iloc[0]["candidate_Latitude"])
            base["resolved_Longitude"] = float(swl.iloc[0]["candidate_Longitude"])
            base["resolution_method"] = "resolved_by_supported_cruise_code_crosswalk_LAU_to_SWL"
            base["confidence_level"] = "medium"
            base["notes"] = (
                "LAU-to-SWL crosswalk is supported by repeated matched LAU2013-2019 station records; "
                "this station had one matching SWL candidate."
            )
            return base

    base["resolution_method"] = "unresolved_conflicting_candidates"
    base["notes"] = "Multiple candidate coordinate sets remain; coordinate resolution requires original cruise station logs."
    return base


def _apply_crosswalk_and_summarize(
    brian: pd.DataFrame,
    unique_matches: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_keys = set(
        crosswalk[
            crosswalk["resolution_method"].astype(str).str.startswith("resolved_")
            & crosswalk["resolved_Latitude"].notna()
            & crosswalk["resolved_Longitude"].notna()
        ]["brian_unique_station_key"]
    )

    unique = unique_matches.copy()
    unique["brian_unique_station_key"] = unique.apply(_unique_station_key, axis=1)
    unique["matched_after_crosswalk"] = unique["matched_yes_no"].astype(bool) | unique["brian_unique_station_key"].isin(resolved_keys)
    unique["target_2011_2019"] = unique["Year_clean"].between(TARGET_MIN_YEAR, TARGET_MAX_YEAR)

    brian2 = brian.copy()
    brian2["brian_unique_station_key"] = brian2.apply(
        lambda r: "|".join(
            [
                _clean_key(r.get("Year_clean")),
                _clean_key(r.get("Cruise_clean")),
                _clean_key(r.get("StationNumber_clean")),
                _clean_key(r.get("StationName_clean")),
            ]
        ),
        axis=1,
    )
    matched_map = unique.set_index("brian_unique_station_key")["matched_after_crosswalk"].to_dict()
    brian2["matched_after_crosswalk"] = brian2["brian_unique_station_key"].map(matched_map).fillna(False)
    brian2["target_2011_2019"] = brian2["Year_clean"].between(TARGET_MIN_YEAR, TARGET_MAX_YEAR)

    summary = pd.DataFrame(
        [
            {
                "scope": "all_years_row_level",
                "matched_count": int(brian2["matched_after_crosswalk"].sum()),
                "total_count": int(len(brian2)),
            },
            {
                "scope": "all_years_unique_station_level",
                "matched_count": int(unique["matched_after_crosswalk"].sum()),
                "total_count": int(len(unique)),
            },
            {
                "scope": "target_2011_2019_row_level",
                "matched_count": int(brian2.loc[brian2["target_2011_2019"], "matched_after_crosswalk"].sum()),
                "total_count": int(brian2["target_2011_2019"].sum()),
            },
            {
                "scope": "target_2011_2019_unique_station_level",
                "matched_count": int(unique.loc[unique["target_2011_2019"], "matched_after_crosswalk"].sum()),
                "total_count": int(unique["target_2011_2019"].sum()),
            },
        ]
    )
    summary["match_rate_percent"] = (summary["matched_count"] / summary["total_count"] * 100).round(3)

    by_year = (
        unique[unique["target_2011_2019"]]
        .groupby("Year_clean")
        .agg(
            matched_unique_station_count=("matched_after_crosswalk", "sum"),
            total_unique_station_count=("matched_after_crosswalk", "size"),
        )
        .reset_index()
    )
    by_year["match_rate_percent"] = (
        by_year["matched_unique_station_count"] / by_year["total_unique_station_count"] * 100
    ).round(3)
    by_year.insert(0, "scope", "target_2011_2019_by_year_unique_station_level")
    by_year = by_year.rename(columns={"Year_clean": "group"})

    by_cruise = (
        unique[unique["target_2011_2019"]]
        .groupby("Cruise_clean")
        .agg(
            matched_unique_station_count=("matched_after_crosswalk", "sum"),
            total_unique_station_count=("matched_after_crosswalk", "size"),
        )
        .reset_index()
    )
    by_cruise["match_rate_percent"] = (
        by_cruise["matched_unique_station_count"] / by_cruise["total_unique_station_count"] * 100
    ).round(3)
    by_cruise.insert(0, "scope", "target_2011_2019_by_cruise_unique_station_level")
    by_cruise = by_cruise.rename(columns={"Cruise_clean": "group"})

    return summary, by_year, by_cruise


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    brian = pd.read_csv(BRIAN_CLEAN)
    coords = pd.read_csv(ALYNNE_CLEAN)
    unique = pd.read_csv(UNIQUE_MATCHES)

    problem = unique[
        unique["Year_clean"].between(TARGET_MIN_YEAR, TARGET_MAX_YEAR)
        & ~unique["matched_yes_no"].astype(bool)
    ].copy()
    family_counts = _build_family_counts(brian)

    candidates = _collect_candidates(problem, coords[coords["coordinate_valid"].astype(bool)].copy(), family_counts)
    classification = _classification(problem, candidates)
    cruise_crosswalk = _cruise_crosswalk_candidates(unique, candidates)
    station_crosswalk = pd.DataFrame([_choose_resolution(r, candidates, classification) for _, r in problem.iterrows()])

    summary, by_year, by_cruise = _apply_crosswalk_and_summarize(brian, unique, station_crosswalk)
    summary_out = pd.concat(
        [
            summary,
            by_year.rename(
                columns={
                    "matched_unique_station_count": "matched_count",
                    "total_unique_station_count": "total_count",
                }
            )[["scope", "group", "matched_count", "total_count", "match_rate_percent"]],
            by_cruise.rename(
                columns={
                    "matched_unique_station_count": "matched_count",
                    "total_unique_station_count": "total_count",
                }
            )[["scope", "group", "matched_count", "total_count", "match_rate_percent"]],
        ],
        ignore_index=True,
        sort=False,
    )

    manual = station_crosswalk[station_crosswalk["confidence_level"].eq("low")].copy()
    if not manual.empty:
        manual = manual.merge(
            classification[
                [
                    "brian_unique_station_key",
                    "candidate_cruises",
                    "coordinate_spread_lat",
                    "coordinate_spread_lon",
                    "ambiguity_class",
                ]
            ],
            on="brian_unique_station_key",
            how="left",
        )
        manual["recommended_manual_action"] = (
            "Resolve ambiguous station coordinates using original cruise station logs; "
            "required fields are the observation cruise, station number/name and candidate coordinate-inventory cruises."
        )

    candidates.to_csv(OUT / "brian_alynne_ambiguous_candidate_matches_v1.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    classification.to_csv(OUT / "brian_alynne_ambiguity_classification_v1.csv", index=False)
    cruise_crosswalk.to_csv(OUT / "brian_alynne_cruise_code_crosswalk_candidates_v1.csv", index=False)
    station_crosswalk.to_csv(OUT / "brian_alynne_station_crosswalk_v1.csv", index=False)
    summary_out.to_csv(OUT / "brian_amphipod_matching_summary_after_crosswalk_v1.csv", index=False)
    manual.to_csv(OUT / "brian_alynne_manual_review_needed_v1.csv", index=False)

    qa = {
        "target_problem_unique_station_count": int(len(problem)),
        "candidate_match_rows": int(len(candidates)),
        "resolved_automatically": int(
            station_crosswalk["resolution_method"].astype(str).str.startswith("resolved_").sum()
        ),
        "resolved_by_identical_coordinates": int(
            station_crosswalk["resolution_method"].eq("resolved_by_identical_coordinates").sum()
        ),
        "resolved_by_supported_cruise_crosswalk": int(
            station_crosswalk["resolution_method"].eq("resolved_by_supported_cruise_code_crosswalk_LAU_to_SWL").sum()
        ),
        "remaining_low_confidence": int(station_crosswalk["confidence_level"].eq("low").sum()),
        "target_2011_2019_unique_station_match_rate_after_crosswalk_percent": float(
            summary_out.loc[
                summary_out["scope"].eq("target_2011_2019_unique_station_level"), "match_rate_percent"
            ].iloc[0]
        ),
        "qa_gate_95_percent_passed": bool(
            summary_out.loc[
                summary_out["scope"].eq("target_2011_2019_unique_station_level"), "match_rate_percent"
            ].iloc[0]
            >= 95
        ),
        "rules": [
            "Automatic high-confidence resolution only for multiple candidates with coordinate spread <= 0.01 degrees.",
            "Medium-confidence LAU-to-SWL crosswalk applied only where a single SWL candidate exists and the pattern is supported by repeated LAU2013-2019 matched records.",
            "W2011, AMB17, and no-candidate records remain low-confidence/manual review when candidate coordinate conflicts persist.",
        ],
    }
    (OUT / "brian_alynne_station_crosswalk_qa_v1.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
