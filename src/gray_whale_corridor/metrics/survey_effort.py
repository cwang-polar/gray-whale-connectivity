"""Matched survey effort and encounter rates from chronological flight events.

Survey mode applies from an event until the next event. Automatic position
updates inherit the preceding mode; mode must never carry between flights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer


ENTRY_MODES = {
    "deadhead": 1, "p on deadhead": 1,
    "start transect": 2, "p on transect": 2, "s on transect": 2,
    "end transect": 4, "search": 4, "p on search": 4, "s on search": 4,
    "p on circling - transect": 5, "s on circling - transect": 5,
    "p on circling - search": 6, "s on circling - search": 6,
    "caps": 7, "p on caps": 7, "s on caps": 7,
    "fov": 8, "p on fov": 8, "s on fov": 8,
    "caps circling": 9, "p on caps circling": 9, "s on caps circling": 9,
    "fgf": 10, "p on fgf": 10, "s on fgf": 10,
    "caps strip": 11, "p on caps strip": 11, "s on caps strip": 11,
}


def resolve_modes(events: pd.DataFrame) -> pd.DataFrame:
    """Order events and recover the mode at each event, including automatic GPS."""
    out = events.sort_values(
        ["flight_key", "datetime", "event_number", "source_id"], kind="stable"
    ).reset_index(drop=True).copy()
    explicit = pd.to_numeric(out["FltType"], errors="coerce")
    explicit = explicit.where(explicit.between(1, 11))
    implied = out["entry_clean"].map(ENTRY_MODES)
    out["mode_text_conflict"] = explicit.notna() & implied.notna() & explicit.ne(implied)
    # Metadata flight-type codes are authoritative when present.
    known = explicit.fillna(implied)
    out["mode"] = known.groupby(out["flight_key"]).ffill()
    unresolved_resume = out["entry_clean"].eq("resume") & known.isna()
    if unresolved_resume.any():
        previous_survey = known.where(known.isin([2, 4])).groupby(out["flight_key"]).ffill()
        known = known.where(~unresolved_resume, previous_survey)
        out["mode"] = known.groupby(out["flight_key"]).ffill()
    return out


def reconstruct_segments(
    events: pd.DataFrame,
    *,
    max_gap_seconds: float = 300,
    max_speed_kmh: float = 600,
    projected_crs: str = "EPSG:3338",
) -> pd.DataFrame:
    """Build adjacent GPS segments without bridging flights or missing locations."""
    out = resolve_modes(events)
    previous = out.groupby("flight_key", sort=False).shift(1)
    same = previous["datetime"].notna()
    seg = out.loc[same, ["flight_key", "year", "month", "source_id", "datetime", "lon", "lat"]].copy()
    for col in ["datetime", "lon", "lat", "mode", "source_id"]:
        seg[f"start_{col}"] = previous.loc[same, col].to_numpy()
    seg["seconds"] = (seg["datetime"] - seg["start_datetime"]).dt.total_seconds()
    loc_ok = (
        seg["lat"].between(50, 80) & seg["start_lat"].between(50, 80)
        & seg["lon"].between(-180, 180) & seg["start_lon"].between(-180, 180)
    )
    seg["raw_km"] = np.nan
    geod = Geod(ellps="WGS84")
    if loc_ok.any():
        a = seg.loc[loc_ok]
        _, _, metres = geod.inv(a["start_lon"], a["start_lat"], a["lon"], a["lat"])
        seg.loc[loc_ok, "raw_km"] = np.asarray(metres) / 1000
    seg["speed_kmh"] = seg["raw_km"] / (seg["seconds"] / 3600)
    seg["valid_location"] = loc_ok
    seg["valid_time"] = seg["seconds"].gt(0)
    seg["valid_speed"] = seg["speed_kmh"].between(0, max_speed_kmh)
    seg["valid_base"] = loc_ok & seg["valid_time"] & seg["valid_speed"] & seg["start_mode"].notna()
    seg["accepted"] = seg["valid_base"] & seg["seconds"].le(max_gap_seconds)
    transform = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    for prefix in ["start_", ""]:
        x, y = transform.transform(seg[prefix + "lon"].to_numpy(), seg[prefix + "lat"].to_numpy())
        seg[prefix + "x"] = x
        seg[prefix + "y"] = y
    return seg.reset_index(drop=True)


def clipped_fractions(segments: pd.DataFrame, polygon) -> np.ndarray:
    """Fraction inside a projected polygon; geodesic lengths retain WGS84 units."""
    fractions = np.zeros(len(segments), dtype=float)
    ok = segments["valid_location"].to_numpy() & segments["valid_time"].to_numpy()
    if not ok.any():
        return fractions
    a = segments.loc[ok]
    coords = np.stack([
        a[["start_x", "start_y"]].to_numpy(), a[["x", "y"]].to_numpy()
    ], axis=1)
    lines = shapely.linestrings(coords)
    length = shapely.length(lines)
    inside = shapely.length(shapely.intersection(lines, polygon))
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.divide(inside, length, out=np.zeros_like(inside), where=length > 0)
    stationary = length == 0
    if stationary.any():
        fraction[stationary] = shapely.covers(
            polygon, shapely.points(a.loc[stationary, ["x", "y"]].to_numpy())
        ).astype(float)
    fractions[ok] = np.clip(fraction, 0, 1)
    return fractions


def point_membership(events: pd.DataFrame, polygon, projected_crs="EPSG:3338") -> np.ndarray:
    membership = np.zeros(len(events), dtype=bool)
    valid = events["lat"].between(50, 80) & events["lon"].between(-180, 180)
    tr = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = tr.transform(events.loc[valid, "lon"].to_numpy(), events.loc[valid, "lat"].to_numpy())
    membership[valid] = shapely.covers(polygon, shapely.points(x, y))
    return membership


def aggregate_flights(
    segments: pd.DataFrame,
    sightings: pd.DataFrame,
    *,
    fraction_column: str,
    member_column: str,
    modes: list[int],
    max_gap_seconds: float = 300,
) -> pd.DataFrame:
    """Include zero-sighting flights and match numerator and denominator modes."""
    selected = (
        segments["valid_base"] & segments["seconds"].le(max_gap_seconds)
        & segments["start_mode"].isin(modes) & segments[fraction_column].gt(0)
    )
    effort = segments.loc[selected].copy()
    effort["survey_km"] = effort["raw_km"] * effort[fraction_column]
    effort["survey_hours"] = effort["seconds"] / 3600 * effort[fraction_column]
    by = ["flight_key", "year", "month"]
    effort = effort.groupby(by, as_index=False).agg(
        survey_km=("survey_km", "sum"), survey_hours=("survey_hours", "sum"),
        segment_count=("source_id", "size"),
    )
    points = sightings.loc[
        sightings[member_column] & sightings["mode"].isin(modes)
        & ~sightings["repeat_clean"].eq("yes")
    ].copy()
    counts = points.groupby(by, as_index=False).agg(
        sighting_count=("source_id", "size"),
        reported_whales=("whale_count", "sum"),
        missing_group_counts=("whale_count", lambda x: int(x.isna().sum())),
        uncertain_repeats=("repeat_clean", lambda x: int(x.eq("unknown").sum())),
    )
    result = effort.merge(counts, on=by, how="outer", validate="one_to_one")
    for c in ["sighting_count", "reported_whales", "missing_group_counts", "uncertain_repeats"]:
        result[c] = result[c].fillna(0)
    result["unmatched_numerator"] = result["survey_km"].isna() & result["sighting_count"].gt(0)
    result.loc[result["missing_group_counts"].gt(0), "reported_whales"] = np.nan
    return result


RATE_FORMULAS = {
    "whales_per_100km": ("reported_whales", "survey_km", 100),
    "sightings_per_100km": ("sighting_count", "survey_km", 100),
    "whales_per_hour": ("reported_whales", "survey_hours", 1),
    "mean_group_size": ("reported_whales", "sighting_count", 1),
}


def summarize_rates(flights: pd.DataFrame, years: list[int], *, metrics=None) -> pd.DataFrame:
    totals = flights.groupby("year").agg(
        survey_km=("survey_km", "sum"), survey_hours=("survey_hours", "sum"),
        sighting_count=("sighting_count", "sum"), reported_whales=("reported_whales", "sum"),
        flight_count=("flight_key", "nunique"), missing_group_counts=("missing_group_counts", "sum"),
        unmatched_numerators=("unmatched_numerator", "sum"),
    ).reindex(years).reset_index()
    for name, (numerator, denominator, multiplier) in RATE_FORMULAS.items():
        if metrics is not None and name not in metrics:
            continue
        totals[name] = (multiplier * totals[numerator] / totals[denominator]).where(totals[denominator].gt(0))
    missing_count_metrics = [name for name in ["whales_per_100km", "whales_per_hour", "mean_group_size"] if name in totals]
    totals.loc[totals["missing_group_counts"].gt(0), missing_count_metrics] = np.nan
    return totals


def flight_bootstrap(
    flights: pd.DataFrame, *, replicates: int = 2000, seed: int = 20261116,
) -> pd.DataFrame:
    """Resample whole flights within year-month; retain zero-sighting effort."""
    rng = np.random.default_rng(seed)
    rows = []
    for year, annual in flights.groupby("year", sort=True):
        missing_whale_counts = not np.isfinite(annual["reported_whales"]).all()
        sums = np.zeros((replicates, 4))
        for _, monthly in annual.groupby("month", sort=True):
            a = monthly[["reported_whales", "sighting_count", "survey_km", "survey_hours"]].to_numpy(float)
            indices = rng.integers(0, len(a), size=(replicates, len(a)))
            sums += a[indices].sum(axis=1)
        columns = {"reported_whales": 0, "sighting_count": 1, "survey_km": 2, "survey_hours": 3}
        for metric, (num, den, mult) in RATE_FORMULAS.items():
            valid = sums[:, columns[den]] > 0
            values = mult * sums[valid, columns[num]] / sums[valid, columns[den]]
            values = values[np.isfinite(values)]
            # A missing group count cannot be handled by retaining only those
            # bootstrap draws that happen not to select the affected flight.
            if missing_whale_counts and num == "reported_whales":
                values = np.array([], dtype=float)
            lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else [np.nan, np.nan]
            rows.append(dict(year=year, metric=metric, lower=lo, upper=hi, valid_replicates=len(values)))
    return pd.DataFrame(rows)
