"""Audit and model observation-informed euphausiid support for Phase 8S.

This isolated phase asks whether available euphausiid observations support a
spatial layer before any least-cost-path analysis is attempted. The model gate
is intentionally asymmetric: year-specific June surfaces may proceed when
they outperform a spatially blocked null, while annual extrapolation requires
successful transfer in both directions between 2017 and 2018.
"""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gray_whale_corridor.surfaces.pelagic_support import (  # noqa: E402
    ModelSpec,
    RelativeSupportRegressor,
    assign_spatial_blocks,
    interpolation_domain,
    nearest_neighbor_quantiles,
    predictive_metrics,
    robust_support_scale,
    within_stratum_relative_score,
)


PHASE = "Phase 8S observation-informed pelagic-support identifiability"
SCRIPT = "scripts/audit_phase8s_observation_informed_pelagic_support.py"
OUT = ROOT / "outputs/pelagic_support_phase8s"
TABLES = OUT / "tables"
SURFACES = OUT / "surfaces"
STATUS = OUT / "phase8s_model_audit_status.json"
MANIFEST = OUT / "phase8s_model_audit_manifest.json"

GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_C3_recurring_support_v2.csv"
EVIDENCE = ROOT / "outputs/pelagic_prey_phase8n/tables/phase8n_euphausiid_grid_evidence.csv"
OBSERVATIONS = ROOT / "outputs/pelagic_prey_phase8n/tables/phase8n_harmonized_euphausiid_observations.csv"
GLORYS = ROOT / "data/raw/ice/cmems_mod_glo_phy_my_0.083deg_P1M-m_1779048578112.nc"

EARLY_ROLE = "quantitative_target_window_abundance"
LATE_ROLES = {
    "quantitative_late_season_abundance_biomass",
    "quantitative_late_season_abundance",
    "quantitative_local_late_season_abundance",
}
PHYSICAL_COLUMNS = (
    "glorys_surface_temperature_c",
    "glorys_surface_salinity",
    "glorys_eastward_current_ms",
    "glorys_northward_current_ms",
)
MODEL_TYPES = ("mean_null", "hydrographic", "spatial", "spatial_hydrographic")
ALPHAS = (0.1, 1.0, 10.0)
KNOTS = (3, 4, 5)
RANDOM_STATE = 8675309
BOOTSTRAP_DRAWS = 200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    for path in (OUT, TABLES, SURFACES):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_stage(stage: str, status: str, **details: object) -> None:
    if STATUS.exists():
        state = json.loads(STATUS.read_text(encoding="utf-8"))
    else:
        state = {
            "phase": PHASE,
            "status": "running",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "stages": {},
            "established_output_modified": False,
            "manuscript_or_si_modified": False,
        }
    state["stages"][stage] = {"status": status, "updated_utc": utc_now(), **details}
    state["updated_utc"] = utc_now()
    if status == "failed":
        state["status"] = "failed"
    write_json(STATUS, state)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_snapshot() -> pd.DataFrame:
    rows = []
    for path in (GRID, EVIDENCE, OBSERVATIONS, GLORYS):
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "phase8s_input_integrity_before.csv", index=False)
    return frame


def source_decisions() -> pd.DataFrame:
    rows = [
        {
            "source_group": "ASGARD June 2017 and 2018",
            "season": "May-July target window",
            "decision": "include_primary",
            "model_role": "year-specific relative-abundance surface",
            "reason": "Comparable station-composite method and quantitative abundance within the study grid.",
        },
        {
            "source_group": "IERP synthesis May-July context",
            "season": "May-July target window",
            "decision": "exclude_from_surface",
            "model_role": "context_only",
            "reason": "Sparse records fall outside the regional ocean analysis grid and use heterogeneous methods.",
        },
        {
            "source_group": "ASGARD lipid samples",
            "season": "May-July target window",
            "decision": "exclude_from_surface",
            "model_role": "quality_context_only",
            "reason": "Prey-quality observations contain no paired quantitative abundance.",
        },
        {
            "source_group": "AMBON 2017",
            "season": "August-October",
            "decision": "include_late_gate",
            "model_role": "late-season transportability test",
            "reason": "Quantitative abundance is available, but method effects require within-stratum scaling.",
        },
        {
            "source_group": "CSESP 2011-2014",
            "season": "August-October",
            "decision": "include_late_gate",
            "model_role": "late-season transportability test",
            "reason": "Quantitative abundance is available for paired mesh methods and is retained by method stratum.",
        },
        {
            "source_group": "Arctic IERP laboratory 2017 and 2019",
            "season": "August-October",
            "decision": "include_late_gate",
            "model_role": "late-season transportability test",
            "reason": "Laboratory abundance is retained separately for each net method.",
        },
        {
            "source_group": "Point Barrow 2011-2015",
            "season": "August-October",
            "decision": "include_late_gate",
            "model_role": "late-season transportability test",
            "reason": "Quantitative local abundance is usable only as its own source-method-year stratum.",
        },
        {
            "source_group": "AMBON 2015",
            "season": "August-October",
            "decision": "exclude_from_surface",
            "model_role": "footprint_only",
            "reason": "Quantitative measurements could not be linked reliably to occurrence events.",
        },
        {
            "source_group": "Arctic IERP RZA 2017 and 2019",
            "season": "August-October",
            "decision": "exclude_primary",
            "model_role": "method_sensitivity_only",
            "reason": "Rapid-analysis counts are a method check and are not pooled with laboratory abundance.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "phase8s_source_inclusion_decisions.csv", index=False)
    return frame


def load_model_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = pd.read_csv(EVIDENCE, dtype={"grid_id": str})
    early = evidence.loc[evidence["data_role"].eq(EARLY_ROLE)].copy()
    late = evidence.loc[evidence["data_role"].isin(LATE_ROLES)].copy()
    early["response_log_abundance"] = np.log1p(early["mean_euphausiid_abundance_m3"])
    early["response_relative_score"] = within_stratum_relative_score(
        early,
        abundance_column="mean_euphausiid_abundance_m3",
        stratum_columns=("year", "method_group"),
    )
    late["response_relative_score"] = within_stratum_relative_score(
        late,
        abundance_column="mean_euphausiid_abundance_m3",
        stratum_columns=("source_id", "year", "method_group"),
    )
    if early["response_log_abundance"].isna().any() or early["response_relative_score"].isna().any():
        raise RuntimeError("Early-season response construction produced missing values")
    if late["response_relative_score"].isna().any():
        raise RuntimeError("Late-season response construction produced missing values")

    coverage = (
        pd.concat(
            [
                early.assign(analysis_set="early_primary"),
                late.assign(analysis_set="late_method_harmonized_gate"),
            ],
            ignore_index=True,
        )
        .groupby(
            ["analysis_set", "source_family", "source_id", "year", "month", "method_group"],
            as_index=False,
            dropna=False,
        )
        .agg(
            grid_record_count=("grid_id", "size"),
            unique_grid_cells=("grid_id", "nunique"),
            positive_abundance_cells=("mean_euphausiid_abundance_m3", lambda x: int((x > 0).sum())),
            zero_abundance_cells=("mean_euphausiid_abundance_m3", lambda x: int((x == 0).sum())),
            longitude_min=("lon", "min"),
            longitude_max=("lon", "max"),
            latitude_min=("lat", "min"),
            latitude_max=("lat", "max"),
        )
    )
    coverage.to_csv(TABLES / "phase8s_model_data_coverage.csv", index=False)
    return evidence, early.reset_index(drop=True), late.reset_index(drop=True)


def sample_glorys(frame: pd.DataFrame, dataset: xr.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    for column in PHYSICAL_COLUMNS:
        output[column] = np.nan
    qa_rows = []
    names = {
        "thetao": "glorys_surface_temperature_c",
        "so": "glorys_surface_salinity",
        "uo": "glorys_eastward_current_ms",
        "vo": "glorys_northward_current_ms",
    }
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    surface = dataset.sel(depth=0.494, method="nearest")
    for (year, month), indices in output.groupby(["year", "month"]).groups.items():
        selected = surface[list(names)].sel(time=f"{int(year):04d}-{int(month):02d}-01", method="nearest")
        points = output.loc[indices]
        longitude = xr.DataArray(points["lon"].to_numpy(dtype=float), dims="point")
        latitude = xr.DataArray(points["lat"].to_numpy(dtype=float), dims="point")
        interpolated = selected.interp(longitude=longitude, latitude=latitude)
        fallback_counts = {}
        fallback_distances = {}
        for source_name, target_name in names.items():
            values = np.asarray(interpolated[source_name].values, dtype=float)
            missing = ~np.isfinite(values)
            fallback_counts[target_name] = int(missing.sum())
            maximum_fallback = 0.0
            if missing.any():
                field = np.asarray(selected[source_name].values, dtype=float)
                valid_rows, valid_columns = np.where(np.isfinite(field))
                valid_lon = selected["longitude"].values[valid_columns]
                valid_lat = selected["latitude"].values[valid_rows]
                valid_x, valid_y = transformer.transform(valid_lon, valid_lat)
                tree = cKDTree(np.column_stack([valid_x, valid_y]))
                query_x, query_y = transformer.transform(
                    points.loc[missing, "lon"].to_numpy(dtype=float),
                    points.loc[missing, "lat"].to_numpy(dtype=float),
                )
                distance, nearest = tree.query(np.column_stack([query_x, query_y]), k=1)
                values[missing] = field[valid_rows[nearest], valid_columns[nearest]]
                maximum_fallback = float(np.max(distance)) if len(distance) else 0.0
                if maximum_fallback > 25_000:
                    raise RuntimeError(
                        f"GLORYS nearest-ocean fallback exceeds 25 km for {year}-{month:02d} {source_name}"
                    )
            output.loc[indices, target_name] = values
            fallback_distances[target_name] = maximum_fallback
        qa_rows.append(
            {
                "year": int(year),
                "month": int(month),
                "record_count": int(len(points)),
                "fallback_record_count_max_across_variables": int(max(fallback_counts.values())),
                "maximum_fallback_distance_km": float(max(fallback_distances.values()) / 1000.0),
                "all_physical_values_complete": bool(
                    np.isfinite(output.loc[indices, list(PHYSICAL_COLUMNS)].to_numpy(dtype=float)).all()
                ),
            }
        )
    output["glorys_current_speed_ms"] = np.sqrt(
        np.square(output["glorys_eastward_current_ms"])
        + np.square(output["glorys_northward_current_ms"])
    )
    return output, pd.DataFrame(qa_rows)


def candidate_specs(model_type: str) -> list[ModelSpec]:
    if model_type == "mean_null":
        return [ModelSpec("mean_null")]
    if model_type == "hydrographic":
        return [ModelSpec(model_type, n_knots=3, alpha=alpha) for alpha in ALPHAS]
    return [ModelSpec(model_type, n_knots=knots, alpha=alpha) for knots in KNOTS for alpha in ALPHAS]


def inner_cv_rmse(
    frame: pd.DataFrame,
    response_column: str,
    block_column: str,
    spec: ModelSpec,
) -> float:
    predictions = np.full(len(frame), np.nan, dtype=float)
    for block in sorted(frame[block_column].unique()):
        train = frame.loc[frame[block_column].ne(block)]
        test = frame.loc[frame[block_column].eq(block)]
        model = RelativeSupportRegressor(spec, PHYSICAL_COLUMNS).fit(train, response_column)
        predictions[test.index.to_numpy()] = model.predict(test)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Inner spatial CV did not produce complete predictions")
    observed = frame[response_column].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(predictions - observed))))


def choose_spec(frame: pd.DataFrame, response_column: str, block_column: str, model_type: str) -> ModelSpec:
    scores = []
    reset = frame.reset_index(drop=True)
    for spec in candidate_specs(model_type):
        score = inner_cv_rmse(reset, response_column, block_column, spec)
        scores.append((score, spec.n_knots, spec.alpha, spec))
    return min(scores, key=lambda item: (item[0], item[1], item[2]))[-1]


def nested_spatial_cv(
    frame: pd.DataFrame,
    response_column: str,
    analysis_set: str,
    n_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy().reset_index(drop=True)
    data["spatial_block"] = assign_spatial_blocks(data, n_blocks=n_blocks, random_state=RANDOM_STATE)
    rows = []
    for model_type in MODEL_TYPES:
        for outer_block in sorted(data["spatial_block"].unique()):
            train = data.loc[data["spatial_block"].ne(outer_block)].copy().reset_index(drop=True)
            test = data.loc[data["spatial_block"].eq(outer_block)].copy()
            if model_type == "mean_null":
                selected = ModelSpec("mean_null")
            else:
                selected = choose_spec(train, response_column, "spatial_block", model_type)
            model = RelativeSupportRegressor(selected, PHYSICAL_COLUMNS).fit(train, response_column)
            predicted = model.predict(test)
            null_prediction = np.full(len(test), train[response_column].mean(), dtype=float)
            for position, (index, record) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "model_type": model_type,
                        "outer_spatial_block": int(outer_block),
                        "record_index": int(index),
                        "grid_id": record["grid_id"],
                        "source_family": record["source_family"],
                        "source_id": record["source_id"],
                        "year": int(record["year"]),
                        "month": int(record["month"]),
                        "observed_response": float(record[response_column]),
                        "predicted_response": float(predicted[position]),
                        "null_prediction": float(null_prediction[position]),
                        "selected_n_knots": selected.n_knots,
                        "selected_alpha": selected.alpha,
                    }
                )
    predictions = pd.DataFrame(rows)
    summaries = []
    for model_type, group in predictions.groupby("model_type"):
        metrics = predictive_metrics(
            group["observed_response"].to_numpy(),
            group["predicted_response"].to_numpy(),
            group["null_prediction"].to_numpy(),
        )
        fold_skill = []
        for _, fold in group.groupby("outer_spatial_block"):
            fold_metrics = predictive_metrics(
                fold["observed_response"].to_numpy(),
                fold["predicted_response"].to_numpy(),
                fold["null_prediction"].to_numpy(),
            )
            fold_skill.append(fold_metrics["rmse_skill"])
        summaries.append(
            {
                "analysis_set": analysis_set,
                "model_type": model_type,
                "record_count": int(len(group)),
                "spatial_block_count": int(group["outer_spatial_block"].nunique()),
                **metrics,
                "positive_skill_fold_fraction": float(np.mean(np.asarray(fold_skill) > 0)),
            }
        )
    return predictions, pd.DataFrame(summaries)


def full_data_spec(frame: pd.DataFrame, response_column: str, model_type: str, n_blocks: int) -> ModelSpec:
    data = frame.copy().reset_index(drop=True)
    data["spatial_block"] = assign_spatial_blocks(data, n_blocks=n_blocks, random_state=RANDOM_STATE)
    return choose_spec(data, response_column, "spatial_block", model_type)


def cross_year_validation(early: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for train_year, test_year in ((2017, 2018), (2018, 2017)):
        train = early.loc[early["year"].eq(train_year)].copy().reset_index(drop=True)
        test = early.loc[early["year"].eq(test_year)].copy().reset_index(drop=True)
        for model_type in MODEL_TYPES:
            if model_type == "mean_null":
                spec = ModelSpec("mean_null")
            else:
                spec = full_data_spec(train, "response_relative_score", model_type, n_blocks=4)
            model = RelativeSupportRegressor(spec, PHYSICAL_COLUMNS).fit(train, "response_relative_score")
            prediction = model.predict(test)
            null = np.full(len(test), train["response_relative_score"].mean(), dtype=float)
            metrics = predictive_metrics(test["response_relative_score"].to_numpy(), prediction, null)
            rows.append(
                {
                    "train_year": train_year,
                    "test_year": test_year,
                    "model_type": model_type,
                    "train_record_count": len(train),
                    "test_record_count": len(test),
                    "selected_n_knots": spec.n_knots,
                    "selected_alpha": spec.alpha,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def leave_source_family_validation(late: pd.DataFrame, model_type: str) -> tuple[pd.DataFrame, dict[str, float]]:
    data = late.copy().reset_index(drop=True)
    spec = full_data_spec(data, "response_relative_score", model_type, n_blocks=5)
    rows = []
    for held_out_family in sorted(data["source_family"].unique()):
        train = data.loc[data["source_family"].ne(held_out_family)]
        test = data.loc[data["source_family"].eq(held_out_family)]
        model = RelativeSupportRegressor(spec, PHYSICAL_COLUMNS).fit(train, "response_relative_score")
        prediction = model.predict(test)
        null = np.full(len(test), train["response_relative_score"].mean(), dtype=float)
        for position, (_, record) in enumerate(test.iterrows()):
            rows.append(
                {
                    "held_out_source_family": held_out_family,
                    "grid_id": record["grid_id"],
                    "year": int(record["year"]),
                    "month": int(record["month"]),
                    "observed_response": float(record["response_relative_score"]),
                    "predicted_response": float(prediction[position]),
                    "null_prediction": float(null[position]),
                    "model_type": model_type,
                    "selected_n_knots": spec.n_knots,
                    "selected_alpha": spec.alpha,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = predictive_metrics(
        predictions["observed_response"].to_numpy(),
        predictions["predicted_response"].to_numpy(),
        predictions["null_prediction"].to_numpy(),
    )
    return predictions, metrics


def select_best_nonnull(summary: pd.DataFrame, analysis_set: str) -> pd.Series:
    candidates = summary.loc[
        summary["analysis_set"].eq(analysis_set) & summary["model_type"].ne("mean_null")
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No non-null candidate models for {analysis_set}")
    return candidates.sort_values(
        ["rmse", "model_type"], ascending=[True, True], kind="mergesort"
    ).iloc[0]


def select_year_surface_model(summary: pd.DataFrame, analysis_set: str) -> pd.Series | None:
    """Choose the lowest-error model that passes all conservative surface gates."""

    candidates = summary.loc[
        summary["analysis_set"].eq(analysis_set)
        & summary["model_type"].ne("mean_null")
        & summary["rmse_skill"].gt(0)
        & summary["spearman_rho"].ge(0.30)
        & summary["positive_skill_fold_fraction"].ge(0.75)
    ].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["rmse", "model_type"], ascending=[True, True], kind="mergesort"
    ).iloc[0]


def bootstrap_surface(
    observations: pd.DataFrame,
    grid: pd.DataFrame,
    response_column: str,
    spec: ModelSpec,
    prediction_domain: np.ndarray,
    lower_anchor: float,
    upper_anchor: float,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = RelativeSupportRegressor(spec, PHYSICAL_COLUMNS).fit(observations, response_column)
    central_response = model.predict(grid)
    central_support = robust_support_scale(central_response, lower_anchor, upper_anchor)
    central_support[~prediction_domain] = 0.0

    data = observations.copy().reset_index(drop=True)
    data["bootstrap_block"] = assign_spatial_blocks(data, n_blocks=n_blocks, random_state=RANDOM_STATE)
    blocks = np.sort(data["bootstrap_block"].unique())
    rng = np.random.default_rng(RANDOM_STATE)
    draws = np.zeros((BOOTSTRAP_DRAWS, len(grid)), dtype=np.float32)
    for draw_index in range(BOOTSTRAP_DRAWS):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        pieces = [data.loc[data["bootstrap_block"].eq(block)] for block in sampled_blocks]
        bootstrap = pd.concat(pieces, ignore_index=True)
        fitted = RelativeSupportRegressor(spec, PHYSICAL_COLUMNS).fit(bootstrap, response_column)
        values = robust_support_scale(fitted.predict(grid), lower_anchor, upper_anchor)
        values[~prediction_domain] = 0.0
        draws[draw_index] = values.astype(np.float32)
    lower = np.quantile(draws, 0.025, axis=0)
    median = np.quantile(draws, 0.50, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    return central_support, lower, median, upper, draws


def build_early_surfaces(
    early: pd.DataFrame,
    grid: pd.DataFrame,
    dataset: xr.Dataset,
    cv_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spacing_rows = []
    surface_qa = []
    q95_values = []
    for year, group in early.groupby("year"):
        spacing = nearest_neighbor_quantiles(group)
        spacing_rows.append({"year": int(year), "unique_cells": group["grid_id"].nunique(), **spacing})
        q95_values.append(spacing["q95_m"])
    maximum_distance_m = math.ceil(max(q95_values) / 10_000.0) * 10_000.0
    spacing_frame = pd.DataFrame(spacing_rows)
    spacing_frame["shared_interpolation_limit_m"] = maximum_distance_m
    spacing_frame.to_csv(TABLES / "phase8s_early_station_spacing_and_interpolation_limit.csv", index=False)

    surface_outputs = []
    for year in (2017, 2018):
        observations = early.loc[early["year"].eq(year)].copy().reset_index(drop=True)
        grid_year = grid.copy()
        grid_year["year"] = year
        grid_year["month"] = 6
        grid_year, physical_qa = sample_glorys(grid_year, dataset)
        physical_qa.to_csv(TABLES / f"phase8s_glorys_grid_sampling_qa_{year}.csv", index=False)
        domain, nearest_distance = interpolation_domain(observations, grid_year, maximum_distance_m)

        analysis_set = f"early_{year}_log_abundance"
        best = select_year_surface_model(cv_summary, analysis_set)
        if best is None:
            raise RuntimeError(f"{year} did not pass the year-specific surface gate")
        spec = full_data_spec(
            observations,
            "response_log_abundance",
            str(best["model_type"]),
            n_blocks=4,
        )
        lower_anchor = float(observations["response_log_abundance"].quantile(0.05))
        upper_anchor = float(observations["response_log_abundance"].quantile(0.95))
        central, lower, median, upper, draws = bootstrap_surface(
            observations=observations,
            grid=grid_year,
            response_column="response_log_abundance",
            spec=spec,
            prediction_domain=domain,
            lower_anchor=lower_anchor,
            upper_anchor=upper_anchor,
            n_blocks=4,
        )
        output = grid_year[
            [
                "grid_id",
                "grid_x",
                "grid_y",
                "x",
                "y",
                "lon",
                "lat",
                "ocean_flag",
                *PHYSICAL_COLUMNS,
                "glorys_current_speed_ms",
            ]
        ].copy()
        output["target_year"] = year
        output["pelagic_evidence_domain"] = domain
        output["nearest_observation_distance_km"] = nearest_distance / 1000.0
        output["P_pelagic_central"] = central
        output["P_pelagic_bootstrap_q025"] = lower
        output["P_pelagic_bootstrap_median"] = median
        output["P_pelagic_bootstrap_q975"] = upper
        output["P_pelagic_bootstrap_interval_width"] = upper - lower
        output["observation_model_type"] = spec.model_type
        output["observation_model_n_knots"] = spec.n_knots
        output["observation_model_alpha"] = spec.alpha
        output["support_scale_lower_log_abundance_q05"] = lower_anchor
        output["support_scale_upper_log_abundance_q95"] = upper_anchor
        path = SURFACES / f"phase8s_observation_informed_pelagic_support_target{year}.csv"
        output.to_csv(path, index=False)
        np.savez_compressed(
            SURFACES / f"phase8s_observation_informed_pelagic_support_bootstrap_target{year}.npz",
            grid_id=output["grid_id"].to_numpy(dtype=str),
            support_draws=draws,
            random_seed=np.asarray([RANDOM_STATE], dtype=np.int64),
        )
        surface_outputs.append(output)
        surface_qa.append(
            {
                "target_year": year,
                "observation_count": len(observations),
                "observation_grid_cell_count": observations["grid_id"].nunique(),
                "prediction_domain_cell_count": int(domain.sum()),
                "prediction_domain_percent_of_grid": float(domain.mean() * 100.0),
                "shared_interpolation_limit_km": maximum_distance_m / 1000.0,
                "model_type": spec.model_type,
                "n_knots": spec.n_knots,
                "alpha": spec.alpha,
                "mean_support_inside_domain": float(central[domain].mean()),
                "median_bootstrap_interval_width_inside_domain": float(np.median((upper - lower)[domain])),
                "outside_domain_nonzero_support_cells": int((central[~domain] > 0).sum()),
            }
        )
    return pd.concat(surface_outputs, ignore_index=True), pd.DataFrame(surface_qa)


def build_gate_table(
    cv_summary: pd.DataFrame,
    cross_year: pd.DataFrame,
    late_leave_source_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for year in (2017, 2018):
        analysis_set = f"early_{year}_log_abundance"
        best = select_year_surface_model(cv_summary, analysis_set)
        passed = best is not None
        if best is None:
            reported = select_best_nonnull(cv_summary, analysis_set)
        else:
            reported = best
        rows.append(
            {
                "gate": f"year_specific_surface_{year}",
                "decision": "pass" if passed else "fail",
                "evidence": (
                    f"Selected blocked-CV model={reported['model_type']}; RMSE skill={reported['rmse_skill']:.3f}; "
                    f"Spearman rho={reported['spearman_rho']:.3f}; positive-skill folds="
                    f"{reported['positive_skill_fold_fraction']:.2f}."
                ),
                "consequence": "Eligible for year-specific corridor sensitivity only." if passed else "No corridor run.",
            }
        )

    transfer_best = (
        cross_year.loc[cross_year["model_type"].ne("mean_null")]
        .sort_values(["train_year", "test_year", "rmse"])
        .groupby(["train_year", "test_year"], as_index=False)
        .first()
    )
    transfer_pass = bool(
        (transfer_best["rmse_skill"] > 0).all() and (transfer_best["spearman_rho"] >= 0.30).all()
    )
    rows.append(
        {
            "gate": "cross_year_transfer_2017_2018",
            "decision": "pass" if transfer_pass else "fail",
            "evidence": "; ".join(
                f"{int(row.train_year)}->{int(row.test_year)} {row.model_type}: "
                f"skill={row.rmse_skill:.3f}, rho={row.spearman_rho:.3f}"
                for row in transfer_best.itertuples()
            ),
            "consequence": "Supports limited comparison between the two sampled years only.",
        }
    )
    rows.append(
        {
            "gate": "annual_2011_2019_dynamic_pelagic_surface",
            "decision": "fail",
            "evidence": (
                "Direct, method-comparable June abundance is available in 2 of 9 target years "
                "(2017 and 2018); cross-year transfer between two years cannot establish annual dynamics."
            ),
            "consequence": "Do not extrapolate pelagic support to 2011-2016 or 2019.",
        }
    )

    late_best = select_best_nonnull(cv_summary, "late_pooled_relative_score")
    late_pass = bool(
        late_best["rmse_skill"] > 0
        and late_best["spearman_rho"] > 0
        and late_leave_source_metrics["rmse_skill"] > 0
        and late_leave_source_metrics["spearman_rho"] > 0
    )
    rows.append(
        {
            "gate": "pooled_august_october_surface",
            "decision": "pass" if late_pass else "fail",
            "evidence": (
                f"Spatial-block best={late_best['model_type']}, skill={late_best['rmse_skill']:.3f}, "
                f"rho={late_best['spearman_rho']:.3f}; leave-source-family skill="
                f"{late_leave_source_metrics['rmse_skill']:.3f}, rho={late_leave_source_metrics['spearman_rho']:.3f}."
            ),
            "consequence": "Eligible for late-season corridor sensitivity." if late_pass else "No pooled late-season corridor run.",
        }
    )
    return pd.DataFrame(rows)


def write_summary(gates: pd.DataFrame, surface_qa: pd.DataFrame, cv_summary: pd.DataFrame) -> None:
    year_rows = surface_qa.sort_values("target_year")
    gate_lines = "\n".join(
        f"- **{row.gate}: {row.decision.upper()}**. {row.evidence} {row.consequence}"
        for row in gates.itertuples()
    )
    surface_lines = "\n".join(
        f"- {int(row.target_year)}: {int(row.observation_count)} observations, "
        f"{int(row.prediction_domain_cell_count)} supported grid cells, {row.model_type} model, "
        f"median bootstrap interval width {row.median_bootstrap_interval_width_inside_domain:.3f}."
        for row in year_rows.itertuples()
    )
    text = f"""# Phase 8S model audit summary

## Scientific question

Can the available euphausiid observations support an empirical spatial layer without treating heterogeneous surveys as interchangeable or extrapolating beyond sampled space and years?

## Decisions

{gate_lines}

## Year-specific June surfaces

{surface_lines}

The surface value is a relative support index. It is derived from observed euphausiid abundance and is not expressed in `kJ m^-2`; its strength in the benthic-pelagic utility combination must therefore remain an explicit sensitivity parameter. Cells outside the sampled convex hull or farther than the data-derived station-spacing limit receive no added support, which denotes absence of empirical support rather than inferred absence of euphausiids.

## Validation rule

Spatially blocked cross-validation was used to compare each fitted model with a training-fold mean null. A year-specific surface passed only when a non-null model improved RMSE, produced Spearman rho >= 0.30 and improved on the null in at least three of four spatial folds. Cross-year transfer was assessed separately, but only two directly sampled years were available and therefore no 2011-2019 annual pelagic surface was permitted. The pooled late-season gate additionally required positive leave-source-family transfer.

## Paper-use boundary

Only surfaces that pass the prediction gate are eligible for corridor integration.
"""
    (OUT / "phase8s_model_audit_summary.md").write_text(text, encoding="utf-8")
    cv_summary.to_csv(TABLES / "phase8s_spatial_block_cv_summary.csv", index=False)


def build_manifest(gates: pd.DataFrame) -> None:
    output_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path != MANIFEST)
    payload = {
        "phase": PHASE,
        "script": SCRIPT,
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (GRID, EVIDENCE, OBSERVATIONS, GLORYS)
        ],
        "model_contract": {
            "physical_covariates": list(PHYSICAL_COLUMNS),
            "candidate_model_types": list(MODEL_TYPES),
            "candidate_alphas": list(ALPHAS),
            "candidate_spatial_knots": list(KNOTS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "target_window": "June observations within the May-July analysis season",
            "response": "log1p abundance for year-specific surfaces; within-stratum relative score for transfer tests",
        },
        "gate_decisions": gates.to_dict(orient="records"),
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ],
        "established_output_modified": False,
        "manuscript_or_si_modified": False,
    }
    write_json(MANIFEST, payload)


def main() -> None:
    ensure_directories()
    set_stage("input_audit", "running")
    input_snapshot()
    source_decisions()
    _, early, late = load_model_data()
    dataset = xr.open_dataset(GLORYS)
    early, early_physical_qa = sample_glorys(early, dataset)
    late, late_physical_qa = sample_glorys(late, dataset)
    early.to_csv(TABLES / "phase8s_early_model_records_with_glorys.csv", index=False)
    late.to_csv(TABLES / "phase8s_late_model_records_with_glorys.csv", index=False)
    pd.concat(
        [
            early_physical_qa.assign(analysis_set="early"),
            late_physical_qa.assign(analysis_set="late"),
        ],
        ignore_index=True,
    ).to_csv(TABLES / "phase8s_glorys_observation_sampling_qa.csv", index=False)
    set_stage(
        "input_audit",
        "completed",
        early_record_count=len(early),
        late_record_count=len(late),
        warnings=[],
    )

    set_stage("model_validation", "running")
    prediction_frames = []
    summary_frames = []
    for year in (2017, 2018):
        predictions, summary = nested_spatial_cv(
            early.loc[early["year"].eq(year)],
            response_column="response_log_abundance",
            analysis_set=f"early_{year}_log_abundance",
            n_blocks=4,
        )
        prediction_frames.append(predictions)
        summary_frames.append(summary)
    pooled_predictions, pooled_summary = nested_spatial_cv(
        early,
        response_column="response_relative_score",
        analysis_set="early_pooled_relative_score",
        n_blocks=5,
    )
    prediction_frames.append(pooled_predictions)
    summary_frames.append(pooled_summary)
    late_predictions, late_summary = nested_spatial_cv(
        late,
        response_column="response_relative_score",
        analysis_set="late_pooled_relative_score",
        n_blocks=5,
    )
    prediction_frames.append(late_predictions)
    summary_frames.append(late_summary)
    cv_predictions = pd.concat(prediction_frames, ignore_index=True)
    cv_summary = pd.concat(summary_frames, ignore_index=True)
    cv_predictions.to_csv(TABLES / "phase8s_spatial_block_cv_predictions.csv", index=False)
    cross_year = cross_year_validation(early)
    cross_year.to_csv(TABLES / "phase8s_early_cross_year_transfer.csv", index=False)
    late_best = select_best_nonnull(cv_summary, "late_pooled_relative_score")
    leave_source, leave_source_metrics = leave_source_family_validation(late, str(late_best["model_type"]))
    leave_source.to_csv(TABLES / "phase8s_late_leave_source_family_predictions.csv", index=False)
    pd.DataFrame([leave_source_metrics]).to_csv(
        TABLES / "phase8s_late_leave_source_family_summary.csv", index=False
    )
    set_stage("model_validation", "completed", cv_prediction_rows=len(cv_predictions), warnings=[])

    set_stage("surface_construction", "running")
    grid = pd.read_csv(GRID, dtype={"grid_id": str})
    grid = grid.loc[grid["ocean_flag"].eq(1)].copy().reset_index(drop=True)
    _, surface_qa = build_early_surfaces(early, grid, dataset, cv_summary)
    surface_qa.to_csv(TABLES / "phase8s_early_surface_qa.csv", index=False)
    gates = build_gate_table(cv_summary, cross_year, leave_source_metrics)
    gates.to_csv(TABLES / "phase8s_model_integration_gates.csv", index=False)
    set_stage(
        "surface_construction",
        "completed",
        surface_count=len(surface_qa),
        gate_decisions=gates[["gate", "decision"]].to_dict(orient="records"),
        warnings=[],
    )

    write_summary(gates, surface_qa, cv_summary)
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    state["status"] = "completed"
    state["updated_utc"] = utc_now()
    write_json(STATUS, state)
    build_manifest(gates)
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
