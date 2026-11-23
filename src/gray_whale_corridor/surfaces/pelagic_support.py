"""Reusable models for observation-informed pelagic-support surfaces.

The helpers in this module deliberately separate the observation model from
the corridor model. They estimate relative spatial support only where survey
data support interpolation; they do not convert euphausiid abundance to the
energy units of the benthic prey surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a penalized relative-support regression."""

    model_type: str
    n_knots: int = 4
    alpha: float = 1.0
    degree: int = 2


class RelativeSupportRegressor:
    """Penalized spatial/hydrographic regression with transparent features."""

    def __init__(self, spec: ModelSpec, physical_columns: Iterable[str] = ()) -> None:
        allowed = {"mean_null", "hydrographic", "spatial", "spatial_hydrographic"}
        if spec.model_type not in allowed:
            raise ValueError(f"Unknown model type: {spec.model_type}")
        self.spec = spec
        self.physical_columns = tuple(physical_columns)
        self.coordinate_scaler: StandardScaler | None = None
        self.physical_scaler: StandardScaler | None = None
        self.x_spline: SplineTransformer | None = None
        self.y_spline: SplineTransformer | None = None
        self.regressor: Ridge | None = None
        self.null_mean: float | None = None

    @property
    def uses_spatial(self) -> bool:
        return self.spec.model_type in {"spatial", "spatial_hydrographic"}

    @property
    def uses_hydrography(self) -> bool:
        return self.spec.model_type in {"hydrographic", "spatial_hydrographic"}

    def _spatial_features(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        if fit:
            self.coordinate_scaler = StandardScaler().fit(coordinates)
        if self.coordinate_scaler is None:
            raise RuntimeError("Coordinate scaler has not been fitted")
        scaled = self.coordinate_scaler.transform(coordinates)
        if fit:
            self.x_spline = SplineTransformer(
                n_knots=self.spec.n_knots,
                degree=self.spec.degree,
                include_bias=False,
                extrapolation="constant",
            ).fit(scaled[:, [0]])
            self.y_spline = SplineTransformer(
                n_knots=self.spec.n_knots,
                degree=self.spec.degree,
                include_bias=False,
                extrapolation="constant",
            ).fit(scaled[:, [1]])
        if self.x_spline is None or self.y_spline is None:
            raise RuntimeError("Spatial spline basis has not been fitted")
        x_basis = self.x_spline.transform(scaled[:, [0]])
        y_basis = self.y_spline.transform(scaled[:, [1]])
        interaction = np.einsum("ij,ik->ijk", x_basis, y_basis).reshape(len(frame), -1)
        return np.column_stack([x_basis, y_basis, interaction])

    def _hydrographic_features(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.physical_columns:
            raise ValueError("Hydrographic model requested without physical columns")
        values = frame.loc[:, self.physical_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Hydrographic predictors contain missing or non-finite values")
        if fit:
            self.physical_scaler = StandardScaler().fit(values)
        if self.physical_scaler is None:
            raise RuntimeError("Hydrographic scaler has not been fitted")
        return self.physical_scaler.transform(values)

    def _features(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        pieces: list[np.ndarray] = []
        if self.uses_spatial:
            pieces.append(self._spatial_features(frame, fit=fit))
        if self.uses_hydrography:
            pieces.append(self._hydrographic_features(frame, fit=fit))
        if not pieces:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(pieces)

    def fit(self, frame: pd.DataFrame, response_column: str) -> "RelativeSupportRegressor":
        response = frame[response_column].to_numpy(dtype=float)
        if not np.isfinite(response).all():
            raise ValueError("Response contains missing or non-finite values")
        if self.spec.model_type == "mean_null":
            self.null_mean = float(np.mean(response))
            return self
        features = self._features(frame, fit=True)
        self.regressor = Ridge(alpha=self.spec.alpha, fit_intercept=True)
        self.regressor.fit(features, response)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.model_type == "mean_null":
            if self.null_mean is None:
                raise RuntimeError("Null model has not been fitted")
            return np.full(len(frame), self.null_mean, dtype=float)
        if self.regressor is None:
            raise RuntimeError("Model has not been fitted")
        return self.regressor.predict(self._features(frame, fit=False))


def assign_spatial_blocks(
    frame: pd.DataFrame,
    n_blocks: int,
    random_state: int = 8675309,
) -> np.ndarray:
    """Assign compact deterministic spatial clusters for blocked validation."""

    if len(frame) < n_blocks:
        raise ValueError("Number of observations is smaller than number of spatial blocks")
    xy = StandardScaler().fit_transform(frame[["x", "y"]].to_numpy(dtype=float))
    return KMeans(n_clusters=n_blocks, random_state=random_state, n_init=50).fit_predict(xy)


def predictive_metrics(observed: np.ndarray, predicted: np.ndarray, null: np.ndarray) -> dict[str, float]:
    """Return paired prediction metrics against a fold-specific null model."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    null = np.asarray(null, dtype=float)
    model_error = predicted - observed
    null_error = null - observed
    rmse = float(np.sqrt(np.mean(np.square(model_error))))
    null_rmse = float(np.sqrt(np.mean(np.square(null_error))))
    mae = float(np.mean(np.abs(model_error)))
    null_mae = float(np.mean(np.abs(null_error)))
    if np.unique(observed).size > 1 and np.unique(predicted).size > 1:
        rho = float(spearmanr(observed, predicted).statistic)
    else:
        rho = float("nan")
    return {
        "rmse": rmse,
        "null_rmse": null_rmse,
        "rmse_skill": float(1.0 - rmse / null_rmse) if null_rmse > 0 else float("nan"),
        "predictive_r2": float(1.0 - np.sum(np.square(model_error)) / np.sum(np.square(null_error)))
        if np.sum(np.square(null_error)) > 0
        else float("nan"),
        "mae": mae,
        "null_mae": null_mae,
        "mae_skill": float(1.0 - mae / null_mae) if null_mae > 0 else float("nan"),
        "spearman_rho": rho,
    }


def within_stratum_relative_score(
    frame: pd.DataFrame,
    abundance_column: str,
    stratum_columns: Iterable[str],
) -> pd.Series:
    """Map zeros to zero and rank positive catches within comparable strata."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    columns = list(stratum_columns)
    for _, group in frame.groupby(columns, dropna=False, sort=False):
        abundance = pd.to_numeric(group[abundance_column], errors="coerce")
        valid = abundance.notna()
        result.loc[group.index[valid & abundance.eq(0)]] = 0.0
        positive = valid & abundance.gt(0)
        if positive.any():
            ranks = abundance.loc[positive].rank(method="average", pct=True)
            result.loc[ranks.index] = ranks.to_numpy(dtype=float)
    return result


def interpolation_domain(
    observations: pd.DataFrame,
    prediction_grid: pd.DataFrame,
    maximum_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict predictions to the sampled hull and a nearest-sample distance."""

    observed_xy = observations[["x", "y"]].drop_duplicates().to_numpy(dtype=float)
    grid_xy = prediction_grid[["x", "y"]].to_numpy(dtype=float)
    if len(observed_xy) < 3:
        raise ValueError("At least three unique sampling locations are needed")
    hull = ConvexHull(observed_xy)
    inside_hull = Delaunay(observed_xy[hull.vertices]).find_simplex(grid_xy) >= 0
    nearest_distance = cKDTree(observed_xy).query(grid_xy, k=1)[0]
    return inside_hull & (nearest_distance <= maximum_distance_m), nearest_distance


def nearest_neighbor_quantiles(observations: pd.DataFrame) -> dict[str, float]:
    """Summarize station spacing for a data-derived interpolation limit."""

    xy = observations[["x", "y"]].drop_duplicates().to_numpy(dtype=float)
    if len(xy) < 2:
        raise ValueError("At least two unique sampling locations are needed")
    distances = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return {
        "minimum_m": float(np.min(distances)),
        "median_m": float(np.median(distances)),
        "q90_m": float(np.quantile(distances, 0.90)),
        "q95_m": float(np.quantile(distances, 0.95)),
        "maximum_m": float(np.max(distances)),
    }


def robust_support_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Scale predictions to [0, 1] using fixed observation-scale anchors."""

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Robust scaling anchors must be finite and ordered")
    return np.clip((np.asarray(values, dtype=float) - lower) / (upper - lower), 0.0, 1.0)
