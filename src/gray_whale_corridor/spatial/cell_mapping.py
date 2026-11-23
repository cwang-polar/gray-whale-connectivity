"""Map point observations to analysis grid cells."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .grid import GridDefinition
from .projection import DEFAULT_SOURCE_CRS, project_coordinates


def cell_indices_from_xy(
    x_coord: Sequence[float] | np.ndarray,
    y_coord: Sequence[float] | np.ndarray,
    grid: GridDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert projected x/y coordinates to integer grid indices."""

    x_values = np.asarray(x_coord, dtype=float)
    y_values = np.asarray(y_coord, dtype=float)
    grid_x = np.floor((x_values - grid.origin_x) / grid.cell_size).astype(int)
    grid_y = np.floor((y_values - grid.origin_y) / grid.cell_size).astype(int)
    return grid_x, grid_y


def assign_points_to_cells(
    data: pd.DataFrame,
    grid: GridDefinition,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    source_crs: str = DEFAULT_SOURCE_CRS,
    keep_projected: bool = True,
    drop_outside: bool = True,
    id_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assign point records with coordinate columns to grid cells.

    The function performs only generic spatial bookkeeping: numeric coordinate
    coercion, projection, finite-value filtering, and cell assignment.
    """

    required = {lon_col, lat_col}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")

    mapped = data.copy()
    mapped[lon_col] = pd.to_numeric(mapped[lon_col], errors="coerce")
    mapped[lat_col] = pd.to_numeric(mapped[lat_col], errors="coerce")
    mapped = mapped.dropna(subset=[lon_col, lat_col]).copy()

    if source_crs == DEFAULT_SOURCE_CRS:
        valid_lonlat = (
            mapped[lon_col].between(-180, 180)
            & mapped[lat_col].between(-90, 90)
        )
        mapped = mapped[valid_lonlat].copy()

    if mapped.empty:
        return _empty_mapping_result(data, id_cols, keep_projected)

    x_coord, y_coord = project_coordinates(
        mapped[lon_col].to_numpy(),
        mapped[lat_col].to_numpy(),
        source_crs=source_crs,
        target_crs=grid.crs,
    )
    finite = np.isfinite(x_coord) & np.isfinite(y_coord)
    mapped = mapped.loc[finite].copy()
    x_coord = x_coord[finite]
    y_coord = y_coord[finite]

    if mapped.empty:
        return _empty_mapping_result(data, id_cols, keep_projected)

    grid_x, grid_y = cell_indices_from_xy(x_coord, y_coord, grid)
    mapped["grid_x"] = grid_x
    mapped["grid_y"] = grid_y
    mapped["grid_id"] = list(zip(mapped["grid_x"], mapped["grid_y"]))

    if keep_projected:
        mapped["x"] = x_coord
        mapped["y"] = y_coord

    if drop_outside:
        in_domain = (
            mapped["grid_x"].between(0, grid.cols - 1)
            & mapped["grid_y"].between(0, grid.rows - 1)
        )
        mapped = mapped[in_domain].copy()

    if id_cols is not None:
        selected_cols = [
            col
            for col in [*id_cols, lon_col, lat_col, "grid_x", "grid_y", "grid_id", "x", "y"]
            if col in mapped.columns
        ]
        mapped = mapped[selected_cols].copy()

    return mapped.reset_index(drop=True)


def _empty_mapping_result(
    data: pd.DataFrame,
    id_cols: Sequence[str] | None,
    keep_projected: bool,
) -> pd.DataFrame:
    columns = list(id_cols) if id_cols is not None else list(data.columns)
    columns.extend(["grid_x", "grid_y", "grid_id"])
    if keep_projected:
        columns.extend(["x", "y"])
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))

