"""Grid construction utilities for projected analysis grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd

from .projection import DEFAULT_SOURCE_CRS, DEFAULT_TARGET_CRS, project_coordinates


@dataclass(frozen=True)
class GridDefinition:
    """Definition of a regular projected analysis grid."""

    origin_x: float
    origin_y: float
    cols: int
    rows: int
    cell_size: float
    crs: str = DEFAULT_TARGET_CRS

    @property
    def max_x(self) -> float:
        return self.origin_x + self.cols * self.cell_size

    @property
    def max_y(self) -> float:
        return self.origin_y + self.rows * self.cell_size


def grid_from_corner_coordinates(
    corners: Iterable[tuple[float, float]],
    cell_size: float,
    source_crs: str = DEFAULT_SOURCE_CRS,
    target_crs: str = DEFAULT_TARGET_CRS,
    coordinate_order: str = "lat_lon",
) -> GridDefinition:
    """Build a grid definition from study-area corner coordinates.

    ``coordinate_order`` can be ``"lat_lon"`` or ``"lon_lat"``. The grid origin
    is the minimum projected x/y coordinate, and rows/columns are rounded up so
    the projected corner bounding box is fully covered.
    """

    corner_list = list(corners)
    if not corner_list:
        raise ValueError("At least one corner coordinate is required.")

    if coordinate_order == "lat_lon":
        latitudes = [lat for lat, _lon in corner_list]
        longitudes = [lon for _lat, lon in corner_list]
    elif coordinate_order == "lon_lat":
        longitudes = [lon for lon, _lat in corner_list]
        latitudes = [lat for _lon, lat in corner_list]
    else:
        raise ValueError("coordinate_order must be 'lat_lon' or 'lon_lat'.")

    x_coord, y_coord = project_coordinates(
        longitudes,
        latitudes,
        source_crs=source_crs,
        target_crs=target_crs,
    )

    origin_x = float(x_coord.min())
    origin_y = float(y_coord.min())
    cols = math.ceil((float(x_coord.max()) - origin_x) / cell_size)
    rows = math.ceil((float(y_coord.max()) - origin_y) / cell_size)

    return GridDefinition(
        origin_x=origin_x,
        origin_y=origin_y,
        cols=cols,
        rows=rows,
        cell_size=float(cell_size),
        crs=target_crs,
    )


def grid_from_coordinates(
    longitude: Iterable[float],
    latitude: Iterable[float],
    cell_size: float,
    source_crs: str = DEFAULT_SOURCE_CRS,
    target_crs: str = DEFAULT_TARGET_CRS,
    padding: float = 0.0,
) -> GridDefinition:
    """Build a grid definition from all sample coordinates.

    This avoids under-covering a projected study extent when a geographic
    bounding box bends under projection.
    """

    x_coord, y_coord = project_coordinates(
        list(longitude),
        list(latitude),
        source_crs=source_crs,
        target_crs=target_crs,
    )
    origin_x = float(x_coord.min()) - padding
    origin_y = float(y_coord.min()) - padding
    max_x = float(x_coord.max()) + padding
    max_y = float(y_coord.max()) + padding
    cols = max(1, math.ceil((max_x - origin_x) / cell_size))
    rows = max(1, math.ceil((max_y - origin_y) / cell_size))

    return GridDefinition(
        origin_x=origin_x,
        origin_y=origin_y,
        cols=cols,
        rows=rows,
        cell_size=float(cell_size),
        crs=target_crs,
    )


def build_full_grid(grid: GridDefinition, include_centers: bool = True) -> pd.DataFrame:
    """Return a DataFrame with one row per grid cell."""

    records = [
        {"grid_x": grid_x, "grid_y": grid_y}
        for grid_y in range(grid.rows)
        for grid_x in range(grid.cols)
    ]
    full_grid = pd.DataFrame.from_records(records)
    full_grid["grid_id"] = list(zip(full_grid["grid_x"], full_grid["grid_y"]))

    if include_centers:
        half_cell = grid.cell_size / 2.0
        full_grid["x"] = grid.origin_x + (full_grid["grid_x"] * grid.cell_size) + half_cell
        full_grid["y"] = grid.origin_y + (full_grid["grid_y"] * grid.cell_size) + half_cell

    return full_grid
