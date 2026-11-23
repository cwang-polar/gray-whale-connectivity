"""Coordinate projection helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pyproj import Transformer


DEFAULT_SOURCE_CRS = "EPSG:4326"
DEFAULT_TARGET_CRS = "EPSG:3338"


def project_coordinates(
    longitude: Sequence[float] | np.ndarray,
    latitude: Sequence[float] | np.ndarray,
    source_crs: str = DEFAULT_SOURCE_CRS,
    target_crs: str = DEFAULT_TARGET_CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude coordinate arrays into a target CRS.

    Parameters are intentionally explicit so future workflows can swap in
    other source or target coordinate systems without changing calling code.
    """

    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x_coord, y_coord = transformer.transform(lon, lat)
    return np.asarray(x_coord, dtype=float), np.asarray(y_coord, dtype=float)

