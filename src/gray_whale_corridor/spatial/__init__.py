"""Shared spatial scaffolding for the corridor pipeline."""

from .cell_mapping import assign_points_to_cells, cell_indices_from_xy
from .grid import (
    GridDefinition,
    build_full_grid,
    grid_from_coordinates,
    grid_from_corner_coordinates,
)
from .projection import project_coordinates

__all__ = [
    "GridDefinition",
    "assign_points_to_cells",
    "build_full_grid",
    "cell_indices_from_xy",
    "grid_from_coordinates",
    "grid_from_corner_coordinates",
    "project_coordinates",
]
