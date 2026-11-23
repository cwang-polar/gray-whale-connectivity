"""Path-based data ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gray_whale_corridor.config import ProjectPaths


@dataclass(frozen=True)
class DataInventory:
    """A lightweight record of available input files."""

    prey_files: tuple[Path, ...]
    ice_files: tuple[Path, ...]
    ship_files: tuple[Path, ...]
    validation_files: tuple[Path, ...]


def collect_input_inventory(paths: ProjectPaths) -> DataInventory:
    """Collect raw input paths without interpreting scientific content yet."""

    return DataInventory(
        prey_files=_list_files(paths.raw_prey),
        ice_files=_list_files(paths.raw_ice),
        ship_files=_list_files(paths.raw_ship),
        validation_files=_list_files(paths.raw_validation),
    )


def _list_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(sorted(path for path in directory.iterdir() if path.is_file()))

