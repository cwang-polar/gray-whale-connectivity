"""Load and validate project configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the pipeline."""

    raw_prey: Path
    raw_ice: Path
    raw_ship: Path
    raw_validation: Path
    processed: Path
    outputs: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline settings."""

    project_root: Path
    paths: ProjectPaths
    analysis: dict[str, Any]


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> PipelineConfig:
    """Load a JSON config file and resolve paths relative to the project root."""

    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    with path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    raw_paths = raw_config.get("paths", {})
    project_paths = ProjectPaths(
        raw_prey=_resolve_path(root, raw_paths["raw_prey"]),
        raw_ice=_resolve_path(root, raw_paths["raw_ice"]),
        raw_ship=_resolve_path(root, raw_paths["raw_ship"]),
        raw_validation=_resolve_path(root, raw_paths["raw_validation"]),
        processed=_resolve_path(root, raw_paths["processed"]),
        outputs=_resolve_path(root, raw_paths["outputs"]),
    )

    return PipelineConfig(
        project_root=root,
        paths=project_paths,
        analysis=raw_config.get("analysis", {}),
    )


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path

