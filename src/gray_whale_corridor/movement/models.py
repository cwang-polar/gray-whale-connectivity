"""Scaffold for movement and connectivity models."""

from __future__ import annotations

from dataclasses import dataclass

from gray_whale_corridor.surfaces import SurfacePlaceholders


@dataclass(frozen=True)
class MovementModelPlan:
    """Description of movement analyses to implement later."""

    candidate_models: tuple[str, ...]
    required_surface: str


def define_movement_model_plan(surfaces: SurfacePlaceholders) -> MovementModelPlan:
    """Return planned model families without running corridor algorithms yet."""

    return MovementModelPlan(
        candidate_models=("least_cost_path", "circuit_theory"),
        required_surface=surfaces.resistance_surface,
    )

