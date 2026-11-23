"""Scaffold for corridor and bottleneck metrics."""

from __future__ import annotations

from dataclasses import dataclass

from gray_whale_corridor.movement import MovementModelPlan


@dataclass(frozen=True)
class MetricPlan:
    """Description of metrics expected from future analysis."""

    planned_metrics: tuple[str, ...]
    based_on_models: tuple[str, ...]


def define_metric_plan(model_plan: MovementModelPlan) -> MetricPlan:
    """Define intended outputs without computing ecological metrics yet."""

    return MetricPlan(
        planned_metrics=(
            "corridor_width",
            "bottleneck_intensity",
            "scenario_difference",
            "validation_overlap",
        ),
        based_on_models=model_plan.candidate_models,
    )

