"""Scaffold for prey, ice, shipping, and resistance surface construction."""

from __future__ import annotations

from dataclasses import dataclass

from gray_whale_corridor.data import DataInventory


@dataclass(frozen=True)
class SurfacePlaceholders:
    """Names of surfaces expected from the future scientific workflow."""

    prey_surface: str
    ice_modifier: str
    ship_modifier: str
    resistance_surface: str


def build_surface_placeholders(inventory: DataInventory) -> SurfacePlaceholders:
    """Define expected surface products without implementing raster logic yet."""

    _ = inventory
    return SurfacePlaceholders(
        prey_surface="pending_prey_surface",
        ice_modifier="pending_ice_modifier",
        ship_modifier="pending_ship_modifier",
        resistance_surface="pending_functional_resistance_surface",
    )

