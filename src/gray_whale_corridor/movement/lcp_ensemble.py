"""Reusable least-cost-path ensemble helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra


@dataclass(frozen=True)
class EnsemblePath:
    source_index: int
    destination_index: int
    cost: float
    node_indices: tuple[int, ...]


def build_weighted_adjacency(edge_pairs: list[tuple[int, int, float]], resistance: np.ndarray) -> csr_matrix:
    """Build a symmetric sparse graph from cell resistance and edge distance."""
    rows: list[int] = []
    cols: list[int] = []
    costs: list[float] = []
    for i, j, distance_km in edge_pairs:
        cost = float((resistance[i] + resistance[j]) / 2.0 * distance_km)
        rows.extend((i, j))
        cols.extend((j, i))
        costs.extend((cost, cost))
    return csr_matrix((costs, (rows, cols)), shape=(len(resistance), len(resistance)))


def reconstruct_path(predecessors: np.ndarray, source: int, destination: int) -> tuple[int, ...]:
    path = [int(destination)]
    node = int(destination)
    for _ in range(len(predecessors) + 1):
        if node == source:
            return tuple(reversed(path))
        node = int(predecessors[node])
        if node < 0:
            return ()
        path.append(node)
    return ()


def run_ensemble(adjacency: csr_matrix, topology: Any) -> tuple[list[EnsemblePath], np.ndarray]:
    """Run all source-destination pairs and count cell use."""
    distances, predecessors = sparse_dijkstra(
        adjacency,
        directed=False,
        indices=topology.source_indices,
        return_predecessors=True,
    )
    cell_counts = np.zeros(adjacency.shape[0], dtype=int)
    paths: list[EnsemblePath] = []
    for source_position, source_index in enumerate(topology.source_indices):
        for destination_index in topology.destination_indices:
            cost = distances[source_position, destination_index]
            if not np.isfinite(cost):
                continue
            nodes = reconstruct_path(predecessors[source_position], source_index, destination_index)
            if not nodes:
                continue
            cell_counts[np.asarray(nodes, dtype=int)] += 1
            paths.append(
                EnsemblePath(
                    source_index=int(source_index),
                    destination_index=int(destination_index),
                    cost=float(cost),
                    node_indices=nodes,
                )
            )
    return paths, cell_counts


def path_length_km(node_indices: tuple[int, ...], grid: pd.DataFrame) -> float:
    xy = grid.iloc[list(node_indices)][["x", "y"]].to_numpy(dtype=float)
    if len(xy) < 2:
        return 0.0
    return float(np.sqrt(np.square(np.diff(xy, axis=0)).sum(axis=1)).sum() / 1000.0)
