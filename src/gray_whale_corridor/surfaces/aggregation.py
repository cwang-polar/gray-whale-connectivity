"""Cell-level aggregation helpers for surface construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CellAggregationConfig:
    """Settings for aggregating sample records to grid cells."""

    group_cols: tuple[str, ...] = ("grid_x", "grid_y", "grid_id")
    value_columns: tuple[str, ...] = ()
    sample_count_col: str | None = None


def aggregate_benthic_samples_to_cells(
    mapped_samples: pd.DataFrame,
    value_columns: tuple[str, ...],
    sample_count_col: str | None = None,
    group_cols: tuple[str, ...] = ("grid_x", "grid_y", "grid_id"),
) -> pd.DataFrame:
    """Aggregate mapped generic benthic sample values to cell summaries."""

    config = CellAggregationConfig(
        group_cols=group_cols,
        value_columns=value_columns,
        sample_count_col=sample_count_col,
    )
    return aggregate_samples_to_cells(mapped_samples, config)


def aggregate_samples_to_cells(
    mapped_samples: pd.DataFrame,
    config: CellAggregationConfig,
) -> pd.DataFrame:
    """Aggregate numeric sample columns by grid cell.

    The output includes row counts, optional summed sample counts, and sum/mean
    summaries for each configured value column.
    """

    _validate_aggregation_inputs(mapped_samples, config)
    df = mapped_samples.copy()
    for column in config.value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    named_aggregations = {
        "sample_rows": (config.group_cols[0], "size"),
    }
    if config.sample_count_col and config.sample_count_col in df.columns:
        df[config.sample_count_col] = pd.to_numeric(
            df[config.sample_count_col],
            errors="coerce",
        )
        named_aggregations["sample_count_sum"] = (config.sample_count_col, "sum")

    for column in config.value_columns:
        named_aggregations[f"{column}_sum"] = (column, "sum")
        named_aggregations[f"{column}_mean"] = (column, "mean")

    aggregated = (
        df.groupby(list(config.group_cols), dropna=False)
        .agg(**named_aggregations)
        .reset_index()
    )

    return aggregated


def _validate_aggregation_inputs(
    mapped_samples: pd.DataFrame,
    config: CellAggregationConfig,
) -> None:
    missing_group_cols = [
        column for column in config.group_cols if column not in mapped_samples.columns
    ]
    if missing_group_cols:
        raise ValueError(f"Missing grouping columns: {missing_group_cols}")

    missing_values = [
        column for column in config.value_columns if column not in mapped_samples.columns
    ]
    if missing_values:
        raise ValueError(f"Missing value columns: {missing_values}")
