"""Inventory ArcticData/Kapsar monthly vessel traffic rasters for Phase 5A."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]
SHIP_ROOT = ROOT / "data/raw/ship/resource_map_doi_10_18739_A2NZ80R4J"
SHIP_DATA = SHIP_ROOT / "data"
OCEAN_GRID = ROOT / "data/processed/study_domain/ocean_mask_grid_10km_large_regional_bbox_v0.csv"
OUT_DIR = ROOT / "data/processed/ship"
MANIFEST = ROOT / "outputs/results_manifest.csv"
REGISTRY = ROOT / "docs/results_registry.md"
README = ROOT / "README.md"
CLAIMS = ROOT / "docs/paper_claims_evidence_matrix.md"
PHASE_MEMO = ROOT / "docs/phase_memos/phase5a_ship_data_inventory.md"
PAPER_MEMO = ROOT / "docs/paper_use_decisions/phase5a_ship_data_inventory_paper_use.md"

TARGET_YEARS = range(2015, 2020)
ALL_YEARS = range(2015, 2021)
MONTHS = range(1, 13)
FOCUS_MONTHS = {5, 6, 7}
FILENAME_RE = re.compile(r"Raster_(?P<year>\d{4})_(?P<month>\d{2})_(?P<vessel_type>[^_]+)_10km\.tif$")
NODATA_SENTINEL = -3.4e38

# GeoTIFF keys indicate NAD83 Albers Equal Area with standard parallels 55/65,
# latitude of origin 50, and central meridian -154.
SHIP_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)


def parse_name(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {"year": None, "month": None, "vessel_type": None}
    data = match.groupdict()
    return {"year": int(data["year"]), "month": int(data["month"]), "vessel_type": data["vessel_type"]}


def tif_metadata(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value
        nodata = tags.get("GDAL_NODATA")
        width = int(page.shape[1])
        height = int(page.shape[0])
        left = float(tie[3])
        top = float(tie[4])
        res_x = float(scale[0])
        res_y = float(scale[1])
        right = left + width * res_x
        bottom = top - height * res_y
        return {
            "width": width,
            "height": height,
            "dtype": str(page.dtype),
            "resolution_x": res_x,
            "resolution_y": res_y,
            "bounds_tuple": (left, bottom, right, top),
            "bounds": f"{left:.3f},{bottom:.3f},{right:.3f},{top:.3f}",
            "nodata_value": nodata.value if nodata else str(NODATA_SENTINEL),
        }


def read_array(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray().astype("float64")
    arr[arr <= NODATA_SENTINEL / 10] = np.nan
    return arr


def load_ocean_grid() -> pd.DataFrame:
    grid = pd.read_csv(OCEAN_GRID)
    if "ocean_flag" in grid.columns:
        grid = grid[grid["ocean_flag"] == 1].copy()
    return grid


def transform_ocean_points(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), SHIP_CRS, always_xy=True)
    xs, ys = transformer.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    return np.asarray(xs), np.asarray(ys)


def points_within_bounds(xs: np.ndarray, ys: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    left, bottom, right, top = bounds
    return (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)


def sample_nearest(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, meta: dict) -> np.ndarray:
    left, bottom, right, top = meta["bounds_tuple"]
    res_x = meta["resolution_x"]
    res_y = meta["resolution_y"]
    col = np.floor((xs - left) / res_x).astype(int)
    row = np.floor((top - ys) / res_y).astype(int)
    valid = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    values = np.full(xs.shape, np.nan, dtype="float64")
    values[valid] = arr[row[valid], col[valid]]
    return values


def safe_stats(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    if not finite.any():
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "missing_count": int(len(values))}
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "missing_count": int((~finite).sum()),
    }


def inventory_files(files: list[Path], ocean_x: np.ndarray, ocean_y: np.ndarray) -> pd.DataFrame:
    rows = []
    for path in sorted(files):
        name = parse_name(path)
        meta = tif_metadata(path)
        within = points_within_bounds(ocean_x, ocean_y, meta["bounds_tuple"])
        coverage_pct = within.mean() * 100
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "file_format": ".tif",
                "year": name["year"],
                "month": name["month"],
                "vessel_type": name["vessel_type"],
                "crs": "NAD83 Albers Equal Area, proj=aea lat_1=55 lat_2=65 lat_0=50 lon_0=-154",
                "bounds": meta["bounds"],
                "resolution_x": meta["resolution_x"],
                "resolution_y": meta["resolution_y"],
                "unit_guess": "meters of vessel-track length per raster cell per month",
                "covers_study_domain": coverage_pct >= 99.5,
                "covers_ocean_grid": coverage_pct >= 99.5,
                "usable_for_may_july_2015_2019": bool(
                    name["year"] in TARGET_YEARS
                    and name["month"] in FOCUS_MONTHS
                    and coverage_pct >= 99.5
                ),
                "notes": f"shape={meta['height']}x{meta['width']}; dtype={meta['dtype']}; nodata={meta['nodata_value']}; ocean_grid_bounds_coverage={coverage_pct:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def completeness_table(inventory: pd.DataFrame) -> pd.DataFrame:
    vessel_types = sorted(inventory["vessel_type"].dropna().unique())
    rows = []
    for year in ALL_YEARS:
        for month in MONTHS:
            for vessel_type in vessel_types:
                subset = inventory[
                    (inventory["year"] == year)
                    & (inventory["month"] == month)
                    & (inventory["vessel_type"] == vessel_type)
                ]
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "vessel_type": vessel_type,
                        "file_count": int(len(subset)),
                        "usable_file_available": bool(
                            len(subset) >= 1
                            and subset["covers_ocean_grid"].fillna(False).all()
                            and subset["usable_for_may_july_2015_2019"].any()
                            if year in TARGET_YEARS and month in FOCUS_MONTHS
                            else len(subset) >= 1
                        ),
                    }
                )
    return pd.DataFrame(rows)


def representative_paths(inventory: pd.DataFrame) -> list[Path]:
    rels: set[str] = set()
    criteria = [(2015, 5, None), (2017, 6, None), (2019, 7, None)]
    for year, month, vessel_type in criteria:
        subset = inventory[(inventory["year"] == year) & (inventory["month"] == month)]
        if vessel_type:
            subset = subset[subset["vessel_type"] == vessel_type]
        for rel in subset.sort_values("vessel_type")["file_path"].head(1):
            rels.add(rel)
    for vessel_type in sorted(inventory["vessel_type"].unique()):
        subset = inventory[
            (inventory["year"] == 2019) & (inventory["month"] == 7) & (inventory["vessel_type"] == vessel_type)
        ]
        if len(subset):
            rels.add(subset.iloc[0]["file_path"])
    return [ROOT / rel for rel in sorted(rels)]


def coverage_qa(paths: Iterable[Path], ocean_x: np.ndarray, ocean_y: np.ndarray, ocean_count: int) -> pd.DataFrame:
    rows = []
    for path in paths:
        name = parse_name(path)
        meta = tif_metadata(path)
        within = points_within_bounds(ocean_x, ocean_y, meta["bounds_tuple"])
        arr = read_array(path)
        values = sample_nearest(arr, ocean_x, ocean_y, meta)
        stats = safe_stats(values)
        rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "year": name["year"],
                "month": name["month"],
                "vessel_type": name["vessel_type"],
                "crs": "NAD83 Albers Equal Area",
                "resolution_x": meta["resolution_x"],
                "resolution_y": meta["resolution_y"],
                "ocean_cell_count": ocean_count,
                "ocean_cells_within_raster_bounds": int(within.sum()),
                "percent_ocean_cells_within_raster_bounds": float(within.mean() * 100),
                "sampled_min": stats["min"],
                "sampled_max": stats["max"],
                "sampled_mean": stats["mean"],
                "sampled_missing_or_nodata_cells": stats["missing_count"],
                "sampled_missing_or_nodata_percent": float(stats["missing_count"] / ocean_count * 100),
                "notes": "Nearest-cell sampling used for inventory QA only; future Phase 5B should preserve raster units during resampling.",
            }
        )
    return pd.DataFrame(rows)


def append_manifest(paths: list[str]) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_registry(summary: dict) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_claims() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_readme() -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def write_memos(summary: dict, qa: pd.DataFrame) -> None:
    """Omit historical manuscript/registry maintenance in release workflows."""
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SHIP_DATA.glob("Raster_*_10km.tif"))
    ocean = load_ocean_grid()
    ocean_x, ocean_y = transform_ocean_points(ocean)

    inventory = inventory_files(files, ocean_x, ocean_y)
    inventory.to_csv(OUT_DIR / "ship_data_inventory_v0.csv", index=False)

    completeness = completeness_table(inventory)
    completeness.to_csv(OUT_DIR / "ship_data_completeness_v0.csv", index=False)

    qa = coverage_qa(representative_paths(inventory), ocean_x, ocean_y, len(ocean))
    qa.to_csv(OUT_DIR / "ship_coverage_qa_v0.csv", index=False)

    ext_counts = {}
    for path in SHIP_ROOT.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower() or "<no_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    summary = {
        "raster_count": int(len(files)),
        "xml_count": int(ext_counts.get(".xml", 0)),
        "txt_count": int(ext_counts.get(".txt", 0)),
        "years": ", ".join(map(str, sorted(inventory["year"].unique()))),
        "months": ", ".join(f"{m:02d}" for m in sorted(inventory["month"].unique())),
        "vessel_types": ", ".join(sorted(inventory["vessel_type"].unique())),
        "focus_complete": bool(
            completeness[
                completeness["year"].isin(TARGET_YEARS)
                & completeness["month"].isin(FOCUS_MONTHS)
            ]["usable_file_available"].all()
        ),
    }
    (OUT_DIR / "ship_data_inventory_summary_v0.json").write_text(json.dumps(summary, indent=2))

    outputs = [
        "data/processed/ship/ship_data_inventory_v0.csv",
        "data/processed/ship/ship_data_completeness_v0.csv",
        "data/processed/ship/ship_coverage_qa_v0.csv",
        "data/processed/ship/ship_data_inventory_summary_v0.json",
        "docs/phase_memos/phase5a_ship_data_inventory.md",
        "docs/paper_use_decisions/phase5a_ship_data_inventory_paper_use.md",
        "scripts/inventory_ship_data_v0.py",
    ]
    append_manifest(outputs)
    write_registry(summary)
    write_claims()
    write_readme()
    write_memos(summary, qa)

    print(json.dumps(summary, indent=2))
    print(qa.to_string(index=False))


if __name__ == "__main__":
    main()
