#!/usr/bin/env python3
"""Export Figure 3 and 4 candidates without in-panel interval explanations."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageStat

import build_phase8t_robustness_candidates_v1 as figures


FIG3_STEM = "gray_whale_figure3_joint_ice_anchors_prey_labels_annotation_free_candidate_v4"
FIG4_STEM = "gray_whale_figure4_vessel_alpha_annotation_free_candidate_v2"
SUFFIXES = (".pdf", ".svg", ".png", ".tiff")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_exports(files: list[Path], forbidden_text: str) -> dict:
    by_suffix = {path.suffix: path for path in files}
    if set(by_suffix) != set(SUFFIXES):
        raise RuntimeError("Incomplete figure export bundle")
    if not by_suffix[".pdf"].read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("Invalid PDF export")
    if forbidden_text.encode() in by_suffix[".svg"].read_bytes():
        raise RuntimeError("In-panel explanation remains in SVG")
    with Image.open(by_suffix[".png"]) as png:
        size = png.size
        nonblank = max(ImageStat.Stat(png.convert("RGB")).stddev) > 2.0
    with Image.open(by_suffix[".tiff"]) as tiff:
        if tiff.size != size:
            raise RuntimeError("PNG and TIFF dimensions differ")
    if not nonblank:
        raise RuntimeError("Blank figure export")
    return {
        "files": [str(path.relative_to(figures.ROOT)) for path in files],
        "raster_size_px": size,
        "raster_nonblank": nonblank,
    }


def main() -> None:
    protected = [
        figures.CANDIDATE / f"{stem}{suffix}"
        for stem in (
            "gray_whale_figure3_joint_ice_anchors_prey_labels_candidate_v3",
            "gray_whale_figure4_vessel_alpha_sensitivity_candidate_v1",
        )
        for suffix in SUFFIXES
    ]
    before = {str(path): digest(path) for path in protected}

    fig3_files = figures.build_figure3(
        comparison_path=figures.PHASE
        / "ice_joint_anchor_v2/joint_ice_anchor_prey_comparison_metrics.csv",
        primary_setting="primary_lower_0_15_upper_0_55",
        stem_name=FIG3_STEM,
        source_name="figure3_joint_ice_anchors_source_data_v4.csv",
        row_labels=("Prey", "Prey + ice"),
        show_range_annotation=False,
    )
    fig4_files = figures.build_figure4(
        stem_name=FIG4_STEM,
        show_range_annotation=False,
        write_source_data=False,
    )

    if before != {str(path): digest(path) for path in protected}:
        raise RuntimeError("A protected earlier candidate was changed")
    manifest = {
        "figure_contract": "Preserve primary estimates and parameter ranges; move interval explanations to captions.",
        "figure3": check_exports(fig3_files, "whiskers: ice-threshold range"),
        "figure4": check_exports(fig4_files, "shading on vessel scenarios: alpha range"),
        "protected_candidates_unchanged": True,
    }
    manifest_path = figures.QA / "phase8t_annotation_free_figures3_4_candidate_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {FIG3_STEM} and {FIG4_STEM}")


if __name__ == "__main__":
    main()
