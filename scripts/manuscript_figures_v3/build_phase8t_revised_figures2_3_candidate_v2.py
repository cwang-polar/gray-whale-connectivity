#!/usr/bin/env python3
"""Export isolated Figure 2 and 3 v2 candidates from completed sensitivity runs."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageStat

import build_phase8t_robustness_candidates_v1 as figures


FIG2_SETTINGS = ("k4_p2", "k8_p1", "primary_k8_p2", "k8_p3")
FIG3_COMPARISON = figures.PHASE / "ice_joint_anchor_v2/joint_ice_anchor_prey_comparison_metrics.csv"
FIG2_STEM = "gray_whale_figure2_idw_k_le_8_candidate_v2"
FIG3_STEM = "gray_whale_figure3_joint_ice_anchors_candidate_v2"


def check_exports(files: list[Path]) -> dict:
    by_suffix = {path.suffix: path for path in files}
    if set(by_suffix) != {".pdf", ".svg", ".png", ".tiff"}:
        raise RuntimeError("Incomplete figure export bundle")
    if not by_suffix[".pdf"].read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("Invalid PDF header")
    if b"<svg" not in by_suffix[".svg"].read_bytes()[:4096]:
        raise RuntimeError("Invalid SVG header")
    with Image.open(by_suffix[".png"]) as image:
        width, height = image.size
        nonblank = max(ImageStat.Stat(image.convert("RGB")).stddev) > 2.0
    with Image.open(by_suffix[".tiff"]) as image:
        if image.size != (width, height):
            raise RuntimeError("PNG and TIFF dimensions differ")
    if not nonblank:
        raise RuntimeError("Blank raster export")
    return {
        "files": [str(path.relative_to(figures.ROOT)) for path in files],
        "width_px": width,
        "height_px": height,
        "raster_nonblank": nonblank,
        "interval_type": "parameter range, not confidence interval",
    }


def main() -> None:
    if not FIG3_COMPARISON.is_file():
        raise FileNotFoundError("Run the joint-anchor sensitivity analysis before exporting Figure 3")
    protected = [
        figures.CANDIDATE / f"gray_whale_figure{number}_{name}_candidate_v1.pdf"
        for number, name in ((2, "idw_parameter_sensitivity"), (3, "ice_threshold_sensitivity"))
    ]
    hashes_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    fig2_files = figures.build_figure2(
        allowed_settings=FIG2_SETTINGS,
        stem_name=FIG2_STEM,
        source_name="figure2_idw_k_le_8_source_data_v2.csv",
        show_range_annotation=False,
    )
    fig3_files = figures.build_figure3(
        comparison_path=FIG3_COMPARISON,
        primary_setting="primary_lower_0_15_upper_0_55",
        stem_name=FIG3_STEM,
        source_name="figure3_joint_ice_anchors_source_data_v2.csv",
    )
    hashes_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if hashes_before != hashes_after:
        raise RuntimeError("Earlier figure candidates were altered")
    manifest = {
        "figure2": {**check_exports(fig2_files), "settings": FIG2_SETTINGS, "max_neighbors": 8},
        "figure3": {
            **check_exports(fig3_files),
            "settings": ["strict (0.10, 0.45)", "primary (0.15, 0.55)", "relaxed (0.20, 0.65)"],
        },
        "earlier_candidates_unchanged": True,
    }
    path = figures.QA / "phase8t_revised_figures2_3_candidate_v2_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {FIG2_STEM} and {FIG3_STEM}; earlier candidates unchanged", flush=True)


if __name__ == "__main__":
    main()
