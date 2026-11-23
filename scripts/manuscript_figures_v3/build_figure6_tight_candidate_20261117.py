#!/usr/bin/env python3
"""Export the established Figure 6 with compact outer canvas margins."""

from __future__ import annotations

# Use the package entry point; standalone execution is disabled.
if __name__ == "__main__":
    raise SystemExit("Use software/v8.py list or plan; analysis/render execution is not enabled.")

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import build_figure5_6_split_20261117 as baseline
import matplotlib.pyplot as plt
import pandas as pd

ROOT = baseline.ROOT
OUT = baseline.OUT / "figure_6"


class TightExport:
    def __init__(self, version, pad_mm, *, release_paths=None):
        self.release_paths = release_paths
        self.out = release_paths.output("figures/Fig6") if release_paths is not None else OUT
        if release_paths is not None:
            for child in ["candidates", "qa"]:
                (self.out / child).mkdir(parents=True, exist_ok=False)
        self.stem = f"gray_whale_figure6_recurrence_exposure_candidate_v{version}"
        self.bundle = self.out / f"layout_tight_20261117_v{version}"
        self.pad_mm = pad_mm
        self.checks = {}
        self.sources = {}
        self.exports = []
        if self.bundle.exists() or (self.out / "candidates" / f"{self.stem}.png").exists():
            raise FileExistsError("Choose an unused candidate version")
        self.bundle.mkdir()

    def relative(self, path):
        return self.release_paths.label(path) if self.release_paths is not None else baseline.rel(path)

    def source(self, path, role):
        path = Path(path)
        self.sources[self.relative(path)] = {"role": role, "sha256": baseline.digest(path)}
        return path

    def export(self, key, fig):
        assert key == "6"
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(self.pad_mm / 25.4)
        self.checks["export_size_mm"] = [tight.width * 25.4, tight.height * 25.4]
        self.checks["outer_padding_mm"] = self.pad_mm
        self.checks["original_size_mm"] = (fig.get_size_inches() * 25.4).tolist()
        for ext in ["png", "pdf", "svg", "tiff"]:
            target = self.out / "candidates" / f"{self.stem}.{ext}"
            fig.savefig(target, dpi=600, bbox_inches=tight, facecolor="white")
            self.exports.append(self.relative(target))
        fig.savefig(self.out / "qa" / f"{self.stem}_review.png", dpi=180,
                    bbox_inches=tight, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--pad-mm", type=float, default=1.0)
    args = parser.parse_args()
    if args.version < 2 or args.pad_mm <= 0:
        raise ValueError("Use a new candidate version and positive outer padding")
    protected = baseline.snapshot()
    protected.pop(baseline.rel(Path(__file__).resolve()), None)
    protected[baseline.rel(Path(baseline.__file__).resolve())] = baseline.digest(Path(baseline.__file__))
    run = TightExport(args.version, args.pad_mm)
    frames = {}
    for name in ["panel_C_unchanged_recurrence", "panel_D_unchanged_exposure",
                 "panel_C_survey_union_cells", "panel_D_survey_union_cells"]:
        source = baseline.OLD / "source_data" / f"{baseline.OLD_STEM}_{name}.csv"
        frames[name] = pd.read_csv(run.source(source, "Unchanged Figure 6 source data"))
    baseline.workflow.style()
    baseline.figure6(run, frames)
    changed = [p for p, checksum in protected.items() if baseline.digest(ROOT / p) != checksum]
    run.checks.update(protected_file_count=len(protected), protected_files_changed=changed,
                      editable_svg_text="<text" in (OUT / "candidates" / f"{run.stem}.svg").read_text(),
                      embedded_pdf_fonts=b"/FontFile2" in (OUT / "candidates" / f"{run.stem}.pdf").read_bytes(),
                      map_extent_changed=False, models_run=False, word_files_modified=False)
    (run.bundle / "run_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": baseline.rel(Path(__file__).resolve()), "script_sha256": baseline.digest(Path(__file__)),
        "sources": run.sources, "outputs": run.exports, "checks": run.checks,
        "protected_input_hashes": protected,
    }, indent=2))
    (run.bundle / "README.md").write_text(
        "# Figure 6 compact-margin candidate\n\n"
        "Uses the established drawing function and frozen sources. Only outer canvas margins "
        "are reduced, to 1 mm beyond the artist bounds. Map extents, cells, classes, exposure "
        "values, survey unions, legends and colorbar are unchanged.\n")
    if changed or not run.checks["editable_svg_text"] or not run.checks["embedded_pdf_fonts"]:
        raise RuntimeError(run.checks)
    print(json.dumps({"bundle": baseline.rel(run.bundle), "checks": run.checks}, indent=2))


if __name__ == "__main__":
    main()
