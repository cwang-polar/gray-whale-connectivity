"""Explicit, isolated execution of the prepared configured study adapters."""

from __future__ import annotations

import importlib
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .control import SOFTWARE, digest
from .paths import RUNS_ROOT, TaskPaths


ALLOWED = {
    **{name: "archived_figures:render_archived_figure" for name in
       ("Fig1", "Fig2", "Fig3", "Fig4", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11", "S12", "S13")},
    "Fig5": "figure_adapters:render_fig5",
    "Fig6": "figure_adapters:render_fig6",
    "S8": "figure_adapters:render_s8",
    **{f"TableS{i}": "table_adapters:export_table" for i in range(1, 6)},
    "annual_rates": "annual_v8:run_annual_rates",
    "annual_correlation": "annual_v8:run_annual_correlation",
    "spatial_matching": "spatial_v8:run_spatial_matching",
    "sensitivity_survey_year": "spatial_v8:run_survey_year_sensitivity",
    "primary_corridors": "corridors_v8:run_primary",
    "sensitivity_ice": "corridors_v8:run_ice",
    "sensitivity_vessel": "corridors_v8:run_vessel",
    "late_season": "late_season_v8:run_late_season",
    "pelagic_support": "pelagic_v8:run_pelagic_support",
    "sensitivity_idw": "idw_v8:run_idw",
    "prey_processing": "prey_v8:run_prey_surfaces",
}

PREPARATION_ALLOWED = {"prepare_ierp": "ierp_v8:prepare_ierp"}


class AccessGuard:
    """Catch accidental Python I/O escape; not a native-code security sandbox."""

    def __init__(self, paths):
        self.paths = paths
        self.reads = set()
        self.violations = []
        self.project = next((p for p in SOFTWARE.parents if (p / "AGENTS.md").is_file()), None)
        prefix = Path(sys.prefix).resolve()
        # A clean local venv is readable, never writable, during scientific runs.
        self.environment = prefix if sys.prefix != sys.base_prefix and (prefix / "pyvenv.cfg").is_file() else None

    def path(self, value):
        if isinstance(value, (str, bytes, os.PathLike)):
            return Path(os.fsdecode(value)).absolute().resolve()
        return None

    def deny(self, event, path):
        message = f"Blocked {event} outside this task's contract: {path}"
        self.violations.append(message)
        raise PermissionError(message)

    def write(self, event, value, *, mkdir=False):
        path = self.path(value)
        if path is None:
            return
        if path.is_relative_to(self.paths.task_root):
            return
        if mkdir and path in (RUNS_ROOT.resolve(), self.paths.run_root):
            return
        self.deny(event, path)

    def __call__(self, event, args):
        if event == "open":
            value, mode, flags = args
            path = self.path(value)
            if path is None:
                return
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                self.write(event, value)
            elif path in getattr(self.paths, 'forbidden_read_paths', set()):
                self.deny('legacy IERP observation-table read', path)
            elif path in getattr(self.paths, 'prepared_read_paths', set()):
                self.reads.add(str(path))
            elif path.is_relative_to(self.paths.data_root) or path.is_relative_to(SOFTWARE):
                self.reads.add(str(path))
            elif path.is_relative_to(self.paths.task_root):
                pass
            elif self.environment and path.is_relative_to(self.environment):
                pass
            elif self.project and path.is_relative_to(self.project):
                self.deny("original-project read", path)
        elif event == "os.mkdir":
            self.write(event, args[0], mkdir=True)
        elif event in ("os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate"):
            self.write(event, args[0])
        elif event in ("os.rename", "os.link"):
            self.write(event, args[0])
            self.write(event, args[1])
        elif event == "os.symlink":
            self.deny(event, args[1])
        elif event in ("socket.connect", "socket.connect_ex", "socket.bind", "os.system"):
            self.deny(event, args)
        elif event == "subprocess.Popen":
            # Matplotlib discovers installed fonts; no workflow shell commands.
            if Path(os.fsdecode(args[0])).name not in ("fc-list", "fc-match"):
                self.deny(event, args[0])


def execute(task, data_root, run_id, *, preparation=False, prepared_ierp_run=None):
    allowed = PREPARATION_ALLOWED if preparation else ALLOWED
    if task not in allowed:
        raise PermissionError(f"Task is not enabled for execution: {task}")
    paths = TaskPaths(task, data_root, run_id,
        policy_file="ierp_input_preparation.json" if preparation else "path_policy.json",
        prepared_ierp_run=prepared_ierp_run)
    if paths.contract["hold_reasons"]:
        raise PermissionError("; ".join(paths.contract["hold_reasons"]))
    target = "gray_whale_corridor.release." + allowed[task]
    if paths.contract.get("adapter") != target:
        raise ValueError("Adapter differs from the registered V8 path contract")
    before = paths.snapshot()
    guard = AccessGuard(paths)
    sys.dont_write_bytecode = True
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(paths.output("cache/matplotlib"))
    os.environ["XDG_CACHE_HOME"] = str(paths.output("cache/xdg"))
    sys.addaudithook(guard)
    started = time.monotonic()
    status, error = "completed", None
    original_reader = None
    try:
        import geopandas
        # Match the reader used in the accepted local run, regardless of Fiona availability.
        original_reader = geopandas.read_file
        geopandas.read_file = partial(original_reader, engine="pyogrio")
        module_name, symbol = target.split(":")
        function = getattr(importlib.import_module(module_name), symbol)
        function(paths)
        if paths.snapshot() != before:
            raise RuntimeError("Frozen task inputs changed")
    except Exception:
        status, error = "failed", traceback.format_exc()
    finally:
        if original_reader is not None:
            geopandas.read_file = original_reader
    report = {
        "task": task, "run_id": run_id, "status": status,
        "prepared_ierp_run": prepared_ierp_run,
        "elapsed_seconds": time.monotonic() - started,
        "adapter": target, "inputs": before,
        "error": error, "blocked_accesses": guard.violations,
        "read_files": sorted(guard.reads),
        "python": sys.version, "executable": sys.executable,
        "geopandas_io_engine": "pyogrio",
        "scientific_parameters_changed": False,
        "validation_note": "Completion is not figure-equivalence certification; see separate comparison report.",
    }
    if paths._reserved:
        qa = paths.output("qa")
        qa.mkdir(exist_ok=True)
        (qa / "execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "completed" else 1
