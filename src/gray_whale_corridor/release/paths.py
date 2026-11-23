"""Explicit frozen inputs and fresh per-task outputs; no scientific imports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .control import CONFIG, SOFTWARE, confined_path, digest, load_catalog

RUNS_ROOT = SOFTWARE.parent / "runs"


def load_policy(filename="path_policy.json"):
    load_catalog()
    if filename not in ("path_policy.json", "ierp_input_preparation.json"):
        raise ValueError("Unknown V8 path-policy file")
    return json.loads((CONFIG / filename).read_text(encoding="utf-8"))


@dataclass
class TaskPaths:
    task: str
    data_root: Path
    run_id: str
    policy_file: str = "path_policy.json"
    prepared_ierp_run: str | None = None
    _reserved: bool = field(default=False, init=False, repr=False)
    prepared_files: dict = field(default_factory=dict, init=False, repr=False)
    prepared_hashes: dict = field(default_factory=dict, init=False, repr=False)
    forbidden_read_paths: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        self.policy = load_policy(self.policy_file)
        if self.task not in self.policy["tasks"]:
            raise ValueError(f"Unknown V8 task: {self.task}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.run_id):
            raise ValueError("Use a new run ID containing only letters, numbers, underscores or hyphens")
        self.data_root = Path(self.data_root).resolve(strict=True)
        marker = confined_path(self.data_root, "source_data/v8_figure_table_index.csv")
        if not marker.is_file() or digest(marker) != self.policy["data_index_sha256"]:
            raise ValueError("Expected the staged V8 dataset index; no fallback to the original project")
        self.run_root = confined_path(RUNS_ROOT, self.run_id)
        self.task_root = confined_path(self.run_root, self.task)
        for protected in (self.data_root, SOFTWARE.resolve()):
            if self.run_root.is_relative_to(protected) or protected.is_relative_to(self.run_root):
                raise ValueError("Output and input/software roots must be disjoint")
        if self.task_root.exists():
            raise FileExistsError(f"Task output already exists; choose a new run ID: {self.task_root}")
        self.contract = self.policy["tasks"][self.task]
        if self.prepared_ierp_run is not None and self.task != 'S11':
            raise ValueError('Prepared IERP input is supported only for S11')
        if self.prepared_ierp_run is not None or self.task == 'prepare_ierp':
            from .ierp_inputs import registry
            self.forbidden_read_paths = {confined_path(self.data_root, spec['legacy_dataset_path'])
                                         for spec in registry()['outputs'].values()}
        if self.prepared_ierp_run is not None:
            from .ierp_inputs import prepared_files
            if self.run_id == self.prepared_ierp_run:
                raise ValueError('Use a distinct run ID for the S11 render')
            self.prepared_files = prepared_files(RUNS_ROOT, self.prepared_ierp_run)
            self.prepared_hashes = {key: digest(path) for key,path in self.prepared_files.items()}
            original = self.contract['aliases']['observations']
            self.contract = {**self.contract,
                'inputs': [p for p in self.contract['inputs'] if p != original],
                'aliases': {k:v for k,v in self.contract['aliases'].items() if k != 'observations'}}

    @property
    def prepared_read_paths(self):
        return set(self.prepared_files.values())

    def require_task(self, *identifiers):
        if self.task not in identifiers:
            raise ValueError(f"This adapter does not support task {self.task}")

    def input_path(self, relative, *, verify=True):
        relative = str(relative)
        if relative not in self.contract["inputs"]:
            raise ValueError(f"Input is not registered for {self.task}: {relative}")
        spec = self.policy["files"][relative]
        if spec.get("hold"):
            raise PermissionError(f"Input remains on hold: {relative}")
        path = confined_path(self.data_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing V8 input; no fallback: {relative}")
        if verify and digest(path) != spec["sha256"]:
            raise ValueError(f"Frozen input checksum differs: {relative}")
        return path

    def input(self, alias):
        if alias == 'observations' and self.prepared_files:
            path = self.prepared_files['s11']
            if digest(path) != self.prepared_hashes['s11']:
                raise ValueError('Prepared S11 observations changed')
            return path
        return self.input_path(self.contract["aliases"][alias])

    def input_glob(self, relative_directory, pattern):
        directory = PurePosixPath(relative_directory)
        found = []
        for name in self.contract["inputs"]:
            relative = PurePosixPath(name)
            if relative.is_relative_to(directory) and relative.relative_to(directory).match(pattern):
                found.append(self.input_path(name))
        return sorted(found)

    def output(self, relative):
        return confined_path(self.task_root, relative)

    def package(self, relative):
        return confined_path(SOFTWARE, relative)

    def label(self, path):
        resolved = Path(path).resolve()
        for key, prepared in self.prepared_files.items():
            if resolved == prepared:
                return 'prepared:' + key
        for prefix, root in (("data", self.data_root), ("software", SOFTWARE.resolve()),
                             ("output", self.task_root)):
            if resolved.is_relative_to(root):
                return f"{prefix}:{resolved.relative_to(root).as_posix()}"
        raise ValueError(f"Path lies outside this task's roots: {path}")

    def resolve_label(self, label):
        prefix, relative = label.split(":", 1)
        if prefix == 'prepared':
            path = self.prepared_files[relative]
            if digest(path) != self.prepared_hashes[relative]:
                raise ValueError('Prepared input changed')
            return path
        if prefix == "data":
            return self.input_path(relative)
        if prefix == "software":
            return self.package(relative)
        if prefix == "output":
            return self.output(relative)
        raise ValueError(f"Unknown path namespace: {prefix}")

    def snapshot(self):
        snapshot = {self.label(self.input_path(name)): self.policy["files"][name]["sha256"]
                    for name in self.contract["inputs"]}
        for key, path in self.prepared_files.items():
            value = digest(path)
            if value != self.prepared_hashes[key]:
                raise ValueError('Prepared input changed: ' + key)
            snapshot['prepared:' + key] = value
        return snapshot

    def reserve_output(self):
        """For a later authorized adapter call, never for the read-only CLI."""
        if self._reserved:
            raise FileExistsError("This task output was already reserved")
        if self.contract["hold_reasons"]:
            raise PermissionError("; ".join(self.contract["hold_reasons"]))
        self.snapshot()
        # Recheck parents before creating anything; never reuse a task directory.
        if confined_path(RUNS_ROOT, f"{self.run_id}/{self.task}") != self.task_root:
            raise ValueError("Output path changed after planning")
        self.task_root.mkdir(parents=True, exist_ok=False)
        self._reserved = True
        with (self.task_root / "path_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump({"task": self.task, "run_id": self.run_id,
                       "data_root": str(self.data_root), "task_root": str(self.task_root),
                       "inputs": self.snapshot(), "parameters": "unchanged frozen V8 scope",
                       "note": "Path reservation is not scientific validation."}, stream, indent=2)
        return self.task_root

    def require_reserved(self):
        if not self._reserved or not self.task_root.is_dir():
            raise RuntimeError("The adapter must reserve its fresh output before importing scientific libraries")

    def figure_reference(self, identifier):
        refs = json.loads((CONFIG / "frozen_figure_references.json").read_text())
        return next(item for item in refs["figures"] if item["id"] == identifier)

    def preview(self):
        entries = []
        for name in self.contract["inputs"]:
            held = self.policy["files"][name].get("hold", False)
            entry = {"relative_path": name, "held": held}
            if held:
                entry["status"] = "held"
            else:
                try:
                    entry.update(path=str(self.input_path(name)), status="present_and_hash_verified")
                except (ValueError, FileNotFoundError) as error:
                    entry.update(status="blocked", reason=str(error))
            entries.append(entry)
        return {"task": self.task, "adapter_status": self.contract["adapter_status"],
                "prepared_ierp_run": self.prepared_ierp_run,
                "prepared_inputs": {key: {"path": str(path), "sha256": self.prepared_hashes[key]}
                                    for key,path in self.prepared_files.items()},
                "adapter_status_describes_input_routing": True,
                "adapter": self.contract.get("adapter"), "inputs": entries,
                "output_root": str(self.task_root),
                "output_sections": {key: str(self.output(key)) for key in self.policy["output_sections"]},
                "hold_reasons": self.contract["hold_reasons"],
                "remaining": self.contract["remaining"], "directories_created": False,
                "execution_enabled": False, "execution_available": self.policy["execution_enabled"],
                "scientific_modules_imported": False}
