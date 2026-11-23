"""Inspect study metadata or explicitly run one configured task in fresh outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

SOFTWARE = Path(__file__).resolve().parents[3]
CONFIG = SOFTWARE / "config/ncc_v8"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def confined_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Expected a package-relative path: {relative}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes package root: {relative}")
    return path


def load_catalog() -> dict:
    catalog = json.loads((CONFIG / "entrypoints.json").read_text(encoding="utf-8"))
    if not isinstance(catalog["execution_enabled"], bool):
        raise ValueError("Invalid execution authorization flag")
    if digest(CONFIG / "scope.json") != catalog["scope_sha256"]:
        raise ValueError("The frozen scope changed; review and version it explicitly")
    if digest(CONFIG / "frozen_figure_references.json") != catalog["figure_references_sha256"]:
        raise ValueError("The frozen figure references changed")
    for relative, expected in catalog["configuration_sha256"].items():
        if digest(confined_path(CONFIG, relative)) != expected:
            raise ValueError(f"The V8 configuration changed: {relative}")
    return catalog


def inspect_task(task_id: str, data_root: Path, *, hashes: bool = False) -> dict:
    catalog = load_catalog()
    if task_id not in catalog["tasks"]:
        raise ValueError(f"Unknown V8 task: {task_id}")
    task = catalog["tasks"][task_id]
    sources = []
    for spec in task["inputs"]:
        path = confined_path(data_root, spec["path"])
        held = spec.get("hold", False)
        result = {**spec, "exists": path.is_file(), "status": "held" if held else "missing"}
        if path.is_file() and not held:
            result["status"] = "present"
            if hashes and spec.get("sha256"):
                result["sha256_matches"] = digest(path) == spec["sha256"]
                if not result["sha256_matches"]:
                    result["status"] = "checksum_mismatch"
        sources.append(result)
    code = []
    for spec in task["implementation"]:
        path = confined_path(SOFTWARE, spec["path"])
        report = {**spec, "exists": path.is_file()}
        if path.is_file():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=spec["path"])
            names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            report["declared_symbols_present"] = all(name in names for name in spec.get("symbols", []))
        code.append(report)
    acceptance_path = SOFTWARE / "environment/runtime_acceptance_clean_v1.json"
    if not acceptance_path.is_file():
        acceptance_path = SOFTWARE / "environment/runtime_acceptance_v1.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.is_file() else {}
    for name in ("runtime_acceptance_archived_figures_v1.json", "runtime_acceptance_archived_context_v1.json",
                 "runtime_acceptance_corridors_v1.json", "runtime_acceptance_late_season_v1.json",
                 "runtime_acceptance_pelagic_v1.json", "runtime_acceptance_idw_v1.json",
                 "runtime_acceptance_idw_v2.json", "runtime_acceptance_benthic_surfaces_v1.json",
                 "runtime_acceptance_fig1_labels_v3.json"):
        supplement = SOFTWARE / "environment" / name
        if supplement.is_file():
            acceptance.setdefault("tasks", {}).update(json.loads(supplement.read_text())["tasks"])
    return {
        "task": task_id,
        "question": task["question"],
        "parameter_contract": task["parameter_contract"],
        "inputs": sources,
        "implementation": code,
        "input_and_scope_limits": task["blockers"],
        "latest_local_acceptance": acceptance.get("tasks", {}).get(task_id, {"status": "held_or_not_tested"}),
        "all_listed_unheld_inputs_present": all(x["status"] == "present" for x in sources if not x.get("hold")),
        "has_held_inputs": any(x.get("hold", False) for x in sources),
        "ready_for_execution": False,
        "execution_enabled": catalog["execution_enabled"],
        "scientific_modules_imported": False,
        "note": "Read-only listed-input check. Local acceptance is separate; this is not an environment preflight on a new machine.",
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List explicit V8 tasks; no default analysis")
    preparation = commands.add_parser("prepare-ierp", help="Prepare adopted IERP inputs only; no fitting, corridors or figures")
    preparation.add_argument("--data-root", type=Path, default=SOFTWARE.parent / "dataset")
    preparation.add_argument("--run-id", required=True)
    preparation.add_argument("--plan", action="store_true", help="Inspect paths without running preparation")
    plan = commands.add_parser("plan", help="Read configuration and inspect paths only")
    plan.add_argument("task", help="One exact task ID from list; no all task")
    plan.add_argument("--data-root", type=Path, default=SOFTWARE.parent / "dataset")
    plan.add_argument("--hashes", action="store_true", help="Verify listed input bytes without computing metrics")
    paths = commands.add_parser("paths", help="Preview inputs and fresh outputs; create nothing")
    paths.add_argument("task")
    paths.add_argument("--data-root", type=Path, default=SOFTWARE.parent / "dataset")
    paths.add_argument("--run-id", required=True)
    paths.add_argument("--prepared-ierp-run", help="S11 only: read accepted locally prepared observations")
    run = commands.add_parser("run", help="Execute one prepared non-held task; refuse existing outputs")
    run.add_argument("task")
    run.add_argument("--data-root", type=Path, default=SOFTWARE.parent / "dataset")
    run.add_argument("--run-id", required=True)
    run.add_argument("--prepared-ierp-run", help="S11 only: read accepted locally prepared observations")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog()
        if args.command == "prepare-ierp":
            if args.plan:
                from .paths import TaskPaths
                print(json.dumps(TaskPaths("prepare_ierp", args.data_root, args.run_id,
                    policy_file="ierp_input_preparation.json").preview(), indent=2))
                return 0
            from .runtime import execute
            return execute("prepare_ierp", args.data_root.resolve(), args.run_id, preparation=True)
        if args.command == "run":
            if not catalog["execution_enabled"]:
                raise ValueError("Task execution has not been enabled")
            from .runtime import execute
            return execute(args.task, args.data_root.resolve(), args.run_id, prepared_ierp_run=args.prepared_ierp_run)
        if args.command == "list":
            payload = {"execution_enabled": catalog["execution_enabled"], "tasks": {key: value["question"] for key, value in catalog["tasks"].items()}}
        elif args.command == "plan":
            payload = inspect_task(args.task, args.data_root.resolve(), hashes=args.hashes)
        else:
            from .paths import TaskPaths
            payload = TaskPaths(args.task, args.data_root, args.run_id, prepared_ierp_run=args.prepared_ierp_run).preview()
    except (ValueError, OSError, SyntaxError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0
