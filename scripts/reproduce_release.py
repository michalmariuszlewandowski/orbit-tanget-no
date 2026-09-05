#!/usr/bin/env python
"""Plan or run experiments from the release registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otno.config import apply_dotted_overrides, load_config
from run_matrix import _command as train_command
from run_matrix import _jobs as train_jobs

STAGES = ("data", "train", "evaluate", "reports", "figures")
TRAIN_FILES = (
    "config.yaml", "meta.json", "rng_state_initial.pt", "rng_state_final.pt",
    "train_metrics.jsonl", "val_metrics.jsonl", "test_metrics.json",
    "checkpoints/best.pt",
)
METRICS = (
    "relative_l2", "orbit_ood_relative_l2", "equivariance_defect_relative",
    "latency_ms_per_sample", "best_val_relative_l2", "parameters",
)


@dataclass
class Step:
    name: str
    stage: str
    command: list[str]
    requires: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    protected_dir: str | None = None
    sha256: str | None = None
    immutable: bool = False
    expected_config: dict[str, Any] | None = None
    fresh_report_inputs: list[str] = field(default_factory=list)
    evaluations: list[tuple[dict[str, Any], str]] = field(default_factory=list)


def _path(root: Path, value: str) -> Path:
    """Release paths must be portable and stay inside the checkout."""
    path = (root / value).resolve()
    if "\\" in value or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a repository-relative forward-slash path: {value}")
    return path


def load_registry(root: Path = ROOT, path: str = "configs/release.yaml") -> dict[str, Any]:
    registry = yaml.safe_load(_path(root, path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("release_version") != 1:
        raise ValueError("Unsupported release registry version")
    if not registry.get("suites") or not registry.get("datasets"):
        raise ValueError("Release registry requires datasets and suites")
    return registry


def selected_suites(registry: dict, requested: list[str]) -> list[str]:
    suites = registry["suites"]
    wanted: set[str] = set()
    visiting: set[str] = set()
    ordered: list[str] = []

    def add(name: str) -> None:
        if name not in suites:
            raise ValueError(f"Unknown suite: {name}; choices: {', '.join(suites)}")
        if name in visiting:
            raise ValueError(f"Cyclic suite dependency: {name}")
        if name in wanted:
            return
        visiting.add(name)
        for dependency in suites[name].get("depends_on", []):
            add(dependency)
        visiting.remove(name)
        wanted.add(name)
        ordered.append(name)

    for name in (list(suites) if "all" in requested else requested):
        add(name)
    return ordered


def source_files(registry: dict, root: Path = ROOT) -> set[str]:
    """List the source and config dependencies of the experiment registry."""
    files = {"scripts/reproduce_release.py", "scripts/run_matrix.py"}
    for data in registry["datasets"].values():
        files.add(data["config"])
    for suite in registry["suites"].values():
        for item in suite.get("training", []):
            files.add(item["matrix"])
            files.update(config for config, _ in train_jobs(_path(root, item["matrix"])))
        for item in suite.get("evaluations", []) + suite.get("adaptations", []):
            files.update((item["matrix"], item["script"]))
            raw = yaml.safe_load(_path(root, item["matrix"]).read_text(encoding="utf-8"))
            files.update(entry["config"] for entry in raw.get("adaptations", []))
        for item in suite.get("reports", []) + suite.get("figures", []):
            files.add(item["command"][0])
            files.update(item.get("sources", []))
    return files


def build_plan(registry: dict, suites: list[str], root: Path = ROOT) -> list[Step]:
    # Share seed expansion with the matrix runners.
    from run_adapt_matrix import _command as adapt_command
    from run_adapt_matrix import _jobs as adapt_jobs
    from run_ood_severity_matrix import _jobs as evaluation_jobs

    steps: list[Step] = []
    dataset_names = {name for suite in suites for name in registry["suites"][suite].get("datasets", [])}
    for name, data in registry["datasets"].items():
        if name in dataset_names:
            steps.append(Step(
                f"data:{name}", "data",
                [sys.executable, "scripts/generate_data.py", "--config", data["config"]],
                produces=[data["path"]], sha256=data.get("required_sha256"),
                immutable=bool(data.get("preserved", False)),
                expected_config=load_config(_path(root, data["config"])),
            ))

    evidence: dict[str, list[str]] = {}
    seen_training: dict[str, list[str]] = {}
    for name in suites:
        suite = registry["suites"][name]
        evidence[name] = []
        for spec in suite.get("training", []):
            jobs = train_jobs(_path(root, spec["matrix"]))
            selected = 0
            for config_path, overrides in jobs:
                overrides = {**overrides, **spec.get("overrides", {})}
                cfg = apply_dotted_overrides(load_config(_path(root, config_path)), overrides)
                run_dir = str(cfg["runtime"]["run_dir"]).rstrip("/")
                if spec.get("include_run_dir") and not re.search(spec["include_run_dir"], run_dir):
                    continue
                selected += 1
                _path(root, run_dir)
                if cfg["runtime"].get("overwrite") or cfg["runtime"].get("resume"):
                    raise ValueError(f"Release training cannot overwrite or resume: {run_dir}")
                outputs = [f"{run_dir}/{file}" for file in TRAIN_FILES]
                evidence[name].extend(outputs)
                command = train_command(config_path, overrides)
                if run_dir in seen_training:
                    if command != seen_training[run_dir]:
                        raise ValueError(f"Conflicting release training jobs: {run_dir}")
                    continue
                seen_training[run_dir] = command
                steps.append(Step(
                    f"train:{name}:{run_dir}", "train", command,
                    requires=[str(cfg["dataset"]["path"])], produces=outputs,
                    protected_dir=run_dir, expected_config=cfg,
                ))
            if not selected:
                raise ValueError(f"No jobs selected from {spec['matrix']}")

        for spec in suite.get("adaptations", []):
            for config_path, overrides in adapt_jobs(_path(root, spec["matrix"])):
                cfg = apply_dotted_overrides(load_config(_path(root, config_path)), overrides)
                run_dir = cfg["runtime"]["run_dir"]
                if cfg["runtime"].get("overwrite"):
                    raise ValueError(f"Release adaptation cannot overwrite: {run_dir}")
                outputs = [f"{run_dir}/adapt_results.json", f"{run_dir}/config.yaml"]
                evidence[name].extend(outputs)
                steps.append(Step(
                    f"adapt:{name}:{run_dir}", "evaluate", adapt_command(config_path, overrides),
                    requires=[cfg["dataset"]["path"], cfg["adaptation"]["checkpoint"]],
                    produces=outputs, protected_dir=run_dir, expected_config=cfg,
                ))

        for spec in suite.get("evaluations", []):
            jobs = evaluation_jobs(_path(root, spec["matrix"]))
            if not jobs:
                raise ValueError(f"No evaluations in {spec['matrix']}")
            if any(job.get("overwrite") for job in jobs):
                raise ValueError(f"Evaluation overwrite enabled: {spec['matrix']}")
            outputs = [f"{job['out_dir']}/{spec.get('metric_file', 'severity_metrics.json')}" for job in jobs]
            evidence[name].extend(outputs)
            steps.append(Step(
                f"evaluate:{name}:{spec['matrix']}", "evaluate",
                [sys.executable, spec["script"], "--matrix", spec["matrix"],
                 "--out-prefix", spec["out_prefix"]],
                requires=list(dict.fromkeys(
                    [job["checkpoint"] for job in jobs]
                    + [registry["datasets"][data]["path"] for data in suite.get("datasets", [])]
                )),
                produces=outputs + [spec["out_prefix"] + ".runs.csv", spec["out_prefix"] + ".aggregate.csv"],
                evaluations=list(zip(jobs, outputs)),
            ))

    for stage in ("reports", "figures"):
        for name in suites:
            suite = registry["suites"][name]
            required_evidence = list(evidence[name])
            for dependency in suite.get("depends_on", []):
                required_evidence.extend(evidence[dependency])
            for index, spec in enumerate(suite.get(stage, [])):
                requires = spec.get("requires", [])
                if stage == "reports":
                    requires = required_evidence + requires
                steps.append(Step(
                    f"{stage}:{name}:{index + 1}", stage,
                    [sys.executable, *spec["command"]], requires=requires,
                    produces=spec.get("outputs", []),
                    fresh_report_inputs=[path for path in evidence[name] if path.endswith("/meta.json")]
                    if spec.get("fresh_report") else [],
                ))
    for step in steps:
        for path in step.requires + step.produces:
            _path(root, path)
    return sorted(steps, key=lambda step: STAGES.index(step.stage))


@lru_cache(maxsize=16)
def _file_hash(path: Path, size: int, modified_ns: int) -> str:
    # The stat fields invalidate the cache when an input changes during a run.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed(step: Step, root: Path, *, verify_evaluation_inputs: bool = False) -> bool:
    from run_ood_severity_matrix import _read_cached_metrics

    for job, output in step.evaluations:
        path = _path(root, output)
        if path.is_file() and (verify_evaluation_inputs or path.with_suffix(".evaluation.json").is_file()):
            _read_cached_metrics(job, path, root, "cpu")
    if not step.produces or not all(_path(root, path).is_file() for path in step.produces):
        return False
    if step.stage == "train":
        metrics = json.loads(_path(root, step.produces[6]).read_text(encoding="utf-8"))
        missing = [key for key in METRICS if not isinstance(metrics.get(key), (int, float))
                   or not math.isfinite(metrics[key])]
        if missing or not metrics.get("method"):
            raise ValueError(f"Invalid completed metrics in {step.protected_dir}: {missing}")
        if step.expected_config:
            from otno.utils import stable_json_hash

            stored = load_config(_path(root, step.produces[0]))
            expected = step.expected_config
            meta = json.loads(_path(root, step.produces[1]).read_text(encoding="utf-8"))
            config_hash = stable_json_hash(stored)[:12]
            if any(item.get("config_hash") != config_hash for item in (meta, metrics)):
                raise ValueError(f"Completed run config hash mismatch: {step.protected_dir}")
            if metrics.get("method") != stored["training"]["method"] or metrics.get("seed") != stored["seed"]:
                raise ValueError(f"Completed run method/seed mismatch: {step.protected_dir}")
            dataset = _path(root, expected["dataset"]["path"])
            if dataset.is_file():
                stat = dataset.stat()
                fingerprint = _file_hash(dataset, stat.st_size, stat.st_mtime_ns)
                if any(item.get("dataset_sha256") != fingerprint for item in (meta, metrics)):
                    raise ValueError(f"Completed run dataset hash mismatch: {step.protected_dir}")
            for section in ("seed", "dataset", "model", "symmetry", "training"):
                values = expected.get(section)
                actual = stored.get(section)
                if isinstance(values, dict) and isinstance(actual, dict):
                    mismatch = [key for key in set(values) | set(actual)
                                if key != "stop_after_epochs" and actual.get(key) != values.get(key)]
                else:
                    mismatch = [] if actual == values else [section]
                if mismatch:
                    raise ValueError(f"Completed run differs from release config: {step.protected_dir}: {section}.{','.join(mismatch)}")
    if step.sha256:
        path = _path(root, step.produces[0])
        stat = path.stat()
        actual = _file_hash(path, stat.st_size, stat.st_mtime_ns)
        if actual != step.sha256:
            raise ValueError(f"Preserved input SHA-256 mismatch: {step.produces[0]}")
    elif step.stage == "data" and step.expected_config:
        from otno.data.generators import validate_existing_dataset

        validate_existing_dataset(_path(root, step.produces[0]), step.expected_config)
    if step.stage == "evaluate":
        for output in step.produces:
            if not output.endswith(".json"):
                continue
            metrics = json.loads(_path(root, output).read_text(encoding="utf-8"))
            groups = [metrics["before"], metrics["after"]] if "before" in metrics else [metrics]
            for values in groups:
                numeric = [value for value in values.values() if isinstance(value, (int, float))]
                if not numeric or not all(math.isfinite(value) for value in numeric):
                    raise ValueError(f"Invalid cached evaluation metrics: {output}")
        if step.expected_config:
            stored = load_config(_path(root, step.produces[1]))
            expected = {**step.expected_config, "runtime": dict(step.expected_config["runtime"])}
            for config in (stored, expected):
                config["runtime"].pop("overwrite", None)
            if stored != expected:
                raise ValueError(f"Completed adaptation differs from release config: {step.protected_dir}")
    return True


def preflight(
    steps: list[Step], root: Path = ROOT, *, verify_evaluation_inputs: bool = False
) -> list[str]:
    """Validate all inputs and dependencies before execution."""
    available: set[str] = set()
    problems: list[str] = []
    for step in steps:
        try:
            report_command(step, root)
            done = completed(step, root, verify_evaluation_inputs=verify_evaluation_inputs)
        except (ValueError, OSError) as exc:
            problems.append(str(exc))
            continue
        if step.immutable and not done:
            problems.append(f"Missing preserved evidence input (cannot regenerate): {step.produces[0]}")
            continue
        if step.protected_dir and not done:
            run_dir = _path(root, step.protected_dir)
            if run_dir.exists() and any(run_dir.iterdir()):
                problems.append(f"Partial protected run; inspect and relocate before rerunning: {step.protected_dir}")
                continue
        if not done or step.stage in ("reports", "figures"):
            for requirement in step.requires:
                if requirement not in available and not _path(root, requirement).is_file():
                    problems.append(f"{step.name}: missing prerequisite {requirement}")
        available.update(step.produces)
    return list(dict.fromkeys(problems))


def report_command(step: Step, root: Path) -> list[str]:
    """Select validation rules for either archived or newly trained runs."""
    if not step.fresh_report_inputs:
        return step.command
    kinds = set()
    for path in step.fresh_report_inputs:
        meta_path = _path(root, path)
        # Missing runs will be trained by this release, with source manifests.
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        kinds.add("fresh" if not meta or meta.get("source_sha256") else "historical")
    if len(kinds) != 1:
        raise ValueError(f"Cannot combine historical and fresh backbone runs: {step.name}")
    return step.command + (["--fresh-runs"] if "fresh" in kinds else [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/release.yaml")
    parser.add_argument("--suite", action="append", help="Repeat to select suites; default: all")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands without writing files (default)")
    mode.add_argument("--preflight", action="store_true", help="Read-only validation of inputs and planned dependencies")
    mode.add_argument("--execute", action="store_true", help="Run the selected stages; training may take days")
    args = parser.parse_args()
    registry = load_registry(ROOT, args.registry)
    suites = selected_suites(registry, args.suite or ["all"])
    for path in source_files(registry):
        if not _path(ROOT, path).is_file():
            raise SystemExit(f"Missing release source: {path}")
    steps = build_plan(registry, suites)
    if args.execute or args.preflight:
        for step in steps:
            if step.stage == "data":
                completed(step, ROOT)
    if args.stage != "all":
        steps = [step for step in steps if step.stage == args.stage or step.immutable]
    print(f"Suites: {', '.join(suites)}; {len(steps)} steps; CPU, OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
    if not args.preflight:
        for step in steps:
            print(f"[{step.stage}] {shlex.join(report_command(step, ROOT))}", flush=True)
    if not args.execute and not args.preflight:
        return
    problems = preflight(steps, ROOT, verify_evaluation_inputs=args.execute)
    if problems:
        raise SystemExit("Release preflight failed:\n" + "\n".join(problems))
    print("RELEASE PREFLIGHT PASSED", flush=True)
    if not args.execute:
        return
    # Backbone runs used CPU with device: auto in their configs.
    # Disable CUDA to reproduce that setting on hosts with a GPU.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    for step in steps:
        if step.stage in ("data", "train", "evaluate") and completed(step, ROOT, verify_evaluation_inputs=True):
            print(f"skip completed: {step.name}", flush=True)
            continue
        subprocess.run(report_command(step, ROOT), cwd=ROOT, env=env, check=True)
        if step.produces and not completed(step, ROOT):
            raise SystemExit(f"Command did not produce its declared outputs: {step.name}")


if __name__ == "__main__":
    main()
