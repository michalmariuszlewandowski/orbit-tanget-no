"""Experiment lifecycle: prepare data, train epochs, checkpoint, and evaluate.

Scientific objective assembly lives in :mod:`otno.training.objectives`; this
module owns the run files and the complete state needed for exact continuation.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from otno.config import save_config, validate_config
from otno.data.datasets import load_tensor_dataset
from otno.data.generators import generate_dataset_from_config, validate_existing_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.symmetry.transforms import BaseTransform
from otno.training.metrics import evaluate_model, measure_inference_latency
from otno.training.objectives import TrainingObjective, build_training_objective, compute_batch_loss
from otno.utils import (
    append_jsonl,
    capture_rng_state,
    check_run_directory,
    count_parameters,
    dump_json,
    ensure_dir,
    environment_manifest,
    file_sha256,
    get_device,
    make_torch_generator,
    prepare_run_directory,
    restore_rng_state,
    seed_worker,
    set_seed,
    source_manifest,
    stable_json_hash,
)


def _prepare_dataset(config: dict[str, Any]) -> Path:
    """Generate a missing cache or verify that its parameters match the run."""
    dataset_cfg = config["dataset"]
    path = Path(dataset_cfg["path"])
    if not path.exists():
        if not dataset_cfg.get("generate_if_missing", True):
            raise FileNotFoundError(path)
        generate_dataset_from_config(config, path)
    else:
        validate_existing_dataset(path, dataset_cfg)
    return path


def _loader(
    path: Path,
    split: str,
    config: dict[str, Any],
    *,
    shuffle: bool,
    fraction_override: float | None = None,
    batch_size_override: int | None = None,
    seed_offset_override: int | None = None,
) -> DataLoader:
    """Select a labeled fraction and attach a split-specific sampler RNG."""
    training_cfg = config.get("training", {})
    if fraction_override is not None:
        fraction = float(fraction_override)
    else:
        fraction = float(training_cfg.get("data_fraction", 1.0)) if split == "train" else 1.0
    dataset, _ = load_tensor_dataset(
        path, split, fraction=fraction, seed=int(config.get("seed", 0))
    )
    split_offset = (
        int(seed_offset_override)
        if seed_offset_override is not None
        else {"train": 0, "val": 10_000, "test": 20_000}.get(split, 30_000)
    )
    generator = make_torch_generator(int(config.get("seed", 0)) + split_offset)
    return DataLoader(
        dataset,
        batch_size=int(batch_size_override or training_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", False)),
        generator=generator,
        worker_init_fn=seed_worker,
    )


_RUN_OUTPUTS = (
    "config.yaml",
    "meta.json",
    "source_manifest.json",
    "train_metrics.jsonl",
    "val_metrics.jsonl",
    "test_metrics.json",
    "partial_metrics.json",
    "rng_state_initial.pt",
    "rng_state_final.pt",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
)


def _rewind_epoch_log(path: Path, completed_epoch: int) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # An interruption can leave the final record partially written.
            raise
        if int(record["epoch"]) <= completed_epoch:
            kept.append(line)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    os.replace(temporary, path)


def _resume_config_hash(config: dict[str, Any]) -> str:
    """Ignore location and interruption controls when comparing experiments."""
    config = copy.deepcopy(config)
    for key in ("resume", "overwrite", "run_dir", "device"):
        config.get("runtime", {}).pop(key, None)
    config.get("training", {}).pop("stop_after_epochs", None)
    config.get("dataset", {}).pop("path", None)
    return stable_json_hash(config)


def _torch_save_atomic(payload: Any, path: Path, *, attempts: int = 20) -> None:
    """Commit a whole checkpoint, retrying transient filesystem failures."""
    ensure_dir(path.parent)
    last_error: BaseException | None = None
    for attempt in range(attempts):
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{attempt}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(min(5.0, 0.5 * (attempt + 1)))
    assert last_error is not None
    raise last_error


def _build_loaders(
    config: dict[str, Any],
    dataset_path: Path,
    *,
    method: str,
) -> dict[str, DataLoader]:
    """Give each split its own RNG stream; semi-supervised data has a fourth."""
    loaders = {
        "train": _loader(dataset_path, "train", config, shuffle=True),
        "val": _loader(dataset_path, "val", config, shuffle=False),
        "test": _loader(dataset_path, "test", config, shuffle=False),
    }
    if method == "semi_aug_orbit":
        training = config.get("training", {})
        # This full-data loader contributes inputs only to the orbit loss.
        loaders["orbit"] = _loader(
            dataset_path,
            "train",
            config,
            shuffle=True,
            fraction_override=float(training.get("orbit_data_fraction", 1.0)),
            batch_size_override=int(
                training.get("orbit_batch_size", training.get("batch_size", 32))
            ),
            seed_offset_override=40_000,
        )
    return loaders


def _load_resume_checkpoint(
    config: dict[str, Any],
    run_dir: Path,
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Validate all resumable state before changing existing run outputs."""
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    resume_path = last_path if last_path.exists() else best_path
    if not resume_path.exists():
        raise FileNotFoundError(f"runtime.resume=true but no checkpoint exists in {run_dir}")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if "rng_state" not in checkpoint or "loader_rng_states" not in checkpoint:
        raise ValueError(
            "This legacy checkpoint lacks complete RNG state; exact resume is unavailable. "
            "Start a fresh run in a new directory."
        )
    required = {"model", "optimizer", "scheduler", "config", "epoch", "best_val", "meta"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint lacks state required for resume: {', '.join(sorted(missing))}"
        )
    if _resume_config_hash(config) != _resume_config_hash(checkpoint["config"]):
        raise ValueError("Resume config changes experiment parameters; start a new run instead.")
    if checkpoint.get("meta", {}).get("dataset_sha256") != dataset_sha256:
        raise ValueError("Resume dataset SHA-256 differs from the checkpoint.")
    if math.isfinite(float(checkpoint["best_val"])) and not best_path.exists():
        raise FileNotFoundError(f"Resume requires the validation-selected checkpoint: {best_path}")
    return checkpoint


def _prepare_run_outputs(
    config: dict[str, Any],
    run_dir: Path,
    *,
    meta: dict[str, Any],
    source: dict[str, Any],
    loaders: dict[str, DataLoader],
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    """Restore sampler state and committed logs, or initialize a fresh run."""
    resume = bool(config.get("runtime", {}).get("resume", False))
    if resume_checkpoint is not None:
        for name, loader in loaders.items():
            loader.generator.set_state(resume_checkpoint["loader_rng_states"][name].cpu())
        restore_rng_state(resume_checkpoint["rng_state"])
        last_path = run_dir / "checkpoints" / "last.pt"
        if not last_path.exists():
            _torch_save_atomic(resume_checkpoint, last_path)
        completed_epoch = int(resume_checkpoint["epoch"])
        for filename in ("train_metrics.jsonl", "val_metrics.jsonl"):
            _rewind_epoch_log(run_dir / filename, completed_epoch)
        for filename in ("test_metrics.json", "partial_metrics.json", "rng_state_final.pt"):
            (run_dir / filename).unlink(missing_ok=True)
    else:
        prepare_run_directory(
            run_dir,
            output_files=_RUN_OUTPUTS,
            overwrite=bool(config.get("runtime", {}).get("overwrite", True)),
        )
    save_config(config, run_dir / "config.yaml")
    dump_json(meta, run_dir / "meta.json")
    dump_json(source, run_dir / "source_manifest.json")
    if not resume or not (run_dir / "rng_state_initial.pt").exists():
        torch.save(capture_rng_state(), run_dir / "rng_state_initial.pt")


def _next_batch(loader: DataLoader, iterator: Iterator) -> tuple[dict[str, torch.Tensor], Iterator]:
    """Cycle a loader when an epoch requests more steps than it contains."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loaders: dict[str, DataLoader],
    objective: TrainingObjective,
    *,
    device: torch.device,
    transform: BaseTransform | None,
    epoch: int,
    training: dict[str, Any],
) -> dict[str, Any]:
    """Take joint optimizer steps and report their mean losses.

    Labeled batches cycle independently of the optional orbit-input batches.
    Iterator creation and the order of loss evaluation are part of the saved
    RNG contract, including epochs with fewer steps than a full loader pass.
    """
    epoch_wall_start = time.perf_counter()
    model.train()
    train_loader = loaders["train"]
    orbit_loader = loaders.get("orbit")
    steps = int(training.get("steps_per_epoch", len(train_loader)))
    if orbit_loader is not None:
        steps = int(training.get("orbit_steps_per_epoch", len(orbit_loader)))
    progress = tqdm(
        total=steps,
        desc=f"epoch {epoch}/{int(training.get('epochs', 100))}",
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    train_iterator = iter(train_loader)
    orbit_iterator = iter(orbit_loader) if orbit_loader is not None else None

    def next_orbit_inputs() -> torch.Tensor:
        nonlocal orbit_iterator
        assert orbit_loader is not None and orbit_iterator is not None
        batch, orbit_iterator = _next_batch(orbit_loader, orbit_iterator)
        return batch["a"].to(device)

    total_loss = 0.0
    totals: dict[str, float] = {}
    num_batches = 0
    try:
        for step in range(steps):
            progress.update(1)
            batch, train_iterator = _next_batch(train_loader, train_iterator)
            inputs, targets = batch["a"].to(device), batch["u"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = compute_batch_loss(
                model,
                inputs,
                targets,
                objective,
                transform=transform,
                orbit_inputs=next_orbit_inputs if orbit_loader is not None else None,
            )
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {step + 1}; "
                    "check dataset values, solver stability, and loss settings."
                )
            loss.backward()
            if training.get("grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            num_batches += 1
            progress.set_postfix(loss=total_loss / num_batches)
    finally:
        progress.close()

    record = {
        "epoch": epoch,
        "train_loss": total_loss / max(1, num_batches),
        "train_supervised_steps": steps,
        "train_epoch_seconds": time.perf_counter() - epoch_wall_start,
        **{f"train_{key}": value / max(1, num_batches) for key, value in totals.items()},
    }
    if orbit_loader is not None:
        record["train_unlabeled_orbit_steps"] = 0  # Joint loss; no separate optimizer steps.
    return record


def _evaluate_best_model(
    config: dict[str, Any],
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    *,
    device: torch.device,
    transform: BaseTransform | None,
    best_path: Path,
) -> dict[str, Any]:
    """Evaluate the validation-selected weights and their inference latency."""
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    training = config.get("training", {})
    test_metrics = evaluate_model(
        model,
        loaders["test"],
        device=device,
        transform=transform,
        n_orbit_samples=int(training.get("eval_orbit_samples", 4)),
        seed=int(training.get("test_eval_seed", int(config.get("seed", 0)) + 200_000)),
    )
    first_batch = next(iter(loaders["test"]))["a"].to(device)
    latency = measure_inference_latency(
        model,
        first_batch,
        repeats=int(training.get("latency_repeats", 20)),
        warmup=int(training.get("latency_warmup", 5)),
    )
    return {**test_metrics, **latency}


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Train or resume one experiment, then evaluate its best validation model.

    Dataset and resume compatibility are checked before replacing run outputs.
    Every epoch commits the model, optimizer, scheduler, and all RNG streams to
    ``last.pt``; ``best.pt`` is selected solely by validation relative L2 error.
    ``stop_after_epochs`` returns partial results that can be resumed exactly.
    """
    validate_config(config)
    training = config.get("training", {})
    method = str(training.get("method", "baseline")).lower()
    if method == "semi_aug_orbit" and int(training.get("unlabeled_orbit_steps_per_epoch", 0)) != 0:
        raise ValueError(
            "Separate unlabeled optimizer steps are unsupported; "
            "use joint semi_aug_orbit with orbit_steps_per_epoch."
        )
    seed = int(config.get("seed", 0))
    runtime = config.get("runtime", {})
    deterministic = bool(runtime.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = get_device(runtime.get("device", "auto"))
    run_dir = Path(runtime.get("run_dir", "runs/default"))
    resume = bool(runtime.get("resume", False))
    if not resume:
        check_run_directory(
            run_dir,
            output_files=_RUN_OUTPUTS,
            overwrite=bool(runtime.get("overwrite", True)),
        )

    dataset_path = _prepare_dataset(config)
    objective = build_training_objective(training)
    epochs = int(training.get("epochs", 100))
    model = build_model(config).to(device)
    transform = build_transform(config.get("symmetry"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    source = source_manifest(Path(__file__).resolve().parents[3])
    meta = {
        "device": str(device),
        "parameters": count_parameters(model),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "method": method,
        "orbit_control": objective.orbit_control,
        "model_name": str(config.get("model", {}).get("name", "fno1d")),
        "dataset_kind": str(config.get("dataset", {}).get("kind", "")),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "seed": seed,
        "deterministic": deterministic,
        "config_hash": stable_json_hash(config)[:12],
        "environment": environment_manifest(config),
        "source_sha256": source["source_sha256"],
        "git_dirty": source["git_dirty"],
    }

    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    best_val, start_epoch, prior_wall_seconds = float("inf"), 1, 0.0
    resume_checkpoint = None
    if resume:
        resume_checkpoint = _load_resume_checkpoint(
            config, run_dir, dataset_sha256=meta["dataset_sha256"]
        )
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        best_val = float(resume_checkpoint["best_val"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        prior_wall_seconds = float(resume_checkpoint.get("train_wall_seconds", 0.0))

    loaders = _build_loaders(config, dataset_path, method=method)
    _prepare_run_outputs(
        config,
        run_dir,
        meta=meta,
        source=source,
        loaders=loaders,
        resume_checkpoint=resume_checkpoint,
    )
    target_epoch = epochs
    if training.get("stop_after_epochs") is not None:
        target_epoch = min(epochs, start_epoch + int(training["stop_after_epochs"]) - 1)
    train_wall_start = time.perf_counter()

    def checkpoint_state(epoch: int) -> dict[str, Any]:
        """Capture the complete continuation state without advancing any RNG."""
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "meta": meta,
            "best_val": best_val,
            "torch_rng": torch.get_rng_state(),
            "rng_state": capture_rng_state(),
            "loader_rng_states": {
                name: loader.generator.get_state() for name, loader in loaders.items()
            },
            "train_wall_seconds": prior_wall_seconds + time.perf_counter() - train_wall_start,
        }

    for epoch in range(start_epoch, target_epoch + 1):
        train_record = _train_epoch(
            model,
            optimizer,
            loaders,
            objective,
            device=device,
            transform=transform,
            epoch=epoch,
            training=training,
        )
        scheduler.step()
        train_record["lr"] = scheduler.get_last_lr()[0]
        append_jsonl(train_record, run_dir / "train_metrics.jsonl")

        if epoch % int(training.get("eval_every", 5)) == 0 or epoch == epochs:
            val_metrics = evaluate_model(
                model,
                loaders["val"],
                device=device,
                transform=transform,
                n_orbit_samples=int(training.get("eval_orbit_samples", 1)),
                seed=int(training.get("eval_seed", seed + 100_000 + epoch)),
            )
            val_record = {
                "epoch": epoch,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            append_jsonl(val_record, run_dir / "val_metrics.jsonl")
            val_key = val_metrics["relative_l2"]
            if val_key < best_val:
                best_val = val_key
                _torch_save_atomic(checkpoint_state(epoch), best_path)
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_record['train_loss']:.6g} "
                f"val_relative_l2={val_key:.6g} "
                f"best_val={best_val:.6g}",
                flush=True,
            )
        _torch_save_atomic(checkpoint_state(epoch), last_path)

    if target_epoch < epochs:
        results = {
            "status": "partial",
            "completed_epoch": target_epoch,
            "target_epochs": epochs,
            "best_val_relative_l2": best_val,
            "train_wall_seconds": prior_wall_seconds + time.perf_counter() - train_wall_start,
            **meta,
        }
        dump_json(results, run_dir / "partial_metrics.json")
        return results

    test_metrics = _evaluate_best_model(
        config,
        model,
        loaders,
        device=device,
        transform=transform,
        best_path=best_path,
    )
    results = {"best_val_relative_l2": best_val, **test_metrics, **meta}
    results["train_wall_seconds"] = prior_wall_seconds + time.perf_counter() - train_wall_start
    dump_json(results, run_dir / "test_metrics.json")
    (run_dir / "partial_metrics.json").unlink(missing_ok=True)
    torch.save(capture_rng_state(), run_dir / "rng_state_final.pt")
    # Evaluation uses best.pt; last.pt retains the final model/optimizer pair.
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    last_checkpoint["results"] = results
    _torch_save_atomic(last_checkpoint, last_path)
    return results
