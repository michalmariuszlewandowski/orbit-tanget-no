from __future__ import annotations

import os
import copy
import json
import math
import sys
import time
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
from otno.training.losses import (
    augmented_supervised_loss,
    orbit_consistency_loss,
    relative_l2_loss,
    tangent_propagation_loss,
    trajectory_nmse_loss,
)
from otno.training.metrics import evaluate_model, measure_inference_latency
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
    training_cfg = config.get("training", {})
    if fraction_override is not None:
        fraction = float(fraction_override)
    else:
        fraction = float(training_cfg.get("data_fraction", 1.0)) if split == "train" else 1.0
    dataset, _ = load_tensor_dataset(path, split, fraction=fraction, seed=int(config.get("seed", 0)))
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
    "config.yaml", "meta.json", "source_manifest.json", "train_metrics.jsonl",
    "val_metrics.jsonl", "test_metrics.json", "partial_metrics.json",
    "rng_state_initial.pt", "rng_state_final.pt", "checkpoints/best.pt", "checkpoints/last.pt",
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
    config = copy.deepcopy(config)
    for key in ("resume", "overwrite", "run_dir", "device"):
        config.get("runtime", {}).pop(key, None)
    config.get("training", {}).pop("stop_after_epochs", None)
    config.get("dataset", {}).pop("path", None)
    return stable_json_hash(config)


def _torch_save_atomic(payload: Any, path: Path, *, attempts: int = 20) -> None:
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


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    training_cfg = config.get("training", {})
    method = str(training_cfg.get("method", "baseline")).lower()
    if method == "semi_aug_orbit" and int(training_cfg.get("unlabeled_orbit_steps_per_epoch", 0)) != 0:
        raise ValueError("Separate unlabeled optimizer steps are unsupported; use joint semi_aug_orbit with orbit_steps_per_epoch.")
    seed = int(config.get("seed", 0))
    runtime_cfg = config.get("runtime", {})
    deterministic = bool(runtime_cfg.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = get_device(runtime_cfg.get("device", "auto"))
    run_dir = Path(runtime_cfg.get("run_dir", "runs/default"))
    resume = bool(runtime_cfg.get("resume", False))
    if not resume:
        check_run_directory(
            run_dir, output_files=_RUN_OUTPUTS,
            overwrite=bool(runtime_cfg.get("overwrite", True)),
        )

    dataset_path = _prepare_dataset(config)
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    epochs = int(training_cfg.get("epochs", 100))
    stop_after_epochs = training_cfg.get("stop_after_epochs", None)
    lambda_orbit = float(training_cfg.get("lambda_orbit", 1.0))
    lambda_tangent = float(training_cfg.get("lambda_tangent", lambda_orbit))
    lambda_aug = float(training_cfg.get("lambda_aug", 1.0))
    normalize_by_epsilon = bool(training_cfg.get("normalize_by_epsilon", True))
    loss_name = str(training_cfg.get("loss", "relative_l2")).lower()
    if loss_name in {"relative_l2", "rel_l2"}:
        supervised_loss_fn = relative_l2_loss
    elif loss_name in {"trajectory_nmse", "lpsda_nmse", "nmse"}:
        supervised_loss_fn = trajectory_nmse_loss
    else:
        raise ValueError(f"Unknown training.loss={loss_name!r}")
    eta = float(training_cfg.get("orbit_eta", 1e-6))
    orbit_control = str(training_cfg.get("orbit_control", "physical")).lower()
    eval_every = int(training_cfg.get("eval_every", 5))
    grad_clip = training_cfg.get("grad_clip", None)
    aug_methods = {
        "aug",
        "augmentation",
        "aug_orbit",
        "orbit_aug",
        "aug_orbit_shuffle",
        "aug_orbit_shuffled",
        "aug_orbit_no_output",
        "aug_orbit_input_only",
        "aug_tangent",
        "tangent_aug",
        "semi_aug_orbit",
    }
    orbit_methods = {
        "orbit",
        "orb",
        "aug_orbit",
        "orbit_aug",
        "aug_orbit_shuffle",
        "aug_orbit_shuffled",
        "aug_orbit_no_output",
        "aug_orbit_input_only",
        "semi_aug_orbit",
    }
    tangent_methods = {
        "tangent",
        "tangent_prop",
        "tangent_propagation",
        "aug_tangent",
        "tangent_aug",
    }
    if method in {"aug_orbit_shuffle", "aug_orbit_shuffled"} and orbit_control == "physical":
        orbit_control = "shuffle_output"
    if method in {"aug_orbit_no_output", "aug_orbit_input_only"} and orbit_control == "physical":
        orbit_control = "no_output_transform"

    model = build_model(config).to(device)
    transform = build_transform(config.get("symmetry"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    source = source_manifest(Path(__file__).resolve().parents[3])
    meta = {
        "device": str(device),
        "parameters": count_parameters(model),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "method": method,
        "orbit_control": orbit_control,
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

    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"

    start_epoch = 1
    prior_wall_seconds = 0.0
    resume_checkpoint = None
    if resume:
        resume_path = last_path if last_path.exists() else best_path
        if not resume_path.exists():
            raise FileNotFoundError(f"runtime.resume=true but no checkpoint exists in {run_dir}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "rng_state" not in ckpt or "loader_rng_states" not in ckpt:
            raise ValueError("This legacy checkpoint lacks complete RNG state; exact resume is unavailable. Start a fresh run in a new directory.")
        missing = {"model", "optimizer", "scheduler", "config", "epoch", "best_val", "meta"} - ckpt.keys()
        if missing:
            raise ValueError(f"Checkpoint lacks state required for resume: {', '.join(sorted(missing))}")
        if _resume_config_hash(config) != _resume_config_hash(ckpt["config"]):
            raise ValueError("Resume config changes experiment parameters; start a new run instead.")
        if ckpt.get("meta", {}).get("dataset_sha256") != meta["dataset_sha256"]:
            raise ValueError("Resume dataset SHA-256 differs from the checkpoint.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val = float(ckpt["best_val"])
        if math.isfinite(best_val) and not best_path.exists():
            raise FileNotFoundError(f"Resume requires the validation-selected checkpoint: {best_path}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        resume_checkpoint = ckpt
        prior_wall_seconds = float(ckpt.get("train_wall_seconds", 0.0))

    train_loader = _loader(
        dataset_path,
        "train",
        config,
        shuffle=True,
    )
    val_loader = _loader(dataset_path, "val", config, shuffle=False)
    test_loader = _loader(dataset_path, "test", config, shuffle=False)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    # The independent full-data loader contributes inputs only to the orbit loss.
    orbit_loader = None
    if method == "semi_aug_orbit":
        orbit_loader = _loader(
            dataset_path, "train", config, shuffle=True,
            fraction_override=float(training_cfg.get("orbit_data_fraction", 1.0)),
            batch_size_override=int(training_cfg.get("orbit_batch_size", training_cfg.get("batch_size", 32))),
            seed_offset_override=40_000,
        )
        loaders["orbit"] = orbit_loader
    if resume_checkpoint is not None:
        for name, loader in loaders.items():
            loader.generator.set_state(resume_checkpoint["loader_rng_states"][name].cpu())
        restore_rng_state(resume_checkpoint["rng_state"])
        if not last_path.exists():
            _torch_save_atomic(resume_checkpoint, last_path)
        for filename in ("train_metrics.jsonl", "val_metrics.jsonl"):
            _rewind_epoch_log(run_dir / filename, start_epoch - 1)
        for filename in ("test_metrics.json", "partial_metrics.json", "rng_state_final.pt"):
            (run_dir / filename).unlink(missing_ok=True)
    else:
        prepare_run_directory(
            run_dir, output_files=_RUN_OUTPUTS,
            overwrite=bool(runtime_cfg.get("overwrite", True)),
        )
    save_config(config, run_dir / "config.yaml")
    dump_json(meta, run_dir / "meta.json")
    dump_json(source, run_dir / "source_manifest.json")
    if not resume or not (run_dir / "rng_state_initial.pt").exists():
        torch.save(capture_rng_state(), run_dir / "rng_state_initial.pt")
    steps_per_epoch = int(training_cfg.get("steps_per_epoch", len(train_loader)))
    if orbit_loader is not None:
        steps_per_epoch = int(training_cfg.get("orbit_steps_per_epoch", len(orbit_loader)))

    target_epoch = epochs
    if stop_after_epochs is not None:
        target_epoch = min(epochs, start_epoch + int(stop_after_epochs) - 1)

    train_wall_start = time.perf_counter()

    def checkpoint_state(epoch: int) -> dict[str, Any]:
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
            "loader_rng_states": {name: loader.generator.get_state() for name, loader in loaders.items()},
            "train_wall_seconds": prior_wall_seconds + time.perf_counter() - train_wall_start,
        }

    for epoch in range(start_epoch, target_epoch + 1):
        epoch_wall_start = time.perf_counter()
        model.train()
        supervised_steps = steps_per_epoch
        progress = tqdm(
            total=supervised_steps,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        train_iter = iter(train_loader)
        orbit_iter = iter(orbit_loader) if orbit_loader is not None else None
        epoch_loss = 0.0
        epoch_logs: dict[str, float] = {}
        num_batches = 0

        for _ in range(supervised_steps):
            progress.update(1)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            a = batch["a"].to(device)
            u = batch["u"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(a)
            loss = supervised_loss_fn(pred, u)
            logs = {"supervised_loss": float(loss.detach().cpu())}

            if method in aug_methods:
                if transform is None:
                    raise ValueError("Augmentation method requires a symmetry transform")
                aug_loss = augmented_supervised_loss(model, a, u, transform, loss_fn=supervised_loss_fn)
                loss = loss + lambda_aug * aug_loss
                logs["aug_loss"] = float(aug_loss.detach().cpu())

            if method in orbit_methods:
                if transform is None:
                    raise ValueError("Orbit method requires a symmetry transform")
                orbit_a, orbit_base_pred = a, pred
                if orbit_iter is not None:
                    try:
                        orbit_batch = next(orbit_iter)
                    except StopIteration:
                        orbit_iter = iter(orbit_loader)
                        orbit_batch = next(orbit_iter)
                    orbit_a = orbit_batch["a"].to(device)
                    orbit_base_pred = model(orbit_a)
                orb_loss, orb_stats = orbit_consistency_loss(
                    model,
                    orbit_a,
                    transform,
                    base_pred=orbit_base_pred,
                    normalize_by_epsilon=normalize_by_epsilon,
                    eta=eta,
                    target_mode=orbit_control,
                )
                loss = loss + lambda_orbit * orb_loss
                logs.update(orb_stats)

            if method in tangent_methods:
                if transform is None:
                    raise ValueError("Tangent method requires a symmetry transform")
                tangent_loss, tangent_stats = tangent_propagation_loss(
                    model,
                    a,
                    transform,
                    normalize_by_epsilon=normalize_by_epsilon,
                    eta=eta,
                )
                loss = loss + lambda_tangent * tangent_loss
                logs.update(tangent_stats)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)
            num_batches += 1
            progress.set_postfix(loss=epoch_loss / num_batches)

        progress.close()

        scheduler.step()
        train_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, num_batches),
            "lr": scheduler.get_last_lr()[0],
            "train_supervised_steps": supervised_steps,
            "train_epoch_seconds": time.perf_counter() - epoch_wall_start,
            **{f"train_{k}": v / max(1, num_batches) for k, v in epoch_logs.items()},
        }
        if orbit_loader is not None:
            train_record["train_unlabeled_orbit_steps"] = 0  # Joint loss, no separate optimizer steps.
        append_jsonl(train_record, run_dir / "train_metrics.jsonl")

        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                transform=transform,
                n_orbit_samples=int(training_cfg.get("eval_orbit_samples", 1)),
                seed=int(training_cfg.get("eval_seed", seed + 100_000 + epoch)),
            )
            val_record = {"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}}
            append_jsonl(val_record, run_dir / "val_metrics.jsonl")
            val_key = val_metrics["relative_l2"]
            if val_key < best_val:
                best_val = val_key
                _torch_save_atomic(
                    checkpoint_state(epoch),
                    best_path,
                )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_record['train_loss']:.6g} "
                f"val_relative_l2={val_key:.6g} "
                f"best_val={best_val:.6g}",
                flush=True,
            )

        _torch_save_atomic(
            checkpoint_state(epoch),
            last_path,
        )

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

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        transform=transform,
        n_orbit_samples=int(training_cfg.get("eval_orbit_samples", 4)),
        seed=int(training_cfg.get("test_eval_seed", seed + 200_000)),
    )
    first_batch = next(iter(test_loader))["a"].to(device)
    latency = measure_inference_latency(
        model,
        first_batch,
        repeats=int(training_cfg.get("latency_repeats", 20)),
        warmup=int(training_cfg.get("latency_warmup", 5)),
    )
    results = {"best_val_relative_l2": best_val, **test_metrics, **latency, **meta}
    results["train_wall_seconds"] = prior_wall_seconds + time.perf_counter() - train_wall_start
    dump_json(results, run_dir / "test_metrics.json")
    (run_dir / "partial_metrics.json").unlink(missing_ok=True)
    torch.save(capture_rng_state(), run_dir / "rng_state_final.pt")
    # Evaluation uses best.pt; last.pt must retain the final model/optimizer pair.
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    last_checkpoint["results"] = results
    _torch_save_atomic(last_checkpoint, last_path)
    return results
