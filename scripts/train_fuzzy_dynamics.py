#!/usr/bin/env python3
"""Train the five-mode fuzzy reasoning dynamical system."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.activation_data import ActivationTrajectoryDataset, load_activation_cache
from src.fuzzy_dynamics import (
    FuzzyDynamicsConfig,
    FuzzyReasoningDynamics,
    LossConfig,
    fuzzy_dynamics_loss,
)
from src.utils import (
    configure_experiment_logging,
    experiment_directory,
    generate_experiment_time,
    validate_experiment_time,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--experiment-time",
        default=None,
        help=(
            "Existing pipeline timestamp in YYYYMMDD_HHMMSS_microseconds format; "
            "normally generated automatically."
        ),
    )
    parser.add_argument("--config", default="configs/fuzzy_dynamics_base.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def mean_metrics(totals: dict[str, float], batches: int) -> dict[str, float]:
    return {name: value / max(1, batches) for name, value in totals.items()}


def sample_training_states(
    cache: dict[str, Any],
    train_indices: list[int],
    sample_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample aligned states across training trajectories and all layers for PCA."""
    layers = int(cache["hidden"].shape[1])
    total = len(train_indices) * layers
    count = min(sample_count, total)
    if count <= 0:
        raise ValueError("projector_fit_samples must be positive.")
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(total, generator=generator)[:count]
    train_lookup = torch.tensor(train_indices, dtype=torch.long)
    trajectories = train_lookup[positions // layers]
    layer_indices = positions % layers
    return (
        cache["hidden"][trajectories, layer_indices],
        cache["belief"][trajectories, layer_indices],
    )


@torch.no_grad()
def fit_state_delta_scale(
    model: FuzzyReasoningDynamics, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Fit per-coordinate delta standard deviations using only the training split."""
    model.eval()
    total = torch.zeros(model.config.state_dim, dtype=torch.float64, device=device)
    total_square = torch.zeros_like(total)
    count = 0
    for batch in loader:
        states, *_ = model._project_batch(move_batch(batch, device))
        delta = (states[:, 1:] - states[:, :-1]).to(torch.float64)
        total += delta.sum(dim=(0, 1))
        total_square += delta.square().sum(dim=(0, 1))
        count += delta.shape[0] * delta.shape[1]
    if count == 0:
        raise ValueError("Cannot fit state delta scales from an empty training loader.")
    variance = (total_square / count - (total / count).square()).clamp_min(0.0)
    return variance.sqrt().to(torch.float32)


def split_indices(
    count: int,
    metadata: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
    split_by: str | None,
) -> tuple[list[int], list[int], str]:
    generator = torch.Generator().manual_seed(seed)
    if split_by and len(metadata) == count:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(metadata):
            value = record.get(split_by) if isinstance(record, dict) else None
            if value not in (None, ""):
                groups[str(value)].append(index)
        if len(groups) >= 2 and sum(map(len, groups.values())) == count:
            names = list(groups)
            order = torch.randperm(len(names), generator=generator).tolist()
            validation_group_count = min(
                len(names) - 1, max(1, round(len(names) * validation_fraction))
            )
            validation_groups = {names[index] for index in order[:validation_group_count]}
            validation = [i for name in validation_groups for i in groups[name]]
            train = [i for name, values in groups.items() if name not in validation_groups for i in values]
            return train, validation, split_by

    indices = torch.randperm(count, generator=generator).tolist()
    validation_count = min(count - 1, max(1, int(count * validation_fraction)))
    return indices[validation_count:], indices[:validation_count], "query"


@torch.no_grad()
def evaluate(
    model: FuzzyReasoningDynamics,
    loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    for batch in loader:
        outputs = model(move_batch(batch, device))
        if loss_config.rollout_weight > 0:
            model.add_short_rollout(outputs, loss_config.rollout_horizon)
        _, metrics = fuzzy_dynamics_loss(outputs, loss_config)
        for name, value in metrics.items():
            totals[name] += float(value)
    return mean_metrics(totals, len(loader))


def main() -> None:
    args = parse_args()
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive.")
    experiment_time = (
        validate_experiment_time(args.experiment_time)
        if args.experiment_time is not None
        else generate_experiment_time()
    )
    experiment_dir = experiment_directory(args.results_root, experiment_time)
    if args.experiment_time is None:
        Path(args.results_root).mkdir(parents=True, exist_ok=True)
        experiment_dir.mkdir(exist_ok=False)
    else:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = experiment_dir / "checkpoint"
    results_dir = experiment_dir / "training"
    experiment_log_dir = Path(args.log_dir) / f"fuzzy_dynamics_{experiment_time}"
    occupied = []
    for path in (checkpoint_dir, results_dir):
        if not path.exists():
            continue
        if not path.is_dir():
            occupied.append(path)
            continue
        entries = list(path.iterdir())
        if args.experiment_time is not None and path == results_dir:
            entries = [item for item in entries if item.name != "pipeline_manifest.json"]
        if entries:
            occupied.append(path)
    if occupied:
        raise FileExistsError(
            f"Experiment time '{experiment_time}' already has artifacts in {occupied[0]}."
        )
    logger, _ = configure_experiment_logging(
        "train_fuzzy_dynamics", experiment_log_dir
    )
    logger.info("Arguments: %s", vars(args))
    logger.info("experiment_time=%s", experiment_time)
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache = load_activation_cache(args.activations)
    with Path(args.config).open(encoding="utf-8") as handle:
        experiment: dict[str, Any] = json.load(handle)

    model_values = dict(experiment["model"])
    model_values.update(
        hidden_size=int(cache["hidden"].shape[-1]),
        belief_input_dim=int(cache["belief"].shape[-1]),
        vocab_size=int(cache.get("vocab_size", 1)),
    )
    model_config = FuzzyDynamicsConfig(**model_values)
    loss_config = LossConfig(**experiment.get("loss", {}))
    training = experiment["training"]

    count = cache["hidden"].shape[0]
    if count < 2:
        raise ValueError("At least two trajectories are required for train/validation splitting.")
    train_indices, validation_indices, split_by = split_indices(
        count,
        cache.get("metadata", []),
        training.get("validation_fraction", 0.1),
        args.seed,
        training.get("split_by"),
    )
    logger.info(
        f"split_by={split_by} train={len(train_indices)} validation={len(validation_indices)}"
    )
    train_loader = DataLoader(
        ActivationTrajectoryDataset(cache, train_indices),
        batch_size=training["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        ActivationTrajectoryDataset(cache, validation_indices),
        batch_size=training["batch_size"],
        shuffle=False,
    )

    model = FuzzyReasoningDynamics(model_config).to(device)
    projector_fit_samples = int(training.get("projector_fit_samples", 8192))
    hidden_samples, belief_samples = sample_training_states(
        cache, train_indices, projector_fit_samples, args.seed
    )
    logger.info(
        "Fitting shared frozen PCA projectors on %d training states across all layers",
        hidden_samples.shape[0],
    )
    model.fit_state_projectors(hidden_samples, belief_samples)
    delta_scale = fit_state_delta_scale(model, train_loader, device)
    model.set_state_delta_scale(delta_scale)
    logger.info(
        "Fitted block-balanced delta scales: min=%.6g median=%.6g max=%.6g",
        float(delta_scale.min()),
        float(delta_scale.median()),
        float(delta_scale.max()),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.01),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints: %s", checkpoint_dir.resolve())
    logger.info("Training results: %s", results_dir.resolve())
    history = []
    split_path = results_dir / "split.json"
    history_path = results_dir / "history.json"
    manifest_path = results_dir / "experiment_manifest.json"
    write_json_atomic(
        split_path,
        {"split_by": split_by, "train": train_indices, "validation": validation_indices},
    )
    write_json_atomic(history_path, history)
    experiment_manifest = {
        "schema_version": 1,
        "experiment_time": experiment_time,
        "status": "running",
        "seed": args.seed,
        "activations": str(Path(args.activations).resolve()),
        "config": str(Path(args.config).resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "experiment_dir": str(experiment_dir.resolve()),
        "training_results_dir": str(results_dir.resolve()),
        "evaluation_results_dir": str(
            (experiment_dir / "evaluation").resolve()
        ),
        "log_dir": str(experiment_log_dir.resolve()),
        "split_file": str(split_path.resolve()),
        "history_file": str(history_path.resolve()),
    }
    write_json_atomic(manifest_path, experiment_manifest)
    best_validation = float("inf")

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        totals: dict[str, float] = defaultdict(float)
        for batch_number, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(move_batch(batch, device))
            if loss_config.rollout_weight > 0:
                model.add_short_rollout(outputs, loss_config.rollout_horizon)
            loss, metrics = fuzzy_dynamics_loss(outputs, loss_config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("gradient_clip", 1.0))
            optimizer.step()
            for name, value in metrics.items():
                totals[name] += float(value)
            if (
                batch_number % args.log_every_batches == 0
                or batch_number == len(train_loader)
            ):
                logger.info(
                    "epoch=%03d train_batch=%d/%d running_loss=%.6f",
                    epoch,
                    batch_number,
                    len(train_loader),
                    totals["loss"] / batch_number,
                )

        train_metrics = mean_metrics(totals, len(train_loader))
        validation_metrics = evaluate(model, validation_loader, loss_config, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        logger.info(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f}"
        )

        checkpoint = model.checkpoint(
            loss_config=loss_config.to_dict(),
            source_activations=str(Path(args.activations).resolve()),
            experiment_time=experiment_time,
            split_file=str(split_path.resolve()),
            training_results_dir=str(results_dir.resolve()),
            epoch=epoch,
            validation_metrics=validation_metrics,
        )
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            logger.info("epoch=%03d saved new best.pt", epoch)
        write_json_atomic(history_path, history)

    experiment_manifest.update(status="complete", best_validation_loss=best_validation)
    write_json_atomic(manifest_path, experiment_manifest)
    logger.info(
        "Best validation loss: %.6f; checkpoints saved in %s",
        best_validation,
        checkpoint_dir,
    )


if __name__ == "__main__":
    main()
