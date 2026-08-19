#!/usr/bin/env python3
"""Evaluate fuzzy-dynamics fidelity, semantic alignment, and rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy  # Import before PyTorch for binary-extension compatibility on HPC nodes.
import torch

from scripts.train_fuzzy_dynamics import resolve_device
from src.data.activation_data import load_activation_cache
from src.evaluation import (
    aggregate_membership_analysis,
    automatic_semantic_event_scores,
    build_trajectory_records,
    load_semantic_event_definitions,
    reconstruction_metrics,
    rollout_metrics,
    semantic_alignment_metrics,
)
from src.fuzzy_dynamics import FuzzyDynamicsConfig, FuzzyReasoningDynamics
from src.fuzzy_dynamics.losses import semantic_reasoning_prior
from src.utils import (
    configure_experiment_logging,
    experiment_directory,
    generate_experiment_time,
    infer_experiment_time_from_checkpoint,
    latest_experiment_time,
    validate_experiment_time,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", default=None)
    selection.add_argument("--experiment-time", default=None)
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--activations", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation-result directory override.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Training split.json. With --subset auto, its validation subset is used.",
    )
    parser.add_argument(
        "--subset",
        choices=("auto", "all", "train", "validation"),
        default="auto",
        help="auto uses the adjacent validation split for the training activation cache.",
    )
    parser.add_argument(
        "--semantic-events",
        default=None,
        help="Optional independent .json/.jsonl transition-event annotations.",
    )
    parser.add_argument("--event-id-field", default="uid")
    parser.add_argument("--allow-missing-events", action="store_true")
    parser.add_argument("--semantic-event-quantile", type=float, default=0.75)
    parser.add_argument(
        "--include-operation-proxies",
        action="store_true",
        help="Also diagnose F1/F2 against MLP/attention norms; never treated as independent.",
    )
    parser.add_argument("--skip-semantic-alignment", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser.parse_args()


def _resolved(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def _load_selection(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    count: int,
) -> tuple[list[int], dict[str, Any]]:
    activation_path = _resolved(args.activations)
    source_path = _resolved(checkpoint.get("source_activations"))
    same_as_training_cache = source_path == activation_path if source_path else None
    stored_split = _resolved(checkpoint.get("split_file"))
    legacy_adjacent_split = _resolved(Path(args.checkpoint).parent / "split.json")
    adjacent_split = (
        stored_split
        if stored_split is not None and stored_split.exists()
        else legacy_adjacent_split
    )
    split_path = _resolved(args.split_file)

    if split_path is None and args.subset == "auto" and same_as_training_cache:
        if adjacent_split is not None and adjacent_split.exists():
            split_path = adjacent_split

    subset = args.subset
    if subset == "auto":
        subset = "validation" if split_path is not None else "all"

    split_by = None
    if subset == "all":
        indices = list(range(count))
    else:
        if split_path is None:
            raise ValueError(f"--subset {subset} requires --split-file.")
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        with split_path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        train_indices = split.get("train")
        validation_indices = split.get("validation")
        if not isinstance(train_indices, list) or not isinstance(validation_indices, list):
            raise ValueError("Split file must contain train and validation index lists.")
        train_set = {int(index) for index in train_indices}
        validation_set = {int(index) for index in validation_indices}
        if train_set & validation_set:
            raise ValueError("Training and validation split indices overlap.")
        split_invalid = [
            index
            for index in train_set | validation_set
            if index < 0 or index >= count
        ]
        if split_invalid:
            raise IndexError(
                f"Split indices outside [0, {count}): {split_invalid[:5]}"
            )
        if subset not in split or not isinstance(split[subset], list):
            raise ValueError(f"Split file has no list named '{subset}'.")
        indices = [int(index) for index in split[subset]]
        split_by = split.get("split_by")

    if not indices:
        raise ValueError("The selected evaluation subset is empty.")
    if len(indices) != len(set(indices)):
        raise ValueError("Evaluation indices contain duplicates.")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise IndexError(f"Evaluation indices outside [0, {count}): {invalid[:5]}")

    if subset == "validation" and same_as_training_cache:
        holdout_status = "verified_training_split_validation"
    elif subset == "validation":
        holdout_status = "validation_indices_on_unverified_cache"
    elif same_as_training_cache:
        holdout_status = "in_sample_or_mixed_training_cache"
    else:
        holdout_status = "external_cache_not_verified_against_training_queries"
    index_digest = hashlib.sha256(
        ",".join(map(str, indices)).encode("utf-8")
    ).hexdigest()
    selection = {
        "subset": subset,
        "split_file": str(split_path) if split_path is not None else None,
        "split_by": split_by,
        "selected_count": len(indices),
        "cache_count": count,
        "selected_indices_sha256": index_digest,
        "same_as_training_activation_cache": same_as_training_cache,
        "holdout_status": holdout_status,
    }
    return indices, selection


def _selected_metadata(
    cache: dict[str, Any], indices: list[int]
) -> list[dict[str, Any]]:
    metadata = cache.get("metadata", [])
    if len(metadata) != cache["hidden"].shape[0]:
        metadata = [{"index": index, "uid": index} for index in range(cache["hidden"].shape[0])]
    selected = []
    for index in indices:
        item = dict(metadata[index]) if isinstance(metadata[index], dict) else {}
        item.setdefault("index", index)
        item.setdefault("uid", item["index"])
        item["cache_index"] = index
        selected.append(item)
    return selected


def _state_dimensions(config: FuzzyDynamicsConfig) -> dict[str, int]:
    return {
        "z": config.z_dim,
        "concept": config.concept_dim,
        "belief": config.belief_dim,
        "uncertainty": 1,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive.")
    loaded_checkpoint: dict[str, Any] | None = None
    if args.experiment_time is not None:
        experiment_time = validate_experiment_time(args.experiment_time)
        checkpoint_path = (
            experiment_directory(args.results_root, experiment_time)
            / "checkpoint"
            / "best.pt"
        )
    elif args.latest:
        experiment_time = latest_experiment_time(args.results_root)
        checkpoint_path = (
            experiment_directory(args.results_root, experiment_time)
            / "checkpoint"
            / "best.pt"
        )
    else:
        checkpoint_path = Path(args.checkpoint)
        try:
            experiment_time = infer_experiment_time_from_checkpoint(checkpoint_path)
        except ValueError:
            loaded_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            stored_time = loaded_checkpoint.get("experiment_time")
            experiment_time = (
                validate_experiment_time(stored_time)
                if stored_time is not None
                else generate_experiment_time()
            )
    output_dir = Path(
        args.output_dir
        or experiment_directory(args.results_root, experiment_time) / "evaluation"
    )
    args.checkpoint = str(checkpoint_path)
    experiment_log_dir = Path(args.log_dir) / f"fuzzy_dynamics_{experiment_time}"
    logger, _ = configure_experiment_logging(
        "evaluate_fuzzy_dynamics", experiment_log_dir
    )
    logger.info("Arguments: %s", vars(args))
    logger.info("experiment_time=%s", experiment_time)
    device = resolve_device(args.device)
    cache = load_activation_cache(args.activations)
    checkpoint = loaded_checkpoint or torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    stored_time = checkpoint.get("experiment_time")
    if stored_time is not None and validate_experiment_time(stored_time) != experiment_time:
        raise ValueError("Checkpoint experiment_time does not match the selected directory.")
    config = FuzzyDynamicsConfig.from_dict(checkpoint["model_config"])
    model = FuzzyReasoningDynamics(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices, selection = _load_selection(args, checkpoint, cache["hidden"].shape[0])
    metadata = _selected_metadata(cache, indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "subset=%s selected=%d holdout_status=%s",
        selection["subset"],
        selection["selected_count"],
        selection["holdout_status"],
    )
    logger.info("Evaluation outputs: %s", output_dir.resolve())

    all_predicted: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    all_memberships: list[torch.Tensor] = []
    all_local_deltas: list[torch.Tensor] = []
    all_priors: list[torch.Tensor] = []
    rollout_predicted: list[torch.Tensor] = []
    rollout_target: list[torch.Tensor] = []
    raw_attention: list[torch.Tensor] = []
    raw_mlp: list[torch.Tensor] = []
    uncertainty: list[torch.Tensor] = []
    margin: list[torch.Tensor] = []
    bridge_logprob: list[torch.Tensor] = []
    answer_logprob: list[torch.Tensor] = []
    all_records: list[dict[str, Any]] = []

    with torch.no_grad():
        starts = range(0, len(indices), args.batch_size)
        total_batches = len(starts)
        for batch_number, start in enumerate(starts, start=1):
            batch_indices = indices[start : start + args.batch_size]
            index_tensor = torch.tensor(batch_indices, dtype=torch.long)
            names = ("hidden", "attention", "mlp", "belief", "uncertainty", "margin")
            optional_names = ("bridge_logprob", "answer_logprob")
            batch = {
                name: cache[name].index_select(0, index_tensor).to(device) for name in names
            }
            batch.update(
                {
                    name: cache[name].index_select(0, index_tensor).to(device)
                    for name in optional_names
                    if name in cache
                }
            )
            outputs = model(batch)
            predicted = outputs["predicted_delta"].cpu()
            target = outputs["target_delta"].cpu()
            memberships = outputs["memberships"].cpu()
            all_predicted.append(predicted)
            all_target.append(target)
            all_memberships.append(memberships)
            all_local_deltas.append(outputs["local_deltas"].cpu())
            all_priors.append(semantic_reasoning_prior(outputs).cpu())
            all_records.extend(
                build_trajectory_records(
                    memberships,
                    predicted,
                    target,
                    outputs["uncertainty"].cpu(),
                    metadata,
                    offset=start,
                )
            )

            if not args.skip_rollout:
                rollout = model.conditional_rollout(batch)
                rollout_predicted.append(rollout["predicted_states"].cpu())
                rollout_target.append(rollout["true_states"].cpu())
            if not args.skip_semantic_alignment and args.semantic_events is None:
                raw_attention.append(batch["attention"].cpu())
                raw_mlp.append(batch["mlp"].cpu())
                uncertainty.append(batch["uncertainty"].cpu())
                margin.append(batch["margin"].cpu())
                if "bridge_logprob" in batch:
                    bridge_logprob.append(batch["bridge_logprob"].cpu())
                if "answer_logprob" in batch:
                    answer_logprob.append(batch["answer_logprob"].cpu())
            if batch_number % args.log_every_batches == 0 or batch_number == total_batches:
                logger.info(
                    "batch=%d/%d evaluated_queries=%d/%d",
                    batch_number,
                    total_batches,
                    min(start + len(batch_indices), len(indices)),
                    len(indices),
                )

    predicted_tensor = torch.cat(all_predicted)
    target_tensor = torch.cat(all_target)
    memberships_tensor = torch.cat(all_memberships)
    dimensions = _state_dimensions(config)
    one_step = reconstruction_metrics(predicted_tensor, target_tensor, dimensions)
    fidelity: dict[str, Any] = {"one_step": one_step}
    if not args.skip_rollout:
        fidelity["conditional_rollout"] = rollout_metrics(
            torch.cat(rollout_predicted), torch.cat(rollout_target), dimensions
        )

    event_manifest: dict[str, Any] | None = None
    semantic_alignment: dict[str, Any] | None = None
    if not args.skip_semantic_alignment:
        if args.semantic_events is not None:
            event_definitions, event_manifest = load_semantic_event_definitions(
                args.semantic_events,
                metadata,
                num_layers=memberships_tensor.shape[1],
                id_field=args.event_id_field,
                allow_missing=args.allow_missing_events,
            )
        else:
            event_definitions = automatic_semantic_event_scores(
                torch.cat(raw_attention),
                torch.cat(raw_mlp),
                torch.cat(uncertainty),
                torch.cat(margin),
                bridge_logprob=torch.cat(bridge_logprob) if bridge_logprob else None,
                answer_logprob=torch.cat(answer_logprob) if answer_logprob else None,
                include_operation_proxies=args.include_operation_proxies,
            )
            event_manifest = {
                "kind": "automatic_training_prior_overlapping_proxies",
                "independent_of_training": False,
                "layer_axis": "transition_s_l_to_s_l_plus_1",
                "include_operation_proxies": args.include_operation_proxies,
                "prediction_refinement_score": (
                    "sqrt(relu(margin_change)*relu(uncertainty_drop))"
                ),
                "bridge_and_answer_scores_used_for": [
                    "concept_composition",
                    "hop_transition",
                ],
            }
        semantic_alignment = semantic_alignment_metrics(
            memberships_tensor,
            event_definitions,
            event_quantile=args.semantic_event_quantile,
        )

    manifest = {
        "schema_version": 3,
        "experiment_time": experiment_time,
        "checkpoint": str(_resolved(args.checkpoint)),
        "activations": str(_resolved(args.activations)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cache_model_name": cache.get("model_name"),
        "device": str(device),
        "selection": selection,
        "semantic_events": event_manifest,
    }
    metrics = {
        "schema_version": 3,
        "evaluation": manifest,
        "fidelity": fidelity,
        "semantic_alignment": semantic_alignment,
    }
    analysis = aggregate_membership_analysis(
        memberships_tensor, torch.cat(all_local_deltas), torch.cat(all_priors)
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    with (output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "membership_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, allow_nan=False)

    logger.info("Metrics:\n%s", json.dumps(metrics, indent=2, allow_nan=False))
    logger.info("holdout_status=%s", selection["holdout_status"])
    logger.info(
        "Saved %d trajectories to %s",
        len(all_records),
        (output_dir / "trajectories.jsonl").resolve(),
    )


if __name__ == "__main__":
    main()
