"""Quantitative fidelity, semantic-alignment, and rollout metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.fuzzy_dynamics.config import REASONING_MODES


STATE_COMPONENTS = ("z", "concept", "belief", "uncertainty")


def _validate_pair(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim < 2:
        raise ValueError("Metric tensors must include samples and a feature dimension.")
    if predicted.shape[-1] == 0:
        raise ValueError("The feature dimension cannot be empty.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Regression metrics require finite predicted and target values.")


def _r2_score(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    """Variance-weighted multi-output R², or ``None`` for a constant target."""
    _validate_pair(predicted, target)
    predicted64 = predicted.to(torch.float64)
    target64 = target.to(torch.float64)
    reduce_dims = tuple(range(target64.ndim - 1))
    target_mean = target64.mean(dim=reduce_dims, keepdim=True)
    total_sum_squares = (target64 - target_mean).square().sum()
    if float(total_sum_squares) == 0.0:
        return None
    residual_sum_squares = (predicted64 - target64).square().sum()
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _regression_summary(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    _validate_pair(predicted, target)
    residual = predicted - target
    r2 = _r2_score(predicted, target)
    reduce_dims = tuple(range(target.ndim - 1))
    centered_target = target.to(torch.float64) - target.to(torch.float64).mean(
        dim=reduce_dims, keepdim=True
    )
    return {
        "mse": float(residual.square().mean()),
        "mae": float(residual.abs().mean()),
        "r2": r2,
        "r2_defined": r2 is not None,
        "r2_reason": None if r2 is not None else "constant_target",
        "target_sum_squares": float(centered_target.square().sum()),
        "target_variance": float(centered_target.square().mean()),
        "cosine_similarity": float(F.cosine_similarity(predicted, target, dim=-1).mean()),
    }


def _component_slices(
    dimensions: Mapping[str, int], state_dim: int
) -> dict[str, slice]:
    missing = [name for name in STATE_COMPONENTS if name not in dimensions]
    if missing:
        raise ValueError(f"Missing state-component dimensions: {', '.join(missing)}")
    slices: dict[str, slice] = {}
    start = 0
    for name in STATE_COMPONENTS:
        size = int(dimensions[name])
        if size <= 0:
            raise ValueError(f"State-component dimension '{name}' must be positive.")
        slices[name] = slice(start, start + size)
        start += size
    if start != state_dim:
        raise ValueError(
            f"State-component dimensions sum to {start}, but the state dimension is {state_dim}."
        )
    return slices


def reconstruction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    component_dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate one-step state-transition reconstruction.

    R² centers each state coordinate independently over queries and layers.  If
    component dimensions are supplied, the function additionally reports the
    four state-block scores and their unweighted macro average.
    """
    metrics: dict[str, Any] = _regression_summary(predicted, target)
    metrics["per_layer"] = []
    for layer in range(predicted.shape[1]):
        layer_summary = _regression_summary(predicted[:, layer], target[:, layer])
        metrics["per_layer"].append({"layer": layer, **layer_summary})

    if component_dimensions is not None:
        slices = _component_slices(component_dimensions, predicted.shape[-1])
        components = {
            name: _regression_summary(predicted[..., section], target[..., section])
            for name, section in slices.items()
        }
        metrics["components"] = components
        component_r2 = [value["r2"] for value in components.values() if value["r2"] is not None]
        metrics["macro_r2"] = (
            sum(component_r2) / len(component_r2)
            if len(component_r2) == len(components)
            else None
        )
        metrics["macro_r2_available_components"] = (
            sum(component_r2) / len(component_r2) if component_r2 else None
        )
        metrics["macro_r2_defined_components"] = len(component_r2)
    return metrics


def rollout_metrics(
    predicted_states: torch.Tensor,
    target_states: torch.Tensor,
    component_dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-layer rollout, excluding the supplied initial state."""
    _validate_pair(predicted_states, target_states)
    if predicted_states.ndim != 3:
        raise ValueError("Rollout states must have shape [queries, layers + 1, state_dim].")
    if predicted_states.shape[1] < 2:
        raise ValueError("A rollout must contain an initial state and at least one prediction.")
    if not torch.allclose(predicted_states[:, 0], target_states[:, 0]):
        raise ValueError("Rollout evaluation requires predicted s_0 to equal the observed s_0.")

    predicted = predicted_states[:, 1:]
    target = target_states[:, 1:]
    squared_l2 = (predicted - target).square().sum(dim=-1)
    summary: dict[str, Any] = {
        "kind": "conditional_on_observed_attention_and_mlp",
        "initial_state_mse": 0.0,
        "rollout_error": float(squared_l2.mean()),
        "mse": float((predicted - target).square().mean()),
        "rmse": float((predicted - target).square().mean().sqrt()),
        "final_state_squared_l2": float(squared_l2[:, -1].mean()),
        "final_state_mse": float((predicted[:, -1] - target[:, -1]).square().mean()),
        "per_horizon": [],
    }
    for horizon in range(predicted.shape[1]):
        horizon_summary = _regression_summary(predicted[:, horizon], target[:, horizon])
        summary["per_horizon"].append(
            {
                "horizon": horizon + 1,
                "squared_l2": float(squared_l2[:, horizon].mean()),
                **horizon_summary,
            }
        )

    if component_dimensions is not None:
        slices = _component_slices(component_dimensions, predicted.shape[-1])
        components = {
            name: _regression_summary(predicted[..., section], target[..., section])
            for name, section in slices.items()
        }
        summary["components"] = components
        component_r2 = [value["r2"] for value in components.values() if value["r2"] is not None]
        summary["macro_r2"] = (
            sum(component_r2) / len(component_r2)
            if len(component_r2) == len(components)
            else None
        )
        summary["macro_r2_available_components"] = (
            sum(component_r2) / len(component_r2) if component_r2 else None
        )
        summary["macro_r2_defined_components"] = len(component_r2)
    return summary


def binary_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute tie-aware AUROC and average precision without sklearn."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape.")
    valid = torch.isfinite(scores)
    if labels.is_floating_point():
        valid &= torch.isfinite(labels)
    if valid_mask is not None:
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must have the same shape as scores.")
        valid &= valid_mask.bool()

    flat_scores = scores[valid].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    selected_labels = labels[valid].detach().to(device="cpu").reshape(-1)
    if not torch.all((selected_labels == 0) | (selected_labels == 1)):
        raise ValueError("labels must contain only binary 0/1 values on valid entries.")
    flat_labels = selected_labels.bool()
    total = int(flat_labels.numel())
    positives = int(flat_labels.sum())
    negatives = total - positives
    result: dict[str, Any] = {
        "available": positives > 0 and negatives > 0,
        "num_examples": total,
        "num_positive": positives,
        "num_negative": negatives,
        "prevalence": positives / total if total else None,
        "auroc": None,
        "average_precision": None,
    }
    if total == 0:
        result["reason"] = "no_valid_events"
        return result
    if positives == 0 or negatives == 0:
        result["reason"] = "both_positive_and_negative_events_are_required"
        return result

    # Mann-Whitney U with average ranks gives an AUROC that is correct under ties.
    ascending = torch.argsort(flat_scores, stable=True)
    sorted_scores = flat_scores[ascending]
    sorted_labels = flat_labels[ascending]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
    average_ranks = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_labels].sum()
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    # Threshold-grouped AP is deterministic when multiple examples share a score.
    descending = torch.argsort(flat_scores, descending=True, stable=True)
    desc_scores = flat_scores[descending]
    desc_labels = flat_labels[descending]
    _, desc_counts = torch.unique_consecutive(desc_scores, return_counts=True)
    group_starts = torch.cat((torch.zeros(1, dtype=torch.long), desc_counts.cumsum(0)[:-1]))
    group_positives = torch.stack(
        [
            desc_labels[start : start + count].sum()
            for start, count in zip(group_starts.tolist(), desc_counts.tolist())
        ]
    ).to(torch.float64)
    cumulative_positives = group_positives.cumsum(0)
    cumulative_total = desc_counts.cumsum(0).to(torch.float64)
    precision = cumulative_positives / cumulative_total
    average_precision = ((group_positives / positives) * precision).sum()

    result["auroc"] = float(auc)
    result["average_precision"] = float(average_precision)
    return result


def _scalar_layers(values: torch.Tensor, expected_layers: int, name: str) -> torch.Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2 or values.shape[1] != expected_layers:
        raise ValueError(
            f"{name} must have shape [queries, {expected_layers}] or "
            f"[queries, {expected_layers}, 1]."
        )
    return values


def automatic_semantic_event_scores(
    attention: torch.Tensor,
    mlp: torch.Tensor,
    uncertainty: torch.Tensor,
    bridge_logprob: torch.Tensor | None = None,
    answer_logprob: torch.Tensor | None = None,
    include_operation_proxies: bool = False,
) -> dict[str, dict[str, Any]]:
    """Construct the event proxies proposed in ``ideas/evaluation metrics.md``.

    F3/F4/F5 are built by default because those are the modes for which the
    ideas document defines events. Optional F1/F2 operation-norm proxies overlap
    even more directly with the semantic training prior. All automatic signals
    measure proxy agreement, not independent semantic validity. Independent
    annotated or intervention-derived labels can instead be passed directly to
    :func:`semantic_alignment_metrics`.
    """
    if attention.ndim != 3 or mlp.shape != attention.shape:
        raise ValueError("attention and mlp must have matching [queries, layers, dim] shapes.")
    num_layers = attention.shape[1]
    uncertainty_values = _scalar_layers(uncertainty, num_layers + 1, "uncertainty")
    result: dict[str, dict[str, Any]] = {}
    if include_operation_proxies:
        result.update(
            {
                "knowledge_enrichment": {
                    "scores": mlp.norm(dim=-1),
                    "valid_mask": torch.isfinite(mlp).all(dim=-1),
                    "source": "high_raw_mlp_output_norm_training_prior_proxy",
                    "positive_only": False,
                    "independent": False,
                },
                "information_routing": {
                    "scores": attention.norm(dim=-1),
                    "valid_mask": torch.isfinite(attention).all(dim=-1),
                    "source": "high_raw_attention_output_norm_training_prior_proxy",
                    "positive_only": False,
                    "independent": False,
                },
            }
        )

    bridge_change = None
    if bridge_logprob is not None:
        bridge = _scalar_layers(bridge_logprob, num_layers + 1, "bridge_logprob")
        bridge_change = bridge[:, 1:] - bridge[:, :-1]
        result["concept_composition"] = {
            "scores": bridge_change,
            "valid_mask": torch.isfinite(bridge_change),
            "source": "positive_first_token_bridge_log_probability_change",
            "positive_only": True,
            "independent": False,
        }

    answer_change = None
    if answer_logprob is not None:
        answer = _scalar_layers(answer_logprob, num_layers + 1, "answer_logprob")
        answer_change = answer[:, 1:] - answer[:, :-1]
        uncertainty_drop = uncertainty_values[:, :-1] - uncertainty_values[:, 1:]
        commitment_score = (
            answer_change.clamp_min(0.0) * uncertainty_drop.clamp_min(0.0)
        ).sqrt()
        result["prediction_refinement"] = {
            "scores": commitment_score,
            "valid_mask": torch.isfinite(answer_change) & torch.isfinite(uncertainty_drop),
            "source": "joint_first_token_answer_logprob_increase_and_uncertainty_decrease",
            "positive_only": True,
            "independent": False,
        }

    if bridge_change is not None and answer_change is not None:
        transition_score = (
            (-bridge_change).clamp_min(0.0) * answer_change.clamp_min(0.0)
        ).sqrt()
        result["hop_transition"] = {
            "scores": transition_score,
            "valid_mask": torch.isfinite(bridge_change) & torch.isfinite(answer_change),
            "source": "joint_first_token_bridge_logprob_decrease_and_answer_logprob_increase",
            "positive_only": True,
            "independent": False,
        }
    return result


def _quantile_event_labels(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float,
    positive_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("event_quantile must be strictly between 0 and 1.")
    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("Event scores and masks must have shape [queries, layers].")
    labels = torch.zeros_like(scores, dtype=torch.bool)
    usable = valid_mask.bool() & torch.isfinite(scores)
    if positive_only:
        usable &= scores > 0
    for query_index in range(scores.shape[0]):
        row_mask = usable[query_index]
        if not row_mask.any():
            continue
        threshold = torch.quantile(scores[query_index, row_mask].float(), quantile)
        labels[query_index] = row_mask & (scores[query_index] >= threshold)
    return labels, valid_mask.bool() & torch.isfinite(scores)


def semantic_alignment_metrics(
    memberships: torch.Tensor,
    event_definitions: Mapping[str, Mapping[str, Any]],
    event_quantile: float = 0.75,
) -> dict[str, Any]:
    """Compute per-mode AUROC/AP and strict five-mode macro scores.

    Each event definition may contain binary ``labels`` directly, or continuous
    ``scores`` that are converted to per-query top-quantile events.  Missing or
    single-class modes remain explicit and are not silently folded into the
    strict five-mode macro score.
    """
    if memberships.ndim != 3 or memberships.shape[-1] != len(REASONING_MODES):
        raise ValueError(
            f"memberships must have shape [queries, layers, {len(REASONING_MODES)}]."
        )
    mode_results: dict[str, Any] = {}
    available_aurocs: list[float] = []
    available_aps: list[float] = []

    for mode_index, mode in enumerate(REASONING_MODES):
        definition = event_definitions.get(mode)
        if definition is None:
            mode_results[mode] = {
                "available": False,
                "reason": "event_definition_unavailable",
                "auroc": None,
                "average_precision": None,
            }
            continue
        valid_mask = definition.get(
            "valid_mask", torch.ones_like(memberships[..., mode_index], dtype=torch.bool)
        )
        if "labels" in definition:
            labels = definition["labels"]
            label_rule = "provided_binary_labels"
        elif "scores" in definition:
            labels, valid_mask = _quantile_event_labels(
                definition["scores"],
                valid_mask,
                event_quantile,
                bool(definition.get("positive_only", False)),
            )
            label_rule = f"per_query_top_{1.0 - event_quantile:.3f}_quantile"
        else:
            raise ValueError(f"Event definition for {mode} needs 'labels' or 'scores'.")

        ranking = binary_ranking_metrics(
            memberships[..., mode_index], labels, valid_mask=valid_mask
        )
        ranking.update(
            {
                "source": definition.get("source", "unspecified"),
                "independent_of_training_prior": bool(definition.get("independent", False)),
                "label_rule": label_rule,
                "num_queries": int(memberships.shape[0]),
                "num_queries_with_valid_layers": int(valid_mask.any(dim=1).sum()),
                "num_queries_with_positive_events": int(
                    (labels.bool() & valid_mask.bool()).any(dim=1).sum()
                ),
                "coverage": float(valid_mask.float().mean()),
            }
        )
        mode_results[mode] = ranking
        if ranking["available"]:
            available_aurocs.append(ranking["auroc"])
            available_aps.append(ranking["average_precision"])

    all_available = len(available_aurocs) == len(REASONING_MODES)
    all_independent = all(
        bool(event_definitions.get(mode, {}).get("independent", False))
        for mode in REASONING_MODES
    )
    strict_available = all_available and all_independent
    available_modes = [
        mode for mode in REASONING_MODES if mode_results[mode].get("available", False)
    ]
    return {
        "modes": mode_results,
        "event_quantile": event_quantile,
        "num_available_modes": len(available_aurocs),
        "available_modes": available_modes,
        "available_mode_fraction": len(available_aurocs) / len(REASONING_MODES),
        "macro_auroc": (
            sum(available_aurocs) / len(available_aurocs) if strict_available else None
        ),
        "macro_average_precision": (
            sum(available_aps) / len(available_aps) if strict_available else None
        ),
        "diagnostic_macro_auroc_available_modes": (
            sum(available_aurocs) / len(available_aurocs) if available_aurocs else None
        ),
        "diagnostic_macro_average_precision_available_modes": (
            sum(available_aps) / len(available_aps) if available_aps else None
        ),
        "strict_macro_requires_all_five_modes": True,
        "strict_macro_requires_independent_events": True,
        "all_events_independent_of_training_prior": all_independent,
        "strict_macro_reason": (
            None
            if strict_available
            else "requires_defined_independent_events_for_all_five_modes"
        ),
    }
