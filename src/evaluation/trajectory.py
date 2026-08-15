"""Metrics and serializable fuzzy reasoning trajectories."""

from __future__ import annotations

from typing import Any

import torch

from src.fuzzy_dynamics.config import REASONING_MODES
from .metrics import reconstruction_metrics


def build_trajectory_records(
    memberships: torch.Tensor,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    uncertainty: torch.Tensor,
    metadata: list[dict[str, Any]],
    offset: int = 0,
) -> list[dict[str, Any]]:
    records = []
    for batch_index in range(memberships.shape[0]):
        item_metadata = metadata[offset + batch_index] if metadata else {"index": offset + batch_index}
        layers = []
        for layer_index in range(memberships.shape[1]):
            mu = memberships[batch_index, layer_index]
            dominant_index = int(mu.argmax())
            layers.append(
                {
                    "layer": layer_index,
                    "membership": {
                        name: float(mu[mode_index])
                        for mode_index, name in enumerate(REASONING_MODES)
                    },
                    "dominant_mode": REASONING_MODES[dominant_index],
                    "uncertainty": float(uncertainty[batch_index, layer_index, 0]),
                    "predicted_delta_norm": float(
                        predicted_delta[batch_index, layer_index].norm()
                    ),
                    "target_delta_norm": float(target_delta[batch_index, layer_index].norm()),
                }
            )
        records.append({**item_metadata, "trajectory": layers})
    return records


def aggregate_membership_analysis(
    memberships: torch.Tensor,
    local_deltas: torch.Tensor,
    semantic_priors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Aggregate layer usage, hard transitions, and weighted mode contributions."""
    mean_by_layer = memberships.mean(dim=0)
    usage = memberships.mean(dim=(0, 1))
    dominant = memberships.argmax(dim=-1)
    transition_counts = torch.zeros(5, 5, dtype=torch.float64)
    for left, right in zip(dominant[:, :-1].reshape(-1), dominant[:, 1:].reshape(-1)):
        transition_counts[int(left), int(right)] += 1
    row_totals = transition_counts.sum(dim=1, keepdim=True)
    transition_probabilities = transition_counts / row_totals.clamp_min(1.0)
    weighted_contribution = (
        memberships * local_deltas.norm(dim=-1)
    ).mean(dim=(0, 1))
    membership_entropy = -(memberships.clamp_min(1e-8) * memberships.clamp_min(1e-8).log()).sum(-1)
    result: dict[str, Any] = {
        "mode_usage": {
            name: float(usage[index]) for index, name in enumerate(REASONING_MODES)
        },
        "weighted_mode_contribution": {
            name: float(weighted_contribution[index])
            for index, name in enumerate(REASONING_MODES)
        },
        "mean_membership_by_layer": [
            {name: float(row[index]) for index, name in enumerate(REASONING_MODES)}
            for row in mean_by_layer
        ],
        "dominant_transition_counts": transition_counts.tolist(),
        "dominant_transition_probabilities": transition_probabilities.tolist(),
        "mode_order": list(REASONING_MODES),
        "mean_membership_entropy": float(membership_entropy.mean()),
        "effective_number_of_modes": float(membership_entropy.mean().exp()),
    }
    if semantic_priors is not None:
        correlations = {}
        for index, name in enumerate(REASONING_MODES):
            membership_values = memberships[..., index].reshape(-1)
            prior_values = semantic_priors[..., index].reshape(-1)
            membership_centered = membership_values - membership_values.mean()
            prior_centered = prior_values - prior_values.mean()
            denominator = (
                membership_centered.square().sum().sqrt()
                * prior_centered.square().sum().sqrt()
            ).clamp_min(1e-12)
            correlations[name] = float(
                (membership_centered * prior_centered).sum() / denominator
            )
        kl = (
            semantic_priors.clamp_min(1e-8)
            * (
                semantic_priors.clamp_min(1e-8).log()
                - memberships.clamp_min(1e-8).log()
            )
        ).sum(-1).mean()
        result["training_prior_agreement"] = {
            "pearson_correlation": correlations,
            "kl_prior_to_membership": float(kl),
            "independent_semantic_validation": False,
        }
    return result
