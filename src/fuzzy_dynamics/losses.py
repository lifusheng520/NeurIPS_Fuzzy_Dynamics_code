"""Learning objectives for fuzzy local reasoning dynamics."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from .config import LossConfig
from .semantic_scores import prediction_refinement_score, route_score


def _standardize_signal(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = values.mean(dim=(0, 1), keepdim=True)
    std = values.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(eps)
    return (values - mean) / std


def semantic_reasoning_prior(
    outputs: dict[str, torch.Tensor], temperature: float = 1.0
) -> torch.Tensor:
    """Construct rho from operation magnitudes, commitment, and concept shifts."""
    enrichment = outputs["mlp"].norm(dim=-1)
    routing = route_score(outputs["attention"])
    composition = (outputs["attention"] * outputs["mlp"]).norm(dim=-1)
    commitment = prediction_refinement_score(
        outputs["margin"], outputs["uncertainty"]
    )
    hop = outputs["concept_change"].norm(dim=-1)
    if "bridge_logprob" in outputs:
        bridge_raw = outputs["bridge_logprob"].squeeze(-1)
        bridge_valid = torch.isfinite(bridge_raw[:, 1:]) & torch.isfinite(bridge_raw[:, :-1])
        bridge = torch.nan_to_num(bridge_raw, nan=0.0)
        bridge_change = bridge[:, 1:] - bridge[:, :-1]
        bridge_change = torch.where(bridge_valid, bridge_change, torch.zeros_like(bridge_change))
        composition = composition + bridge_change.clamp_min(0.0)
        hop = hop + bridge_change.abs()
    scores = torch.stack((enrichment, routing, composition, commitment, hop), dim=-1)
    scores = _standardize_signal(scores)
    return torch.softmax(scores / max(temperature, 1e-6), dim=-1)


def diversity_loss(local_deltas: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = F.normalize(local_deltas, dim=-1, eps=eps)
    terms = []
    for left, right in itertools.combinations(range(values.shape[-2]), 2):
        cosine = (values[..., left, :] * values[..., right, :]).sum(dim=-1)
        terms.append(cosine.square().mean())
    return torch.stack(terms).mean()


def _block_balanced_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    dimensions: dict[str, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    standardized_error = (predicted - target) / scale.detach()
    block_losses: dict[str, torch.Tensor] = {}
    start = 0
    for name in ("z", "concept", "belief", "uncertainty"):
        width = int(dimensions[name])
        block_losses[name] = standardized_error[..., start : start + width].square().mean()
        start += width
    if start != predicted.shape[-1]:
        raise ValueError("State-component dimensions do not match prediction width.")
    return torch.stack(tuple(block_losses.values())).mean(), block_losses


def fuzzy_dynamics_loss(
    outputs: dict[str, torch.Tensor], config: LossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    memberships = outputs["memberships"].clamp_min(1e-8)
    predicted_delta = outputs["predicted_delta"]
    target_delta = outputs["target_delta"]
    scale = outputs.get("state_delta_scale")
    dimensions = outputs.get("component_dimensions")
    if scale is None or dimensions is None:
        raise ValueError("Block-balanced dynamics requires delta scales and dimensions.")
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("State delta scales must be finite and positive.")
    dynamics, block_losses = _block_balanced_mse(
        predicted_delta, target_delta, scale, dimensions
    )

    rollout = torch.zeros((), device=predicted_delta.device)
    rollout_horizons: dict[str, torch.Tensor] = {}
    if config.rollout_weight > 0:
        predicted_states = outputs.get("short_rollout_predicted_states")
        target_states = outputs.get("short_rollout_target_states")
        if predicted_states is None or target_states is None:
            raise ValueError("Positive rollout_weight requires a differentiable short rollout.")
        if predicted_states.shape != target_states.shape:
            raise ValueError("Short-rollout predictions and targets must have matching shapes.")
        if predicted_states.shape[1] != config.rollout_horizon + 1:
            raise ValueError("Short-rollout output does not match rollout_horizon.")
        horizon_losses = []
        for horizon in range(2, config.rollout_horizon + 1):
            value, _ = _block_balanced_mse(
                predicted_states[:, horizon], target_states[:, horizon], scale, dimensions
            )
            rollout_horizons[f"rollout_h{horizon}"] = value
            horizon_losses.append(value)
        rollout = torch.stack(horizon_losses).mean()

    prior = semantic_reasoning_prior(outputs, config.semantic_temperature)
    semantic = (prior * (prior.clamp_min(1e-8).log() - memberships.log())).sum(-1).mean()
    diversity = diversity_loss(outputs["local_deltas"])

    # This is negative entropy. Minimizing it discourages premature hard assignments.
    fuzzy_entropy = (memberships * memberships.log()).sum(dim=-1).mean()
    mean_membership = memberships.mean(dim=(0, 1))
    uniform = torch.full_like(mean_membership, 1.0 / mean_membership.numel())
    balance = (mean_membership * (mean_membership.log() - uniform.log())).sum()
    concept_probe = torch.zeros((), device=memberships.device)
    if "bridge_logprob" in outputs:
        bridge_logprob = outputs["bridge_logprob"]
        valid = torch.isfinite(bridge_logprob)
        if valid.any():
            bridge_probability = torch.nan_to_num(bridge_logprob, nan=-100.0).exp().clamp(0.0, 1.0)
            concept_probe = F.binary_cross_entropy_with_logits(
                outputs["bridge_probe_logits"][valid], bridge_probability[valid]
            )

    total = (
        config.dynamics_weight * dynamics
        + config.rollout_weight * rollout
        + config.semantic_weight * semantic
        + config.diversity_weight * diversity
        + config.fuzzy_entropy_weight * fuzzy_entropy
        + config.balance_weight * balance
        + config.concept_probe_weight * concept_probe
    )
    metrics = {
        "loss": total.detach(),
        "dynamics": dynamics.detach(),
        **{f"dynamics_{name}": value.detach() for name, value in block_losses.items()},
        "rollout": rollout.detach(),
        **{name: value.detach() for name, value in rollout_horizons.items()},
        "semantic": semantic.detach(),
        "diversity": diversity.detach(),
        "fuzzy_entropy": (-fuzzy_entropy).detach(),
        "balance": balance.detach(),
        "concept_probe": concept_probe.detach(),
    }
    return total, metrics
