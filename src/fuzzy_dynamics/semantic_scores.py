"""Explicit semantic scores used by the paper and implementation."""

from __future__ import annotations

import torch


def route_score(attention: torch.Tensor) -> torch.Tensor:
    """Return RouteScore(a) = ||a||_2 over the attention feature dimension."""
    if attention.ndim < 2 or attention.shape[-1] == 0:
        raise ValueError("attention must include a non-empty feature dimension.")
    return attention.norm(dim=-1)


def phi_a(gamma_change: torch.Tensor, uncertainty_drop: torch.Tensor) -> torch.Tensor:
    """Combine confidence gain and uncertainty reduction as a soft conjunction.

    The paper definition is phi_A(x, y) = sqrt([x]_+ [y]_+), where
    [v]_+ = max(v, 0).  The score is positive only when both signals improve.
    """
    if gamma_change.shape != uncertainty_drop.shape:
        raise ValueError("gamma_change and uncertainty_drop must have the same shape.")
    return (
        gamma_change.clamp_min(0.0) * uncertainty_drop.clamp_min(0.0)
    ).sqrt()


def prediction_refinement_score(
    margin: torch.Tensor, uncertainty: torch.Tensor
) -> torch.Tensor:
    """Compute r^A_l = phi_A(gamma_{l+1}-gamma_l, u_l-u_{l+1})."""
    if margin.shape != uncertainty.shape:
        raise ValueError("margin and uncertainty must have the same shape.")
    if margin.ndim < 2 or margin.shape[-2] < 2:
        raise ValueError("margin and uncertainty must contain at least two layer states.")
    if margin.shape[-1] != 1:
        raise ValueError("margin and uncertainty must have a singleton scalar dimension.")
    gamma_change = margin[..., 1:, 0] - margin[..., :-1, 0]
    uncertainty_drop = uncertainty[..., :-1, 0] - uncertainty[..., 1:, 0]
    return phi_a(gamma_change, uncertainty_drop)
