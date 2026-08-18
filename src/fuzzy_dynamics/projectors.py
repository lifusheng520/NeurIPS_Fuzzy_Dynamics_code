"""Shared, train-fitted projections from activations to reasoning variables."""

from __future__ import annotations

import torch
from torch import nn


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class FrozenPCAProjector(nn.Module):
    """One PCA coordinate system fitted on training examples across all layers."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim > input_dim:
            raise ValueError("PCA output_dim cannot exceed input_dim.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("components", torch.zeros(output_dim, input_dim))
        self.register_buffer("scale", torch.ones(output_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, samples: torch.Tensor) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.input_dim:
            raise ValueError(
                f"PCA samples must have shape [N, {self.input_dim}], got {tuple(samples.shape)}."
            )
        if samples.shape[0] <= self.output_dim:
            raise ValueError("PCA requires more samples than output dimensions.")
        values = samples.detach().to(device=self.mean.device, dtype=torch.float32)
        mean = values.mean(dim=0)
        centered = values - mean
        _, singular_values, vectors = torch.pca_lowrank(
            centered, q=self.output_dim, center=False, niter=3
        )
        scale = singular_values / max(1, values.shape[0] - 1) ** 0.5
        self.mean.copy_(mean)
        self.components.copy_(vectors.T)
        self.scale.copy_(scale.clamp_min(1e-6))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("FrozenPCAProjector must be fitted on the training split first.")
        projected = torch.matmul(values.float() - self.mean, self.components.T)
        return (projected / self.scale).to(values.dtype)


class ReasoningProjectors(nn.Module):
    """Shared frozen P_h/P_c/P_b and trainable operation projectors P_a/P_m."""

    def __init__(
        self,
        hidden_size: int,
        belief_input_dim: int,
        z_dim: int,
        concept_dim: int,
        belief_dim: int,
        operation_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.concept_dim = concept_dim
        self.hidden_state = FrozenPCAProjector(hidden_size, z_dim + concept_dim)
        self.belief = FrozenPCAProjector(belief_input_dim, belief_dim)
        self.attention = FeatureProjector(hidden_size, operation_dim, dropout)
        self.mlp = FeatureProjector(hidden_size, operation_dim, dropout)

    @torch.no_grad()
    def fit_state_projectors(
        self, hidden_samples: torch.Tensor, belief_samples: torch.Tensor
    ) -> None:
        self.hidden_state.fit(hidden_samples)
        self.belief.fit(belief_samples)

    def hidden_and_concept(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = self.hidden_state(values)
        return coordinates.split((self.z_dim, self.concept_dim), dim=-1)
