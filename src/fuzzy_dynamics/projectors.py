"""Trainable projections from model activations to reasoning variables."""

from __future__ import annotations

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

    def forward(self, values):
        return self.network(values)


class ReasoningProjectors(nn.Module):
    """Implements P_h, P_c, P_b, P_a, and P_m."""

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
        self.hidden = FeatureProjector(hidden_size, z_dim, dropout)
        self.concept = FeatureProjector(hidden_size, concept_dim, dropout)
        self.belief = FeatureProjector(belief_input_dim, belief_dim, dropout)
        self.attention = FeatureProjector(hidden_size, operation_dim, dropout)
        self.mlp = FeatureProjector(hidden_size, operation_dim, dropout)

