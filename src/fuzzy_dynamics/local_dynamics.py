"""Five reasoning-semantic local dynamical systems."""

from __future__ import annotations

import torch
from torch import nn


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class LocalReasoningDynamics(nn.Module):
    """Structured implementations of F1 through F5 from the paper proposal."""

    def __init__(
        self,
        z_dim: int,
        concept_dim: int,
        belief_dim: int,
        operation_dim: int,
        state_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.knowledge_enrichment = DynamicsMLP(
            z_dim + operation_dim, hidden_dim, state_dim, dropout
        )
        self.information_routing = DynamicsMLP(
            z_dim + operation_dim, hidden_dim, state_dim, dropout
        )
        self.concept_composition = DynamicsMLP(
            z_dim + concept_dim + 3 * operation_dim,
            hidden_dim,
            state_dim,
            dropout,
        )
        self.prediction_refinement = DynamicsMLP(
            z_dim + belief_dim + 1, hidden_dim, state_dim, dropout
        )
        self.hop_transition = DynamicsMLP(
            z_dim + 2 * concept_dim + operation_dim,
            hidden_dim,
            state_dim,
            dropout,
        )

    def forward(
        self,
        z: torch.Tensor,
        concept: torch.Tensor,
        belief: torch.Tensor,
        uncertainty: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
        concept_change: torch.Tensor,
    ) -> torch.Tensor:
        f1 = self.knowledge_enrichment(torch.cat((z, mlp), dim=-1))
        f2 = self.information_routing(torch.cat((z, attention), dim=-1))
        f3 = self.concept_composition(
            torch.cat((z, concept, attention, mlp, attention * mlp), dim=-1)
        )
        f4 = self.prediction_refinement(torch.cat((z, belief, uncertainty), dim=-1))
        f5 = self.hop_transition(
            torch.cat((z, concept, concept_change, attention), dim=-1)
        )
        return torch.stack((f1, f2, f3, f4, f5), dim=-2)

