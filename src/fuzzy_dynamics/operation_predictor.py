"""Predict attention and MLP operation coordinates for autonomous rollouts."""

from __future__ import annotations

import torch
from torch import nn


class OperationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class OperationPredictor(nn.Module):
    """Close the surrogate loop by predicting operations from its current state.

    The fixed context state anchors the prompt information. Attention is
    predicted first because the Transformer block's MLP output is downstream of
    the attention update; the predicted attention therefore also conditions the
    MLP predictor.
    """

    def __init__(
        self,
        state_dim: int,
        operation_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        common_dim = 2 * state_dim + 1
        self.attention = OperationMLP(
            common_dim, hidden_dim, operation_dim, dropout
        )
        self.mlp = OperationMLP(
            common_dim + operation_dim, hidden_dim, operation_dim, dropout
        )

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        layer_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != context.shape:
            raise ValueError("Operation state and context must have matching shapes.")
        if layer_position.shape != (*state.shape[:-1], 1):
            raise ValueError("Operation layer position must have shape [..., 1].")
        common = torch.cat((state, context, layer_position), dim=-1)
        attention = self.attention(common)
        mlp = self.mlp(torch.cat((common, attention), dim=-1))
        return attention, mlp
