"""Recurrent fuzzy reasoning-mode membership dynamics."""

from __future__ import annotations

import torch
from torch import nn


class MembershipDynamics(nn.Module):
    def __init__(
        self,
        state_dim: int,
        operation_dim: int,
        num_modes: int = 5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_modes = num_modes
        self.temperature = temperature
        self.initial_logits = nn.Parameter(torch.zeros(num_modes))
        self.transition = nn.Linear(num_modes, num_modes, bias=False)
        self.state_projection = nn.Linear(state_dim, num_modes, bias=False)
        self.attention_projection = nn.Linear(operation_dim, num_modes, bias=False)
        self.mlp_projection = nn.Linear(operation_dim, num_modes, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_modes))

        with torch.no_grad():
            self.transition.weight.copy_(torch.eye(num_modes))

    def initial_membership(self, batch_size: int) -> torch.Tensor:
        """Return the learned membership prior for a batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return torch.softmax(self.initial_logits, dim=-1).expand(batch_size, -1)

    def step(
        self,
        state: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the recurrent membership dynamics by one Transformer layer."""
        logits = (
            self.transition(previous)
            + self.state_projection(state)
            + self.attention_projection(attention)
            + self.mlp_projection(mlp)
            + self.bias
        )
        return torch.softmax(logits / self.temperature, dim=-1)

    def forward(
        self,
        states: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
    ) -> torch.Tensor:
        """Return memberships with shape [batch, layers, modes]."""
        batch_size, num_layers, _ = states.shape
        previous = self.initial_membership(batch_size)
        memberships = []
        for layer_index in range(num_layers):
            previous = self.step(
                states[:, layer_index],
                attention[:, layer_index],
                mlp[:, layer_index],
                previous,
            )
            memberships.append(previous)
        return torch.stack(memberships, dim=1)
