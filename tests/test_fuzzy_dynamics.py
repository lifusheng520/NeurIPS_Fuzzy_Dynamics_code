from __future__ import annotations

import unittest
import types

import torch
from torch import nn

from src.fuzzy_dynamics import (
    FuzzyDynamicsConfig,
    FuzzyReasoningDynamics,
    LossConfig,
    fuzzy_dynamics_loss,
)


class IdenticalZDynamics(nn.Module):
    """Every mode predicts delta-z=z and zero for all other state blocks."""

    def forward(self, z, concept, belief, uncertainty, attention, mlp, concept_change):
        del concept, belief, uncertainty, attention, mlp, concept_change
        delta = torch.zeros(*z.shape[:-1], 4, device=z.device, dtype=z.dtype)
        delta[..., 0] = z[..., 0]
        return delta.unsqueeze(-2).expand(*delta.shape[:-1], 5, 4)


class FuzzyDynamicsTest(unittest.TestCase):
    def test_conditional_rollout_uses_predicted_not_true_intermediate_state(self) -> None:
        config = FuzzyDynamicsConfig(
            hidden_size=1,
            belief_input_dim=1,
            vocab_size=2,
            z_dim=1,
            concept_dim=1,
            belief_dim=1,
            operation_dim=1,
            dynamics_hidden_dim=2,
        )
        model = FuzzyReasoningDynamics(config)
        model.local_dynamics = IdenticalZDynamics()

        def fake_project_batch(self, batch):
            states = batch["states"]
            zeros = torch.zeros(states.shape[0], states.shape[1] - 1, 1)
            return (
                states,
                states[..., 0:1],
                states[..., 1:2],
                states[..., 2:3],
                states[..., 3:4],
                zeros,
                zeros,
            )

        model._project_batch = types.MethodType(fake_project_batch, model)
        true_states = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]]]
        )

        rollout = model.conditional_rollout({"states": true_states})

        self.assertTrue(
            torch.allclose(
                rollout["predicted_states"][0, :, 0], torch.tensor([1.0, 2.0, 4.0])
            )
        )

    def test_shapes_memberships_and_backward(self) -> None:
        batch_size, layers, hidden_size, belief_size = 3, 4, 16, 8
        config = FuzzyDynamicsConfig(
            hidden_size=hidden_size,
            belief_input_dim=belief_size,
            vocab_size=101,
            z_dim=8,
            concept_dim=6,
            belief_dim=5,
            operation_dim=7,
            dynamics_hidden_dim=12,
        )
        model = FuzzyReasoningDynamics(config)
        batch = {
            "hidden": torch.randn(batch_size, layers + 1, hidden_size),
            "attention": torch.randn(batch_size, layers, hidden_size),
            "mlp": torch.randn(batch_size, layers, hidden_size),
            "belief": torch.randn(batch_size, layers + 1, belief_size),
            "uncertainty": torch.rand(batch_size, layers + 1, 1),
        }
        outputs = model(batch)
        self.assertEqual(outputs["states"].shape, (batch_size, layers + 1, config.state_dim))
        self.assertEqual(outputs["local_deltas"].shape, (batch_size, layers, 5, config.state_dim))
        self.assertEqual(outputs["memberships"].shape, (batch_size, layers, 5))
        self.assertTrue(
            torch.allclose(outputs["memberships"].sum(dim=-1), torch.ones(batch_size, layers))
        )

        model.eval()
        with torch.no_grad():
            deterministic_outputs = model(batch)
            rollout = model.conditional_rollout(batch)
        self.assertEqual(
            rollout["predicted_states"].shape,
            (batch_size, layers + 1, config.state_dim),
        )
        self.assertTrue(
            torch.allclose(rollout["predicted_states"][:, 0], deterministic_outputs["states"][:, 0])
        )
        self.assertTrue(
            torch.allclose(
                rollout["predicted_delta"][:, 0],
                deterministic_outputs["predicted_delta"][:, 0],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                rollout["memberships"][:, 0],
                deterministic_outputs["memberships"][:, 0],
                atol=1e-6,
            )
        )

        model.train()
        loss, metrics = fuzzy_dynamics_loss(outputs, LossConfig())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("dynamics", metrics)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
