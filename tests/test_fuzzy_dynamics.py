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

    def forward(
        self, z, concept, belief, uncertainty, attention, mlp, concept_change,
        layer_position=None,
    ):
        del concept, belief, uncertainty, attention, mlp, concept_change, layer_position
        delta = torch.zeros(*z.shape[:-1], 4, device=z.device, dtype=z.dtype)
        delta[..., 0] = z[..., 0]
        return delta.unsqueeze(-2).expand(*delta.shape[:-1], 5, 4)


class FuzzyDynamicsTest(unittest.TestCase):
    def test_legacy_observed_configuration_keeps_checkpoint_structure(self) -> None:
        legacy = FuzzyDynamicsConfig.from_dict(
            {
                "hidden_size": 4,
                "belief_input_dim": 2,
                "vocab_size": 11,
                "z_dim": 2,
                "concept_dim": 1,
                "belief_dim": 1,
                "operation_dim": 2,
                "dynamics_hidden_dim": 4,
            }
        )
        original = FuzzyReasoningDynamics(legacy)
        restored = FuzzyReasoningDynamics(FuzzyDynamicsConfig.from_dict(legacy.to_dict()))
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(legacy.operation_source, "observed")
        self.assertEqual(legacy.operation_projection, "learned")
        self.assertIsNone(restored.operation_predictor)
        self.assertFalse(
            any(name.startswith("operation_predictor") for name in restored.state_dict())
        )

    def test_conditional_rollout_uses_predicted_not_true_intermediate_state(self) -> None:
        config = FuzzyDynamicsConfig(
            hidden_size=2,
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
            "margin": torch.rand(batch_size, layers + 1, 1),
        }
        model.fit_state_projectors(
            batch["hidden"].reshape(-1, hidden_size),
            batch["belief"].reshape(-1, belief_size),
        )
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
        for name in ("z", "concept", "belief", "uncertainty"):
            self.assertIn(f"dynamics_{name}", metrics)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_layer_condition_and_short_rollout_are_differentiable(self) -> None:
        batch_size, layers, hidden_size, belief_size = 3, 5, 8, 4
        config = FuzzyDynamicsConfig(
            hidden_size=hidden_size,
            belief_input_dim=belief_size,
            vocab_size=19,
            z_dim=3,
            concept_dim=2,
            belief_dim=2,
            operation_dim=3,
            dynamics_hidden_dim=7,
            dynamics_dropout=0.0,
            use_layer_condition=True,
        )
        model = FuzzyReasoningDynamics(config)
        batch = {
            "hidden": torch.randn(batch_size, layers + 1, hidden_size),
            "attention": torch.randn(batch_size, layers, hidden_size),
            "mlp": torch.randn(batch_size, layers, hidden_size),
            "belief": torch.randn(batch_size, layers + 1, belief_size),
            "uncertainty": torch.rand(batch_size, layers + 1, 1),
            "margin": torch.rand(batch_size, layers + 1, 1),
        }
        model.fit_state_projectors(
            batch["hidden"].reshape(-1, hidden_size),
            batch["belief"].reshape(-1, belief_size),
        )
        outputs = model(batch)
        self.assertTrue(
            torch.allclose(
                outputs["layer_position"][0, :, 0],
                torch.linspace(0.0, 1.0, layers),
            )
        )
        model.add_short_rollout(outputs, horizon=4)
        self.assertEqual(
            outputs["short_rollout_predicted_states"].shape,
            (batch_size, 5, config.state_dim),
        )
        loss, metrics = fuzzy_dynamics_loss(
            outputs, LossConfig(rollout_weight=0.25, rollout_horizon=4)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("rollout_h4", metrics)
        loss.backward()
        self.assertIsNotNone(
            model.local_dynamics.knowledge_enrichment.network[-1].weight.grad
        )

    def test_autonomous_rollout_does_not_read_observed_future_operations(self) -> None:
        batch_size, layers, hidden_size, belief_size = 3, 5, 8, 4
        config = FuzzyDynamicsConfig(
            hidden_size=hidden_size,
            belief_input_dim=belief_size,
            vocab_size=19,
            z_dim=3,
            concept_dim=2,
            belief_dim=2,
            operation_dim=3,
            dynamics_hidden_dim=7,
            dynamics_dropout=0.0,
            operation_source="predicted",
            operation_projection="frozen_pca",
            operation_predictor_hidden_dim=9,
            autonomous_context_layer=1,
        )
        generator = torch.Generator().manual_seed(23)
        batch = {
            "hidden": torch.randn(
                batch_size, layers + 1, hidden_size, generator=generator
            ),
            "attention": torch.randn(
                batch_size, layers, hidden_size, generator=generator
            ),
            "mlp": torch.randn(batch_size, layers, hidden_size, generator=generator),
            "belief": torch.randn(
                batch_size, layers + 1, belief_size, generator=generator
            ),
            "uncertainty": torch.rand(
                batch_size, layers + 1, 1, generator=generator
            ),
            "margin": torch.rand(batch_size, layers + 1, 1, generator=generator),
        }
        model = FuzzyReasoningDynamics(config)
        model.fit_state_projectors(
            batch["hidden"].reshape(-1, hidden_size),
            batch["belief"].reshape(-1, belief_size),
            batch["attention"].reshape(-1, hidden_size),
            batch["mlp"].reshape(-1, hidden_size),
        )
        model.eval()
        with torch.no_grad():
            original = model.autonomous_rollout(batch)
            state_only_batch = {
                name: value
                for name, value in batch.items()
                if name not in {"attention", "mlp"}
            }
            altered = model.autonomous_rollout(state_only_batch)
        self.assertEqual(
            original["predicted_states"].shape,
            (batch_size, layers, config.state_dim),
        )
        self.assertTrue(
            torch.allclose(original["predicted_states"], altered["predicted_states"])
        )
        self.assertEqual(original["start_layer"], 1)
        self.assertEqual(
            original["rollout_kind"], "autonomous_predicted_attention_and_mlp"
        )

    def test_autonomous_operation_and_rollout_losses_are_differentiable(self) -> None:
        batch_size, layers, hidden_size, belief_size = 3, 5, 8, 4
        config = FuzzyDynamicsConfig(
            hidden_size=hidden_size,
            belief_input_dim=belief_size,
            vocab_size=19,
            z_dim=3,
            concept_dim=2,
            belief_dim=2,
            operation_dim=3,
            dynamics_hidden_dim=7,
            dynamics_dropout=0.0,
            operation_source="predicted",
            operation_projection="frozen_pca",
            operation_predictor_hidden_dim=9,
            autonomous_context_layer=1,
        )
        batch = {
            "hidden": torch.randn(batch_size, layers + 1, hidden_size),
            "attention": torch.randn(batch_size, layers, hidden_size),
            "mlp": torch.randn(batch_size, layers, hidden_size),
            "belief": torch.randn(batch_size, layers + 1, belief_size),
            "uncertainty": torch.rand(batch_size, layers + 1, 1),
            "margin": torch.rand(batch_size, layers + 1, 1),
        }
        model = FuzzyReasoningDynamics(config)
        model.fit_state_projectors(
            batch["hidden"].reshape(-1, hidden_size),
            batch["belief"].reshape(-1, belief_size),
            batch["attention"].reshape(-1, hidden_size),
            batch["mlp"].reshape(-1, hidden_size),
        )
        outputs = model(batch, operation_teacher_probability=0.25)
        model.add_short_rollout(
            outputs, horizon=3, operation_teacher_probability=0.25
        )
        loss, metrics = fuzzy_dynamics_loss(
            outputs,
            LossConfig(
                rollout_weight=0.5,
                rollout_horizon=3,
                operation_weight=1.0,
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(metrics["operation"]), 0.0)
        self.assertEqual(
            outputs["short_rollout_kind"],
            "autonomous_predicted_attention_and_mlp",
        )
        loss.backward()
        self.assertIsNotNone(
            model.operation_predictor.attention.network[-1].weight.grad
        )


if __name__ == "__main__":
    unittest.main()
