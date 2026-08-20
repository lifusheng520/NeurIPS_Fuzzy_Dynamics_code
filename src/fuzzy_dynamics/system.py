"""End-to-end fuzzy reasoning dynamical system."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import FuzzyDynamicsConfig
from .local_dynamics import LocalReasoningDynamics
from .membership import MembershipDynamics
from .operation_predictor import OperationPredictor
from .projectors import ReasoningProjectors


class FuzzyReasoningDynamics(nn.Module):
    def __init__(self, config: FuzzyDynamicsConfig) -> None:
        super().__init__()
        self.config = config
        self.projectors = ReasoningProjectors(
            hidden_size=config.hidden_size,
            belief_input_dim=config.belief_input_dim,
            z_dim=config.z_dim,
            concept_dim=config.concept_dim,
            belief_dim=config.belief_dim,
            operation_dim=config.operation_dim,
            dropout=config.projector_dropout,
            operation_projection=config.operation_projection,
        )
        self.local_dynamics = LocalReasoningDynamics(
            z_dim=config.z_dim,
            concept_dim=config.concept_dim,
            belief_dim=config.belief_dim,
            operation_dim=config.operation_dim,
            state_dim=config.state_dim,
            hidden_dim=config.dynamics_hidden_dim,
            dropout=config.dynamics_dropout,
            use_layer_condition=config.use_layer_condition,
        )
        self.membership_dynamics = MembershipDynamics(
            state_dim=config.state_dim,
            operation_dim=config.operation_dim,
            num_modes=config.num_modes,
            temperature=config.membership_temperature,
        )
        self.bridge_concept_probe = nn.Linear(config.concept_dim, 1)
        self.operation_predictor = (
            OperationPredictor(
                state_dim=config.state_dim,
                operation_dim=config.operation_dim,
                hidden_dim=config.operation_predictor_hidden_dim,
                dropout=config.dynamics_dropout,
            )
            if config.operation_source == "predicted"
            else None
        )
        self.register_buffer("state_delta_scale", torch.ones(config.state_dim))
        self.register_buffer("state_delta_scale_fitted", torch.tensor(False))

    @property
    def supports_autonomous_rollout(self) -> bool:
        return self.operation_predictor is not None

    @torch.no_grad()
    def fit_state_projectors(
        self,
        hidden_samples: torch.Tensor,
        belief_samples: torch.Tensor,
        attention_samples: torch.Tensor | None = None,
        mlp_samples: torch.Tensor | None = None,
    ) -> None:
        self.projectors.fit_state_projectors(
            hidden_samples, belief_samples, attention_samples, mlp_samples
        )

    @torch.no_grad()
    def set_state_delta_scale(self, scale: torch.Tensor) -> None:
        if scale.shape != (self.config.state_dim,):
            raise ValueError(f"Expected state delta scale [{self.config.state_dim}].")
        self.state_delta_scale.copy_(scale.to(self.state_delta_scale).clamp_min(1e-4))
        self.state_delta_scale_fitted.fill_(True)

    def _project_states(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project cached LLM states without reading observed layer operations."""
        hidden = batch["hidden"]
        belief_raw = batch["belief"]
        uncertainty_raw = batch["uncertainty"]

        z, concept = self.projectors.hidden_and_concept(hidden)
        belief = self.projectors.belief(belief_raw)
        max_entropy = max(1.0, float(torch.log(torch.tensor(self.config.vocab_size))))
        uncertainty = uncertainty_raw / max_entropy
        states = torch.cat((z, concept, belief, uncertainty), dim=-1)
        return states, z, concept, belief, uncertainty

    def _project_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project cached LLM states and observed operations."""
        states, z, concept, belief, uncertainty = self._project_states(batch)
        attention = self.projectors.attention(batch["attention"])
        mlp = self.projectors.mlp(batch["mlp"])
        return states, z, concept, belief, uncertainty, attention, mlp

    @staticmethod
    def _layer_positions(
        num_layers: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if num_layers <= 0:
            raise ValueError("At least one transition is required.")
        denominator = max(1, num_layers - 1)
        values = torch.arange(num_layers, device=device, dtype=dtype) / denominator
        return values.view(1, num_layers, 1).expand(batch_size, -1, -1)

    def _validate_context_layer(self, num_layers: int) -> int:
        context_layer = self.config.autonomous_context_layer
        if context_layer >= num_layers:
            raise ValueError(
                f"autonomous_context_layer={context_layer} must be smaller than "
                f"the number of transitions ({num_layers})."
            )
        return context_layer

    def _predict_operations(
        self,
        states: torch.Tensor,
        layer_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.operation_predictor is None:
            raise RuntimeError("This configuration has no autonomous operation predictor.")
        num_layers = layer_positions.shape[1]
        context_layer = self._validate_context_layer(num_layers)
        current_states = states[:, :-1]
        context = states[:, context_layer : context_layer + 1].expand_as(current_states)
        return self.operation_predictor(current_states, context, layer_positions)

    @staticmethod
    def _mix_operations(
        predicted: torch.Tensor,
        observed: torch.Tensor,
        teacher_probability: float,
    ) -> torch.Tensor:
        if not 0.0 <= teacher_probability <= 1.0:
            raise ValueError("operation_teacher_probability must lie in [0, 1].")
        return (
            teacher_probability * observed
            + (1.0 - teacher_probability) * predicted
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        operation_teacher_probability: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        states, z, concept, belief, uncertainty, attention, mlp = self._project_batch(batch)

        current_z = z[:, :-1]
        current_concept = concept[:, :-1]
        current_belief = belief[:, :-1]
        current_uncertainty = uncertainty[:, :-1]
        concept_change = torch.zeros_like(current_concept)
        concept_change[:, 1:] = current_concept[:, 1:] - current_concept[:, :-1]
        layer_position = self._layer_positions(
            attention.shape[1], attention.shape[0], attention.device, attention.dtype
        )

        dynamics_attention = attention
        dynamics_mlp = mlp
        dynamics_concept_change = concept_change
        predicted_attention: torch.Tensor | None = None
        predicted_mlp: torch.Tensor | None = None
        operation_mask: torch.Tensor | None = None
        if self.operation_predictor is not None:
            predicted_attention, predicted_mlp = self._predict_operations(
                states, layer_position
            )
            mixed_attention = self._mix_operations(
                predicted_attention, attention, operation_teacher_probability
            )
            mixed_mlp = self._mix_operations(
                predicted_mlp, mlp, operation_teacher_probability
            )
            context_layer = self._validate_context_layer(attention.shape[1])
            operation_mask = (
                torch.arange(attention.shape[1], device=attention.device)
                .ge(context_layer)
                .view(1, -1, 1)
            )
            # The autonomous simulation is initialized at the contextualized
            # state s_context. Earlier transitions remain the observed baseline
            # and are excluded from operation reconstruction metrics/losses.
            dynamics_attention = torch.where(operation_mask, mixed_attention, attention)
            dynamics_mlp = torch.where(operation_mask, mixed_mlp, mlp)
            dynamics_concept_change = concept_change.clone()
            dynamics_concept_change[:, context_layer] = 0.0

        local_deltas = self.local_dynamics(
            current_z,
            current_concept,
            current_belief,
            current_uncertainty,
            dynamics_attention,
            dynamics_mlp,
            dynamics_concept_change,
            layer_position,
        )
        memberships = self.membership_dynamics(
            states[:, :-1], dynamics_attention, dynamics_mlp
        )
        predicted_delta = (memberships.unsqueeze(-1) * local_deltas).sum(dim=-2)
        target_delta = states[:, 1:] - states[:, :-1]

        outputs = {
            "states": states,
            "z": z,
            "concept": concept,
            "belief": belief,
            "uncertainty": uncertainty,
            "attention": attention,
            "mlp": mlp,
            "dynamics_attention": dynamics_attention,
            "dynamics_mlp": dynamics_mlp,
            "concept_change": concept_change,
            "dynamics_concept_change": dynamics_concept_change,
            "layer_position": layer_position,
            "bridge_probe_logits": self.bridge_concept_probe(concept),
            "local_deltas": local_deltas,
            "memberships": memberships,
            "predicted_delta": predicted_delta,
            "target_delta": target_delta,
            "state_delta_scale": self.state_delta_scale,
            "component_dimensions": {
                "z": self.config.z_dim,
                "concept": self.config.concept_dim,
                "belief": self.config.belief_dim,
                "uncertainty": 1,
            },
        }
        if predicted_attention is not None and predicted_mlp is not None:
            outputs.update(
                predicted_attention=predicted_attention,
                predicted_mlp=predicted_mlp,
                operation_mask=operation_mask,
                operation_teacher_probability=operation_teacher_probability,
            )
        for name in ("bridge_logprob", "answer_logprob", "margin"):
            if name in batch:
                outputs[name] = batch[name]
        return outputs

    def conditional_rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Roll the learned state dynamics forward without teacher-forced states.

        Attention and MLP operation features are still taken from the observed
        LLM trajectory.  The result is therefore a *conditional* rollout, not an
        autonomous simulation of the Transformer.  Starting at ``s_0``, every
        later state, concept change, and membership recurrence uses the model's
        own preceding prediction.
        """
        true_states, _, _, _, _, attention, mlp = self._project_batch(batch)
        num_layers = attention.shape[1]
        if num_layers == 0:
            raise ValueError("At least one layer is required for a rollout.")

        current = true_states[:, 0]
        predicted_states = [current]
        predicted_deltas = []
        rollout_memberships = []
        rollout_local_deltas = []
        previous_membership = self.membership_dynamics.initial_membership(current.shape[0])
        previous_concept: torch.Tensor | None = None
        component_sizes = (
            self.config.z_dim,
            self.config.concept_dim,
            self.config.belief_dim,
            1,
        )

        for layer_index in range(num_layers):
            z, concept, belief, uncertainty = current.split(component_sizes, dim=-1)
            concept_change = (
                torch.zeros_like(concept)
                if previous_concept is None
                else concept - previous_concept
            )
            layer_attention = attention[:, layer_index]
            layer_mlp = mlp[:, layer_index]
            local_deltas = self.local_dynamics(
                z.unsqueeze(1),
                concept.unsqueeze(1),
                belief.unsqueeze(1),
                uncertainty.unsqueeze(1),
                layer_attention.unsqueeze(1),
                layer_mlp.unsqueeze(1),
                concept_change.unsqueeze(1),
                self._layer_positions(
                    num_layers, current.shape[0], current.device, current.dtype
                )[:, layer_index : layer_index + 1],
            ).squeeze(1)
            membership = self.membership_dynamics.step(
                current,
                layer_attention,
                layer_mlp,
                previous_membership,
            )
            delta = (membership.unsqueeze(-1) * local_deltas).sum(dim=-2)

            previous_concept = concept
            previous_membership = membership
            current = current + delta
            predicted_states.append(current)
            predicted_deltas.append(delta)
            rollout_memberships.append(membership)
            rollout_local_deltas.append(local_deltas)

        return {
            "true_states": true_states,
            "predicted_states": torch.stack(predicted_states, dim=1),
            "predicted_delta": torch.stack(predicted_deltas, dim=1),
            "memberships": torch.stack(rollout_memberships, dim=1),
            "local_deltas": torch.stack(rollout_local_deltas, dim=1),
            "rollout_kind": "conditional_on_observed_attention_and_mlp",
        }

    def autonomous_rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Roll forward using only operations predicted on the simulated trajectory.

        The rollout is initialized at ``s_context`` rather than the embedding
        state ``s_0``. The first contextualized last-token state contains prompt
        information that is unavailable in the isolated final-token embedding.
        No observed attention or MLP feature is read after initialization.
        """
        if self.operation_predictor is None:
            raise RuntimeError(
                "autonomous_rollout requires operation_source='predicted'."
            )
        true_states, _, _, _, _ = self._project_states(batch)
        num_layers = true_states.shape[1] - 1
        context_layer = self._validate_context_layer(num_layers)
        positions = self._layer_positions(
            num_layers, true_states.shape[0], true_states.device, true_states.dtype
        )

        context = true_states[:, context_layer]
        current = context
        predicted_states = [current]
        predicted_deltas = []
        predicted_attention = []
        predicted_mlp = []
        rollout_memberships = []
        rollout_local_deltas = []
        previous_membership = self.membership_dynamics.initial_membership(current.shape[0])
        previous_concept: torch.Tensor | None = None
        component_sizes = (
            self.config.z_dim,
            self.config.concept_dim,
            self.config.belief_dim,
            1,
        )

        for layer_index in range(context_layer, num_layers):
            z, concept, belief, uncertainty = current.split(component_sizes, dim=-1)
            concept_change = (
                torch.zeros_like(concept)
                if previous_concept is None
                else concept - previous_concept
            )
            layer_position = positions[:, layer_index]
            layer_attention, layer_mlp = self.operation_predictor(
                current, context, layer_position
            )
            local_deltas = self.local_dynamics(
                z.unsqueeze(1),
                concept.unsqueeze(1),
                belief.unsqueeze(1),
                uncertainty.unsqueeze(1),
                layer_attention.unsqueeze(1),
                layer_mlp.unsqueeze(1),
                concept_change.unsqueeze(1),
                layer_position.unsqueeze(1),
            ).squeeze(1)
            membership = self.membership_dynamics.step(
                current,
                layer_attention,
                layer_mlp,
                previous_membership,
            )
            delta = (membership.unsqueeze(-1) * local_deltas).sum(dim=-2)

            previous_concept = concept
            previous_membership = membership
            current = current + delta
            predicted_states.append(current)
            predicted_deltas.append(delta)
            predicted_attention.append(layer_attention)
            predicted_mlp.append(layer_mlp)
            rollout_memberships.append(membership)
            rollout_local_deltas.append(local_deltas)

        return {
            "true_states": true_states[:, context_layer:],
            "predicted_states": torch.stack(predicted_states, dim=1),
            "predicted_delta": torch.stack(predicted_deltas, dim=1),
            "predicted_attention": torch.stack(predicted_attention, dim=1),
            "predicted_mlp": torch.stack(predicted_mlp, dim=1),
            "memberships": torch.stack(rollout_memberships, dim=1),
            "local_deltas": torch.stack(rollout_local_deltas, dim=1),
            "start_layer": context_layer,
            "rollout_kind": "autonomous_predicted_attention_and_mlp",
        }

    def add_short_rollout(
        self,
        outputs: dict[str, Any],
        horizon: int,
        operation_teacher_probability: float = 0.0,
    ) -> dict[str, Any]:
        """Attach a differentiable random-window conditional rollout for training."""
        if self.operation_predictor is not None:
            return self._add_autonomous_short_rollout(
                outputs, horizon, operation_teacher_probability
            )
        if horizon < 2:
            raise ValueError("rollout_horizon must be at least 2.")
        states = outputs["states"]
        attention = outputs["attention"]
        mlp = outputs["mlp"]
        teacher_memberships = outputs["memberships"]
        batch_size, num_layers = attention.shape[:2]
        if horizon > num_layers:
            raise ValueError(f"rollout_horizon={horizon} exceeds {num_layers} transitions.")
        num_starts = num_layers - horizon + 1
        if self.training:
            starts = torch.randint(num_starts, (batch_size,), device=states.device)
        else:
            starts = torch.arange(batch_size, device=states.device) % num_starts
        rows = torch.arange(batch_size, device=states.device)
        current = states[rows, starts]
        predicted_states = [current]
        target_states = [current]
        previous_membership = self.membership_dynamics.initial_membership(batch_size)
        has_prefix = starts > 0
        if has_prefix.any():
            previous_membership = previous_membership.clone()
            previous_membership[has_prefix] = teacher_memberships[
                rows[has_prefix], starts[has_prefix] - 1
            ]
        concept_start = self.config.z_dim
        concept_end = concept_start + self.config.concept_dim
        previous_concept = current[:, concept_start:concept_end].clone()
        if has_prefix.any():
            previous_concept[has_prefix] = states[
                rows[has_prefix], starts[has_prefix] - 1, concept_start:concept_end
            ]
        component_sizes = (
            self.config.z_dim,
            self.config.concept_dim,
            self.config.belief_dim,
            1,
        )
        all_positions = self._layer_positions(
            num_layers, batch_size, states.device, states.dtype
        )
        for step in range(horizon):
            layer_indices = starts + step
            z, concept, belief, uncertainty = current.split(component_sizes, dim=-1)
            concept_change = concept - previous_concept
            layer_attention = attention[rows, layer_indices]
            layer_mlp = mlp[rows, layer_indices]
            local_deltas = self.local_dynamics(
                z.unsqueeze(1),
                concept.unsqueeze(1),
                belief.unsqueeze(1),
                uncertainty.unsqueeze(1),
                layer_attention.unsqueeze(1),
                layer_mlp.unsqueeze(1),
                concept_change.unsqueeze(1),
                all_positions[rows, layer_indices].unsqueeze(1),
            ).squeeze(1)
            membership = self.membership_dynamics.step(
                current, layer_attention, layer_mlp, previous_membership
            )
            delta = (membership.unsqueeze(-1) * local_deltas).sum(dim=-2)
            previous_concept = concept
            previous_membership = membership
            current = current + delta
            predicted_states.append(current)
            target_states.append(states[rows, layer_indices + 1])
        outputs["short_rollout_predicted_states"] = torch.stack(predicted_states, dim=1)
        outputs["short_rollout_target_states"] = torch.stack(target_states, dim=1)
        outputs["short_rollout_starts"] = starts
        outputs["short_rollout_kind"] = "conditional_on_observed_attention_and_mlp"
        return outputs

    def _add_autonomous_short_rollout(
        self,
        outputs: dict[str, Any],
        horizon: int,
        operation_teacher_probability: float,
    ) -> dict[str, Any]:
        """Attach a differentiable rollout whose operations follow predicted states."""
        if self.operation_predictor is None:
            raise RuntimeError("Autonomous short rollout requires an operation predictor.")
        if horizon < 2:
            raise ValueError("rollout_horizon must be at least 2.")
        states = outputs["states"]
        attention = outputs["attention"]
        mlp = outputs["mlp"]
        batch_size, num_layers = attention.shape[:2]
        context_layer = self._validate_context_layer(num_layers)
        if context_layer + horizon > num_layers:
            raise ValueError(
                f"rollout_horizon={horizon} from context layer {context_layer} "
                f"exceeds {num_layers} transitions."
            )
        num_starts = num_layers - horizon - context_layer + 1
        if self.training:
            starts = context_layer + torch.randint(
                num_starts, (batch_size,), device=states.device
            )
        else:
            starts = context_layer + (
                torch.arange(batch_size, device=states.device) % num_starts
            )
        rows = torch.arange(batch_size, device=states.device)
        context = states[:, context_layer]
        current = states[rows, starts]
        predicted_states = [current]
        target_states = [current]
        predicted_attention = []
        predicted_mlp = []
        previous_membership = self.membership_dynamics.initial_membership(batch_size)
        previous_concept: torch.Tensor | None = None
        component_sizes = (
            self.config.z_dim,
            self.config.concept_dim,
            self.config.belief_dim,
            1,
        )
        all_positions = self._layer_positions(
            num_layers, batch_size, states.device, states.dtype
        )

        for step in range(horizon):
            layer_indices = starts + step
            z, concept, belief, uncertainty = current.split(component_sizes, dim=-1)
            concept_change = (
                torch.zeros_like(concept)
                if previous_concept is None
                else concept - previous_concept
            )
            layer_position = all_positions[rows, layer_indices]
            layer_predicted_attention, layer_predicted_mlp = self.operation_predictor(
                current, context, layer_position
            )
            layer_attention = self._mix_operations(
                layer_predicted_attention,
                attention[rows, layer_indices],
                operation_teacher_probability,
            )
            layer_mlp = self._mix_operations(
                layer_predicted_mlp,
                mlp[rows, layer_indices],
                operation_teacher_probability,
            )
            local_deltas = self.local_dynamics(
                z.unsqueeze(1),
                concept.unsqueeze(1),
                belief.unsqueeze(1),
                uncertainty.unsqueeze(1),
                layer_attention.unsqueeze(1),
                layer_mlp.unsqueeze(1),
                concept_change.unsqueeze(1),
                layer_position.unsqueeze(1),
            ).squeeze(1)
            membership = self.membership_dynamics.step(
                current, layer_attention, layer_mlp, previous_membership
            )
            delta = (membership.unsqueeze(-1) * local_deltas).sum(dim=-2)
            previous_concept = concept
            previous_membership = membership
            current = current + delta
            predicted_states.append(current)
            target_states.append(states[rows, layer_indices + 1])
            predicted_attention.append(layer_predicted_attention)
            predicted_mlp.append(layer_predicted_mlp)

        outputs["short_rollout_predicted_states"] = torch.stack(predicted_states, dim=1)
        outputs["short_rollout_target_states"] = torch.stack(target_states, dim=1)
        outputs["short_rollout_predicted_attention"] = torch.stack(
            predicted_attention, dim=1
        )
        outputs["short_rollout_predicted_mlp"] = torch.stack(predicted_mlp, dim=1)
        outputs["short_rollout_starts"] = starts
        outputs["short_rollout_kind"] = "autonomous_predicted_attention_and_mlp"
        return outputs

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {
            "model_config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
            **metadata,
        }
