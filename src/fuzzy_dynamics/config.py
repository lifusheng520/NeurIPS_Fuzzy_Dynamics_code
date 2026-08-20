"""Configuration objects for the fuzzy reasoning dynamical system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REASONING_MODES = (
    "knowledge_enrichment",
    "information_routing",
    "concept_composition",
    "prediction_refinement",
    "hop_transition",
)


@dataclass
class FuzzyDynamicsConfig:
    hidden_size: int
    belief_input_dim: int
    vocab_size: int
    z_dim: int = 64
    concept_dim: int = 32
    belief_dim: int = 32
    operation_dim: int = 32
    dynamics_hidden_dim: int = 128
    projector_dropout: float = 0.0
    dynamics_dropout: float = 0.1
    membership_temperature: float = 1.0
    use_layer_condition: bool = False
    operation_source: str = "observed"
    operation_projection: str = "learned"
    operation_predictor_hidden_dim: int = 256
    autonomous_context_layer: int = 1

    def __post_init__(self) -> None:
        if self.operation_source not in {"observed", "predicted"}:
            raise ValueError("operation_source must be 'observed' or 'predicted'.")
        if self.operation_projection not in {"learned", "frozen_pca"}:
            raise ValueError(
                "operation_projection must be 'learned' or 'frozen_pca'."
            )
        if self.operation_source == "predicted" and self.operation_projection != "frozen_pca":
            raise ValueError(
                "Predicted operations require fixed targets; set "
                "operation_projection='frozen_pca'."
            )
        if self.operation_predictor_hidden_dim <= 0:
            raise ValueError("operation_predictor_hidden_dim must be positive.")
        if self.autonomous_context_layer < 0:
            raise ValueError("autonomous_context_layer cannot be negative.")

    @property
    def state_dim(self) -> int:
        return self.z_dim + self.concept_dim + self.belief_dim + 1

    @property
    def num_modes(self) -> int:
        return len(REASONING_MODES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FuzzyDynamicsConfig":
        return cls(**values)


@dataclass
class LossConfig:
    dynamics_weight: float = 1.0
    semantic_weight: float = 0.1
    diversity_weight: float = 0.05
    fuzzy_entropy_weight: float = 0.01
    balance_weight: float = 0.01
    concept_probe_weight: float = 0.05
    semantic_temperature: float = 1.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 4
    operation_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LossConfig":
        return cls(**values)
