"""Serialization and Dataset helpers for extracted reasoning trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_TENSORS = ("hidden", "attention", "mlp", "belief", "uncertainty", "margin")
OPTIONAL_TENSORS = ("bridge_logprob", "answer_logprob")


def validate_activation_cache(cache: dict[str, Any]) -> None:
    for name in REQUIRED_TENSORS:
        if name not in cache or not isinstance(cache[name], torch.Tensor):
            raise ValueError(f"Activation cache is missing tensor '{name}'.")
    hidden = cache["hidden"]
    attention = cache["attention"]
    mlp = cache["mlp"]
    if hidden.ndim != 3 or attention.ndim != 3 or mlp.ndim != 3:
        raise ValueError("hidden, attention, and mlp tensors must have shape [N, L, D].")
    if hidden.shape[0] != attention.shape[0] or attention.shape != mlp.shape:
        raise ValueError("Activation tensors have inconsistent batch/layer dimensions.")
    if hidden.shape[1] != attention.shape[1] + 1:
        raise ValueError("hidden must contain L+1 states while operations contain L layers.")
    if cache["belief"].shape[:2] != hidden.shape[:2]:
        raise ValueError("belief must align with every hidden state.")
    if cache["uncertainty"].shape[:2] != hidden.shape[:2]:
        raise ValueError("uncertainty must align with every hidden state.")
    for name in ("uncertainty", "margin"):
        if cache[name].shape != (*hidden.shape[:2], 1):
            raise ValueError(f"{name} must have shape [N, L+1, 1].")
    for name in OPTIONAL_TENSORS:
        if name in cache and cache[name].shape != (*hidden.shape[:2], 1):
            raise ValueError(f"{name} must have shape [N, L+1, 1].")


def load_activation_cache(path: str | Path) -> dict[str, Any]:
    cache = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError("Activation cache must be a dictionary.")
    validate_activation_cache(cache)
    return cache


class ActivationTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: dict[str, Any], indices: list[int] | None = None):
        validate_activation_cache(cache)
        self.cache = cache
        self.indices = indices if indices is not None else list(range(cache["hidden"].shape[0]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item_index = self.indices[index]
        names = REQUIRED_TENSORS + tuple(name for name in OPTIONAL_TENSORS if name in self.cache)
        return {name: self.cache[name][item_index] for name in names}
