"""Architecture adapters for extracting internal Transformer activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


def _get_nested_attr(obj: object, path: str) -> object | None:
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _first_module(obj: object, paths: Iterable[str]) -> nn.Module | None:
    for path in paths:
        value = _get_nested_attr(obj, path)
        if isinstance(value, nn.Module):
            return value
    return None


def module_device(module: nn.Module) -> torch.device:
    """Return a module's device, including modules containing only buffers."""
    parameter = next(module.parameters(recurse=True), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(recurse=True), None)
    return buffer.device if buffer is not None else torch.device("cpu")


@dataclass(frozen=True)
class CausalLMAdapter:
    """Resolved modules needed by the fuzzy-dynamics extraction pipeline."""

    layers: Sequence[nn.Module]
    attention_modules: Sequence[nn.Module]
    mlp_modules: Sequence[nn.Module]
    final_norm: nn.Module
    output_embedding: nn.Module

    @classmethod
    def from_model(cls, model: nn.Module) -> "CausalLMAdapter":
        layers_obj = None
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            candidate = _get_nested_attr(model, path)
            if isinstance(candidate, (nn.ModuleList, list, tuple)):
                layers_obj = candidate
                break
        if layers_obj is None or len(layers_obj) == 0:
            raise ValueError(
                "Unsupported model architecture: expected model.layers, "
                "transformer.h, or gpt_neox.layers."
            )

        layers = list(layers_obj)
        attention_modules: list[nn.Module] = []
        mlp_modules: list[nn.Module] = []
        for index, layer in enumerate(layers):
            attention = _first_module(layer, ("self_attn", "attn", "attention"))
            mlp = _first_module(layer, ("mlp", "feed_forward", "ffn"))
            if attention is None or mlp is None:
                raise ValueError(
                    f"Could not identify attention/MLP modules in layer {index} "
                    f"({type(layer).__name__})."
                )
            attention_modules.append(attention)
            mlp_modules.append(mlp)

        final_norm = _first_module(
            model,
            ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"),
        )
        if final_norm is None:
            raise ValueError("Could not identify the model's final normalization layer.")

        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        output_embedding = get_output_embeddings() if callable(get_output_embeddings) else None
        if not isinstance(output_embedding, nn.Module):
            output_embedding = _first_module(model, ("lm_head", "embed_out"))
        if output_embedding is None:
            raise ValueError("Could not identify the model's output embedding/lm_head.")

        return cls(
            layers=layers,
            attention_modules=attention_modules,
            mlp_modules=mlp_modules,
            final_norm=final_norm,
            output_embedding=output_embedding,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

