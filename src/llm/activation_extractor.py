"""Extract layer-wise reasoning signals from Hugging Face causal LMs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_adapter import CausalLMAdapter, module_device


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output does not contain a tensor: {type(value).__name__}")


def _last_non_padding_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_positions = attention_mask.flip(dims=(1,)).to(torch.int64).argmax(dim=1)
    return attention_mask.shape[1] - 1 - reversed_positions


def _select_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    positions = positions.to(hidden.device)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, positions]


class _ActivationHooks(AbstractContextManager["_ActivationHooks"]):
    def __init__(self, adapter: CausalLMAdapter):
        self.adapter = adapter
        self.handles: list[Any] = []
        self.layer_input: torch.Tensor | None = None
        self.layer_outputs: list[torch.Tensor | None] = [None] * adapter.num_layers
        self.attention_outputs: list[torch.Tensor | None] = [None] * adapter.num_layers
        self.mlp_outputs: list[torch.Tensor | None] = [None] * adapter.num_layers

    def reset(self) -> None:
        self.layer_input = None
        self.layer_outputs = [None] * self.adapter.num_layers
        self.attention_outputs = [None] * self.adapter.num_layers
        self.mlp_outputs = [None] * self.adapter.num_layers

    def __enter__(self) -> "_ActivationHooks":
        def capture_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            self.layer_input = _first_tensor(inputs)

        self.handles.append(self.adapter.layers[0].register_forward_pre_hook(capture_input))
        for index, (layer, attention, mlp) in enumerate(
            zip(self.adapter.layers, self.adapter.attention_modules, self.adapter.mlp_modules)
        ):
            self.handles.append(
                layer.register_forward_hook(
                    lambda _m, _i, output, idx=index: self.layer_outputs.__setitem__(
                        idx, _first_tensor(output)
                    )
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    lambda _m, _i, output, idx=index: self.attention_outputs.__setitem__(
                        idx, _first_tensor(output)
                    )
                )
            )
            self.handles.append(
                mlp.register_forward_hook(
                    lambda _m, _i, output, idx=index: self.mlp_outputs.__setitem__(
                        idx, _first_tensor(output)
                    )
                )
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def selected(self, positions: torch.Tensor) -> tuple[list[torch.Tensor], ...]:
        if self.layer_input is None or any(x is None for x in self.layer_outputs):
            raise RuntimeError("Layer hooks did not capture all hidden states.")
        if any(x is None for x in self.attention_outputs):
            raise RuntimeError("Attention hooks did not capture all outputs.")
        if any(x is None for x in self.mlp_outputs):
            raise RuntimeError("MLP hooks did not capture all outputs.")

        hidden_values = [self.layer_input, *self.layer_outputs]
        return (
            [_select_positions(x, positions) for x in hidden_values if x is not None],
            [_select_positions(x, positions) for x in self.attention_outputs if x is not None],
            [_select_positions(x, positions) for x in self.mlp_outputs if x is not None],
        )


@dataclass(frozen=True)
class BeliefFeatures:
    projected_log_probs: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    topk_ids: torch.Tensor
    topk_probs: torch.Tensor
    bridge_logprob: torch.Tensor | None = None
    answer_logprob: torch.Tensor | None = None


class ActivationExtractor:
    """Extract hidden, operation, and latent-belief features at the last token."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        belief_dim: int = 32,
        belief_projection_seed: int = 17,
        top_k: int = 5,
        tuned_lens: nn.Module | None = None,
    ) -> None:
        if belief_dim <= 0:
            raise ValueError("belief_dim must be positive.")
        if top_k < 2:
            raise ValueError("top_k must be at least 2 to compute a prediction margin.")
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = CausalLMAdapter.from_model(model)
        self.belief_dim = belief_dim
        self.belief_projection_seed = belief_projection_seed
        self.top_k = top_k
        self.tuned_lens = tuned_lens
        self._projection_cache: dict[tuple[str, int], torch.Tensor] = {}

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token

    def _belief_projection(self, device: torch.device, vocab_size: int) -> torch.Tensor:
        key = (str(device), vocab_size)
        if key not in self._projection_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.belief_projection_seed)
            projection = torch.randn(
                vocab_size,
                self.belief_dim,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device) / math.sqrt(vocab_size)
            self._projection_cache[key] = projection
        return self._projection_cache[key]

    @torch.inference_mode()
    def _first_target_ids(self, values: Sequence[str | None] | None) -> torch.Tensor | None:
        if values is None:
            return None
        token_ids = []
        for value in values:
            if not value:
                token_ids.append(-1)
                continue
            tokens = self.tokenizer.encode(" " + str(value), add_special_tokens=False)
            if not tokens:
                tokens = self.tokenizer.encode(str(value), add_special_tokens=False)
            token_ids.append(tokens[0] if tokens else -1)
        return torch.tensor(token_ids, dtype=torch.long)

    @torch.inference_mode()
    def _belief_features(
        self,
        hidden_by_layer: Sequence[torch.Tensor],
        bridge_ids: torch.Tensor | None = None,
        answer_ids: torch.Tensor | None = None,
    ) -> BeliefFeatures:
        norm_device = module_device(self.adapter.final_norm)
        output_device = module_device(self.adapter.output_embedding)
        lens_device = module_device(self.tuned_lens) if self.tuned_lens is not None else None
        projected, entropies, margins, ids, probs = [], [], [], [], []
        bridge_logprobs, answer_logprobs = [], []
        for layer_index, hidden in enumerate(hidden_by_layer):
            if self.tuned_lens is None:
                normalized = self.adapter.final_norm(hidden.to(norm_device))
                logits = self.adapter.output_embedding(normalized.to(output_device)).float()
            elif layer_index < len(self.tuned_lens):
                logits = self.tuned_lens(hidden.to(lens_device), layer_index).float()
            else:
                # Tuned Lens has translators for intermediate states only. The
                # final block output is decoded by its standard unembed module.
                logits = self.tuned_lens.unembed(hidden.to(lens_device)).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            entropy = -(probabilities * log_probs).sum(dim=-1)
            top_probs, top_ids = probabilities.topk(self.top_k, dim=-1)
            projection = self._belief_projection(logits.device, logits.shape[-1])
            projected.append((log_probs @ projection).cpu())
            entropies.append(entropy.cpu())
            margins.append((top_probs[:, 0] - top_probs[:, 1]).cpu())
            ids.append(top_ids.cpu())
            probs.append(top_probs.cpu())

            for target_ids, destination in (
                (bridge_ids, bridge_logprobs),
                (answer_ids, answer_logprobs),
            ):
                if target_ids is not None:
                    device_ids = target_ids.to(log_probs.device)
                    valid = device_ids.ge(0)
                    gathered = log_probs.gather(
                        1, device_ids.clamp_min(0).unsqueeze(1)
                    ).squeeze(1)
                    destination.append(
                        torch.where(valid, gathered, torch.full_like(gathered, float("nan"))).cpu()
                    )

        return BeliefFeatures(
            projected_log_probs=torch.stack(projected, dim=1),
            entropy=torch.stack(entropies, dim=1).unsqueeze(-1),
            margin=torch.stack(margins, dim=1).unsqueeze(-1),
            topk_ids=torch.stack(ids, dim=1),
            topk_probs=torch.stack(probs, dim=1),
            bridge_logprob=(
                torch.stack(bridge_logprobs, dim=1).unsqueeze(-1)
                if bridge_logprobs
                else None
            ),
            answer_logprob=(
                torch.stack(answer_logprobs, dim=1).unsqueeze(-1)
                if answer_logprobs
                else None
            ),
        )

    @torch.inference_mode()
    def extract_batch(
        self,
        prompts: Sequence[str],
        max_length: int = 512,
        bridge_entities: Sequence[str | None] | None = None,
        answers: Sequence[str | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not prompts:
            raise ValueError("Cannot extract an empty prompt batch.")
        tokenized = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_device = module_device(self.adapter.layers[0])
        model_inputs = {key: value.to(input_device) for key, value in tokenized.items()}
        positions = _last_non_padding_positions(model_inputs["attention_mask"])

        self.model.eval()
        with _ActivationHooks(self.adapter) as hooks:
            hooks.reset()
            self.model(**model_inputs, use_cache=False, return_dict=True)
            hidden, attention, mlp = hooks.selected(positions)
            belief = self._belief_features(
                hidden,
                bridge_ids=self._first_target_ids(bridge_entities),
                answer_ids=self._first_target_ids(answers),
            )

        result = {
            "hidden": torch.stack([x.float().cpu() for x in hidden], dim=1),
            "attention": torch.stack([x.float().cpu() for x in attention], dim=1),
            "mlp": torch.stack([x.float().cpu() for x in mlp], dim=1),
            "belief": belief.projected_log_probs.float(),
            "uncertainty": belief.entropy.float(),
            "margin": belief.margin.float(),
            "topk_ids": belief.topk_ids,
            "topk_probs": belief.topk_probs.float(),
            "last_token_ids": model_inputs["input_ids"][
                torch.arange(len(prompts), device=input_device), positions
            ].cpu(),
        }
        if belief.bridge_logprob is not None:
            result["bridge_logprob"] = belief.bridge_logprob.float()
        if belief.answer_logprob is not None:
            result["answer_logprob"] = belief.answer_logprob.float()
        return result
