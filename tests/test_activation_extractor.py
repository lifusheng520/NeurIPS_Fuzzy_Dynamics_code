from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from src.llm.activation_extractor import ActivationExtractor


class FakeAttention(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.projection = nn.Linear(size, size)

    def forward(self, hidden: torch.Tensor):
        return self.projection(hidden), None


class FakeMLP(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.projection = nn.Linear(size, size)

    def forward(self, hidden: torch.Tensor):
        return self.projection(hidden)


class FakeLayer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.self_attn = FakeAttention(size)
        self.mlp = FakeMLP(size)

    def forward(self, hidden: torch.Tensor):
        hidden = hidden + self.self_attn(hidden)[0]
        hidden = hidden + self.mlp(hidden)
        return (hidden,)


class FakeBackbone(nn.Module):
    def __init__(self, layers: int, size: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(size) for _ in range(layers)])
        self.norm = nn.LayerNorm(size)


class FakeModel(nn.Module):
    def __init__(self, layers: int = 3, size: int = 12, vocab: int = 31):
        super().__init__()
        self.embedding = nn.Embedding(vocab, size)
        self.model = FakeBackbone(layers, size)
        self.lm_head = nn.Linear(size, vocab, bias=False)
        self.config = types.SimpleNamespace(vocab_size=vocab)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return {"logits": self.lm_head(self.model.norm(hidden))}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(char) % 20) for char in text] or [1]

    def __call__(self, prompts, **_kwargs):
        encoded = [self.encode(prompt) for prompt in prompts]
        length = max(map(len, encoded))
        input_ids, masks = [], []
        for tokens in encoded:
            padding = [0] * (length - len(tokens))
            input_ids.append(padding + tokens)
            masks.append([0] * len(padding) + [1] * len(tokens))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
        }


class ActivationExtractorTest(unittest.TestCase):
    def test_extracts_aligned_layer_states(self) -> None:
        model = FakeModel()
        extractor = ActivationExtractor(model, FakeTokenizer(), belief_dim=7, top_k=3)
        result = extractor.extract_batch(
            ["alpha", "longer prompt"],
            bridge_entities=["bridge", "entity"],
            answers=["answer", "target"],
        )
        self.assertEqual(result["hidden"].shape, (2, 4, 12))
        self.assertEqual(result["attention"].shape, (2, 3, 12))
        self.assertEqual(result["mlp"].shape, (2, 3, 12))
        self.assertEqual(result["belief"].shape, (2, 4, 7))
        self.assertEqual(result["uncertainty"].shape, (2, 4, 1))
        self.assertEqual(result["bridge_logprob"].shape, (2, 4, 1))


if __name__ == "__main__":
    unittest.main()

