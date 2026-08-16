#!/usr/bin/env python3
"""Extract the layer-wise signals used to train fuzzy reasoning dynamics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy  # Import before PyTorch for binary-extension compatibility on HPC nodes.
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.data import load_prompt_records
from src.llm.activation_extractor import ActivationExtractor
from src.utils import configure_experiment_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--input", required=True, help="CSV, JSON, JSONL, or TXT prompts.")
    parser.add_argument("--output", required=True, help="Output .pt activation cache.")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--limit", type=int, default=256, help="0 means no limit.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--belief-dim", type=int, default=32)
    parser.add_argument("--belief-seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lens", choices=("logit", "tuned"), default="logit")
    parser.add_argument(
        "--lens-resource",
        default=None,
        help="Tuned Lens local directory or Hub resource; defaults to --model.",
    )
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-dir", default="logs/extract_activations")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser.parse_args()


def resolve_dtype(name: str):
    return "auto" if name == "auto" else getattr(torch, name)


def main() -> None:
    args = parse_args()
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive.")
    cache_name = Path(args.output).stem
    logger, _ = configure_experiment_logging(
        "extract_activations", Path(args.log_dir) / cache_name
    )
    logger.info("Arguments: %s", vars(args))
    logger.info("cache_name=%s", cache_name)
    records = load_prompt_records(
        args.input,
        prompt_field=args.prompt_field,
        limit=None if args.limit == 0 else args.limit,
    )
    # Hugging Face progress bars use stderr by default, which makes successful
    # checkpoint loading look like an error in Slurm. Our INFO messages and
    # extraction progress provide stdout status instead.
    transformers_logging.disable_progress_bar()
    logger.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model_kwargs = {
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    logger.info("Loading model from %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model.to(torch.device(args.device))
    logger.info("Loaded model on device=%s", args.device)

    tuned_lens = None
    if args.lens == "tuned":
        try:
            from tuned_lens.nn.lenses import TunedLens
        except ImportError as error:
            raise RuntimeError(
                "--lens tuned requires the tuned-lens package from "
                "requirements.txt"
            ) from error
        tuned_lens = TunedLens.from_model_and_pretrained(
            model,
            lens_resource_id=args.lens_resource or args.model,
        )
        model_parameter = next(model.parameters())
        tuned_lens.to(device=model_parameter.device, dtype=model_parameter.dtype)
        tuned_lens.eval()
        logger.info(
            "Loaded Tuned Lens on device=%s dtype=%s",
            model_parameter.device,
            model_parameter.dtype,
        )

    extractor = ActivationExtractor(
        model,
        tokenizer,
        belief_dim=args.belief_dim,
        belief_projection_seed=args.belief_seed,
        top_k=args.top_k,
        tuned_lens=tuned_lens,
    )
    tensor_batches: dict[str, list[torch.Tensor]] = {}
    starts = range(0, len(records), args.batch_size)
    total_batches = len(starts)
    for batch_number, start in enumerate(
        tqdm(starts, desc="Extracting", file=sys.stdout), start=1
    ):
        batch_records = records[start : start + args.batch_size]
        batch = extractor.extract_batch(
            [record["prompt"] for record in batch_records],
            max_length=args.max_length,
            bridge_entities=[record.get("bridge_entity") for record in batch_records],
            answers=[record.get("answer") for record in batch_records],
        )
        for name, values in batch.items():
            tensor_batches.setdefault(name, []).append(values)
        if batch_number % args.log_every_batches == 0 or batch_number == total_batches:
            logger.info(
                "batch=%d/%d extracted_queries=%d/%d",
                batch_number,
                total_batches,
                min(start + len(batch_records), len(records)),
                len(records),
            )

    tensors = {name: torch.cat(values, dim=0) for name, values in tensor_batches.items()}
    cache = {
        "format_version": 1,
        "model_name": args.model,
        "metadata": records,
        "prompts": [record["prompt"] for record in records],
        "num_layers": extractor.adapter.num_layers,
        "hidden_size": tensors["hidden"].shape[-1],
        "vocab_size": model.config.vocab_size,
        "belief_projection": {
            "kind": "gaussian_log_probability_projection",
            "dimension": args.belief_dim,
            "seed": args.belief_seed,
            "lens": args.lens,
            "lens_resource": args.lens_resource,
        },
        **tensors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    logger.info("Saved %d trajectories to %s", len(records), output.resolve())
    logger.info(
        "hidden=%s, attention=%s, belief=%s",
        tuple(cache["hidden"].shape),
        tuple(cache["attention"].shape),
        tuple(cache["belief"].shape),
    )


if __name__ == "__main__":
    main()
