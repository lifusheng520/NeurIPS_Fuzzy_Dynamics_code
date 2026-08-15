from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluate_fuzzy_dynamics import main
from src.fuzzy_dynamics import FuzzyDynamicsConfig, FuzzyReasoningDynamics


class EvaluationCliTest(unittest.TestCase):
    def test_auto_selects_validation_and_writes_all_metric_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "activations.pt"
            experiment_time = "20260816_120000_000007"
            results_root = root / "results"
            experiment_dir = results_root / f"fuzzy_dynamics_{experiment_time}"
            checkpoint_dir = experiment_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True)
            training_dir = experiment_dir / "training"
            training_dir.mkdir(parents=True)
            output_dir = experiment_dir / "evaluation"
            config = FuzzyDynamicsConfig(
                hidden_size=6,
                belief_input_dim=3,
                vocab_size=17,
                z_dim=2,
                concept_dim=2,
                belief_dim=2,
                operation_dim=3,
                dynamics_hidden_dim=5,
                dynamics_dropout=0.0,
            )
            generator = torch.Generator().manual_seed(7)
            queries, layers = 4, 3
            cache = {
                "hidden": torch.randn(queries, layers + 1, 6, generator=generator),
                "attention": torch.randn(queries, layers, 6, generator=generator),
                "mlp": torch.randn(queries, layers, 6, generator=generator),
                "belief": torch.randn(queries, layers + 1, 3, generator=generator),
                "uncertainty": torch.rand(queries, layers + 1, 1, generator=generator),
                "bridge_logprob": torch.randn(
                    queries, layers + 1, 1, generator=generator
                ),
                "answer_logprob": torch.randn(
                    queries, layers + 1, 1, generator=generator
                ),
                "metadata": [
                    {"index": index, "uid": f"q{index}", "prompt": f"query {index}"}
                    for index in range(queries)
                ],
                "model_name": "test-model",
                "vocab_size": 17,
            }
            torch.save(cache, cache_path)
            model = FuzzyReasoningDynamics(config)
            split_path = training_dir / "split.json"
            checkpoint = model.checkpoint(
                source_activations=str(cache_path.resolve()),
                split_file=str(split_path.resolve()),
                experiment_time=experiment_time,
                epoch=1,
            )
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            with split_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"split_by": "query", "train": [0, 1], "validation": [2, 3]},
                    handle,
                )

            argv = [
                "evaluate_fuzzy_dynamics",
                "--experiment-time",
                experiment_time,
                "--results-root",
                str(results_root),
                "--activations",
                str(cache_path),
                "--device",
                "cpu",
                "--log-dir",
                str(root / "logs"),
            ]
            with (
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                main()

            with (output_dir / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(
                metrics["evaluation"]["selection"]["subset"], "validation"
            )
            self.assertEqual(
                metrics["evaluation"]["selection"]["selected_count"], 2
            )
            self.assertIn("one_step", metrics["fidelity"])
            self.assertIn("components", metrics["fidelity"]["one_step"])
            self.assertIn("conditional_rollout", metrics["fidelity"])
            self.assertIn("modes", metrics["semantic_alignment"])
            with (output_dir / "trajectories.jsonl").open(encoding="utf-8") as handle:
                trajectories = [json.loads(line) for line in handle]
            self.assertEqual([row["cache_index"] for row in trajectories], [2, 3])
            logs = list(
                (root / "logs" / f"fuzzy_dynamics_{experiment_time}").glob(
                    "evaluate_fuzzy_dynamics_*.log"
                )
            )
            self.assertEqual(len(logs), 1)
            self.assertIn("batch=1/1", logs[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
