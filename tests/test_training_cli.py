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

from scripts.train_fuzzy_dynamics import main


class TrainingCliTest(unittest.TestCase):
    def test_trains_autonomous_configuration_with_fixed_operation_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "activations.pt"
            config_path = root / "autonomous.json"
            generator = torch.Generator().manual_seed(31)
            queries, layers, hidden_size = 6, 3, 5
            torch.save(
                {
                    "hidden": torch.randn(
                        queries, layers + 1, hidden_size, generator=generator
                    ),
                    "attention": torch.randn(
                        queries, layers, hidden_size, generator=generator
                    ),
                    "mlp": torch.randn(
                        queries, layers, hidden_size, generator=generator
                    ),
                    "belief": torch.randn(
                        queries, layers + 1, 2, generator=generator
                    ),
                    "uncertainty": torch.rand(
                        queries, layers + 1, 1, generator=generator
                    ),
                    "margin": torch.rand(
                        queries, layers + 1, 1, generator=generator
                    ),
                    "metadata": [
                        {"uid": f"q{index}", "category": f"c{index // 2}"}
                        for index in range(queries)
                    ],
                    "vocab_size": 13,
                },
                cache_path,
            )
            config_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "z_dim": 2,
                            "concept_dim": 1,
                            "belief_dim": 1,
                            "operation_dim": 2,
                            "dynamics_hidden_dim": 4,
                            "dynamics_dropout": 0.0,
                            "operation_source": "predicted",
                            "operation_projection": "frozen_pca",
                            "operation_predictor_hidden_dim": 5,
                            "autonomous_context_layer": 1,
                        },
                        "loss": {
                            "operation_weight": 1.0,
                            "rollout_weight": 0.5,
                            "rollout_horizon": 2,
                        },
                        "training": {
                            "epochs": 1,
                            "batch_size": 2,
                            "learning_rate": 0.001,
                            "projector_fit_samples": 16,
                            "validation_fraction": 0.34,
                            "split_by": "category",
                            "rollout_curriculum": [2],
                            "operation_teacher_probability_start": 0.0,
                            "operation_teacher_probability_end": 0.0,
                            "operation_teacher_decay_epochs": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            experiment_time = "20260816_120002_000003"
            argv = [
                "train_fuzzy_dynamics",
                "--activations",
                str(cache_path),
                "--config",
                str(config_path),
                "--seed",
                "42",
                "--results-root",
                str(root / "results"),
                "--log-dir",
                str(root / "logs"),
                "--device",
                "cpu",
                "--log-every-batches",
                "1",
            ]
            with (
                patch(
                    "scripts.train_fuzzy_dynamics.generate_experiment_time",
                    return_value=experiment_time,
                ),
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                main()

            experiment_dir = root / "results" / f"fuzzy_dynamics_{experiment_time}"
            checkpoint = torch.load(
                experiment_dir / "checkpoint" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["model_config"]["operation_source"], "predicted")
            history = json.loads(
                (experiment_dir / "training" / "history.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(history[0]["train"]["operation"], 0.0)
            self.assertIn("rollout_h2", history[0]["validation"])

    def test_experiment_times_isolate_independent_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "activations.pt"
            config_path = root / "config.json"
            generator = torch.Generator().manual_seed(3)
            queries, layers = 4, 2
            torch.save(
                {
                    "hidden": torch.randn(queries, layers + 1, 4, generator=generator),
                    "attention": torch.randn(queries, layers, 4, generator=generator),
                    "mlp": torch.randn(queries, layers, 4, generator=generator),
                    "belief": torch.randn(queries, layers + 1, 2, generator=generator),
                    "uncertainty": torch.rand(queries, layers + 1, 1, generator=generator),
                    "margin": torch.rand(queries, layers + 1, 1, generator=generator),
                    "metadata": [
                        {"uid": f"q{index}", "category": f"c{index // 2}"}
                        for index in range(queries)
                    ],
                    "vocab_size": 13,
                },
                cache_path,
            )
            config_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "z_dim": 2,
                            "concept_dim": 1,
                            "belief_dim": 1,
                            "operation_dim": 2,
                            "dynamics_hidden_dim": 4,
                            "projector_dropout": 0.0,
                            "dynamics_dropout": 0.0,
                            "membership_temperature": 1.0,
                        },
                        "loss": {},
                        "training": {
                            "epochs": 1,
                            "batch_size": 2,
                            "learning_rate": 0.001,
                            "validation_fraction": 0.5,
                            "split_by": "category",
                        },
                    }
                ),
                encoding="utf-8",
            )
            results_root = root / "results"
            log_root = root / "logs"

            experiment_times = (
                "20260816_120000_000001",
                "20260816_120001_000002",
            )
            with patch(
                "scripts.train_fuzzy_dynamics.generate_experiment_time",
                side_effect=experiment_times,
            ):
                for seed in (1, 2):
                    argv = [
                        "train_fuzzy_dynamics",
                        "--activations",
                        str(cache_path),
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--results-root",
                        str(results_root),
                        "--log-dir",
                        str(log_root),
                        "--device",
                        "cpu",
                        "--log-every-batches",
                        "1",
                    ]
                    with (
                        patch.object(sys, "argv", argv),
                        redirect_stdout(io.StringIO()),
                        redirect_stderr(io.StringIO()),
                    ):
                        main()

            for experiment_time in experiment_times:
                experiment_dir = results_root / f"fuzzy_dynamics_{experiment_time}"
                self.assertTrue((experiment_dir / "checkpoint" / "best.pt").exists())
                self.assertTrue((experiment_dir / "checkpoint" / "last.pt").exists())
                training = experiment_dir / "training"
                self.assertTrue((training / "history.json").exists())
                self.assertTrue((training / "split.json").exists())
                manifest = json.loads(
                    (training / "experiment_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["experiment_time"], experiment_time)
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(
                    len(list((log_root / f"fuzzy_dynamics_{experiment_time}").glob("*.log"))), 1
                )

            with (
                patch(
                    "scripts.train_fuzzy_dynamics.generate_experiment_time",
                    return_value=experiment_times[-1],
                ),
                patch.object(sys, "argv", argv),
                self.assertRaises(FileExistsError),
            ):
                main()


if __name__ == "__main__":
    unittest.main()
