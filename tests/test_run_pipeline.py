from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_pipeline import StageProcessError, main


EXPERIMENT_TIME = "20260816_120000_000123"


def _module_name(command: list[str]) -> str:
    return command[command.index("-m") + 1]


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class RunPipelineTest(unittest.TestCase):
    @contextmanager
    def _patched_pipeline(
        self,
        root: Path,
        *extra_args: str,
        fail_stage: str | None = None,
    ):
        activations = root / "activations.pt"
        config = root / "config.json"
        activations.write_bytes(b"activation-cache-placeholder")
        config.write_text("{}", encoding="utf-8")
        results_root = root / "results"
        log_root = root / "logs"
        argv = [
            "run_pipeline",
            "--activations",
            str(activations),
            "--config",
            str(config),
            "--results-root",
            str(results_root),
            "--log-dir",
            str(log_root),
            "--device",
            "cpu",
            *extra_args,
        ]
        experiment_dir = (
            results_root.resolve() / f"fuzzy_dynamics_{EXPERIMENT_TIME}"
        )

        def write_artifact(path: Path, value: str = "placeholder") -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")

        def fake_popen(command, **_kwargs):
            module = _module_name(command)
            stage = {
                "scripts.train_fuzzy_dynamics": "training",
                "scripts.evaluate_fuzzy_dynamics": "evaluation",
                "scripts.plot_trajectory": "plotting",
            }[module]
            failed = stage == fail_stage
            if not failed and stage == "training":
                write_artifact(experiment_dir / "checkpoint" / "best.pt")
                write_artifact(experiment_dir / "checkpoint" / "last.pt")
                write_artifact(experiment_dir / "training" / "history.json", "{}")
                write_artifact(experiment_dir / "training" / "split.json", "{}")
                write_artifact(
                    experiment_dir / "training" / "experiment_manifest.json",
                    json.dumps(
                        {
                            "experiment_time": EXPERIMENT_TIME,
                            "status": "complete",
                        }
                    ),
                )
            elif not failed and stage == "evaluation":
                write_artifact(experiment_dir / "evaluation" / "metrics.json", "{}")
                write_artifact(
                    experiment_dir / "evaluation" / "evaluation_manifest.json",
                    json.dumps({"experiment_time": EXPERIMENT_TIME}),
                )
                write_artifact(
                    experiment_dir / "evaluation" / "membership_analysis.json",
                    "{}",
                )
                write_artifact(
                    experiment_dir / "evaluation" / "trajectories.jsonl",
                    "{}\n",
                )
            elif not failed:
                plot_index = _argument(command, "--index")
                write_artifact(
                    experiment_dir / "evaluation" / f"trajectory_{plot_index}.png"
                )
            child = MagicMock()
            child.pid = 12345
            child.wait.return_value = 7 if failed else 0
            child.poll.return_value = 7 if failed else 0
            return child

        runner = MagicMock(side_effect=fake_popen)
        logger = MagicMock()

        def fake_logging(_command_name, directory):
            return logger, Path(directory) / "run_pipeline_test.log"

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch(
                "scripts.run_pipeline.generate_experiment_time",
                return_value=EXPERIMENT_TIME,
            ) as generate_time,
            patch(
                "scripts.run_pipeline.configure_experiment_logging",
                side_effect=fake_logging,
            ),
            patch("scripts.run_pipeline.subprocess.Popen", runner),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            yield {
                "runner": runner,
                "generate_time": generate_time,
                "results_root": results_root.resolve(),
                "log_root": log_root.resolve(),
                "stdout": stdout,
            }

    @staticmethod
    def _manifest(results_root: Path) -> dict:
        path = (
            results_root
            / f"fuzzy_dynamics_{EXPERIMENT_TIME}"
            / "training"
            / "pipeline_manifest.json"
        )
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def test_runs_stages_in_order_with_one_timestamp_and_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._patched_pipeline(root) as context:
                main()

            runner = context["runner"]
            commands = [item.args[0] for item in runner.call_args_list]
            self.assertEqual(
                [_module_name(command) for command in commands],
                [
                    "scripts.train_fuzzy_dynamics",
                    "scripts.evaluate_fuzzy_dynamics",
                    "scripts.plot_trajectory",
                ],
            )
            context["generate_time"].assert_called_once_with()
            for command in commands:
                self.assertEqual(
                    _argument(command, "--experiment-time"), EXPERIMENT_TIME
                )
            for item in runner.call_args_list:
                self.assertTrue(item.kwargs["start_new_session"])
                self.assertEqual(
                    item.kwargs["cwd"], Path(__file__).resolve().parents[1]
                )

            experiment_dir = (
                context["results_root"]
                / f"fuzzy_dynamics_{EXPERIMENT_TIME}"
            )
            self.assertTrue(experiment_dir.is_dir())
            self.assertEqual(
                Path(_argument(commands[2], "--trajectories")),
                experiment_dir / "evaluation" / "trajectories.jsonl",
            )
            manifest = self._manifest(context["results_root"])
            self.assertEqual(manifest["experiment_time"], EXPERIMENT_TIME)
            self.assertEqual(manifest["experiment_dir"], str(experiment_dir))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                [manifest["stages"][name]["status"] for name in (
                    "training",
                    "evaluation",
                    "plotting",
                )],
                ["complete", "complete", "complete"],
            )
            self.assertEqual(manifest["stages"]["training"]["command"], commands[0])
            self.assertEqual(manifest["stages"]["evaluation"]["command"], commands[1])
            self.assertEqual(manifest["stages"]["plotting"]["command"], commands[2])
            self.assertIn(f"EXPERIMENT_TIME={EXPERIMENT_TIME}", context["stdout"].getvalue())
            self.assertIn(f"EXPERIMENT_DIR={experiment_dir}", context["stdout"].getvalue())

    def test_skip_plot_runs_only_training_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._patched_pipeline(root, "--skip-plot") as context:
                main()

            commands = [item.args[0] for item in context["runner"].call_args_list]
            self.assertEqual(
                [_module_name(command) for command in commands],
                ["scripts.train_fuzzy_dynamics", "scripts.evaluate_fuzzy_dynamics"],
            )
            manifest = self._manifest(context["results_root"])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["stages"]["plotting"], {"status": "skipped"})

    def test_training_failure_stops_pipeline_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._patched_pipeline(root, fail_stage="training") as context:
                with self.assertRaises(StageProcessError):
                    main()

            self.assertEqual(context["runner"].call_count, 1)
            command = context["runner"].call_args.args[0]
            self.assertEqual(_module_name(command), "scripts.train_fuzzy_dynamics")
            manifest = self._manifest(context["results_root"])
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_stage"], "training")
            self.assertEqual(manifest["stages"]["training"]["status"], "failed")
            self.assertEqual(
                manifest["stages"]["training"]["error_type"],
                "StageProcessError",
            )
            self.assertNotIn("evaluation", manifest["stages"])
            self.assertNotIn("plotting", manifest["stages"])

    def test_evaluation_failure_prevents_plotting_and_preserves_training_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._patched_pipeline(root, fail_stage="evaluation") as context:
                with self.assertRaises(StageProcessError):
                    main()

            commands = [item.args[0] for item in context["runner"].call_args_list]
            self.assertEqual(
                [_module_name(command) for command in commands],
                ["scripts.train_fuzzy_dynamics", "scripts.evaluate_fuzzy_dynamics"],
            )
            manifest = self._manifest(context["results_root"])
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_stage"], "evaluation")
            self.assertEqual(manifest["stages"]["training"]["status"], "complete")
            self.assertEqual(manifest["stages"]["evaluation"]["status"], "failed")
            self.assertNotIn("plotting", manifest["stages"])


if __name__ == "__main__":
    unittest.main()
