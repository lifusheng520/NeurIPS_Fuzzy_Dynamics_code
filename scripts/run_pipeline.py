#!/usr/bin/env python3
"""Run one isolated fuzzy-dynamics training, evaluation, and plotting pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import (
    configure_experiment_logging,
    experiment_directory,
    generate_experiment_time,
    write_json_atomic,
)


class StageProcessError(RuntimeError):
    """A pipeline child process returned a non-zero status."""

    def __init__(self, stage: str, returncode: int) -> None:
        self.returncode = returncode
        super().__init__(f"Pipeline stage {stage!r} exited with status {returncode}.")


class PipelineInterrupted(KeyboardInterrupt):
    """The pipeline received an interrupt/termination signal."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.returncode = 128 + signum
        super().__init__(f"Pipeline received signal {signum}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one independently timestamped experiment."
    )
    parser.add_argument("--activations", required=True, help="Training activation cache.")
    parser.add_argument(
        "--evaluation-activations",
        default=None,
        help="Optional separate validation/test cache; defaults to --activations.",
    )
    parser.add_argument("--config", default="configs/fuzzy_dynamics_base.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument(
        "--subset",
        choices=("auto", "all", "train", "validation"),
        default="auto",
    )
    parser.add_argument("--semantic-events", default=None)
    parser.add_argument("--event-id-field", default="uid")
    parser.add_argument("--allow-missing-events", action="store_true")
    parser.add_argument("--semantic-event-quantile", type=float, default=0.75)
    parser.add_argument("--include-operation-proxies", action="store_true")
    parser.add_argument("--skip-semantic-alignment", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--plot-index", type=int, default=0)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_existing(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _require_unchanged(path: Path, expected: dict[str, int], label: str) -> None:
    if not path.is_file() or _file_identity(path) != expected:
        raise RuntimeError(f"{label} changed while the pipeline was running: {path}")


def _validate_with_unchanged_input(
    validate: Callable[[], dict[str, str]],
    path: Path,
    expected: dict[str, int],
    label: str,
) -> dict[str, str]:
    artifacts = validate()
    _require_unchanged(path, expected, label)
    return artifacts


def _reserve_experiment(results_root: Path) -> tuple[str, Path]:
    """Atomically reserve a timestamp directory for this pipeline process."""
    results_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        experiment_time = generate_experiment_time()
        directory = experiment_directory(results_root, experiment_time)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return experiment_time, directory
    raise FileExistsError("Could not reserve a unique experiment timestamp directory.")


def _run_stage(
    stage: str,
    command: list[str],
    manifest: dict[str, Any],
    manifest_path: Path,
    logger: Any,
    project_root: Path,
    validate_artifacts: Callable[[], dict[str, str]],
    preflight: Callable[[], None] | None = None,
) -> None:
    manifest.update(current_stage=stage, updated_at=_utc_now())
    manifest["stages"][stage] = {
        "status": "running",
        "started_at": _utc_now(),
        "command": command,
    }
    write_json_atomic(manifest_path, manifest)
    logger.info("Starting %s: %s", stage, shlex.join(command))
    child: subprocess.Popen[Any] | None = None
    returncode: int | None = None
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        raise PipelineInterrupted(signum)

    try:
        if preflight is not None:
            preflight()
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        child = subprocess.Popen(command, cwd=project_root, start_new_session=True)
        returncode = child.wait()
        if returncode != 0:
            raise StageProcessError(stage, returncode)
        artifacts = validate_artifacts()
    except BaseException as error:
        if isinstance(error, PipelineInterrupted) and child is not None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest["stages"][stage].update(
            status="interrupted" if interrupted else "failed",
            finished_at=_utc_now(),
            error_type=type(error).__name__,
            error=str(error),
            returncode=getattr(error, "returncode", returncode),
            signal=getattr(error, "signum", None),
        )
        manifest.update(
            status="interrupted" if interrupted else "failed",
            failed_stage=stage,
            finished_at=_utc_now(),
            updated_at=_utc_now(),
        )
        write_json_atomic(manifest_path, manifest)
        logger.exception("Pipeline stage %s did not complete", stage)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    manifest["stages"][stage].update(
        status="complete",
        finished_at=_utc_now(),
        returncode=returncode,
        artifacts=artifacts,
    )
    manifest.update(current_stage=None, updated_at=_utc_now())
    write_json_atomic(manifest_path, manifest)
    logger.info("Completed %s", stage)


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} was not produced or is empty: {path}")
    return path


def _load_required_json(path: Path, label: str) -> dict[str, Any]:
    _required_file(path, label)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_training_artifacts(
    experiment_dir: Path,
    experiment_time: str,
) -> dict[str, str]:
    checkpoint_dir = experiment_dir / "checkpoint"
    training_dir = experiment_dir / "training"
    best = _required_file(checkpoint_dir / "best.pt", "Best checkpoint")
    last = _required_file(checkpoint_dir / "last.pt", "Last checkpoint")
    history = _required_file(training_dir / "history.json", "Training history")
    split = _required_file(training_dir / "split.json", "Training split")
    manifest_path = training_dir / "experiment_manifest.json"
    training_manifest = _load_required_json(manifest_path, "Training manifest")
    if training_manifest.get("experiment_time") != experiment_time:
        raise RuntimeError("Training manifest has the wrong experiment_time.")
    if training_manifest.get("status") != "complete":
        raise RuntimeError("Training manifest does not report status=complete.")
    return {
        "best_checkpoint": str(best.resolve()),
        "last_checkpoint": str(last.resolve()),
        "history": str(history.resolve()),
        "split": str(split.resolve()),
        "training_manifest": str(manifest_path.resolve()),
    }


def _validate_evaluation_artifacts(
    experiment_dir: Path,
    experiment_time: str,
) -> dict[str, str]:
    evaluation_dir = experiment_dir / "evaluation"
    metrics = evaluation_dir / "metrics.json"
    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    trajectories = _required_file(
        evaluation_dir / "trajectories.jsonl", "Trajectory records"
    )
    membership = _required_file(
        evaluation_dir / "membership_analysis.json", "Membership analysis"
    )
    _load_required_json(metrics, "Evaluation metrics")
    evaluation_manifest = _load_required_json(
        evaluation_manifest_path, "Evaluation manifest"
    )
    if evaluation_manifest.get("experiment_time") != experiment_time:
        raise RuntimeError("Evaluation manifest has the wrong experiment_time.")
    return {
        "metrics": str(metrics.resolve()),
        "evaluation_manifest": str(evaluation_manifest_path.resolve()),
        "membership_analysis": str(membership.resolve()),
        "trajectories": str(trajectories.resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive.")
    if args.evaluation_batch_size <= 0:
        raise ValueError("--evaluation-batch-size must be positive.")
    if args.plot_index < 0:
        raise ValueError("--plot-index must be non-negative.")

    project_root = Path(__file__).resolve().parents[1]
    activations = _resolve_existing(args.activations, "Activation cache")
    evaluation_activations = _resolve_existing(
        args.evaluation_activations or args.activations,
        "Evaluation activation cache",
    )
    config = _resolve_existing(args.config, "Experiment config")
    semantic_events = (
        _resolve_existing(args.semantic_events, "Semantic-event file")
        if args.semantic_events
        else None
    )
    activation_identity = _file_identity(activations)
    evaluation_activation_identity = _file_identity(evaluation_activations)
    results_root = Path(args.results_root).expanduser().resolve()
    log_root = Path(args.log_dir).expanduser().resolve()
    experiment_time, experiment_dir = _reserve_experiment(results_root)
    experiment_log_dir = log_root / f"fuzzy_dynamics_{experiment_time}"
    logger, pipeline_log = configure_experiment_logging(
        "run_pipeline", experiment_log_dir
    )
    training_dir = experiment_dir / "training"
    training_dir.mkdir()
    manifest_path = training_dir / "pipeline_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_time": experiment_time,
        "status": "running",
        "started_at": _utc_now(),
        "seed": args.seed,
        "device": args.device,
        "activations": str(activations),
        "activation_file_identity": activation_identity,
        "evaluation_activations": str(evaluation_activations),
        "evaluation_activation_file_identity": evaluation_activation_identity,
        "config": str(config),
        "semantic_events": str(semantic_events) if semantic_events else None,
        "experiment_dir": str(experiment_dir),
        "log_dir": str(experiment_log_dir),
        "pipeline_log": str(pipeline_log.resolve()),
        "stages": {},
    }
    write_json_atomic(manifest_path, manifest)
    logger.info("experiment_time=%s", experiment_time)
    logger.info("Experiment directory: %s", experiment_dir)

    common_log_args = [
        "--log-dir",
        str(log_root),
        "--log-every-batches",
        str(args.log_every_batches),
    ]
    train_command = [
        sys.executable,
        "-m",
        "scripts.train_fuzzy_dynamics",
        "--activations",
        str(activations),
        "--config",
        str(config),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--results-root",
        str(results_root),
        "--experiment-time",
        experiment_time,
        *common_log_args,
    ]
    _run_stage(
        "training",
        train_command,
        manifest,
        manifest_path,
        logger,
        project_root,
        lambda: _validate_with_unchanged_input(
            lambda: _validate_training_artifacts(experiment_dir, experiment_time),
            activations,
            activation_identity,
            "Training activation cache",
        ),
        lambda: _require_unchanged(
            activations, activation_identity, "Training activation cache"
        ),
    )

    evaluation_command = [
        sys.executable,
        "-m",
        "scripts.evaluate_fuzzy_dynamics",
        "--experiment-time",
        experiment_time,
        "--activations",
        str(evaluation_activations),
        "--results-root",
        str(results_root),
        "--device",
        args.device,
        "--batch-size",
        str(args.evaluation_batch_size),
        "--subset",
        args.subset,
        "--semantic-event-quantile",
        str(args.semantic_event_quantile),
        "--event-id-field",
        args.event_id_field,
        *common_log_args,
    ]
    if semantic_events:
        evaluation_command.extend(("--semantic-events", str(semantic_events)))
    for enabled, flag in (
        (args.allow_missing_events, "--allow-missing-events"),
        (args.include_operation_proxies, "--include-operation-proxies"),
        (args.skip_semantic_alignment, "--skip-semantic-alignment"),
        (args.skip_rollout, "--skip-rollout"),
    ):
        if enabled:
            evaluation_command.append(flag)
    _run_stage(
        "evaluation",
        evaluation_command,
        manifest,
        manifest_path,
        logger,
        project_root,
        lambda: _validate_with_unchanged_input(
            lambda: _validate_evaluation_artifacts(experiment_dir, experiment_time),
            evaluation_activations,
            evaluation_activation_identity,
            "Evaluation activation cache",
        ),
        lambda: _require_unchanged(
            evaluation_activations,
            evaluation_activation_identity,
            "Evaluation activation cache",
        ),
    )

    if args.skip_plot:
        manifest["stages"]["plotting"] = {"status": "skipped"}
        write_json_atomic(manifest_path, manifest)
    else:
        trajectories = experiment_dir / "evaluation" / "trajectories.jsonl"
        plot_command = [
            sys.executable,
            "-m",
            "scripts.plot_trajectory",
            "--trajectories",
            str(trajectories),
            "--index",
            str(args.plot_index),
            "--experiment-time",
            experiment_time,
            "--log-dir",
            str(log_root),
        ]
        _run_stage(
            "plotting",
            plot_command,
            manifest,
            manifest_path,
            logger,
            project_root,
            lambda: {
                "trajectory_plot": str(
                    _required_file(
                        experiment_dir
                        / "evaluation"
                        / f"trajectory_{args.plot_index}.png",
                        "Trajectory plot",
                    ).resolve()
                )
            },
        )

    manifest.update(
        status="complete",
        current_stage=None,
        finished_at=_utc_now(),
        updated_at=_utc_now(),
    )
    write_json_atomic(manifest_path, manifest)
    logger.info("Pipeline complete: %s", experiment_dir)
    print(f"EXPERIMENT_TIME={experiment_time}")
    print(f"EXPERIMENT_DIR={experiment_dir}")


if __name__ == "__main__":
    try:
        main()
    except PipelineInterrupted as error:
        raise SystemExit(error.returncode) from None
    except StageProcessError as error:
        exit_code = error.returncode if error.returncode > 0 else 128 - error.returncode
        raise SystemExit(exit_code) from None
