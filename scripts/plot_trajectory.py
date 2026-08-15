#!/usr/bin/env python3
"""Plot one query's five-mode fuzzy membership trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fuzzy_dynamics import REASONING_MODES
from src.utils import (
    configure_experiment_logging,
    generate_experiment_time,
    validate_experiment_time,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--index", type=int, default=0, help="Zero-based JSONL record index.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image; defaults beside trajectories.jsonl.",
    )
    parser.add_argument("--experiment-time", default=None)
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectories)
    inferred_time = None
    if trajectory_path.parent.name == "evaluation":
        experiment_name = trajectory_path.parent.parent.name
        prefix = "fuzzy_dynamics_"
        if experiment_name.startswith(prefix):
            inferred_time = experiment_name[len(prefix) :]
    try:
        experiment_time = validate_experiment_time(args.experiment_time or inferred_time or "")
    except ValueError:
        experiment_time = generate_experiment_time()
    logger, _ = configure_experiment_logging(
        "plot_trajectory", Path(args.log_dir) / f"fuzzy_dynamics_{experiment_time}"
    )
    logger.info("Arguments: %s", vars(args))
    logger.info("experiment_time=%s", experiment_time)
    import matplotlib.pyplot as plt

    with trajectory_path.open(encoding="utf-8") as handle:
        records = (json.loads(line) for line in handle if line.strip())
        record = next(value for index, value in enumerate(records) if index == args.index)

    layers = [item["layer"] for item in record["trajectory"]]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for mode in REASONING_MODES:
        axis.plot(
            layers,
            [item["membership"][mode] for item in record["trajectory"]],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=mode.replace("_", " ").title(),
        )
    axis.set(xlabel="Transformer layer", ylabel="Fuzzy membership", ylim=(0.0, 1.0))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    title = str(record.get("prompt", f"Query {args.index}"))
    axis.set_title(title if len(title) <= 100 else title[:97] + "...")
    figure.tight_layout()
    output = Path(args.output or trajectory_path.parent / f"trajectory_{args.index}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    logger.info("Saved %s", output.resolve())


if __name__ == "__main__":
    main()
