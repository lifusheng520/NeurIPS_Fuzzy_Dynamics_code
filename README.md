# Fuzzy Reasoning Dynamical System

This repository implements a fuzzy local dynamical-system view of layer-wise
LLM reasoning. For every prompt and Transformer layer it extracts residual,
attention, MLP, latent-belief, and uncertainty signals, then learns five
reasoning-semantic local dynamics:

1. knowledge enrichment;
2. information retrieval/routing;
3. concept composition/inference;
4. prediction refinement/answer commitment;
5. hop transition/reasoning shift.

The learned membership vector remains soft and sums to one at every layer. The
primary training target is reconstruction of the observed state transition.

## Project layout

```text
src/llm/                         model adapters and activation extraction
src/fuzzy_dynamics/              projectors, F1--F5, memberships, losses
src/data/                        prompt and activation-cache datasets
src/evaluation/                  reconstruction and trajectory export
scripts/                         extraction, training, and evaluation CLIs
cluster/                         Slurm single-run and job-array entry points
configs/fuzzy_dynamics_base.json default experiment configuration
docs/IMPLEMENTATION.md             formula-to-code mapping and experiment cautions
data/datasets/latent_multi_hop/  TwoHopFact and SOCRATES v1
```

The author's earlier FuzzyControl code is retained under
`src/fuzzy_dynamics/fuzzy_control` as a separate downstream-control baseline.

## Installation

Create a Python 3.10+ environment, then install the base dependencies:

```bash
pip install -r requirements.txt
```

The single requirements file also contains the optional Tuned Lens and
platform-gated vLLM dependencies used by the full experiments.

## 1. Extract activations

Start with a small extraction to verify model compatibility and memory use:

```bash
python -m scripts.extract_activations \
  --model Qwen/Qwen2.5-1.5B \
  --input data/datasets/latent_multi_hop/SOCRATES_v1.csv \
  --output cache/socrates_qwen_256.pt \
  --limit 256 \
  --batch-size 2 \
  --max-length 256
```

The cache contains:

- `hidden`: `[N, L+1, D]`, including the input to layer 0;
- `attention` and `mlp`: `[N, L, D]` operation outputs;
- `belief`: a fixed Gaussian projection of layer-wise log probabilities;
- `uncertainty`, `margin`, top-k predictions;
- bridge-entity and answer first-token latent log probabilities when labels are
  available in the input dataset.

The fixed belief projection makes the cache tractable. It is a logit-lens
baseline for the paper's general `P_b(log p_l)` term; experiments using a
trained Tuned Lens should be reported separately. The unified
`requirements.txt` includes Tuned Lens; enable it with `--lens tuned` and
optionally provide `--lens-resource` for a local or Hub lens checkpoint.

## 2. Train fuzzy dynamics

Every independent experiment receives an automatic microsecond timestamp.
Checkpoints, numerical results, and logs are separated by that time:

```text
results/fuzzy_dynamics_<experiment-time>/
├── checkpoint/
├── training/
└── evaluation/
logs/fuzzy_dynamics_<experiment-time>/
```

```bash
python -m scripts.train_fuzzy_dynamics \
  --activations cache/socrates_qwen_256.pt \
  --config configs/fuzzy_dynamics_base.json \
  --seed 42
```

The objective combines transition reconstruction, semantic-prior KL,
pairwise dynamics diversity, fuzzy entropy, and global mode balance. `best.pt`
and `last.pt` go under the experiment's `checkpoint/`; split indices, training
history, and an experiment manifest go under `training/`. SOCRATES/TwoHopFact experiments default to a category-level split;
the trainer falls back to a seeded query split when category metadata is absent.

The generated directory name has the form `YYYYMMDD_HHMMSS_microseconds`.
Multiple independent seeds require no manually chosen name:

```bash
python -m scripts.train_fuzzy_dynamics --activations cache/socrates_qwen_256.pt --seed 41
python -m scripts.train_fuzzy_dynamics --activations cache/socrates_qwen_256.pt --seed 42
python -m scripts.train_fuzzy_dynamics --activations cache/socrates_qwen_256.pt --seed 43
```

Each invocation receives a new timestamp, so the three runs cannot overwrite
one another. The seed/config/cache mapping is stored in `experiment_manifest.json`.

Ready-to-run loss ablations are provided under `configs/ablation_*.json`.

## 3. Export reasoning trajectories

Evaluation artifacts go to the selected run's `evaluation/` directory.

```bash
python -m scripts.evaluate_fuzzy_dynamics \
  --experiment-time 20260816_143025_123456 \
  --activations cache/socrates_qwen_256.pt
```

With `--experiment-time`, evaluation automatically loads
`results/fuzzy_dynamics_<experiment-time>/checkpoint/best.pt`, reads the recorded training
split, selects validation queries, and writes to
`results/fuzzy_dynamics_<experiment-time>/evaluation/`. Use `--latest` to select
the newest completed timestamp. Explicit `--checkpoint`,
`--split-file`, and `--output-dir` remain available for legacy/custom layouts.
Passing a separate test cache evaluates all of that cache and marks its holdout
status as externally unverified in the manifest.

`metrics.json` contains:

- one-step MSE, MAE, cosine similarity, overall dynamics R-squared, and separate
  `z`/`concept`/`belief`/`uncertainty` block R-squared values plus Macro-R-squared;
- operation-conditioned multi-step rollout error, coordinate MSE, final-state
  error, and per-horizon errors;
- per-mode semantic-event AUROC and average precision when the required bridge
  and answer signals or external event annotations are available.

The automatic semantic events implement the proposal's F3/F4/F5 diagnostics.
They overlap with signals used by the training prior and are therefore marked
as non-independent. F1/F2 attention/MLP norm proxies are available only with
`--include-operation-proxies`. A strict five-mode Semantic Alignment macro is
reported only when independent labels are supplied for all modes:

```bash
python -m scripts.evaluate_fuzzy_dynamics \
  --experiment-time 20260816_143025_123456 \
  --activations cache/socrates_qwen_256.pt \
  --semantic-events data/evaluation/semantic_events.jsonl
```

Each event JSONL row is joined by `uid`; arrays contain one `0`, `1`, or `null`
per transition `s_l -> s_(l+1)`:

```json
{"uid":"q1","events":{"hop_transition":[0,null,1]},"provenance":{"source":"human-v1","independent_of_training":true}}
```

`trajectories.jsonl` contains the full five-dimensional membership and dominant
mode for every query and layer. `membership_analysis.json` contains aggregate
mode usage, weighted contributions, layer profiles, the dominant-mode
transition matrix, and explicitly labelled training-prior agreement diagnostics.
`evaluation_manifest.json` records the selected subset and holdout status.

Plot an individual trajectory with:

```bash
python -m scripts.plot_trajectory \
  --trajectories results/fuzzy_dynamics_20260816_143025_123456/evaluation/trajectories.jsonl \
  --index 0
```

## 4. Run one complete experiment

After extracting an activation cache once, use the pipeline entry point for a
complete independent training, validation evaluation, and trajectory plot:

```bash
python -m scripts.run_pipeline \
  --activations cache/socrates_qwen_256.pt \
  --config configs/fuzzy_dynamics_base.json \
  --seed 42 \
  --device cuda
```

The pipeline atomically reserves one microsecond timestamp and passes that exact
timestamp to every stage. It never uses `--latest`, so concurrent GPU jobs cannot
evaluate another job's checkpoint. The resulting layout is:

```text
results/fuzzy_dynamics_<experiment-time>/
├── checkpoint/{best.pt,last.pt}
├── training/{history.json,split.json,experiment_manifest.json,pipeline_manifest.json}
└── evaluation/{metrics.json,evaluation_manifest.json,membership_analysis.json,trajectories.jsonl,trajectory_0.png}
```

`pipeline_manifest.json` records every stage command, status, return code, and
validated artifacts. A failed or interrupted stage stops all later stages. Use
`--skip-plot` on a headless/minimal environment, or
`--evaluation-activations path/to/test_cache.pt` to evaluate a separate cache.
The pipeline treats all input caches as read-only and detects if one changes
while the experiment is running.

## Cluster/Slurm runs

The Slurm jobs run the training, evaluation, and plotting pipeline on an existing
activation cache; extract that cache once before submitting seed/configuration
sweeps. Slurm opens its output
file before the job script starts, so create the scheduler-log directory first.
Activate the Python environment before submission, or export `PYTHON_BIN` as
the environment's Python executable. Run the submission commands from the
project root so the scheduler's relative log path resolves to `logs/slurm/`.

Submit one base-config experiment:

```bash
mkdir -p logs/slurm
export ACTIVATIONS=cache/socrates_qwen_256.pt
sbatch --export=ALL cluster/run_single_experiment.sbatch
```

`CONFIG`, `SEED`, `RESULTS_ROOT`, `LOG_ROOT`, `DEVICE`, and
`EVAL_BATCH_SIZE` can be exported to override their defaults. Set
`SEMANTIC_EVENTS` to evaluate independent event annotations. Each job calls the
pipeline entry point, which reserves one timestamp and passes it explicitly to
training, evaluation, and plotting. It does not use `--latest`, which would be
unsafe when jobs overlap.

The array script defaults to three seeds (`41`, `42`, `43`) across the base
configuration and all three supplied ablations, for 12 tasks:

```bash
mkdir -p logs/slurm
export ACTIVATIONS=cache/socrates_qwen_256.pt
sbatch --export=ALL cluster/run_experiment_array.sbatch
```

Use colon-separated `SEEDS` and `CONFIGS` for a custom Cartesian product, and
set the array range to `0` through `number-of-seeds * number-of-configs - 1`:

```bash
export SEEDS=41:42:43:44:45
export CONFIGS=configs/fuzzy_dynamics_base.json:configs/ablation_no_semantic.json
sbatch --array=0-9 --export=ALL cluster/run_experiment_array.sbatch
```

Override site-specific resources such as partition, account, GPU type, memory,
or wall time with normal `sbatch` options. Scheduler stdout/stderr goes under
`logs/slurm/`; the Python experiment logs retain the timestamped `logs/` layout
described below.

## Tests

```bash
python -m unittest discover -s tests -v
```

The extraction test uses a tiny local mock Transformer and does not download a
model. A real-model smoke test should still be run with the exact model version
used in the paper because Hugging Face module layouts can differ.

## Runtime logs

All command-line experiments write timestamped logs to the project-level
`logs/` directory while continuing to show the same progress in the terminal:

```text
logs/
├── extract_activations/<cache-name>/extract_activations_<timestamp>.log
├── fuzzy_dynamics_<experiment-time>/
│   ├── run_pipeline_<timestamp>.log
│   ├── train_fuzzy_dynamics_<timestamp>.log
│   ├── evaluate_fuzzy_dynamics_<timestamp>.log
│   └── plot_trajectory_<timestamp>.log
└── latent_multi_hop/<experiment>_<timestamp>.log
```

Use `--log-dir` (or `--log_dir` for the imported latent-multi-hop CLIs) to
override the directory. Extraction, training, and evaluation record batch
progress every 10 batches by default; change this with
`--log-every-batches`. Training additionally records every epoch, saves
`last.pt` every epoch, and atomically refreshes `history.json` after every
epoch. Follow a running job with:

```bash
tail -f logs/fuzzy_dynamics_20260816_143025_123456/train_fuzzy_dynamics_*.log
```

## Experimental cautions

- A low reconstruction error alone does not establish that the five modes have
  the intended semantics. Report semantic-alignment, intervention, and ablation
  experiments as separate evidence.
- The current rollout is conditioned on attention and MLP operations extracted
  from the real LLM trajectory. It is not an autonomous Transformer simulation.
- Bridge and answer event scores currently use only each target's first token.
  Do not describe them as full multi-token entity probabilities.
- Norm-based routing/enrichment scores are weak priors. Attention-flow scores,
  trained concept probes, and Patchscopes results should be evaluated as
  stronger alternatives.
- Keep the train/validation split at the query or fact-composition level when
  making paper claims. The base config requests a category-level split and
  falls back to a seeded query-level split when complete category labels are
  unavailable.
- The entropy term in the proposal has an easy sign ambiguity. This
  implementation minimizes negative entropy so that positive weight encourages
  soft membership instead of hard assignments.
