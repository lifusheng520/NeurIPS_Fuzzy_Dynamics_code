#!/usr/bin/env bash
#SBATCH --job-name=fuzzy-dynamics
#SBATCH --partition=hopper140
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err




set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ -n "${FUZZY_PROJECT_ROOT:-}" ]]; then
    FUZZY_PROJECT_ROOT="$(cd -- "${FUZZY_PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    FUZZY_PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    FUZZY_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${FUZZY_PROJECT_ROOT}/scripts/run_pipeline.py" ]]; then
    echo "Project root is invalid: ${FUZZY_PROJECT_ROOT}" >&2
    echo "Submit from the project root or export FUZZY_PROJECT_ROOT explicitly." >&2
    exit 2
fi
cd "${FUZZY_PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVATIONS="${ACTIVATIONS:?Set ACTIVATIONS to an existing activation-cache .pt file.}"
CONFIG="${CONFIG:-configs/fuzzy_dynamics_base.json}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-10}"
SUBSET="${SUBSET:-auto}"
PLOT_INDEX="${PLOT_INDEX:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${ACTIVATIONS}" ]]; then
    echo "Activation cache not found: ${ACTIVATIONS}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Experiment config not found: ${CONFIG}" >&2
    exit 2
fi
if [[ -n "${EVALUATION_ACTIVATIONS:-}" && ! -f "${EVALUATION_ACTIVATIONS}" ]]; then
    echo "Evaluation activation cache not found: ${EVALUATION_ACTIVATIONS}" >&2
    exit 2
fi
if [[ -n "${SEMANTIC_EVENTS:-}" && ! -f "${SEMANTIC_EVENTS}" ]]; then
    echo "Semantic-event file not found: ${SEMANTIC_EVENTS}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "project_root=${FUZZY_PROJECT_ROOT}"
echo "activations=${ACTIVATIONS}"
echo "config=${CONFIG} seed=${SEED} device=${DEVICE}"

PIPELINE_COMMAND=(
    "${PYTHON_BIN}" -m scripts.run_pipeline
    --activations "${ACTIVATIONS}"
    --config "${CONFIG}"
    --seed "${SEED}"
    --results-root "${RESULTS_ROOT}"
    --device "${DEVICE}"
    --log-dir "${LOG_ROOT}"
    --log-every-batches "${LOG_EVERY_BATCHES}"
    --evaluation-batch-size "${EVAL_BATCH_SIZE}"
    --subset "${SUBSET}"
    --plot-index "${PLOT_INDEX}"
)
if [[ -n "${EVALUATION_ACTIVATIONS:-}" ]]; then
    PIPELINE_COMMAND+=(--evaluation-activations "${EVALUATION_ACTIVATIONS}")
fi
if [[ -n "${SEMANTIC_EVENTS:-}" ]]; then
    PIPELINE_COMMAND+=(
        --semantic-events "${SEMANTIC_EVENTS}"
        --event-id-field "${EVENT_ID_FIELD:-uid}"
    )
fi
if [[ "${ALLOW_MISSING_EVENTS:-0}" == "1" ]]; then
    PIPELINE_COMMAND+=(--allow-missing-events)
fi
if [[ "${INCLUDE_OPERATION_PROXIES:-0}" == "1" ]]; then
    PIPELINE_COMMAND+=(--include-operation-proxies)
fi
if [[ "${SKIP_SEMANTIC_ALIGNMENT:-0}" == "1" ]]; then
    PIPELINE_COMMAND+=(--skip-semantic-alignment)
fi
if [[ "${SKIP_ROLLOUT:-0}" == "1" ]]; then
    PIPELINE_COMMAND+=(--skip-rollout)
fi
if [[ "${SKIP_PLOT:-0}" == "1" ]]; then
    PIPELINE_COMMAND+=(--skip-plot)
fi
if [[ -n "${SEMANTIC_EVENT_QUANTILE:-}" ]]; then
    PIPELINE_COMMAND+=(--semantic-event-quantile "${SEMANTIC_EVENT_QUANTILE}")
fi

"${PIPELINE_COMMAND[@]}"
