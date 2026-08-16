#!/usr/bin/env bash
#SBATCH --job-name=fuzzy-array
#SBATCH --array=0-11
#SBATCH --partition=hopper140
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/slurm-%x_%A_%a.out
#SBATCH --error=logs/slurm-%x_%A_%a.err

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
if [[ ! -f "${FUZZY_PROJECT_ROOT}/cluster/run_single_experiment.sbatch" ]]; then
    echo "Project root is invalid: ${FUZZY_PROJECT_ROOT}" >&2
    echo "Submit from the project root or export FUZZY_PROJECT_ROOT explicitly." >&2
    exit 2
fi
export FUZZY_PROJECT_ROOT

# Colon-separated values are convenient to export through sbatch. The defaults
# form 4 configs x 3 seeds = 12 tasks, matching the default --array range.
SEEDS="${SEEDS:-41:42:43}"
CONFIGS="${CONFIGS:-configs/fuzzy_dynamics_base.json:configs/ablation_no_semantic.json:configs/ablation_no_diversity.json:configs/ablation_no_fuzzy_regularization.json}"
IFS=':' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=':' read -r -a CONFIG_LIST <<< "${CONFIGS}"

if (( ${#SEED_LIST[@]} == 0 || ${#CONFIG_LIST[@]} == 0 )); then
    echo "SEEDS and CONFIGS must each contain at least one value." >&2
    exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm job array.}"
SEED_COUNT="${#SEED_LIST[@]}"
CONFIG_COUNT="${#CONFIG_LIST[@]}"
TASK_COUNT=$((SEED_COUNT * CONFIG_COUNT))
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
    echo "Array task ${TASK_ID} is outside the configured 0-$((TASK_COUNT - 1)) range." >&2
    echo "Submit with: sbatch --array=0-$((TASK_COUNT - 1)) cluster/run_experiment_array.sbatch" >&2
    exit 2
fi

CONFIG_INDEX=$((TASK_ID / SEED_COUNT))
SEED_INDEX=$((TASK_ID % SEED_COUNT))
export CONFIG="${CONFIG_LIST[CONFIG_INDEX]}"
export SEED="${SEED_LIST[SEED_INDEX]}"

echo "array_task=${TASK_ID}/${TASK_COUNT} config_index=${CONFIG_INDEX} seed_index=${SEED_INDEX}"
echo "config=${CONFIG} seed=${SEED}"

# The single-job script is also the array worker. Its #SBATCH lines are comments
# here, so the resource request and Slurm log path above remain authoritative.
exec bash "${FUZZY_PROJECT_ROOT}/cluster/run_single_experiment.sbatch"
