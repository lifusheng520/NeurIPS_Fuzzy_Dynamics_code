#!/usr/bin/env bash
#SBATCH --job-name=extract-activations
#SBATCH --partition=hopper140
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/cluster/slurm-%x_%j.out
#SBATCH --error=logs/cluster/slurm-%x_%j.err

set -euo pipefail

# Compute nodes must use the model, dataset, and Tuned Lens already copied into
# this repository. A missing local artifact should fail instead of downloading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

# Do not mix cluster-wide Python packages with the project virtual environment.
# The repository is the working directory, so it remains importable without a
# custom PYTHONPATH.
unset PYTHONHOME
unset PYTHONPATH

# Slurm executes a spool copy of this file, so BASH_SOURCE points under
# /var/spool/slurmd rather than to the repository. Jobs submitted from the
# project root can reliably use SLURM_SUBMIT_DIR.
if [[ -n "${FUZZY_PROJECT_ROOT:-}" ]]; then
    FUZZY_PROJECT_ROOT="$(cd -- "${FUZZY_PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    FUZZY_PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    FUZZY_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${FUZZY_PROJECT_ROOT}/scripts/extract_activations.py" ]]; then
    echo "Project root is invalid: ${FUZZY_PROJECT_ROOT}" >&2
    echo "Submit from the project root or export FUZZY_PROJECT_ROOT explicitly." >&2
    exit 2
fi
cd "${FUZZY_PROJECT_ROOT}"

PYTHON_EXECUTABLE="${FUZZY_PROJECT_ROOT}/.venv/bin/python"

MODEL_PATH="${MODEL_PATH:-data/models/Meta-Llama-3-8B}"
INPUT_PATH="${INPUT_PATH:-data/datasets/latent_multi_hop/SOCRATES_v1.csv}"
LENS_PATH="${LENS_PATH:-data/tuned_lens/lens/meta-llama/Meta-Llama-3-8B}"
OUTPUT_PATH="${OUTPUT_PATH:-cache/socrates_llama3_8b_tuned_256.pt}"
LIMIT="${LIMIT:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-10}"
LOG_DIR="${LOG_DIR:-logs/extract_activations}"

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
    echo "Project virtual-environment Python not found: ${PYTHON_EXECUTABLE}" >&2
    echo "Create .venv in the project root and install requirements.txt before submitting." >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "Complete local model directory not found: ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local tokenizer file not found: ${MODEL_PATH}/tokenizer.json" >&2
    exit 2
fi
if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Local input dataset not found: ${INPUT_PATH}" >&2
    exit 2
fi
if [[ ! -f "${LENS_PATH}/config.json" || ! -f "${LENS_PATH}/params.pt" ]]; then
    echo "Complete local Tuned Lens directory not found: ${LENS_PATH}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_PATH}" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "Output already exists: ${OUTPUT_PATH}" >&2
    echo "Set OVERWRITE=1 only if replacing it is intentional." >&2
    exit 2
fi

echo "host=$(hostname)"
echo "project_root=${FUZZY_PROJECT_ROOT}"
echo "python=${PYTHON_EXECUTABLE}"
echo "model=${MODEL_PATH}"
echo "dataset=${INPUT_PATH}"
echo "tuned_lens=${LENS_PATH}"
echo "output=${OUTPUT_PATH} limit=${LIMIT} batch_size=${BATCH_SIZE}"

# Import NumPy before PyTorch in the same interpreter. This avoids a binary
# extension re-initialization issue observed with the cluster's Python stack.
"${PYTHON_EXECUTABLE}" -c \
    'import numpy, runpy; runpy.run_module("scripts.check_environment", run_name="__main__")' \
    --require-cuda

"${PYTHON_EXECUTABLE}" -c \
    'import numpy, runpy; runpy.run_module("scripts.extract_activations", run_name="__main__")' \
    --model "${MODEL_PATH}" \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --limit "${LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --lens tuned \
    --lens-resource "${LENS_PATH}" \
    --dtype "${DTYPE}" \
    --device cuda \
    --log-dir "${LOG_DIR}" \
    --log-every-batches "${LOG_EVERY_BATCHES}"
