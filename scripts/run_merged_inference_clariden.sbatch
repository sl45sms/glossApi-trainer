#!/bin/bash
#SBATCH -A a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=apertus-merged-infer
#SBATCH --output=%x-%j.out

# This script runs a smoke test inference prompt against a merged Apertus model.
# It loads the merged model and tokenizer from a specified directory,
# and generates a response to a given prompt, printing the result as JSON.
# you can set the prompt to test via the INFER_PROMPT environment variable when submitting the job, e.g.
# INFER_PROMPT="Ποιος είναι ο σκοπός του Βιβλίου του Δασκάλου;" sbatch scripts/run_merged


set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${repo_root}" || ! -d "${repo_root}" ]]; then
	repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${repo_root}"

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH is not set." >&2
	exit 1
fi

export REPO_ROOT="${repo_root}"
export UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
export UENV_VIEW="${UENV_VIEW:-default}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-uenv}"
export MODEL_DIR="${MODEL_DIR:-${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke_merged}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/hf_datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton-cache}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFER_PROMPT="${INFER_PROMPT:-}"

if [[ ! -d "${VENV_DIR}" ]]; then
	echo "Virtual environment not found: ${VENV_DIR}" >&2
	echo "Run bash scripts/setup_uenv_python.sh first." >&2
	exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
	echo "Merged model directory not found: ${MODEL_DIR}" >&2
	exit 1
fi

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

uenv run "${UENV_IMAGE}" --view="${UENV_VIEW}" -- bash -lc '
set -euo pipefail
source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

cmd=(python "${REPO_ROOT}/scripts/smoke_test_merged_model.py" --model_dir "${MODEL_DIR}")
if [[ -n "${INFER_PROMPT}" ]]; then
	cmd+=(--prompt "${INFER_PROMPT}")
fi

"${cmd[@]}"
'