#!/bin/bash
#SBATCH -A a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=apertus-lora-clr
#SBATCH --output=%x-%j.out

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
export DATASET_PATH="${DATASET_PATH:-${SCRATCH}/glossapi-trainer/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/hf_datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton-cache}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-1024}"
export TRAIN_GRADIENT_CHECKPOINTING="${TRAIN_GRADIENT_CHECKPOINTING:-false}"

if [[ ! -d "${VENV_DIR}" ]]; then
	echo "Virtual environment not found: ${VENV_DIR}" >&2
	echo "Run bash scripts/setup_uenv_python.sh first." >&2
	exit 1
fi

if [[ ! -d "${DATASET_PATH}" && ! -f "${DATASET_PATH}" ]]; then
	echo "Dataset path not found: ${DATASET_PATH}" >&2
	echo "Run bash scripts/stage_glossapi_dataset.sh first." >&2
	exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

uenv run "${UENV_IMAGE}" --view="${UENV_VIEW}" -- bash -lc '
set -euo pipefail
source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"
python "${REPO_ROOT}/scripts/sft_train_glossapi.py" \
	--config "${REPO_ROOT}/configs/sft_lora_glossapi.yaml" \
	--dataset_name "${DATASET_PATH}" \
	--output_dir "${OUTPUT_DIR}" \
	--max_steps "${TRAIN_MAX_STEPS}" \
	--per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
	--max_length "${TRAIN_MAX_LENGTH}" \
	--gradient_checkpointing "${TRAIN_GRADIENT_CHECKPOINTING}"
'
