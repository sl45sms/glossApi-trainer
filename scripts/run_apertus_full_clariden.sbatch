#!/bin/bash
#SBATCH -A a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=apertus-full-clr
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
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/glossapi-trainer/output/apertus_full}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/hf_datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton-cache}"
export TOKENIZERS_PARALLELISM="false"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-nmn0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-nmn0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
unset CUDA_VISIBLE_DEVICES || true

uenv run "${UENV_IMAGE}" --view="${UENV_VIEW}" -- bash -lc '
set -euo pipefail
source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"
accelerate launch --config_file "${REPO_ROOT}/configs/zero3_clariden.yaml" \
	"${REPO_ROOT}/scripts/sft_train_glossapi.py" \
	--config "${REPO_ROOT}/configs/sft_full_glossapi.yaml" \
	--dataset_name "${DATASET_PATH}" \
	--output_dir "${OUTPUT_DIR}"
'
