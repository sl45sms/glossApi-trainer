#!/bin/bash
#SBATCH -A a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=apertus-dual-ui
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
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-uenv}"
export APERTUS_UI_PORT="${APERTUS_UI_PORT:-8631}"
export APERTUS_BASE_DEVICE="${APERTUS_BASE_DEVICE:-cuda:0}"
export APERTUS_MERGED_DEVICE="${APERTUS_MERGED_DEVICE:-cuda:1}"
export APERTUS_MERGED_MODEL="${APERTUS_MERGED_MODEL:-${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged}"
export APERTUS_BASE_MODEL="${APERTUS_BASE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
export ENABLE_UI_REVERSE_TUNNEL="${ENABLE_UI_REVERSE_TUNNEL:-0}"
export UI_REVERSE_TUNNEL_HOST="${UI_REVERSE_TUNNEL_HOST:-${SLURM_SUBMIT_HOST:-}}"
export UI_REVERSE_TUNNEL_PORT="${UI_REVERSE_TUNNEL_PORT:-${APERTUS_UI_PORT}}"

echo "Allocated node: $(hostname -s)"
echo "UI job: ${SLURM_JOB_ID}"
echo "UI port: ${APERTUS_UI_PORT}"
echo "Base device: ${APERTUS_BASE_DEVICE}"
echo "Merged device: ${APERTUS_MERGED_DEVICE}"
if [[ "${ENABLE_UI_REVERSE_TUNNEL}" == "1" ]]; then
	echo "Reverse tunnel requested on submit host ${UI_REVERSE_TUNNEL_HOST}:${UI_REVERSE_TUNNEL_PORT}"
fi

srun --cpu-bind=none bash "${REPO_ROOT}/scripts/run_dual_model_ui.sh"