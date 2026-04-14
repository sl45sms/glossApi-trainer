#!/bin/bash

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH is not set." >&2
	exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export APERTUS_UI_PORT="${APERTUS_UI_PORT:-8631}"
export APERTUS_BASE_DEVICE="${APERTUS_BASE_DEVICE:-cuda:0}"
export APERTUS_MERGED_DEVICE="${APERTUS_MERGED_DEVICE:-cuda:1}"
export APERTUS_MERGED_MODEL="${APERTUS_MERGED_MODEL:-${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged}"
export APERTUS_BASE_MODEL="${APERTUS_BASE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"

echo "Requesting an interactive Clariden node for the dual-model UI..."
echo "Port: ${APERTUS_UI_PORT}"
echo "Base device: ${APERTUS_BASE_DEVICE}"
echo "Merged device: ${APERTUS_MERGED_DEVICE}"

exec srun \
	--account="${SLURM_ACCOUNT_OVERRIDE:-a0140}" \
	--partition="${SLURM_PARTITION_OVERRIDE:-debug}" \
	--nodes=1 \
	--ntasks=1 \
	--gpus-per-node="${SLURM_GPUS_PER_NODE_OVERRIDE:-4}" \
	--cpus-per-task="${SLURM_CPUS_PER_TASK_OVERRIDE:-72}" \
	--mem="${SLURM_MEM_OVERRIDE:-128G}" \
	--time="${SLURM_TIME_OVERRIDE:-00:30:00}" \
	--job-name="${SLURM_JOB_NAME_OVERRIDE:-apertus-dual-ui}" \
	--unbuffered \
	bash "${repo_root}/scripts/run_dual_model_ui.sh"