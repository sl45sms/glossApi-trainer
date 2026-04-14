#!/bin/bash

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH is not set." >&2
	exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export APERTUS_UI_PORT="${APERTUS_UI_PORT:-8631}"
export APERTUS_BASE_MODEL="${APERTUS_BASE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
export APERTUS_MERGED_MODEL="${APERTUS_MERGED_MODEL:-${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged}"
export APERTUS_BASE_DEVICE="${APERTUS_BASE_DEVICE:-cuda:0}"
export APERTUS_MERGED_DEVICE="${APERTUS_MERGED_DEVICE:-cuda:1}"
export APERTUS_BASE_DTYPE="${APERTUS_BASE_DTYPE:-${APERTUS_UI_DTYPE:-float32}}"
export APERTUS_MERGED_DTYPE="${APERTUS_MERGED_DTYPE:-${APERTUS_UI_DTYPE:-bfloat16}}"
export ENABLE_UI_REVERSE_TUNNEL="${ENABLE_UI_REVERSE_TUNNEL:-0}"
export UI_REVERSE_TUNNEL_HOST="${UI_REVERSE_TUNNEL_HOST:-}"
export UI_REVERSE_TUNNEL_PORT="${UI_REVERSE_TUNNEL_PORT:-${APERTUS_UI_PORT}}"
export APERTUS_UI_JOB_NAME="${APERTUS_UI_JOB_NAME:-apertus-dual-ui}"
export VERIFY_UI_MODEL_PATH_ON_SUBMIT="${VERIFY_UI_MODEL_PATH_ON_SUBMIT:-0}"
export APERTUS_UI_PARTITION="${APERTUS_UI_PARTITION:-normal}"
export APERTUS_UI_GPUS_PER_NODE="${APERTUS_UI_GPUS_PER_NODE:-2}"
export APERTUS_UI_CPUS_PER_TASK="${APERTUS_UI_CPUS_PER_TASK:-36}"
export APERTUS_UI_MEM="${APERTUS_UI_MEM:-128G}"
export APERTUS_UI_TIME="${APERTUS_UI_TIME:-02:30:00}"

if [[ "${VERIFY_UI_MODEL_PATH_ON_SUBMIT}" == "1" ]]; then
	if [[ ! -d "${APERTUS_MERGED_MODEL}" ]]; then
		echo "Merged model directory not found: ${APERTUS_MERGED_MODEL}" >&2
		exit 1
	fi
fi

echo "Submitting dual-model UI job" >&2
echo "Merged model: ${APERTUS_MERGED_MODEL}" >&2
echo "Port: ${APERTUS_UI_PORT}" >&2
echo "Base device: ${APERTUS_BASE_DEVICE}" >&2
echo "Merged device: ${APERTUS_MERGED_DEVICE}" >&2
echo "Base dtype: ${APERTUS_BASE_DTYPE}" >&2
echo "Merged dtype: ${APERTUS_MERGED_DTYPE}" >&2
echo "Partition: ${APERTUS_UI_PARTITION}" >&2
echo "GPUs per node: ${APERTUS_UI_GPUS_PER_NODE}" >&2
echo "CPUs per task: ${APERTUS_UI_CPUS_PER_TASK}" >&2
echo "Memory: ${APERTUS_UI_MEM}" >&2
if [[ "${ENABLE_UI_REVERSE_TUNNEL}" == "1" ]]; then
	echo "Reverse tunnel: enabled" >&2
	if [[ -n "${UI_REVERSE_TUNNEL_HOST}" ]]; then
		echo "Reverse tunnel host: ${UI_REVERSE_TUNNEL_HOST}" >&2
	fi
fi

exec sbatch \
	--job-name="${APERTUS_UI_JOB_NAME}" \
	--partition="${APERTUS_UI_PARTITION}" \
	--gpus-per-node="${APERTUS_UI_GPUS_PER_NODE}" \
	--cpus-per-task="${APERTUS_UI_CPUS_PER_TASK}" \
	--mem="${APERTUS_UI_MEM}" \
	--time="${APERTUS_UI_TIME}" \
	"$@" \
	"${repo_root}/scripts/run_dual_model_ui_clariden.sbatch"