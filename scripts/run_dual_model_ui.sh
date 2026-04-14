#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

discover_default_merged_model() {
	if [[ -z "${SCRATCH:-}" ]]; then
		return 1
	fi

	local candidate
	for candidate in \
		"${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged" \
		"${SCRATCH}/glossapi-trainer/output/apertus_lora_short100_merged" \
		"${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke_merged"
	do
		if [[ -d "${candidate}" ]]; then
			printf '%s\n' "${candidate}"
			return 0
		fi
	done

	printf '%s\n' "${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged"
	return 0
}

if [[ ! -d "${repo_root}/UI" ]]; then
	echo "UI directory not found: ${repo_root}/UI" >&2
	exit 1
fi

export REPO_ROOT="${repo_root}"
export UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
export UENV_VIEW="${UENV_VIEW:-default}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-uenv}"
export APERTUS_UI_HOST="${APERTUS_UI_HOST:-0.0.0.0}"
export APERTUS_UI_PORT="${APERTUS_UI_PORT:-8631}"
export APERTUS_BASE_DEVICE="${APERTUS_BASE_DEVICE:-cuda:0}"
export APERTUS_MERGED_DEVICE="${APERTUS_MERGED_DEVICE:-cuda:1}"
export APERTUS_BASE_DTYPE="${APERTUS_BASE_DTYPE:-${APERTUS_UI_DTYPE:-float32}}"
export APERTUS_MERGED_DTYPE="${APERTUS_MERGED_DTYPE:-${APERTUS_UI_DTYPE:-bfloat16}}"
export TOKENIZERS_PARALLELISM="false"
export ENABLE_UI_REVERSE_TUNNEL="${ENABLE_UI_REVERSE_TUNNEL:-0}"
export UI_REVERSE_TUNNEL_HOST="${UI_REVERSE_TUNNEL_HOST:-${SLURM_SUBMIT_HOST:-}}"
export UI_REVERSE_TUNNEL_PORT="${UI_REVERSE_TUNNEL_PORT:-${APERTUS_UI_PORT}}"

if [[ -z "${APERTUS_MERGED_MODEL:-}" ]]; then
	if merged_model_default="$(discover_default_merged_model 2>/dev/null)"; then
		export APERTUS_MERGED_MODEL="${merged_model_default}"
	fi
fi

if [[ -n "${SCRATCH:-}" ]]; then
	export HF_HOME="${HF_HOME:-${SCRATCH}/hf}"
	export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/hf_datasets}"
	export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton-cache}"
	mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
	echo "Virtual environment not found: ${VENV_DIR}" >&2
	echo "Run bash scripts/setup_uenv_python.sh first." >&2
	exit 1
fi

if [[ -n "${APERTUS_MERGED_MODEL:-}" && ! -d "${APERTUS_MERGED_MODEL}" ]]; then
	echo "Merged model directory not found: ${APERTUS_MERGED_MODEL}" >&2
	exit 1
fi

tunnel_pid=""
cleanup() {
	if [[ -n "${tunnel_pid}" ]]; then
		kill "${tunnel_pid}" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT

echo "Launching Apertus UI on host $(hostname -s) port ${APERTUS_UI_PORT}"
echo "Base device: ${APERTUS_BASE_DEVICE}"
echo "Merged device: ${APERTUS_MERGED_DEVICE}"
echo "Base dtype: ${APERTUS_BASE_DTYPE}"
echo "Merged dtype: ${APERTUS_MERGED_DTYPE}"
if [[ -n "${APERTUS_MERGED_MODEL:-}" ]]; then
	echo "Merged model: ${APERTUS_MERGED_MODEL}"
fi

if [[ "${ENABLE_UI_REVERSE_TUNNEL}" == "1" ]]; then
	if ! command -v ssh >/dev/null 2>&1; then
		echo "ssh is not available for reverse tunnel setup." >&2
		exit 1
	fi
	if [[ -z "${UI_REVERSE_TUNNEL_HOST}" ]]; then
		echo "UI_REVERSE_TUNNEL_HOST is required when ENABLE_UI_REVERSE_TUNNEL=1." >&2
		exit 1
	fi
	echo "Opening reverse tunnel on ${UI_REVERSE_TUNNEL_HOST}:${UI_REVERSE_TUNNEL_PORT} -> $(hostname -s):${APERTUS_UI_PORT}"
	ssh -nNT \
		-o ExitOnForwardFailure=yes \
		-o ServerAliveInterval=30 \
		-o ServerAliveCountMax=3 \
		-o StrictHostKeyChecking=accept-new \
		-R "127.0.0.1:${UI_REVERSE_TUNNEL_PORT}:127.0.0.1:${APERTUS_UI_PORT}" \
		"${USER}@${UI_REVERSE_TUNNEL_HOST}" &
	tunnel_pid="$!"
fi

uenv run "${UENV_IMAGE}" --view="${UENV_VIEW}" -- bash -lc '
set -euo pipefail
source "${VENV_DIR}/bin/activate"

if ! python -c "import gradio" >/dev/null 2>&1; then
	python -m pip install -r "${REPO_ROOT}/UI/requirements.txt"
fi

cd "${REPO_ROOT}"
python "${REPO_ROOT}/UI/app.py" --host "${APERTUS_UI_HOST}" --port "${APERTUS_UI_PORT}"
'