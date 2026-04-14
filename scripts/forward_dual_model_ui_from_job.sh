#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

job_id="${1:-${UI_JOB_ID:-}}"
job_name="${APERTUS_UI_JOB_NAME:-apertus-dual-ui}"
local_port="${LOCAL_UI_PORT:-8631}"
remote_port="${REMOTE_UI_PORT:-8631}"

list_listener_pids() {
	if command -v lsof >/dev/null 2>&1; then
		lsof -tiTCP:"${local_port}" -sTCP:LISTEN 2>/dev/null | sort -u
		return 0
	fi

	if command -v fuser >/dev/null 2>&1; then
		fuser -n tcp "${local_port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u
		return 0
	fi

	echo "Need lsof or fuser to inspect local port ${local_port}." >&2
	return 1
}

reclaim_stale_local_proxy() {
	local pid owner args
	local -a listener_pids=()

	mapfile -t listener_pids < <(list_listener_pids || true)
	if [[ ${#listener_pids[@]} -eq 0 ]]; then
		return 0
	fi

	for pid in "${listener_pids[@]}"; do
		owner="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)"
		args="$(ps -o args= -p "${pid}" 2>/dev/null || true)"

		if [[ "${owner}" == "${USER}" && "${args}" == *"${repo_root}/scripts/tcp_port_proxy.py"* && "${args}" == *"--listen-port ${local_port}"* ]]; then
			echo "Stopping stale UI proxy on port ${local_port} (pid ${pid})." >&2
			kill "${pid}" 2>/dev/null || true
			if ps -p "${pid}" >/dev/null 2>&1; then
				kill -9 "${pid}" 2>/dev/null || true
			fi
			continue
		fi

		echo "Local port ${local_port} is already in use by a different process (pid ${pid})." >&2
		if [[ -n "${args}" ]]; then
			echo "Process: ${args}" >&2
		fi
		echo "Stop it manually or set LOCAL_UI_PORT to a different port." >&2
		exit 1
	done

	mapfile -t listener_pids < <(list_listener_pids || true)
	if [[ ${#listener_pids[@]} -gt 0 ]]; then
		echo "Local port ${local_port} is still busy after proxy cleanup." >&2
		exit 1
	fi
}

if [[ -z "${job_id}" ]]; then
	job_id="$(squeue -h -u "${USER}" -n "${job_name}" -t RUNNING -o '%A' | head -n 1)"
fi

if [[ -z "${job_id}" ]]; then
	echo "No running UI job found. Pass a job id explicitly or set UI_JOB_ID." >&2
	exit 1
fi

job_state="$(squeue -h -j "${job_id}" -o '%T')"
job_node="$(squeue -h -j "${job_id}" -o '%N')"

if [[ -z "${job_state}" || -z "${job_node}" ]]; then
	echo "Could not resolve state/node for job ${job_id}." >&2
	exit 1
fi

if [[ "${job_state}" != "RUNNING" ]]; then
	echo "Job ${job_id} is not RUNNING (state=${job_state})." >&2
	exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 is required to run the local port proxy." >&2
	exit 1
fi

reclaim_stale_local_proxy

echo "Forwarding login-node port ${local_port} to ${job_node}:${remote_port} for job ${job_id}" >&2
exec python3 "${repo_root}/scripts/tcp_port_proxy.py" \
	--listen-host 127.0.0.1 \
	--listen-port "${local_port}" \
	--target-host "${job_node}" \
	--target-port "${remote_port}"