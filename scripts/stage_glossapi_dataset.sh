#!/bin/bash
# this script stages the dataset for the GlossAPI trainer 
# by copying and optionally splitting a source dataset 
# into train and validation sets in the destination directory.

set -euo pipefail

source_dataset="${SOURCE_DATASET:-/users/p-skarvelis/GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl}"

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH is not set." >&2
	exit 1
fi

dest_dir="${DEST_DIR:-${SCRATCH}/glossapi-trainer/data}"
validation_rows="${VALIDATION_ROWS:-0}"

if [[ ! -f "${source_dataset}" ]]; then
	echo "Dataset file not found: ${source_dataset}" >&2
	exit 1
fi

if [[ ! "${validation_rows}" =~ ^[0-9]+$ ]]; then
	echo "VALIDATION_ROWS must be a non-negative integer." >&2
	exit 1
fi

mkdir -p "${dest_dir}"

if [[ "${validation_rows}" == "0" ]]; then
	cp "${source_dataset}" "${dest_dir}/train.jsonl"
	rm -f "${dest_dir}/validation.jsonl"
	printf 'Staged train file at %s\n' "${dest_dir}/train.jsonl"
	exit 0
fi

total_rows="$(wc -l < "${source_dataset}")"
if (( validation_rows >= total_rows )); then
	echo "VALIDATION_ROWS=${validation_rows} is too large for a dataset with ${total_rows} rows." >&2
	exit 1
fi

train_rows=$((total_rows - validation_rows))

head -n "${train_rows}" "${source_dataset}" > "${dest_dir}/train.jsonl"
tail -n "${validation_rows}" "${source_dataset}" > "${dest_dir}/validation.jsonl"

printf 'Staged train file at %s\n' "${dest_dir}/train.jsonl"
printf 'Staged validation file at %s\n' "${dest_dir}/validation.jsonl"
