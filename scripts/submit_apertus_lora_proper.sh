#!/bin/bash

# This script submits a training job for the Apertus LoRA adapter on Clariden.
# It sets up the necessary environment variables and then calls sbatch to run the training script.
# The training script will handle the actual training and merging of the LoRA adapter into the base
# model. After training, the merged model can be tested using the smoke test script.


set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
	echo "SCRATCH is not set." >&2
	exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DATASET_PATH="${DATASET_PATH:-${SCRATCH}/glossapi-trainer/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/glossapi-trainer/output/apertus_lora_proper}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-${repo_root}/configs/sft_lora_glossapi_proper.yaml}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
export TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-}"
export TRAIN_GRADIENT_CHECKPOINTING="${TRAIN_GRADIENT_CHECKPOINTING:-false}"
export MERGE_AFTER_TRAIN="${MERGE_AFTER_TRAIN:-1}"
export MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${OUTPUT_DIR}_merged}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec sbatch --job-name="apertus-lora-proper" scripts/run_apertus_lora_clariden.sbatch