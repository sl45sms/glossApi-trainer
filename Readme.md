fine tuning for Apertus on ALPS Clariden using question answer pairs prepared from the GSDG dataset.

For the first test use:

- `../GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl`

This repository now includes a minimal Clariden starter scaffold:

- `scripts/setup_uenv_python.sh` bootstraps a `uenv` Python environment and installs the Apertus recipe dependencies
- `scripts/stage_glossapi_dataset.sh` stages the JSONL into `${SCRATCH}/glossapi-trainer/data`
- `scripts/sft_train_glossapi.py` mirrors the upstream Apertus training entrypoint but adds local JSONL loading
- `scripts/merge_lora_into_base.py` merges a LoRA adapter into a standalone merged model directory
- `scripts/smoke_test_merged_model.py` runs one prompt against a merged model and prints the response as JSON
- `scripts/run_apertus_lora_clariden.sbatch` runs the first single-process LoRA smoke test on a full Clariden node
- `scripts/submit_apertus_lora_proper.sh` submits a validation-enabled non-smoke LoRA run
- `scripts/run_merged_inference_clariden.sbatch` runs one merged-model inference prompt on Clariden
- `scripts/run_dual_model_ui.sh` launches a side-by-side UI on port `8631` for the base 8B model and a merged local model
- `scripts/run_dual_model_ui_clariden.sbatch` submits the dual-model UI as a long-running Clariden job
- `scripts/srun_dual_model_ui_clariden.sh` requests an interactive Clariden allocation and starts the same UI there
- `scripts/submit_dual_model_ui.sh` provides a one-command batch submission path for the dual-model UI
- `scripts/forward_dual_model_ui_from_job.sh` forwards login-node port `8631` to the compute node of a running UI job without requiring reverse SSH from the node
- `scripts/run_apertus_full_clariden.sbatch` runs single-node 4-GPU full fine-tuning with `accelerate`
- `configs/` contains repository-specific training configs and the single-node ZeRO-3 config

Quick start:

```bash
bash scripts/setup_uenv_python.sh
bash scripts/stage_glossapi_dataset.sh
sbatch scripts/run_apertus_lora_clariden.sbatch
```

The LoRA job now merges the adapter into a standalone model on the same Clariden node after training finishes.

Useful overrides:

```bash
# disable merge if you only want adapter output
MERGE_AFTER_TRAIN=0 sbatch scripts/run_apertus_lora_clariden.sbatch

# choose a custom merged-model output path
MERGED_OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke_merged sbatch scripts/run_apertus_lora_clariden.sbatch
```

Merged-model inference smoke test:

```bash
sbatch scripts/run_merged_inference_clariden.sbatch
```

Dual-model UI on port `8631`:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
APERTUS_BASE_DEVICE=cuda:0 \
APERTUS_MERGED_DEVICE=cuda:1 \
APERTUS_BASE_DTYPE=float32 \
APERTUS_MERGED_DTYPE=bfloat16 \
bash scripts/run_dual_model_ui.sh
```

Dual-model UI on a Clariden compute node:

```bash
bash scripts/submit_dual_model_ui.sh
```

with explicit merged model path:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
bash scripts/submit_dual_model_ui.sh
```

If reverse SSH from the compute node is unavailable, forward the UI port from the login node once the job is running:

```bash
bash scripts/forward_dual_model_ui_from_job.sh JOB_ID
```

or interactively:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
bash scripts/srun_dual_model_ui_clariden.sh
```

Proper LoRA training run (with validation and checkpoint selection):

```bash
# create train/validation split first
VALIDATION_ROWS=24 bash scripts/stage_glossapi_dataset.sh

# submit a non-smoke run (defaults to configs/sft_lora_glossapi_proper.yaml)
bash scripts/submit_apertus_lora_proper.sh
```

The setup script now defaults to the validated Clariden install path for PyTorch:

- `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128`

You can still override that by setting `TORCH_INSTALL_COMMAND` before running setup.

Reference docs:

- Apertus fine-tuning: `https://apertvs.ai/docs/tech/fine-tuning/`
- Clariden: `https://docs.cscs.ch/clusters/clariden/`
- uenv: `https://docs.cscs.ch/software/uenv/`

