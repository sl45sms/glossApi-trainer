# TLDR: Proper LoRA Training Commands (Clariden)

## 1) Stage dataset with validation split

```bash
SOURCE_DATASET=/users/p-skarvelis/GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl VALIDATION_ROWS=24 bash scripts/stage_glossapi_dataset.sh
```

## 2) Submit proper training run

```bash
bash scripts/submit_apertus_lora_proper.sh
```
you can optionally specify output directories with environment variables:

```bash
OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper 
MERGED_OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged 
```

## 3) Verify output directories

```bash
ls ${SCRATCH}/glossapi-trainer/output/apertus_lora_proper
ls ${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged
```

## 4) Run inference smoke test on the merged model

```bash
sbatch --parsable scripts/run_merged_inference_clariden.sbatch
```

## 5) Launch side-by-side UI on port 8631
```bash
JOB_ID=$(ENABLE_UI_REVERSE_TUNNEL=1 \
APERTUS_BASE_DTYPE=float32 \
APERTUS_MERGED_DTYPE=bfloat16 \
bash scripts/submit_dual_model_ui.sh --parsable) && \
echo "JOBID=$JOB_ID"
```

if reverse tunnel fails but the job is running, forward the port from the login node to the compute node:
```bash
bash scripts/forward_dual_model_ui_from_job.sh JOB_ID
```
script closes old local proxy on `8631` if it finds one from a previous run.

## 6) Run the UI on a Clariden node with srun

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
bash scripts/srun_dual_model_ui_clariden.sh
```
