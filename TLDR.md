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
jobid=$(MODEL_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
sbatch --parsable scripts/run_merged_inference_clariden.sbatch) && \
echo "JOBID=$jobid" && \
while squeue -h -j "$jobid" | grep -q .; do
	squeue -h -j "$jobid" -o 'STATE=%T ELAPSED=%M REASON=%R'
	sleep 20
done && \
sacct -j "$jobid" --format=JobID,JobName%30,State,ExitCode,Elapsed%20 -P
```
