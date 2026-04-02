# TLDR: Proper LoRA Training Commands (Clariden)

## 1) Stage dataset with validation split

```bash
VALIDATION_ROWS=24 bash scripts/stage_glossapi_dataset.sh
```

## 2) Submit proper training run

```bash
bash scripts/submit_apertus_lora_proper.sh
```

## 3) Submit proper run with explicit output directories

```bash
OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2 \
MERGED_OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2_merged \
bash scripts/submit_apertus_lora_proper.sh
```

## 4) Submit and actively track job status

```bash
jobid=$(OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2 \
MERGED_OUTPUT_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2_merged \
bash scripts/submit_apertus_lora_proper.sh | awk '{print $4}') && \
echo "JOBID=$jobid" && \
while squeue -h -j "$jobid" | grep -q .; do
	squeue -h -j "$jobid" -o 'STATE=%T ELAPSED=%M REASON=%R'
	sleep 30
done && \
sacct -j "$jobid" --format=JobID,JobName%30,State,ExitCode,Elapsed%20 -P
```

## 5) Verify output directories

```bash
ls ${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2
ls ${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2_merged
```

## 6) Run inference smoke test on the merged model

```bash
jobid=$(MODEL_DIR=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_v2_merged \
sbatch --parsable scripts/run_merged_inference_clariden.sbatch) && \
echo "JOBID=$jobid" && \
while squeue -h -j "$jobid" | grep -q .; do
	squeue -h -j "$jobid" -o 'STATE=%T ELAPSED=%M REASON=%R'
	sleep 20
done && \
sacct -j "$jobid" --format=JobID,JobName%30,State,ExitCode,Elapsed%20 -P
```
