# Agents / Runbook (Apertus Fine-Tuning on CSCS Alps Clariden)

This repository is intended to fine-tune Apertus on CSCS Alps, targeting Clariden, using question-answer pairs prepared from the GSDG dataset.

The first test dataset is the existing JSONL file:

- `/users/p-skarvelis/GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl`

The practical goal is to run Apertus supervised fine-tuning recipes on Clariden with the minimum number of moving parts:

- use `uenv` for environment bootstrap and CPU-side preparation
- use Slurm for scheduling on Clariden
- keep caches and outputs on `${SCRATCH}`
- start with a single-process LoRA smoke test on Apertus 8B
- move to 4-GPU full fine-tuning only after the data path is validated

This runbook adapts the upstream Apertus fine-tuning instructions for Clariden instead of following them literally.

## 1. Platform assumptions

- Cluster: CSCS Alps, Clariden
- Scheduler: Slurm
- Architecture: `aarch64`
- Preferred control-plane tooling: `uenv`
- Training codebase: `https://github.com/swiss-ai/apertus-finetuning-recipes`
- Base models:
	- `swiss-ai/Apertus-8B-Instruct-2509` for the first LoRA run
	- `swiss-ai/Apertus-70B-Instruct-2509` only after the pipeline is stable

Important Clariden constraints:

- Clariden is `aarch64`, so do not assume upstream `x86_64` wheel instructions apply unchanged.
- The upstream quickstart uses a CUDA wheel index meant as a generic recipe example. Treat it as an entrypoint reference, not as a guaranteed Clariden install recipe.
- Prefer a validated Clariden Python stack through `uenv`, and layer only the missing Python packages on top.
- Keep Hugging Face and dataset caches on `${SCRATCH}`.

Recommended operating split:

- Use `uenv` for preparation, validation, config editing, and lightweight preprocessing.
- Run actual training through Slurm on Clariden compute nodes.
- Do not write large caches or model outputs into `${HOME}`.

### Debug partition

For the first smoke test, prefer the debug partition if the expected runtime fits within the cluster limit. It is the fastest way to validate the data loader, tokenizer path, and trainer startup before using a longer allocation.

## 2. What is already verified about the dataset

The first JSONL file has already been checked.

Each row is a conversational training sample with:

- a `messages` array
- a `user` turn containing the source text and instruction
- an `assistant` turn containing the expected JSON answer
- a `meta` object with dataset metadata

That means the data is already in a chat-style SFT shape and should be treated as the canonical first test input.

Important practical note:

- the file appears to contain a single training stream, not a separate validation split
- for the first smoke test, disable evaluation or create a small held-out validation JSONL explicitly

## 3. High-level workflow

The intended workflow for this repository is:

1. Bootstrap a Clariden-compatible Python environment with `uenv`.
2. Clone the upstream Apertus fine-tuning recipes.
3. Stage the GSDG JSONL on `${SCRATCH}`.
4. Add the minimum local adaptation needed so the recipe can load a local JSONL file on Clariden.
5. Run a single-GPU LoRA smoke test with Apertus 8B.
6. Only after that succeeds, consider multi-GPU or full-parameter training.

## 4. Use the upstream Apertus recipes, but adapt the data path

Upstream documentation and code currently point to:

- `python sft_train.py --config configs/sft_lora.yaml`
- `accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full.yaml`

The important limitation is that the upstream `sft_train.py` loads data with:

```python
load_dataset(script_args.dataset_name, name=script_args.dataset_config)
```

That means a plain local JSONL file is not wired in directly by the documented examples.

For this repository, the correct approach is:

1. Keep using the upstream recipe codebase.
2. Add a minimal local adaptation so the trainer can load a JSONL file from disk.
3. Do not invent a separate training stack unless the upstream recipe proves unworkable.

Preferred minimal adaptation:

- allow the recipe to accept local `data_files`
- point it at the GSDG JSONL on `${SCRATCH}`
- disable eval for the first test unless a validation file is provided

In other words, the first engineering task for this workflow is usually a small patch or wrapper around the upstream recipe, not a rewrite.

## 5. uenv bootstrap pattern

Use `uenv` for the preparation layer.

Example bootstrap pattern:

```bash
uenv image pull prgenv-gnu/24.11:v1
uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc 'python3 --version; which python3'
```

Create a local environment for preparation work:

```bash
uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc '
python3 -m venv .venv-uenv
source .venv-uenv/bin/activate
python -m pip install --upgrade pip uv
git clone https://github.com/swiss-ai/apertus-finetuning-recipes.git
cd apertus-finetuning-recipes
uv pip install -r requirements.txt
'
```

Use this layer for:

- cloning and preparing the upstream recipe repo
- editing configs
- checking imports
- validating the JSONL structure
- staging data under `${SCRATCH}`

Do not assume the upstream `torch` installation command is the right one for Clariden. On `aarch64`, reuse a validated Clariden-compatible PyTorch stack if one already exists in your environment.

## 6. Data staging on Clariden

Stage the first test dataset onto `${SCRATCH}` so training jobs do not read it from a home-directory path repeatedly.

Example:

```bash
mkdir -p ${SCRATCH}/glossapi-trainer/data
cp /users/p-skarvelis/GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl \
	${SCRATCH}/glossapi-trainer/data/train.jsonl
```

If you want evaluation in the first round, create a tiny held-out file such as:

- `${SCRATCH}/glossapi-trainer/data/train.jsonl`
- `${SCRATCH}/glossapi-trainer/data/validation.jsonl`

If you do not create a validation split, configure the first run with no evaluation.

## 7. Recommended first training target

Start with LoRA on `swiss-ai/Apertus-8B-Instruct-2509`.

On Clariden, this recommendation is about execution topology, not about asking Slurm for a smaller allocation. If the scheduler gives you a full 4-GPU node by default, you can still run the first smoke test as a 1-GPU trainer process on that node and leave the other GPUs unused.

Why:

- the upstream Apertus docs explicitly position LoRA as the lightweight starting path
- it isolates data-loading, tokenization, and config issues from distributed-training failures
- it avoids bringing `accelerate`, ZeRO-3, multi-process launch, and NCCL behavior into the first debug step
- it gives you a cleaner failure mode if the only remaining uncertainty is the local JSONL integration

Treat full fine-tuning and 70B as second-phase work.

## 8. Suggested config direction

Use the upstream `configs/sft_lora.yaml` as the base and create a repository-specific training config for the GSDG data.

The repository-specific config should reflect at least these choices:

- `model_name_or_path: swiss-ai/Apertus-8B-Instruct-2509`
- `output_dir` on `${SCRATCH}`
- `dtype: bfloat16`
- `use_peft: true`
- `gradient_checkpointing: true`
- `eval_strategy: no` for the first single-file smoke test

After the local JSONL loader is wired in, the dataset target should resolve to the staged file on `${SCRATCH}` rather than a hosted Hugging Face dataset.

## 9. Example Slurm launch shape for the first smoke test

Use a single trainer process first, even if the node allocation contains 4 GPUs.

Example shape:

```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=apertus-glossapi-lora
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export HF_HOME=${SCRATCH}/hf
export HF_DATASETS_CACHE=${SCRATCH}/hf_datasets
export CUDA_VISIBLE_DEVICES=0

uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc '
source /path/to/.venv-uenv/bin/activate
cd /path/to/apertus-finetuning-recipes
python sft_train.py --config configs/sft_lora_glossapi.yaml
'
```

This is the intended first checkpoint:

- the job starts cleanly on Clariden
- the local JSONL is loaded successfully
- tokenizer and trainer initialize
- adapter checkpoints are written to `${SCRATCH}`

## 10. Full fine-tuning path after the smoke test

Only after the single-GPU LoRA path works should you attempt a 4-GPU run such as:

```bash
accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full_glossapi.yaml
```

Use that path for:

- full-parameter fine-tuning
- larger experiments
- possible 70B runs if the environment has already been validated on Clariden

Do not start there.

## 11. Output expectations

According to the Apertus recipes:

- LoRA runs save adapters under a directory like `Apertus-FT/output/apertus_lora/`
- full fine-tuning saves a full model directory under a path like `Apertus-FT/output/apertus_full/`

For Clariden, prefer repository-specific output roots on `${SCRATCH}`, for example:

- `${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke`
- `${SCRATCH}/glossapi-trainer/output/apertus_full`

## 12. Working rules for this repository

When updating this repository or adjacent training scripts, follow these rules:

- Prefer minimal, explicit changes over introducing a parallel custom trainer.
- Keep the Apertus upstream recipe structure recognizable.
- Preserve the verified GSDG conversational JSONL shape as the first training input.
- Optimize first for successful startup and data loading on Clariden.
- Avoid x86-specific installation assumptions.
- Keep caches, checkpoints, and temporary artifacts on `${SCRATCH}`.
- If validation data does not exist yet, disable eval instead of fabricating a split implicitly.

## 13. References

- Apertus fine-tuning docs: `https://apertvs.ai/docs/tech/fine-tuning/`
- Apertus recipes: `https://github.com/swiss-ai/apertus-finetuning-recipes`
- Clariden docs: `https://docs.cscs.ch/clusters/clariden/`
- uenv docs: `https://docs.cscs.ch/software/uenv/`
