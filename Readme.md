fine tuning for Apertus on ALPS Clariden using question answer pairs prepared from the GSDG dataset.

For the first test use:

- `../GSDG/outputs/synthetic_glossAPI_Sxolika_vivlia_Qwen_Qwen3.5-397B-A17B-FP8_manual_job1700178.jsonl`

This repository now includes a minimal Clariden starter scaffold:

- `scripts/setup_uenv_python.sh` bootstraps a `uenv` Python environment and installs the Apertus recipe dependencies
- `scripts/stage_glossapi_dataset.sh` stages the JSONL into `${SCRATCH}/glossapi-trainer/data`
- `scripts/sft_train_glossapi.py` mirrors the upstream Apertus training entrypoint but adds local JSONL loading
- `scripts/run_apertus_lora_clariden.sbatch` runs the first single-process LoRA smoke test on a full Clariden node
- `scripts/run_apertus_full_clariden.sbatch` runs single-node 4-GPU full fine-tuning with `accelerate`
- `configs/` contains repository-specific training configs and the single-node ZeRO-3 config

Quick start:

```bash
bash scripts/setup_uenv_python.sh
bash scripts/stage_glossapi_dataset.sh
sbatch scripts/run_apertus_lora_clariden.sbatch
```

The setup script now defaults to the validated Clariden install path for PyTorch:

- `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128`

You can still override that by setting `TORCH_INSTALL_COMMAND` before running setup.

Reference docs:

- Apertus fine-tuning: `https://apertvs.ai/docs/tech/fine-tuning/`
- Clariden: `https://docs.cscs.ch/clusters/clariden/`
- uenv: `https://docs.cscs.ch/software/uenv/`

