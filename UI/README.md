# Apertus Dual Console

Το UI αυτό φορτώνει το base `Apertus 8B` και το local merged model και σου δίνει:

- κοινό prompt που στέλνεται και στα δύο μοντέλα παράλληλα
- δύο ανεξάρτητες λωρίδες με ξεχωριστό prompt για single-model checks
- serve σε `0.0.0.0:8631` ώστε να κάνεις port forward από VS Code

## Quick run

```bash
bash scripts/run_dual_model_ui.sh
```

## Run on a Clariden compute node

Batch job:

```bash
bash scripts/submit_dual_model_ui.sh
```

Το helper κάνει by default submit με `normal`, `2 GPUs`, `36 CPUs`, `128G`.

Με explicit merged path:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
bash scripts/submit_dual_model_ui.sh
```

Interactive `srun`:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
bash scripts/srun_dual_model_ui_clariden.sh
```

Αν θες να δοκιμάσεις reverse tunnel προς το login node για να βλέπει πιο εύκολα το port forwarding του VS Code την πόρτα, μπορείς να βάλεις:

```bash
ENABLE_UI_REVERSE_TUNNEL=1 \
UI_REVERSE_TUNNEL_HOST=${SLURM_SUBMIT_HOST:-$(hostname -s)} \
bash scripts/submit_dual_model_ui.sh
```

Αυτό προϋποθέτει ότι από compute node μπορείς να κάνεις non-interactive `ssh` πίσω στο submit host.

## Αν το reverse tunnel δεν δουλεύει

Αν το job τρέχει αλλά βλέπεις `Permission denied (publickey)` για το reverse tunnel, μπορείς να προωθήσεις το port από το login node προς το compute node χωρίς ssh back-tunnel:

```bash
bash scripts/forward_dual_model_ui_from_job.sh JOB_ID
```

ή αυτόματα για το πρώτο running `apertus-dual-ui` job:

```bash
bash scripts/forward_dual_model_ui_from_job.sh
```

Μετά κάνεις VS Code port forward στο local `8631` του login node.

Αν θέλεις να ορίσεις ρητά merged path και GPU split:

```bash
APERTUS_MERGED_MODEL=${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged \
APERTUS_BASE_DEVICE=cuda:0 \
APERTUS_MERGED_DEVICE=cuda:1 \
APERTUS_BASE_DTYPE=float32 \
APERTUS_MERGED_DTYPE=bfloat16 \
bash scripts/run_dual_model_ui.sh
```

## Environment overrides

- `APERTUS_BASE_MODEL`: default `swiss-ai/Apertus-8B-Instruct-2509`
- `APERTUS_MERGED_MODEL`: local merged model directory
- `APERTUS_BASE_DEVICE`: π.χ. `cuda:0`
- `APERTUS_MERGED_DEVICE`: π.χ. `cuda:1`
- `APERTUS_UI_PORT`: default `8631`
- `APERTUS_BASE_DTYPE`: default `float32`
- `APERTUS_MERGED_DTYPE`: default `bfloat16`
- `APERTUS_UI_DTYPE`: legacy override που εφαρμόζει το ίδιο dtype και στα δύο models
- `APERTUS_UI_ATTN_IMPLEMENTATION`: αν χρειαστεί π.χ. `eager`

Το app ψάχνει αυτόματα πρώτα για:

1. `${SCRATCH}/glossapi-trainer/output/apertus_lora_proper_merged`
2. `${SCRATCH}/glossapi-trainer/output/apertus_lora_short100_merged`
3. `${SCRATCH}/glossapi-trainer/output/apertus_lora_smoke_merged`

Αν το environment έχει τουλάχιστον 2 GPUs, τα defaults είναι `cuda:0` για το base model και `cuda:1` για το merged model.