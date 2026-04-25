# JUSUF HPC Configuration Guide

## Overview
JUSUF (Jülich Supercomputing Facility) is a high-performance computing system at Forschungszentrum Jülich. This guide covers data preparation, environment setup, and running training/inference jobs on JUSUF.

## 1. Data Preparation

The pipeline expects an `aligned_v2` dataset with a `metadata.json` and per-case directories.

**Required layout on the cluster:**
```
<project_root>/
  data/
    aligned_v2/
      metadata.json
      <case_id>/
        t2/          # T2-weighted slices (0000.png, 0001.png, ...)
        adc/         # ADC maps (optional per case)
        calc/        # Calculated DWI (optional per case)
        mask_prostate/
        mask_target1/
    splits/
      <YYYY-MM-DD>.yaml   # frozen paper split (train/val/test case IDs)
```

**How to get data onto JUSUF:**
1. Transfer the `aligned_v2/` directory to the cluster (e.g. via `rsync`).
2. Place or symlink it at `data/aligned_v2/` relative to the repo root.
3. Ensure `data/splits/<YYYY-MM-DD>.yaml` (or your dated split file) is committed in the repo.

**Generate a new split (if needed):**
```bash
python tools/generate_splits.py \
  --metadata data/aligned_v2/metadata.json \
  --output data/splits/YYYY-MM-DD.yaml
```

## 2. Environment Setup (`.env`)

Create a `.env` file at the repo root. The HPC scripts (`scripts/new/*`) source it automatically.

```bash
# .env  (do NOT commit this file)

# Singularity container (set to run inside container on HPC)
MRI_TRAIN_CONTAINER_IMAGE=/p/scratch/ebrains-0000006/<user>/singularity_images/mri-train3.sif
# SIF_EXTRA_BINDS=/extra/path:/extra/path   # additional bind mounts if needed

# Required for research-smoke and import_tcia_aligned.py
SOURCE_DATA=/p/project/ebrains-0000006/<user>/tcia-handler/data/aligned_v2

# Optional overrides (defaults shown)
PYTHON_BIN=python
DATA_DIR=$PWD/data
CHECKPOINT_DIR=$PWD/checkpoints
PREDICTIONS_DIR=$PWD/predictions
WANDB_DIR=$PWD/wandb
WANDB_MODE=offline
WANDB_API_KEY=<your-key>       # only needed if WANDB_MODE=online
```

**Singularity container (recommended):**

The scripts run inside a Singularity container when `MRI_TRAIN_CONTAINER_IMAGE` is set. Add it to `.env`:

```bash
MRI_TRAIN_CONTAINER_IMAGE=/p/scratch/ebrains-0000006/<user>/singularity_images/mri-train3.sif
```

The image contents come from the repo [Dockerfile](/p/scratch/ebrains-0000006/kim27/cancer_detector/Dockerfile:1), which installs [requirements.txt](/p/scratch/ebrains-0000006/kim27/cancer_detector/requirements.txt:1). Rebuild it after dependency changes such as adding `einops`.

One reliable path is:

1. Build a Docker image from the repo on a machine where Docker is available:
```bash
docker build -t cancer-detector:mri-train3 .
docker save cancer-detector:mri-train3 -o /tmp/cancer-detector-mri-train3.tar
```

2. Convert that archive into the cluster `.sif` image:
```bash
apptainer build /p/scratch/ebrains-0000006/<user>/singularity_images/mri-train3.sif \
  docker-archive:///tmp/cancer-detector-mri-train3.tar
```

If your site requires privileged image builds, use `apptainer build --fakeroot ...` or build the `.sif` wherever your Apptainer/Singularity setup allows it.

Quick verification:
```bash
apptainer exec /p/scratch/ebrains-0000006/<user>/singularity_images/mri-train3.sif \
  python -c "import monai, einops; print(monai.__version__)"
```

**Native Python (alternative):**
```bash
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## 3. Training

### Segmentation

```bash
# Interactive (bash)
bash scripts/new/train --config mri/config/task/segmentation.yaml

# SLURM job
sbatch scripts/new/train --config mri/config/task/segmentation.yaml

# With overrides
sbatch scripts/new/train --config mri/config/task/segmentation.yaml \
  --epochs 200 --lr 1e-4 --run_name seg-exp1
```

Default: SegResNet, 100 epochs, lr 5e-5, batch 4, Dice+BCE loss.
Checkpoints go to `checkpoints/seg/<run_name>/`.

### Classification

Classification requires segmentation predictions to exist first (see Inference below).

```bash
# Interactive
bash scripts/new/train --config mri/config/task/classification.yaml

# SLURM job
sbatch scripts/new/train --config mri/config/task/classification.yaml \
  --run_name cls-exp1
```

Default: Swin, 50 epochs, lr 1e-4, batch 2, cross-entropy loss.
Checkpoints go to `checkpoints/cls/<run_name>/`.

The classification config points to `data.seg_pred_dir: data/seg_preds` by default. Update this to wherever your segmentation predictions live (see Section 4).

## 4. Inference

### Segmentation inference (produces predictions for classification)

```bash
# Run on all three splits to generate predictions for downstream classification
sbatch scripts/new/inference \
  --config mri/config/task/segmentation.yaml \
  --split train \
  --checkpoint checkpoints/seg/<run_name>/<run_name>_best.pt \
  --output_dir data/seg_preds

sbatch scripts/new/inference \
  --config mri/config/task/segmentation.yaml \
  --split val \
  --checkpoint checkpoints/seg/<run_name>/<run_name>_best.pt \
  --output_dir data/seg_preds

sbatch scripts/new/inference \
  --config mri/config/task/segmentation.yaml \
  --split test \
  --checkpoint checkpoints/seg/<run_name>/<run_name>_best.pt \
  --output_dir data/seg_preds
```

This writes per-case prediction files under `data/seg_preds/<case_id>/`.

### Classification inference

```bash
sbatch scripts/new/inference \
  --config mri/config/task/classification.yaml \
  --split test \
  --checkpoint checkpoints/cls/<run_name>/<run_name>_best.pt
```

Results go to `predictions/` by default (or `--output_dir` override).

## 5. Finding Checkpoints

The trainer saves two checkpoints per run inside `<output_dir>/<run_name>/`:

| File | Description |
|------|-------------|
| `<run_name>_best.pt` | Best validation metric (Dice for seg, macro-F1 for cls) |
| `<run_name>_last.pt` | End of final epoch |

**Always use `_best.pt` for inference and paper results.**

Each run also writes these files alongside checkpoints:

| File | Description |
|------|-------------|
| `resolved_config.yaml` | Exact config used (all layers merged) |
| `run_manifest.json` | Full metadata: git commit, SLURM job ID, W&B URL, best metric, paths |
| `metrics_history.csv` | Per-epoch train and val metrics |
| `run_summary.json` | Final summary with best metric and epoch |

**To find the best segmentation run:**
```bash
# List all completed seg run manifests sorted by Dice
find checkpoints/seg -name run_manifest.json \
  -exec grep -l '"status": "completed"' {} \; \
  | xargs -I{} sh -c 'echo "$(python -c "import json; m=json.load(open(\"{}\")); print(m.get(\"summary\",{}).get(\"best_metric\",0), m[\"run_name\"])")"' \
  | sort -rn
```

Or inspect a single run:
```bash
python -c "
import json, sys
m = json.load(open(sys.argv[1]))
print(f\"Run:        {m['run_name']}\")
print(f\"Status:     {m['status']}\")
print(f\"Best metric:{m.get('summary',{}).get('best_metric')}\")
print(f\"Best epoch: {m.get('summary',{}).get('best_epoch')}\")
print(f\"Checkpoint: {m.get('artifacts',{}).get('best_checkpoint')}\")
" checkpoints/seg/<run_name>/run_manifest.json
```

## SLURM Reference

### GPU Partition

**Critical**: GPU jobs must use the `gpus` partition, NOT `gpu`.

```bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --account=ebrains-0000006
```

Using the wrong partition causes `NVIDIA-SMI has failed` errors even with `--gres=gpu:1`.

Find your available accounts: `jutil user projects`

### GPU Node Specs

- **GPU**: NVIDIA V100 PCIe (16 GB)
- **GPUs per node**: 1
- **Partition**: `gpus`

### Job Monitoring

```bash
squeue -u $USER              # check job status
cat slurm-<JOBID>.out        # view stdout
cat slurm-<JOBID>.err        # view stderr
scancel <JOBID>              # cancel job
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NVIDIA-SMI has failed` | Wrong partition (`gpu` instead of `gpus`) | `--partition=gpus` |
| `please specify the job's account` | Missing account | `--account=ebrains-0000006` |
| `No such file or directory` in SLURM | Relative path resolution | Use absolute paths or `SLURM_SUBMIT_DIR` |
