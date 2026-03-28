# Pipeline Steps

This project is currently a two-stage pipeline:

1. train a segmentation model
2. generate segmentation predictions for all required splits
3. train a classification model using those segmentation predictions
4. run classification inference

For now, we are training segmentation first. Classification is a separate next step and cannot be trained correctly until segmentation inference has been run on the needed cases.

## 1. How To Train Segmentation

Recommended HPC path: submit the wrapper through SLURM and let it hand off to Singularity automatically.

From anywhere on the cluster:

```bash
export PROJECT_DIR=/p/scratch/ebrains-0000006/kim27/cancer_detector

sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always,WANDB_MODE=offline \
  "${PROJECT_DIR}/scripts/new/train" \
  --config mri/config/task/segmentation.yaml \
  --output-dir "${PROJECT_DIR}/checkpoints/seg" \
  --run-name segresnet-$(date +%Y-%m-%d-%H-%M-%S)
```

Notes:

- `MRI_TRAIN_USE_CONTAINER=always` forces the Singularity/Apptainer path.
- `mri/config/task/segmentation.yaml` is the full segmentation training config.
- `--run-name` makes the output directory easy to find later.
- `WANDB_MODE=offline` keeps tracking local on the cluster.

If you only want to validate the launcher without starting training:

```bash
sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always \
  "${PROJECT_DIR}/scripts/new/train" \
  --config mri/config/task/segmentation.yaml \
  --dry-run
```

## 2. How To Monitor Training

Check the SLURM job:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

Check the SLURM logs:

```bash
tail -f "${PROJECT_DIR}/checkpoints/slurm-train-<job_id>.out"
tail -f "${PROJECT_DIR}/checkpoints/slurm-train-<job_id>.err"
```

What you want to see in `stdout`:

- the `MRI Training (Singularity HPC)` block
- the `MRI Training (Native HPC)` block
- `Starting training for ... epochs`
- `Epoch 1/... - training`

## 3. After Segmentation Training Finishes, What To Check

First confirm that SLURM finished successfully:

```bash
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed
```

The expected state is `COMPLETED` with exit code `0:0`.

Then inspect the run directory:

```bash
ls "${PROJECT_DIR}/checkpoints/seg/<run_name>"
```

Expected files:

- `resolved_config.yaml`
- `run_manifest.json`
- `run_summary.json`
- `metrics_history.csv`
- `train.log`
- `<run_name>_best.pt`
- `<run_name>_last.pt`

Important checks:

- `run_summary.json`: confirm training finished and produced validation metrics
- `metrics_history.csv`: inspect whether `val/dice`, `val/precision`, and `val/recall` improved across epochs
- `<run_name>_best.pt`: this is the checkpoint to use for segmentation inference
- `train.log`: check for NaNs, exploding loss, or other runtime problems

If `run_summary.json` is missing, or the checkpoint files are missing, treat the run as incomplete even if the SLURM log exists.

## 4. Next Step After Segmentation Training

Segmentation training alone is not enough for downstream classification.

Before classification training, run segmentation inference on every split needed by classification:

- `train`
- `val`
- usually `test` too, if you also want final evaluation later

Example:

```bash
export PROJECT_DIR=/p/scratch/ebrains-0000006/kim27/cancer_detector
export SEG_RUN=<run_name>
export SEG_CKPT="${PROJECT_DIR}/checkpoints/seg/${SEG_RUN}/${SEG_RUN}_best.pt"
export SEG_OUT="${PROJECT_DIR}/data/seg_preds"

sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always,WANDB_MODE=offline \
  "${PROJECT_DIR}/scripts/new/inference" \
  --config mri/config/task/segmentation.yaml \
  --checkpoint "${SEG_CKPT}" \
  --output_dir "${SEG_OUT}" \
  --split train

sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always,WANDB_MODE=offline \
  "${PROJECT_DIR}/scripts/new/inference" \
  --config mri/config/task/segmentation.yaml \
  --checkpoint "${SEG_CKPT}" \
  --output_dir "${SEG_OUT}" \
  --split val

sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always,WANDB_MODE=offline \
  "${PROJECT_DIR}/scripts/new/inference" \
  --config mri/config/task/segmentation.yaml \
  --checkpoint "${SEG_CKPT}" \
  --output_dir "${SEG_OUT}" \
  --split test
```

After inference, verify that the segmentation prediction root contains per-case outputs such as:

- `<case_id>/prostate_prob.npy`
- `<case_id>/target_prob.npy`
- `<case_id>/overlays/<slice_idx>.png`

## 5. Before Training Classification

Classification training depends on segmentation predictions.

Check [classification.yaml](../mri/config/task/classification.yaml) and make sure:

- `data.seg_pred_dir` points at the segmentation prediction root
- the predictions exist for all cases in the `train` and `val` splits

If you used the default output above, `data.seg_pred_dir: data/seg_preds` already matches the expected location.

## 6. How To Train Classification

Once segmentation predictions are ready, start classification training:

```bash
export PROJECT_DIR=/p/scratch/ebrains-0000006/kim27/cancer_detector

sbatch \
  --export=ALL,MRI_TRAIN_USE_CONTAINER=always,WANDB_MODE=offline \
  "${PROJECT_DIR}/scripts/new/train" \
  --config mri/config/task/classification.yaml \
  --output-dir "${PROJECT_DIR}/checkpoints/cls" \
  --run-name classification-$(date +%Y-%m-%d-%H-%M-%S)
```

Monitor it the same way as segmentation training.

## 7. After Classification Training

Repeat the same checks as for segmentation:

- SLURM state is `COMPLETED`
- `run_summary.json` exists
- `metrics_history.csv` exists
- best and last checkpoints exist
- the validation metrics look reasonable

Then you can run classification inference on the desired split using the trained classification checkpoint.

## 8. Recommended End-To-End Order

Use this order for the full pipeline:

1. train segmentation
2. check segmentation metrics and checkpoints
3. run segmentation inference on `train`
4. run segmentation inference on `val`
5. run segmentation inference on `test`
6. confirm `data.seg_pred_dir` is correct
7. train classification
8. check classification metrics and checkpoints
9. run classification inference
