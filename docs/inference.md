# Inference

This repo uses a two-stage pipeline:

1. train a segmentation model
2. run segmentation inference and save probability maps for every case
3. train a classification model using the original MRI plus those saved probability maps
4. run classification inference for the final case-level prediction

Segmentation is not optional in the current downstream setup. The classification dataset expects saved segmentation predictions to already exist.

## Why We Need Segmentation First

The downstream classifier predicts one label per case, but the actual cancer target is small compared with the full study. If we feed the whole volume directly, most of the input is background or non-target tissue.

The segmentation model solves that localization problem. Its outputs are used to:

- find the most relevant slice neighborhood
- decide whether to focus on target tissue or fall back to the prostate region
- build the ROI crop for the classifier

Important detail: the classifier uses the saved probability maps, not the overlay PNGs and not only hard masks.

## What Is The Input?

### Segmentation Input

Default segmentation samples come from `mri.data.datasets.segmentation.SegmentationDataset`.

- input type: slice-level 2.5D tensor
- default channels: `5` T2 context slices + `1` ADC slice + `1` calculated high-b slice
- default total channels: `7`
- image size: `256 x 256`
- output channels: `2` masks
- output classes: `prostate`, `target`

So the default segmentation input shape is roughly:

```text
(7, 256, 256)
```

If we use `stack_depth = 7`, the input becomes `9` channels instead of `7`.

### Classification Input

Default classification samples come from `mri.data.datasets.classification.ClassificationDataset`.

- input type: case-level 3D tensor
- modalities: `t2`, `adc`, `calc`
- default channels: `3`
- default depth window: `16` slices
- ROI crop: `192 x 192`
- resized output: `256 x 256`
- zero-padding is used when `adc` or `calc` is missing

So the default classification input shape after preprocessing is roughly:

```text
(3, 16, 256, 256)
```

The default classifier config is a 3D `resnet101` with:

- `n_input_channels = 3`
- `num_classes = 5`

Labels come from `case_info["class"]`, except cases without a target are mapped to class `0`.

## How Segmentation Helps The Classifier

At classification time, we load:

- `<case_id>/target_prob.npy`
- `<case_id>/prostate_prob.npy`

Then the dataset uses those probabilities to:

- score slices and choose a center slice
- detect whether a confident target is present
- threshold the probability volume at `data.selection.min_prob` (default `0.3`) to build an ROI mask
- crop around that ROI before resizing and sending the volume into the classifier

If the target probability is weak, the pipeline falls back to the prostate probability map so we still get an anatomically meaningful crop.

This is the main reason a stronger segmentation model directly helps downstream classification.

## Current Segmentation Summary

Based on [best_jobs.html](../checkpoints/reports/best_jobs.html), generated on `2026-04-04`:

- `81` segmentation jobs were included in the leaderboard
- best `val/precision_target`: `seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons`
- that run reached `val/precision_target = 0.2603` at epoch `65`
- the same run reached `val/threshold_sweep_target_best_dice = 0.3216`
- recipe summary: `SimpleUNet`, `stack_depth=5`, positive-only pool, geometric augmentation, gentle modality dropout, light target weighting, conservative `OneCycle`

Exact config of that current precision leader:

```text
checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml
```

There is also a stronger threshold-sweep Dice leader:

- run: `seg-auto-seg-apr04-autopilot-debug-w02-r11-dynu-s5-prec-md0-w1-cons`
- best `val/threshold_sweep_target_best_dice = 0.4025`
- but its raw `val/precision_target` is lower at `0.1030`

So for downstream experiments, the natural first checkpoint to try is the current precision leader, and the threshold-sweep Dice leader is a good comparison if the ROI looks too tight or too loose.

## How We Train And Run The Full Pipeline

### 1. Train Segmentation

Baseline training:

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
```

If we want to reproduce the current precision leader exactly:

```bash
python mri/cli/train.py \
  --config checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml
```

### 2. Run Segmentation Inference On `train`, `val`, And `test`

Use one shared output root, because classification expects a single `data.seg_pred_dir` containing all required case folders.

Important: the config must match the checkpoint architecture. For example, a `SimpleUNet` checkpoint must not be loaded with the baseline `segmentation.yaml` `SegResNet` config.

```bash
SEG_CFG=checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml
SEG_CKPT=checkpoints/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons_best.pt
SEG_OUT=data/seg_preds/seg_apr04_precision_leader

for split in train val test; do
  python mri/cli/infer.py \
    --config "$SEG_CFG" \
    --split "$split" \
    --checkpoint "$SEG_CKPT" \
    --output_dir "$SEG_OUT" \
    --run_name "seg_apr04_precision_leader_${split}"
done
```

### 3. Train Classification

Point `data.seg_pred_dir` in `mri/config/task/classification.yaml` at the same `SEG_OUT` directory, then train:

```bash
python mri/cli/train.py --config mri/config/task/classification.yaml
```

During classification training, the crop is still prediction-driven. The dataset uses the segmentation probabilities to choose the center slice and ROI, with a small random center jitter for robustness.

### 4. Run Classification Inference

```bash
python mri/cli/infer.py \
  --config mri/config/task/classification.yaml \
  --split test \
  --checkpoint checkpoints/cls/<run>/<run>_best.pt
```

### 5. One Command Using Existing Checkpoints

If you already have both trained checkpoints and want to skip retraining, use:

```bash
python mri/cli/pipeline_infer.py \
  --seg-config checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml \
  --seg-checkpoint checkpoints/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons_best.pt \
  --cls-config mri/config/task/classification.yaml \
  --cls-checkpoint checkpoints/cls/<run>/<run>_best.pt \
  --cls-inference-split test
```

This runner:

- runs segmentation inference first
- writes segmentation probabilities under one shared output root
- generates a temporary classification override config with `data.seg_pred_dir` pointing at those fresh predictions
- runs classification inference without starting any training job

Native HPC wrapper:

```bash
bash scripts/new/pipeline-inference \
  --seg-config checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml \
  --seg-checkpoint checkpoints/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons/seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons_best.pt \
  --cls-config mri/config/task/classification.yaml \
  --cls-checkpoint checkpoints/cls/<run>/<run>_best.pt \
  --dry-run
```

## What Segmentation Inference Writes

For each case:

- `<case_id>/prostate_prob.npy`
- `<case_id>/target_prob.npy`
- `<case_id>/overlays/<slice_idx>.png`

For each inference run:

- `<run_name>_inference_summary.json`
- `<run_name>_inference_manifest.json`
- `<run_name>_resolved_config.yaml`

The classifier reads the `.npy` probability volumes. The overlay PNGs are only for visual QC.

## What Classification Inference Writes

- `predictions.csv`
- `<run_name>_inference_summary.json`
- `<run_name>_inference_manifest.json`
- `<run_name>_resolved_config.yaml`

`predictions.csv` contains `case_id`, predicted class, top confidence, ground-truth label, and per-class probabilities.

## Common Failure Modes

`data.seg_pred_dir must be set`

- classification config is missing the segmentation prediction root

`Missing segmentation predictions for ...`

- segmentation inference was not run for every required case
- for classification training, the shared prediction root must contain both `train` and `val` cases

checkpoint/config mismatch during inference

- the config defines the model architecture before the checkpoint is loaded
- use the config that matches the checkpoint you are trying to run

## Short Takeaway

Our downstream model is not trained directly on raw full studies. We first segment prostate and target regions, export probability maps, and then use those maps to build a smaller, lesion-aware 3D classification input. That is why the stronger segmentation results in [best_jobs.html](../checkpoints/reports/best_jobs.html) are the key unlock for the next classification round.
