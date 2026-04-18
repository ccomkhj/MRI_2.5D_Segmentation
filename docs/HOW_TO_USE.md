# How To Use This Project

This project analyzes prostate MRI data in two steps:

1. `segmentation`: find the prostate and likely target region
2. `classification`: predict the final case-level class

Most users do not need to think about the internal details. If you already have trained model files, the easiest way to use this project is the one-command pipeline below.

## The Easiest Option

Use this when you already have:

- a segmentation checkpoint
- a classification checkpoint
- the matching config files

Recommended first step:

```bash
python mri/cli/pipeline_infer.py \
  --seg-config <segmentation-config.yaml> \
  --seg-checkpoint <segmentation-best.pt> \
  --cls-config <classification-config.yaml> \
  --cls-checkpoint <classification-best.pt> \
  --dry-run
```

`--dry-run` is safe. It only shows what would run.

If that looks correct, run the real command:

```bash
python mri/cli/pipeline_infer.py \
  --seg-config <segmentation-config.yaml> \
  --seg-checkpoint <segmentation-best.pt> \
  --cls-config <classification-config.yaml> \
  --cls-checkpoint <classification-best.pt>
```

What this does:

- runs segmentation inference first
- saves segmentation probability maps
- uses those maps for classification
- writes the final predictions

## What You Need Before Running

You need these items:

- MRI data prepared under `data/aligned_v2`
- a split file such as `data/splits/<YYYY-MM-DD>.yaml`
- a segmentation config that matches the segmentation checkpoint
- a classification config that matches the classification checkpoint

Important:

- the segmentation config must match the segmentation checkpoint architecture
- the classification config must match the classification checkpoint architecture
- classification cannot run correctly unless segmentation predictions are available first

## Where The Results Go

The one-command pipeline writes results under:

```text
experiments/pipeline_inference/<run-name>/
```

Main outputs:

- `predictions/segmentation/`: segmentation probability maps
- `predictions/classification/predictions.csv`: final classification results
- `manifests/`: records of what was run
- `configs/`: generated temporary config files used for this run

## If You Want Only The Final Test Prediction

The default classification split is `test`.

If you want a different split:

```bash
python mri/cli/pipeline_infer.py \
  --seg-config <segmentation-config.yaml> \
  --seg-checkpoint <segmentation-best.pt> \
  --cls-config <classification-config.yaml> \
  --cls-checkpoint <classification-best.pt> \
  --cls-inference-split val \
  --seg-inference-splits val
```

Use the same split for both commands unless you have a specific reason not to.

## If You Need To Train New Models

Use the full research workflow instead of the checkpoint-only workflow:

```bash
python mri/cli/research.py \
  --source-data /path/to/aligned_v2 \
  --dest-data data/aligned_v2 \
  --split-file data/splits/<YYYY-MM-DD>.yaml \
  --disable-wandb \
  --dry-run
```

Then remove `--dry-run` to launch the full run.

This full workflow:

- prepares or reuses the data
- prepares or reuses the split
- trains segmentation
- runs segmentation inference
- trains classification
- runs classification inference

## Common Problems

`Missing segmentation predictions`

- segmentation inference did not run
- or classification is looking in the wrong prediction folder

`checkpoint/config mismatch`

- the config file does not match the model checkpoint
- use the config that was used to create that checkpoint

`data.seg_pred_dir must be set`

- classification does not know where the segmentation outputs are
- the one-command pipeline handles this automatically

## Simple Rule Of Thumb

If you already have trained models, use:

```bash
python mri/cli/pipeline_infer.py ...
```

If you need to train new models, use:

```bash
python mri/cli/research.py ...
```

## Related Docs

- [inference.md](inference.md): more detail about inference
- [research.md](research.md): full end-to-end training and inference
- [README.md](README.md): documentation index
