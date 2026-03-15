# Smoke Workflows

Smoke workflows exist to verify that the pipeline wiring works. They are not intended to produce meaningful research metrics.

## 3-Case One-Command Smoke Run

Use the checked-in wrapper:

```bash
bash scripts/new/research-smoke --dry-run
bash scripts/new/research-smoke
```

This uses:

- [segmentation_smoke.yaml](../mri/config/task/segmentation_smoke.yaml)
- [classification_smoke.yaml](../mri/config/task/classification_smoke.yaml)
- [smoke_3case.yaml](../data/splits/smoke_3case.yaml)

## 5-Case Full Pipeline Validation

For a slightly stronger local workflow check, use the checked-in 5-case split:

```bash
python mri/cli/research.py \
  --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest-data data/aligned_v2 \
  --import-mode link \
  --split-file data/splits/smoke_5case.yaml \
  --seg-config mri/config/task/segmentation_smoke.yaml \
  --cls-config mri/config/task/classification_smoke.yaml \
  --output-root experiments/research \
  --disable-wandb \
  --device cpu \
  --run-name smoke-5case-full
```

That uses:

- [smoke_5case.yaml](../data/splits/smoke_5case.yaml)

## What The Smoke Workflows Validate

- aligned dataset import or reuse
- split loading
- segmentation training
- segmentation inference
- classification training
- classification inference
- manifest generation
- generated override configs

## What They Do Not Validate

- convergence quality
- stable benchmark metrics
- production-ready hyperparameters

## Recommended Usage

- use the 3-case workflow for a fast CPU sanity check
- use the 5-case workflow when you want slightly broader pipeline coverage before starting a larger run
