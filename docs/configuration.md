# Configuration Guide

The modular pipeline uses layered YAML configs loaded by [mri/config/loader.py](../mri/config/loader.py).

This is the core rule:

1. load [defaults.yaml](../mri/config/defaults.yaml)
2. recursively load any `extends` entries
3. deep-merge the current file on top

Later values override earlier values.

## Config Layers

Common config locations:

- [mri/config/defaults.yaml](../mri/config/defaults.yaml): repo-wide defaults
- [mri/config/preset/](../mri/config/preset): task training presets
- [mri/config/model/](../mri/config/model): model family presets
- [mri/config/task/](../mri/config/task): runnable task configs
- [mri/config/sweep/](../mri/config/sweep): sweep definitions

## How `extends` Works

Example task config:

```yaml
extends:
  - ../preset/segmentation_train.yaml
  - ../model/segmentation/segresnet.yaml

data:
  metadata: data/aligned_v2/metadata.json
  split_file: data/splits/2026-03-08.yaml
  stack_depth: 5
```

That means:

- start from `defaults.yaml`
- merge `segmentation_train.yaml`
- merge `segresnet.yaml`
- merge the current file

Relative `extends` paths are resolved from the YAML file that declares them.

## Current Base Task Configs

- [segmentation.yaml](../mri/config/task/segmentation.yaml)
- [classification.yaml](../mri/config/task/classification.yaml)
- [segmentation_smoke.yaml](../mri/config/task/segmentation_smoke.yaml)
- [classification_smoke.yaml](../mri/config/task/classification_smoke.yaml)

## Important Sections

`task`

- `name`: `segmentation` or `classification`
- `stage`: usually `train`

`data`

- `metadata`
- `split_file`
- `modalities`
- `zero_pad_missing`
- segmentation-only: `stack_depth`
- classification-only: `seg_pred_dir`, `selection`, `depth`, `roi`

`model`

- `name`
- `params`

`train`

- `batch_size`
- `epochs`
- `lr`
- `weight_decay`
- `device`
- `output_dir`

`inference`

- `batch_size`
- `device`
- `output_dir`
- `checkpoint`

`tracking`

- `wandb.enabled`

## Current Presets

- [segmentation_train.yaml](../mri/config/preset/segmentation_train.yaml)
- [classification_train.yaml](../mri/config/preset/classification_train.yaml)
- [segresnet.yaml](../mri/config/model/segmentation/segresnet.yaml)
- [swin.yaml](../mri/config/model/classification/swin.yaml)

## CLI Overrides Vs YAML Edits

Use CLI overrides for temporary run-level changes:

```bash
python mri/cli/train.py \
  --config mri/config/task/segmentation.yaml \
  --epochs 10 \
  --batch_size 2 \
  --device cpu
```

Use YAML files when the change should be stable and reusable:

- a new model family
- a paper run config
- a smoke config
- a sweep definition

## Research Runner Generated Configs

[mri/cli/research.py](../mri/cli/research.py) does not edit the base task config in place.

It writes generated override configs into:

```text
experiments/research/<run-name>/configs/
```

Those generated configs:

- pin the metadata path
- pin the split file
- set train and inference output directories
- wire classification `data.seg_pred_dir` to segmentation outputs

## Recommended Workflow

- keep reusable task defaults in `mri/config/task/`
- keep model family defaults in `mri/config/model/`
- keep training defaults in `mri/config/preset/`
- use generated configs only as run artifacts, not as source-controlled config templates

## Common Mistakes

`Config not found`

- the `--config` path is wrong

`Config extends cycle detected`

- two YAML files extend each other, directly or indirectly

Unexpected inherited model parameters

- a child config switched model families but inherited stale params from a previous base
- this repo now filters unsupported model kwargs more safely, but keep configs clean anyway

## What To Read Next

- [train.md](train.md) for task execution
- [research.md](research.md) for generated run configs
- [paper_run_checklist.md](paper_run_checklist.md) for a full non-smoke execution checklist
