# Local Research Workflow

`mri/cli/research.py` is the end-to-end local workflow runner.

It is the right entrypoint when you want one command that handles:

1. dataset import or validation
2. split generation or reuse
3. segmentation training
4. segmentation inference for `train`, `val`, and `test`
5. classification training
6. classification inference

## Dry-Run First

```bash
python mri/cli/research.py \
  --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest-data data/aligned_v2 \
  --split-file data/splits/2026-03-15.yaml \
  --disable-wandb \
  --dry-run
```

Dry-run writes:

- a research manifest
- generated segmentation and classification configs
- split metadata when the split is generated or reused

It does not launch training or inference.

## Full Run

```bash
python mri/cli/research.py \
  --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest-data data/aligned_v2 \
  --split-file data/splits/2026-03-15.yaml \
  --disable-wandb
```

## Important Flags

- `--source-data`: source aligned dataset path
- `--dest-data`: repository-local aligned dataset path
- `--import-mode copy|link`: how to materialize the dataset locally
- `--skip-import`: reuse an existing local dataset without syncing
- `--force-import`: replace a mismatched destination dataset
- `--validate-import-files`: enable sample-level file validation
- `--seg-config`: base segmentation task config
- `--cls-config`: base classification task config
- `--split-file`: reuse or generate a specific split YAML
- `--regenerate-split`: overwrite an existing split file
- `--label-space`: `downstream_5class` or `original`
- `--seg-inference-splits`: default is `train,val,test`
- `--cls-inference-split`: default is `test`
- `--device`: override both training and inference devices
- `--disable-wandb`: disable W&B in the generated configs
- `--output-root`: root directory for the research run
- `--run-name`: stable name for the run directory

## Output Layout

Each run creates:

```text
experiments/research/<run-name>/
  configs/
  manifests/
  checkpoints/segmentation/
  checkpoints/classification/
  predictions/segmentation/
  predictions/classification/
```

## Generated Configs

The runner does not mutate the base task configs directly.

Instead, it writes generated override configs that:

- point at the chosen metadata path
- point at the chosen split file
- set output directories for train and inference
- wire classification `data.seg_pred_dir` to the segmentation prediction root

## Research Manifest

The manifest records stage-by-stage execution for:

- import
- split
- config generation
- segmentation training
- segmentation inference
- classification training
- classification inference

This is the main artifact to inspect when you need to understand what happened in a run.

## When Not To Use This Entry Point

Do not use `mri/cli/research.py` for:

- large parameter sweeps
- HPC batch experiments that should be scheduled as independent jobs

Use [sweeps.md](sweeps.md) for that.
