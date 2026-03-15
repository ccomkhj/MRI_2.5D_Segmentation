# Sweeps And Downstream Promotion

Use `mri/cli/sweep.py` for bounded experiment sweeps and downstream promotion from segmentation into classification.

The CLI requires exactly one of:

- `--config`
- `--sweep_dir`
- `--downstream-config`

## Segmentation Sweep

Dry-run:

```bash
python mri/cli/sweep.py \
  --config mri/config/sweep/segmentation/stack_depth_grid.yaml \
  --dry-run
```

Launch:

```bash
python mri/cli/sweep.py \
  --config mri/config/sweep/segmentation/stack_depth_grid.yaml
```

## Summarize An Existing Sweep

```bash
python mri/cli/sweep.py --sweep_dir experiments/segmentation/<sweep-name>
```

## Downstream Top-1 Promotion

Dry-run:

```bash
python mri/cli/sweep.py \
  --downstream-config mri/config/sweep/classification/downstream_top1.yaml \
  --dry-run
```

Launch:

```bash
python mri/cli/sweep.py \
  --downstream-config mri/config/sweep/classification/downstream_top1.yaml
```

## What Downstream Promotion Does

It:

- finds the best completed upstream segmentation run from local manifests
- prepares segmentation inference jobs for `train`, `val`, and `test`
- wires the resulting prediction root into a generated classification sweep config
- launches or dry-runs the downstream classification sweep

## Important Constraint

Downstream promotion requires real completed upstream segmentation runs.

A segmentation sweep dry-run is not enough. If no completed segmentation run manifests exist, downstream promotion fails by design.

## Output Locations

Sweep summaries and generated configs are written under `experiments/segmentation/` or `experiments/classification/`, depending on the task.

## When To Use Research Instead

Use [research.md](research.md) when you want a single local run over one fixed split.

Use sweeps when you want to compare multiple configs or automate downstream promotion after ranking upstream segmentation runs.
