# SLURM

The supported HPC path is the native wrapper pair in `scripts/new/`.

## Train

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
```

## Inference

```bash
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

## Dry Run

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/classification.yaml --split test
```

## Sweep Runner

```bash
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml --dry-run
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml
python mri/cli/sweep.py --downstream-config mri/config/sweep/classification/downstream_top1.yaml --dry-run
```

Container-era SLURM scripts were moved to `archive/scripts/`.
