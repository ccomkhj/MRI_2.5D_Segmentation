# Scripts Overview

The supported HPC wrappers are:

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
bash scripts/new/pipeline-inference --seg-checkpoint checkpoints/seg/<run>/<run>_best.pt --cls-checkpoint checkpoints/cls/<run>/<run>_best.pt
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/pipeline-inference --seg-checkpoint checkpoints/seg/<run>/<run>_best.pt --cls-checkpoint checkpoints/cls/<run>/<run>_best.pt
```

Dry-run validation:

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/classification.yaml --split test
bash scripts/new/pipeline-inference --dry-run --seg-checkpoint checkpoints/seg/<run>/<run>_best.pt --cls-checkpoint checkpoints/cls/<run>/<run>_best.pt
```

Older container wrappers were moved to `archive/scripts/`.
