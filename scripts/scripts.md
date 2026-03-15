# Scripts Overview

The supported HPC wrappers are:

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

Dry-run validation:

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/classification.yaml --split test
```

Older container wrappers were moved to `archive/scripts/`.
