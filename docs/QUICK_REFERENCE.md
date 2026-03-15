# Quick Reference

## Setup

```bash
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## Prepare Splits

```bash
python tools/generate_splits.py --metadata data/aligned_v2/metadata.json --output data/splits/2026-03-08.yaml
```

## Train

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
python mri/cli/train.py --config mri/config/task/classification.yaml
python mri/cli/train.py --config mri/config/task/segmentation.yaml --epochs 10 --batch_size 2
```

## Inference

```bash
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test
python mri/cli/infer.py --config mri/config/task/classification.yaml --split test
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test --checkpoint checkpoints/seg/<run>/<file>.pt
```

## Sweeps

```bash
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml --dry-run
python mri/cli/sweep.py --downstream-config mri/config/sweep/classification/downstream_top1.yaml --dry-run
```

## Validation And Utilities

```bash
python service/validate_data.py
python tools/validation/test_dataset_basic.py --manifest data/processed/class2/manifest.csv
python tools/deployment/data_backup.py
python tools/deployment/data_restore.py backup.zip
```
