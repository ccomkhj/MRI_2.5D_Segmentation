# Training

Training is split into two stages:

1. segmentation training
2. classification training

Segmentation must run first. Classification training requires segmentation probability maps to already exist for every case in the train and validation splits.

## Local Training

### Segmentation

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
```

### Classification

Before classification, make sure `data.seg_pred_dir` points at completed segmentation inference outputs.

```bash
python mri/cli/train.py --config mri/config/task/classification.yaml
```

If predictions are missing, the CLI now fails early with a clear error instead of failing deep inside the dataset loader.

## Common CLI Overrides

`mri/cli/train.py` supports:

- `--output_dir`
- `--run_name`
- `--epochs`
- `--batch_size`
- `--lr`
- `--device`

Example:

```bash
python mri/cli/train.py \
  --config mri/config/task/segmentation.yaml \
  --epochs 10 \
  --batch_size 2 \
  --device cpu \
  --run_name seg-local-debug
```

## Default Task Configs

- [segmentation.yaml](../mri/config/task/segmentation.yaml)
- [classification.yaml](../mri/config/task/classification.yaml)

These task configs use layered YAML via `extends`.

## Output Files

Each training run writes a run directory under `train.output_dir/<run_name>/` with:

- `resolved_config.yaml`
- `run_manifest.json`
- `run_summary.json`
- `metrics_history.csv`
- `<run_name>_best.pt`
- `<run_name>_last.pt`

## Native HPC Training

Direct run:

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
```

SLURM submission:

```bash
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
```

Dry-run:

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
```

The wrapper:

- auto-loads `.env` when present
- sets cache directories for Torch and Hugging Face
- writes into `checkpoints/`, `wandb/`, and `.cache/matplotlib`
- uses `srun` automatically inside a SLURM allocation

## Training Order For The Full Pipeline

Recommended sequence:

1. train segmentation
2. run segmentation inference on `train`, `val`, and `test`
3. train classification
4. run classification inference

If you want that sequence managed for you, use [research.md](research.md).
