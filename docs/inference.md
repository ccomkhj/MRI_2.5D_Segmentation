# Inference

Inference is also split by task.

Segmentation inference produces the probability maps used by downstream classification. Classification inference produces the final case-level predictions.

## Local Inference

### Segmentation

```bash
python mri/cli/infer.py \
  --config mri/config/task/segmentation.yaml \
  --split test \
  --checkpoint checkpoints/segmentation/<run>/<run>_best.pt
```

Run segmentation inference on `train`, `val`, and `test` when you are preparing classification inputs.

### Classification

Classification inference requires:

- a trained classification checkpoint
- `data.seg_pred_dir` pointing at the segmentation prediction root

```bash
python mri/cli/infer.py \
  --config mri/config/task/classification.yaml \
  --split test \
  --checkpoint checkpoints/classification/<run>/<run>_best.pt
```

## Common CLI Overrides

`mri/cli/infer.py` supports:

- `--split`
- `--checkpoint`
- `--output_dir`
- `--run_name`
- `--batch_size`
- `--device`

Example:

```bash
python mri/cli/infer.py \
  --config mri/config/task/segmentation.yaml \
  --split val \
  --checkpoint checkpoints/segmentation/seg-run/seg-run_best.pt \
  --output_dir predictions/segmentation \
  --run_name seg-run-val \
  --device cpu
```

## Segmentation Outputs

Segmentation inference writes:

- `<case_id>/prostate_prob.npy`
- `<case_id>/target_prob.npy`
- `<run_name>_inference_summary.json`
- `<run_name>_inference_manifest.json`
- `<run_name>_resolved_config.yaml`

The classification dataset accepts either the two-`.npy` case layout or an `.npz` file containing `prostate_prob` and `target_prob`.

## Classification Outputs

Classification inference writes:

- `predictions.csv`
- `<run_name>_inference_summary.json`
- `<run_name>_inference_manifest.json`
- `<run_name>_resolved_config.yaml`

## Native HPC Inference

Direct run:

```bash
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

SLURM submission:

```bash
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

Dry-run:

```bash
bash scripts/new/inference --dry-run --config mri/config/task/classification.yaml --split test
```

The wrapper accepts `--output` as an alias for `--output_dir`.

## Common Failure Modes

`data.seg_pred_dir must be set`

- classification config is missing the segmentation prediction root

`Missing segmentation predictions for ...`

- segmentation inference did not run for every required case
- rerun segmentation inference for the relevant split

## What To Read Next

- Use [research.md](research.md) if you want training and inference orchestrated together
- Use [smoke.md](smoke.md) for short CPU validation runs
