# Paper Run Checklist

Use this checklist before starting a real segmentation-plus-classification experiment that you intend to compare, report, or publish.

This checklist is intentionally stricter than the smoke workflow.

## 1. Freeze The Environment

Run from the repository root:

```bash
conda activate mri
python -m pip install -r requirements.txt
python -m pip install pytest
```

Record:

- current git branch
- current commit SHA
- Python version
- CUDA availability if you are using GPU

## 2. Materialize The Dataset Locally

```bash
python tools/dataset/import_tcia_aligned.py \
  --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest data/aligned_v2 \
  --mode link \
  --validate-files
```

Record:

- import manifest path
- metadata hash
- case count
- sample count

## 3. Create One Dated Split For The Entire Run

```bash
python tools/generate_splits.py \
  --metadata data/aligned_v2/metadata.json \
  --output data/splits/2026-03-15.yaml \
  --label-space downstream_5class
```

Inspect the generated summary JSON before training starts:

- confirm `train`, `val`, and `test` are non-empty
- confirm downstream label coverage is reasonable
- keep this split fixed for both segmentation and classification

Do not create separate split files for the two tasks.

## 4. Pin The Task Configs

Pick the exact config files for:

- segmentation
- classification

If you need paper-specific variants, create dedicated config files under `mri/config/task/` rather than relying only on ad-hoc CLI overrides.

Read [configuration.md](configuration.md) before creating new config layers.

## 5. Run Segmentation Training

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
```

Check the run directory and keep:

- `resolved_config.yaml`
- `run_manifest.json`
- `run_summary.json`
- `metrics_history.csv`
- best checkpoint path

## 6. Run Segmentation Inference For Train, Val, And Test

```bash
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split train --checkpoint checkpoints/seg/<run>/<run>_best.pt
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split val --checkpoint checkpoints/seg/<run>/<run>_best.pt
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test --checkpoint checkpoints/seg/<run>/<run>_best.pt
```

Confirm that every required case has:

- `prostate_prob.npy`
- `target_prob.npy`

Classification depends on these files.

## 7. Point Classification At The Segmentation Prediction Root

Before training classification, confirm `data.seg_pred_dir` resolves to the segmentation prediction directory you just generated.

If this path is wrong or incomplete, classification now fails early by design.

## 8. Run Classification Training

```bash
python mri/cli/train.py --config mri/config/task/classification.yaml
```

Keep:

- `resolved_config.yaml`
- `run_manifest.json`
- `run_summary.json`
- `metrics_history.csv`
- best checkpoint path

## 9. Run Classification Inference On The Evaluation Split

```bash
python mri/cli/infer.py --config mri/config/task/classification.yaml --split test --checkpoint checkpoints/cls/<run>/<run>_best.pt
```

Keep:

- `predictions.csv`
- inference summary JSON
- inference manifest JSON

## 10. Archive The Run Metadata

For the final record, save:

- dataset import manifest
- split YAML
- split summary JSON
- segmentation run manifest and summary
- segmentation inference summaries
- classification run manifest and summary
- classification inference summary
- exact commands used

If you use `mri/cli/research.py`, the research manifest already ties these stages together.

## 11. Sanity Checks Before Comparing Metrics

Verify:

- segmentation and classification used the same split file
- classification used segmentation predictions from the intended segmentation run
- model family and hyperparameters match the run you think you executed
- W&B mode or offline mode matches your intended tracking behavior
- no smoke config was used by accident

## 12. Prefer A Dry-Run Before Expensive Changes

For new configs or a new machine, run:

```bash
python mri/cli/research.py \
  --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest-data data/aligned_v2 \
  --split-file data/splits/2026-03-15.yaml \
  --disable-wandb \
  --dry-run
```

Then launch the real run only after the generated config and split artifacts look correct.

## Minimum Deliverables For A Serious Run

- one dated split file
- one segmentation best checkpoint
- segmentation predictions for `train`, `val`, and `test`
- one classification best checkpoint
- one classification prediction CSV
- manifests and summaries for every stage

## Related Guides

- [setup.md](setup.md)
- [data.md](data.md)
- [splits.md](splits.md)
- [configuration.md](configuration.md)
- [train.md](train.md)
- [inference.md](inference.md)
- [research.md](research.md)
