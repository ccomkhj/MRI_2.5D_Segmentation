# MRI Modular Pipeline (New Architecture)

This repository uses a config-driven modular architecture for prostate MRI segmentation and classification.

**Official execution paths:**
- `mri/cli/train.py`
- `mri/cli/infer.py`
- `mri/cli/sweep.py`
- `mri/cli/research.py`
- `scripts/new/train`
- `scripts/new/inference`

Compatibility wrappers exist in `service/train.py` and `service/inference.py`, but new work should use the `mri/cli/*` and `scripts/new/*` entrypoints.

## Documentation

The root README is the overview. Detailed workflow guides live under `docs/`.

Recommended starting points:

- [docs/README.md](docs/README.md): documentation index
- [docs/setup.md](docs/setup.md): environment setup and sanity checks
- [docs/data.md](docs/data.md): import or sync `aligned_v2`
- [docs/splits.md](docs/splits.md): generate the shared dated split
- [docs/configuration.md](docs/configuration.md): layered YAML config composition
- [docs/train.md](docs/train.md): segmentation-first training workflow
- [docs/inference.md](docs/inference.md): segmentation and classification inference
- [docs/research.md](docs/research.md): end-to-end local research runner
- [docs/smoke.md](docs/smoke.md): short CPU smoke workflows
- [docs/sweeps.md](docs/sweeps.md): sweep and downstream promotion flow
- [docs/paper_run_checklist.md](docs/paper_run_checklist.md): checklist for real paper runs

## Architecture Overview

The pipeline is organized around reusable modules in `mri/`:

- `mri/config/`: YAML config loading (`defaults + task config`)
- `mri/data/`: metadata loading, split/index builders, datasets
- `mri/models/`: segmentation/classification model registry + builders
- `mri/tasks/`: task-specific loss/metric handling
- `mri/training/`: trainer loop, checkpointing, device resolution
- `mri/inference/`: segmentation/classification inference runners
- `mri/cli/`: train/infer/sweep/research entrypoints

## Pipeline Flow

```mermaid
flowchart TD
    A["Source dataset"]
    B["Import or sync<br/>`tools/dataset/import_tcia_aligned.py`"]
    C["Local aligned dataset<br/>`data/aligned_v2`"]
    D["Shared dated split<br/>`data/splits/YYYY-MM-DD.yaml`"]
    E["Segmentation stage<br/>train then infer"]
    F["Segmentation probability maps"]
    G["Classification stage<br/>train then infer"]
    H["Predictions, checkpoints, manifests, metrics"]

    I["`mri/cli/research.py`<br/>runs the full local pipeline"]
    J["`mri/cli/sweep.py`<br/>runs config sweeps and downstream promotion"]

    A --> B --> C --> D --> E --> F --> G --> H

    I -. orchestrates .-> B
    I -. orchestrates .-> E
    I -. orchestrates .-> G

    J -. compares segmentation runs .-> E
    J -. promotes best upstream run .-> G
```

The important constraint is structural: segmentation runs first, and classification consumes the selected MRI inputs together with segmentation prediction outputs.

## Repository Structure

```text
mri/
  cli/
    train.py
    infer.py
    sweep.py
    research.py
  config/
    defaults.yaml
    task/
      segmentation.yaml
      classification.yaml
  data/
  models/
  tasks/
  training/
  inference/

scripts/
  new/
    train
    inference

tools/
  generate_splits.py
```

## Environment Setup

```bash
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
```

## Data Split Preparation

Import or sync aligned data produced by the sibling `tcia-handler` repository:

```bash
python tools/dataset/import_tcia_aligned.py --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 --dest data/aligned_v2 --mode link
```

Use `--mode copy` if you need a physical copy instead of a symlink.

Generate split YAMLs from metadata:

```bash
python tools/generate_splits.py --metadata data/aligned_v2/metadata.json --output data/splits/2026-03-15.yaml --label-space downstream_5class
```

That command also writes a JSON summary next to the split file with per-split label histograms.

Use one dated split artifact for both segmentation and classification paper runs.

## Training

### Local

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
python mri/cli/train.py --config mri/config/task/classification.yaml
```

`classification` requires segmentation predictions to already exist under `data.seg_pred_dir`.

### HPC (Native, No Container)

```bash
bash scripts/new/train --config mri/config/task/segmentation.yaml
sbatch scripts/new/train --config mri/config/task/segmentation.yaml
```

You can pass selected runtime overrides on the CLI, for example:

```bash
sbatch scripts/new/train --config mri/config/task/segmentation.yaml --epochs 50 --batch_size 16
```

## Inference

Set inference values in the config (especially `inference.checkpoint` and `inference.output_dir`), then run:

```bash
python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test
python mri/cli/infer.py --config mri/config/task/classification.yaml --split test
```

HPC native launch:

```bash
bash scripts/new/inference --config mri/config/task/segmentation.yaml --split test
sbatch scripts/new/inference --config mri/config/task/segmentation.yaml --split test
```

## Native HPC Script Behavior

`scripts/new/train` and `scripts/new/inference`:
- run directly on node Python environment
- support both direct run and `sbatch`
- auto-load `.env` if present
- auto-export runtime paths/caches (`DATA_DIR`, `CHECKPOINT_DIR`, `WANDB_DIR`, `TORCH_HOME`, `HF_HOME`)
- use `srun` automatically inside a SLURM allocation

Dry-run validation:

```bash
bash scripts/new/train --dry-run --config mri/config/task/segmentation.yaml
bash scripts/new/inference --dry-run --config mri/config/task/segmentation.yaml --split test
```

## Sweeps And Downstream Promotion

Launch a bounded grid sweep from a sweep config:

```bash
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml --dry-run
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml
```

Promote the top-1 completed segmentation run into downstream classification preparation:

```bash
python mri/cli/sweep.py --downstream-config mri/config/sweep/classification/downstream_top1.yaml --dry-run
```

That downstream flow:
- selects the best completed segmentation training run by Dice from local manifests
- prepares segmentation inference jobs for downstream classification `train`, `val`, and `test`
- wires the resulting prediction root into a generated classification sweep config
- launches or dry-runs the downstream classification sweep

## Local Research Workflow

Run the full local workflow from aligned data import through downstream classification:

```bash
python mri/cli/research.py --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 --dest-data data/aligned_v2 --split-file data/splits/2026-03-15.yaml --disable-wandb --dry-run
python mri/cli/research.py --source-data /Users/huijokim/personal/tcia-handler/data/aligned_v2 --dest-data data/aligned_v2 --split-file data/splits/2026-03-15.yaml --disable-wandb
```

That workflow:
- syncs or validates `data/aligned_v2`
- generates or reuses one dated split file
- trains segmentation and writes a best checkpoint
- runs segmentation inference for `train`, `val`, and `test`
- trains classification using those segmentation predictions
- runs classification inference on the selected split

## Smoke Workflow

Run the checked-in 3-case CPU smoke test with one command:

```bash
bash scripts/new/research-smoke --dry-run
bash scripts/new/research-smoke
```

That smoke workflow uses:
- [segmentation_smoke.yaml](/Users/huijokim/personal/mri/mri/config/task/segmentation_smoke.yaml)
- [classification_smoke.yaml](/Users/huijokim/personal/mri/mri/config/task/classification_smoke.yaml)
- [smoke_3case.yaml](/Users/huijokim/personal/mri/data/splits/smoke_3case.yaml)

It is intended only to verify the end-to-end pipeline on CPU, not to produce meaningful research metrics.

## Configuration Contract

Task configs are expected to define:
- `task.name`: `segmentation` or `classification`
- `data.metadata`
- `data.split_file`
- `model.name` and optional `model.params`
- `train.*` and `inference.*` sections (missing fields fall back to `mri/config/defaults.yaml`)

## Notes

- This README reflects only the new architecture.
- For production runs, prefer `scripts/new/*` on HPC and `mri/cli/*` locally.
- Use `service/train.py` and `service/inference.py` only for compatibility with older scripts.
