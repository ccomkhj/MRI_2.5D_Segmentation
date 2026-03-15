# Setup

This guide prepares a local development environment for the modular MRI pipeline.

## Prerequisites

- Conda is installed locally
- You are running commands from the repository root
- The aligned dataset source exists at `/Users/huijokim/personal/tcia-handler/data/aligned_v2`

## Create The Environment

```bash
conda create -n mri python=3.12 -y
conda activate mri
python -m pip install -r requirements.txt
```

`pytest` is useful for local validation and is not guaranteed to be included in `requirements.txt`:

```bash
python -m pip install pytest
```

## Repository Assumptions

- Modern entrypoints live under `mri/cli/`
- Native HPC wrappers live under `scripts/new/`
- Compatibility wrappers under `service/` are not the recommended path for new work

## Minimum Sanity Checks

Validate Python imports and basic smoke configs:

```bash
python -m compileall mri service tools tests
python -m pytest tests/test_smoke_configs.py -q
```

Check that the one-command smoke workflow resolves correctly:

```bash
bash scripts/new/research-smoke --dry-run
```

## What To Read Next

- Use [data.md](data.md) to materialize `data/aligned_v2`
- Use [splits.md](splits.md) to create the shared dated split file
- Use [research.md](research.md) if you want one command for the full local workflow
