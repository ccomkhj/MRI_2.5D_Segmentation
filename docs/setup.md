# Setup

This guide prepares a local development environment for the modular MRI pipeline.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) is installed locally
- `dicom_mapper` is cloned next to `cancer_detector` (it is a path dependency in `pyproject.toml`)
- You are running commands from the repository root
- The aligned dataset source exists at `/Users/huijokim/personal/tcia-handler/data/aligned_v2`

## Create The Environment

```bash
uv sync
```

`uv sync` creates `.venv/` and installs the locked dependency set (including editable `dicom_mapper`). It also installs the `dev` dependency group, which provides `pytest` for local validation.

Run commands without activating the venv:

```bash
uv run python -m mri.service.ui
uv run pytest tests/test_smoke_configs.py -q
```

## Repository Assumptions

- Modern entrypoints live under `mri/cli/`
- Native HPC wrappers live under `scripts/new/`
- Compatibility wrappers under `service/` are not the recommended path for new work

## Minimum Sanity Checks

Validate Python imports and basic smoke configs:

```bash
uv run python -m compileall mri service tools tests
uv run pytest tests/test_smoke_configs.py -q
```

Check that the one-command smoke workflow resolves correctly:

```bash
bash scripts/new/research-smoke --dry-run
```

## What To Read Next

- Use [data.md](data.md) to materialize `data/aligned_v2`
- Use [splits.md](splits.md) to create the shared dated split file
- Use [research.md](research.md) if you want one command for the full local workflow
