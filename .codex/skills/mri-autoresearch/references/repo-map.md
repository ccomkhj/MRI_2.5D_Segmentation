# Repo Map

Load this file first when the skill triggers.

## Entry Points

- `mri/cli/research.py`: one end-to-end local segmentation to classification run with generated configs and a `research_manifest.json`
- `mri/cli/sweep.py`: bounded segmentation or classification sweeps plus downstream top-1 promotion
- `scripts/segmentation_autopilot.py`: unattended general segmentation controller
- `scripts/swinunetr_autopilot.py`: unattended SwinUNETR-only controller
- `scripts/new/train`: canonical training wrapper for direct run or `sbatch`
- `scripts/new/inference`: canonical inference wrapper for direct run or `sbatch`
- `scripts/submit_swinunetr_apr22_diverse_12.sh`: current helper for the April 22 SwinUNETR manual wave

## Primary Artifacts

- `experiments/research/<run>/manifests/research_manifest.json`
- `experiments/segmentation/<sweep>/sweep_manifest.json`
- `experiments/segmentation/<sweep>/reports/runs.csv`
- `experiments/classification/<stage>/downstream_manifest.json`
- `checkpoints/<task>/<run>/run_manifest.json`
- `checkpoints/<task>/<run>/run_summary.json`
- `checkpoints/<task>/<run>/resolved_config.yaml`
- `checkpoints/autopilot/<campaign>/state.json`
- `checkpoints/autopilot/<campaign>/autopilot.log`
- `checkpoints/reports/latest_jobs.html`
- `checkpoints/reports/best_jobs.html`

## First Commands By Lane

- `research`

```bash
bash scripts/new/research-smoke --dry-run
python mri/cli/research.py --help
```

- `sweep`

```bash
python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml --dry-run
python mri/cli/sweep.py --downstream-config mri/config/sweep/classification/downstream_top1.yaml --dry-run
```

- `manual-wave`

```bash
bash scripts/new/train --dry-run --config mri/config/task/apr22_swin_diverse/waveA01_base_bs1_lr2p50e04.yaml
bash scripts/submit_swinunetr_apr22_diverse_12.sh A
```

- `segmentation-autopilot`

```bash
python scripts/segmentation_autopilot.py --campaign seg-apr22 --mode cadence --dry-run
```

- `swinunetr-autopilot`

```bash
python scripts/swinunetr_autopilot.py --help
python scripts/swinunetr_autopilot.py --campaign swin-apr22 --waves 3
```

## Useful Search Patterns

```bash
rg --files mri/config/task mri/config/sweep
rg -n "primary_metric_name|threshold_sweep|batch_size|lr|scheduler" mri/config/task mri/config/sweep
find checkpoints -name run_manifest.json
find experiments -name sweep_manifest.json
```

## Read More Only If Needed

- `docs/research.md`
- `docs/sweeps.md`
- `docs/slurm.md`
- `docs/progress/auto_pilot.md`
- `docs/JUSUF.md`
