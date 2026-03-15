# Tools

Utility scripts live under `tools/`. The modular train and inference path is `mri/cli/*`; tools exist to prepare data, validate datasets, and move artifacts.

## Directories

- `preprocessing/`: DICOM, overlay, and metadata preparation helpers
- `tcia/`: TCIA manifest generation
- `deployment/`: backup and restore helpers for moving datasets between machines
- `validation/`: dataset and mask diagnostics
- `dataset/`: older standalone dataset helpers that are not used by `mri/cli/*`

## Common Commands

```bash
python tools/dataset/import_tcia_aligned.py --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 --dest data/aligned_v2 --mode link
python tools/generate_splits.py --metadata data/aligned_v2/metadata.json --output data/splits/2026-03-15.yaml --label-space downstream_5class
bash scripts/new/research-smoke --dry-run
python tools/preprocessing/dicom_converter.py --all
python tools/preprocessing/process_overlay_aligned.py
python tools/validation/test_dataset_basic.py --manifest data/processed/class2/manifest.csv
python tools/deployment/data_backup.py
python tools/deployment/data_restore.py backup.zip
```

`generate_splits.py` also writes `<split>_summary.json` with label histograms for `train`, `val`, and `test`.

For training and inference commands, use [README.md](../README.md).
