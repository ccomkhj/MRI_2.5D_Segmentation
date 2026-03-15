# Deployment Tools

These helpers package runtime data for moving the repo between local machines, cloud instances, and clusters.

## Backup

```bash
python tools/deployment/data_backup.py
python tools/deployment/data_backup.py --include-checkpoints
python tools/deployment/data_backup.py --full
```

## Restore

```bash
python tools/deployment/data_restore.py backup.zip
python tools/deployment/data_restore.py backup.zip --dry-run
python tools/deployment/data_restore.py backup.zip --force
```

After restore, continue with the modular workflow:

```bash
python mri/cli/train.py --config mri/config/task/segmentation.yaml
```
