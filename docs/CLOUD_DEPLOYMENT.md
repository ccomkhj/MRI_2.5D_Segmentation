# Cloud Deployment

Use the modular pipeline for cloud runs. The old service-based workflow is archived.

## Local Machine

```bash
python tools/deployment/data_backup.py
scp mri_data_backup_*.zip user@cloud:/path/to/repo/
```

## Cloud Machine

```bash
git clone <repo-url>
cd <repo>
conda create -n mri python=3.12 -y
conda activate mri
pip install -r requirements.txt
python tools/deployment/data_restore.py mri_data_backup_*.zip
python mri/cli/train.py --config mri/config/task/segmentation.yaml
```

If you run on an HPC-style cluster, prefer `scripts/new/train` and `scripts/new/inference`.
