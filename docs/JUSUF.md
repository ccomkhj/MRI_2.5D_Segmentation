# JUSUF HPC Configuration Guide

## Overview
JUSUF (Jülich Supercomputing Facility) is a high-performance computing system at Forschungszentrum Jülich. This guide documents key configuration requirements and common issues when running GPU-accelerated jobs.

## SLURM Job Submission

### GPU Partition Configuration

**Critical**: GPU jobs must use the `gpus` partition, NOT `gpu`.

```bash
# Correct - will allocate GPU nodes
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1

# Incorrect - will allocate CPU-only nodes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

**Why this matters**: Even with `--gres=gpu:1`, using the wrong partition means your job runs on CPU-only nodes, causing NVIDIA-SMI to fail with:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

### Account Requirements

JUSUF requires specifying a budget account for all job submissions:

```bash
#SBATCH --account=ebrains-0000006
```

**To find your available accounts:**
```bash
jutil user projects
```

### Complete Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=mri-train-wandb
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --account=ebrains-0000006
```

## GPU Node Specifications

- **GPU Type**: NVIDIA V100 PCIe (16 GB memory per GPU)
- **GPUs per node**: 1
- **Partition**: `gpus`
- **Default GPU allocation**: `--gres=gpu:1` is default, no need to specify explicitly

## Common Issues and Solutions

### 1. Script Path Resolution in SLURM
**Symptom**: `/var/spool/parastation/jobs/JOBID: line X: /var/spool/parastation/jobs/submit_slurm.sh: No such file or directory`
**Cause**: SLURM copies scripts to a temporary location, breaking relative path resolution
**Solution**: Use absolute paths in SLURM scripts instead of relative paths

**Bad:**
```bash
exec "$(dirname "${BASH_SOURCE[0]}")/submit_slurm.sh" ${ARGS} "$@"
```

**Good:**
```bash
exec "/full/path/to/project/scripts/new/train" "$@"
```

### 2. NVIDIA Driver Communication Error
**Symptom**: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`
**Cause**: Job running on CPU partition instead of GPU partition
**Solution**: Change `--partition=gpu` to `--partition=gpus`

### 2. Account Not Specified Error
**Symptom**: `sbatch: error: job_submit_filter: please specify the job's account`
**Cause**: Missing `--account` directive
**Solution**: Add `--account=<your-project-account>` to job script

### 3. Container Runtime Issues
**Available container runtimes**:
- Apptainer (preferred, newer Singularity replacement)
- Singularity (legacy)

**Auto-detection in scripts**:
```bash
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: Neither 'singularity' nor 'apptainer' found in PATH."
    exit 1
fi
```

## GPU Visibility and Affinity

SLURM automatically manages GPU visibility:
- Sets `CUDA_VISIBLE_DEVICES` environment variable
- Ensures CPU cores have affinity to allocated GPUs
- Default: one task per GPU, GPU appears as device 0 to applications

## Job Monitoring

**Check job status**:
```bash
squeue -u $USER
```

**View job output**:
```bash
# Replace JOBID with actual job ID
cat slurm-JOBID.out
cat slurm-JOBID.err
```

**Cancel job**:
```bash
scancel JOBID
```

## Best Practices

1. **Always use `sbatch`, never `bash`** for SLURM scripts
2. **Verify partition name**: `gpus` for GPU jobs, not `gpu`
3. **Specify account**: Required for all job submissions
4. **Test with short jobs**: Use short runtimes for testing (`--time=00:15:00`)
5. **Check GPU info**: Include `nvidia-smi` calls in scripts for verification
6. **Use absolute paths**: Avoid relative paths that may break in batch environment

## Key Resources

- [JUSUF User Documentation](https://apps.fz-juelich.de/jsc/hps/jusuf/)
- [GPU Computing Guide](https://apps.fz-juelich.de/jsc/hps/jusuf/gpu-computing.html)
- [SLURM Documentation](https://apps.fz-juelich.de/jsc/hps/jusuf/batch-system.html)

## Recent Changes

- **Partition naming**: Confirmed `gpus` is correct for GPU access (not `gpu`)
- **Account requirement**: All jobs require explicit account specification
- **Container support**: Apptainer preferred over legacy Singularity
