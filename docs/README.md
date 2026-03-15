# MRI Documentation

This directory is the step-by-step documentation set for the modular `mri/` workflow.

The root [README.md](../README.md) is the project overview. Use the guides below when you need the detailed operational steps.

## Recommended Reading Order

1. [setup.md](setup.md): create the environment and verify the repo is runnable
2. [data.md](data.md): import or sync `aligned_v2` from `tcia-handler`
3. [splits.md](splits.md): generate the dated split YAML used by both tasks
4. [train.md](train.md): run segmentation first, then classification training
5. [inference.md](inference.md): generate segmentation probabilities and classification predictions
6. [research.md](research.md): run the full local segmentation-to-classification pipeline

## Detailed Workflow Guides

- [setup.md](setup.md): Python environment, dependency install, and local sanity checks
- [data.md](data.md): dataset import, validation, and repository-local data contract
- [splits.md](splits.md): downstream-aware split generation and split summary artifacts
- [configuration.md](configuration.md): layered YAML config composition and reusable config structure
- [train.md](train.md): local training and native HPC training wrappers
- [inference.md](inference.md): local inference and native HPC inference wrappers
- [sweeps.md](sweeps.md): segmentation sweeps and downstream top-1 promotion
- [research.md](research.md): end-to-end local research workflow with generated configs and manifests
- [smoke.md](smoke.md): small CPU smoke workflows for 3-case and 5-case validation
- [paper_run_checklist.md](paper_run_checklist.md): checklist for real non-smoke research runs

## Supporting Docs

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md): short command-only reference
- [slurm.md](slurm.md): native HPC wrapper usage
- [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md): backup and restore helpers
- [JUSUF.md](JUSUF.md): cluster-specific notes

## Legacy Docs

Older service-era, container-era, and exploratory notes live in `docs/archive/`.
