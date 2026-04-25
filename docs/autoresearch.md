# Autoresearch

This repo includes a project-local Codex skill at `.codex/skills/mri-autoresearch`.

Use it when you want an autoresearch-style loop in this repository without inventing a new workflow from scratch. The skill keeps one experiment lane and one editable surface in scope, creates an untracked `experiments/autoresearch/<tag>/` workspace, and points Codex at the repo's existing manifests and reports.

Bootstrap a run with:

```bash
python .codex/skills/mri-autoresearch/scripts/bootstrap_run.py apr22-swin --lane manual-wave --goal "Improve SwinUNETR segmentation without widening scope"
```

Then invoke Codex with the skill explicitly, for example:

```text
Use $mri-autoresearch at .codex/skills/mri-autoresearch to run a bounded SwinUNETR experiment loop in this repo.
```
