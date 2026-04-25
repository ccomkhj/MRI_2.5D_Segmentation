---
name: mri-autoresearch
description: Run bounded autonomous research loops in the cancer_detector repository. Use when Codex needs an autoresearch-style workflow for MRI segmentation or classification experiments by choosing one repo-native lane (`mri/cli/research.py`, `mri/cli/sweep.py`, `scripts/segmentation_autopilot.py`, `scripts/swinunetr_autopilot.py`, or a small hand-curated config wave), keeping a narrow editable surface, creating an untracked `experiments/autoresearch/tag-name/` workspace, launching dry-runs or SLURM jobs, reading manifests and reports, and deciding which changes to keep or discard.
---

# MRI Autoresearch

Use this skill to adapt Karpathy's `autoresearch` pattern to this repo without pretending the repo is a single-file trainer. Keep one experiment lane and one editable surface in scope, create an untracked `experiments/autoresearch/<tag>/` workspace, start with a dry-run or baseline submission, and only keep changes supported by the repo's own manifests or reports.

## Choose The Lane

- Use `research` for one end-to-end local segmentation-to-classification run through `mri/cli/research.py`.
- Use `sweep` for bounded config comparisons or downstream top-1 promotion through `mri/cli/sweep.py`.
- Use `manual-wave` for a hand-curated 6-12 config batch, for example the current `mri/config/task/apr22_swin_diverse/` SwinUNETR overlays.
- Use `segmentation-autopilot` for unattended segmentation campaigns through `scripts/segmentation_autopilot.py`.
- Use `swinunetr-autopilot` for unattended SwinUNETR-only campaigns through `scripts/swinunetr_autopilot.py`.

Prefer the narrowest lane that can answer the user's question. Do not mix lanes inside one run tag unless the earlier lane has already produced a clear winner and you are intentionally promoting it.

## Choose The Editable Surface

- Treat the repo like `train.py` plus `program.md`, but adapt the writable surface to this project.
- Keep exactly one surface writable for a run tag:
  - one task-config family under `mri/config/task/`
  - one sweep config under `mri/config/sweep/`
  - one controller or submit script under `scripts/`
- Treat these as read-only during a run unless the user explicitly asks to widen scope:
  - data import and validation code under `tools/dataset/` and `tools/validation/`
  - shared metric and reporting code under `mri/tasks/`, `mri/experiments/latest_jobs_report.py`, `mri/cli/train.py`, and `mri/cli/infer.py`
  - baseline docs that only describe prior runs

This keeps diffs reviewable and makes keep/discard decisions legible.

## Bootstrap The Run

1. Pick a short tag such as `apr22-swin-a`.
2. If the worktree is clean and isolation matters, create a branch or worktree such as `autoresearch/<tag>`. If the worktree is dirty, stay on the current branch and record the starting `HEAD` instead of switching branches.
3. Run the bootstrap helper:

```bash
python .codex/skills/mri-autoresearch/scripts/bootstrap_run.py <tag> --lane <lane> --goal "<goal>"
```

4. Read the generated `program.md` in `experiments/autoresearch/<tag>/`.
5. Verify prerequisites before the first real run:
   - the relevant dataset or split exists
   - the chosen config path exists
   - the launcher passes a `--dry-run` or `--help` check when possible
6. Record the baseline before making the first keep/discard decision.

Use `--surface`, `--seed-path`, and `--metric` on the bootstrap script when the editable surface or success metric is already known.

## Run The Loop

1. Start from a baseline dry-run, smoke run, or already-completed seed run.
2. Make one meaningful change at a time inside the chosen surface.
3. Launch the narrowest validating command first:
   - local `--dry-run`
   - `bash scripts/new/train --dry-run ...`
   - sweep dry-run
   - autopilot dry-run
4. Only after the setup passes, launch the real run or submission.
5. Capture artifacts, not narratives. Prefer manifest paths, sweep summaries, `state.json`, and report HTML over long terminal logs.
6. Append a row to `results.tsv` after each attempt. Leave the workspace untracked.
7. Keep a change only when the repo's own metrics say it helped and the complexity tradeoff is justified.

## Judge Outcomes

- Use the repo's primary metric as ground truth. Do not invent a new scoreboard.
- For segmentation or classification single runs, inspect `run_manifest.json` and `run_summary.json`.
- For `research.py`, inspect `experiments/research/<run>/manifests/research_manifest.json` plus the generated configs.
- For `sweep.py`, inspect `sweep_manifest.json` and the generated reports under `reports/`.
- For downstream promotion, inspect `downstream_manifest.json`.
- For autopilots, inspect `checkpoints/autopilot/<campaign>/state.json`, `autopilot.log`, and the refreshed `checkpoints/reports/latest_jobs.html` and `best_jobs.html`.
- When the signal is mixed, prefer simpler changes or revert to the previous best known state.

## Repo Defaults

- Start with [references/repo-map.md](references/repo-map.md).
- For the currently active SwinUNETR lane, inspect `mri/config/task/apr22_swin_diverse/` and `scripts/submit_swinunetr_apr22_diverse_12.sh`.
- Use `docs/research.md`, `docs/sweeps.md`, `docs/slurm.md`, and `docs/progress/auto_pilot.md` only when the chosen lane needs extra detail.

## Guardrails

- Keep `results.tsv`, scratch notes, and generated `program.md` under `experiments/autoresearch/<tag>/`; do not commit them.
- Prefer config overlays over edits to shared training code.
- Do not change both the search policy and the model or config surface in the same experiment unless the user explicitly requests a larger jump.
- Do not discard a good run only because a secondary metric moved slightly; note the tradeoff in `results.tsv`.
- When a run crashes, log it, inspect the smallest useful traceback or manifest field, fix only the obvious issue, and move on.
- If the user asks for a one-off analysis instead of an indefinite loop, keep the same structure but stop after the requested scope.
