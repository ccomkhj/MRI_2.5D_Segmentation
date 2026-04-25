# Segmentation error-analysis & label-audit tool — design

**Date:** 2026-04-25
**Status:** spec, not yet implemented
**Scope:** segmentation only (classification stage untouched)

## Problem

The current segmentation leader sits at `val/precision_target = 0.26` and `val/threshold_sweep_target_best_dice = 0.32` after the autopilot has already swept the obvious modeling knobs (focal loss, class-imbalance handling, geometric augmentation, modality dropout, OneCycle scheduling, four backbones). Further sweeps are unlikely to break the ceiling because we do not know where the model is failing.

Validation runs today only log aggregate metrics. Per-case predictions are not saved, so error attribution requires re-running inference. The val set is small (~20–60 cases) and each case has a gland mask, a lesion mask, and a 0–4 class label — enough to do real attribution if the artifacts are written to disk.

The next concrete improvement is therefore *not* a model change. It is a tool that turns the current best checkpoint into evidence about where it fails, including evidence that the GT itself may be the bottleneck.

## Goal

Ship one CLI that, given a finished run directory, produces:

1. Per-case error metrics for the lesion channel that distinguish *localization* errors (FP outside the gland) from *discrimination* errors (FP inside the gland), stratified by 0–4 class label.
2. A label-audit queue that surfaces GT cases likely to be wrong or inconsistent, ranked for human review.
3. A single HTML report that ties (1) and (2) together with per-case slice visualizations.

The tool surfaces; the human decides. No automatic exclusion of cases, no retraining, no modeling fix is part of this spec.

## Non-goals

- Comparing two runs side-by-side.
- Auto-excluding flagged cases from training.
- Test-set evaluation (val-only).
- Any modeling fix — those come *after* this tool tells us where to fix.
- Editing the trainer to save predictions during training (this tool runs post-hoc on a finished checkpoint).
- Re-sweeping the operating threshold (we use the threshold the run already chose).

## Architecture

The tool is a CLI that operates on a finished run directory — the same artifact the UI already consumes (`<run_dir>/<name>_best.pt` + `resolved_config.yaml`). It reuses the existing val split, model registry, and inference path so it cannot disagree with how training measures the same checkpoint.

### File layout

```
mri/
  cli/
    diagnose.py           # entry point: `python -m mri.cli.diagnose <run_dir>`
  diagnostics/
    __init__.py
    dump.py               # run inference on val, save per-case prob maps + GT
    attribute.py          # per-case error metrics (in/out gland, by class)
    audit.py              # label-noise heuristics
    report.py             # render HTML report
  templates/
    diagnostic_report.html.j2

tests/diagnostics/        # unit tests for attribute.py + audit.py on synthetic cases
```

### Output layout

Outputs land under `<run_dir>/diagnostic/`:

```
diagnostic/
  predictions/<case_id>/{prob.npz, gt.npz, meta.json}
  metrics_by_case.csv
  metrics_by_class.csv
  label_audit.csv
  report.html
```

Two reasons for this layout:

1. `diagnostic/` lives next to the checkpoint so it is obvious which run it describes and is naturally version-controlled with the run.
2. The per-case `predictions/` cache means re-running just the analysis (faster iteration on heuristics) does not re-run inference.

## Components

### `dump.py`

Loads `resolved_config.yaml` and `*_best.pt` from the run directory, builds the val dataloader the same way the trainer does, runs inference one case at a time, and writes:

- `prob.npz` — per-channel probability volume (gland, lesion) at the model's native resolution.
- `gt.npz` — GT gland mask, GT lesion mask, in the same volume space.
- `meta.json` — `case_id`, `class_label` (0–4), `voxel_spacing`, `shape`, threshold values used by the run.

If the per-case directory already exists with a non-empty `prob.npz`, `dump.py` skips inference for that case unless `--force` is passed. This makes heuristic iteration cheap.

### `attribute.py`

Reads the dumped predictions and produces `metrics_by_case.csv` with one row per case. Thresholds are pulled from `meta.json` — we use the run's existing operating threshold, not a re-sweep. We do not emit two columns per metric for "best-Dice threshold"; that would muddy the diagnosis by reporting on a different operating point than the run was selected at.

For the splitting boundary between FP-inside-gland and FP-outside-gland we use the **GT** gland mask, not the predicted gland. Reason: we want to attribute *lesion-model* errors, not compound them with gland-model errors. Gland-model quality shows up separately in `gland_dice`.

Columns:

**Lesion channel** (the one that matters for `precision_target`):

| Column | Meaning |
|---|---|
| `dice` | Dice on the lesion channel at the run's threshold |
| `precision` | Precision on the lesion channel |
| `recall` | Recall on the lesion channel |
| `fp_voxels_inside_gland` | FP lesion voxels falling inside GT-gland |
| `fp_voxels_outside_gland` | FP lesion voxels falling outside GT-gland |
| `fn_voxels` | Missed lesion voxels |
| `tp_voxels` | True-positive lesion voxels |
| `fp_outside_ratio` | `fp_outside / (fp_outside + fp_inside + tp)` — single number for "is localization or discrimination the problem?" |

**Gland channel** (sanity check):

| Column | Meaning |
|---|---|
| `gland_dice` | Dice on the gland channel — flags cases where the lesion ROI is being fed wrong |

**Stratification keys:**

| Column | Meaning |
|---|---|
| `class_label` | 0–4, joined to classification label |
| `lesion_volume_gt_voxels` | For "do we miss small lesions?" |
| `case_id` | |
| `status` | `ok` or `failed` (with NaN metrics if `failed`) |

`metrics_by_class.csv` aggregates these by `class_label` (one row per class 0..4), with `dice`, `precision`, `recall`, `fp_outside_ratio` averaged over cases in that class.

### `audit.py`

Emits `label_audit.csv` with one row per case that triggers at least one heuristic. Columns: `case_id`, `class_label`, `flags` (semicolon-joined list of flag names), `priority` (1=high, 2=medium, 3=low), `reason` (one-line per flag, semicolon-joined).

Six heuristics, all conservative — surface candidates for human review, do not auto-exclude.

1. **`class_mask_inconsistent`** (priority 1). Case has `class_label > 0` but empty lesion mask, or `class_label == 0` but non-empty lesion mask. Pure metadata check, no model needed. These are usually data-loading bugs or annotation drift; either way the case is unusable as-is.

2. **`high_confidence_disagreement`** (priority 1). Model predicts lesion with mean probability > 0.8 over a 3D connected component of ≥ 50 voxels that lies inside the GT gland and overlaps zero GT-lesion voxels. Classic "missed annotation" signal.

3. **`tiny_gt_island`** (priority 2). GT lesion mask contains a 3D connected component < 10 voxels and the model predicts ~0 probability there. Possible annotation noise (a stray brush stroke).

4. **`gt_volume_outlier`** (priority 3). GT lesion volume is in the top or bottom 5% of non-empty cases. Outliers are not errors, but they are disproportionately expensive in a small val set. Surfaced for review, not for exclusion.

5. **`erratic_slice_consistency`** (priority 2). GT lesion appears, disappears, and reappears across z-slices with gaps > 1 slice between components. Real lesions are usually contiguous in z.

6. **`class_severity_mismatch`** (priority 2). Model predicts very high lesion probability mass on a `class_label == 1` case (low-severity) or near-zero mass on a `class_label == 4` case (high-severity), at a magnitude that is an outlier vs. the rest of the class.

Default thresholds (50 voxels, 10 voxels, 5%, 0.8) are hardcoded in `audit.py`. They become a YAML config only if a follow-up actually changes them.

### `report.py`

Renders `report.html` from a Jinja template. Uses the same lightweight Jinja-based pattern as `mri/service/pipeline.py`'s clinician report — no new templating stack.

Layout, top to bottom:

1. **Header** — run name, checkpoint path, val split name, date, and headline numbers (overall lesion Dice, overall `precision_target`, gland Dice). One sentence each.
2. **Per-class breakdown table** — `metrics_by_class.csv` as a table, five rows (class 0–4), conditional formatting on Dice (green > 0.5, red < 0.2).
3. **Audit queue** — `label_audit.csv` as a sortable table, sorted by priority. Each row links to the case section below.
4. **Per-case sections** — one section per case in the audit queue at priorities 1 and 2 by default; priority 3 elided unless `--include-low-priority` is passed. Each section shows `case_id`, `class_label`, lesion Dice, `fp_outside_ratio`, audit flags, and a 3-slice grid centered on the GT-lesion-mass-weighted central z-slice (slice index `s*`, then `s*-1` and `s*+1`; for cases with empty GT lesion, fall back to the predicted-mass-weighted central slice). Each slice shows three panels: GT lesion overlay on T2, predicted-lesion-probability heatmap on T2, and a binary disagreement panel (FP red, FN blue, TP green at the run's threshold). Anchors so the audit queue links jump straight here.
5. **Worst-cases-without-flags section** — top 5 cases by lesion Dice in the bottom decile that did not trigger any audit flag. These are the cases where the model is genuinely struggling and the labels look fine, i.e. real model errors to plan modeling fixes around.

Slice rendering uses matplotlib PNGs embedded as base64. With ~60 val cases, page generation should complete in under ~30 seconds. No interactive 3D viewer in v1.

## Data flow

1. User runs `python -m mri.cli.diagnose <run_dir>`.
2. `diagnose.py` validates the run directory has both `*_best.pt` and `resolved_config.yaml`. Hard error at startup if either is missing, with a message pointing at the expected layout.
3. `dump.py` loads the val split via the same dataloader the trainer uses, runs inference per case, writes per-case `prob.npz` + `gt.npz` + `meta.json`. Skips cases that already have a non-empty `prob.npz` unless `--force`.
4. `attribute.py` reads each per-case directory, computes metrics, writes `metrics_by_case.csv` and `metrics_by_class.csv`.
5. `audit.py` reads each per-case directory + the metric CSVs, applies the six heuristics, writes `label_audit.csv`.
6. `report.py` reads all three CSVs and the per-case predictions, renders `report.html`.

## Edge cases

- **Empty GT lesion mask.** Dice / precision / recall reported as `NaN`, not 0. NaN means "undefined"; 0 implies a measurement. Excluded from class aggregates.
- **Inference failure on a case.** Skipped with a warning. Listed in `metrics_by_case.csv` with `status=failed` and NaN metrics. The report header reports how many were skipped.
- **Missing `resolved_config.yaml` or `*_best.pt`.** Hard error at startup. Message lists the expected files.
- **Threshold absent from the run's logged metrics.** Fall back to 0.5 with a warning. Do not silently pick something else.

## Testing

- **Unit tests** on `attribute.py` and `audit.py` against synthetic cases — small numpy arrays where TPs / FPs / FNs are constructed by hand. Verifies per-case metric math and each audit heuristic in isolation. Fast, no model, no I/O.
- **Smoke test** on `dump.py` using a tiny dummy checkpoint + 2-case dummy split (CPU, < 30s). Verifies the inference→disk path does not break when configs change.
- **No end-to-end test on real data.** The point of the tool is to *run* on real runs; pinning behavior on real data fights its purpose.

## Risks

- **The diagnostic runs and tells us the labels are fine and the model errors are spread evenly.** Then the tool has narrowed the search but not pointed at a fix. Mitigation: this is still strictly more information than we have today, and the tool is reusable on every future checkpoint.
- **Heuristic thresholds are wrong for our data.** Mitigation: the heuristics are conservative and the audit is human-in-the-loop. If a heuristic fires on every case or no cases, we tune the threshold once after first use.
- **Inference at val time uses a slightly different transform stack than training.** Mitigation: reuse the existing val dataloader exactly — do not build a new one. If the existing val path is wrong, the diagnostic exposes that, which is itself a valuable finding.

## Acceptance

The tool is done when, for the current segmentation leader, it produces:

- A `metrics_by_case.csv` with the columns above.
- A `metrics_by_class.csv` with one row per class 0–4.
- A `label_audit.csv` with at least one heuristic having fired (or, if none fired, a header note in the report saying so).
- An `report.html` that opens in a browser, renders the per-class table, and links from the audit queue to per-case sections with slice visualizations.

The downstream brainstorm — what model change to make based on these outputs — is a separate spec.
