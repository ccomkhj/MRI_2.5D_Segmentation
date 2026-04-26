# Diagnosing a segmentation run

This is the practical runbook for using `mri.cli.diagnose` on a real segmentation checkpoint and turning its output into a concrete decision about what to fix next. It assumes you have a finished run directory and that aggregate-metric sweeps have already plateaued.

The CLI reference is in [`diagnostic.md`](diagnostic.md). This document is the workflow on top of that reference.

## When to run this

Run the diagnosis when at least one of these is true:

- Aggregate metrics have stopped moving and you don't know **why**.
- A new sweep is being planned but you can't articulate what hypothesis it tests.
- A clinician is asking why a particular case looks wrong and you can't answer from the saved metrics alone.

Do **not** run it after every routine training job. It's a once-per-noteworthy-checkpoint exercise — the artifacts are saved and reusable.

## Prerequisites

| Need | How to check |
|---|---|
| Run directory containing `*_best.pt` and `resolved_config.yaml` (or `*_resolved_config.yaml`) | `ls path/to/run_dir` |
| The val split file referenced by the resolved config exists on disk | `python -c "import yaml; print(yaml.safe_load(open('path/to/run_dir/resolved_config.yaml'))['data']['split_file'])"` then `ls` that path |
| The `aligned_v2` (or equivalent) data root is present locally | check `data.metadata` from the same config |
| `uv` venv synced | `uv sync` |

If the val set is on a remote cluster, sync it first; the diagnosis loads the same dataloader the trainer used.

## Step 1 — Run the tool

```bash
uv run python -m mri.cli.diagnose path/to/run_dir
```

This takes roughly 30 seconds per case on CPU and 2–5 seconds per case on a GPU. For a 30-case val set, plan on 5–15 minutes.

What to look for in the console output:

- `[diagnose] dump: {'cases_written': N, 'cases_skipped_cached': 0, 'cases_failed_inference': [], ...}` — `cases_failed_inference` should be empty. If it isn't, those cases will appear in the report header as "skipped" and they need investigating before you trust the rest of the report.
- `[diagnose] report: <path>/diagnostic/report.html` — the path to open in a browser.

If you re-run the analysis after tweaking heuristics or thresholds, the dump step is cached — only the analysis re-runs, in a few seconds. Pass `--force` to re-do inference.

## Step 2 — Read the report in this order

Open `<run_dir>/diagnostic/report.html`. Read the sections **top to bottom** but with these priorities:

### 2a. Header — sanity check

Look at the three headline numbers (overall lesion Dice, overall precision, overall gland Dice). Confirm they match what you remember from training. If they don't, the val split or threshold being used here may differ from the run's logged metrics — investigate before proceeding.

If the header says "N case(s) skipped (inference failure)", **stop**. The diagnosis is incomplete. Look at the run console for which cases failed and why.

### 2b. Per-class breakdown table — the highest-leverage view

This is the single table that most often answers "where does the model fail?".

The expected shape on a struggling model is one of:

- **One class dominates Dice and the others collapse.** E.g. class 4 at Dice 0.6, class 1 at Dice 0.05. Says: the model has learned the easy aggressive cases but not the subtler ones. Fix candidates: focal loss tuning per class, oversampling rare classes, harder negative mining.
- **Dice is uniformly mediocre across classes.** Says: not a class problem. Likely a localization or labeling problem. Move to 2c.
- **Mean FP-outside-ratio is high (> 0.3) on most classes.** Says: localization is the bottleneck — the lesion model is firing outside the gland. Fix candidates: cascade architecture (gland mask → lesion within ROI), gland-masked loss term, gland-aware augmentations.
- **Mean FP-outside-ratio is low (< 0.1) but Dice is still bad.** Says: localization is fine, but the model can't tell lesion from benign tissue inside the gland. Fix candidates: better lesion features (multi-modal fusion review, ADC weighting), harder positive mining, attention-style models.

Write down which of those four patterns you see before moving on. The audit queue can mislead you if you don't have the per-class story straight first.

### 2c. Audit queue — work the priorities top to bottom

Each row corresponds to a case where at least one heuristic fired. Click through to the linked per-case section and look at the slice grid.

#### Priority 1 cases — review first

- **`class_mask_inconsistent`** — fix or exclude immediately. A `class_label = 3` case with an empty lesion mask (or vice versa) is a data-pipeline bug, not a model failure. Don't burn cycles on the model until these are resolved.
- **`high_confidence_disagreement`** — open the per-case slices. Decide between:
  - The model is right and the GT missed a lesion → flag for re-annotation, expect Dice to rise after fix.
  - The model is wrong (false positive in a region that *looks* like lesion to the model) → keep the case, but it's now a positive signal for "real model error" — track it.

#### Priority 2 cases

- **`tiny_gt_island`** — usually annotation noise (a stray brushstroke). Strong candidate for cleanup. After you remove these tiny components from the GT, training Dice will rise; the rise is a measurement effect, not a real improvement, so don't celebrate.
- **`erratic_slice_consistency`** — a single annotated slice surrounded by gaps. Often the annotator labeled the obvious slice and didn't propagate. Decide whether to expand the annotation or accept it.
- **`class_severity_mismatch`** — the model predicts very differently from what the class label suggests. Look at the case. If the model is right, the class label may be wrong. If the model is wrong, the case is a hard negative worth keeping.

#### Priority 3 cases — context, not action

- **`gt_volume_outlier`** — extreme cases pulling Dice in unhelpful directions. Useful as context when interpreting per-class numbers. Not a bug.

### 2d. Worst cases without flags — your real model errors

This section is the most useful one for planning the **next modeling fix**. These are cases where the labels look fine and the model is genuinely struggling.

For each: open the slices and write a one-sentence diagnosis in your head. After 5 cases, you'll usually see a pattern (e.g. "all of these are small lesions in the transition zone", "all of these have low ADC contrast", "all of these are at the apex of the prostate"). That pattern is the hypothesis for the next sweep.

## Step 3 — Decide the next move

After reading the report, you should be able to fill in this paragraph:

> The model's bottleneck is **[localization | discrimination | per-class | label noise]**. Specifically, **[short observation from per-class table or worst-cases section]**. The next intervention should be **[concrete change]**, expected to improve **[which metric]** by addressing **[which failure mode]**.

If you can't fill this in, the report has not given you what you need — re-read the audit queue and the worst-cases section and refine the hypothesis. Don't start a new sweep until the paragraph is written.

## Step 4 — Act on what you found

Map each of the four bottleneck classes to a concrete next step:

| Bottleneck | Next step |
|---|---|
| Label noise dominates (many priority-1 / 2 audit cases) | Triage the audit queue with a clinician, fix or exclude flagged cases, re-train the **same** recipe on the cleaned set. Expected: Dice ceiling rises by 0.05–0.15. Run the diagnosis again on the new run. |
| Localization (high FP-outside-ratio) | Build a gland-localization cascade: train a gland-only model first, then constrain lesion training to the gland ROI. Expected: precision rises substantially, Dice may not move much because recall was already inside the gland. |
| Discrimination (FPs are inside the gland) | Improve lesion features. Audit modality usage (is ADC actually being used?), try harder negatives, attention modules. Slowest path; expect small gains per iteration. |
| Per-class collapse | Add class-aware loss weighting or per-class oversampling. Risk: degrades the strong class. Validate by running diagnosis on the new run and comparing the per-class table. |

## Step 5 — Re-run after the fix

After your intervention, run the diagnosis on the new checkpoint. Compare:

- The per-class table — did the targeted class actually move?
- Worst-cases-without-flags — is it the *same* set of cases or a different set? Same set means the fix didn't address the right failure mode.
- FP-outside-ratio per case — the metric should drop materially if you intervened on localization, and stay flat otherwise.

The diagnosis is a **loop**, not a one-time gate. Each run produces a hypothesis; each hypothesis produces a fix; the next diagnosis tests the fix.

## Common gotchas

- **Cached predictions go stale.** If you change the val split or the data on disk but reuse the same run directory, `prob.npz` files no longer match. Pass `--force` to re-dump, or `rm -rf <run_dir>/diagnostic/predictions/`.
- **The threshold matters.** `metrics_by_case.csv` uses the threshold from `resolved_config.yaml`'s `metrics.segmentation_threshold`. Some runs use the threshold-swept best operating point at training time but never persist it. If `metrics.segmentation_threshold` is missing, the tool defaults to 0.5 and warns. The numbers in the report may differ from what you remember from training in that case — this is expected, not a bug.
- **Empty audit CSVs are not failures.** If `label_audit.csv` is empty, that's information: the heuristics found nothing suspicious. Read the worst-cases-without-flags section instead.
- **Class buckets with fewer than 20 cases skip the cohort heuristics.** Short val sets (per the spec: ~20–60 cases) may not produce any cohort-level findings. That's by design — small samples make percentile cuts meaningless.

## See also

- [`diagnostic.md`](diagnostic.md) — CLI reference (flags, outputs, heuristic table).
- [`current_leader.md`](current_leader.md) — current segmentation leader to run the diagnosis against.
