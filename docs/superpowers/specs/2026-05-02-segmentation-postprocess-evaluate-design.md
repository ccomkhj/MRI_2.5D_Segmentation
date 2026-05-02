# Segmentation postprocess + per-lesion evaluation — design

**Status:** approved (brainstorming)
**Date:** 2026-05-02
**Scope:** analysis-only. No changes to training, inference, the clinician HTML
report, or downstream classification.

## Goal

Add two CLIs that operate on a finished segmentation run directory:

1. `mri.cli.postprocess` — apply two anatomy-aware masking rules to the
   predicted lesion masks.
2. `mri.cli.evaluate` — score the postprocessed predictions with a per-lesion
   detection criterion (3D-connected GT components, `IoU > 0.1` on any slice
   the component spans) and a negative-case false-alarm criterion (predicted
   area `> 2%` of any slice).

Both CLIs consume the per-case dump artifacts already produced by
`mri/diagnostics/dump.py:dump_predictions` (used today by `mri.cli.diagnose`).
Outputs are CSVs, a `summary.json`, and an interactive 3D Plotly HTML per case.

## Why

The existing voxel-level Dice / precision / recall metrics tell us how
well-aligned predictions are at the voxel scale, but not whether each
clinically-relevant lesion was found. Two recurring failure modes are also
unaddressed:

- **Lesion predicted outside the prostate** — anatomically impossible; should
  be masked away before scoring.
- **Lesion predicted in a case where no prostate is detected** — high-confidence
  noise; should be suppressed entirely.

Together, the postprocess rules + per-lesion detection metric give a metric
the clinical workflow cares about: "did the model find each lesion that was
there, and did it stay quiet on healthy cases?"

## Non-goals

- No change to `mri/inference/segmentation.py`, the clinician web UI, or the
  HTML clinician report. Inference outputs and downstream classification
  consume the raw predictions as today.
- No threshold sweeping over `--correctness-iou` or `--negative-area-frac`. Both
  are CLI flags; sweeps can be wrapped externally if needed (re-running
  `evaluate` is cheap because it consumes cached postprocessed masks).
- No tiny-component noise filter at postprocess time. The three rules are
  voxel-wise; per-component cleanup is deferred until evidence warrants it.

## High-level architecture

```
<run_dir>/
  *_best.pt                        # already exists
  resolved_config.yaml             # already exists
  diagnostic/
    predictions/<case>/            # produced by mri.diagnostics.dump.dump_predictions
      prob.npz                     # gland_prob, lesion_prob (Z,H,W float32)
      gt.npz                       # gland_gt,   lesion_gt   (Z,H,W uint8)
      meta.json                    # case_id, class_label, num_slices, thresholds
    postprocessed/<case>/          # NEW — produced by mri.cli.postprocess
      lesion_mask.npz              # gland-constrained, no-gland-suppressed (Z,H,W uint8)
      gland_mask.npz               # binarized gland (Z,H,W uint8)
      meta.json                    # thresholds, gland_present, voxel counts
    evaluation/                    # NEW — produced by mri.cli.evaluate
      metrics_by_lesion.csv        # one row per 3D GT lesion
      metrics_by_case.csv          # per-case detection roll-up
      summary.json                 # cohort lesion-recall, negative accuracy, params
      visuals/
        <case_id>.html             # interactive 3D Plotly figure
        index.html                 # gallery linking to per-case HTMLs
```

**Module layout (new):**

| Module | Purpose | Pure NumPy? |
|---|---|---|
| `mri/diagnostics/postprocess.py` | `apply_postprocess(lesion_prob, gland_prob, *, lesion_threshold, gland_threshold) -> (lesion_mask, gland_mask, gland_present)` implementing rules 1+2 | yes |
| `mri/diagnostics/detection.py` | 3D connected-component labeling, per-lesion per-slice IoU, negative-case 2%-area rule, dataclasses + CSV writers | yes (scipy.ndimage for CC) |
| `mri/diagnostics/visualization.py` | `build_case_figure(...) -> plotly.graph_objects.Figure`, `write_case_html(fig, path, *, header_meta, use_cdn)` | NumPy + Plotly |
| `mri/cli/postprocess.py` | CLI: ensure dump cache → apply postprocess per case → write `postprocessed/` | — |
| `mri/cli/evaluate.py` | CLI: read postprocessed + GT → write CSVs, summary, visuals | — |

`mri/cli/diagnose.py` and the existing diagnostics modules (`attribute.py`,
`audit.py`, `dump.py`, `report.py`) are unchanged.

**New dependency:** `plotly>=5.18` added to `pyproject.toml` and
`requirements.txt`.

## Postprocess rules (CLI 1)

Both rules are voxel-wise. Multi-lesion cases are handled implicitly: each
predicted lesion voxel is masked the same way regardless of which connected
component it belongs to.

```python
def apply_postprocess(
    lesion_prob: np.ndarray,    # (Z,H,W) float
    gland_prob: np.ndarray,     # (Z,H,W) float
    *,
    lesion_threshold: float,
    gland_threshold: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    gland_mask  = (gland_prob  >= gland_threshold).astype(np.uint8)
    lesion_mask = (lesion_prob >= lesion_threshold).astype(np.uint8)

    gland_present = bool(gland_mask.any())
    if not gland_present:
        # Rule 2: no prostate detected ⇒ all target ignored.
        lesion_mask = np.zeros_like(lesion_mask)
    else:
        # Rule 1: target outside prostate is ignored.
        lesion_mask = lesion_mask & gland_mask

    return lesion_mask, gland_mask, gland_present
```

Threshold for the gland constraint comes from `--gland-threshold` →
`metrics.segmentation_threshold` in resolved config → `0.5` (warn). Same
precedence for the lesion threshold. Both thresholds are persisted into the
per-case `meta.json` for traceability.

### `mri.cli.postprocess`

```
uv run python -m mri.cli.postprocess <run_dir> [--split val] [--force] \
    [--lesion-threshold FLOAT] [--gland-threshold FLOAT] [--device DEV]
```

1. Resolve `<run_dir>` via the existing `mri.cli.diagnose.resolve_run_dir`.
2. Resolve thresholds (precedence above).
3. Ensure `<run_dir>/diagnostic/predictions/` exists. If missing or `--force`,
   build the val dataloader + model the same way `diagnose` does and call
   `dump_predictions`.
4. For each case under `predictions/`:
   - Load `prob.npz` → `gland_prob`, `lesion_prob`.
   - Run `apply_postprocess`.
   - Write `postprocessed/<case_id>/lesion_mask.npz`, `gland_mask.npz`,
     `meta.json`.

Per-case `meta.json` schema:

```json
{
  "case_id": "...",
  "lesion_threshold": 0.5,
  "gland_threshold": 0.5,
  "gland_present": true,
  "lesion_voxels_raw":   1234,
  "lesion_voxels_post":   987,
  "gland_voxels":       45678
}
```

**Edge cases:**

- `prob.npz` missing for a case (failed inference): skip with warning, keep
  going.
- `--force` regenerates both `predictions/` (via `dump_predictions(force=True)`)
  and `postprocessed/`.
- `lesion_prob.shape != gland_prob.shape`: assertion error (programming bug
  upstream).

## Evaluation rules (CLI 2)

Per-case algorithm:

1. Load GT (`lesion_gt`, `gland_gt`) from `predictions/<case>/gt.npz` and the
   postprocessed prediction (`lesion_mask`) from
   `postprocessed/<case>/lesion_mask.npz`.
2. 3D connected-component labeling on `lesion_gt` using
   `scipy.ndimage.label`. Default 6-connectivity in 3D
   (`scipy.ndimage.generate_binary_structure(3, 1)`); flag-overridable to 26
   (`...generate_binary_structure(3, 3)`).
3. **If GT has ≥ 1 component (positive case)**, for each component `k` spanning
   slices `S_k = {z : gt_lesion_k[z].any()}`:
   - For each `z ∈ S_k`, compute slice IoU on the full slice:
     - `inter = (gt_lesion_k[z] & pred_lesion[z]).sum()`
     - `union = (gt_lesion_k[z] | pred_lesion[z]).sum()`
     - `iou_z = inter / union if union > 0 else 0.0`
   - `max_slice_iou_k = max(iou_z for z in S_k)`
   - `argmax_slice_k = z` achieving the max (lowest `z` on ties)
   - `detected_k = (max_slice_iou_k > correctness_iou)` — strict `>`.
   - Emit one row per component to `metrics_by_lesion.csv`.
4. **If GT has zero components (negative case)**:
   - For each slice `z`, `pred_area_frac_z = pred_lesion[z].sum() / (H*W)`.
   - `max_pred_area_frac = max(pred_area_frac_z for z)`.
   - `negative_correct = (max_pred_area_frac <= negative_area_frac)`. Strict
     `>` is the FALSE side, matching the user's "more than 2% ⇒ FALSE".
   - Emit one row to `metrics_by_case.csv` with `case_kind = "negative"` and
     no per-lesion rows.

### `mri.cli.evaluate`

```
uv run python -m mri.cli.evaluate <run_dir> \
    [--correctness-iou 0.1] [--negative-area-frac 0.02] \
    [--connectivity 6] \
    [--visualize-only all|failed|none] [--downsample-vis 1] [--plotly-cdn]
```

Reads `<run_dir>/diagnostic/postprocessed/` and
`<run_dir>/diagnostic/predictions/<case>/{gt.npz,meta.json}`. Errors clearly
if `postprocessed/` is missing (must run `mri.cli.postprocess` first). Writes
`<run_dir>/diagnostic/evaluation/`.

### CSV schemas

`metrics_by_lesion.csv` — one row per 3D GT lesion in the cohort:

| column | type | meaning |
|---|---|---|
| `case_id` | str | |
| `class_label` | int | from per-case `meta.json` |
| `lesion_id` | int | 1..K within the case (CC labeling order) |
| `lesion_voxels` | int | voxel count of this GT component |
| `slices` | str | `;`-joined `z` indices the component spans |
| `n_slices` | int | `len(S_k)` |
| `max_slice_iou` | float | `max_slice_iou_k` |
| `argmax_slice` | int | `argmax_slice_k` |
| `detected` | bool | `max_slice_iou > correctness_iou` |

`metrics_by_case.csv` — one row per case (positive or negative):

| column | type | meaning |
|---|---|---|
| `case_id` | str | |
| `class_label` | int | |
| `case_kind` | str | `"positive"` or `"negative"` |
| `n_gt_lesions` | int | `0` for negatives |
| `n_detected_lesions` | int | `0` for negatives |
| `lesion_recall` | float | `n_detected / n_gt_lesions`; empty for negatives |
| `max_pred_area_frac` | float | empty for positives |
| `negative_correct` | bool | empty for positives, `True`/`False` for negatives |

**CSV NaN convention.** "Empty" means the cell is written as the empty string
(Python's `csv` default for `None`); downstream readers (`pandas.read_csv`,
etc.) interpret it as NaN. No `nan` literal is written.

`summary.json`:

```json
{
  "params": {
    "correctness_iou": 0.1,
    "negative_area_frac": 0.02,
    "connectivity": 6,
    "lesion_threshold": 0.5,
    "gland_threshold": 0.5
  },
  "positives": {
    "n_cases": 42,
    "n_gt_lesions": 57,
    "n_detected_lesions": 49,
    "lesion_recall": 0.8596
  },
  "negatives": {
    "n_cases": 18,
    "n_correct": 15,
    "negative_accuracy": 0.8333
  },
  "cases_skipped": []
}
```

### Edge cases

- A GT component that is a single voxel: still labeled, still evaluated.
  Acceptable — matches "≥ 1 slice with IoU > 0.1".
- Empty postprocessed prediction on a positive case: every IoU is 0 ⇒ all
  components FALSE.
- A case present in `predictions/` but missing from `postprocessed/`: logged
  to `summary.json:cases_skipped`, not counted in metrics.
- Boundary semantics:
  - `max_slice_iou == 0.1` ⇒ `detected = False` (strict `>`).
  - `max_pred_area_frac == 0.02` ⇒ `negative_correct = True` (strict `>` on
    the FALSE side).

## Interactive 3D visualization

Per case, `mri.cli.evaluate` writes a self-contained
`evaluation/visuals/<case_id>.html` containing a single rotatable Plotly
scene with three kinds of `go.Isosurface` traces:

| trace | source | color | opacity | toggleable |
|---|---|---|---|---|
| GT gland | `gland_gt` | pale yellow | 0.15 | yes |
| GT lesion (per 3D component) | each CC component | green if detected, gray if not | 0.55 | yes (per-component) |
| Predicted lesion (postprocessed) | `lesion_mask.npz` | red | 0.45 | yes |

Each GT-lesion component is its own trace so the legend doubles as a
per-lesion verdict view.

**Coordinate system.** Axes are array indices `(z, y, x)` with
`aspectmode='data'`. No physical-spacing transform — all arrays are on the
resampled grid the model trained on; the goal is qualitative inspection, not
mm-accurate rendering.

**Header bar** above the figure (HTML, not part of the Plotly figure):
`case_id`, `class_label`, `n_gt_lesions`, `n_detected_lesions`,
`lesion_recall`, `negative_correct` (whichever applies).

**`evaluation/visuals/index.html`** — gallery: a sortable table linking
to each case's HTML, with the headline columns from `metrics_by_case.csv`.

**Performance / scale knobs:**

- `--visualize-only none` — skip rendering entirely; CSVs and `summary.json`
  still produced.
- `--visualize-only failed` — only render cases with any missed lesion or
  `negative_correct = False`.
- `--visualize-only all` (default).
- `--downsample-vis K` — int stride applied to voxel grids before isosurface
  extraction. Default `1` (no downsample).
- `--plotly-cdn` — emit Plotly via CDN reference instead of inlining
  (smaller HTML, requires network at view time).

**Module layout for visualization:**

- `mri/diagnostics/visualization.py`:
  - `build_case_figure(*, gt_gland, gt_lesion_components, pred_lesion, downsample) -> plotly.graph_objects.Figure`
  - `write_case_html(fig, path, *, header_meta, use_cdn) -> None`
  - `write_index_html(case_summaries, path) -> None`

  Pure functions over NumPy arrays; the CLI is responsible for I/O.

**Edge cases:**

- Negative case (no GT lesion components) → only GT gland + predicted lesion
  traces; `negative_correct` shown in the title.
- Empty predicted lesion → predicted-lesion trace omitted (Plotly's
  Isosurface needs ≥ 1 voxel above threshold).
- Empty GT gland → gland trace omitted; lesion traces still rendered.

## Testing

Pure-NumPy modules — exhaustive unit tests, no fixtures needed:

**`tests/diagnostics/test_postprocess.py`:**

- Empty gland → lesion fully zeroed (rule 2).
- Lesion partly outside gland → outside voxels removed, inside kept (rule 1).
- Lesion entirely inside gland → unchanged.
- Lesion entirely outside gland → fully zeroed.
- Multi-lesion case → both lesions independently masked.
- Threshold edges: prob exactly equal to threshold uses `>=`.

**`tests/diagnostics/test_detection.py`:**

- 3D CC labeling: two GT lesions on adjacent slices, spatially disjoint → 2
  components; one lesion across 3 slices → 1 component; touching diagonally
  only → connectivity flag (6 vs 26) gives different counts.
- Per-lesion IoU: hand-built `(Z,H,W)` GT + pred where `max_slice_iou = 0.15`
  → `detected = True` at `correctness_iou = 0.1`, `False` at `0.2`.
- Multi-lesion case: one detected, one missed → CSV has 2 rows with the
  correct `detected` flags; case-level `lesion_recall == 0.5`.
- Negative case: pred covers 1.5% on every slice →
  `negative_correct = True` at `negative_area_frac = 0.02`; pred covers 3% on
  one slice → `negative_correct = False`.
- Boundaries: `max_slice_iou == 0.1` → `detected = False`. Pred area exactly
  `0.02` → `negative_correct = True`.
- All-zero pred on a positive case → `max_slice_iou = 0`, `detected = False`
  for every component.
- All-zero pred on a negative case → `max_pred_area_frac = 0`,
  `negative_correct = True`.

**`tests/diagnostics/test_visualization.py`:**

- Build a tiny `(8,8,8)` synthetic volume with one GT component + a partly-
  overlapping predicted-lesion. Assert `build_case_figure` returns a Figure
  with the expected number of traces and legend names.
- Negative case → no green/gray lesion traces; predicted-lesion trace still
  present if non-empty.
- `write_case_html` produces a non-empty file containing `"plotly"` and the
  case_id.

**CLI tests — mock the slow paths:**

**`tests/cli/test_postprocess_cli.py`:**

- Build a tiny synthetic `<run_dir>` with two cached
  `predictions/<case>/{prob.npz, gt.npz, meta.json}` (no model load needed).
  Run the CLI; assert
  `postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}` are
  created with expected contents.
- `--force` regenerates an existing `postprocessed/`.
- `prob.npz` missing for one case → that case skipped, others succeed, exit
  code 0.

**`tests/cli/test_evaluate_cli.py`:**

- Same synthetic `<run_dir>` extended with hand-crafted `postprocessed/`.
  Run CLI; assert `evaluation/{metrics_by_lesion.csv, metrics_by_case.csv,
  summary.json, visuals/index.html}` match expected counts.
- A positive case + a negative case → both flow paths covered, summary
  aggregates correctly.
- `--correctness-iou 0.5` reduces detected count vs the default.
- `--visualize-only failed` → only the failing case's HTML appears.
- `--visualize-only none` → no `visuals/` directory.
- `postprocessed/` missing entirely → exit non-zero with a clear message
  pointing at `mri.cli.postprocess`.

No GPU, no real model, no real data — every test runs in milliseconds on
CPU. Plotly figure tests use Plotly's own data-only inspection
(`fig.data`, `fig.layout`) rather than rendering.

## Out of scope

- Threshold sweep tooling (re-run `evaluate` for each setting; it's cheap).
- Tiny-component noise filter at postprocess time.
- Physical-space (mm) coordinate rendering in the Plotly figure.
- HTML report integration with `mri.cli.diagnose`'s `report.html` (the
  diagnose report stays focused on voxel-level error attribution; this work
  produces its own gallery).

## Open questions

None at this time.
