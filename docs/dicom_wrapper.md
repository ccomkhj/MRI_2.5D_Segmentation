# DICOM Wrapper

`mri/service/pipeline.py` provides a single-call entry point that goes from a raw vendor DICOM zip to segmentation outputs plus a self-contained HTML report. It is meant for quick visual inspection of one study at a time, not for benchmark or research runs.

Internally it reuses the sibling `tcia-handler` preprocessing (`dicom_mapper.cli.pipeline._process_single_vendor_zip` + `tools/generate_training_metadata.py`) to produce an aligned PNG tree and `metadata.json`, then runs the normal `mri.inference.segmentation.run_segmentation_inference` with a caller-provided checkpoint.

## When To Use It

- **Use this wrapper** when the input is a raw vendor DICOM zip and you want one-shot segmentation output + an HTML report.
- **Use `mri/cli/infer.py`** (see [inference.md](inference.md)) when the data is already in `aligned_v2` form with a dated split, or whenever you need the two-stage segmentation → classification research loop.

## Prerequisites

Install the sibling `tcia-handler` repository as an editable package so `dicom_mapper` is importable:

```bash
pip install -e /Users/huijokim/personal/tcia-handler
```

That pulls in `pydicom`, `SimpleITK`, `highdicom`, `click`, and `tqdm`. Fresh installs may bump `numpy` past the version pinned in `requirements.txt`; torch stays compatible with numpy 2.x, but if you see ABI errors, reinstall a torch build matching the new numpy.

If the editable install is not possible, the wrapper falls back to adding `/Users/huijokim/personal/tcia-handler` to `sys.path` automatically. The editable install is preferred because it makes the dependency explicit.

## Python Usage

```python
from mri.service.pipeline import run_dicom_segmentation

result = run_dicom_segmentation(
    zip_path="/Users/huijokim/personal/tcia-handler/test_sample/PHilips AMbition 1,5T GL7B.zip",
    checkpoint_path="checkpoints/seg-auto-seg-apr17-24h-20260417T113109Z-w11-r01-simple-s5-prec-md1-w1-cons",
    output_dir="/tmp/mri_wrapper_smoke",
)
print(result)
```

Then open the report:

```bash
open /tmp/mri_wrapper_smoke/report.html
```

## Batch Usage

For multiple zips, use `run_dicom_segmentation_batch`. It accepts a directory (auto-globs `*.zip`), a list of zip paths, or a single zip, runs each case into its own `<output_dir>/<case_stem>/`, and writes a top-level `index.html` dashboard that links to every per-case report.

```python
from mri.service.pipeline import run_dicom_segmentation_batch

r = run_dicom_segmentation_batch(
    zip_paths="/Users/huijokim/personal/tcia-handler/test_sample",
    checkpoint_path="checkpoints/seg-auto-seg-apr17-24h-20260417T113109Z-w11-r01-simple-s5-prec-md1-w1-cons",
    output_dir="/tmp/mri_wrapper_batch",
)
print("ok:", r["num_ok"], "failed:", r["num_failed"], "->", r["index_html"])
```

Then:

```bash
open /tmp/mri_wrapper_batch/index.html
```

Batch-specific behaviour:

- Each case gets its own isolated output tree (`<case_stem>/_aligned/`, `<case_stem>/predictions/`, `<case_stem>/report.html`). Nothing is shared or overwritten between cases.
- `continue_on_error=True` (default) captures failures as red rows in `index.html` and lets the batch keep going. Pass `continue_on_error=False` to stop on the first exception.
- The checkpoint is resolved once and passed to every case, so the model config is loaded from disk a single time.

Returned dict: `output_dir`, `index_html`, `num_ok`, `num_failed`, `results` (list of per-case dicts matching `run_dicom_segmentation`'s return value), `errors` (list of `{zip, error}` for failed cases).

## Function Signature

```python
run_dicom_segmentation(
    zip_path,
    checkpoint_path,
    output_dir,
    *,
    resolved_config_path=None,
    threshold=None,
    group_name="case_auto",
)
```

- `zip_path`: path to a vendor DICOM zip, e.g. the samples under `tcia-handler/test_sample/`.
- `checkpoint_path`: either a training run directory or a direct `.pt` file (see next section).
- `output_dir`: destination for `_aligned/`, `predictions/`, and `report.html`.
- `resolved_config_path`: optional explicit path to a `resolved_config.yaml`. Required only when `checkpoint_path` points at a `.pt` that has no sibling `resolved_config.yaml`.
- `prostate_threshold`: probability threshold for the prostate mask. Defaults to `metrics.segmentation_threshold` from the checkpoint config (typically `0.5`).
- `target_thresholds`: sweep of target thresholds precomputed into the report. Defaults to `(0.1, 0.2, 0.3, 0.5, 0.7)`. The HTML slider flips between these values client-side without rerunning inference.
- `default_target_threshold`: initial slider position, and the threshold used to populate `target_slices` in the returned dict and the batch `index.html`. Defaults to `summary.best_val_metrics.threshold_sweep_target_best_threshold` from the checkpoint's `run_summary.json` (falls back to `min(target_thresholds)`).
- `threshold`: legacy single-threshold override. If given, it is applied to both classes and the sweep collapses to that one value (slider disabled).
- `group_name`: subdirectory name under `_aligned/` used to scope this auto-preprocessed case. The synthesised `case_id` becomes `"{group_name}/{case_name}"`.

### Why the target sweep

This checkpoint's training `threshold_sweep` picked `0.1` as the best target threshold (`val/dice_target` went from `0.13` at `0.5` to `0.29` at `0.1`). The wrapper reads that value from `run_summary.json` and uses it as the default, but keeps the surrounding thresholds embedded so you can confirm visually that the choice is reasonable for your data without rerunning anything. Prostate detection is robust at `0.5` so it is fixed, not swept.

## `checkpoint_path` Forms

- **Run directory**: the wrapper auto-discovers `*_best.pt` (falling back to `*_last.pt`) and loads `resolved_config.yaml` from the same directory. This is the default way to reference trained runs.
- **Direct `.pt` file**: the wrapper uses the provided weights. Pass `resolved_config_path=` or make sure a `resolved_config.yaml` lives next to the checkpoint.

## What It Writes

```
<output_dir>/
  _aligned/
    metadata.json                                 # synthesised for this one case
    split.yaml                                    # single-case test split
    <group_name>/<case_name>/
      t2/NNNN.png
      adc/NNNN.png
      calc/NNNN.png
      mask_prostate/NNNN.png                      # zero masks
      mask_target1/NNNN.png                       # zero masks
  predictions/
    <case_id>/
      prostate_prob.npy                           # (num_slices, H, W) float32
      target_prob.npy                             # (num_slices, H, W) float32
      overlays/NNNN.png                           # one per slice predicted by the dataset
  report.html                                     # single-file; 1 overlay/slice × target_threshold, ~7-12 MB
```

## Return Value

`run_dicom_segmentation` returns a dict:

- `case_id`: `"{group_name}/{case_name}"`
- `output_dir`, `staging_dir`, `predictions_dir`, `overlays_dir`, `html_report_path`: absolute paths
- `prostate_slices`: list of slice indices where `prostate_prob.max() >= prostate_threshold`
- `target_slices`: list of slice indices where `target_prob.max() >= default_target_threshold`
- `target_slices_by_threshold`: dict `{threshold (str): [slice indices]}` for every value in the sweep
- `num_slices`: total slices in the volume
- `prostate_threshold`, `target_thresholds`, `default_target_threshold`: the values actually used
- `checkpoint`: resolved `.pt` path
- `summary`: the raw summary dict from `run_segmentation_inference` (`cases_written`, `num_samples`, `overlay_pngs_written`, `segmentation_threshold`, `mean_dice`, etc.)

## HTML Report

Self-contained single file. Structure:

- Header summary: `case_id`, source zip, checkpoint, prostate threshold, target threshold sweep, default target threshold, total slices, prostate-positive slice ranges, target-positive slice ranges at **every** swept threshold, generated timestamp.
- Threshold slider at the top: flips every slice card between precomputed target-threshold variants in the browser. No rerun needed.
- Slice grid: one card per slice. Each card carries one embedded overlay per swept target threshold; the slider toggles which is visible.
- Yellow overlay = prostate mask (≥ `prostate_threshold`), red overlay = target mask (≥ slider-selected threshold).
- Green/cyan contour overlays (aligned_v2 flow only): drawn as a 1-px outline when GT masks exist in `mask_prostate/` / `mask_target1/`. Green = GT prostate, cyan = GT target. Hide/show with the **show ground truth** checkbox next to the slider.
- Card border colour: red if target is detected at the current slider value, yellow if only prostate is detected, plain otherwise. The border updates live as the slider moves.

## Aligned Data (no DICOM)

`run_aligned_segmentation(_batch)` runs the same inference + report flow on cases already in `aligned_v2` PNG form. It skips the DICOM preprocessing step entirely — the case's `metadata.json` entry and `t2/`, `adc/`, `calc/`, `mask_prostate/`, `mask_target1/` directories are used directly.

```python
from mri.service.pipeline import run_aligned_segmentation_batch

r = run_aligned_segmentation_batch(
    case_selector="class4",   # str prefix, single case_id, or list of case_ids
    checkpoint_path="checkpoints/seg-auto-seg-apr17-24h-20260417T113109Z-w11-r01-simple-s5-prec-md1-w1-cons",
    output_dir="/tmp/mri_aligned_batch",
    metadata_path="data/aligned_v2/metadata.json",  # default
)
print("ok:", r["num_ok"], "->", r["index_html"])
```

Key differences vs the DICOM flow:

- `case_selector` is a case_id prefix (`"class4"`), a single id (`"class4/case_0006"`), or a list of ids. The function walks `metadata.json` to expand the selector.
- No `_aligned/` staging tree is written — the existing `data/aligned_v2` layout is used as-is.
- GT masks auto-light up in the HTML report (green/cyan contours) with the checkbox to toggle them. Turn off when comparing to a colleague who shouldn't see labels.
- The resolved config and split YAML are still materialised into `<output_dir>/split.yaml` so the run is fully reproducible.

## Common Failure Modes

`ImportError: dicom-mapper not importable. Install with: pip install -e /Users/huijokim/personal/tcia-handler`

- The sibling repo is missing or not importable and the `sys.path` fallback did not find it. Install it editable.

`RuntimeError: DICOM preprocessing failed for <zip>`

- `tcia-handler` could not classify the series. The most common cause is `No T2 series found` in the log — the zip does not contain a recognisable T2 series. Inspect the zip contents or extend `dicom_mapper.io.vendor.classify_series`.

`ValueError: resolved_config_path is required when checkpoint_path points to a .pt file ...`

- Pass a run directory as `checkpoint_path`, or pass `resolved_config_path=` explicitly.

Checkpoint vs config mismatch (e.g. `SimpleUNet` weights with a `SegResNet` config)

- The wrapper always loads the `resolved_config.yaml` colocated with the checkpoint, so this is usually avoided. If you override `resolved_config_path=`, make sure it matches the architecture the checkpoint was trained with.

## Short Takeaway

This wrapper is the shortest path from a raw vendor DICOM zip to a human-readable segmentation result. For the full seg → cls research loop on `aligned_v2`, continue to use [inference.md](inference.md) and `mri/cli/infer.py`.
