# Segmentation postprocess + per-lesion evaluation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two analysis-only CLIs (`mri.cli.postprocess`, `mri.cli.evaluate`) that gland-constrain a finished run's lesion predictions, score each 3D GT lesion by max-over-slices IoU, flag negative cases with > 2% predicted area, and emit an interactive 3D Plotly HTML per case.

**Architecture:** Pure-NumPy core in three new modules under `mri/diagnostics/` (`postprocess.py`, `detection.py`, `visualization.py`), driven by two new CLIs under `mri/cli/`. Both CLIs operate on the per-case dump artifacts already written by `mri/diagnostics/dump.py:dump_predictions` (which `mri.cli.diagnose` already invokes). No changes to training, inference, the clinician HTML report, or downstream classification.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.ndimage.label`), Plotly (new dependency), Pytest.

**Spec:** [docs/superpowers/specs/2026-05-02-segmentation-postprocess-evaluate-design.md](../specs/2026-05-02-segmentation-postprocess-evaluate-design.md)

---

## File structure

**New files:**

| Path | Responsibility |
|---|---|
| `mri/diagnostics/postprocess.py` | Pure-NumPy: `apply_postprocess(lesion_prob, gland_prob, *, lesion_threshold, gland_threshold) -> (lesion_mask, gland_mask, gland_present)` implementing rules 1+2. |
| `mri/diagnostics/detection.py` | Pure-NumPy: 3D CC labeling, per-lesion per-slice IoU, negative-case area rule, dataclasses (`LesionRow`, `CaseRow`), CSV/JSON writers. |
| `mri/diagnostics/visualization.py` | Plotly: `build_case_figure(...)`, `write_case_html(...)`, `write_index_html(...)`. |
| `mri/cli/postprocess.py` | CLI: resolve run dir → ensure dump cache → apply postprocess per case → write `postprocessed/`. |
| `mri/cli/evaluate.py` | CLI: read postprocessed + GT → write CSVs, summary, visuals. |
| `tests/test_diagnostics_postprocess.py` | Unit tests for `apply_postprocess`. |
| `tests/test_diagnostics_detection.py` | Unit tests for CC labeling, per-lesion IoU, evaluate_case, writers. |
| `tests/test_diagnostics_visualization.py` | Unit tests for Plotly figure construction + HTML writing. |
| `tests/test_postprocess_cli.py` | CLI tests for `mri.cli.postprocess`. |
| `tests/test_evaluate_cli.py` | CLI tests for `mri.cli.evaluate`. |

**Modified files:**

| Path | Change |
|---|---|
| `pyproject.toml` | Add `plotly>=5.18` to `dependencies`. |
| `requirements.txt` | Add `plotly>=5.18` (kept in sync though not the source of truth). |

---

## Task 1: Add Plotly dependency

**Files:**
- Modify: `pyproject.toml` (the `dependencies` array)
- Modify: `requirements.txt` (under "Additional utilities")

- [ ] **Step 1.1: Add plotly to `pyproject.toml`**

Open `pyproject.toml` and edit the `dependencies` list to include `plotly>=5.18` immediately after the existing `seaborn` line. The relevant slice should look like:

```toml
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.18",
    "loguru",
```

- [ ] **Step 1.2: Add plotly to `requirements.txt`**

Add a single line under the "Additional utilities" section, between `seaborn>=0.12.0` and `loguru`:

```
plotly>=5.18
```

- [ ] **Step 1.3: Sync the venv**

Run: `uv sync`
Expected: completes without error; `plotly` appears in the install summary or `uv.lock` is updated.

- [ ] **Step 1.4: Sanity-check the import**

Run: `uv run python -c "import plotly.graph_objects as go; print(go.Figure().to_json()[:20])"`
Expected: prints something like `{"data":[],"layout":`. Any traceback is a failure.

- [ ] **Step 1.5: Commit**

```bash
git add pyproject.toml requirements.txt uv.lock
git commit -m "deps: add plotly for interactive 3D segmentation visualization"
```

---

## Task 2: `apply_postprocess` core function

**Files:**
- Create: `mri/diagnostics/postprocess.py`
- Test: `tests/test_diagnostics_postprocess.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_diagnostics_postprocess.py` with the full content below.

```python
"""Unit tests for gland-constrained lesion postprocess (rules 1 + 2)."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.postprocess import apply_postprocess


def _zeros(shape=(3, 4, 4)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_no_gland_zeros_lesion_completely() -> None:
    # Rule 2: when there's no prostate, all target is ignored.
    lesion_prob = _zeros()
    lesion_prob[1, 2, 2] = 0.99
    gland_prob = _zeros()  # nothing above any reasonable threshold

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is False
    assert lesion_mask.sum() == 0
    assert gland_mask.sum() == 0
    assert lesion_mask.dtype == np.uint8
    assert gland_mask.dtype == np.uint8


def test_lesion_partly_outside_gland_is_clipped() -> None:
    # Rule 1: target outside prostate is ignored; inside is kept.
    lesion_prob = _zeros()
    lesion_prob[1, 1, 1] = 0.9   # inside gland
    lesion_prob[1, 3, 3] = 0.9   # outside gland
    gland_prob = _zeros()
    gland_prob[1, 0:2, 0:2] = 0.9  # 2x2 gland on slice 1

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask[1, 1, 1] == 1
    assert lesion_mask[1, 3, 3] == 0
    assert lesion_mask.sum() == 1


def test_lesion_entirely_inside_gland_is_unchanged() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 1, 1] = 0.8
    lesion_prob[1, 1, 2] = 0.8
    gland_prob = np.full(lesion_prob.shape, 0.9, dtype=np.float32)

    lesion_mask, _, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask.sum() == 2
    assert lesion_mask[0, 1, 1] == 1
    assert lesion_mask[1, 1, 2] == 1


def test_lesion_entirely_outside_gland_is_zeroed() -> None:
    lesion_prob = _zeros()
    lesion_prob[1, 3, 3] = 0.99
    gland_prob = _zeros()
    gland_prob[1, 0, 0] = 0.99

    lesion_mask, _, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask.sum() == 0


def test_multi_lesion_each_masked_independently() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 1, 1] = 0.9   # inside
    lesion_prob[2, 3, 3] = 0.9   # outside
    gland_prob = _zeros()
    gland_prob[0, 1, 1] = 0.9
    gland_prob[2, 0, 0] = 0.9

    lesion_mask, _, _ = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert lesion_mask[0, 1, 1] == 1
    assert lesion_mask[2, 3, 3] == 0


def test_threshold_edge_uses_greater_or_equal() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 0, 0] = 0.5  # exactly at threshold
    gland_prob = np.full(lesion_prob.shape, 0.5, dtype=np.float32)

    lesion_mask, gland_mask, _ = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert lesion_mask[0, 0, 0] == 1
    assert gland_mask[0, 0, 0] == 1


def test_shape_mismatch_raises() -> None:
    lesion_prob = np.zeros((3, 4, 4), dtype=np.float32)
    gland_prob = np.zeros((3, 4, 5), dtype=np.float32)

    import pytest
    with pytest.raises(AssertionError):
        apply_postprocess(
            lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
        )
```

- [ ] **Step 2.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_postprocess.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'mri.diagnostics.postprocess'`.

- [ ] **Step 2.3: Implement `apply_postprocess`**

Create `mri/diagnostics/postprocess.py` with this exact content:

```python
"""Anatomy-aware postprocessing of segmentation predictions.

Two voxel-wise rules:

1. Target voxels outside the predicted prostate are ignored.
2. When no prostate is detected anywhere, all target voxels are ignored.

Both rules consume the *predicted* gland (not the GT gland), so this
function is suitable for evaluation paths that mirror inference-time
decision making. Pure NumPy, no I/O.
"""

from __future__ import annotations

import numpy as np


def apply_postprocess(
    lesion_prob: np.ndarray,
    gland_prob: np.ndarray,
    *,
    lesion_threshold: float,
    gland_threshold: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Binarize and gland-constrain the lesion prediction.

    Args:
      lesion_prob: (Z, H, W) float array of lesion probabilities.
      gland_prob:  (Z, H, W) float array of gland probabilities.
      lesion_threshold: probability threshold for binarizing the lesion mask
          (uses ``>=``).
      gland_threshold: probability threshold for binarizing the gland mask
          (uses ``>=``).

    Returns:
      ``(lesion_mask, gland_mask, gland_present)`` where:
        - ``lesion_mask`` is a (Z, H, W) uint8 array with rules 1+2 applied.
        - ``gland_mask`` is a (Z, H, W) uint8 array (binarized gland, no
          masking applied to it).
        - ``gland_present`` is True iff any gland voxel passed the
          threshold; when False, ``lesion_mask`` is fully zeroed.
    """
    assert lesion_prob.shape == gland_prob.shape, (
        f"shape mismatch: lesion {lesion_prob.shape} vs gland {gland_prob.shape}"
    )

    gland_mask = (gland_prob >= gland_threshold).astype(np.uint8)
    lesion_mask = (lesion_prob >= lesion_threshold).astype(np.uint8)

    gland_present = bool(gland_mask.any())
    if not gland_present:
        lesion_mask = np.zeros_like(lesion_mask)
    else:
        lesion_mask = (lesion_mask & gland_mask).astype(np.uint8)

    return lesion_mask, gland_mask, gland_present
```

- [ ] **Step 2.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_postprocess.py -v`
Expected: 7 passed.

- [ ] **Step 2.5: Commit**

```bash
git add mri/diagnostics/postprocess.py tests/test_diagnostics_postprocess.py
git commit -m "feat(diagnostics): apply_postprocess (gland-constrain + no-gland-suppress)"
```

---

## Task 3: 3D connected-component labeling for GT lesions

**Files:**
- Create: `mri/diagnostics/detection.py` (start of file)
- Test: `tests/test_diagnostics_detection.py` (start of file)

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_diagnostics_detection.py` with:

```python
"""Unit tests for per-3D-lesion detection scoring."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.detection import label_lesion_components


def test_single_lesion_across_three_slices_is_one_component() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 2, 2] = 1
    gt[2, 2, 2] = 1
    gt[3, 2, 2] = 1

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 1
    assert labels.shape == gt.shape
    assert labels.dtype.kind == "i"
    assert (labels[gt == 1] == 1).all()
    assert (labels[gt == 0] == 0).all()


def test_two_disjoint_lesions_are_two_components() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 4, 4] = 1  # spatially disjoint on the same slice

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 2
    assert sorted(np.unique(labels[gt == 1]).tolist()) == [1, 2]


def test_diagonal_only_split_under_6_connectivity_joined_under_26() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 2, 2] = 1  # diagonal in-plane

    _, n6 = label_lesion_components(gt, connectivity_rank=1)
    _, n26 = label_lesion_components(gt, connectivity_rank=3)

    assert n6 == 2
    assert n26 == 1


def test_empty_gt_yields_zero_components() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 0
    assert (labels == 0).all()
```

- [ ] **Step 3.2: Run the test; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_detection.py::test_single_lesion_across_three_slices_is_one_component -v`
Expected: `ModuleNotFoundError: No module named 'mri.diagnostics.detection'`.

- [ ] **Step 3.3: Implement `label_lesion_components`**

Create `mri/diagnostics/detection.py` with:

```python
"""Per-3D-lesion detection scoring for postprocessed segmentation predictions.

Pure NumPy + scipy.ndimage. The CLI in mri/cli/evaluate.py is responsible
for I/O.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def label_lesion_components(
    gt_lesion: np.ndarray,
    *,
    connectivity_rank: int = 1,
) -> tuple[np.ndarray, int]:
    """3D connected-component labeling of a binary GT lesion volume.

    Args:
      gt_lesion: (Z, H, W) uint8 array; non-zero voxels are foreground.
      connectivity_rank: passed straight to
          ``scipy.ndimage.generate_binary_structure(3, rank)``. Use
          ``1`` for 6-connectivity (default) or ``3`` for 26-connectivity.

    Returns:
      ``(labels, n_components)`` where ``labels`` has the same shape as
      ``gt_lesion`` with components numbered 1..n (0 = background).
    """
    structure = ndimage.generate_binary_structure(3, connectivity_rank)
    labels, n = ndimage.label(gt_lesion.astype(bool), structure=structure)
    return labels.astype(np.int32), int(n)
```

- [ ] **Step 3.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_detection.py -v`
Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add mri/diagnostics/detection.py tests/test_diagnostics_detection.py
git commit -m "feat(diagnostics): 3D connected-component labeling for GT lesions"
```

---

## Task 4: Per-lesion per-slice IoU

**Files:**
- Modify: `mri/diagnostics/detection.py` (append)
- Modify: `tests/test_diagnostics_detection.py` (append)

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/test_diagnostics_detection.py`:

```python
from mri.diagnostics.detection import compute_lesion_iou


def test_lesion_iou_max_across_slices_with_argmax() -> None:
    # Component spans z=1..3. Pred overlaps best at z=2.
    component = np.zeros((5, 4, 4), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    component[2, 1, 2] = True
    component[3, 1, 1] = True

    pred = np.zeros((5, 4, 4), dtype=bool)
    pred[1, 1, 1] = True             # iou = 1/1 = 1.0  (single voxel exact)
    pred[2, 1, 1] = True             # iou = 1/2 on z=2 (component has 2 voxels)
    pred[3, 0, 0] = True             # iou = 0 on z=3

    result = compute_lesion_iou(component, pred)

    assert result.slices == (1, 2, 3)
    # z=1: 1/1, z=2: 1/2, z=3: 0/(1+1)=0
    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_argmax_breaks_ties_with_lowest_z() -> None:
    component = np.zeros((4, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    pred = np.zeros((4, 3, 3), dtype=bool)
    pred[1, 1, 1] = True
    pred[2, 1, 1] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_all_zero_pred_is_zero() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    pred = np.zeros((3, 3, 3), dtype=bool)

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.0
    assert result.argmax_slice == 1


def test_lesion_iou_partial_overlap_value() -> None:
    # Component on z=0 = 4 voxels. Pred on z=0 = 2 voxels overlapping. iou = 2/4 = 0.5
    component = np.zeros((1, 4, 4), dtype=bool)
    component[0, 1:3, 1:3] = True  # 4 voxels
    pred = np.zeros((1, 4, 4), dtype=bool)
    pred[0, 1, 1] = True
    pred[0, 1, 2] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.5
    assert result.argmax_slice == 0
```

- [ ] **Step 4.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_detection.py -k "lesion_iou" -v`
Expected: ImportError on `compute_lesion_iou`.

- [ ] **Step 4.3: Implement `compute_lesion_iou`**

Append to `mri/diagnostics/detection.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class LesionIoUResult:
    """Outcome of evaluating one 3D GT component against a prediction volume."""
    slices: tuple[int, ...]      # z indices the component spans, ascending
    max_slice_iou: float
    argmax_slice: int            # z index achieving max_slice_iou; lowest-z on ties


def compute_lesion_iou(
    component_mask: np.ndarray,
    pred_lesion_mask: np.ndarray,
) -> LesionIoUResult:
    """Per-slice IoU of one GT lesion component vs the full predicted lesion.

    For each slice z that the component spans, compute IoU between the
    component's slice mask and the *entire* predicted-lesion slice (no
    cropping). Returns the max IoU across those slices and the lowest z
    that achieves it.

    Args:
      component_mask: (Z, H, W) bool array with one connected GT component
          set to True; False elsewhere.
      pred_lesion_mask: (Z, H, W) bool/uint8 array of postprocessed
          predicted lesion voxels.

    Returns:
      ``LesionIoUResult`` with ``slices`` empty if the component is empty.
    """
    component = component_mask.astype(bool)
    pred = pred_lesion_mask.astype(bool)
    assert component.shape == pred.shape, (
        f"shape mismatch: component {component.shape} vs pred {pred.shape}"
    )

    z_axis = 0
    slice_has_component = component.any(axis=(1, 2))
    slices = tuple(int(z) for z in np.flatnonzero(slice_has_component))

    if not slices:
        return LesionIoUResult(slices=(), max_slice_iou=0.0, argmax_slice=0)

    max_iou = -1.0
    argmax_z = slices[0]
    for z in slices:
        gt_z = component[z]
        pr_z = pred[z]
        inter = int(np.logical_and(gt_z, pr_z).sum())
        union = int(np.logical_or(gt_z, pr_z).sum())
        iou = (inter / union) if union > 0 else 0.0
        if iou > max_iou:
            max_iou = iou
            argmax_z = z

    return LesionIoUResult(
        slices=slices,
        max_slice_iou=float(max_iou),
        argmax_slice=int(argmax_z),
    )
```

(Remove the unused `z_axis = 0` if you prefer; it's only a comment for the reader.)

- [ ] **Step 4.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_detection.py -v`
Expected: 8 passed (4 new + 4 from Task 3).

- [ ] **Step 4.5: Commit**

```bash
git add mri/diagnostics/detection.py tests/test_diagnostics_detection.py
git commit -m "feat(diagnostics): per-lesion per-slice IoU with argmax-slice tracking"
```

---

## Task 5: Per-case evaluation (positives + negatives)

**Files:**
- Modify: `mri/diagnostics/detection.py` (append)
- Modify: `tests/test_diagnostics_detection.py` (append)

- [ ] **Step 5.1: Write the failing tests**

Append to `tests/test_diagnostics_detection.py`:

```python
from mri.diagnostics.detection import (
    LesionRow, CaseRow, evaluate_case,
)


def test_evaluate_case_positive_two_lesions_one_detected() -> None:
    gt = np.zeros((4, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1   # lesion A — single voxel on z=1
    gt[2, 4, 4] = 1   # lesion B — single voxel on z=2

    pred = np.zeros((4, 6, 6), dtype=np.uint8)
    pred[1, 1, 1] = 1  # exact hit on lesion A → IoU = 1.0
    # nothing for lesion B

    case_row, lesion_rows = evaluate_case(
        case_id="c1", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert case_row.n_gt_lesions == 2
    assert case_row.n_detected_lesions == 1
    assert case_row.lesion_recall == 0.5
    assert case_row.max_pred_area_frac is None
    assert case_row.negative_correct is None

    assert len(lesion_rows) == 2
    detected_ids = {row.lesion_id for row in lesion_rows if row.detected}
    assert len(detected_ids) == 1


def test_evaluate_case_negative_below_threshold_is_correct() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    # 1 voxel of 100 = 1% on slice 0; below 2% ⇒ negative_correct = True
    pred[0, 0, 0] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c2", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "negative"
    assert case_row.n_gt_lesions == 0
    assert case_row.n_detected_lesions == 0
    assert case_row.lesion_recall is None
    assert case_row.max_pred_area_frac == 0.01
    assert case_row.negative_correct is True
    assert lesion_rows == []


def test_evaluate_case_negative_above_threshold_is_false() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    # 3 voxels of 100 = 3% on slice 0; > 2% ⇒ negative_correct = False
    pred[0, 0, 0:3] = 1

    case_row, _ = evaluate_case(
        case_id="c3", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is False
    assert case_row.max_pred_area_frac == 0.03


def test_evaluate_case_negative_at_threshold_is_correct() -> None:
    # Exactly 2% ⇒ TRUE (strict > on the FALSE side).
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    pred[0, 0, 0:2] = 1  # 2/100 = 0.02

    case_row, _ = evaluate_case(
        case_id="c4", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is True


def test_evaluate_case_positive_iou_at_threshold_is_not_detected() -> None:
    # Strict > on detection. iou = 0.1 exactly ⇒ detected = False.
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    gt[0, 0, 0:10] = 1   # row of 10 voxels
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    # Need iou exactly 0.1: intersection / union = 0.1.
    # Pred = 1 voxel inside GT, 8 voxels outside  ⇒ inter=1, union=10+8=18 → 0.0555
    # Easier: pred = 1 voxel inside, 0 outside  ⇒ inter=1, union=10 → 0.1 exactly.
    pred[0, 0, 0] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c5", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert lesion_rows[0].max_slice_iou == 0.1
    assert lesion_rows[0].detected is False
    assert case_row.n_detected_lesions == 0
```

- [ ] **Step 5.2: Run the tests; expect ImportError on the new symbols**

Run: `uv run pytest tests/test_diagnostics_detection.py -k "evaluate_case" -v`
Expected: ImportError.

- [ ] **Step 5.3: Implement the dataclasses + `evaluate_case`**

Append to `mri/diagnostics/detection.py`:

```python
@dataclass(frozen=True)
class LesionRow:
    """One row of metrics_by_lesion.csv (per 3D GT component)."""
    case_id: str
    class_label: int
    lesion_id: int
    lesion_voxels: int
    slices: str           # ";"-joined z indices the component spans
    n_slices: int
    max_slice_iou: float
    argmax_slice: int
    detected: bool


@dataclass(frozen=True)
class CaseRow:
    """One row of metrics_by_case.csv. Mixed positive/negative case schema.

    For positive cases: ``max_pred_area_frac`` and ``negative_correct`` are None.
    For negative cases: ``lesion_recall`` is None.
    Writers translate None to empty CSV cells.
    """
    case_id: str
    class_label: int
    case_kind: str        # "positive" | "negative"
    n_gt_lesions: int
    n_detected_lesions: int
    lesion_recall: float | None
    max_pred_area_frac: float | None
    negative_correct: bool | None


def evaluate_case(
    *,
    case_id: str,
    class_label: int,
    gt_lesion: np.ndarray,
    pred_lesion: np.ndarray,
    correctness_iou: float,
    negative_area_frac: float,
    connectivity_rank: int,
) -> tuple[CaseRow, list[LesionRow]]:
    """Score one case under the per-3D-lesion + negative-area rule.

    Returns:
      ``(case_row, lesion_rows)``. ``lesion_rows`` is empty for negative
      cases.
    """
    assert gt_lesion.shape == pred_lesion.shape, (
        f"shape mismatch: gt {gt_lesion.shape} vs pred {pred_lesion.shape}"
    )

    labels, n_components = label_lesion_components(
        gt_lesion, connectivity_rank=connectivity_rank,
    )

    if n_components == 0:
        # Negative case: max-over-slices predicted area fraction.
        Z, H, W = pred_lesion.shape
        per_slice_voxels = pred_lesion.astype(bool).reshape(Z, -1).sum(axis=1)
        per_slice_frac = per_slice_voxels / float(H * W)
        max_frac = float(per_slice_frac.max()) if Z > 0 else 0.0
        return (
            CaseRow(
                case_id=case_id,
                class_label=class_label,
                case_kind="negative",
                n_gt_lesions=0,
                n_detected_lesions=0,
                lesion_recall=None,
                max_pred_area_frac=max_frac,
                negative_correct=(max_frac <= negative_area_frac),
            ),
            [],
        )

    # Positive case: per-3D-component scoring.
    pred_bool = pred_lesion.astype(bool)
    lesion_rows: list[LesionRow] = []
    detected = 0
    for k in range(1, n_components + 1):
        component = (labels == k)
        ious = compute_lesion_iou(component, pred_bool)
        is_detected = ious.max_slice_iou > correctness_iou
        if is_detected:
            detected += 1
        lesion_rows.append(
            LesionRow(
                case_id=case_id,
                class_label=class_label,
                lesion_id=k,
                lesion_voxels=int(component.sum()),
                slices=";".join(str(z) for z in ious.slices),
                n_slices=len(ious.slices),
                max_slice_iou=ious.max_slice_iou,
                argmax_slice=ious.argmax_slice,
                detected=is_detected,
            )
        )

    return (
        CaseRow(
            case_id=case_id,
            class_label=class_label,
            case_kind="positive",
            n_gt_lesions=n_components,
            n_detected_lesions=detected,
            lesion_recall=detected / n_components,
            max_pred_area_frac=None,
            negative_correct=None,
        ),
        lesion_rows,
    )
```

- [ ] **Step 5.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_detection.py -v`
Expected: 13 passed (5 new + 8 from prior tasks).

- [ ] **Step 5.5: Commit**

```bash
git add mri/diagnostics/detection.py tests/test_diagnostics_detection.py
git commit -m "feat(diagnostics): per-case evaluation (positive lesion-recall + negative area rule)"
```

---

## Task 6: CSV + summary writers

**Files:**
- Modify: `mri/diagnostics/detection.py` (append)
- Modify: `tests/test_diagnostics_detection.py` (append)

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_diagnostics_detection.py`:

```python
import csv
import json
from pathlib import Path

from mri.diagnostics.detection import (
    write_lesion_csv, write_case_csv, build_summary, write_summary_json,
)


def _make_pos_rows() -> tuple[CaseRow, list[LesionRow]]:
    case = CaseRow(
        case_id="c1", class_label=2, case_kind="positive",
        n_gt_lesions=2, n_detected_lesions=1, lesion_recall=0.5,
        max_pred_area_frac=None, negative_correct=None,
    )
    rows = [
        LesionRow(case_id="c1", class_label=2, lesion_id=1, lesion_voxels=4,
                  slices="1;2", n_slices=2, max_slice_iou=0.42, argmax_slice=2,
                  detected=True),
        LesionRow(case_id="c1", class_label=2, lesion_id=2, lesion_voxels=3,
                  slices="3", n_slices=1, max_slice_iou=0.05, argmax_slice=3,
                  detected=False),
    ]
    return case, rows


def _make_neg_row() -> CaseRow:
    return CaseRow(
        case_id="c2", class_label=0, case_kind="negative",
        n_gt_lesions=0, n_detected_lesions=0, lesion_recall=None,
        max_pred_area_frac=0.015, negative_correct=True,
    )


def test_lesion_csv_columns_and_values(tmp_path: Path) -> None:
    _, rows = _make_pos_rows()
    out = tmp_path / "metrics_by_lesion.csv"

    write_lesion_csv(rows, out)

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert reader.fieldnames == [
        "case_id", "class_label", "lesion_id", "lesion_voxels",
        "slices", "n_slices", "max_slice_iou", "argmax_slice", "detected",
    ]
    assert records[0]["lesion_id"] == "1"
    assert records[0]["detected"] == "True"
    assert records[1]["detected"] == "False"


def test_case_csv_writes_empty_string_for_none(tmp_path: Path) -> None:
    case_pos, _ = _make_pos_rows()
    case_neg = _make_neg_row()
    out = tmp_path / "metrics_by_case.csv"

    write_case_csv([case_pos, case_neg], out)

    # Read raw lines to verify empty cells (no "None" or "nan" literals).
    text = out.read_text()
    assert "None" not in text
    assert "nan" not in text.lower()

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert records[0]["max_pred_area_frac"] == ""
    assert records[0]["negative_correct"] == ""
    assert records[1]["lesion_recall"] == ""
    assert records[0]["lesion_recall"] == "0.5"
    assert records[1]["max_pred_area_frac"] == "0.015"
    assert records[1]["negative_correct"] == "True"


def test_build_summary_aggregates_positive_and_negative(tmp_path: Path) -> None:
    case_pos, rows_pos = _make_pos_rows()
    case_neg = _make_neg_row()

    summary = build_summary(
        case_rows=[case_pos, case_neg],
        lesion_rows=rows_pos,
        params={
            "correctness_iou": 0.1, "negative_area_frac": 0.02,
            "connectivity": 6, "lesion_threshold": 0.5, "gland_threshold": 0.5,
        },
        cases_skipped=[],
    )

    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["lesion_recall"] == 0.5
    assert summary["negatives"]["n_cases"] == 1
    assert summary["negatives"]["n_correct"] == 1
    assert summary["negatives"]["negative_accuracy"] == 1.0
    assert summary["params"]["correctness_iou"] == 0.1
    assert summary["cases_skipped"] == []


def test_write_summary_json_round_trip(tmp_path: Path) -> None:
    summary = {"params": {"correctness_iou": 0.1}, "positives": {"n_cases": 0}}
    out = tmp_path / "summary.json"

    write_summary_json(summary, out)

    loaded = json.loads(out.read_text())
    assert loaded == summary
```

- [ ] **Step 6.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_detection.py -k "csv or summary" -v`
Expected: ImportError.

- [ ] **Step 6.3: Implement the writers**

Append to `mri/diagnostics/detection.py`:

```python
import csv as _csv
import json as _json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _empty_for_none(value: Any) -> Any:
    """Translate None to empty string for CSV cells. Other types pass through."""
    return "" if value is None else value


def write_lesion_csv(rows: Iterable[LesionRow], path: Path) -> None:
    """Write metrics_by_lesion.csv. Empty list ⇒ header-only file."""
    rows = list(rows)
    fieldnames = [
        "case_id", "class_label", "lesion_id", "lesion_voxels",
        "slices", "n_slices", "max_slice_iou", "argmax_slice", "detected",
    ]
    with Path(path).open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_case_csv(rows: Iterable[CaseRow], path: Path) -> None:
    """Write metrics_by_case.csv. None values become empty cells."""
    rows = list(rows)
    fieldnames = [
        "case_id", "class_label", "case_kind",
        "n_gt_lesions", "n_detected_lesions", "lesion_recall",
        "max_pred_area_frac", "negative_correct",
    ]
    with Path(path).open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            writer.writerow({k: _empty_for_none(v) for k, v in d.items()})


def build_summary(
    *,
    case_rows: Iterable[CaseRow],
    lesion_rows: Iterable[LesionRow],
    params: Mapping[str, Any],
    cases_skipped: Iterable[str],
) -> dict[str, Any]:
    """Aggregate cohort metrics into the summary.json shape."""
    case_rows = list(case_rows)
    lesion_rows = list(lesion_rows)

    pos_cases = [c for c in case_rows if c.case_kind == "positive"]
    neg_cases = [c for c in case_rows if c.case_kind == "negative"]

    n_gt_lesions = sum(c.n_gt_lesions for c in pos_cases)
    n_detected = sum(c.n_detected_lesions for c in pos_cases)
    lesion_recall = (n_detected / n_gt_lesions) if n_gt_lesions > 0 else 0.0

    n_neg_correct = sum(1 for c in neg_cases if c.negative_correct)
    neg_accuracy = (n_neg_correct / len(neg_cases)) if neg_cases else 0.0

    return {
        "params": dict(params),
        "positives": {
            "n_cases": len(pos_cases),
            "n_gt_lesions": n_gt_lesions,
            "n_detected_lesions": n_detected,
            "lesion_recall": lesion_recall,
        },
        "negatives": {
            "n_cases": len(neg_cases),
            "n_correct": n_neg_correct,
            "negative_accuracy": neg_accuracy,
        },
        "cases_skipped": list(cases_skipped),
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    Path(path).write_text(_json.dumps(summary, indent=2))
```

- [ ] **Step 6.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_detection.py -v`
Expected: 17 passed (4 new + 13 from prior tasks).

- [ ] **Step 6.5: Commit**

```bash
git add mri/diagnostics/detection.py tests/test_diagnostics_detection.py
git commit -m "feat(diagnostics): CSV + summary.json writers for per-lesion evaluation"
```

---

## Task 7: `mri.cli.postprocess` — argparse + run-dir resolution

**Files:**
- Create: `mri/cli/postprocess.py`
- Test: `tests/test_postprocess_cli.py`

- [ ] **Step 7.1: Write the failing tests**

Create `tests/test_postprocess_cli.py` with:

```python
"""Smoke tests for `python -m mri.cli.postprocess` argparse + run-dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mri.cli.postprocess import resolve_postprocess_thresholds


def test_resolve_thresholds_explicit_flags_win() -> None:
    cfg = {"metrics": {"segmentation_threshold": 0.7}}
    lesion, gland = resolve_postprocess_thresholds(
        cfg, lesion_arg=0.4, gland_arg=0.3,
    )
    assert lesion == 0.4
    assert gland == 0.3


def test_resolve_thresholds_falls_back_to_config() -> None:
    cfg = {"metrics": {"segmentation_threshold": 0.6}}
    lesion, gland = resolve_postprocess_thresholds(
        cfg, lesion_arg=None, gland_arg=None,
    )
    assert lesion == 0.6
    assert gland == 0.6


def test_resolve_thresholds_warns_when_no_config_value() -> None:
    cfg: dict = {}
    with pytest.warns(UserWarning, match="segmentation_threshold"):
        lesion, gland = resolve_postprocess_thresholds(
            cfg, lesion_arg=None, gland_arg=None,
        )
    assert lesion == 0.5
    assert gland == 0.5
```

- [ ] **Step 7.2: Run the test; expect ImportError**

Run: `uv run pytest tests/test_postprocess_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'mri.cli.postprocess'`.

- [ ] **Step 7.3: Implement the module skeleton**

Create `mri/cli/postprocess.py` with:

```python
"""CLI entry point for gland-constrained segmentation postprocessing.

Usage::

    python -m mri.cli.postprocess <run_dir> [--split val] [--force] \
        [--lesion-threshold FLOAT] [--gland-threshold FLOAT] [--device DEV]

Reads ``<run_dir>/diagnostic/predictions/<case>/prob.npz`` (running
``mri.diagnostics.dump.dump_predictions`` first if the cache is missing or
``--force`` is set) and writes
``<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Sequence


def resolve_postprocess_thresholds(
    cfg: dict,
    *,
    lesion_arg: float | None,
    gland_arg: float | None,
) -> tuple[float, float]:
    """Resolve lesion/gland thresholds with the precedence:
    flag > resolved_config[metrics.segmentation_threshold] > 0.5 (warn).
    """
    metrics_cfg = cfg.get("metrics") or {}
    cfg_threshold = metrics_cfg.get("segmentation_threshold")
    if cfg_threshold is None and (lesion_arg is None or gland_arg is None):
        warnings.warn(
            "No metrics.segmentation_threshold in resolved_config — "
            "falling back to 0.5 for missing flag(s).",
            stacklevel=2,
        )

    lesion = float(lesion_arg) if lesion_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    gland = float(gland_arg) if gland_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    return lesion, gland


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gland-constrain + no-gland-suppress postprocessing "
                    "to a finished run's segmentation predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if cached predictions exist.")
    parser.add_argument("--lesion-threshold", type=float, default=None)
    parser.add_argument("--gland-threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    raise NotImplementedError("Wired in Task 8.")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_postprocess_cli.py -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add mri/cli/postprocess.py tests/test_postprocess_cli.py
git commit -m "feat(cli): postprocess CLI scaffold with threshold resolution"
```

---

## Task 8: `mri.cli.postprocess` — per-case loop with cached `prob.npz`

**Files:**
- Modify: `mri/cli/postprocess.py`
- Modify: `tests/test_postprocess_cli.py`

- [ ] **Step 8.1: Write the failing tests**

Append to `tests/test_postprocess_cli.py`:

```python
import json

import numpy as np

from mri.cli import postprocess as postprocess_cli


def _seed_predictions(run_dir: Path, case_id: str, lesion_prob, gland_prob,
                      gt_lesion=None, gt_gland=None) -> None:
    """Write a synthetic prob.npz/gt.npz/meta.json for one case."""
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "prob.npz",
                         gland=gland_prob.astype(np.float32),
                         lesion=lesion_prob.astype(np.float32))
    if gt_lesion is None:
        gt_lesion = np.zeros_like(lesion_prob, dtype=np.uint8)
    if gt_gland is None:
        gt_gland = np.zeros_like(gland_prob, dtype=np.uint8)
    np.savez_compressed(pdir / "gt.npz",
                         gland=gt_gland.astype(np.uint8),
                         lesion=gt_lesion.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id, "class_label": 2,
        "spatial_shape": list(lesion_prob.shape[1:]),
        "num_slices": int(lesion_prob.shape[0]),
        "predicted_slices": list(range(int(lesion_prob.shape[0]))),
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
        "data:\n  metadata: dummy\n  split_file: dummy\n"
        "model:\n  name: simple_unet\n  params: {}\n"
    )
    return run_dir


def test_postprocess_cli_writes_artifacts_per_case(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    lesion_prob = np.zeros((3, 4, 4), dtype=np.float32)
    gland_prob = np.zeros((3, 4, 4), dtype=np.float32)
    lesion_prob[1, 1, 1] = 0.9
    lesion_prob[1, 3, 3] = 0.9
    gland_prob[1, 0:2, 0:2] = 0.9
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)

    rc = postprocess_cli.main([str(run_dir)])

    assert rc == 0
    pdir = run_dir / "diagnostic" / "postprocessed" / "case_a"
    lesion = np.load(pdir / "lesion_mask.npz")["mask"]
    gland = np.load(pdir / "gland_mask.npz")["mask"]
    meta = json.loads((pdir / "meta.json").read_text())
    assert lesion[1, 1, 1] == 1
    assert lesion[1, 3, 3] == 0
    assert lesion.dtype == np.uint8
    assert gland.dtype == np.uint8
    assert meta["gland_present"] is True
    assert meta["lesion_voxels_raw"] == 2
    assert meta["lesion_voxels_post"] == 1
    assert meta["lesion_threshold"] == 0.5


def test_postprocess_cli_skips_cases_with_missing_prob(tmp_path: Path,
                                                       capsys) -> None:
    run_dir = _seed_run_dir(tmp_path)
    # case_a is fine.
    lesion_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    gland_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)
    # case_b has the directory but no prob.npz — simulate failed inference.
    (run_dir / "diagnostic" / "predictions" / "case_b").mkdir(parents=True)

    rc = postprocess_cli.main([str(run_dir)])

    assert rc == 0
    assert (run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz").exists()
    assert not (run_dir / "diagnostic" / "postprocessed" / "case_b").exists()


def test_postprocess_cli_force_regenerates_postprocessed(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    lesion_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    gland_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)

    # First run — populate cache.
    assert postprocess_cli.main([str(run_dir)]) == 0
    out = run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz"
    first_mtime = out.stat().st_mtime_ns

    # Second run with --force should rewrite.
    import time as _t; _t.sleep(0.01)
    assert postprocess_cli.main([str(run_dir), "--force"]) == 0
    second_mtime = out.stat().st_mtime_ns
    assert second_mtime > first_mtime
```

- [ ] **Step 8.2: Run the tests; expect failure**

Run: `uv run pytest tests/test_postprocess_cli.py -v`
Expected: the new tests raise `NotImplementedError`. Threshold tests still pass.

- [ ] **Step 8.3: Implement the per-case loop (cached path only — dump fallback comes in Task 9)**

Replace the body of `mri/cli/postprocess.py:main` and add helpers. The full module should now read:

```python
"""CLI entry point for gland-constrained segmentation postprocessing.

Usage::

    python -m mri.cli.postprocess <run_dir> [--split val] [--force] \
        [--lesion-threshold FLOAT] [--gland-threshold FLOAT] [--device DEV]

Reads ``<run_dir>/diagnostic/predictions/<case>/prob.npz`` (running
``mri.diagnostics.dump.dump_predictions`` first if the cache is missing or
``--force`` is set; that wiring is added in Task 9) and writes
``<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from mri.cli.diagnose import resolve_run_dir
from mri.diagnostics.postprocess import apply_postprocess


def resolve_postprocess_thresholds(
    cfg: dict,
    *,
    lesion_arg: float | None,
    gland_arg: float | None,
) -> tuple[float, float]:
    metrics_cfg = cfg.get("metrics") or {}
    cfg_threshold = metrics_cfg.get("segmentation_threshold")
    if cfg_threshold is None and (lesion_arg is None or gland_arg is None):
        warnings.warn(
            "No metrics.segmentation_threshold in resolved_config — "
            "falling back to 0.5 for missing flag(s).",
            stacklevel=2,
        )
    lesion = float(lesion_arg) if lesion_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    gland = float(gland_arg) if gland_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    return lesion, gland


def _process_case(
    *,
    case_dir: Path,
    output_dir: Path,
    lesion_threshold: float,
    gland_threshold: float,
) -> bool:
    """Postprocess one cached case. Returns True if written, False if skipped."""
    prob_path = case_dir / "prob.npz"
    if not prob_path.exists():
        warnings.warn(
            f"[postprocess] {case_dir.name}: prob.npz missing, skipping.",
            stacklevel=2,
        )
        return False
    arrays = np.load(prob_path)
    lesion_prob = arrays["lesion"]
    gland_prob = arrays["gland"]

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob,
        lesion_threshold=lesion_threshold,
        gland_threshold=gland_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "lesion_mask.npz", mask=lesion_mask)
    np.savez_compressed(output_dir / "gland_mask.npz", mask=gland_mask)
    (output_dir / "meta.json").write_text(json.dumps({
        "case_id": case_dir.name,
        "lesion_threshold": lesion_threshold,
        "gland_threshold": gland_threshold,
        "gland_present": gland_present,
        "lesion_voxels_raw": int((lesion_prob >= lesion_threshold).sum()),
        "lesion_voxels_post": int(lesion_mask.sum()),
        "gland_voxels": int(gland_mask.sum()),
    }, indent=2))
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gland-constrain + no-gland-suppress postprocessing "
                    "to a finished run's segmentation predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lesion-threshold", type=float, default=None)
    parser.add_argument("--gland-threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    cfg = yaml.safe_load(paths.resolved_config.read_text()) or {}
    lesion_threshold, gland_threshold = resolve_postprocess_thresholds(
        cfg, lesion_arg=args.lesion_threshold, gland_arg=args.gland_threshold,
    )

    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    postprocessed_dir = diag_root / "postprocessed"

    if not predictions_dir.exists():
        # Dump-fallback wiring lands in Task 9. For now, error clearly.
        raise SystemExit(
            f"[postprocess] no cached predictions at {predictions_dir}. "
            "Run `python -m mri.cli.diagnose` first (or wait for Task 9 "
            "which auto-runs the dump)."
        )

    case_dirs = sorted(p for p in predictions_dir.iterdir() if p.is_dir())
    n_written = 0
    for case_dir in case_dirs:
        out_dir = postprocessed_dir / case_dir.name
        if out_dir.exists() and not args.force:
            # Cached output; touching mtime is not necessary here. Skip work
            # but count as "kept" for the caller's information.
            print(f"[postprocess] {case_dir.name}: cached, skipping (use --force to regenerate).")
            continue
        if out_dir.exists() and args.force:
            for f in out_dir.iterdir():
                f.unlink()
        if _process_case(
            case_dir=case_dir, output_dir=out_dir,
            lesion_threshold=lesion_threshold, gland_threshold=gland_threshold,
        ):
            n_written += 1

    print(f"[postprocess] wrote {n_written} case(s) to {postprocessed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_postprocess_cli.py -v`
Expected: 6 passed (3 from Task 7 + 3 here).

- [ ] **Step 8.5: Commit**

```bash
git add mri/cli/postprocess.py tests/test_postprocess_cli.py
git commit -m "feat(cli): postprocess CLI per-case loop over cached prob.npz"
```

---

## Task 9: `mri.cli.postprocess` — dump fallback when cache is missing

**Files:**
- Modify: `mri/cli/postprocess.py`
- Modify: `tests/test_postprocess_cli.py`

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_postprocess_cli.py`:

```python
from unittest.mock import patch

import torch


def _stub_dataloader_factory():
    def loader():
        for slice_idx in range(2):
            images = torch.zeros(1, 5, 4, 4)
            masks = torch.zeros(1, 2, 4, 4)
            masks[0, 0, 1:3, 1:3] = 1
            meta = {"case_id": ["case_a"], "slice_idx": [slice_idx], "class": [2]}
            yield (images, masks, meta)
    return list(loader())


class _StubModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0   # gland always firing → gland_present=True
        return out


def test_postprocess_cli_runs_dump_when_predictions_missing(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)

    def fake_build(cfg, split):
        return _stub_dataloader_factory(), {"case_a": 2}

    from mri.cli import diagnose, postprocess as ppl_cli
    with patch.object(ppl_cli, "_build_model_and_dataloader", fake_build), \
         patch.object(ppl_cli, "_load_checkpoint",
                      lambda model, path, device: None), \
         patch.object(ppl_cli, "_create_segmentation_model",
                      lambda name, **params: _StubModel()):
        rc = ppl_cli.main([str(run_dir), "--split", "val"])

    assert rc == 0
    assert (run_dir / "diagnostic" / "predictions" / "case_a" / "prob.npz").exists()
    assert (run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz").exists()
```

- [ ] **Step 9.2: Run the test; expect failure**

Run: `uv run pytest tests/test_postprocess_cli.py::test_postprocess_cli_runs_dump_when_predictions_missing -v`
Expected: SystemExit "no cached predictions" or AttributeError on the patched names (because they don't exist yet).

- [ ] **Step 9.3: Wire up the dump fallback**

Edit `mri/cli/postprocess.py`. Add three module-level helpers (mirroring the names `mri.cli.diagnose` uses, so the tests can monkey-patch them):

```python
from mri.cli.diagnose import (
    _build_model_and_dataloader,
    _create_segmentation_model,
    _load_checkpoint,
)
from mri.diagnostics.dump import dump_predictions
from mri.training.trainer import resolve_device
```

(Place the imports near the top, alongside the existing `from mri.cli.diagnose import resolve_run_dir`.)

Replace the `if not predictions_dir.exists(): raise SystemExit(...)` block with:

```python
    if not predictions_dir.exists() or args.force:
        loader, num_slices_per_case = _build_model_and_dataloader(cfg, args.split)
        model = _create_segmentation_model(
            cfg["model"]["name"], **(cfg["model"].get("params") or {}),
        )
        device = resolve_device(args.device)
        _load_checkpoint(model, paths.checkpoint, device)
        dump_summary = dump_predictions(
            model=model,
            dataloader=loader,
            device=device,
            output_dir=predictions_dir,
            num_slices_per_case=num_slices_per_case,
            force=args.force,
            lesion_threshold=lesion_threshold,
            gland_threshold=gland_threshold,
        )
        print(f"[postprocess] dump: {dump_summary}")
```

- [ ] **Step 9.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_postprocess_cli.py -v`
Expected: 7 passed.

- [ ] **Step 9.5: Commit**

```bash
git add mri/cli/postprocess.py tests/test_postprocess_cli.py
git commit -m "feat(cli): postprocess CLI auto-runs dump when prediction cache is missing"
```

---

## Task 10: `mri.cli.evaluate` — argparse + CSVs/summary end-to-end (no visuals)

**Files:**
- Create: `mri/cli/evaluate.py`
- Test: `tests/test_evaluate_cli.py`

- [ ] **Step 10.1: Write the failing tests**

Create `tests/test_evaluate_cli.py` with:

```python
"""End-to-end CLI tests for `python -m mri.cli.evaluate`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mri.cli import evaluate as evaluate_cli


def _seed_predictions(run_dir: Path, case_id: str, *, gt_lesion, gt_gland) -> None:
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    Z, H, W = gt_lesion.shape
    np.savez_compressed(pdir / "prob.npz",
                         gland=np.zeros((Z, H, W), dtype=np.float32),
                         lesion=np.zeros((Z, H, W), dtype=np.float32))
    np.savez_compressed(pdir / "gt.npz",
                         gland=gt_gland.astype(np.uint8),
                         lesion=gt_lesion.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id, "class_label": 2 if gt_lesion.any() else 0,
        "spatial_shape": [H, W], "num_slices": Z,
        "predicted_slices": list(range(Z)),
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def _seed_postprocessed(run_dir: Path, case_id: str, *, lesion_mask, gland_mask) -> None:
    pdir = run_dir / "diagnostic" / "postprocessed" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "lesion_mask.npz", mask=lesion_mask.astype(np.uint8))
    np.savez_compressed(pdir / "gland_mask.npz", mask=gland_mask.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id,
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
        "gland_present": bool(gland_mask.any()),
        "lesion_voxels_raw": int(lesion_mask.sum()),
        "lesion_voxels_post": int(lesion_mask.sum()),
        "gland_voxels": int(gland_mask.sum()),
    }))


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
    )
    return run_dir


def test_evaluate_cli_writes_lesion_case_csv_and_summary(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 3, 10, 10

    # Positive case: 2 lesions, 1 detected.
    gt_pos = np.zeros((Z, H, W), dtype=np.uint8)
    gt_pos[1, 1, 1] = 1
    gt_pos[2, 8, 8] = 1
    pred_pos = np.zeros((Z, H, W), dtype=np.uint8)
    pred_pos[1, 1, 1] = 1   # exact hit on lesion 1
    _seed_predictions(run_dir, "case_pos",
                      gt_lesion=gt_pos, gt_gland=np.zeros_like(gt_pos))
    _seed_postprocessed(run_dir, "case_pos",
                         lesion_mask=pred_pos, gland_mask=np.zeros_like(gt_pos))

    # Negative case: 1% predicted area, below 2% ⇒ correct.
    gt_neg = np.zeros((1, 10, 10), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    pred_neg[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_neg",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_neg",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "none"])

    assert rc == 0
    eval_dir = run_dir / "diagnostic" / "evaluation"
    assert (eval_dir / "metrics_by_lesion.csv").exists()
    assert (eval_dir / "metrics_by_case.csv").exists()
    assert (eval_dir / "summary.json").exists()
    assert not (eval_dir / "visuals").exists()  # --visualize-only none

    with (eval_dir / "metrics_by_lesion.csv").open() as f:
        lesion_rows = list(csv.DictReader(f))
    assert len(lesion_rows) == 2
    assert {row["detected"] for row in lesion_rows} == {"True", "False"}

    with (eval_dir / "metrics_by_case.csv").open() as f:
        case_rows = list(csv.DictReader(f))
    assert {row["case_kind"] for row in case_rows} == {"positive", "negative"}

    summary = json.loads((eval_dir / "summary.json").read_text())
    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["negatives"]["n_correct"] == 1


def test_evaluate_cli_correctness_iou_flag_changes_detection(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    gt = np.zeros((Z, H, W), dtype=np.uint8)
    gt[0, 0, 0:5] = 1   # 5 voxels
    pred = np.zeros((Z, H, W), dtype=np.uint8)
    pred[0, 0, 0:1] = 1  # iou = 1/5 = 0.2
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))

    # Default 0.1 → detected.
    assert evaluate_cli.main([str(run_dir), "--visualize-only", "none"]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 1

    # Raise to 0.5 → not detected.
    assert evaluate_cli.main([
        str(run_dir), "--correctness-iou", "0.5", "--visualize-only", "none",
    ]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 0


def test_evaluate_cli_errors_when_postprocessed_missing(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    # predictions/ exists but postprocessed/ does not.
    (run_dir / "diagnostic" / "predictions").mkdir(parents=True)

    with pytest.raises(SystemExit, match="postprocess"):
        evaluate_cli.main([str(run_dir), "--visualize-only", "none"])
```

- [ ] **Step 10.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_evaluate_cli.py -v`
Expected: ImportError on `mri.cli.evaluate`.

- [ ] **Step 10.3: Implement `mri/cli/evaluate.py`**

Create the file with:

```python
"""CLI entry point for per-3D-lesion evaluation of postprocessed predictions.

Usage::

    python -m mri.cli.evaluate <run_dir> \\
        [--correctness-iou 0.1] [--negative-area-frac 0.02] \\
        [--connectivity 6] \\
        [--visualize-only all|failed|none] [--downsample-vis 1] [--plotly-cdn]

Reads ``<run_dir>/diagnostic/postprocessed/<case>/lesion_mask.npz`` and the
matching ``<run_dir>/diagnostic/predictions/<case>/{gt.npz, meta.json}``
and writes ``<run_dir>/diagnostic/evaluation/{metrics_by_lesion.csv,
metrics_by_case.csv, summary.json, visuals/...}``.

Visual rendering is wired in Task 13 + Task 14; Task 10 emits CSV/JSON only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from mri.cli.diagnose import resolve_run_dir
from mri.diagnostics.detection import (
    LesionRow, CaseRow, evaluate_case,
    write_lesion_csv, write_case_csv, build_summary, write_summary_json,
)


_CONNECTIVITY_TO_RANK = {6: 1, 26: 3}


def _load_case(predictions_dir: Path, postprocessed_dir: Path, case_id: str):
    gt = np.load(predictions_dir / case_id / "gt.npz")
    pred = np.load(postprocessed_dir / case_id / "lesion_mask.npz")
    meta = json.loads((predictions_dir / case_id / "meta.json").read_text())
    return gt["lesion"], pred["mask"], int(meta.get("class_label", 0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-3D-lesion evaluation of postprocessed predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--correctness-iou", type=float, default=0.1)
    parser.add_argument("--negative-area-frac", type=float, default=0.02)
    parser.add_argument("--connectivity", type=int, choices=[6, 26], default=6)
    parser.add_argument(
        "--visualize-only", choices=["all", "failed", "none"], default="all",
    )
    parser.add_argument("--downsample-vis", type=int, default=1)
    parser.add_argument("--plotly-cdn", action="store_true")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    cfg = yaml.safe_load(paths.resolved_config.read_text()) or {}
    seg_threshold = (cfg.get("metrics") or {}).get("segmentation_threshold", 0.5)

    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    postprocessed_dir = diag_root / "postprocessed"
    eval_dir = diag_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not postprocessed_dir.exists() or not any(postprocessed_dir.iterdir()):
        raise SystemExit(
            f"[evaluate] no postprocessed predictions at {postprocessed_dir}. "
            "Run `python -m mri.cli.postprocess <run_dir>` first."
        )

    case_ids = sorted(p.name for p in postprocessed_dir.iterdir() if p.is_dir())
    case_rows: list[CaseRow] = []
    lesion_rows: list[LesionRow] = []
    cases_skipped: list[str] = []

    connectivity_rank = _CONNECTIVITY_TO_RANK[args.connectivity]

    for case_id in case_ids:
        gt_path = predictions_dir / case_id / "gt.npz"
        if not gt_path.exists():
            warnings.warn(
                f"[evaluate] {case_id}: missing predictions/gt.npz, skipping.",
                stacklevel=2,
            )
            cases_skipped.append(case_id)
            continue
        gt_lesion, pred_lesion, class_label = _load_case(
            predictions_dir, postprocessed_dir, case_id,
        )
        case_row, rows = evaluate_case(
            case_id=case_id, class_label=class_label,
            gt_lesion=gt_lesion, pred_lesion=pred_lesion,
            correctness_iou=args.correctness_iou,
            negative_area_frac=args.negative_area_frac,
            connectivity_rank=connectivity_rank,
        )
        case_rows.append(case_row)
        lesion_rows.extend(rows)

    write_lesion_csv(lesion_rows, eval_dir / "metrics_by_lesion.csv")
    write_case_csv(case_rows, eval_dir / "metrics_by_case.csv")
    summary = build_summary(
        case_rows=case_rows, lesion_rows=lesion_rows,
        params={
            "correctness_iou": args.correctness_iou,
            "negative_area_frac": args.negative_area_frac,
            "connectivity": args.connectivity,
            "lesion_threshold": float(seg_threshold),
            "gland_threshold": float(seg_threshold),
        },
        cases_skipped=cases_skipped,
    )
    write_summary_json(summary, eval_dir / "summary.json")

    if args.visualize_only != "none":
        # Wired in Task 14.
        pass

    print(f"[evaluate] wrote evaluation/ to {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_evaluate_cli.py -v`
Expected: 3 passed.

- [ ] **Step 10.5: Commit**

```bash
git add mri/cli/evaluate.py tests/test_evaluate_cli.py
git commit -m "feat(cli): evaluate CLI emits per-lesion CSV, per-case CSV, summary.json"
```

---

## Task 11: Visualization — `build_case_figure`

**Files:**
- Create: `mri/diagnostics/visualization.py`
- Test: `tests/test_diagnostics_visualization.py`

- [ ] **Step 11.1: Write the failing tests**

Create `tests/test_diagnostics_visualization.py` with:

```python
"""Unit tests for the per-case Plotly figure builder."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.visualization import build_case_figure, ComponentSpec


def _solid_cube(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
                shape=(8, 8, 8)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[z0:z1, y0:y1, x0:x1] = 1
    return arr


def test_positive_case_has_three_kinds_of_traces() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    component_a = _solid_cube(3, 5, 3, 5, 3, 5)
    component_b = _solid_cube(2, 3, 5, 6, 5, 6)
    pred_lesion = _solid_cube(3, 5, 3, 5, 3, 5)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[
            ComponentSpec(component_a, lesion_id=1, detected=True),
            ComponentSpec(component_b, lesion_id=2, detected=False),
        ],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" in names
    assert "Predicted lesion" in names
    assert "GT lesion 1 (detected)" in names
    assert "GT lesion 2 (missed)" in names
    assert len(fig.data) == 4


def test_negative_case_omits_lesion_traces() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    pred_lesion = _solid_cube(3, 4, 3, 4, 3, 4)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" in names
    assert "Predicted lesion" in names
    assert all("GT lesion" not in n for n in names)


def test_empty_pred_omits_pred_trace() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    pred_lesion = np.zeros_like(gt_gland)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "Predicted lesion" not in names


def test_empty_gland_omits_gland_trace() -> None:
    gt_gland = np.zeros((4, 4, 4), dtype=np.uint8)
    pred_lesion = _solid_cube(1, 2, 1, 2, 1, 2, shape=(4, 4, 4))

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" not in names
    assert "Predicted lesion" in names


def test_downsample_reduces_grid_size() -> None:
    gt_gland = np.ones((8, 8, 8), dtype=np.uint8)
    pred_lesion = np.ones((8, 8, 8), dtype=np.uint8)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=2,
    )

    # The Isosurface trace stores its scalar grid in the `value` array of
    # length Z*Y*X. Downsample=2 ⇒ 4*4*4 = 64 entries instead of 512.
    pred_trace = next(t for t in fig.data if t.name == "Predicted lesion")
    assert len(pred_trace.value) == 64
```

- [ ] **Step 11.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_visualization.py -v`
Expected: ImportError.

- [ ] **Step 11.3: Implement the figure builder**

Create `mri/diagnostics/visualization.py` with:

```python
"""Plotly-based 3D visualization of postprocessed segmentation results.

`build_case_figure` produces a single rotatable scene with three kinds of
isosurface traces:

- GT gland (pale yellow, low opacity) — anatomical context.
- One trace per GT lesion 3D component, color-coded by detection verdict
  (green if detected, gray if missed).
- Postprocessed predicted lesion (red).

Each trace is toggleable via the legend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class ComponentSpec:
    """One GT lesion 3D component to render."""
    mask: np.ndarray
    lesion_id: int
    detected: bool


_GLAND_COLOR = "#f4e285"
_PRED_COLOR = "#d6334b"
_DETECTED_COLOR = "#3fa34d"
_MISSED_COLOR = "#8a8a8a"


def _downsample(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    return arr[::k, ::k, ::k]


def _isosurface_from_mask(
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    opacity: float,
    showlegend: bool = True,
) -> go.Isosurface | None:
    """Wrap a binary mask into a Plotly Isosurface trace.

    Returns None if the mask has no foreground voxels (Plotly rejects empty
    isosurfaces).
    """
    if not mask.any():
        return None
    Z, Y, X = mask.shape
    z_idx, y_idx, x_idx = np.mgrid[0:Z, 0:Y, 0:X]
    return go.Isosurface(
        x=x_idx.flatten(),
        y=y_idx.flatten(),
        z=z_idx.flatten(),
        value=mask.astype(np.float32).flatten(),
        isomin=0.5,
        isomax=1.0,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
    )


def build_case_figure(
    *,
    gt_gland: np.ndarray,
    gt_lesion_components: list[ComponentSpec],
    pred_lesion: np.ndarray,
    downsample: int = 1,
) -> go.Figure:
    """Build the per-case 3D Plotly figure (no I/O)."""
    fig = go.Figure()

    gland_trace = _isosurface_from_mask(
        _downsample(gt_gland, downsample),
        name="GT gland", color=_GLAND_COLOR, opacity=0.15,
    )
    if gland_trace is not None:
        fig.add_trace(gland_trace)

    for comp in gt_lesion_components:
        suffix = "detected" if comp.detected else "missed"
        color = _DETECTED_COLOR if comp.detected else _MISSED_COLOR
        trace = _isosurface_from_mask(
            _downsample(comp.mask, downsample),
            name=f"GT lesion {comp.lesion_id} ({suffix})",
            color=color, opacity=0.55,
        )
        if trace is not None:
            fig.add_trace(trace)

    pred_trace = _isosurface_from_mask(
        _downsample(pred_lesion, downsample),
        name="Predicted lesion", color=_PRED_COLOR, opacity=0.45,
    )
    if pred_trace is not None:
        fig.add_trace(pred_trace)

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig
```

- [ ] **Step 11.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_visualization.py -v`
Expected: 5 passed.

- [ ] **Step 11.5: Commit**

```bash
git add mri/diagnostics/visualization.py tests/test_diagnostics_visualization.py
git commit -m "feat(diagnostics): build_case_figure (Plotly isosurfaces, per-component coloring)"
```

---

## Task 12: Visualization — `write_case_html` + `write_index_html`

**Files:**
- Modify: `mri/diagnostics/visualization.py` (append)
- Modify: `tests/test_diagnostics_visualization.py` (append)

- [ ] **Step 12.1: Write the failing tests**

Append to `tests/test_diagnostics_visualization.py`:

```python
from pathlib import Path

from mri.diagnostics.visualization import (
    write_case_html, write_index_html, CaseSummary,
)


def test_write_case_html_contains_plotly_and_case_id(tmp_path: Path) -> None:
    gt_gland = np.ones((4, 4, 4), dtype=np.uint8)
    fig = build_case_figure(
        gt_gland=gt_gland, gt_lesion_components=[],
        pred_lesion=np.zeros_like(gt_gland), downsample=1,
    )
    out = tmp_path / "c1.html"

    write_case_html(
        fig, out,
        header_meta={"case_id": "c1", "class_label": 2,
                     "n_gt_lesions": 0, "n_detected_lesions": 0,
                     "lesion_recall": None, "negative_correct": True},
        use_cdn=False,
    )

    text = out.read_text()
    assert "plotly" in text.lower()
    assert "c1" in text
    assert "negative_correct" in text or "Negative correct" in text


def test_write_case_html_cdn_yields_smaller_file(tmp_path: Path) -> None:
    gt_gland = np.ones((4, 4, 4), dtype=np.uint8)
    fig = build_case_figure(
        gt_gland=gt_gland, gt_lesion_components=[],
        pred_lesion=np.zeros_like(gt_gland), downsample=1,
    )
    inline = tmp_path / "inline.html"
    cdn = tmp_path / "cdn.html"

    write_case_html(fig, inline, header_meta={"case_id": "c1"}, use_cdn=False)
    write_case_html(fig, cdn,    header_meta={"case_id": "c1"}, use_cdn=True)

    assert inline.stat().st_size > cdn.stat().st_size


def test_write_index_html_lists_cases(tmp_path: Path) -> None:
    summaries = [
        CaseSummary(case_id="c1", case_kind="positive",
                    n_gt_lesions=2, n_detected_lesions=1,
                    lesion_recall=0.5, negative_correct=None),
        CaseSummary(case_id="c2", case_kind="negative",
                    n_gt_lesions=0, n_detected_lesions=0,
                    lesion_recall=None, negative_correct=True),
    ]
    out = tmp_path / "index.html"

    write_index_html(summaries, out)

    text = out.read_text()
    assert 'href="c1.html"' in text
    assert 'href="c2.html"' in text
    assert "positive" in text
    assert "negative" in text
```

- [ ] **Step 12.2: Run the tests; expect ImportError**

Run: `uv run pytest tests/test_diagnostics_visualization.py -k "write_" -v`
Expected: ImportError.

- [ ] **Step 12.3: Implement the writers**

Append to `mri/diagnostics/visualization.py`:

```python
from pathlib import Path


@dataclass(frozen=True)
class CaseSummary:
    """One row in the visuals/index.html gallery."""
    case_id: str
    case_kind: str
    n_gt_lesions: int
    n_detected_lesions: int
    lesion_recall: float | None
    negative_correct: bool | None


def _format_header(meta: dict) -> str:
    """Render a small HTML header bar above the figure."""
    fields = []
    for key in ("case_id", "class_label", "n_gt_lesions",
                "n_detected_lesions", "lesion_recall", "negative_correct"):
        if key in meta and meta[key] is not None:
            fields.append(f"<b>{key}:</b> {meta[key]}")
    return "<div style='font-family:sans-serif;padding:8px;'>" + " &nbsp; ".join(fields) + "</div>"


def write_case_html(
    fig: go.Figure,
    path: Path,
    *,
    header_meta: dict,
    use_cdn: bool = False,
) -> None:
    """Write a self-contained per-case HTML.

    Args:
      fig: figure produced by ``build_case_figure``.
      path: destination ``.html`` path.
      header_meta: rendered as a small header bar above the figure. Any of
          ``case_id``, ``class_label``, ``n_gt_lesions``, ``n_detected_lesions``,
          ``lesion_recall``, ``negative_correct`` keys are surfaced.
      use_cdn: when True, the Plotly JS is loaded from a CDN at view time
          (smaller files, requires network); otherwise it's inlined.
    """
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if use_cdn else "inline",
    )
    header = _format_header(header_meta)
    case_id = header_meta.get("case_id", "case")
    full = (
        "<!doctype html><html><head>"
        f"<title>{case_id}</title>"
        "<meta charset='utf-8'></head><body>"
        f"{header}{plot_div}"
        "</body></html>"
    )
    Path(path).write_text(full)


def write_index_html(summaries: list[CaseSummary], path: Path) -> None:
    """Write a gallery linking to per-case HTMLs."""
    rows = []
    for s in summaries:
        cells = [
            f"<a href='{s.case_id}.html'>{s.case_id}</a>",
            s.case_kind,
            str(s.n_gt_lesions),
            str(s.n_detected_lesions),
            "" if s.lesion_recall is None else f"{s.lesion_recall:.3f}",
            "" if s.negative_correct is None else str(s.negative_correct),
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    header = (
        "<tr>"
        "<th>case_id</th><th>case_kind</th><th>n_gt_lesions</th>"
        "<th>n_detected_lesions</th><th>lesion_recall</th>"
        "<th>negative_correct</th>"
        "</tr>"
    )
    html = (
        "<!doctype html><html><head>"
        "<title>evaluation index</title>"
        "<meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;}"
        "table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
        "</style></head><body>"
        f"<h2>Per-case evaluation visuals</h2>"
        f"<table>{header}{''.join(rows)}</table>"
        "</body></html>"
    )
    Path(path).write_text(html)
```

- [ ] **Step 12.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_diagnostics_visualization.py -v`
Expected: 8 passed (5 from Task 11 + 3 here).

- [ ] **Step 12.5: Commit**

```bash
git add mri/diagnostics/visualization.py tests/test_diagnostics_visualization.py
git commit -m "feat(diagnostics): per-case + index HTML writers for evaluation visuals"
```

---

## Task 13: Wire visualization into `mri.cli.evaluate`

**Files:**
- Modify: `mri/cli/evaluate.py`
- Modify: `tests/test_evaluate_cli.py`

- [ ] **Step 13.1: Write the failing tests**

Append to `tests/test_evaluate_cli.py`:

```python
def test_evaluate_cli_visualize_all_writes_html_per_case(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 4, 8, 8
    gt = np.zeros((Z, H, W), dtype=np.uint8); gt[1, 2, 2] = 1
    pred = np.zeros((Z, H, W), dtype=np.uint8); pred[1, 2, 2] = 1
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))
    gt_neg = np.zeros((1, 8, 8), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    _seed_predictions(run_dir, "case_b",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_b",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "all"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert (visuals / "case_a.html").exists()
    assert (visuals / "case_b.html").exists()
    assert (visuals / "index.html").exists()


def test_evaluate_cli_visualize_failed_only_renders_failures(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    # case_pass: lesion exactly hit ⇒ detected.
    gt_pass = np.zeros((Z, H, W), dtype=np.uint8); gt_pass[0, 0, 0] = 1
    pred_pass = np.zeros_like(gt_pass); pred_pass[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_pass",
                      gt_lesion=gt_pass, gt_gland=np.zeros_like(gt_pass))
    _seed_postprocessed(run_dir, "case_pass",
                         lesion_mask=pred_pass,
                         gland_mask=np.zeros_like(gt_pass))
    # case_fail: lesion missed.
    gt_fail = np.zeros((Z, H, W), dtype=np.uint8); gt_fail[0, 5, 5] = 1
    pred_fail = np.zeros_like(gt_fail)
    _seed_predictions(run_dir, "case_fail",
                      gt_lesion=gt_fail, gt_gland=np.zeros_like(gt_fail))
    _seed_postprocessed(run_dir, "case_fail",
                         lesion_mask=pred_fail,
                         gland_mask=np.zeros_like(gt_fail))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "failed"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert not (visuals / "case_pass.html").exists()
    assert (visuals / "case_fail.html").exists()
    assert (visuals / "index.html").exists()
```

- [ ] **Step 13.2: Run the tests; expect failure**

Run: `uv run pytest tests/test_evaluate_cli.py -k "visualize" -v`
Expected: visuals are not created (the `pass` placeholder still no-ops).

- [ ] **Step 13.3: Wire visualization into `evaluate.main`**

In `mri/cli/evaluate.py`, add imports near the top:

```python
from mri.diagnostics.visualization import (
    build_case_figure, write_case_html, write_index_html,
    ComponentSpec, CaseSummary,
)
from mri.diagnostics.detection import label_lesion_components
```

Add a helper above `main`:

```python
def _is_failed_case(case_row: CaseRow, case_lesion_rows: list[LesionRow]) -> bool:
    if case_row.case_kind == "negative":
        return case_row.negative_correct is False
    return any(not r.detected for r in case_lesion_rows)
```

Replace the `if args.visualize_only != "none":` block in `main` with:

```python
    if args.visualize_only != "none":
        visuals_dir = eval_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        case_id_to_lesion_rows: dict[str, list[LesionRow]] = {}
        for row in lesion_rows:
            case_id_to_lesion_rows.setdefault(row.case_id, []).append(row)

        rendered: list[CaseSummary] = []
        for case_row in case_rows:
            case_lesion_rows = case_id_to_lesion_rows.get(case_row.case_id, [])
            if args.visualize_only == "failed" and not _is_failed_case(
                case_row, case_lesion_rows,
            ):
                continue

            gt = np.load(predictions_dir / case_row.case_id / "gt.npz")
            gland_gt = gt["gland"]
            lesion_gt = gt["lesion"]
            pred_lesion = np.load(
                postprocessed_dir / case_row.case_id / "lesion_mask.npz",
            )["mask"]

            labels, n_components = label_lesion_components(
                lesion_gt, connectivity_rank=connectivity_rank,
            )
            detected_ids = {row.lesion_id for row in case_lesion_rows if row.detected}
            components = [
                ComponentSpec(
                    mask=(labels == k),
                    lesion_id=k,
                    detected=(k in detected_ids),
                )
                for k in range(1, n_components + 1)
            ]

            fig = build_case_figure(
                gt_gland=gland_gt,
                gt_lesion_components=components,
                pred_lesion=pred_lesion,
                downsample=args.downsample_vis,
            )
            write_case_html(
                fig,
                visuals_dir / f"{case_row.case_id}.html",
                header_meta={
                    "case_id": case_row.case_id,
                    "class_label": case_row.class_label,
                    "n_gt_lesions": case_row.n_gt_lesions,
                    "n_detected_lesions": case_row.n_detected_lesions,
                    "lesion_recall": case_row.lesion_recall,
                    "negative_correct": case_row.negative_correct,
                },
                use_cdn=args.plotly_cdn,
            )
            rendered.append(CaseSummary(
                case_id=case_row.case_id,
                case_kind=case_row.case_kind,
                n_gt_lesions=case_row.n_gt_lesions,
                n_detected_lesions=case_row.n_detected_lesions,
                lesion_recall=case_row.lesion_recall,
                negative_correct=case_row.negative_correct,
            ))

        write_index_html(rendered, visuals_dir / "index.html")
```

- [ ] **Step 13.4: Run the tests; expect all pass**

Run: `uv run pytest tests/test_evaluate_cli.py -v`
Expected: 5 passed (3 from Task 10 + 2 here).

- [ ] **Step 13.5: Final integration sanity check**

Run: `uv run pytest tests/test_diagnostics_postprocess.py tests/test_diagnostics_detection.py tests/test_diagnostics_visualization.py tests/test_postprocess_cli.py tests/test_evaluate_cli.py -v`
Expected: all tests pass (7 + 17 + 8 + 7 + 5 = 44 tests).

- [ ] **Step 13.6: Commit**

```bash
git add mri/cli/evaluate.py tests/test_evaluate_cli.py
git commit -m "feat(cli): evaluate emits per-case + index Plotly HTMLs (--visualize-only)"
```

---

## Task 14: User documentation

**Files:**
- Create: `docs/postprocess-evaluate.md`
- Modify: `docs/README.md` (add to the index)

- [ ] **Step 14.1: Add the new doc**

Create `docs/postprocess-evaluate.md` with:

```markdown
# Postprocess + per-lesion evaluation

Two CLIs that turn a finished segmentation run directory into a postprocessed
prediction set and a per-3D-lesion detection score, with an interactive 3D
Plotly HTML per case.

## When to run this

Run after `mri.cli.diagnose` (or after a fresh training run) when you want to
answer:

- "Did the model find each lesion that was actually there?"
- "Is the model staying quiet on healthy cases?"

The voxel-level Dice / precision / recall in the diagnostic report don't
answer either of those questions on their own.

## Pipeline

```
mri.cli.diagnose  →  mri.cli.postprocess  →  mri.cli.evaluate
```

`postprocess` will auto-run the dump step (the same one diagnose uses) if
the prediction cache is missing, so in practice you can skip diagnose if you
only care about the postprocessed evaluation.

## Step 1 — Postprocess

```bash
uv run python -m mri.cli.postprocess <run_dir>
```

Applies two voxel-wise rules:

1. **Target outside prostate is ignored** — `lesion_mask &= gland_mask`.
2. **No prostate ⇒ no target** — when no gland voxel passes its threshold,
   the lesion mask is fully zeroed.

Both use the *predicted* gland (not GT), so the rules mirror what would
happen at deployment.

Output: `<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}`.

Useful flags:

- `--lesion-threshold 0.4 --gland-threshold 0.3` — override the thresholds
  (default: `metrics.segmentation_threshold` from the resolved config).
- `--force` — re-run inference (regenerates `predictions/` and
  `postprocessed/`).

## Step 2 — Evaluate

```bash
uv run python -m mri.cli.evaluate <run_dir>
```

For each case:

- **Positive case (≥ 1 GT lesion).** GT is split into 3D-connected
  components. Each component is scored by the *max* IoU across the slices
  it spans; a component is detected iff that max IoU > `--correctness-iou`
  (default 0.1). Detection is per-lesion, not per-case: a 2-lesion case
  with one hit and one miss contributes 1/2 to lesion-level recall.
- **Negative case (no GT lesion).** Correct iff the postprocessed
  prediction covers ≤ `--negative-area-frac` (default 0.02 = 2%) of *every*
  slice.

Outputs under `<run_dir>/diagnostic/evaluation/`:

- `metrics_by_lesion.csv` — one row per 3D GT lesion in the cohort.
- `metrics_by_case.csv` — one row per case (positive or negative).
- `summary.json` — cohort lesion-recall, negative-case accuracy, and the
  parameters used.
- `visuals/<case>.html` — interactive 3D Plotly figure per case (toggleable
  per-component, color-coded by detection verdict).
- `visuals/index.html` — gallery linking to all per-case HTMLs.

Useful flags:

- `--correctness-iou 0.2` — tighten the detection bar.
- `--negative-area-frac 0.01` — tighten the false-alarm bar.
- `--connectivity 26` — 26-connectivity in 3D for GT CC labeling
  (default 6).
- `--visualize-only failed` — only render HTMLs for cases with any missed
  lesion or any negative-case false alarm. Use `none` to skip rendering
  entirely.
- `--downsample-vis 2` — downsample voxel grids 2x along each axis before
  isosurface extraction; cuts HTML size for large volumes.
- `--plotly-cdn` — load Plotly from CDN instead of inlining ~3 MB per HTML.
```

- [ ] **Step 14.2: Add the new doc to the index**

Open `docs/README.md` and add this line to the documentation index, after
the `diagnostic.md` entry:

```
- [postprocess-evaluate.md](postprocess-evaluate.md) — per-lesion detection metrics with interactive 3D visualization
```

- [ ] **Step 14.3: Sanity check**

Run: `uv run python -m mri.cli.postprocess --help`
Expected: argparse usage prints, exits 0.

Run: `uv run python -m mri.cli.evaluate --help`
Expected: argparse usage prints, exits 0.

- [ ] **Step 14.4: Commit**

```bash
git add docs/postprocess-evaluate.md docs/README.md
git commit -m "docs: postprocess + per-lesion evaluation runbook"
```

---

## Self-review

**Spec coverage:**

- Goal — both CLIs ✅ (Tasks 7–10, 13).
- Postprocess rules 1+2 ✅ (Task 2).
- Per-3D-lesion detection with max-over-slices IoU ✅ (Tasks 3–5).
- Strict `>` semantics on `correctness_iou` and on the negative-area FALSE side ✅ (Task 5 boundary tests).
- Per-lesion granularity (multi-lesion case ⇒ multiple rows) ✅ (Task 5).
- Negative-case 2%-area rule ✅ (Task 5).
- CSV NaN convention (empty cells, no `nan` literal) ✅ (Task 6).
- summary.json schema ✅ (Task 6).
- Plotly Isosurface per case + per-component coloring ✅ (Task 11).
- Inline vs CDN HTML ✅ (Task 12).
- Index gallery ✅ (Task 12).
- `--visualize-only all|failed|none`, `--downsample-vis`, `--plotly-cdn` ✅ (Task 13).
- Dump-fallback in postprocess CLI ✅ (Task 9).
- `postprocessed/` missing → clear evaluate error ✅ (Task 10).
- Docs ✅ (Task 14).

**Placeholder scan:** Every step has executable code or an exact command. The Task 7 stub `raise NotImplementedError("Wired in Task 8.")` is intentional and the next task replaces it. No "TODO", "TBD", or "fill in" in the implementation steps.

**Type consistency check:**

- `LesionRow`, `CaseRow`, `LesionIoUResult`, `ComponentSpec`, `CaseSummary` — all defined once and referenced consistently.
- `connectivity_rank` (1 or 3) is the internal name; CLI flag `--connectivity` (6 or 26) maps to it via `_CONNECTIVITY_TO_RANK` in Task 10. `label_lesion_components` and `evaluate_case` both take `connectivity_rank`.
- `compute_lesion_iou` returns `LesionIoUResult` (used by `evaluate_case`).
- `apply_postprocess` signature: `(lesion_prob, gland_prob, *, lesion_threshold, gland_threshold) -> (lesion_mask, gland_mask, gland_present)` — same in tests, CLI, and module.
- `gt.npz` keys: `gland`, `lesion` (matches existing `dump.py`).
- `prob.npz` keys: `gland`, `lesion` (matches existing `dump.py`).
- `lesion_mask.npz` / `gland_mask.npz` key: `mask` — same in writer (Task 8) and reader (Tasks 10, 13).
- `meta.json` fields persisted by postprocess: `case_id`, `lesion_threshold`, `gland_threshold`, `gland_present`, `lesion_voxels_raw`, `lesion_voxels_post`, `gland_voxels` — matches the spec table.

No discrepancies found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-segmentation-postprocess-evaluate.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
