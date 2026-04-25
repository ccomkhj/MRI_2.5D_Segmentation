# Segmentation Error-Analysis & Label-Audit Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a CLI (`python -m mri.cli.diagnose <run_dir>`) that turns a finished segmentation run into per-case error attribution (FP inside vs outside gland, stratified by 0–4 class), a label-audit queue, and a single HTML report.

**Architecture:** New `mri.diagnostics` package with four pure-Python components — `dump` (inference → npz), `attribute` (metrics CSV), `audit` (heuristics CSV), `report` (Jinja → HTML) — plus a thin CLI in `mri.cli.diagnose`. Reuses the existing config loader, model registry, val dataloader, and Jinja-rendered HTML pattern. No changes to training or to existing inference paths.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.ndimage` for connected components), PyTorch (inference only), Jinja2 (already used by `mri/inference/html_report.py`), matplotlib (already used by the project), pytest.

**Spec:** [`docs/superpowers/specs/2026-04-25-segmentation-error-analysis-design.md`](../specs/2026-04-25-segmentation-error-analysis-design.md)

---

## File map

**Create:**
- `mri/diagnostics/__init__.py` — package marker
- `mri/diagnostics/dump.py` — per-case inference → `prob.npz`/`gt.npz`/`meta.json`
- `mri/diagnostics/attribute.py` — pure-numpy metrics, no I/O of its own beyond the CSV
- `mri/diagnostics/audit.py` — pure-numpy heuristics, no I/O of its own beyond the CSV
- `mri/diagnostics/report.py` — Jinja-rendered HTML
- `mri/diagnostics/templates/diagnostic_report.html.j2` — single-file template
- `mri/cli/diagnose.py` — CLI wiring, argparse, run-dir resolution, orchestration
- `tests/test_diagnostics_attribute.py` — unit tests for `attribute.py`
- `tests/test_diagnostics_audit.py` — unit tests for `audit.py`
- `tests/test_diagnostics_dump.py` — smoke test for `dump.py` on a tiny dummy case
- `tests/test_diagnose_cli.py` — argparse / run-dir-validation smoke test

**Modify:**
- None of the existing source files. The trainer, inference path, model registry, dataset, and existing HTML report are read-only for this work.

**Test layout note:** existing project uses a flat `tests/` directory (no subpackages). New tests follow that convention. Spec's `tests/diagnostics/` is overridden by codebase pattern.

**Path conventions for run dirs:** the project has two on-disk layouts for finished runs:
- Training-side: `<run_dir>/<run_name>_best.pt` + `<run_dir>/<run_name>_resolved_config.yaml`
- Release/UI-side: `<run_dir>/*_best.pt` + `<run_dir>/resolved_config.yaml`

`mri/cli/diagnose.py` resolves both: prefer exact `resolved_config.yaml` if present, else first `*_resolved_config.yaml`; pick the unique `*_best.pt`.

---

## Conventions used throughout this plan

- **TDD:** every code task is test-first — write failing test, run to verify it fails, implement minimum, run to verify pass, commit.
- **Test command:** `uv run pytest tests/<file> -v` (this project uses `uv`; `pyproject.toml` confirms pytest is the test runner).
- **Commit cadence:** one commit per task minimum, more if natural.
- **`weights_only=True`** when loading checkpoints, matching the pattern in `mri/cli/infer.py:31`.
- **No emojis in committed code.**

---

## Task 1: Package scaffolding + CLI skeleton

**Files:**
- Create: `mri/diagnostics/__init__.py`
- Create: `mri/cli/diagnose.py`
- Create: `tests/test_diagnose_cli.py`

- [ ] **Step 1: Write the failing test for run-dir validation**

```python
# tests/test_diagnose_cli.py
"""Smoke tests for `python -m mri.cli.diagnose` argparse + run-dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mri.cli.diagnose import resolve_run_dir, RunDirError


def test_resolve_run_dir_release_layout(tmp_path: Path) -> None:
    (tmp_path / "model_best.pt").write_bytes(b"")
    (tmp_path / "resolved_config.yaml").write_text("task:\n  name: segmentation\n")

    paths = resolve_run_dir(tmp_path)

    assert paths.checkpoint == tmp_path / "model_best.pt"
    assert paths.resolved_config == tmp_path / "resolved_config.yaml"


def test_resolve_run_dir_training_layout(tmp_path: Path) -> None:
    (tmp_path / "myrun_best.pt").write_bytes(b"")
    (tmp_path / "myrun_resolved_config.yaml").write_text("task:\n  name: segmentation\n")

    paths = resolve_run_dir(tmp_path)

    assert paths.checkpoint == tmp_path / "myrun_best.pt"
    assert paths.resolved_config == tmp_path / "myrun_resolved_config.yaml"


def test_resolve_run_dir_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "resolved_config.yaml").write_text("")
    with pytest.raises(RunDirError, match="No .*_best\\.pt"):
        resolve_run_dir(tmp_path)


def test_resolve_run_dir_missing_config(tmp_path: Path) -> None:
    (tmp_path / "model_best.pt").write_bytes(b"")
    with pytest.raises(RunDirError, match="resolved_config.yaml"):
        resolve_run_dir(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnose_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mri.cli.diagnose'`

- [ ] **Step 3: Create the empty package marker**

```python
# mri/diagnostics/__init__.py
"""Post-hoc segmentation diagnostics: error attribution + label audit."""
```

- [ ] **Step 4: Create the CLI module with run-dir resolution only**

```python
# mri/cli/diagnose.py
"""CLI entry point for segmentation diagnostics.

Usage::

    python -m mri.cli.diagnose <run_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class RunDirError(RuntimeError):
    """Raised when a run directory does not match either expected layout."""


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint: Path
    resolved_config: Path


def resolve_run_dir(run_dir: Path) -> RunPaths:
    """Locate the checkpoint and resolved config in a finished run directory.

    Supports two on-disk layouts:

    1. Release/UI: ``<run_dir>/*_best.pt`` + ``<run_dir>/resolved_config.yaml``.
    2. Training:   ``<run_dir>/<name>_best.pt`` + ``<run_dir>/<name>_resolved_config.yaml``.
    """
    if not run_dir.is_dir():
        raise RunDirError(f"Run directory does not exist: {run_dir}")

    best_candidates = sorted(run_dir.glob("*_best.pt"))
    if not best_candidates:
        raise RunDirError(f"No *_best.pt checkpoint found in {run_dir}")
    if len(best_candidates) > 1:
        raise RunDirError(
            f"Multiple *_best.pt checkpoints in {run_dir}: {[p.name for p in best_candidates]}"
        )
    checkpoint = best_candidates[0]

    exact = run_dir / "resolved_config.yaml"
    if exact.exists():
        config = exact
    else:
        cfg_candidates = sorted(run_dir.glob("*_resolved_config.yaml"))
        if not cfg_candidates:
            raise RunDirError(
                f"No resolved_config.yaml or *_resolved_config.yaml in {run_dir}"
            )
        config = cfg_candidates[0]

    return RunPaths(run_dir=run_dir, checkpoint=checkpoint, resolved_config=config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segmentation diagnostics for a finished run")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (containing *_best.pt + resolved_config.yaml)")
    parser.add_argument("--split", default="val", help="Split key (default: val)")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cached predictions exist")
    parser.add_argument("--include-low-priority", action="store_true", help="Include priority-3 audit cases in the report")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    print(f"[diagnose] checkpoint: {paths.checkpoint}")
    print(f"[diagnose] resolved_config: {paths.resolved_config}")
    print("[diagnose] (orchestration not yet wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnose_cli.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add mri/diagnostics/__init__.py mri/cli/diagnose.py tests/test_diagnose_cli.py
git commit -m "diagnostics: scaffold mri.diagnostics package + diagnose CLI run-dir resolver"
```

---

## Task 2: Error attribution math (`attribute.py`)

The lesion-channel metric math, in pure NumPy. No I/O. Operates on dense numpy volumes that the dump step will produce later.

**Files:**
- Create: `mri/diagnostics/attribute.py`
- Create: `tests/test_diagnostics_attribute.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics_attribute.py
"""Unit tests for per-case lesion-channel attribution math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mri.diagnostics.attribute import (
    CaseAttribution,
    attribute_case,
    aggregate_by_class,
)


def _zeros(shape=(4, 4, 4)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_perfect_prediction_has_dice_one_and_no_fp() -> None:
    gt_lesion = _zeros()
    gt_lesion[1, 1, 1] = 1
    gt_gland = np.ones_like(gt_lesion)
    pred_lesion_prob = (gt_lesion > 0).astype(np.float32)  # already 0/1
    pred_gland_prob = pred_lesion_prob.copy()

    out = attribute_case(
        case_id="c1",
        class_label=2,
        pred_lesion_prob=pred_lesion_prob,
        pred_gland_prob=pred_gland_prob,
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert out.dice == pytest.approx(1.0)
    assert out.precision == pytest.approx(1.0)
    assert out.recall == pytest.approx(1.0)
    assert out.fp_voxels_inside_gland == 0
    assert out.fp_voxels_outside_gland == 0
    assert out.fn_voxels == 0
    assert out.tp_voxels == 1
    assert out.fp_outside_ratio == pytest.approx(0.0)
    assert out.gland_dice == pytest.approx(1.0)
    assert out.lesion_volume_gt_voxels == 1
    assert out.class_label == 2
    assert out.status == "ok"


def test_fp_outside_gland_is_counted_separately() -> None:
    gt_lesion = _zeros()
    gt_lesion[2, 2, 2] = 1
    gt_gland = _zeros()
    gt_gland[1:3, 1:3, 1:3] = 1  # 8-voxel gland
    pred_lesion = _zeros()
    pred_lesion[2, 2, 2] = 1   # TP (inside gland)
    pred_lesion[0, 0, 0] = 1   # FP outside gland
    pred_lesion[1, 1, 1] = 1   # FP inside gland (not in GT lesion, but in gland)

    out = attribute_case(
        case_id="c2",
        class_label=3,
        pred_lesion_prob=pred_lesion,
        pred_gland_prob=gt_gland.astype(np.float32),
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert out.tp_voxels == 1
    assert out.fp_voxels_inside_gland == 1
    assert out.fp_voxels_outside_gland == 1
    assert out.fn_voxels == 0
    # ratio = 1 / (1 + 1 + 1) = 1/3
    assert out.fp_outside_ratio == pytest.approx(1.0 / 3.0)


def test_empty_gt_lesion_yields_nan_metrics() -> None:
    gt_lesion = _zeros()
    gt_gland = np.ones_like(gt_lesion)
    pred_lesion = _zeros()

    out = attribute_case(
        case_id="c3",
        class_label=0,
        pred_lesion_prob=pred_lesion,
        pred_gland_prob=gt_gland.astype(np.float32),
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert math.isnan(out.dice)
    assert math.isnan(out.precision)
    assert math.isnan(out.recall)
    # FP/FN/TP are still well-defined integers.
    assert out.tp_voxels == 0
    assert out.fp_voxels_inside_gland == 0
    assert out.fp_voxels_outside_gland == 0
    assert out.fn_voxels == 0
    assert out.lesion_volume_gt_voxels == 0


def test_aggregate_by_class_excludes_nan_cases() -> None:
    cases = [
        CaseAttribution(
            case_id="a", class_label=0, dice=float("nan"), precision=float("nan"),
            recall=float("nan"), fp_voxels_inside_gland=0, fp_voxels_outside_gland=0,
            fn_voxels=0, tp_voxels=0, fp_outside_ratio=float("nan"),
            gland_dice=1.0, lesion_volume_gt_voxels=0, status="ok",
        ),
        CaseAttribution(
            case_id="b", class_label=2, dice=0.4, precision=0.5, recall=0.3,
            fp_voxels_inside_gland=2, fp_voxels_outside_gland=1, fn_voxels=4,
            tp_voxels=2, fp_outside_ratio=0.2, gland_dice=0.9,
            lesion_volume_gt_voxels=6, status="ok",
        ),
        CaseAttribution(
            case_id="c", class_label=2, dice=0.6, precision=0.7, recall=0.5,
            fp_voxels_inside_gland=1, fp_voxels_outside_gland=0, fn_voxels=2,
            tp_voxels=2, fp_outside_ratio=0.0, gland_dice=0.85,
            lesion_volume_gt_voxels=4, status="ok",
        ),
    ]

    rows = aggregate_by_class(cases)

    by_class = {row["class_label"]: row for row in rows}
    # Class 0: only a NaN case → n_cases=1, mean_dice NaN
    assert by_class[0]["n_cases"] == 1
    assert math.isnan(by_class[0]["mean_dice"])
    # Class 2: cases b,c → mean_dice = 0.5
    assert by_class[2]["n_cases"] == 2
    assert by_class[2]["mean_dice"] == pytest.approx(0.5)
    assert by_class[2]["mean_precision"] == pytest.approx(0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics_attribute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mri.diagnostics.attribute'`

- [ ] **Step 3: Implement `attribute.py`**

```python
# mri/diagnostics/attribute.py
"""Per-case error attribution: lesion-channel metrics split by gland location.

Pure NumPy, no I/O beyond the optional CSV writer at the bottom. The CLI orchestrator
is responsible for loading numpy arrays from disk and passing them in.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class CaseAttribution:
    case_id: str
    class_label: int
    dice: float
    precision: float
    recall: float
    fp_voxels_inside_gland: int
    fp_voxels_outside_gland: int
    fn_voxels: int
    tp_voxels: int
    fp_outside_ratio: float
    gland_dice: float
    lesion_volume_gt_voxels: int
    status: str  # "ok" or "failed"


def _binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    return arr >= threshold


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two boolean volumes; NaN if both are empty."""
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return float("nan")
    inter = int(np.logical_and(a, b).sum())
    return (2.0 * inter) / (a_sum + b_sum)


def attribute_case(
    *,
    case_id: str,
    class_label: int,
    pred_lesion_prob: np.ndarray,
    pred_gland_prob: np.ndarray,
    gt_lesion: np.ndarray,
    gt_gland: np.ndarray,
    lesion_threshold: float,
    gland_threshold: float,
) -> CaseAttribution:
    """Compute per-case error attribution for the lesion channel.

    NaN policy: when ``gt_lesion`` is empty, dice/precision/recall/fp_outside_ratio
    are NaN (undefined), but TP/FP/FN voxel counts are still well-defined integers.
    """
    pred_lesion_bin = _binarize(pred_lesion_prob, lesion_threshold)
    pred_gland_bin = _binarize(pred_gland_prob, gland_threshold)
    gt_lesion_bin = gt_lesion.astype(bool)
    gt_gland_bin = gt_gland.astype(bool)

    tp = np.logical_and(pred_lesion_bin, gt_lesion_bin)
    fp = np.logical_and(pred_lesion_bin, np.logical_not(gt_lesion_bin))
    fn = np.logical_and(np.logical_not(pred_lesion_bin), gt_lesion_bin)

    fp_inside = np.logical_and(fp, gt_gland_bin)
    fp_outside = np.logical_and(fp, np.logical_not(gt_gland_bin))

    tp_n = int(tp.sum())
    fp_inside_n = int(fp_inside.sum())
    fp_outside_n = int(fp_outside.sum())
    fn_n = int(fn.sum())
    fp_total = fp_inside_n + fp_outside_n

    gt_volume = int(gt_lesion_bin.sum())

    if gt_volume == 0:
        dice = float("nan")
        precision = float("nan")
        recall = float("nan")
        fp_outside_ratio = float("nan")
    else:
        denom_dice = 2 * tp_n + fp_total + fn_n
        dice = (2 * tp_n) / denom_dice if denom_dice > 0 else float("nan")
        denom_p = tp_n + fp_total
        precision = (tp_n / denom_p) if denom_p > 0 else float("nan")
        denom_r = tp_n + fn_n
        recall = (tp_n / denom_r) if denom_r > 0 else float("nan")
        denom_ratio = fp_total + tp_n
        fp_outside_ratio = (fp_outside_n / denom_ratio) if denom_ratio > 0 else float("nan")

    gland_dice = _dice(pred_gland_bin, gt_gland_bin)

    return CaseAttribution(
        case_id=case_id,
        class_label=class_label,
        dice=dice,
        precision=precision,
        recall=recall,
        fp_voxels_inside_gland=fp_inside_n,
        fp_voxels_outside_gland=fp_outside_n,
        fn_voxels=fn_n,
        tp_voxels=tp_n,
        fp_outside_ratio=fp_outside_ratio,
        gland_dice=gland_dice,
        lesion_volume_gt_voxels=gt_volume,
        status="ok",
    )


def aggregate_by_class(cases: Iterable[CaseAttribution]) -> List[dict]:
    """Group attributions by class_label (0..4) and average the float metrics.

    NaN cases (empty GT) are excluded from the means but counted in ``n_cases``.
    """
    buckets: dict[int, List[CaseAttribution]] = {c: [] for c in range(5)}
    for case in cases:
        buckets.setdefault(case.class_label, []).append(case)

    rows: List[dict] = []
    for class_label in sorted(buckets):
        group = buckets[class_label]
        if not group:
            continue
        rows.append({
            "class_label": class_label,
            "n_cases": len(group),
            "mean_dice": _nanmean([c.dice for c in group]),
            "mean_precision": _nanmean([c.precision for c in group]),
            "mean_recall": _nanmean([c.recall for c in group]),
            "mean_fp_outside_ratio": _nanmean([c.fp_outside_ratio for c in group]),
            "mean_gland_dice": _nanmean([c.gland_dice for c in group]),
        })
    return rows


def _nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def write_metrics_by_case(cases: Iterable[CaseAttribution], path: Path) -> None:
    cases = list(cases)
    if not cases:
        path.write_text("")
        return
    fieldnames = list(asdict(cases[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(asdict(case))


def write_metrics_by_class(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnostics_attribute.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add mri/diagnostics/attribute.py tests/test_diagnostics_attribute.py
git commit -m "diagnostics: per-case error attribution math (FP inside/outside gland)"
```

---

## Task 3: Label-audit heuristics (`audit.py`)

Six heuristics, all conservative. One test per heuristic. Each must verify both a triggering and a non-triggering case.

**Files:**
- Create: `mri/diagnostics/audit.py`
- Create: `tests/test_diagnostics_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics_audit.py
"""Unit tests for label-audit heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from mri.diagnostics.audit import (
    AuditFinding,
    audit_case,
    AUDIT_DEFAULTS,
)


def _empty(shape=(6, 8, 8)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _flag_names(findings: list[AuditFinding]) -> set[str]:
    return {f.flag for f in findings}


def test_class_mask_inconsistent_class_positive_empty_mask() -> None:
    findings = audit_case(
        case_id="x", class_label=3,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=_empty(), gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" in _flag_names(findings)


def test_class_mask_inconsistent_class_zero_with_mask() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1
    findings = audit_case(
        case_id="x", class_label=0,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" in _flag_names(findings)


def test_class_mask_consistent_does_not_trigger() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" not in _flag_names(findings)


def test_high_confidence_disagreement_triggers_inside_gland() -> None:
    # 50-voxel block of high-prob pred inside gland with no GT lesion.
    pred = _empty()
    pred[2, 1:6, 1:6] = 0.9  # 25 voxels per slice
    pred[3, 1:6, 1:6] = 0.9  # +25 = 50 voxels, all inside gland
    gland = np.zeros((6, 8, 8), dtype=bool)
    gland[1:5, 0:7, 0:7] = True

    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=pred, pred_gland_prob=gland.astype(np.float32),
        gt_lesion=_empty(), gt_gland=gland,
    )
    assert "high_confidence_disagreement" in _flag_names(findings)


def test_high_confidence_disagreement_does_not_trigger_below_min_voxels() -> None:
    pred = _empty()
    pred[2, 1, 1] = 0.95  # only 1 voxel
    gland = np.ones((6, 8, 8), dtype=bool)

    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=pred, pred_gland_prob=gland.astype(np.float32),
        gt_lesion=_empty(), gt_gland=gland,
    )
    assert "high_confidence_disagreement" not in _flag_names(findings)


def test_tiny_gt_island_triggers() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1  # 1-voxel island
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "tiny_gt_island" in _flag_names(findings)


def test_tiny_gt_island_does_not_trigger_for_large_lesion() -> None:
    gt = _empty()
    gt[1:5, 1:5, 1:5] = 1  # 64-voxel block
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "tiny_gt_island" not in _flag_names(findings)


def test_erratic_slice_consistency_triggers_on_gap() -> None:
    gt = _empty()
    gt[1, 4, 4] = 1
    # gap at slice 2
    gt[3, 4, 4] = 1  # 1-voxel islands; ignore via tiny-island filter? No - this heuristic
    # operates on the GT slice presence regardless of size. Make islands big enough to not be tiny.
    gt = _empty()
    gt[1, 1:5, 1:5] = 1
    gt[3, 1:5, 1:5] = 1  # gap at slice 2
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "erratic_slice_consistency" in _flag_names(findings)


def test_erratic_slice_consistency_does_not_trigger_on_contiguous() -> None:
    gt = _empty()
    gt[1:4, 1:5, 1:5] = 1  # slices 1,2,3 contiguous
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "erratic_slice_consistency" not in _flag_names(findings)


def test_audit_findings_have_priority_and_reason() -> None:
    findings = audit_case(
        case_id="x", class_label=3,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=_empty(), gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert any(f.flag == "class_mask_inconsistent" and f.priority == 1 for f in findings)
    assert all(f.reason for f in findings)


def test_audit_defaults_are_documented() -> None:
    # These are the values the spec promises. If they change, both the docs and tests should update.
    assert AUDIT_DEFAULTS["high_conf_min_voxels"] == 50
    assert AUDIT_DEFAULTS["high_conf_min_prob"] == 0.8
    assert AUDIT_DEFAULTS["tiny_island_max_voxels"] == 10
    assert AUDIT_DEFAULTS["volume_outlier_pct"] == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `audit.py`**

```python
# mri/diagnostics/audit.py
"""Conservative label-audit heuristics for segmentation GT.

Each heuristic surfaces a candidate for human review. None of them auto-exclude.

Heuristics (priority 1 = highest, 3 = lowest):

  1. class_mask_inconsistent (priority 1)
  2. high_confidence_disagreement (priority 1)
  3. tiny_gt_island (priority 2)
  4. gt_volume_outlier (priority 3)  -- requires per-case volumes from the cohort
  5. erratic_slice_consistency (priority 2)
  6. class_severity_mismatch (priority 2) -- requires cohort-level mass distribution

Per-case heuristics live in ``audit_case``; cohort-level ones (4 and 6) live in
``audit_cohort`` so they can see all cases.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from scipy import ndimage


AUDIT_DEFAULTS = {
    "high_conf_min_voxels": 50,
    "high_conf_min_prob": 0.8,
    "tiny_island_max_voxels": 10,
    "volume_outlier_pct": 5.0,
    "severity_mismatch_outlier_pct": 5.0,
}


@dataclass(frozen=True)
class AuditFinding:
    case_id: str
    class_label: int
    flag: str
    priority: int
    reason: str


def _connected_components_3d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """3D 6-connectivity connected components."""
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled, n = ndimage.label(mask, structure=structure)
    return labeled, n


def _component_sizes(labeled: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=np.int64)
    return np.bincount(labeled.ravel())[1 : n + 1]


def _check_class_mask_inconsistent(
    case_id: str, class_label: int, gt_lesion: np.ndarray
) -> List[AuditFinding]:
    has_lesion = bool(gt_lesion.any())
    if class_label > 0 and not has_lesion:
        return [AuditFinding(
            case_id=case_id, class_label=class_label,
            flag="class_mask_inconsistent", priority=1,
            reason=f"class_label={class_label} but lesion mask is empty",
        )]
    if class_label == 0 and has_lesion:
        return [AuditFinding(
            case_id=case_id, class_label=class_label,
            flag="class_mask_inconsistent", priority=1,
            reason="class_label=0 but lesion mask is non-empty",
        )]
    return []


def _check_high_confidence_disagreement(
    case_id: str, class_label: int,
    pred_lesion_prob: np.ndarray, gt_lesion: np.ndarray, gt_gland: np.ndarray,
    *, min_voxels: int, min_prob: float,
) -> List[AuditFinding]:
    high_prob = pred_lesion_prob >= min_prob
    inside_gland = np.logical_and(high_prob, gt_gland)
    not_in_gt = np.logical_and(inside_gland, np.logical_not(gt_lesion.astype(bool)))
    labeled, n = _connected_components_3d(not_in_gt)
    if n == 0:
        return []
    sizes = _component_sizes(labeled, n)
    if not (sizes >= min_voxels).any():
        return []
    largest = int(sizes.max())
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="high_confidence_disagreement", priority=1,
        reason=f"pred lesion prob >= {min_prob} over {largest} voxels inside GT-gland with no GT-lesion overlap",
    )]


def _check_tiny_gt_island(
    case_id: str, class_label: int,
    gt_lesion: np.ndarray, *, max_voxels: int,
) -> List[AuditFinding]:
    labeled, n = _connected_components_3d(gt_lesion.astype(bool))
    if n == 0:
        return []
    sizes = _component_sizes(labeled, n)
    tiny = sizes[sizes < max_voxels]
    if tiny.size == 0:
        return []
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="tiny_gt_island", priority=2,
        reason=f"GT lesion has {tiny.size} connected component(s) smaller than {max_voxels} voxels",
    )]


def _check_erratic_slice_consistency(
    case_id: str, class_label: int, gt_lesion: np.ndarray,
) -> List[AuditFinding]:
    """GT lesion appears, disappears, and reappears across z-slices.

    Defined precisely: the set of z-slices with non-empty GT lesion has at least
    one gap of length >= 2 separating two non-empty slices.
    """
    per_slice = gt_lesion.reshape(gt_lesion.shape[0], -1).any(axis=1)
    nonzero_slices = np.where(per_slice)[0]
    if nonzero_slices.size < 2:
        return []
    gaps = np.diff(nonzero_slices)
    if not (gaps >= 2).any():
        return []
    max_gap = int(gaps.max())
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="erratic_slice_consistency", priority=2,
        reason=f"GT lesion has a z-gap of {max_gap} slices between non-empty slices",
    )]


def audit_case(
    *,
    case_id: str,
    class_label: int,
    pred_lesion_prob: np.ndarray,
    pred_gland_prob: np.ndarray,  # currently unused (placeholder for future heuristics)
    gt_lesion: np.ndarray,
    gt_gland: np.ndarray,
    defaults: dict | None = None,
) -> List[AuditFinding]:
    """Run all per-case heuristics. Cohort-level ones live in ``audit_cohort``."""
    cfg = {**AUDIT_DEFAULTS, **(defaults or {})}
    findings: List[AuditFinding] = []
    findings.extend(_check_class_mask_inconsistent(case_id, class_label, gt_lesion))
    findings.extend(_check_high_confidence_disagreement(
        case_id, class_label, pred_lesion_prob, gt_lesion, gt_gland.astype(bool),
        min_voxels=int(cfg["high_conf_min_voxels"]),
        min_prob=float(cfg["high_conf_min_prob"]),
    ))
    findings.extend(_check_tiny_gt_island(
        case_id, class_label, gt_lesion,
        max_voxels=int(cfg["tiny_island_max_voxels"]),
    ))
    findings.extend(_check_erratic_slice_consistency(case_id, class_label, gt_lesion))
    return findings


@dataclass(frozen=True)
class CohortCase:
    case_id: str
    class_label: int
    gt_lesion_volume: int
    pred_lesion_mass: float  # sum of prob over all voxels


def audit_cohort(
    cases: Sequence[CohortCase], defaults: dict | None = None,
) -> List[AuditFinding]:
    """Run cohort-level heuristics that need the full distribution.

    Heuristics:
    - gt_volume_outlier: top/bottom N% of non-empty GT volumes (priority 3)
    - class_severity_mismatch: pred mass outlier within a class (priority 2)
    """
    cfg = {**AUDIT_DEFAULTS, **(defaults or {})}
    findings: List[AuditFinding] = []

    # Volume outliers among non-empty GT cases.
    nonzero = [c for c in cases if c.gt_lesion_volume > 0]
    if len(nonzero) >= 2:
        volumes = np.array([c.gt_lesion_volume for c in nonzero], dtype=np.float64)
        pct = float(cfg["volume_outlier_pct"])
        lo = np.percentile(volumes, pct)
        hi = np.percentile(volumes, 100 - pct)
        for c, v in zip(nonzero, volumes):
            if v <= lo or v >= hi:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=c.class_label,
                    flag="gt_volume_outlier", priority=3,
                    reason=f"GT lesion volume {int(v)} voxels is in the outer {pct:.1f}% (range {lo:.0f}..{hi:.0f})",
                ))

    # Severity mismatch: pred mass outlier within each class bucket.
    by_class: dict[int, list[CohortCase]] = {}
    for c in cases:
        by_class.setdefault(c.class_label, []).append(c)
    pct = float(cfg["severity_mismatch_outlier_pct"])
    for class_label, group in by_class.items():
        if len(group) < 2:
            continue
        masses = np.array([c.pred_lesion_mass for c in group], dtype=np.float64)
        lo = np.percentile(masses, pct)
        hi = np.percentile(masses, 100 - pct)
        for c, m in zip(group, masses):
            if class_label == 1 and m >= hi:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=class_label,
                    flag="class_severity_mismatch", priority=2,
                    reason=f"class_label=1 but pred lesion mass ({m:.1f}) is in the top {pct:.1f}% of class 1",
                ))
            elif class_label == 4 and m <= lo:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=class_label,
                    flag="class_severity_mismatch", priority=2,
                    reason=f"class_label=4 but pred lesion mass ({m:.1f}) is in the bottom {pct:.1f}% of class 4",
                ))
    return findings


def write_audit_csv(findings: Iterable[AuditFinding], path: Path) -> None:
    rows = []
    by_case: dict[tuple[str, int], list[AuditFinding]] = {}
    for f in findings:
        by_case.setdefault((f.case_id, f.class_label), []).append(f)
    for (case_id, class_label), fs in sorted(by_case.items(), key=lambda kv: (min(f.priority for f in kv[1]), kv[0][0])):
        flags = ";".join(f.flag for f in fs)
        priority = min(f.priority for f in fs)
        reason = "; ".join(f.reason for f in fs)
        rows.append({
            "case_id": case_id,
            "class_label": class_label,
            "flags": flags,
            "priority": priority,
            "reason": reason,
        })
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnostics_audit.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add mri/diagnostics/audit.py tests/test_diagnostics_audit.py
git commit -m "diagnostics: label-audit heuristics (per-case + cohort)"
```

---

## Task 4: Per-case prediction dump (`dump.py`)

Loads the val split via the existing config + dataloader path, runs inference, and writes per-case `prob.npz`, `gt.npz`, `meta.json`. Skips cases that already have a non-empty `prob.npz` unless `force=True`.

**Files:**
- Create: `mri/diagnostics/dump.py`
- Create: `tests/test_diagnostics_dump.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics_dump.py
"""Smoke test for `dump_predictions` with a stub model and a tiny synthetic loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mri.diagnostics.dump import dump_predictions


class _StubModel(torch.nn.Module):
    """Returns logits that decode to ones in channel 0 and zeros in channel 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0  # gland prob ≈ 1
        return out


def _stub_dataloader(num_cases: int = 2, slices_per_case: int = 4):
    """Yield batches of (images, masks, meta_dict). Mimics SegmentationDataset's collate."""
    batches = []
    for case_idx in range(num_cases):
        for slice_idx in range(slices_per_case):
            images = torch.zeros(1, 5, 8, 8)
            masks = torch.zeros(1, 2, 8, 8)
            masks[0, 0, 2:6, 2:6] = 1  # gland GT
            if slice_idx == 2:
                masks[0, 1, 3:5, 3:5] = 1  # lesion GT only on slice 2
            meta = {
                "case_id": [f"case_{case_idx:02d}"],
                "slice_idx": [slice_idx],
                "class": [case_idx + 1],  # class label per case
            }
            batches.append((images, masks, meta))
    return batches


def test_dump_writes_per_case_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    model = _StubModel()
    loader = _stub_dataloader(num_cases=2, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4, "case_01": 4},
        spatial_shape=(8, 8),
    )

    assert summary["cases_written"] == 2
    for case_id in ("case_00", "case_01"):
        case_dir = out_dir / case_id
        assert (case_dir / "prob.npz").exists()
        assert (case_dir / "gt.npz").exists()
        assert (case_dir / "meta.json").exists()

        prob = np.load(case_dir / "prob.npz")
        assert prob["gland"].shape == (4, 8, 8)
        assert prob["lesion"].shape == (4, 8, 8)
        # gland channel decoded to ~1 by the stub
        assert np.all(prob["gland"] > 0.99)

        gt = np.load(case_dir / "gt.npz")
        assert gt["gland"].sum() > 0
        assert gt["lesion"].sum() > 0  # lesion present on slice 2

        meta = json.loads((case_dir / "meta.json").read_text())
        assert meta["case_id"] == case_id
        assert "class_label" in meta
        assert meta["spatial_shape"] == [8, 8]


def test_dump_skips_cached_cases(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    case_dir = out_dir / "case_00"
    case_dir.mkdir(parents=True)
    # Pre-existing non-empty prob.npz
    np.savez_compressed(case_dir / "prob.npz", gland=np.ones((4, 8, 8), dtype=np.float32),
                        lesion=np.zeros((4, 8, 8), dtype=np.float32))

    model = _StubModel()
    loader = _stub_dataloader(num_cases=1, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4},
        spatial_shape=(8, 8),
        force=False,
    )

    assert summary["cases_skipped_cached"] == 1
    # gt.npz should still not have been written, since we skipped this case
    assert not (case_dir / "gt.npz").exists()


def test_dump_force_overrides_cache(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    case_dir = out_dir / "case_00"
    case_dir.mkdir(parents=True)
    np.savez_compressed(case_dir / "prob.npz", gland=np.zeros((4, 8, 8), dtype=np.float32),
                        lesion=np.zeros((4, 8, 8), dtype=np.float32))

    model = _StubModel()
    loader = _stub_dataloader(num_cases=1, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4},
        spatial_shape=(8, 8),
        force=True,
    )

    assert summary["cases_skipped_cached"] == 0
    assert (case_dir / "gt.npz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics_dump.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `dump.py`**

```python
# mri/diagnostics/dump.py
"""Per-case prediction dump for finished segmentation runs.

Loads the val dataloader the same way the trainer / inference CLI does, runs
inference one batch at a time, accumulates per-case probability volumes, and
writes ``prob.npz`` + ``gt.npz`` + ``meta.json`` per case.

Pure orchestration over a model + dataloader: it does NOT know how to build
either of them. The CLI in ``mri/cli/diagnose.py`` is responsible for that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


def _coerce_meta_list(metas: Any) -> list[dict]:
    """The seg dataloader collates meta as a dict-of-lists OR a list-of-dicts.

    Mirror the handling in ``mri/inference/segmentation.py`` so behaviour matches.
    """
    if isinstance(metas, dict):
        first = metas[next(iter(metas))]
        return [{k: metas[k][i] for k in metas} for i in range(len(first))]
    return list(metas)


def dump_predictions(
    *,
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    output_dir: Path,
    num_slices_per_case: Mapping[str, int],
    spatial_shape: tuple[int, int],
    force: bool = False,
) -> Dict[str, Any]:
    """Run inference and write per-case artifacts under ``output_dir/<case_id>/``.

    Per-case files:
      - ``prob.npz``     keys: ``gland`` (Z,H,W float32), ``lesion`` (Z,H,W float32)
      - ``gt.npz``       keys: ``gland`` (Z,H,W uint8),   ``lesion`` (Z,H,W uint8)
      - ``meta.json``    fields: case_id, class_label, spatial_shape, num_slices, predicted_slices

    Args:
      num_slices_per_case: mapping of case_id -> total Z, used to allocate buffers.
        Pulled by the CLI from the metadata index.
      spatial_shape: (H, W) of the model output. Pulled from the first batch.
      force: when True, ignore any pre-existing ``prob.npz`` and re-dump.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: identify cases with an already-cached prob.npz.
    cached: set[str] = set()
    if not force:
        for case_id in num_slices_per_case:
            p = output_dir / case_id / "prob.npz"
            if p.exists() and p.stat().st_size > 0:
                cached.add(case_id)

    model = model.to(device)
    model.eval()

    h, w = spatial_shape
    buffers: Dict[str, Dict[str, Any]] = {}
    for case_id, n_z in num_slices_per_case.items():
        if case_id in cached:
            continue
        buffers[case_id] = {
            "gland_prob": np.zeros((n_z, h, w), dtype=np.float32),
            "lesion_prob": np.zeros((n_z, h, w), dtype=np.float32),
            "gland_gt": np.zeros((n_z, h, w), dtype=np.uint8),
            "lesion_gt": np.zeros((n_z, h, w), dtype=np.uint8),
            "predicted_slices": set(),
            "class_label": None,
        }

    with torch.no_grad():
        for batch in dataloader:
            images, masks, metas = batch[0], batch[1], batch[2]
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            meta_list = _coerce_meta_list(metas)
            mask_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks

            for i, m in enumerate(meta_list):
                case_id = str(m["case_id"])
                if case_id not in buffers:
                    continue  # cached or not in scope
                slice_idx = int(m["slice_idx"])
                buf = buffers[case_id]
                buf["gland_prob"][slice_idx] = probs[i, 0]
                if probs.shape[1] > 1:
                    buf["lesion_prob"][slice_idx] = probs[i, 1]
                buf["gland_gt"][slice_idx] = (mask_np[i, 0] > 0.5).astype(np.uint8)
                if mask_np.shape[1] > 1:
                    buf["lesion_gt"][slice_idx] = (mask_np[i, 1] > 0.5).astype(np.uint8)
                buf["predicted_slices"].add(slice_idx)
                if buf["class_label"] is None and "class" in m:
                    cls = m["class"]
                    if cls is not None:
                        buf["class_label"] = int(cls)

    # Write artifacts.
    cases_written = 0
    for case_id, buf in buffers.items():
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "prob.npz",
            gland=buf["gland_prob"],
            lesion=buf["lesion_prob"],
        )
        np.savez_compressed(
            case_dir / "gt.npz",
            gland=buf["gland_gt"],
            lesion=buf["lesion_gt"],
        )
        meta_doc = {
            "case_id": case_id,
            "class_label": buf["class_label"] if buf["class_label"] is not None else 0,
            "spatial_shape": list(spatial_shape),
            "num_slices": int(buf["gland_prob"].shape[0]),
            "predicted_slices": sorted(int(s) for s in buf["predicted_slices"]),
        }
        (case_dir / "meta.json").write_text(json.dumps(meta_doc, indent=2))
        cases_written += 1

    return {
        "cases_written": cases_written,
        "cases_skipped_cached": len(cached),
        "output_dir": str(output_dir),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnostics_dump.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mri/diagnostics/dump.py tests/test_diagnostics_dump.py
git commit -m "diagnostics: per-case prediction dump (prob.npz + gt.npz + meta.json)"
```

---

## Task 5: HTML report (`report.py` + template)

**Files:**
- Create: `mri/diagnostics/report.py`
- Create: `mri/diagnostics/templates/diagnostic_report.html.j2`

This task does not get a TDD round of unit tests — the report is a rendered artifact and would be brittle to pin. We verify it via the end-to-end smoke in Task 6 (it generates and the file is non-empty / contains expected anchors).

- [ ] **Step 1: Create the Jinja template**

```html
{# mri/diagnostics/templates/diagnostic_report.html.j2 #}
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diagnostic — {{ run_name }}</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
  table { border-collapse: collapse; margin: 1rem 0; }
  td, th { border: 1px solid #ccc; padding: 0.4rem 0.6rem; }
  th { background: #f4f4f4; }
  .dice-good { background: #d9f7d9; }
  .dice-bad  { background: #f7d9d9; }
  .case { border-top: 1px solid #ddd; padding-top: 1rem; margin-top: 2rem; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
  .grid img { width: 100%; }
  .meta { color: #555; font-size: 0.9rem; }
  .flag { display: inline-block; background: #ffeec8; padding: 0 0.4rem; margin-right: 0.3rem; border-radius: 3px; }
  h2 { margin-top: 2.5rem; }
</style>
</head>
<body>
<h1>Diagnostic — {{ run_name }}</h1>
<p class="meta">
  Checkpoint: <code>{{ checkpoint_path }}</code><br>
  Split: <code>{{ split }}</code> &middot; Cases: {{ n_cases }} &middot; Generated: {{ generated_at }}
</p>
<p>
  Overall lesion Dice: <b>{{ overall.dice }}</b> &middot;
  Overall precision: <b>{{ overall.precision }}</b> &middot;
  Overall gland Dice: <b>{{ overall.gland_dice }}</b>
  {% if n_failed %}<br>{{ n_failed }} case(s) skipped (inference failure).{% endif %}
</p>

<h2>Per-class breakdown</h2>
<table>
<tr><th>class</th><th>n</th><th>mean Dice</th><th>mean precision</th><th>mean recall</th><th>mean FP-outside ratio</th><th>mean gland Dice</th></tr>
{% for row in class_table %}
<tr>
  <td>{{ row.class_label }}</td>
  <td>{{ row.n_cases }}</td>
  <td class="{{ row.dice_class }}">{{ row.mean_dice }}</td>
  <td>{{ row.mean_precision }}</td>
  <td>{{ row.mean_recall }}</td>
  <td>{{ row.mean_fp_outside_ratio }}</td>
  <td>{{ row.mean_gland_dice }}</td>
</tr>
{% endfor %}
</table>

<h2>Audit queue</h2>
{% if audit_rows %}
<table>
<tr><th>priority</th><th>case</th><th>class</th><th>flags</th><th>reason</th></tr>
{% for row in audit_rows %}
<tr>
  <td>{{ row.priority }}</td>
  <td><a href="#case-{{ row.case_id }}">{{ row.case_id }}</a></td>
  <td>{{ row.class_label }}</td>
  <td>{% for flag in row.flag_list %}<span class="flag">{{ flag }}</span>{% endfor %}</td>
  <td>{{ row.reason }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p><i>No audit flags fired on this run.</i></p>
{% endif %}

<h2>Per-case detail</h2>
{% for case in cases %}
<div class="case" id="case-{{ case.case_id }}">
  <h3>{{ case.case_id }} <span class="meta">(class {{ case.class_label }}, Dice {{ case.dice }}, FP-outside ratio {{ case.fp_outside_ratio }})</span></h3>
  {% if case.flag_list %}<p>{% for flag in case.flag_list %}<span class="flag">{{ flag }}</span>{% endfor %}</p>{% endif %}
  <div class="grid">
    {% for panel in case.panels %}
    <div>
      <div class="meta">slice {{ panel.slice_idx }} — {{ panel.title }}</div>
      <img src="data:image/png;base64,{{ panel.png_b64 }}" alt="slice {{ panel.slice_idx }} {{ panel.title }}">
    </div>
    {% endfor %}
  </div>
</div>
{% endfor %}

{% if worst_unflagged %}
<h2>Worst cases without audit flags</h2>
<p class="meta">Bottom decile by lesion Dice with no audit findings — likely real model errors.</p>
<table>
<tr><th>case</th><th>class</th><th>Dice</th><th>FP-outside ratio</th></tr>
{% for row in worst_unflagged %}
<tr>
  <td><a href="#case-{{ row.case_id }}">{{ row.case_id }}</a></td>
  <td>{{ row.class_label }}</td>
  <td>{{ row.dice }}</td>
  <td>{{ row.fp_outside_ratio }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

</body>
</html>
```

- [ ] **Step 2: Implement `report.py`**

```python
# mri/diagnostics/report.py
"""Render the diagnostic HTML report from the dumped artifacts and CSV outputs."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mri.diagnostics.attribute import CaseAttribution
from mri.diagnostics.audit import AuditFinding


def _format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


def _dice_class(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value > 0.5:
        return "dice-good"
    if value < 0.2:
        return "dice-bad"
    return ""


def _png_b64_from_array(rgb: np.ndarray) -> str:
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 1) * 255
    arr = arr.astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _heatmap(prob: np.ndarray) -> np.ndarray:
    """Quick red-channel heatmap, no colormap dependency."""
    base = _gray_to_rgb(np.zeros_like(prob, dtype=np.float32))
    overlay = base.copy()
    overlay[..., 0] = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    return overlay


def _disagreement_panel(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """TP=green, FP=red, FN=blue, on black background."""
    h, w = pred_bin.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin))
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin)
    rgb[tp] = (0, 200, 0)
    rgb[fp] = (220, 0, 0)
    rgb[fn] = (0, 100, 220)
    return rgb


def _gt_overlay(gt_lesion: np.ndarray) -> np.ndarray:
    return _heatmap(gt_lesion.astype(np.float32))


def _pick_central_slice(gt_lesion: np.ndarray, pred_lesion_prob: np.ndarray) -> int:
    """Pick the z-slice with the most GT lesion mass; fall back to predicted mass."""
    per_slice_gt = gt_lesion.reshape(gt_lesion.shape[0], -1).sum(axis=1)
    if per_slice_gt.max() > 0:
        return int(np.argmax(per_slice_gt))
    per_slice_pred = pred_lesion_prob.reshape(pred_lesion_prob.shape[0], -1).sum(axis=1)
    return int(np.argmax(per_slice_pred))


def _build_panels(
    pred_lesion_prob: np.ndarray, gt_lesion: np.ndarray, lesion_threshold: float,
) -> list[dict]:
    z_star = _pick_central_slice(gt_lesion, pred_lesion_prob)
    z_max = pred_lesion_prob.shape[0] - 1
    slice_idxs = sorted({max(0, z_star - 1), z_star, min(z_max, z_star + 1)})

    panels = []
    for z in slice_idxs:
        gt_panel = _gt_overlay(gt_lesion[z])
        prob_panel = _heatmap(pred_lesion_prob[z])
        disagreement = _disagreement_panel(
            pred_lesion_prob[z] >= lesion_threshold,
            gt_lesion[z].astype(bool),
        )
        for title, panel in (("GT", gt_panel), ("pred prob", prob_panel), ("disagreement", disagreement)):
            panels.append({
                "slice_idx": z,
                "title": title,
                "png_b64": _png_b64_from_array(panel),
            })
    return panels


@dataclass(frozen=True)
class CaseArtifact:
    case_id: str
    class_label: int
    pred_lesion_prob: np.ndarray
    gt_lesion: np.ndarray


def render_report(
    *,
    output_path: Path,
    template_path: Path,
    run_name: str,
    checkpoint_path: Path,
    split: str,
    lesion_threshold: float,
    case_attributions: Sequence[CaseAttribution],
    audit_findings: Sequence[AuditFinding],
    case_artifacts: dict[str, CaseArtifact],
    include_low_priority: bool = False,
) -> None:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template(template_path.name)

    n_cases = len(case_attributions)
    n_failed = sum(1 for c in case_attributions if c.status == "failed")

    overall = {
        "dice": _format_metric(_safe_mean([c.dice for c in case_attributions])),
        "precision": _format_metric(_safe_mean([c.precision for c in case_attributions])),
        "gland_dice": _format_metric(_safe_mean([c.gland_dice for c in case_attributions])),
    }

    from mri.diagnostics.attribute import aggregate_by_class
    class_rows = aggregate_by_class(case_attributions)
    class_table = []
    for row in class_rows:
        class_table.append({
            "class_label": row["class_label"],
            "n_cases": row["n_cases"],
            "mean_dice": _format_metric(row["mean_dice"]),
            "mean_precision": _format_metric(row["mean_precision"]),
            "mean_recall": _format_metric(row["mean_recall"]),
            "mean_fp_outside_ratio": _format_metric(row["mean_fp_outside_ratio"]),
            "mean_gland_dice": _format_metric(row["mean_gland_dice"]),
            "dice_class": _dice_class(row["mean_dice"]),
        })

    by_case_findings: dict[str, list[AuditFinding]] = {}
    for f in audit_findings:
        by_case_findings.setdefault(f.case_id, []).append(f)

    audit_rows = []
    rendered_case_ids: list[str] = []
    for case_id, fs in sorted(by_case_findings.items(), key=lambda kv: (min(f.priority for f in kv[1]), kv[0])):
        priority = min(f.priority for f in fs)
        if priority == 3 and not include_low_priority:
            continue
        audit_rows.append({
            "priority": priority,
            "case_id": case_id,
            "class_label": fs[0].class_label,
            "flag_list": [f.flag for f in fs],
            "reason": "; ".join(f.reason for f in fs),
        })
        rendered_case_ids.append(case_id)

    by_case_attrs = {c.case_id: c for c in case_attributions}
    cases = []
    for case_id in rendered_case_ids:
        attr = by_case_attrs.get(case_id)
        artifact = case_artifacts.get(case_id)
        if attr is None or artifact is None:
            continue
        cases.append({
            "case_id": case_id,
            "class_label": attr.class_label,
            "dice": _format_metric(attr.dice),
            "fp_outside_ratio": _format_metric(attr.fp_outside_ratio),
            "flag_list": [f.flag for f in by_case_findings.get(case_id, [])],
            "panels": _build_panels(artifact.pred_lesion_prob, artifact.gt_lesion, lesion_threshold),
        })

    # Worst-cases-without-flags: bottom-decile Dice with no findings.
    flagged_set = set(by_case_findings)
    valid = [c for c in case_attributions if not (isinstance(c.dice, float) and math.isnan(c.dice))]
    valid.sort(key=lambda c: c.dice)
    if valid:
        cutoff = valid[max(0, len(valid) // 10 - 1)].dice
        worst_unflagged_cases = [
            c for c in valid if c.dice <= cutoff and c.case_id not in flagged_set
        ][:5]
    else:
        worst_unflagged_cases = []
    worst_unflagged = [
        {
            "case_id": c.case_id,
            "class_label": c.class_label,
            "dice": _format_metric(c.dice),
            "fp_outside_ratio": _format_metric(c.fp_outside_ratio),
        }
        for c in worst_unflagged_cases
    ]

    rendered = template.render(
        run_name=run_name,
        checkpoint_path=str(checkpoint_path),
        split=split,
        n_cases=n_cases,
        n_failed=n_failed,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        overall=overall,
        class_table=class_table,
        audit_rows=audit_rows,
        cases=cases,
        worst_unflagged=worst_unflagged,
    )
    output_path.write_text(rendered)


def _safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))
```

- [ ] **Step 3: Smoke-render the report locally to confirm it imports and runs**

Run:
```bash
uv run python -c "
from pathlib import Path
import numpy as np
from mri.diagnostics.attribute import CaseAttribution
from mri.diagnostics.audit import AuditFinding
from mri.diagnostics.report import render_report, CaseArtifact

cases = [
    CaseAttribution(case_id='c1', class_label=2, dice=0.4, precision=0.5,
                    recall=0.3, fp_voxels_inside_gland=2, fp_voxels_outside_gland=1,
                    fn_voxels=4, tp_voxels=2, fp_outside_ratio=0.2, gland_dice=0.9,
                    lesion_volume_gt_voxels=6, status='ok'),
]
findings = [AuditFinding(case_id='c1', class_label=2, flag='tiny_gt_island',
                         priority=2, reason='test')]
artifacts = {'c1': CaseArtifact(case_id='c1', class_label=2,
                                pred_lesion_prob=np.zeros((4, 8, 8), dtype=np.float32),
                                gt_lesion=np.zeros((4, 8, 8), dtype=np.uint8))}
render_report(
    output_path=Path('/tmp/diag.html'),
    template_path=Path('mri/diagnostics/templates/diagnostic_report.html.j2'),
    run_name='smoke', checkpoint_path=Path('/tmp/x_best.pt'), split='val',
    lesion_threshold=0.5, case_attributions=cases, audit_findings=findings,
    case_artifacts=artifacts,
)
print('rendered:', Path('/tmp/diag.html').stat().st_size, 'bytes')
"
```

Expected: prints `rendered: <some kbytes>` with a non-zero size, no exception.

- [ ] **Step 4: Add jinja2 dependency check**

`gradio` already pulls jinja2 into the env. Verify with: `uv run python -c "import jinja2; print(jinja2.__version__)"`.

If the import fails, add `jinja2` to the `dependencies` list in `pyproject.toml` (at the same level as `gradio>=4.0.0`):
```
"jinja2>=3.0",
```
then `uv sync`.

- [ ] **Step 5: Commit**

```bash
git add mri/diagnostics/report.py mri/diagnostics/templates/diagnostic_report.html.j2
git commit -m "diagnostics: HTML report renderer (Jinja template + matplotlib-free panels)"
```

---

## Task 6: CLI orchestration end-to-end (`mri/cli/diagnose.py`)

Wire the four components in `main()`. Build the val dataloader exactly the way `mri/cli/infer.py` does. Add an end-to-end CLI smoke test that monkeypatches the model + dataloader.

**Files:**
- Modify: `mri/cli/diagnose.py`
- Modify: `tests/test_diagnose_cli.py` (extend)

- [ ] **Step 1: Extend `tests/test_diagnose_cli.py` with an end-to-end smoke test**

Append to `tests/test_diagnose_cli.py`:

```python
import sys
from unittest.mock import patch

import numpy as np
import torch

from mri.cli import diagnose


def _stub_dataloader_factory():
    def loader():
        for case_idx in range(2):
            for slice_idx in range(3):
                images = torch.zeros(1, 5, 8, 8)
                masks = torch.zeros(1, 2, 8, 8)
                masks[0, 0, 2:6, 2:6] = 1
                if slice_idx == 1 and case_idx == 0:
                    masks[0, 1, 3:5, 3:5] = 1
                meta = {
                    "case_id": [f"case_{case_idx:02d}"],
                    "slice_idx": [slice_idx],
                    "class": [case_idx + 1],
                }
                yield (images, masks, meta)
    return list(loader())


class _StubModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0
        return out


def test_diagnose_main_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "task:\n  name: segmentation\n"
        "metrics:\n  segmentation_threshold: 0.5\n"
        "data:\n  metadata: dummy\n  split_file: dummy\n  num_workers: 0\n  stack_depth: 5\n"
        "model:\n  name: simple_unet\n  params: {}\n"
        "inference:\n  batch_size: 1\n  device: cpu\n"
    )

    def fake_build(cfg, split):
        cases = {"case_00": 3, "case_01": 3}
        return _stub_dataloader_factory(), cases, (8, 8)

    with patch.object(diagnose, "_build_model_and_dataloader", fake_build), \
         patch.object(diagnose, "_load_checkpoint", lambda model, path, device: None), \
         patch.object(diagnose, "_create_segmentation_model", lambda name, **params: _StubModel()):
        rc = diagnose.main([str(run_dir), "--split", "val"])

    assert rc == 0
    diag = run_dir / "diagnostic"
    assert (diag / "predictions" / "case_00" / "prob.npz").exists()
    assert (diag / "metrics_by_case.csv").exists()
    assert (diag / "metrics_by_class.csv").exists()
    assert (diag / "report.html").exists()
    html = (diag / "report.html").read_text()
    assert "Diagnostic" in html
    assert "case_00" in html or "case_01" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnose_cli.py::test_diagnose_main_end_to_end -v`
Expected: FAIL — `_build_model_and_dataloader` not yet defined.

- [ ] **Step 3: Implement orchestration in `mri/cli/diagnose.py`**

Replace the placeholder `main()` body. Full file:

```python
"""CLI entry point for segmentation diagnostics.

Usage::

    python -m mri.cli.diagnose <run_dir>
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch


class RunDirError(RuntimeError):
    """Raised when a run directory does not match either expected layout."""


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint: Path
    resolved_config: Path


def resolve_run_dir(run_dir: Path) -> RunPaths:
    if not run_dir.is_dir():
        raise RunDirError(f"Run directory does not exist: {run_dir}")
    best_candidates = sorted(run_dir.glob("*_best.pt"))
    if not best_candidates:
        raise RunDirError(f"No *_best.pt checkpoint found in {run_dir}")
    if len(best_candidates) > 1:
        raise RunDirError(
            f"Multiple *_best.pt checkpoints in {run_dir}: {[p.name for p in best_candidates]}"
        )
    checkpoint = best_candidates[0]
    exact = run_dir / "resolved_config.yaml"
    if exact.exists():
        config = exact
    else:
        cfg_candidates = sorted(run_dir.glob("*_resolved_config.yaml"))
        if not cfg_candidates:
            raise RunDirError(
                f"No resolved_config.yaml or *_resolved_config.yaml in {run_dir}"
            )
        config = cfg_candidates[0]
    return RunPaths(run_dir=run_dir, checkpoint=checkpoint, resolved_config=config)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load weights using the same convention as ``mri/cli/infer.py``."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)


def _create_segmentation_model(name: str, **params):
    """Indirection for monkeypatching in tests; mirrors mri.models.create_segmentation_model."""
    from mri.models import create_segmentation_model
    return create_segmentation_model(name, **params)


def _build_model_and_dataloader(cfg: Dict[str, Any], split: str):
    """Build the val dataloader the same way ``mri/cli/infer.py:_build_dataloader`` does.

    Returns (dataloader, num_slices_per_case, spatial_shape).
    """
    from torch.utils.data import DataLoader
    from mri.data.metadata import load_metadata
    from mri.data.index_builders import build_segmentation_index, load_split_file
    from mri.data.datasets.segmentation import SegmentationDataset

    meta = load_metadata(cfg["data"]["metadata"])
    splits = load_split_file(cfg["data"]["split_file"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    stack_depth = cfg["data"].get("stack_depth", meta.config.get("t2_context_window", 5))

    split_index = build_segmentation_index(meta, splits[split])
    ds = SegmentationDataset(
        metadata_path=cfg["data"]["metadata"],
        samples_index=split_index,
        stack_depth=stack_depth,
        normalize=True,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("inference", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=num_workers,
    )

    # Per-case slice counts come from metadata.
    num_slices_per_case = {}
    for sample in split_index:
        cid = sample["case_id"]
        if cid not in num_slices_per_case:
            num_slices_per_case[cid] = int(meta.cases[cid]["num_slices"])

    # Spatial shape is fixed at 256x256 by SegmentationDataset (see _load_image).
    spatial_shape = (256, 256)
    return loader, num_slices_per_case, spatial_shape


def _resolve_lesion_threshold(cfg: Dict[str, Any]) -> float:
    metrics_cfg = cfg.get("metrics", {}) or {}
    threshold = metrics_cfg.get("segmentation_threshold")
    if threshold is None:
        warnings.warn("No metrics.segmentation_threshold in resolved_config — falling back to 0.5", stacklevel=2)
        return 0.5
    return float(threshold)


def _load_per_case_artifact(predictions_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    case_dir = predictions_dir / case_id
    prob = np.load(case_dir / "prob.npz")
    gt = np.load(case_dir / "gt.npz")
    import json
    meta = json.loads((case_dir / "meta.json").read_text())
    return prob["gland"], prob["lesion"], gt["gland"], gt["lesion"], int(meta.get("class_label", 0))


def main(argv: Sequence[str] | None = None) -> int:
    import yaml
    from mri.diagnostics.attribute import (
        attribute_case, write_metrics_by_case, write_metrics_by_class, aggregate_by_class,
    )
    from mri.diagnostics.audit import audit_case, audit_cohort, write_audit_csv, CohortCase
    from mri.diagnostics.dump import dump_predictions
    from mri.diagnostics.report import render_report, CaseArtifact

    parser = argparse.ArgumentParser(description="Segmentation diagnostics for a finished run")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (containing *_best.pt + resolved_config.yaml)")
    parser.add_argument("--split", default="val", help="Split key (default: val)")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cached predictions exist")
    parser.add_argument("--include-low-priority", action="store_true", help="Include priority-3 audit cases in the report")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    print(f"[diagnose] checkpoint: {paths.checkpoint}")
    print(f"[diagnose] resolved_config: {paths.resolved_config}")

    cfg = yaml.safe_load(paths.resolved_config.read_text()) or {}
    device = torch.device(args.device)
    lesion_threshold = _resolve_lesion_threshold(cfg)
    gland_threshold = lesion_threshold  # same operating point unless future spec splits them

    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    diag_root.mkdir(parents=True, exist_ok=True)

    loader, num_slices_per_case, spatial_shape = _build_model_and_dataloader(cfg, args.split)
    model = _create_segmentation_model(cfg["model"]["name"], **(cfg["model"].get("params") or {}))
    _load_checkpoint(model, paths.checkpoint, device)

    dump_summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=device,
        output_dir=predictions_dir,
        num_slices_per_case=num_slices_per_case,
        spatial_shape=spatial_shape,
        force=args.force,
    )
    print(f"[diagnose] dump: {dump_summary}")

    # Attribution + audit pass.
    case_attrs = []
    findings = []
    cohort_cases = []
    artifacts: dict[str, CaseArtifact] = {}
    for case_id in num_slices_per_case:
        case_dir = predictions_dir / case_id
        if not (case_dir / "prob.npz").exists():
            continue
        gland_prob, lesion_prob, gland_gt, lesion_gt, class_label = _load_per_case_artifact(predictions_dir, case_id)
        attr = attribute_case(
            case_id=case_id, class_label=class_label,
            pred_lesion_prob=lesion_prob, pred_gland_prob=gland_prob,
            gt_lesion=lesion_gt, gt_gland=gland_gt,
            lesion_threshold=lesion_threshold, gland_threshold=gland_threshold,
        )
        case_attrs.append(attr)
        findings.extend(audit_case(
            case_id=case_id, class_label=class_label,
            pred_lesion_prob=lesion_prob, pred_gland_prob=gland_prob,
            gt_lesion=lesion_gt, gt_gland=gland_gt,
        ))
        cohort_cases.append(CohortCase(
            case_id=case_id, class_label=class_label,
            gt_lesion_volume=int(lesion_gt.sum()),
            pred_lesion_mass=float(lesion_prob.sum()),
        ))
        artifacts[case_id] = CaseArtifact(
            case_id=case_id, class_label=class_label,
            pred_lesion_prob=lesion_prob, gt_lesion=lesion_gt,
        )
    findings.extend(audit_cohort(cohort_cases))

    write_metrics_by_case(case_attrs, diag_root / "metrics_by_case.csv")
    write_metrics_by_class(aggregate_by_class(case_attrs), diag_root / "metrics_by_class.csv")
    write_audit_csv(findings, diag_root / "label_audit.csv")

    template_path = Path(__file__).resolve().parents[1] / "diagnostics" / "templates" / "diagnostic_report.html.j2"
    render_report(
        output_path=diag_root / "report.html",
        template_path=template_path,
        run_name=paths.checkpoint.stem.removesuffix("_best"),
        checkpoint_path=paths.checkpoint,
        split=args.split,
        lesion_threshold=lesion_threshold,
        case_attributions=case_attrs,
        audit_findings=findings,
        case_artifacts=artifacts,
        include_low_priority=args.include_low_priority,
    )
    print(f"[diagnose] report: {diag_root / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diagnose_cli.py -v`
Expected: 5 passed (4 from Task 1 + 1 new end-to-end).

- [ ] **Step 5: Run the full diagnostics test suite**

Run: `uv run pytest tests/test_diagnose_cli.py tests/test_diagnostics_attribute.py tests/test_diagnostics_audit.py tests/test_diagnostics_dump.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add mri/cli/diagnose.py tests/test_diagnose_cli.py
git commit -m "diagnostics: wire CLI orchestration end-to-end (dump+attribute+audit+report)"
```

---

## Task 7: Documentation

Add a short doc page that points future readers (including future-you) at the tool. Update the docs index.

**Files:**
- Create: `docs/diagnostic.md`
- Modify: `docs/README.md`

- [ ] **Step 1: Create `docs/diagnostic.md`**

```markdown
# Segmentation diagnostics

Post-hoc error attribution + label audit for a finished segmentation run.

## Usage

```bash
uv run python -m mri.cli.diagnose path/to/run_dir
```

Optional flags:

- `--split val` — dataloader split key (default `val`).
- `--force` — re-run inference even if `diagnostic/predictions/<case_id>/prob.npz` is cached.
- `--include-low-priority` — include priority-3 audit findings in the HTML report.
- `--device cuda` — override torch device (default `cpu`).

## Outputs

Written to `<run_dir>/diagnostic/`:

- `predictions/<case_id>/{prob.npz, gt.npz, meta.json}` — per-case raw artifacts.
- `metrics_by_case.csv` — per-case lesion-channel metrics, including FP-inside vs FP-outside-gland counts.
- `metrics_by_class.csv` — same metrics aggregated by 0–4 class label.
- `label_audit.csv` — flagged cases ranked by priority (1 high → 3 low).
- `report.html` — single-file report tying it all together.

## Heuristics

The audit surfaces — never auto-excludes — cases that match conservative noise patterns:

| Flag | Priority | Meaning |
|---|---|---|
| `class_mask_inconsistent`     | 1 | Class label disagrees with mask presence (class>0 + empty mask, or class=0 + non-empty mask). |
| `high_confidence_disagreement`| 1 | Pred lesion prob ≥ 0.8 over a 3D component ≥ 50 voxels inside GT-gland with no GT-lesion overlap. |
| `tiny_gt_island`              | 2 | GT lesion has a 3D component < 10 voxels. |
| `erratic_slice_consistency`   | 2 | GT lesion has a z-gap ≥ 2 slices between non-empty slices. |
| `class_severity_mismatch`     | 2 | Pred lesion mass is an outlier within its 0–4 class bucket. |
| `gt_volume_outlier`           | 3 | GT lesion volume is in the top or bottom 5% of non-empty cases. |

Defaults are hardcoded in `mri/diagnostics/audit.py:AUDIT_DEFAULTS`. Promote to config only if a follow-up actually changes them.

## Design

Spec: [`docs/superpowers/specs/2026-04-25-segmentation-error-analysis-design.md`](superpowers/specs/2026-04-25-segmentation-error-analysis-design.md).
```

- [ ] **Step 2: Add a link in `docs/README.md`**

Add a line in the existing list of docs (next to `inference.md` is a sensible spot):

```
- [docs/diagnostic.md](diagnostic.md) — post-hoc error attribution and label audit
```

- [ ] **Step 3: Commit**

```bash
git add docs/diagnostic.md docs/README.md
git commit -m "docs: add diagnostic.md describing the segmentation diagnostics CLI"
```

---

## Self-review checklist (run after all tasks complete)

- [ ] `uv run pytest tests/test_diagnostics_attribute.py tests/test_diagnostics_audit.py tests/test_diagnostics_dump.py tests/test_diagnose_cli.py -v` — all green.
- [ ] `uv run python -m mri.cli.diagnose --help` — prints argparse help.
- [ ] `uv run python -m mri.cli.diagnose checkpoints/default` (or another real run dir) — produces a `diagnostic/` folder with all four CSVs/HTML and no exception. (Note: requires the val split file referenced by the resolved config to actually exist on disk.)
- [ ] `report.html` opens in a browser; the per-class table renders; the audit queue links jump to per-case sections.
- [ ] No new files committed under `runs/` or `checkpoints/`.
- [ ] No changes to existing source files outside the new `mri/diagnostics/` package and `mri/cli/diagnose.py`.
