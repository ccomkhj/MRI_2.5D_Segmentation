"""Postprocess + 3D visualization stage for the inference pipeline.

Bridges the on-disk format that ``mri/cli/infer.py`` writes
(``<case>/{prostate_prob.npy, target_prob.npy}``) with the gland-constrain
+ no-prostate-suppress rules from ``mri.diagnostics.postprocess`` and the
per-case Plotly figure builder in ``mri.diagnostics.visualization``. Designed
for use by ``mri/cli/pipeline_infer.py`` after segmentation inference and
before classification.

Important name mapping: ``prostate_prob.npy`` carries the gland channel and
``target_prob.npy`` carries the lesion channel. We pass them into
``apply_postprocess`` accordingly so rules 1+2 (lesion outside gland masked;
no gland ⇒ all lesion zeroed) apply to the inference output.

No torch dependency. Pure NumPy + Plotly + JSON I/O.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mri.diagnostics.postprocess import apply_postprocess
from mri.diagnostics.visualization import (
    build_case_figure, write_case_html,
)


def process_case(
    case_dir: Path,
    *,
    lesion_threshold: float,
    gland_threshold: float,
    write_3d: bool = True,
    use_cdn: bool = False,
    case_id: str | None = None,
) -> dict[str, Any] | None:
    """Postprocess one inference case dir + (optionally) write its 3D HTML.

    Reads ``prostate_prob.npy`` (gland channel) and ``target_prob.npy``
    (lesion channel). Writes back into the same ``case_dir``:

    - ``lesion_mask_postprocessed.npy`` — uint8 (Z,H,W) gland-constrained
      lesion mask (rules 1+2 applied).
    - ``gland_mask.npy`` — uint8 (Z,H,W) binarized prostate mask.
    - ``postprocess_meta.json`` — thresholds, voxel counts, gland_present.
    - ``visual_3d.html`` — self-contained Plotly figure (only if
      ``write_3d=True``); the GT traces are omitted because no GT is
      available at inference time, so only the predicted-lesion trace
      renders.

    Args:
      case_dir: per-case directory under
          ``<run_root>/predictions/segmentation/<case_id>``.
      lesion_threshold: probability threshold for the lesion mask (>=).
      gland_threshold: probability threshold for the gland mask (>=).
      write_3d: when False, ``visual_3d.html`` is not produced.
      use_cdn: pass-through to ``write_case_html`` — when True the Plotly
          JS is loaded from a CDN at view time.
      case_id: full case identifier to record in ``postprocess_meta.json``
          and the HTML title. Defaults to ``case_dir.name`` (which is
          wrong for nested layouts like ``class3/case_0310``).

    Returns:
      Per-case stat dict, or ``None`` if either probability file is missing.
    """
    if case_id is None:
        case_id = case_dir.name

    prostate_path = case_dir / "prostate_prob.npy"
    target_path = case_dir / "target_prob.npy"
    if not (prostate_path.exists() and target_path.exists()):
        warnings.warn(
            f"[postprocess_visualize] {case_id}: missing prostate_prob.npy "
            f"or target_prob.npy, skipping.",
            stacklevel=2,
        )
        return None

    gland_prob = np.load(prostate_path)
    lesion_prob = np.load(target_path)

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob,
        lesion_threshold=lesion_threshold,
        gland_threshold=gland_threshold,
    )

    np.save(case_dir / "lesion_mask_postprocessed.npy", lesion_mask)
    np.save(case_dir / "gland_mask.npy", gland_mask)

    lesion_voxels_raw = int((lesion_prob >= lesion_threshold).sum())
    lesion_voxels_post = int(lesion_mask.sum())
    gland_voxels = int(gland_mask.sum())
    (case_dir / "postprocess_meta.json").write_text(json.dumps({
        "case_id": case_id,
        "lesion_threshold": lesion_threshold,
        "gland_threshold": gland_threshold,
        "gland_present": gland_present,
        "lesion_voxels_raw": lesion_voxels_raw,
        "lesion_voxels_post": lesion_voxels_post,
        "gland_voxels": gland_voxels,
    }, indent=2))

    html_written = False
    if write_3d:
        fig = build_case_figure(
            gt_gland=np.zeros_like(gland_mask),
            gt_lesion_components=[],
            pred_lesion=lesion_mask,
            downsample=1,
        )
        write_case_html(
            fig, case_dir / "visual_3d.html",
            header_meta={
                "case_id": case_id,
                "gland_present": gland_present,
                "lesion_voxels_post": lesion_voxels_post,
            },
            use_cdn=use_cdn,
        )
        html_written = True

    return {
        "case_id": case_id,
        "gland_present": gland_present,
        "lesion_voxels_raw": lesion_voxels_raw,
        "lesion_voxels_post": lesion_voxels_post,
        "gland_voxels": gland_voxels,
        "html_written": html_written,
    }


def run_stage(
    seg_pred_root: Path,
    *,
    lesion_threshold: float,
    gland_threshold: float,
    write_3d: bool,
    use_cdn: bool,
) -> dict[str, Any]:
    """Sweep every case dir under ``seg_pred_root`` and run ``process_case``.

    Case dirs are detected via ``rglob("prostate_prob.npy")`` to support
    both flat case_ids (``case_a``) and nested ones (``class3/case_0310``).

    Returns a summary dict aggregating counts across the cohort.
    """
    seg_pred_root = Path(seg_pred_root)
    prostate_paths = sorted(seg_pred_root.rglob("prostate_prob.npy"))

    cases_processed = 0
    cases_skipped = 0
    gland_present_count = 0
    html_written = 0
    lesion_voxels_raw_total = 0
    lesion_voxels_post_total = 0

    for prostate_path in prostate_paths:
        case_dir = prostate_path.parent
        case_id = case_dir.relative_to(seg_pred_root).as_posix()
        stats = process_case(
            case_dir,
            lesion_threshold=lesion_threshold,
            gland_threshold=gland_threshold,
            write_3d=write_3d,
            use_cdn=use_cdn,
            case_id=case_id,
        )
        if stats is None:
            cases_skipped += 1
            continue
        cases_processed += 1
        if stats["gland_present"]:
            gland_present_count += 1
        if stats["html_written"]:
            html_written += 1
        lesion_voxels_raw_total += stats["lesion_voxels_raw"]
        lesion_voxels_post_total += stats["lesion_voxels_post"]

    return {
        "cases_processed": cases_processed,
        "cases_skipped": cases_skipped,
        "gland_present_count": gland_present_count,
        "html_written": html_written,
        "lesion_voxels_raw_total": lesion_voxels_raw_total,
        "lesion_voxels_post_total": lesion_voxels_post_total,
        "lesion_threshold": lesion_threshold,
        "gland_threshold": gland_threshold,
    }
