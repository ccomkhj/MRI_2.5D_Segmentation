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

from mri.diagnostics.attribute import CaseAttribution, aggregate_by_class, nanmean
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


def png_b64_from_array(rgb: np.ndarray) -> str:
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


def _load_t2_slice(metadata_root: Path, case_id: str, slice_idx: int, shape: tuple[int, int]) -> np.ndarray | None:
    """Load a single T2 slice as a uint8 grayscale array sized to ``shape``.

    Returns None when the file is missing — callers should fall back to a black background.
    Mirrors ``mri/inference/segmentation.py:_load_case_t2_slice`` so we render the same
    image the model saw at inference time.
    """
    image_path = metadata_root / case_id / "t2" / f"{slice_idx:04d}.png"
    if not image_path.exists():
        return None
    from PIL import Image

    image = Image.open(image_path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)


def base_rgb(t2: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Build the H,W,3 base image for a panel — T2 grayscale if available, else black."""
    if t2 is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    return np.stack([t2, t2, t2], axis=-1).astype(np.uint8)


def alpha_blend(base: np.ndarray, color: tuple[int, int, int], alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend a single color over a base image. ``alpha`` is H,W in [0, 1]."""
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = (1.0 - a) * base.astype(np.float32) + a * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _heatmap(prob: np.ndarray, t2: np.ndarray | None = None) -> np.ndarray:
    """Red probability heatmap, optionally alpha-blended over a T2 grayscale base."""
    base = base_rgb(t2, prob.shape)
    # Slightly compressed alpha so even mid-confidence regions are visible.
    alpha = np.clip(prob, 0.0, 1.0) * 0.85
    return alpha_blend(base, (255, 0, 0), alpha)


def _disagreement_panel(
    pred_bin: np.ndarray, gt_bin: np.ndarray, t2: np.ndarray | None = None,
) -> np.ndarray:
    """TP=green, FP=red, FN=cyan, alpha-blended over a T2 grayscale base."""
    base = base_rgb(t2, pred_bin.shape)
    out = base.copy()
    tp = np.logical_and(pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin))
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin)
    if tp.any():
        out = alpha_blend(out, (0, 200, 0), tp.astype(np.float32) * 0.6)
    if fp.any():
        out = alpha_blend(out, (220, 0, 0), fp.astype(np.float32) * 0.6)
    if fn.any():
        out = alpha_blend(out, (0, 180, 220), fn.astype(np.float32) * 0.6)
    return out


def _gt_overlay(gt_lesion: np.ndarray, t2: np.ndarray | None = None) -> np.ndarray:
    """GT lesion shown as a cyan tint over a T2 grayscale base (matches html_report.py convention)."""
    base = base_rgb(t2, gt_lesion.shape)
    alpha = (gt_lesion.astype(bool).astype(np.float32)) * 0.6
    return alpha_blend(base, (0, 200, 255), alpha)


def _pick_central_slices(
    gt_lesion: np.ndarray,
    pred_lesion_prob: np.ndarray,
    lesion_threshold: float,
) -> list[int]:
    """Pick anchor z-slices for the per-case grid.

    Returns up to two anchors — one driven by GT mass, one by predicted mass —
    so cases where the model fires in completely different z-positions than the
    GT (a real failure mode) are still visible. Each anchor expands to ``a-1, a, a+1``;
    the union is returned, sorted and deduped.
    """
    z_max_idx = pred_lesion_prob.shape[0] - 1
    anchors: list[int] = []

    per_slice_gt = gt_lesion.reshape(gt_lesion.shape[0], -1).sum(axis=1)
    if per_slice_gt.max() > 0:
        anchors.append(int(np.argmax(per_slice_gt)))

    pred_above = (pred_lesion_prob >= lesion_threshold).reshape(pred_lesion_prob.shape[0], -1).sum(axis=1)
    if pred_above.max() > 0:
        pred_anchor = int(np.argmax(pred_above))
        if pred_anchor not in anchors:
            anchors.append(pred_anchor)

    if not anchors:
        # Truly nothing — fall back to whichever channel has the most mass at all.
        per_slice_pred_mass = pred_lesion_prob.reshape(pred_lesion_prob.shape[0], -1).sum(axis=1)
        anchors.append(int(np.argmax(per_slice_pred_mass)))

    expanded: set[int] = set()
    for a in anchors:
        for offset in (-1, 0, 1):
            expanded.add(max(0, min(z_max_idx, a + offset)))
    return sorted(expanded)


def _build_panels(
    pred_lesion_prob: np.ndarray,
    gt_lesion: np.ndarray,
    lesion_threshold: float,
    *,
    case_id: str | None = None,
    metadata_root: Path | None = None,
) -> list[dict]:
    slice_idxs = _pick_central_slices(gt_lesion, pred_lesion_prob, lesion_threshold)
    spatial = (pred_lesion_prob.shape[1], pred_lesion_prob.shape[2])

    panels = []
    for z in slice_idxs:
        t2 = None
        if metadata_root is not None and case_id is not None:
            t2 = _load_t2_slice(metadata_root, case_id, z, spatial)
        gt_panel = _gt_overlay(gt_lesion[z], t2)
        prob_panel = _heatmap(pred_lesion_prob[z], t2)
        disagreement = _disagreement_panel(
            pred_lesion_prob[z] >= lesion_threshold,
            gt_lesion[z].astype(bool),
            t2,
        )
        for title, panel in (("GT", gt_panel), ("pred prob", prob_panel), ("disagreement", disagreement)):
            panels.append({
                "slice_idx": z,
                "title": title,
                "png_b64": png_b64_from_array(panel),
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
    metadata_root: Path | None = None,
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
        "dice": _format_metric(nanmean([c.dice for c in case_attributions])),
        "precision": _format_metric(nanmean([c.precision for c in case_attributions])),
        "gland_dice": _format_metric(nanmean([c.gland_dice for c in case_attributions])),
    }

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

    # Worst-cases-without-flags: bottom-decile Dice with no findings.
    flagged_set = set(by_case_findings)
    valid = [c for c in case_attributions if not (isinstance(c.dice, float) and math.isnan(c.dice))]
    valid.sort(key=lambda c: c.dice)
    if valid:
        k = max(1, len(valid) // 10)
        cutoff = valid[k - 1].dice
        worst_unflagged_cases = [
            c for c in valid if c.dice <= cutoff and c.case_id not in flagged_set
        ][:5]
    else:
        worst_unflagged_cases = []

    cases_to_render: list[str] = list(rendered_case_ids)
    for c in worst_unflagged_cases:
        if c.case_id not in cases_to_render:
            cases_to_render.append(c.case_id)

    by_case_attrs = {c.case_id: c for c in case_attributions}
    cases = []
    for case_id in cases_to_render:
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
            "panels": _build_panels(
                artifact.pred_lesion_prob,
                artifact.gt_lesion,
                lesion_threshold,
                case_id=case_id,
                metadata_root=metadata_root,
            ),
        })

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
    output_path.write_text(rendered, encoding="utf-8")
