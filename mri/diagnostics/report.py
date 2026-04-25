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

from mri.diagnostics.attribute import CaseAttribution, aggregate_by_class
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
    h, w = prob.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
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
            "panels": _build_panels(artifact.pred_lesion_prob, artifact.gt_lesion, lesion_threshold),
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


def _safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))
