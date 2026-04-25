"""CLI entry point for segmentation diagnostics.

Usage::

    python -m mri.cli.diagnose <run_dir>
"""

from __future__ import annotations

import argparse
import json
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
        if len(cfg_candidates) > 1:
            raise RunDirError(
                f"Multiple *_resolved_config.yaml files in {run_dir}: {[p.name for p in cfg_candidates]}"
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

    Returns (dataloader, num_slices_per_case). The spatial shape is discovered
    from the first batch by ``dump_predictions``, not hardcoded here.
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

    return loader, num_slices_per_case


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
    meta = json.loads((case_dir / "meta.json").read_text())
    return prob["gland"], prob["lesion"], gt["gland"], gt["lesion"], int(meta.get("class_label", 0))


def main(argv: Sequence[str] | None = None) -> int:
    import yaml
    from mri.diagnostics.attribute import (
        attribute_case, write_metrics_by_case, write_metrics_by_class, aggregate_by_class,
        CaseAttribution,
    )
    from mri.diagnostics.audit import audit_case, audit_cohort, write_audit_csv, CohortCase
    from mri.diagnostics.dump import dump_predictions
    from mri.diagnostics.report import render_report, CaseArtifact

    parser = argparse.ArgumentParser(description="Segmentation diagnostics for a finished run")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (containing *_best.pt + resolved_config.yaml or *_resolved_config.yaml)")
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

    loader, num_slices_per_case = _build_model_and_dataloader(cfg, args.split)
    model = _create_segmentation_model(cfg["model"]["name"], **(cfg["model"].get("params") or {}))
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
    print(f"[diagnose] dump: {dump_summary}")
    failed_case_ids = set(dump_summary.get("cases_failed_inference", []))

    # Attribution + audit pass.
    case_attrs = []
    findings = []
    cohort_cases = []
    artifacts: dict[str, CaseArtifact] = {}
    for case_id in num_slices_per_case:
        case_dir = predictions_dir / case_id
        if not (case_dir / "prob.npz").exists():
            if case_id in failed_case_ids:
                case_attrs.append(CaseAttribution(
                    case_id=case_id, class_label=0,
                    dice=float("nan"), precision=float("nan"), recall=float("nan"),
                    fp_voxels_inside_gland=0, fp_voxels_outside_gland=0,
                    fn_voxels=0, tp_voxels=0,
                    fp_outside_ratio=float("nan"),
                    gland_dice=float("nan"),
                    lesion_volume_gt_voxels=0,
                    status="failed",
                ))
            continue
        try:
            gland_prob, lesion_prob, gland_gt, lesion_gt, class_label = _load_per_case_artifact(predictions_dir, case_id)
            attr = attribute_case(
                case_id=case_id, class_label=class_label,
                pred_lesion_prob=lesion_prob, pred_gland_prob=gland_prob,
                gt_lesion=lesion_gt, gt_gland=gland_gt,
                lesion_threshold=lesion_threshold, gland_threshold=gland_threshold,
            )
            case_findings = audit_case(
                case_id=case_id, class_label=class_label,
                pred_lesion_prob=lesion_prob, pred_gland_prob=gland_prob,
                gt_lesion=lesion_gt, gt_gland=gland_gt,
            )
            cohort_case = CohortCase(
                case_id=case_id, class_label=class_label,
                gt_lesion_volume=int(lesion_gt.sum()),
                pred_lesion_mass=float(lesion_prob.sum()),
            )
            artifact = CaseArtifact(
                case_id=case_id, class_label=class_label,
                pred_lesion_prob=lesion_prob, gt_lesion=lesion_gt,
            )
        except Exception as exc:  # noqa: BLE001 — per-case error isolation
            warnings.warn(
                f"[diagnose] skipped {case_id}: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            case_attrs.append(CaseAttribution(
                case_id=case_id, class_label=0,
                dice=float("nan"), precision=float("nan"), recall=float("nan"),
                fp_voxels_inside_gland=0, fp_voxels_outside_gland=0,
                fn_voxels=0, tp_voxels=0,
                fp_outside_ratio=float("nan"),
                gland_dice=float("nan"),
                lesion_volume_gt_voxels=0,
                status="failed",
            ))
            continue
        case_attrs.append(attr)
        findings.extend(case_findings)
        cohort_cases.append(cohort_case)
        artifacts[case_id] = artifact
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
