"""Checkpoint-driven segmentation-to-classification inference runner."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mri.cli.postprocess import resolve_postprocess_thresholds
from mri.config.loader import load_config
from mri.experiments.runtime import utc_now_iso, write_json, write_yaml
from mri.inference.postprocess_visualize import run_stage as run_postprocess_visualize_stage


def _set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _build_override_config(
    *,
    base_config_path: Path,
    generated_config_path: Path,
    overrides: Dict[str, Any],
) -> Path:
    payload: Dict[str, Any] = {
        "extends": [str(base_config_path.resolve())],
    }
    for key, value in overrides.items():
        _set_nested_value(payload, key, value)
    write_yaml(generated_config_path, payload)
    load_config(generated_config_path)
    return generated_config_path


def _csv_values(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_cli(fn, args: List[str], stage_name: str) -> None:
    exit_code = fn(args)
    if exit_code != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {exit_code}")


def _stage_record(name: str, status: str, **payload: Any) -> Dict[str, Any]:
    record = {"name": name, "status": status}
    record.update(payload)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run segmentation inference and downstream classification inference from existing checkpoints"
    )
    parser.add_argument(
        "--seg-config",
        default="mri/config/task/segmentation.yaml",
        help="Segmentation config that matches the segmentation checkpoint architecture",
    )
    parser.add_argument(
        "--cls-config",
        default="mri/config/task/classification.yaml",
        help="Classification config that matches the classification checkpoint architecture",
    )
    parser.add_argument(
        "--seg-checkpoint",
        required=True,
        help="Existing segmentation checkpoint used to generate probability maps",
    )
    parser.add_argument(
        "--cls-checkpoint",
        required=True,
        help="Existing classification checkpoint used for final case-level prediction",
    )
    parser.add_argument(
        "--seg-inference-splits",
        help="Comma-separated segmentation inference splits. Defaults to the classification inference split.",
    )
    parser.add_argument(
        "--cls-inference-split",
        default="test",
        help="Classification inference split",
    )
    parser.add_argument(
        "--run-name",
        help="Pipeline inference run name. Defaults to pipeline-infer-<timestamp>",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/pipeline_inference",
        help="Root directory for generated configs, manifests, and predictions",
    )
    parser.add_argument("--device", help="Device override passed to inference CLI")
    parser.add_argument(
        "--lesion-threshold",
        type=float,
        default=None,
        help="Lesion (target) probability threshold for postprocessing. Defaults to "
             "metrics.segmentation_threshold from the segmentation config.",
    )
    parser.add_argument(
        "--gland-threshold",
        type=float,
        default=None,
        help="Gland (prostate) probability threshold for postprocessing. Defaults to "
             "metrics.segmentation_threshold from the segmentation config.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip the postprocess + 3D visualization stage entirely.",
    )
    parser.add_argument(
        "--no-3d-visuals",
        action="store_true",
        help="Run postprocess but skip the per-case visual_3d.html emission.",
    )
    parser.add_argument(
        "--plotly-cdn",
        action="store_true",
        help="Load Plotly from CDN in per-case HTML instead of inlining (~3 MB smaller per file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare configs and manifest without launching inference",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seg_base_config = Path(args.seg_config).resolve()
    cls_base_config = Path(args.cls_config).resolve()
    seg_checkpoint = Path(args.seg_checkpoint).resolve()
    cls_checkpoint = Path(args.cls_checkpoint).resolve()
    if not seg_base_config.exists():
        raise FileNotFoundError(f"Segmentation config not found: {seg_base_config}")
    if not cls_base_config.exists():
        raise FileNotFoundError(f"Classification config not found: {cls_base_config}")
    if not seg_checkpoint.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {seg_checkpoint}")
    if not cls_checkpoint.exists():
        raise FileNotFoundError(f"Classification checkpoint not found: {cls_checkpoint}")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = args.run_name or f"pipeline-infer-{timestamp}"
    run_root = Path(args.output_root) / run_name
    configs_dir = run_root / "configs"
    manifests_dir = run_root / "manifests"
    seg_pred_root = run_root / "predictions" / "segmentation"
    cls_pred_root = run_root / "predictions" / "classification"
    for path in (configs_dir, manifests_dir, seg_pred_root, cls_pred_root):
        path.mkdir(parents=True, exist_ok=True)

    seg_splits = _csv_values(args.seg_inference_splits) if args.seg_inference_splits else [args.cls_inference_split]
    if not seg_splits:
        raise ValueError("At least one segmentation inference split is required.")
    if args.cls_inference_split not in seg_splits:
        raise ValueError(
            f"Classification split '{args.cls_inference_split}' must also be included in --seg-inference-splits."
        )

    seg_config_path = configs_dir / "segmentation.yaml"
    cls_config_path = configs_dir / "classification.yaml"
    manifest_path = manifests_dir / "pipeline_infer_manifest.json"

    _build_override_config(
        base_config_path=seg_base_config,
        generated_config_path=seg_config_path,
        overrides={
            "experiment.tags": ["pipeline_inference"],
            "inference.checkpoint": str(seg_checkpoint),
            "inference.output_dir": str(seg_pred_root.resolve()),
        },
    )
    _build_override_config(
        base_config_path=cls_base_config,
        generated_config_path=cls_config_path,
        overrides={
            "experiment.tags": ["pipeline_inference"],
            "data.seg_pred_dir": str(seg_pred_root.resolve()),
            "inference.checkpoint": str(cls_checkpoint),
            "inference.output_dir": str(cls_pred_root.resolve()),
        },
    )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "status": "dry_run" if args.dry_run else "running",
        "name": run_name,
        "run_root": str(run_root.resolve()),
        "segmentation_base_config": str(seg_base_config),
        "classification_base_config": str(cls_base_config),
        "segmentation_checkpoint": str(seg_checkpoint),
        "classification_checkpoint": str(cls_checkpoint),
        "segmentation_splits": seg_splits,
        "classification_split": args.cls_inference_split,
        "stages": [
            _stage_record(
                "config_generation",
                "completed",
                segmentation_config=str(seg_config_path),
                classification_config=str(cls_config_path),
            )
        ],
    }
    write_json(manifest_path, manifest)

    infer_common_args: List[str] = []
    if args.device:
        infer_common_args.extend(["--device", args.device])

    for split_name in seg_splits:
        seg_infer_run_name = f"{run_name}-seg-{split_name}"
        seg_infer_args = [
            "--config",
            str(seg_config_path),
            "--split",
            split_name,
            "--checkpoint",
            str(seg_checkpoint),
            "--output_dir",
            str(seg_pred_root.resolve()),
            "--run_name",
            seg_infer_run_name,
            *infer_common_args,
        ]
        manifest["stages"].append(
            _stage_record(
                f"segmentation_inference_{split_name}",
                "planned" if args.dry_run else "running",
                command=seg_infer_args,
            )
        )

    seg_resolved_cfg = load_config(seg_config_path)
    lesion_threshold, gland_threshold = resolve_postprocess_thresholds(
        seg_resolved_cfg,
        lesion_arg=args.lesion_threshold,
        gland_arg=args.gland_threshold,
    )
    postprocess_status = "skipped" if args.no_postprocess else (
        "planned" if args.dry_run else "running"
    )
    manifest["stages"].append(
        _stage_record(
            "postprocess_visualize",
            postprocess_status,
            output_root=str(seg_pred_root.resolve()),
            lesion_threshold=lesion_threshold,
            gland_threshold=gland_threshold,
            html_enabled=not args.no_3d_visuals,
            artifact_pattern="<case>/{lesion_mask_postprocessed.npy, gland_mask.npy, postprocess_meta.json, visual_3d.html}",
        )
    )

    cls_infer_run_name = f"{run_name}-cls-{args.cls_inference_split}"
    cls_infer_args = [
        "--config",
        str(cls_config_path),
        "--split",
        args.cls_inference_split,
        "--checkpoint",
        str(cls_checkpoint),
        "--output_dir",
        str(cls_pred_root.resolve()),
        "--run_name",
        cls_infer_run_name,
        *infer_common_args,
    ]
    manifest["stages"].append(
        _stage_record(
            "classification_inference",
            "planned" if args.dry_run else "running",
            command=cls_infer_args,
        )
    )
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(f"Pipeline inference dry-run prepared: {run_root}")
        print(f"Manifest: {manifest_path}")
        print(f"Seg config: {seg_config_path}")
        print(f"Cls config: {cls_config_path}")
        print(f"Seg predictions: {seg_pred_root}")
        print(f"Cls predictions: {cls_pred_root}")
        return 0

    from mri.cli.infer import main as infer_main

    for index, split_name in enumerate(seg_splits, start=1):
        seg_infer_run_name = f"{run_name}-seg-{split_name}"
        seg_infer_args = manifest["stages"][index]["command"]
        _run_cli(infer_main, seg_infer_args, f"segmentation inference ({split_name})")
        manifest["stages"][index]["status"] = "completed"
        manifest["stages"][index]["summary_path"] = str(seg_pred_root / f"{seg_infer_run_name}_inference_summary.json")
        manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, manifest)

    postprocess_stage_index = len(seg_splits) + 1
    if not args.no_postprocess:
        summary = run_postprocess_visualize_stage(
            seg_pred_root,
            lesion_threshold=lesion_threshold,
            gland_threshold=gland_threshold,
            write_3d=not args.no_3d_visuals,
            use_cdn=args.plotly_cdn,
        )
        manifest["stages"][postprocess_stage_index].update({
            "status": "completed",
            "cases_processed": summary["cases_processed"],
            "cases_skipped": summary["cases_skipped"],
            "gland_present_count": summary["gland_present_count"],
            "html_written": summary["html_written"],
            "lesion_voxels_raw_total": summary["lesion_voxels_raw_total"],
            "lesion_voxels_post_total": summary["lesion_voxels_post_total"],
        })
        manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, manifest)

    cls_stage_index = len(manifest["stages"]) - 1
    _run_cli(infer_main, cls_infer_args, f"classification inference ({args.cls_inference_split})")
    manifest["stages"][cls_stage_index]["status"] = "completed"
    manifest["stages"][cls_stage_index]["summary_path"] = str(
        cls_pred_root / f"{cls_infer_run_name}_inference_summary.json"
    )
    manifest["stages"][cls_stage_index]["predictions_csv"] = str(cls_pred_root / "predictions.csv")
    manifest["status"] = "completed"
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)

    print(f"Pipeline inference completed: {run_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Seg predictions: {seg_pred_root}")
    print(f"Cls predictions: {cls_pred_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
