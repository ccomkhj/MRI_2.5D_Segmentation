"""Inference entrypoint for the scalable MRI pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from mri.config.loader import load_config
from mri.data.metadata import load_metadata
from mri.data.index_builders import load_split_file, build_segmentation_index, build_classification_index
from mri.data.datasets.segmentation import SegmentationDataset
from mri.data.datasets.classification import ClassificationDataset, find_missing_segmentation_predictions
from mri.experiments.runtime import build_run_manifest, finalize_run_manifest, write_json, write_yaml
from mri.models import create_segmentation_model, create_classification_model
from mri.training.trainer import resolve_device
from mri.inference.segmentation import run_segmentation_inference
from mri.inference.classification import run_classification_inference


def _validate_classification_inputs(seg_pred_dir: str | None, case_ids: list[str], context: str) -> None:
    if not seg_pred_dir:
        raise ValueError(f"data.seg_pred_dir must be set for {context}.")

    missing = find_missing_segmentation_predictions(seg_pred_dir, case_ids)
    if not missing:
        return

    preview = ", ".join(missing[:5])
    extra = len(missing) - min(len(missing), 5)
    suffix = f", ... (+{extra} more)" if extra > 0 else ""
    raise FileNotFoundError(
        f"Missing segmentation predictions for {len(missing)} case(s) in {seg_pred_dir} while preparing {context}: "
        f"{preview}{suffix}\n"
        "Run segmentation inference first to populate that directory."
    )


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)


def _build_dataloader(cfg: Dict[str, Any], task_name: str, split_key: str):
    meta = load_metadata(cfg["data"]["metadata"])
    splits = load_split_file(cfg["data"]["split_file"])
    num_workers = cfg["data"].get("num_workers", 4)

    if task_name == "segmentation":
        stack_depth = cfg["data"].get("stack_depth", meta.config.get("t2_context_window", 5))
        split_index = build_segmentation_index(meta, splits[split_key])
        ds = SegmentationDataset(
            metadata_path=cfg["data"]["metadata"],
            samples_index=split_index,
            stack_depth=stack_depth,
            normalize=True,
        )
        return DataLoader(ds, batch_size=cfg["inference"]["batch_size"], shuffle=False, num_workers=num_workers)

    if task_name == "classification":
        split_index = build_classification_index(meta, splits[split_key])
        _validate_classification_inputs(
            cfg["data"].get("seg_pred_dir"),
            [record["case_id"] for record in split_index],
            f"classification inference split '{split_key}'",
        )
        ds = ClassificationDataset(
            metadata_path=cfg["data"]["metadata"],
            cases_index=split_index,
            seg_pred_dir=cfg["data"]["seg_pred_dir"],
            depth=cfg["data"]["depth"]["depth"],
            crop_size=cfg["data"]["roi"]["crop_size"],
            output_size=cfg["data"]["roi"]["output_size"],
            modalities=tuple(cfg["data"]["modalities"]),
            zero_pad_missing=cfg["data"].get("zero_pad_missing", True),
            selection_source="pred",
            selection_jitter=0,
            min_prob=cfg["data"]["selection"].get("min_prob", 0.3),
            use_roi=cfg["data"]["roi"].get("use_roi", True),
            normalize=True,
        )
        return DataLoader(ds, batch_size=cfg["inference"]["batch_size"], shuffle=False, num_workers=num_workers)

    raise ValueError(f"Unknown task: {task_name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MRI unified inference CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--split", default="test", help="Split key to run inference on")
    parser.add_argument("--checkpoint", help="Override inference.checkpoint")
    parser.add_argument("--output_dir", help="Override inference.output_dir")
    parser.add_argument("--run_name", help="Override experiment.name / generated inference run name")
    parser.add_argument("--batch_size", type=int, help="Override inference.batch_size")
    parser.add_argument("--device", help="Override inference.device")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["inference"]["checkpoint"] = args.checkpoint
    if args.output_dir:
        cfg["inference"]["output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["inference"]["batch_size"] = args.batch_size
    if args.device:
        cfg["inference"]["device"] = args.device
    if args.run_name:
        cfg.setdefault("experiment", {})["name"] = args.run_name
    task_name = cfg["task"]["name"]
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("purpose", task_name)
    device = resolve_device(cfg["inference"].get("device", "auto"))

    dataloader = _build_dataloader(cfg, task_name, args.split)
    output_dir = Path(cfg["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = cfg["inference"].get("checkpoint")
    run_name = cfg["experiment"].get("name")
    if not run_name:
        if checkpoint_name:
            run_name = Path(checkpoint_name).stem
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{task_name}_infer_{timestamp}"
        cfg["experiment"]["name"] = run_name

    resolved_config_path = output_dir / f"{run_name}_resolved_config.yaml"
    summary_path = output_dir / f"{run_name}_inference_summary.json"
    manifest_path = output_dir / f"{run_name}_inference_manifest.json"
    write_yaml(resolved_config_path, cfg)
    manifest = build_run_manifest(
        run_type="inference",
        config_path=Path(args.config),
        resolved_config_path=resolved_config_path,
        cfg=cfg,
        run_name=run_name,
        run_dir=output_dir,
        command=[sys.executable, "mri/cli/infer.py", *(argv if argv is not None else sys.argv[1:])],
        tracker_info={"enabled": False},
    )
    write_json(manifest_path, manifest)

    try:
        if task_name == "segmentation":
            model = create_segmentation_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
            if cfg["inference"].get("checkpoint"):
                _load_checkpoint(model, cfg["inference"]["checkpoint"], device)
            summary = run_segmentation_inference(
                model=model,
                dataloader=dataloader,
                metadata_path=cfg["data"]["metadata"],
                output_dir=cfg["inference"]["output_dir"],
                device=device,
                threshold=cfg.get("metrics", {}).get("segmentation_threshold", 0.5),
            )
            write_json(summary_path, summary)
            write_json(
                manifest_path,
                finalize_run_manifest(
                    manifest,
                    status="completed",
                    summary=summary,
                    artifacts={"summary_json": summary_path},
                ),
            )
            return 0

        if task_name == "classification":
            model = create_classification_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
            if cfg["inference"].get("checkpoint"):
                _load_checkpoint(model, cfg["inference"]["checkpoint"], device)
            output_csv = Path(cfg["inference"]["output_dir"]) / "predictions.csv"
            inference_result = run_classification_inference(
                model=model,
                dataloader=dataloader,
                device=device,
                output_csv=output_csv,
            )
            write_json(summary_path, inference_result["summary"])
            write_json(
                manifest_path,
                finalize_run_manifest(
                    manifest,
                    status="completed",
                    summary=inference_result["summary"],
                    artifacts={
                        "summary_json": summary_path,
                        "predictions_csv": output_csv,
                    },
                ),
            )
            return 0
    except Exception as exc:
        write_json(
            manifest_path,
            finalize_run_manifest(
                manifest,
                status="failed",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            ),
        )
        raise

    raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    raise SystemExit(main())
