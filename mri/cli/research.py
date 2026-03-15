"""Local research workflow runner for segmentation-to-classification experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mri.cli.infer import main as infer_main
from mri.cli.train import main as train_main
from mri.data.index_builders import load_split_file
from mri.experiments.runtime import utc_now_iso, write_json, write_yaml
from tools.dataset.import_tcia_aligned import DEFAULT_SOURCE, sync_aligned_dataset, validate_aligned_dataset
from tools.generate_splits import build_splits, summarize_splits, write_split_artifacts


def _relative_path(from_dir: Path, target: Path) -> str:
    return str(Path(os.path.relpath(target, from_dir)))


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the local research workflow end-to-end")
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE), help="Source aligned_v2 directory")
    parser.add_argument("--dest-data", default="data/aligned_v2", help="Repository-local aligned_v2 directory")
    parser.add_argument("--import-mode", choices=["copy", "link"], default="copy", help="How to materialize source data locally")
    parser.add_argument("--force-import", action="store_true", help="Replace a mismatched destination dataset")
    parser.add_argument("--validate-import-files", action="store_true", help="Validate sample-level files during import")
    parser.add_argument("--skip-import", action="store_true", help="Reuse the existing local aligned dataset without syncing")
    parser.add_argument("--seg-config", default="mri/config/task/segmentation.yaml", help="Base segmentation task config")
    parser.add_argument("--cls-config", default="mri/config/task/classification.yaml", help="Base classification task config")
    parser.add_argument("--split-file", help="Split YAML path. Defaults to data/splits/<today>.yaml")
    parser.add_argument("--split-ratios", default="0.7,0.15,0.15", help="Comma-separated train,val,test ratios")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for split generation")
    parser.add_argument("--label-space", choices=["original", "downstream_5class"], default="downstream_5class", help="Label space used for split stratification")
    parser.add_argument("--no-stratify", action="store_true", help="Disable split stratification")
    parser.add_argument("--regenerate-split", action="store_true", help="Overwrite an existing split file")
    parser.add_argument("--seg-inference-splits", default="train,val,test", help="Comma-separated segmentation inference splits")
    parser.add_argument("--cls-inference-split", default="test", help="Classification inference split")
    parser.add_argument("--run-name", help="Research run name. Defaults to research-<timestamp>")
    parser.add_argument("--output-root", default="experiments/research", help="Root directory for research run outputs")
    parser.add_argument("--device", help="Device override passed to train/infer CLIs")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB in generated task configs")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and manifests without launching training or inference")
    args = parser.parse_args(argv)

    ratios = [float(part) for part in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be three numbers summing to 1.0")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = args.run_name or f"research-{timestamp}"
    run_root = Path(args.output_root) / run_name
    configs_dir = run_root / "configs"
    manifests_dir = run_root / "manifests"
    seg_ckpt_root = run_root / "checkpoints" / "segmentation"
    cls_ckpt_root = run_root / "checkpoints" / "classification"
    seg_pred_root = run_root / "predictions" / "segmentation"
    cls_pred_root = run_root / "predictions" / "classification"
    for path in (configs_dir, manifests_dir, seg_ckpt_root, cls_ckpt_root, seg_pred_root, cls_pred_root):
        path.mkdir(parents=True, exist_ok=True)

    source_root = Path(args.source_data)
    dest_root = Path(args.dest_data)
    split_file = Path(args.split_file) if args.split_file else Path("data/splits") / f"{datetime.now().date().isoformat()}.yaml"
    split_summary_path = split_file.with_name(f"{split_file.stem}_summary.json")
    seg_base_config = Path(args.seg_config).resolve()
    cls_base_config = Path(args.cls_config).resolve()
    seg_config_path = configs_dir / "segmentation.yaml"
    cls_config_path = configs_dir / "classification.yaml"
    seg_run_name = f"{run_name}-seg"
    cls_run_name = f"{run_name}-cls"

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "status": "dry_run" if args.dry_run else "running",
        "name": run_name,
        "run_root": str(run_root.resolve()),
        "source_data": str(source_root),
        "dest_data": str(dest_root),
        "split_file": str(split_file),
        "label_space": args.label_space,
        "segmentation_base_config": str(seg_base_config),
        "classification_base_config": str(cls_base_config),
        "stages": [],
    }
    manifest_path = manifests_dir / "research_manifest.json"

    def _record_stage(record: Dict[str, Any]) -> None:
        manifest["stages"].append(record)
        manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, manifest)

    write_json(manifest_path, manifest)

    if args.skip_import:
        dataset_summary = validate_aligned_dataset(dest_root, validate_files=args.validate_import_files)
        _record_stage(
            _stage_record(
                "import",
                "reused",
                summary=dataset_summary,
            )
        )
        data_root_for_execution = dest_root
        metadata_for_split = dest_root / "metadata.json"
    else:
        import_result = sync_aligned_dataset(
            source_root,
            dest_root,
            mode=args.import_mode,
            force=args.force_import,
            validate_files=args.validate_import_files,
            dry_run=args.dry_run,
        )
        _record_stage(
            _stage_record(
                "import",
                "dry_run" if args.dry_run else import_result["action"],
                result=import_result,
            )
        )
        data_root_for_execution = dest_root
        metadata_for_split = (dest_root if dest_root.exists() else source_root) / "metadata.json"

    if split_file.exists() and not args.regenerate_split:
        splits = load_split_file(split_file)
        split_summary = summarize_splits(metadata_for_split, splits, label_space=args.label_space)
        if not args.dry_run:
            write_split_artifacts(
                splits=splits,
                summary=split_summary,
                output_path=split_file,
                summary_path=split_summary_path,
            )
        split_status = "reused"
    else:
        splits = build_splits(
            metadata_for_split,
            ratios,
            args.seed,
            stratify=not args.no_stratify,
            label_space=args.label_space,
        )
        split_summary = summarize_splits(metadata_for_split, splits, label_space=args.label_space)
        if not args.dry_run:
            write_split_artifacts(
                splits=splits,
                summary=split_summary,
                output_path=split_file,
                summary_path=split_summary_path,
            )
        split_status = "generated"

    _record_stage(
        _stage_record(
            "split",
            "dry_run" if args.dry_run else split_status,
            output_path=str(split_file),
            summary_path=str(split_summary_path),
            summary=split_summary,
        )
    )

    common_overrides = {
        "data.metadata": str((data_root_for_execution / "metadata.json").resolve() if (data_root_for_execution / "metadata.json").exists() else data_root_for_execution / "metadata.json"),
        "data.split_file": str(split_file.resolve()),
        "experiment.tags": ["research"],
    }
    if args.disable_wandb:
        common_overrides["tracking.wandb.enabled"] = False

    _build_override_config(
        base_config_path=seg_base_config,
        generated_config_path=seg_config_path,
        overrides={
            **common_overrides,
            "train.output_dir": str(seg_ckpt_root.resolve()),
            "inference.output_dir": str(seg_pred_root.resolve()),
        },
    )
    _build_override_config(
        base_config_path=cls_base_config,
        generated_config_path=cls_config_path,
        overrides={
            **common_overrides,
            "data.seg_pred_dir": str(seg_pred_root.resolve()),
            "train.output_dir": str(cls_ckpt_root.resolve()),
            "inference.output_dir": str(cls_pred_root.resolve()),
        },
    )

    _record_stage(
        _stage_record(
            "config_generation",
            "completed",
            segmentation_config=str(seg_config_path),
            classification_config=str(cls_config_path),
        )
    )

    if args.dry_run:
        manifest["status"] = "dry_run"
        manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, manifest)
        print(f"Research dry-run prepared: {run_root}")
        print(f"Manifest: {manifest_path}")
        print(f"Seg config: {seg_config_path}")
        print(f"Cls config: {cls_config_path}")
        print(f"Planned split file: {split_file}")
        return 0

    train_common_args: List[str] = []
    infer_common_args: List[str] = []
    if args.device:
        train_common_args.extend(["--device", args.device])
        infer_common_args.extend(["--device", args.device])

    seg_train_args = [
        "--config",
        str(seg_config_path),
        "--run_name",
        seg_run_name,
        "--output_dir",
        str(seg_ckpt_root.resolve()),
        *train_common_args,
    ]
    _run_cli(train_main, seg_train_args, "segmentation training")
    seg_run_dir = seg_ckpt_root / seg_run_name
    seg_checkpoint = seg_run_dir / f"{seg_run_name}_best.pt"
    if not seg_checkpoint.exists():
        raise FileNotFoundError(f"Segmentation best checkpoint not found: {seg_checkpoint}")
    _record_stage(
        _stage_record(
            "segmentation_train",
            "completed",
            command=seg_train_args,
            run_dir=str(seg_run_dir),
            best_checkpoint=str(seg_checkpoint),
        )
    )

    for split_name in _csv_values(args.seg_inference_splits):
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
        _run_cli(infer_main, seg_infer_args, f"segmentation inference ({split_name})")
        _record_stage(
            _stage_record(
                f"segmentation_inference_{split_name}",
                "completed",
                command=seg_infer_args,
                summary_path=str(seg_pred_root / f"{seg_infer_run_name}_inference_summary.json"),
            )
        )

    cls_train_args = [
        "--config",
        str(cls_config_path),
        "--run_name",
        cls_run_name,
        "--output_dir",
        str(cls_ckpt_root.resolve()),
        *train_common_args,
    ]
    _run_cli(train_main, cls_train_args, "classification training")
    cls_run_dir = cls_ckpt_root / cls_run_name
    cls_checkpoint = cls_run_dir / f"{cls_run_name}_best.pt"
    if not cls_checkpoint.exists():
        raise FileNotFoundError(f"Classification best checkpoint not found: {cls_checkpoint}")
    _record_stage(
        _stage_record(
            "classification_train",
            "completed",
            command=cls_train_args,
            run_dir=str(cls_run_dir),
            best_checkpoint=str(cls_checkpoint),
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
    _run_cli(infer_main, cls_infer_args, f"classification inference ({args.cls_inference_split})")
    _record_stage(
        _stage_record(
            "classification_inference",
            "completed",
            command=cls_infer_args,
            summary_path=str(cls_pred_root / f"{cls_infer_run_name}_inference_summary.json"),
            predictions_csv=str(cls_pred_root / "predictions.csv"),
        )
    )

    manifest["status"] = "completed"
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)
    print(f"Research workflow completed: {run_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Seg checkpoint: {seg_checkpoint}")
    print(f"Cls checkpoint: {cls_checkpoint}")
    print(f"Seg predictions: {seg_pred_root}")
    print(f"Cls predictions: {cls_pred_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
