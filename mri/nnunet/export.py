"""Export segmentation samples into a MONAI nnU-Net V2 workspace."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from mri.config.loader import load_config
from mri.data.index_builders import build_segmentation_index, load_split_file
from mri.data.metadata import load_metadata
from mri.experiments.runtime import utc_now_iso, write_json, write_yaml


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ExportPaths:
    run_root: Path
    dataset_root: Path
    configs_dir: Path
    manifests_dir: Path
    datalist_path: Path
    input_config_path: Path
    manifest_path: Path
    resolved_task_config_path: Path


def _csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _sanitize_name(value: str) -> str:
    return _SANITIZE_RE.sub("_", value).strip("._") or "sample"


def _materialized_sample_name(source_split: str, meta: dict[str, Any]) -> str:
    sample_id = str(meta.get("sample_id") or f"{meta.get('case_id', 'case')}_slice_{meta.get('slice_idx', 0):04d}")
    return _sanitize_name(f"{source_split}__{sample_id}")


def _combine_mask_channels(mask_stack: np.ndarray) -> np.ndarray:
    if mask_stack.ndim != 3 or mask_stack.shape[0] < 2:
        raise ValueError(f"Expected mask stack shaped like (2, H, W), got {mask_stack.shape}")

    prostate = mask_stack[0] > 0.5
    target = mask_stack[1] > 0.5

    label = np.zeros(mask_stack.shape[1:], dtype=np.uint8)
    label[prostate] = 1
    label[target] = 2
    return label


def _safe_std(value: float | None) -> float:
    if value is None or value <= 1e-6:
        return 1.0
    return float(value)


def _load_png(path: Path, *, resample: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((256, 256), dtype=np.float32)
    img = Image.open(path).convert("L")
    if img.size != (256, 256):
        img = img.resize((256, 256), resample)
    return np.array(img, dtype=np.float32)


def _load_mask(path: Path) -> np.ndarray:
    return (_load_png(path, resample=Image.NEAREST) > 127).astype(np.float32)


def _filter_segmentation_samples(
    samples_index: list[dict[str, Any]], *, require_complete: bool, require_positive: bool
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for sample in samples_index:
        if require_complete and not (sample.get("has_adc", False) and sample.get("has_calc", False)):
            continue
        if require_positive and not sample.get("has_prostate", False):
            continue
        filtered.append(sample)
    return filtered


def _build_sample_arrays(
    *,
    metadata_root: Path,
    sample: dict[str, Any],
    stack_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    case_dir = metadata_root / sample["case_id"]

    context_indices = list(sample["t2_context_indices"])
    if stack_depth < len(context_indices):
        start = (len(context_indices) - stack_depth) // 2
        context_indices = context_indices[start : start + stack_depth]
    elif stack_depth > len(context_indices):
        diff = stack_depth - len(context_indices)
        context_indices = [context_indices[0]] * (diff // 2) + context_indices + [context_indices[-1]] * (diff - diff // 2)

    t2_slices = [_load_png(case_dir / "t2" / f"{slice_idx:04d}.png", resample=Image.BILINEAR) for slice_idx in context_indices]

    if sample.get("has_adc", False):
        adc_file = sample["files"].get("adc", f"{sample['slice_idx']:04d}.png")
        adc_img = _load_png(case_dir / "adc" / adc_file, resample=Image.BILINEAR)
    else:
        adc_img = np.zeros((256, 256), dtype=np.float32)

    if sample.get("has_calc", False):
        calc_file = sample["files"].get("calc", f"{sample['slice_idx']:04d}.png")
        calc_img = _load_png(case_dir / "calc" / calc_file, resample=Image.BILINEAR)
    else:
        calc_img = np.zeros((256, 256), dtype=np.float32)

    image_stack = np.stack(t2_slices + [adc_img, calc_img], axis=0)

    prostate_file = sample["files"].get("mask_prostate", f"{sample['slice_idx']:04d}.png")
    mask_prostate = _load_mask(case_dir / "mask_prostate" / prostate_file)

    target1_file = sample["files"].get("mask_target1", f"{sample['slice_idx']:04d}.png")
    mask_target1 = _load_mask(case_dir / "mask_target1" / target1_file)
    mask_target2 = _load_mask(case_dir / "mask_target2" / target1_file)
    mask_target = np.maximum(mask_target1, mask_target2)
    mask_stack = np.stack([mask_prostate, mask_target], axis=0)
    return image_stack, mask_stack


def _normalize_image_stack(image_stack: np.ndarray, *, stack_depth: int, global_stats: dict[str, Any]) -> np.ndarray:
    if not global_stats:
        return image_stack / 255.0

    normalized = image_stack.copy()
    t2_mean = global_stats["t2"]["mean"]
    t2_std = _safe_std(global_stats["t2"]["std"])
    normalized[:stack_depth] = (normalized[:stack_depth] - t2_mean) / t2_std

    adc_channel = normalized[stack_depth]
    if adc_channel.max() > 0:
        adc_mean = global_stats["adc"]["mean"]
        adc_std = _safe_std(global_stats["adc"]["std"])
        normalized[stack_depth] = (adc_channel - adc_mean) / adc_std

    calc_channel = normalized[stack_depth + 1]
    if calc_channel.max() > 0:
        calc_mean = global_stats["calc"]["mean"]
        calc_std = _safe_std(global_stats["calc"]["std"])
        normalized[stack_depth + 1] = (calc_channel - calc_mean) / calc_std

    return normalized


def _default_modality_names(cfg: dict[str, Any]) -> list[str]:
    data_cfg = cfg.get("data", {})
    modalities = [str(item).upper() for item in data_cfg.get("modalities", ["t2", "adc", "calc"])]
    stack_depth = int(data_cfg.get("stack_depth", 5))
    primary = modalities[:1] or ["T2"]
    return [primary[0]] * stack_depth + modalities[1:]


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "nibabel is required for nnU-Net dataset export. Install nibabel together with MONAI nnU-Net dependencies."
        ) from exc
    return nib


def _write_image_nifti(path: Path, image_stack: np.ndarray) -> None:
    if image_stack.ndim != 3:
        raise ValueError(f"Expected image stack shaped like (C, H, W), got {image_stack.shape}")

    nib = _require_nibabel()
    # Store 2D multi-channel data as (H, W, 1, C) so MONAI can recover the channel axis.
    data = np.moveaxis(image_stack.astype(np.float32), 0, -1)[:, :, None, :]
    nib.save(nib.Nifti1Image(data, np.eye(4, dtype=np.float32)), str(path))


def _write_label_nifti(path: Path, label_map: np.ndarray) -> None:
    if label_map.ndim != 2:
        raise ValueError(f"Expected label map shaped like (H, W), got {label_map.shape}")

    nib = _require_nibabel()
    data = label_map.astype(np.uint8)[:, :, None]
    nib.save(nib.Nifti1Image(data, np.eye(4, dtype=np.float32)), str(path))


def _export_split_records(
    *,
    cfg: dict[str, Any],
    metadata_root: Path,
    global_stats: dict[str, Any],
    split_name: str,
    samples_index: list[dict[str, Any]],
    dataset_root: Path,
    dry_run: bool,
    fold_value: int | None,
) -> tuple[list[dict[str, Any]], int]:
    data_cfg = cfg.get("data", {})
    stack_depth = int(data_cfg.get("stack_depth", 5))
    filtered_samples = _filter_segmentation_samples(
        samples_index,
        require_complete=bool(data_cfg.get("require_complete", False)),
        require_positive=bool(data_cfg.get("require_positive", False)),
    )

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for sample in filtered_samples:
        image_stack, mask_stack = _build_sample_arrays(
            metadata_root=metadata_root,
            sample=sample,
            stack_depth=stack_depth,
        )
        image_stack = _normalize_image_stack(image_stack, stack_depth=stack_depth, global_stats=global_stats)
        label_map = _combine_mask_channels(mask_stack)
        meta = {
            "case_id": sample.get("case_id"),
            "sample_id": sample.get("sample_id"),
            "slice_idx": sample.get("slice_idx"),
        }
        sample_name = _materialized_sample_name(split_name, meta)
        image_rel = Path("images") / f"{sample_name}.nii.gz"
        label_rel = Path("labels") / f"{sample_name}.nii.gz"

        if not dry_run:
            _write_image_nifti(dataset_root / image_rel, image_stack)
            _write_label_nifti(dataset_root / label_rel, label_map)

        record: dict[str, Any] = {
            "image": str(image_rel),
            "label": str(label_rel),
            "case_id": meta.get("case_id"),
            "sample_id": meta.get("sample_id"),
            "slice_idx": meta.get("slice_idx"),
            "source_split": split_name,
        }
        if fold_value is not None:
            record["fold"] = fold_value
        records.append(record)

    return records, len(filtered_samples)


def _resolved_paths(output_root: Path, run_name: str) -> ExportPaths:
    run_root = output_root / run_name
    configs_dir = run_root / "configs"
    manifests_dir = run_root / "manifests"
    dataset_root = run_root / "dataset"
    return ExportPaths(
        run_root=run_root,
        dataset_root=dataset_root,
        configs_dir=configs_dir,
        manifests_dir=manifests_dir,
        datalist_path=configs_dir / "datalist.json",
        input_config_path=configs_dir / "input.yaml",
        manifest_path=manifests_dir / "nnunet_manifest.json",
        resolved_task_config_path=configs_dir / "resolved_task_config.yaml",
    )


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


def export_nnunet_workspace(
    *,
    config_path: str | Path,
    output_root: str | Path = "experiments/nnunet",
    run_name: str,
    training_splits: Sequence[str] = ("train", "val"),
    testing_splits: Sequence[str] = ("test",),
    dataset_name_or_id: int = 901,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if cfg.get("task", {}).get("name") != "segmentation":
        raise ValueError("nnU-Net export currently supports segmentation task configs only.")

    paths = _resolved_paths(Path(output_root), run_name)
    if paths.run_root.exists():
        if force:
            shutil.rmtree(paths.run_root)
        elif paths.manifest_path.exists() and paths.input_config_path.exists():
            manifest = _load_existing_manifest(paths.manifest_path)
            if not dry_run and not manifest.get("materialized_dataset", False):
                raise RuntimeError(
                    f"Existing nnU-Net workspace '{paths.run_root}' was created as dry-run only. "
                    "Re-run with --force to materialize the dataset."
                )
            return manifest
        else:
            raise FileExistsError(f"nnU-Net workspace already exists: {paths.run_root}. Use --force to overwrite it.")

    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        paths.dataset_root.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(cfg["data"]["metadata"])
    split_map = load_split_file(cfg["data"]["split_file"])
    metadata_root = meta.path.parent
    all_requested_splits = list(training_splits) + list(testing_splits)
    unknown = [name for name in all_requested_splits if name not in split_map]
    if unknown:
        raise KeyError(f"Unknown split(s) in split file: {', '.join(unknown)}")

    training_records: list[dict[str, Any]] = []
    testing_records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    for split_name in training_splits:
        split_index = build_segmentation_index(meta, split_map[split_name])
        fold_value = 0 if split_name == "val" else 1
        records, count = _export_split_records(
            cfg=cfg,
            metadata_root=metadata_root,
            global_stats=meta.raw.get("global_stats", {}),
            split_name=split_name,
            samples_index=split_index,
            dataset_root=paths.dataset_root,
            dry_run=dry_run,
            fold_value=fold_value,
        )
        training_records.extend(records)
        split_counts[split_name] = count

    for split_name in testing_splits:
        split_index = build_segmentation_index(meta, split_map[split_name])
        records, count = _export_split_records(
            cfg=cfg,
            metadata_root=metadata_root,
            global_stats=meta.raw.get("global_stats", {}),
            split_name=split_name,
            samples_index=split_index,
            dataset_root=paths.dataset_root,
            dry_run=dry_run,
            fold_value=None,
        )
        testing_records.extend(records)
        split_counts[split_name] = count

    if not training_records:
        raise ValueError("No segmentation samples were exported for nnU-Net training.")

    datalist_payload: dict[str, Any] = {"training": training_records}
    if testing_records:
        datalist_payload["testing"] = testing_records

    input_payload = {
        "datalist": str(paths.datalist_path.resolve()),
        "dataroot": str(paths.dataset_root.resolve()),
        "modality": _default_modality_names(cfg),
        "dataset_name_or_id": int(dataset_name_or_id),
        "nnunet_raw": str((paths.run_root / "nnunet_raw").resolve()),
        "nnunet_preprocessed": str((paths.run_root / "nnunet_preprocessed").resolve()),
        "nnunet_results": str((paths.run_root / "nnunet_results").resolve()),
    }

    write_yaml(paths.resolved_task_config_path, cfg)
    write_json(paths.datalist_path, datalist_payload)
    write_yaml(paths.input_config_path, input_payload)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "status": "dry_run" if dry_run else "prepared",
        "run_name": run_name,
        "task_config": str(Path(config_path).resolve()),
        "materialized_dataset": not dry_run,
        "training_splits": list(training_splits),
        "testing_splits": list(testing_splits),
        "split_counts": split_counts,
        "num_training_samples": len(training_records),
        "num_testing_samples": len(testing_records),
        "stack_depth": cfg.get("data", {}).get("stack_depth"),
        "dataset_name_or_id": int(dataset_name_or_id),
        "modality": input_payload["modality"],
        "artifacts": {
            "run_root": str(paths.run_root.resolve()),
            "dataset_root": str(paths.dataset_root.resolve()),
            "datalist_json": str(paths.datalist_path.resolve()),
            "input_yaml": str(paths.input_config_path.resolve()),
            "resolved_task_config": str(paths.resolved_task_config_path.resolve()),
        },
        "stages": [
            {
                "name": "export",
                "status": "dry_run" if dry_run else "completed",
            }
        ],
    }
    write_json(paths.manifest_path, manifest)
    return manifest


def parse_split_args(training_splits: str, testing_splits: str | None) -> tuple[list[str], list[str]]:
    train_values = _csv_values(training_splits)
    test_values = _csv_values(testing_splits or "")
    return train_values, test_values


def update_manifest_stage(
    manifest_path: Path,
    *,
    stage_name: str,
    status: str,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest = _load_existing_manifest(manifest_path)
    manifest["updated_at"] = utc_now_iso()
    manifest["status"] = status
    manifest.setdefault("stages", []).append(
        {
            "name": stage_name,
            "status": status,
            "command": list(command or []),
        }
    )
    write_json(manifest_path, manifest)
    return manifest
