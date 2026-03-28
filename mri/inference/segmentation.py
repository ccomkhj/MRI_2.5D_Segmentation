"""Segmentation inference and prediction export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import numpy as np
from PIL import Image

from mri.data.metadata import load_metadata

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


_PROSTATE_OVERLAY_RGB = (255, 255, 0)
_TARGET_OVERLAY_RGB = (255, 0, 0)
_PROSTATE_ALPHA = 0.25
_TARGET_ALPHA = 0.45


def create_segmentation_overlay(
    base_image: np.ndarray,
    prostate_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Render a segmentation overlay PNG on top of a grayscale base image."""

    if base_image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {base_image.shape}")

    if base_image.dtype != np.uint8:
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)

    overlay = np.stack([base_image, base_image, base_image], axis=-1).astype(np.float32)

    def _apply_mask(mask: np.ndarray, rgb: tuple[int, int, int], alpha: float) -> None:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return
        color = np.array(rgb, dtype=np.float32)
        overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color

    _apply_mask(prostate_mask, _PROSTATE_OVERLAY_RGB, _PROSTATE_ALPHA)
    _apply_mask(target_mask, _TARGET_OVERLAY_RGB, _TARGET_ALPHA)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _load_case_t2_slice(metadata_root: Path, case_id: str, slice_idx: int, image_shape: tuple[int, int]) -> np.ndarray:
    image_path = metadata_root / case_id / "t2" / f"{slice_idx:04d}.png"
    if not image_path.exists():
        return np.zeros(image_shape, dtype=np.uint8)

    image = Image.open(image_path).convert("L")
    if image.size != (image_shape[1], image_shape[0]):
        image = image.resize((image_shape[1], image_shape[0]), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)


def run_segmentation_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metadata_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float | int | str | None]:
    import torch
    from mri.tasks.segmentation_ops import compute_dice_score

    model = model.to(device)
    model.eval()

    meta = load_metadata(metadata_path)
    metadata_root = meta.path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_buffers: Dict[str, Dict[str, np.ndarray]] = {}
    dice_scores = []
    num_samples = 0
    overlay_pngs_written = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            masks = batch[1]
            metas = batch[2]
            if isinstance(metas, dict):
                meta_list = [
                    {k: metas[k][i] for k in metas}
                    for i in range(len(metas[next(iter(metas))]))
                ]
            else:
                meta_list = metas

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            dice_scores.append(compute_dice_score(logits.cpu(), masks.cpu(), threshold=threshold))
            num_samples += images.shape[0]

            for i, m in enumerate(meta_list):
                case_id = m["case_id"]
                slice_idx = int(m["slice_idx"])
                if case_id not in case_buffers:
                    num_slices = int(meta.cases[case_id]["num_slices"])
                    h, w = probs.shape[2], probs.shape[3]
                    case_buffers[case_id] = {
                        "target": np.zeros((num_slices, h, w), dtype=np.float32),
                        "prostate": np.zeros((num_slices, h, w), dtype=np.float32),
                        "predicted_slices": set(),
                    }
                case_buffers[case_id]["prostate"][slice_idx] = probs[i, 0]
                if probs.shape[1] > 1:
                    case_buffers[case_id]["target"][slice_idx] = probs[i, 1]
                case_buffers[case_id]["predicted_slices"].add(slice_idx)

    for case_id, arrays in case_buffers.items():
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "prostate_prob.npy", arrays["prostate"])
        np.save(case_dir / "target_prob.npy", arrays["target"])

        overlay_dir = case_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        image_shape = arrays["prostate"].shape[1:]
        for slice_idx in sorted(arrays["predicted_slices"]):
            base_image = _load_case_t2_slice(metadata_root, case_id, slice_idx, image_shape)
            overlay = create_segmentation_overlay(
                base_image=base_image,
                prostate_mask=arrays["prostate"][slice_idx] >= threshold,
                target_mask=arrays["target"][slice_idx] >= threshold,
            )
            Image.fromarray(overlay).save(overlay_dir / f"{slice_idx:04d}.png")
            overlay_pngs_written += 1

    return {
        "output_dir": str(output_dir),
        "cases_written": len(case_buffers),
        "num_samples": num_samples,
        "overlay_pngs_written": overlay_pngs_written,
        "segmentation_threshold": threshold,
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else None,
        "primary_metric_name": "mean_dice",
        "best_metric": float(np.mean(dice_scores)) if dice_scores else None,
    }
