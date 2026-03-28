"""Classification dataset: per-case volumes derived from aligned_v2 + seg predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from mri.data.metadata import load_metadata


def _safe_std(value: float) -> float:
    if value is None or value <= 1e-6:
        return 1.0
    return float(value)


def _case_prediction_dir(seg_pred_dir: Path, case_id: str) -> Path:
    return seg_pred_dir / case_id


def _case_prediction_npz(seg_pred_dir: Path, case_id: str) -> Path:
    return seg_pred_dir / f"{case_id.replace('/', '_')}.npz"


def has_segmentation_predictions(seg_pred_dir: Union[str, Path], case_id: str) -> bool:
    seg_pred_dir = Path(seg_pred_dir)
    case_dir = _case_prediction_dir(seg_pred_dir, case_id)
    if case_dir.is_dir():
        if (case_dir / "target_prob.npy").exists() and (case_dir / "prostate_prob.npy").exists():
            return True
    return _case_prediction_npz(seg_pred_dir, case_id).exists()


def find_missing_segmentation_predictions(
    seg_pred_dir: Union[str, Path],
    case_ids: List[str],
) -> List[str]:
    unique_case_ids = sorted(dict.fromkeys(case_ids))
    return [case_id for case_id in unique_case_ids if not has_segmentation_predictions(seg_pred_dir, case_id)]


class ClassificationDataset(Dataset):
    def __init__(
        self,
        metadata_path: Union[str, Path],
        cases_index: List[Dict],
        seg_pred_dir: Union[str, Path],
        depth: int = 16,
        crop_size: int = 192,
        output_size: int = 256,
        modalities: Tuple[str, ...] = ("t2", "adc", "calc"),
        zero_pad_missing: bool = True,
        selection_source: str = "pred",
        selection_jitter: int = 2,
        min_prob: float = 0.3,
        use_roi: bool = True,
        normalize: bool = True,
    ) -> None:
        self.meta = load_metadata(metadata_path)
        self.base_dir = self.meta.path.parent
        self.cases = cases_index
        self.seg_pred_dir = Path(seg_pred_dir)
        self.depth = depth
        self.crop_size = crop_size
        self.output_size = output_size
        self.modalities = modalities
        self.zero_pad_missing = zero_pad_missing
        self.selection_source = selection_source
        self.selection_jitter = selection_jitter
        self.min_prob = min_prob
        self.use_roi = use_roi
        self.normalize = normalize
        self.global_stats = self.meta.raw.get("global_stats", {})

    def __len__(self) -> int:
        return len(self.cases)

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((256, 256), dtype=np.float32)
        img = Image.open(path).convert("L")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((256, 256), dtype=np.float32)
        img = Image.open(path).convert("L")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.NEAREST)
        return (np.array(img, dtype=np.float32) > 127).astype(np.float32)

    def _load_seg_preds(self, case_id: str) -> Dict[str, np.ndarray]:
        case_dir = _case_prediction_dir(self.seg_pred_dir, case_id)
        if case_dir.is_dir():
            target_path = case_dir / "target_prob.npy"
            prostate_path = case_dir / "prostate_prob.npy"
            if target_path.exists() and prostate_path.exists():
                return {
                    "target": np.load(target_path),
                    "prostate": np.load(prostate_path),
                }
        npz_path = _case_prediction_npz(self.seg_pred_dir, case_id)
        if npz_path.exists():
            data = np.load(npz_path)
            return {
                "target": data["target_prob"],
                "prostate": data["prostate_prob"],
            }
        raise FileNotFoundError(f"Seg predictions not found for {case_id} in {self.seg_pred_dir}")

    def _normalize_modalities(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.global_stats:
            return volume / 255.0
        # volume shape: (C, D, H, W)
        stats = {
            "t2": self.global_stats.get("t2", {}),
            "adc": self.global_stats.get("adc", {}),
            "calc": self.global_stats.get("calc", {}),
        }
        for idx, modality in enumerate(self.modalities):
            mean = stats.get(modality, {}).get("mean", 0.0)
            std = _safe_std(stats.get(modality, {}).get("std", 1.0))
            if modality in {"adc", "calc"}:
                present_slices = torch.amax(volume[idx], dim=(-1, -2)) > 0
                if not bool(torch.any(present_slices)):
                    continue
                volume[idx, present_slices] = (volume[idx, present_slices] - mean) / std
                continue
            volume[idx] = (volume[idx] - mean) / std
        return volume

    def _center_crop_pad_indices(self, center: int, num_slices: int) -> List[int]:
        start = center - (self.depth // 2)
        indices = []
        for i in range(start, start + self.depth):
            i_clamped = max(0, min(num_slices - 1, i))
            indices.append(i_clamped)
        return indices

    def _select_center_index(
        self, scores: np.ndarray, num_slices: int, apply_jitter: bool
    ) -> int:
        if scores.size == 0:
            center = num_slices // 2
        else:
            center = int(scores.argmax())
        if apply_jitter and self.selection_jitter > 0:
            shift = random.randint(-self.selection_jitter, self.selection_jitter)
            center = max(0, min(num_slices - 1, center + shift))
        return center

    def _compute_scores_from_mask(self, mask_volume: np.ndarray) -> np.ndarray:
        # mask_volume shape: (num_slices, H, W)
        if mask_volume.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return mask_volume.reshape(mask_volume.shape[0], -1).sum(axis=1)

    def _compute_scores_from_prob(self, prob_volume: np.ndarray) -> np.ndarray:
        if prob_volume.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return prob_volume.reshape(prob_volume.shape[0], -1).max(axis=1)

    def _center_from_case_info(self, case_info: Dict, has_target_flag: bool) -> int:
        key = "slices_with_target" if has_target_flag else "slices_with_prostate"
        slices = case_info.get(key) or []
        if isinstance(slices, int):
            slices = list(range(slices))
        if not slices:
            return int(case_info.get("num_slices", 0) // 2)
        return int(slices[len(slices) // 2])

    def _load_roi_masks(self, case_dir: Path, slice_indices: List[int], use_target: bool) -> np.ndarray:
        masks = []
        for slice_idx in slice_indices:
            prostate_path = case_dir / "mask_prostate" / f"{slice_idx:04d}.png"
            mask_prostate = self._load_mask(prostate_path)
            if use_target:
                target1_path = case_dir / "mask_target1" / f"{slice_idx:04d}.png"
                target2_path = case_dir / "mask_target2" / f"{slice_idx:04d}.png"
                mask_target = np.maximum(self._load_mask(target1_path), self._load_mask(target2_path))
                masks.append(mask_target)
            else:
                masks.append(mask_prostate)
        return np.stack(masks, axis=0)

    def _get_roi_bbox(self, mask_volume: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # mask_volume shape: (D, H, W)
        if mask_volume.size == 0:
            return None
        union_mask = mask_volume.max(axis=0)
        ys, xs = np.where(union_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        return y_min, y_max, x_min, x_max

    def _crop_volume(self, volume: torch.Tensor, bbox: Optional[Tuple[int, int, int, int]]) -> torch.Tensor:
        # volume: (C, D, H, W)
        _, _, h, w = volume.shape
        if bbox is None:
            center_y, center_x = h // 2, w // 2
        else:
            y_min, y_max, x_min, x_max = bbox
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2

        half = self.crop_size // 2
        y_start = center_y - half
        x_start = center_x - half
        y_end = y_start + self.crop_size
        x_end = x_start + self.crop_size

        out = torch.zeros((volume.shape[0], volume.shape[1], self.crop_size, self.crop_size), dtype=volume.dtype)

        src_y0 = max(0, y_start)
        src_x0 = max(0, x_start)
        src_y1 = min(h, y_end)
        src_x1 = min(w, x_end)

        dst_y0 = src_y0 - y_start
        dst_x0 = src_x0 - x_start
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = volume[:, :, src_y0:src_y1, src_x0:src_x1]
        return out

    def _resize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        # volume: (C, D, H, W) -> resize H/W to output_size
        if volume.shape[-1] == self.output_size and volume.shape[-2] == self.output_size:
            return volume
        vol = volume.unsqueeze(0)  # (1, C, D, H, W)
        vol = F.interpolate(vol, size=(volume.shape[1], self.output_size, self.output_size), mode="trilinear", align_corners=False)
        return vol.squeeze(0)

    def __getitem__(self, idx: int):
        record = self.cases[idx]
        case_id = record["case_id"]
        label = int(record["label"])
        case_dir = self.base_dir / case_id
        case_info = self.meta.cases[case_id]
        num_slices = int(case_info.get("num_slices", 0))

        use_pred = self.selection_source in {"pred", "hybrid"}
        use_gt = self.selection_source == "gt"
        apply_jitter = self.selection_source == "hybrid"

        target_scores = None
        prostate_scores = None
        roi_mask_volume = None
        has_target_flag = record["has_target"]

        if use_pred:
            preds = self._load_seg_preds(case_id)
            target_prob = preds["target"]
            prostate_prob = preds["prostate"]
            target_scores = self._compute_scores_from_prob(target_prob)
            prostate_scores = self._compute_scores_from_prob(prostate_prob)
            has_target_flag = bool(target_scores.max() > self.min_prob)
            # ROI mask from probabilities
            roi_mask_volume = (target_prob if has_target_flag else prostate_prob) > self.min_prob
        if use_gt:
            center_idx = self._center_from_case_info(case_info, record["has_target"])
            if apply_jitter and self.selection_jitter > 0:
                shift = random.randint(-self.selection_jitter, self.selection_jitter)
                center_idx = max(0, min(num_slices - 1, center_idx + shift))
        else:
            scores = target_scores if has_target_flag else prostate_scores
            center_idx = self._select_center_index(scores, num_slices, apply_jitter)
        slice_indices = self._center_crop_pad_indices(center_idx, num_slices)
        if use_gt:
            roi_mask_volume = self._load_roi_masks(case_dir, slice_indices, record["has_target"])

        # Load modality volumes
        modality_volumes = []
        for modality in self.modalities:
            if modality == "adc" and not case_info.get("has_adc", False) and self.zero_pad_missing:
                modality_volumes.append(np.zeros((self.depth, 256, 256), dtype=np.float32))
                continue
            if modality == "calc" and not case_info.get("has_calc", False) and self.zero_pad_missing:
                modality_volumes.append(np.zeros((self.depth, 256, 256), dtype=np.float32))
                continue

            slices = []
            for slice_idx in slice_indices:
                img_path = case_dir / modality / f"{slice_idx:04d}.png"
                slices.append(self._load_image(img_path))
            modality_volumes.append(np.stack(slices, axis=0))

        volume = np.stack(modality_volumes, axis=0)  # (C, D, H, W)
        volume_t = torch.from_numpy(volume).float()

        if self.use_roi and roi_mask_volume is not None:
            if use_gt:
                roi_mask_selected = roi_mask_volume
            else:
                roi_mask_selected = roi_mask_volume[slice_indices]
            bbox = self._get_roi_bbox(roi_mask_selected)
            volume_t = self._crop_volume(volume_t, bbox)

        volume_t = self._resize_volume(volume_t)
        if self.normalize:
            volume_t = self._normalize_modalities(volume_t)

        meta = {
            "case_id": case_id,
            "slice_indices": slice_indices,
            "center_idx": center_idx,
        }

        return volume_t, label, meta
