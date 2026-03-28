"""Segmentation dataset built from aligned_v2 metadata.json."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _safe_std(value: float) -> float:
    if value is None or value <= 1e-6:
        return 1.0
    return float(value)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata_path: Union[str, Path],
        samples_index: Optional[List[Dict]] = None,
        stack_depth: int = 5,
        transform=None,
        require_complete: bool = False,
        require_positive: bool = False,
        normalize: bool = True,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with self.metadata_path.open() as f:
            self.metadata = json.load(f)

        self.base_dir = self.metadata_path.parent
        self.transform = transform
        self.normalize_enabled = normalize
        self.global_stats = self.metadata.get("global_stats", {})
        self.stack_depth = stack_depth

        if samples_index is not None:
            self.samples = samples_index
        else:
            self.samples = self.metadata.get("samples", [])

        filtered = []
        for sample in self.samples:
            if require_complete and not (sample.get("has_adc", False) and sample.get("has_calc", False)):
                continue
            if require_positive and not sample.get("has_prostate", False):
                continue
            filtered.append(sample)
        self.samples = filtered

    def __len__(self) -> int:
        return len(self.samples)

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

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if not self.global_stats:
            return image / 255.0

        t2_mean = self.global_stats["t2"]["mean"]
        t2_std = _safe_std(self.global_stats["t2"]["std"])
        image[: self.stack_depth] = (image[: self.stack_depth] - t2_mean) / t2_std

        adc_channel = image[self.stack_depth]
        if adc_channel.max() > 0:
            adc_mean = self.global_stats["adc"]["mean"]
            adc_std = _safe_std(self.global_stats["adc"]["std"])
            image[self.stack_depth] = (adc_channel - adc_mean) / adc_std

        calc_channel = image[self.stack_depth + 1]
        if calc_channel.max() > 0:
            calc_mean = self.global_stats["calc"]["mean"]
            calc_std = _safe_std(self.global_stats["calc"]["std"])
            image[self.stack_depth + 1] = (calc_channel - calc_mean) / calc_std

        return image

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        case_id = sample["case_id"]
        case_dir = self.base_dir / case_id

        context_indices = list(sample["t2_context_indices"])
        if self.stack_depth < len(context_indices):
            start = (len(context_indices) - self.stack_depth) // 2
            context_indices = context_indices[start : start + self.stack_depth]
        elif self.stack_depth > len(context_indices):
            diff = self.stack_depth - len(context_indices)
            context_indices = (
                [context_indices[0]] * (diff // 2)
                + context_indices
                + [context_indices[-1]] * (diff - diff // 2)
            )

        t2_slices = []
        for slice_idx in context_indices:
            t2_path = case_dir / "t2" / f"{slice_idx:04d}.png"
            t2_slices.append(self._load_image(t2_path))

        if sample.get("has_adc", False):
            adc_file = sample["files"].get("adc", f"{sample['slice_idx']:04d}.png")
            adc_path = case_dir / "adc" / adc_file
            adc_img = self._load_image(adc_path)
        else:
            adc_img = np.zeros((256, 256), dtype=np.float32)

        if sample.get("has_calc", False):
            calc_file = sample["files"].get("calc", f"{sample['slice_idx']:04d}.png")
            calc_path = case_dir / "calc" / calc_file
            calc_img = self._load_image(calc_path)
        else:
            calc_img = np.zeros((256, 256), dtype=np.float32)

        image_stack = np.stack(t2_slices + [adc_img, calc_img], axis=0)

        prostate_file = sample["files"].get("mask_prostate", f"{sample['slice_idx']:04d}.png")
        prostate_path = case_dir / "mask_prostate" / prostate_file
        mask_prostate = self._load_mask(prostate_path)

        target1_file = sample["files"].get("mask_target1", f"{sample['slice_idx']:04d}.png")
        target1_path = case_dir / "mask_target1" / target1_file
        mask_target1 = self._load_mask(target1_path)

        target2_path = case_dir / "mask_target2" / target1_file
        mask_target2 = self._load_mask(target2_path)

        mask_target = np.maximum(mask_target1, mask_target2)
        mask_stack = np.stack([mask_prostate, mask_target], axis=0)

        if self.transform:
            image_stack, mask_stack = self.transform(image_stack, mask_stack)

        if self.normalize_enabled:
            image_stack = self._normalize(image_stack)

        meta = {
            "case_id": case_id,
            "slice_idx": sample.get("slice_idx"),
            "sample_id": sample.get("sample_id"),
            "class": sample.get("class"),
        }

        return (
            torch.from_numpy(image_stack).float(),
            torch.from_numpy(mask_stack).float(),
            meta,
        )
