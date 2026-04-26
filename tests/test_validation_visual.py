"""Smoke test for the best-checkpoint validation-visual renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mri.training.validation_visual import save_validation_visual


class _StubModel(torch.nn.Module):
    """Returns gland prob ≈ 1, lesion prob centred in the image."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0
        # lesion prob non-zero in a central 4x4 patch so the overlay row is visible
        cy, cx = h // 2, w // 2
        out[:, 1, cy - 2:cy + 2, cx - 2:cx + 2] = 5.0
        return out


def _make_modality(case_dir: Path, modality: str, n_slices: int, hw: tuple[int, int], value: int = 80) -> None:
    d = case_dir / modality
    d.mkdir(parents=True, exist_ok=True)
    for z in range(n_slices):
        Image.fromarray(np.full(hw, value, dtype=np.uint8)).save(d / f"{z:04d}.png")


def _make_mask(case_dir: Path, mask_name: str, n_slices: int, hw: tuple[int, int], z_range: tuple[int, int]) -> None:
    d = case_dir / mask_name
    d.mkdir(parents=True, exist_ok=True)
    h, w = hw
    for z in range(n_slices):
        if z_range[0] <= z <= z_range[1]:
            arr = np.zeros(hw, dtype=np.uint8)
            arr[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4] = 255
        else:
            arr = np.zeros(hw, dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{z:04d}.png")


def _stub_loader(case_id: str, n_slices: int, hw: tuple[int, int]):
    batches = []
    for z in range(n_slices):
        images = torch.zeros(1, 5, hw[0], hw[1])
        masks = torch.zeros(1, 2, hw[0], hw[1])
        meta = {"case_id": [case_id], "slice_idx": [z], "class": [2]}
        batches.append((images, masks, meta))
    return batches


def test_save_validation_visual_writes_html_with_modality_rows(tmp_path: Path) -> None:
    case_id = "class2/case_xx"
    n_slices, hw = 6, (16, 16)
    metadata_root = tmp_path / "data"
    case_dir = metadata_root / case_id
    for modality in ("t2", "adc", "calc"):
        _make_modality(case_dir, modality, n_slices, hw)
    _make_mask(case_dir, "mask_prostate", n_slices, hw, z_range=(1, 4))
    _make_mask(case_dir, "mask_target1", n_slices, hw, z_range=(2, 3))

    output_path = tmp_path / "best_val_visual.html"
    save_validation_visual(
        model=_StubModel(),
        val_loader=_stub_loader(case_id, n_slices, hw),
        device=torch.device("cpu"),
        metadata_root=metadata_root,
        output_path=output_path,
        threshold=0.5,
        n_strip=3,
        title_extra="epoch 1 · dice=0.42",
    )

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert case_id in html
    assert "T2" in html and "ADC" in html and "CALC" in html
    assert "T2 + GT" in html and "T2 + pred" in html
    assert "epoch 1 · dice=0.42" in html

    # Side-car PNG dir was created with thumbnails.
    asset_dir = tmp_path / "best_val_visual_assets"
    assert asset_dir.is_dir()
    pngs = list(asset_dir.rglob("*.png"))
    # 3 modalities × 3 row-types × 3 slices = 27 thumbnails for one case
    assert len(pngs) == 27, f"expected 27 thumbnails, got {len(pngs)}"


def test_save_validation_visual_skips_case_without_modality_dir(tmp_path: Path) -> None:
    """A case whose T2 dir doesn't exist on disk is silently skipped (no crash)."""
    case_id = "class2/case_yy"
    n_slices, hw = 4, (16, 16)
    metadata_root = tmp_path / "data"
    # Don't create case_dir at all.

    output_path = tmp_path / "best_val_visual.html"
    save_validation_visual(
        model=_StubModel(),
        val_loader=_stub_loader(case_id, n_slices, hw),
        device=torch.device("cpu"),
        metadata_root=metadata_root,
        output_path=output_path,
        threshold=0.5,
        n_strip=3,
    )
    html = output_path.read_text(encoding="utf-8")
    assert "Cases rendered: 0" in html
