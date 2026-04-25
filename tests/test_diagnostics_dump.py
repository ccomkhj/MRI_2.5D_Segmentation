"""Smoke test for `dump_predictions` with a stub model and a tiny synthetic loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mri.diagnostics.dump import dump_predictions


class _StubModel(torch.nn.Module):
    """Returns logits that decode to ones in channel 0 and zeros in channel 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0  # gland prob ≈ 1
        return out


def _stub_dataloader(num_cases: int = 2, slices_per_case: int = 4):
    """Yield batches of (images, masks, meta_dict). Mimics SegmentationDataset's collate."""
    batches = []
    for case_idx in range(num_cases):
        for slice_idx in range(slices_per_case):
            images = torch.zeros(1, 5, 8, 8)
            masks = torch.zeros(1, 2, 8, 8)
            masks[0, 0, 2:6, 2:6] = 1  # gland GT
            if slice_idx == 2:
                masks[0, 1, 3:5, 3:5] = 1  # lesion GT only on slice 2
            meta = {
                "case_id": [f"case_{case_idx:02d}"],
                "slice_idx": [slice_idx],
                "class": [case_idx + 1],  # class label per case
            }
            batches.append((images, masks, meta))
    return batches


def test_dump_writes_per_case_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    model = _StubModel()
    loader = _stub_dataloader(num_cases=2, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4, "case_01": 4},
        spatial_shape=(8, 8),
    )

    assert summary["cases_written"] == 2
    for case_id in ("case_00", "case_01"):
        case_dir = out_dir / case_id
        assert (case_dir / "prob.npz").exists()
        assert (case_dir / "gt.npz").exists()
        assert (case_dir / "meta.json").exists()

        prob = np.load(case_dir / "prob.npz")
        assert prob["gland"].shape == (4, 8, 8)
        assert prob["lesion"].shape == (4, 8, 8)
        # gland channel decoded to ~1 by the stub
        assert np.all(prob["gland"] > 0.99)

        gt = np.load(case_dir / "gt.npz")
        assert gt["gland"].sum() > 0
        assert gt["lesion"].sum() > 0  # lesion present on slice 2

        meta = json.loads((case_dir / "meta.json").read_text())
        assert meta["case_id"] == case_id
        assert "class_label" in meta
        assert meta["spatial_shape"] == [8, 8]


def test_dump_skips_cached_cases(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    case_dir = out_dir / "case_00"
    case_dir.mkdir(parents=True)
    # Pre-existing non-empty prob.npz
    np.savez_compressed(case_dir / "prob.npz", gland=np.ones((4, 8, 8), dtype=np.float32),
                        lesion=np.zeros((4, 8, 8), dtype=np.float32))

    model = _StubModel()
    loader = _stub_dataloader(num_cases=1, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4},
        spatial_shape=(8, 8),
        force=False,
    )

    assert summary["cases_skipped_cached"] == 1
    # gt.npz should still not have been written, since we skipped this case
    assert not (case_dir / "gt.npz").exists()


def test_dump_force_overrides_cache(tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    case_dir = out_dir / "case_00"
    case_dir.mkdir(parents=True)
    np.savez_compressed(case_dir / "prob.npz", gland=np.zeros((4, 8, 8), dtype=np.float32),
                        lesion=np.zeros((4, 8, 8), dtype=np.float32))

    model = _StubModel()
    loader = _stub_dataloader(num_cases=1, slices_per_case=4)

    summary = dump_predictions(
        model=model,
        dataloader=loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_slices_per_case={"case_00": 4},
        spatial_shape=(8, 8),
        force=True,
    )

    assert summary["cases_skipped_cached"] == 0
    assert (case_dir / "gt.npz").exists()
