"""Per-case prediction dump for finished segmentation runs.

Loads the val dataloader the same way the trainer / inference CLI does, runs
inference one batch at a time, accumulates per-case probability volumes, and
writes ``prob.npz`` + ``gt.npz`` + ``meta.json`` per case.

Pure orchestration over a model + dataloader: it does NOT know how to build
either of them. The CLI in ``mri/cli/diagnose.py`` is responsible for that.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


def _coerce_meta_list(metas: Any) -> list[dict]:
    """The seg dataloader collates meta as a dict-of-lists OR a list-of-dicts.

    Mirror the handling in ``mri/inference/segmentation.py`` so behaviour matches.
    """
    if isinstance(metas, dict):
        first = metas[next(iter(metas))]
        return [{k: metas[k][i] for k in metas} for i in range(len(first))]
    return list(metas)


def dump_predictions(
    *,
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    output_dir: Path,
    num_slices_per_case: Mapping[str, int],
    force: bool = False,
    lesion_threshold: float | None = None,
    gland_threshold: float | None = None,
) -> Dict[str, Any]:
    """Run inference and write per-case artifacts under ``output_dir/<case_id>/``.

    Per-case files:
      - ``prob.npz``     keys: ``gland`` (Z,H,W float32), ``lesion`` (Z,H,W float32)
      - ``gt.npz``       keys: ``gland`` (Z,H,W uint8),   ``lesion`` (Z,H,W uint8)
      - ``meta.json``    fields: case_id, class_label, spatial_shape, num_slices,
                                 predicted_slices, lesion_threshold, gland_threshold

    Args:
      num_slices_per_case: mapping of case_id -> total Z, used to allocate buffers.
        Pulled by the CLI from the metadata index.
      force: when True, ignore any pre-existing ``prob.npz`` and re-dump.
      lesion_threshold, gland_threshold: when provided, persisted into each
        case's ``meta.json`` so downstream consumers know the operating point.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: identify cases with an already-cached prob.npz.
    cached: set[str] = set()
    if not force:
        for case_id in num_slices_per_case:
            p = output_dir / case_id / "prob.npz"
            if p.exists() and p.stat().st_size > 0:
                cached.add(case_id)

    model = model.to(device)
    model.eval()

    spatial_shape: tuple[int, int] | None = None
    buffers: Dict[str, Dict[str, Any]] = {}

    def _ensure_buffer(case_id: str) -> None:
        if case_id in buffers:
            return
        assert spatial_shape is not None
        h, w = spatial_shape
        n_z = num_slices_per_case[case_id]
        buffers[case_id] = {
            "gland_prob": np.zeros((n_z, h, w), dtype=np.float32),
            "lesion_prob": np.zeros((n_z, h, w), dtype=np.float32),
            "gland_gt": np.zeros((n_z, h, w), dtype=np.uint8),
            "lesion_gt": np.zeros((n_z, h, w), dtype=np.uint8),
            "predicted_slices": set(),
            "class_label": None,
        }

    failed_inference_cases: set[str] = set()
    with torch.no_grad():
        for batch in dataloader:
            images, masks, metas = batch[0], batch[1], batch[2]
            if spatial_shape is None:
                spatial_shape = (int(images.shape[-2]), int(images.shape[-1]))
            images = images.to(device)
            try:
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
            except Exception as exc:  # noqa: BLE001 — per-batch error isolation
                meta_list_failed = _coerce_meta_list(metas)
                for m in meta_list_failed:
                    cid = m["case_id"]
                    if cid in num_slices_per_case and cid not in cached:
                        failed_inference_cases.add(cid)
                warnings.warn(
                    f"[dump] inference failed on batch ({len(meta_list_failed)} item(s)): "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                continue

            meta_list = _coerce_meta_list(metas)
            mask_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks

            for i, m in enumerate(meta_list):
                case_id = m["case_id"]
                if case_id in cached:
                    continue
                if case_id not in num_slices_per_case:
                    continue
                _ensure_buffer(case_id)
                slice_idx = int(m["slice_idx"])
                buf = buffers[case_id]
                if slice_idx >= buf["gland_prob"].shape[0]:
                    raise ValueError(
                        f"slice_idx={slice_idx} out of range for case {case_id} "
                        f"(num_slices={buf['gland_prob'].shape[0]}); "
                        "metadata index may be stale"
                    )
                buf["gland_prob"][slice_idx] = probs[i, 0]
                if probs.shape[1] > 1:
                    buf["lesion_prob"][slice_idx] = probs[i, 1]
                buf["gland_gt"][slice_idx] = (mask_np[i, 0] > 0.5).astype(np.uint8)
                if mask_np.shape[1] > 1:
                    buf["lesion_gt"][slice_idx] = (mask_np[i, 1] > 0.5).astype(np.uint8)
                buf["predicted_slices"].add(slice_idx)
                if buf["class_label"] is None and "class" in m:
                    cls = m["class"]
                    if cls is not None:
                        buf["class_label"] = int(cls)

    # Cases whose every batch errored out: don't write partial artifacts.
    fully_failed = {
        cid for cid in failed_inference_cases
        if cid not in buffers or not buffers[cid]["predicted_slices"]
    }

    # Write artifacts.
    cases_written = 0
    for case_id, buf in buffers.items():
        if case_id in fully_failed:
            continue
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "prob.npz",
            gland=buf["gland_prob"],
            lesion=buf["lesion_prob"],
        )
        np.savez_compressed(
            case_dir / "gt.npz",
            gland=buf["gland_gt"],
            lesion=buf["lesion_gt"],
        )
        meta_doc = {
            "case_id": case_id,
            "class_label": buf["class_label"] if buf["class_label"] is not None else 0,
            "spatial_shape": list(spatial_shape) if spatial_shape is not None else None,
            "num_slices": int(buf["gland_prob"].shape[0]),
            "predicted_slices": sorted(int(s) for s in buf["predicted_slices"]),
            "lesion_threshold": lesion_threshold,
            "gland_threshold": gland_threshold,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta_doc, indent=2))
        cases_written += 1

    cases_incomplete = sum(
        1
        for cid, buf in buffers.items()
        if cid not in fully_failed
        and len(buf["predicted_slices"]) < buf["gland_prob"].shape[0]
    )

    return {
        "cases_written": cases_written,
        "cases_skipped_cached": len(cached),
        "cases_incomplete": cases_incomplete,
        "cases_failed_inference": sorted(fully_failed),
        "output_dir": str(output_dir),
    }
