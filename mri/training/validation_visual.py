"""Save a per-case T2/ADC/CALC + GT mask + prediction overlay HTML at best-checkpoint time.

Triggered by ``Trainer.fit`` whenever a new best checkpoint is saved and the
config flag ``train.save_validation_visual`` is true. The output is written
next to the checkpoint as ``<run_name>_best_val_visual.html`` plus a sidecar
PNG directory, so the artifact stays alongside the model that produced it.

Reuses the panel primitives from :mod:`mri.diagnostics.report` so the layout
matches the diagnostic CLI's per-case strips: each case shows three modality
rows (T2, ADC, CALC) and three panel kinds (plain / GT mask overlay /
prediction overlay) at a small set of z-slices.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from mri.diagnostics.report import _alpha_blend, _base_rgb


_GT_PROSTATE_RGB = (255, 255, 0)   # yellow
_GT_LESION_RGB = (0, 200, 255)     # cyan
_PRED_LESION_RGB = (255, 0, 0)     # red

_MODALITIES = ("t2", "adc", "calc")


def _coerce_meta_list(metas):
    """Mirror ``mri.inference.segmentation``'s collation handling."""
    if isinstance(metas, dict):
        first = metas[next(iter(metas))]
        return [{k: metas[k][i] for k in metas} for i in range(len(first))]
    return list(metas)


def _load_volume(d: Path) -> np.ndarray | None:
    if not d.exists():
        return None
    files = sorted(d.glob("*.png"))
    if not files:
        return None
    return np.stack([np.array(Image.open(f).convert("L")) for f in files])


def _pick_strip(lesion: np.ndarray, prostate: np.ndarray, n_slices: int, n_strip: int) -> list[int]:
    """Pick ``n_strip`` distinct z-slices spanning the GT lesion (or prostate) extent.

    When the lesion z-extent is narrower than ``n_strip`` slices, pads symmetrically
    into the surrounding volume so the returned indices are still distinct.
    """
    per_lesion = (lesion > 127).reshape(lesion.shape[0], -1).sum(axis=1)
    nz = np.where(per_lesion > 0)[0]
    if nz.size == 0:
        per_prost = (prostate > 127).reshape(prostate.shape[0], -1).sum(axis=1)
        nz = np.where(per_prost > 0)[0]
    if nz.size == 0:
        return [int(round(i * (n_slices - 1) / max(1, n_strip - 1))) for i in range(n_strip)]
    z_lo, z_hi = int(nz.min()), int(nz.max())
    # Pad symmetrically when the range is narrower than n_strip; clamp to volume.
    needed = n_strip - 1
    if z_hi - z_lo < needed:
        deficit = needed - (z_hi - z_lo)
        pad_lo = deficit // 2
        pad_hi = deficit - pad_lo
        z_lo = max(0, z_lo - pad_lo)
        z_hi = min(n_slices - 1, z_hi + pad_hi)
        # If clamped at one end, push the other end out to keep n_strip distinct.
        if z_hi - z_lo < needed and z_lo == 0:
            z_hi = min(n_slices - 1, needed)
        if z_hi - z_lo < needed and z_hi == n_slices - 1:
            z_lo = max(0, n_slices - 1 - needed)
    if n_strip == 1:
        return [(z_lo + z_hi) // 2]
    return [int(round(z_lo + i * (z_hi - z_lo) / (n_strip - 1))) for i in range(n_strip)]


def _png_path(rgb: np.ndarray, asset_dir: Path, key: str, html_dir: Path) -> str:
    """Write an RGB array as a PNG and return a path relative to the HTML file."""
    out = asset_dir / f"{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out)
    try:
        return str(out.relative_to(html_dir))
    except ValueError:
        return str(out)


def _modality_with_overlay(
    mod_slice: np.ndarray,
    prostate_slice: np.ndarray,
    lesion_slice: np.ndarray,
) -> np.ndarray:
    """Plain modality + yellow prostate + cyan GT lesion."""
    base = _base_rgb(mod_slice, mod_slice.shape)
    out = _alpha_blend(base, _GT_PROSTATE_RGB, (prostate_slice > 127).astype(np.float32) * 0.25)
    return _alpha_blend(out, _GT_LESION_RGB, (lesion_slice > 127).astype(np.float32) * 0.55)


def _modality_with_pred(
    mod_slice: np.ndarray,
    pred_lesion: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Modality + soft-red predicted lesion probability."""
    base = _base_rgb(mod_slice, mod_slice.shape)
    alpha = np.clip(pred_lesion, 0.0, 1.0) * 0.6
    out = _alpha_blend(base, _PRED_LESION_RGB, alpha)
    # Outline the binarized prediction at the operating threshold so it's visible
    # even in regions where probability is low but above threshold.
    return _alpha_blend(
        out, _PRED_LESION_RGB, (pred_lesion >= threshold).astype(np.float32) * 0.25
    )


def _accumulate_predictions(
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
) -> dict[str, dict]:
    """Run inference on ``val_loader`` once and return per-case probability volumes.

    Returns a mapping case_id -> {"lesion": (Z, H, W) float32, "gland": (Z, H, W) float32}
    where ``Z`` is the number of slices the loader actually yielded for the case
    (max slice_idx + 1). Slices not seen in the loader stay at zero.
    """
    model.eval()
    # First, collect everything as a flat list of (case_id, slice_idx, gland_prob, lesion_prob)
    # so we don't have to iterate the loader twice (each pass costs an inference run).
    flat: dict[str, list[tuple[int, np.ndarray, np.ndarray | None]]] = {}
    spatial: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            metas = batch[2]
            probs = torch.sigmoid(model(images)).cpu().numpy()
            meta_list = _coerce_meta_list(metas)
            for i, m in enumerate(meta_list):
                cid = str(m["case_id"])
                z = int(m["slice_idx"])
                gland = probs[i, 0]
                lesion = probs[i, 1] if probs.shape[1] > 1 else None
                flat.setdefault(cid, []).append((z, gland, lesion))
                spatial[cid] = (gland.shape[0], gland.shape[1])

    cases: dict[str, dict] = {}
    for cid, entries in flat.items():
        max_z = max(z for z, _, _ in entries)
        h, w = spatial[cid]
        gland_vol = np.zeros((max_z + 1, h, w), dtype=np.float32)
        lesion_vol = np.zeros((max_z + 1, h, w), dtype=np.float32)
        for z, g, l in entries:
            gland_vol[z] = g
            if l is not None:
                lesion_vol[z] = l
        cases[cid] = {
            "lesion": lesion_vol,
            "gland": gland_vol,
            "n_slices": max_z + 1,
            "spatial_shape": (h, w),
        }
    return cases


def save_validation_visual(
    *,
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
    metadata_root: Path,
    output_path: Path,
    threshold: float = 0.5,
    n_strip: int = 5,
    title_extra: str = "",
) -> Path:
    """Render the best-epoch validation visual to ``output_path``.

    PNG assets are written to ``output_path.with_suffix("") + "_assets/"`` so
    the HTML file itself stays small and the browser lazy-loads thumbnails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_path.parent / f"{output_path.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_path.parent

    cases = _accumulate_predictions(model, val_loader, device)
    rows_html: list[str] = []
    n_rendered = 0
    for case_id, slot in sorted(cases.items()):
        case_dir = metadata_root / case_id
        t2 = _load_volume(case_dir / "t2")
        if t2 is None:
            continue
        adc = _load_volume(case_dir / "adc")
        calc = _load_volume(case_dir / "calc")
        prostate = _load_volume(case_dir / "mask_prostate")
        lesion = _load_volume(case_dir / "mask_target1")
        if prostate is None:
            prostate = np.zeros_like(t2)
        if lesion is None:
            lesion = np.zeros_like(t2)

        slice_idxs = _pick_strip(lesion, prostate, t2.shape[0], n_strip)
        safe = case_id.replace("/", "_")
        modality_blocks: list[str] = []
        for mod_name, mod_vol in (("t2", t2), ("adc", adc), ("calc", calc)):
            if mod_vol is None:
                continue
            plain_cells: list[str] = []
            gt_cells: list[str] = []
            pred_cells: list[str] = []
            for z in slice_idxs:
                cz = min(z, mod_vol.shape[0] - 1)
                plain = _base_rgb(mod_vol[cz], mod_vol[cz].shape)
                gt = _modality_with_overlay(mod_vol[cz], prostate[cz], lesion[cz])
                pred = _modality_with_pred(mod_vol[cz], slot["lesion"][min(z, slot["lesion"].shape[0] - 1)], threshold)
                plain_cells.append(_panel(_png_path(plain, asset_dir, f"{safe}/{mod_name}_plain_z{cz}", html_dir), f"z={cz}"))
                gt_cells.append(_panel(_png_path(gt, asset_dir, f"{safe}/{mod_name}_gt_z{cz}", html_dir), f"GT z={cz}"))
                pred_cells.append(_panel(_png_path(pred, asset_dir, f"{safe}/{mod_name}_pred_z{cz}", html_dir), f"pred z={cz}"))
            modality_blocks.append(_modality_block(mod_name.upper(), plain_cells, gt_cells, pred_cells))
        rows_html.append(f'<div class="case"><div class="case-id">{case_id}</div>{"".join(modality_blocks)}</div>')
        n_rendered += 1

    html = _build_html(rows_html, n_rendered, title_extra)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _panel(rel_path: str, caption: str) -> str:
    return (
        '<div class="panel">'
        f'<img src="{rel_path}" loading="lazy" alt="{caption}">'
        f'<div class="caption">{caption}</div>'
        '</div>'
    )


def _modality_block(mod: str, plain: list[str], gt: list[str], pred: list[str]) -> str:
    def _row(label: str, cells: list[str]) -> str:
        return f'<div class="strip"><div class="strip-label">{label}</div>{"".join(cells)}</div>'
    return (
        '<div class="modality-block">'
        + _row(mod, plain)
        + _row(f"{mod} + GT", gt)
        + _row(f"{mod} + pred", pred)
        + '</div>'
    )


def _build_html(rows: list[str], n_cases: int, title_extra: str) -> str:
    title = "Best-checkpoint validation visual"
    if title_extra:
        title = f"{title} — {title_extra}"
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1900px; margin: 1.5rem auto; padding: 0 1rem; }
    h1 { margin-bottom: 0.2rem; }
    .meta { color: #555; font-size: 0.9rem; margin-bottom: 1rem; }
    .legend { background: #fbf6e8; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem; }
    .case { border: 1px solid #ddd; border-radius: 4px; padding: 0.6rem 0.8rem; margin-bottom: 1.2rem; }
    .case-id { font-family: monospace; font-size: 0.95rem; margin-bottom: 0.4rem; }
    .modality-block { margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px dashed #eee; }
    .modality-block:last-child { border-bottom: 0; padding-bottom: 0; }
    .strip { display: flex; gap: 0.3rem; margin-bottom: 0.3rem; align-items: flex-start; }
    .strip-label { font-size: 0.8rem; color: #555; min-width: 7rem; padding-top: 0.5rem; font-family: monospace; }
    .panel { display: flex; flex-direction: column; align-items: center; }
    .panel img { width: 200px; display: block; }
    .caption { font-size: 0.75rem; color: #777; margin-top: 0.1rem; }
    """
    legend = (
        "Yellow = GT prostate. Cyan = GT lesion. Red = predicted lesion (probability heatmap "
        "+ a stronger red where prediction ≥ threshold). Each case has three modality blocks "
        "(T2 / ADC / CALC), each block has three rows: plain modality, GT overlay, prediction overlay."
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>{title}</title>"
        f"<style>{css}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class=\"meta\">Cases rendered: {n_cases} · Generated: {datetime.now().isoformat(timespec='seconds')}</p>"
        f"<div class=\"legend\">{legend}</div>"
        + "".join(rows)
        + "</body></html>"
    )
