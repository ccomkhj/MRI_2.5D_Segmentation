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

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from mri.diagnostics.dump import _coerce_meta_list
from mri.diagnostics.report import alpha_blend, base_rgb


_GT_PROSTATE_RGB = (255, 255, 0)   # yellow
_GT_LESION_RGB = (0, 200, 255)     # cyan
_PRED_LESION_RGB = (255, 0, 0)     # red


def _load_slices(d: Path, slice_idxs: list[int] | None = None) -> np.ndarray | None:
    """Load a directory of indexed PNGs as a uint8 ``(Z, H, W)`` volume.

    When ``slice_idxs`` is given, only those indices are decoded (clamped to
    the valid range), which avoids loading 30 PNGs to use 5. Returns None if
    the directory has no PNGs.
    """
    files = sorted(d.glob("*.png")) if d.exists() else []
    if not files:
        return None
    if slice_idxs is None:
        return np.stack([np.array(Image.open(f).convert("L")) for f in files])
    n = len(files)
    return np.stack([
        np.array(Image.open(files[max(0, min(z, n - 1))]).convert("L"))
        for z in slice_idxs
    ])


def _pick_strip(lesion: np.ndarray, prostate: np.ndarray, n_slices: int, n_strip: int) -> list[int]:
    """Pick ``n_strip`` distinct z-slices spanning the GT lesion (or prostate) extent."""
    per_lesion = (lesion > 127).reshape(lesion.shape[0], -1).sum(axis=1)
    nz = np.where(per_lesion > 0)[0]
    if nz.size == 0:
        per_prost = (prostate > 127).reshape(prostate.shape[0], -1).sum(axis=1)
        nz = np.where(per_prost > 0)[0]
    if nz.size == 0:
        return [int(round(i * (n_slices - 1) / max(1, n_strip - 1))) for i in range(n_strip)]
    z_lo, z_hi = int(nz.min()), int(nz.max())
    # Single-pass clamp: ensure z_hi - z_lo >= n_strip - 1 by shifting z_lo up to
    # n_slices-1-needed if the volume can't grow z_hi any further; works for any
    # case (range too narrow, range at low edge, range at high edge).
    needed = n_strip - 1
    z_lo = max(0, min(z_lo, n_slices - 1 - needed))
    z_hi = min(n_slices - 1, max(z_hi, z_lo + needed))
    if n_strip == 1:
        return [(z_lo + z_hi) // 2]
    return [int(round(z_lo + i * (z_hi - z_lo) / (n_strip - 1))) for i in range(n_strip)]


def _resize_to(arr: np.ndarray, target_hw: tuple[int, int], mode: str) -> np.ndarray:
    """Resize a 2D array (uint8 mask or float32 probability) to ``target_hw``.

    ``mode`` is ``"nearest"`` for binary masks (preserves 0/255 values) or
    ``"bilinear"`` for continuous probability maps. The model is trained at a
    smaller working resolution than the on-disk modality slices, so each
    overlay needs the prediction / mask resampled up to match the slice
    being rendered before ``alpha_blend`` can broadcast.
    """
    if arr.shape == target_hw:
        return arr
    pil_mode = Image.NEAREST if mode == "nearest" else Image.BILINEAR
    if arr.dtype == np.float32:
        # PIL doesn't resize float32 cleanly; route through uint8 *255 then back.
        scaled = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(scaled).resize((target_hw[1], target_hw[0]), pil_mode)
        return np.asarray(img, dtype=np.float32) / 255.0
    img = Image.fromarray(arr.astype(np.uint8)).resize((target_hw[1], target_hw[0]), pil_mode)
    return np.asarray(img)


def _save_panel(rgb: np.ndarray, case_dir: Path, name: str, rel_prefix: str) -> str:
    """Write a panel PNG into ``case_dir`` and return the path relative to the HTML.

    ``case_dir`` is expected to exist (the caller mkdir's it once per case so we
    don't pay the syscall per panel). ``rel_prefix`` is the case-relative
    asset-directory prefix used in the ``<img src=...>`` attribute.
    """
    arr = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(case_dir / f"{name}.png")
    return f"{rel_prefix}/{name}.png"


def _modality_with_overlay(
    mod_slice: np.ndarray,
    prostate_slice: np.ndarray,
    lesion_slice: np.ndarray,
) -> np.ndarray:
    """Plain modality + yellow prostate + cyan GT lesion."""
    base = base_rgb(mod_slice, mod_slice.shape)
    out = alpha_blend(base, _GT_PROSTATE_RGB, (prostate_slice > 127).astype(np.float32) * 0.25)
    return alpha_blend(out, _GT_LESION_RGB, (lesion_slice > 127).astype(np.float32) * 0.55)


def _modality_with_pred(
    mod_slice: np.ndarray,
    pred_lesion: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Modality + soft-red predicted lesion probability."""
    base = base_rgb(mod_slice, mod_slice.shape)
    alpha = np.clip(pred_lesion, 0.0, 1.0) * 0.6
    out = alpha_blend(base, _PRED_LESION_RGB, alpha)
    # Outline the binarized prediction at the operating threshold so it's visible
    # even in regions where probability is low but above threshold.
    return alpha_blend(
        out, _PRED_LESION_RGB, (pred_lesion >= threshold).astype(np.float32) * 0.25
    )


def _accumulate_predictions(
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference on ``val_loader`` once and return per-case lesion-prob volumes.

    Returns ``case_id -> (Z, H, W) float32`` where ``Z`` is ``max(slice_idx)+1``
    seen for the case. Slices the loader didn't yield stay at zero. We only
    persist the lesion channel since the gland channel isn't rendered.
    """
    model.eval()
    flat: dict[str, list[tuple[int, np.ndarray]]] = {}
    spatial: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            metas = batch[2]
            probs = torch.sigmoid(model(images)).cpu().numpy()
            for i, m in enumerate(_coerce_meta_list(metas)):
                cid = str(m["case_id"])
                z = int(m["slice_idx"])
                lesion = probs[i, 1] if probs.shape[1] > 1 else np.zeros_like(probs[i, 0])
                flat.setdefault(cid, []).append((z, lesion))
                spatial[cid] = (lesion.shape[0], lesion.shape[1])

    volumes: dict[str, np.ndarray] = {}
    for cid, entries in flat.items():
        max_z = max(z for z, _ in entries)
        h, w = spatial[cid]
        vol = np.zeros((max_z + 1, h, w), dtype=np.float32)
        for z, lesion in entries:
            vol[z] = lesion
        volumes[cid] = vol
    return volumes


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
    # Wipe stale PNGs from prior best-epoch runs so the asset dir doesn't grow
    # unboundedly across long training runs (case set can shift between epochs
    # if val cases are added/removed).
    if asset_dir.exists():
        shutil.rmtree(asset_dir, ignore_errors=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_rel = asset_dir.name  # relative to html_dir; HTML lives next to asset_dir

    pred_volumes = _accumulate_predictions(model, val_loader, device)
    rows_html: list[str] = []
    n_rendered = 0
    for case_id, pred_lesion_vol in sorted(pred_volumes.items()):
        case_data_dir = metadata_root / case_id
        if not (case_data_dir / "t2").exists():
            continue
        # First, decide which slices we'll show — needs the full mask volumes,
        # which are uint8 and small.
        prostate = _load_slices(case_data_dir / "mask_prostate")
        lesion = _load_slices(case_data_dir / "mask_target1")
        # Fall back to T2 slice count to determine the strip if masks are absent.
        ref_n = pred_lesion_vol.shape[0]
        if prostate is None:
            prostate = np.zeros((ref_n, 1, 1), dtype=np.uint8)
        if lesion is None:
            lesion = np.zeros((ref_n, 1, 1), dtype=np.uint8)
        slice_idxs = _pick_strip(lesion, prostate, max(prostate.shape[0], ref_n), n_strip)

        # Now load only the strip slices for each modality (lazy).
        mods = {
            "t2": _load_slices(case_data_dir / "t2", slice_idxs),
            "adc": _load_slices(case_data_dir / "adc", slice_idxs),
            "calc": _load_slices(case_data_dir / "calc", slice_idxs),
        }
        if mods["t2"] is None:
            continue
        # And only the strip slices of the masks too — but we already loaded
        # the whole mask volumes for the strip picker. Index them.
        gt_prostate = np.stack([prostate[min(z, prostate.shape[0] - 1)] for z in slice_idxs])
        gt_lesion = np.stack([lesion[min(z, lesion.shape[0] - 1)] for z in slice_idxs])
        pred_strip = np.stack([
            pred_lesion_vol[min(z, pred_lesion_vol.shape[0] - 1)] for z in slice_idxs
        ])

        safe = case_id.replace("/", "_")
        case_asset_dir = asset_dir / safe
        case_asset_dir.mkdir(parents=True, exist_ok=True)
        rel_prefix = f"{asset_rel}/{safe}"

        modality_blocks: list[str] = []
        for mod_name, mod_strip in mods.items():
            if mod_strip is None:
                continue
            plain_cells, gt_cells, pred_cells = [], [], []
            target_hw = (mod_strip.shape[1], mod_strip.shape[2])
            for i, z in enumerate(slice_idxs):
                mod_slice = mod_strip[i]
                gt_p = _resize_to(gt_prostate[i], target_hw, "nearest")
                gt_l = _resize_to(gt_lesion[i], target_hw, "nearest")
                pred_p = _resize_to(pred_strip[i], target_hw, "bilinear")
                plain = base_rgb(mod_slice, mod_slice.shape)
                gt = _modality_with_overlay(mod_slice, gt_p, gt_l)
                pred = _modality_with_pred(mod_slice, pred_p, threshold)
                plain_cells.append(_panel(
                    _save_panel(plain, case_asset_dir, f"{mod_name}_plain_z{z}", rel_prefix),
                    f"z={z}",
                ))
                gt_cells.append(_panel(
                    _save_panel(gt, case_asset_dir, f"{mod_name}_gt_z{z}", rel_prefix),
                    f"GT z={z}",
                ))
                pred_cells.append(_panel(
                    _save_panel(pred, case_asset_dir, f"{mod_name}_pred_z{z}", rel_prefix),
                    f"pred z={z}",
                ))
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
