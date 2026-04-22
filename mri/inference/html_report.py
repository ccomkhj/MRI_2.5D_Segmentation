"""Self-contained HTML report for segmentation inference results."""

from __future__ import annotations

import base64
import html as html_lib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from mri.inference.segmentation import create_segmentation_overlay


_REPORT_PROSTATE_RGB = (255, 255, 0)
_REPORT_TARGET_RGB = (255, 0, 0)
_REPORT_PROSTATE_ALPHA = 0.25
_REPORT_TARGET_ALPHA = 0.85  # stronger than the on-disk overlay so threshold sweeps are visibly different

_GT_PROSTATE_RGBA = (0, 255, 0, 255)    # green outline
_GT_TARGET_RGBA = (0, 200, 255, 255)    # cyan outline


def _contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    up = np.roll(m, -1, axis=0); up[-1, :] = False
    down = np.roll(m, 1, axis=0); down[0, :] = False
    left = np.roll(m, -1, axis=1); left[:, -1] = False
    right = np.roll(m, 1, axis=1); right[:, 0] = False
    return m & ~(up & down & left & right)


def _gt_contour_rgba(prostate_gt: np.ndarray, target_gt: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    rgba = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    p_edge = _contour(prostate_gt) if prostate_gt is not None else None
    t_edge = _contour(target_gt) if target_gt is not None else None
    if p_edge is not None and p_edge.any():
        rgba[p_edge] = _GT_PROSTATE_RGBA
    if t_edge is not None and t_edge.any():
        rgba[t_edge] = _GT_TARGET_RGBA  # target drawn second so it wins on overlap
    return rgba


def _load_mask(mask_path: Path, shape: Tuple[int, int], positive_value: int = 127) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    img = Image.open(mask_path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    return arr > positive_value


def _report_overlay(base_image: np.ndarray, prostate_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    if base_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {base_image.shape}")
    if base_image.dtype != np.uint8:
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)
    overlay = np.stack([base_image, base_image, base_image], axis=-1).astype(np.float32)

    p_bool = prostate_mask.astype(bool)
    if p_bool.any():
        overlay[p_bool] = (
            (1.0 - _REPORT_PROSTATE_ALPHA) * overlay[p_bool]
            + _REPORT_PROSTATE_ALPHA * np.array(_REPORT_PROSTATE_RGB, dtype=np.float32)
        )

    t_bool = target_mask.astype(bool)
    if t_bool.any():
        overlay[t_bool] = (
            (1.0 - _REPORT_TARGET_ALPHA) * overlay[t_bool]
            + _REPORT_TARGET_ALPHA * np.array(_REPORT_TARGET_RGB, dtype=np.float32)
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _load_t2_slice(metadata_root: Path, case_id: str, slice_idx: int, target_shape: Tuple[int, int]) -> np.ndarray:
    image_path = metadata_root / case_id / "t2" / f"{slice_idx:04d}.png"
    if not image_path.exists():
        return np.zeros(target_shape, dtype=np.uint8)
    image = Image.open(image_path).convert("L")
    if image.size != (target_shape[1], target_shape[0]):
        image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)


def _png_to_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _contiguous_ranges(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    sorted_idx = sorted(indices)
    ranges = []
    start = prev = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def _format_ranges(ranges: List[Tuple[int, int]]) -> str:
    if not ranges:
        return "none"
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def generate_html_report(
    case_output_dir: Path,
    case_id: str,
    metadata_root: Path,
    report_path: Path,
    *,
    prostate_threshold: float = 0.5,
    target_thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7),
    default_target_threshold: Optional[float] = None,
    source_zip: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    threshold: Optional[float] = None,  # back-compat: if given, used for both classes, slider disabled
) -> Dict[str, object]:
    """Render a self-contained HTML segmentation report with a target-threshold slider.

    One overlay is precomputed per (slice × target_threshold) and embedded inline.
    The report returns slice lists at the default target threshold so a batch
    dashboard can summarise consistently.
    """

    case_output_dir = Path(case_output_dir)
    metadata_root = Path(metadata_root)
    report_path = Path(report_path)

    prostate = np.load(case_output_dir / "prostate_prob.npy")
    target = np.load(case_output_dir / "target_prob.npy")
    if prostate.shape != target.shape:
        raise ValueError(f"prostate/target shape mismatch: {prostate.shape} vs {target.shape}")

    if threshold is not None:
        prostate_threshold = float(threshold)
        target_thresholds = (float(threshold),)
        default_target_threshold = float(threshold)

    target_ts = sorted({float(t) for t in target_thresholds})
    if not target_ts:
        raise ValueError("target_thresholds must not be empty")
    if default_target_threshold is None:
        default_target_threshold = target_ts[0]
    default_target_threshold = float(default_target_threshold)
    if default_target_threshold not in target_ts:
        target_ts = sorted({*target_ts, default_target_threshold})
    default_idx = target_ts.index(default_target_threshold)

    num_slices, h, w = prostate.shape
    prostate_slices = [i for i in range(num_slices) if prostate[i].max() >= prostate_threshold]
    target_slices_by_threshold: Dict[float, List[int]] = {
        t: [i for i in range(num_slices) if target[i].max() >= t] for t in target_ts
    }
    target_pixels_by_threshold: Dict[float, int] = {
        t: int((target >= t).sum()) for t in target_ts
    }
    default_target_slices = target_slices_by_threshold[default_target_threshold]

    per_slice_overlays: List[List[str]] = []  # [slice][threshold_idx] -> base64 PNG
    per_slice_gt: List[Optional[str]] = []    # [slice] -> base64 RGBA contour PNG (or None)
    per_slice_meta = []
    case_dir = metadata_root / case_id
    has_any_gt = False
    for i in range(num_slices):
        base = _load_t2_slice(metadata_root, case_id, i, (h, w))
        pm = prostate[i] >= prostate_threshold
        variants = []
        for t in target_ts:
            tm = target[i] >= t
            overlay = _report_overlay(base, pm, tm)
            variants.append(_png_to_base64(overlay))
        per_slice_overlays.append(variants)

        fname = f"{i:04d}.png"
        gt_prostate = _load_mask(case_dir / "mask_prostate" / fname, (h, w))
        gt_target1 = _load_mask(case_dir / "mask_target1" / fname, (h, w))
        gt_target2 = _load_mask(case_dir / "mask_target2" / fname, (h, w))
        gt_target = None
        if gt_target1 is not None or gt_target2 is not None:
            gt_target = np.zeros((h, w), dtype=bool)
            if gt_target1 is not None:
                gt_target |= gt_target1
            if gt_target2 is not None:
                gt_target |= gt_target2
        slice_has_gt = (gt_prostate is not None and gt_prostate.any()) or (gt_target is not None and gt_target.any())
        if slice_has_gt:
            has_any_gt = True
            rgba = _gt_contour_rgba(gt_prostate, gt_target, (h, w))
            per_slice_gt.append(_png_to_base64(rgba))
        else:
            per_slice_gt.append(None)

        per_slice_meta.append({
            "p_max": float(prostate[i].max()),
            "t_max": float(target[i].max()),
            "p_hit": bool(prostate[i].max() >= prostate_threshold),
            "gt_p": bool(gt_prostate is not None and gt_prostate.any()),
            "gt_t": bool(gt_target is not None and gt_target.any()),
        })

    slice_cards = []
    for i in range(num_slices):
        meta = per_slice_meta[i]
        p_max, t_max, p_hit = meta["p_max"], meta["t_max"], meta["p_hit"]
        imgs = "".join(
            f"<img class='layer{' active' if ti == default_idx else ''}' data-ti='{ti}' "
            f"src='data:image/png;base64,{per_slice_overlays[i][ti]}' alt='slice {i} t={t}' />"
            for ti, t in enumerate(target_ts)
        )
        gt_img = (
            f"<img class='gt-layer' src='data:image/png;base64,{per_slice_gt[i]}' alt='slice {i} GT' />"
            if per_slice_gt[i] is not None else ""
        )
        gt_info = (
            f"<div class='row'>"
            f"<span class='dot gt-prostate'></span>GT prostate {'✓' if meta['gt_p'] else ''}"
            f"<span class='dot gt-target' style='margin-left:10px'></span>GT target {'✓' if meta['gt_t'] else ''}"
            f"</div>"
            if has_any_gt else ""
        )
        slice_cards.append(
            f"<div class='slice' data-slice='{i}' data-tmax='{t_max:.6f}' data-pmax='{p_max:.6f}' "
            f"data-phit='{int(p_hit)}'>"
            f"<div class='imgwrap'>{imgs}{gt_img}</div>"
            f"<div class='meta'>"
            f"<div class='idx'>slice {i:04d}</div>"
            f"<div class='row'><span class='dot prostate'></span>prostate "
            f"max={p_max:.3f} {'✓' if p_hit else ''}</div>"
            f"<div class='row'><span class='dot target'></span>target "
            f"max={t_max:.3f} <span class='thit'></span></div>"
            f"{gt_info}"
            f"</div></div>"
        )

    def _fmt_path(p: Optional[Path]) -> str:
        return html_lib.escape(str(p)) if p is not None else "—"

    target_summary_items = [
        (f"target @ {t:.2f}",
         f"{len(target_slices_by_threshold[t])} slices "
         f"({_format_ranges(_contiguous_ranges(target_slices_by_threshold[t]))}), "
         f"{target_pixels_by_threshold[t]} px")
        for t in target_ts
    ]

    banner_by_ti: List[Dict[str, str]] = []
    for t in target_ts:
        slices = target_slices_by_threshold[t]
        if slices:
            ranges = _format_ranges(_contiguous_ranges(slices))
            banner_by_ti.append({
                "level": "warn",
                "text": (
                    f"Suspicious lesion regions on slices {ranges} at target "
                    f"threshold {t:.2f}. Clinical review is recommended — "
                    f"this tool is assistive and not a diagnostic conclusion."
                ),
            })
        else:
            banner_by_ti.append({
                "level": "ok",
                "text": (
                    f"No lesion regions at target threshold {t:.2f}. "
                    f"Try lower thresholds with the slider below."
                ),
            })

    summary_items = [
        ("case_id", html_lib.escape(case_id)),
        ("source_zip", _fmt_path(source_zip)),
        ("checkpoint", _fmt_path(checkpoint_path)),
        ("prostate_threshold", f"{prostate_threshold:.2f}"),
        ("target_thresholds", ", ".join(f"{t:.2f}" for t in target_ts)),
        ("default_target_threshold", f"{default_target_threshold:.2f}"),
        ("total_slices", str(num_slices)),
        ("prostate_positive", f"{len(prostate_slices)} ({_format_ranges(_contiguous_ranges(prostate_slices))})"),
    ] + target_summary_items + [
        ("generated_at", datetime.now().isoformat(timespec="seconds")),
    ]
    summary_html = "".join(
        f'<div class="s-row"><span class="s-key">{k}</span><span class="s-val">{v}</span></div>'
        for k, v in summary_items
    )

    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #222; background: #f7f7f8; }
    h1 { margin: 0 0 8px; font-size: 20px; }
    h2 { margin: 24px 0 12px; font-size: 16px; color: #444; }
    .summary { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; max-width: 900px; }
    .s-row { display: flex; gap: 12px; padding: 2px 0; font-size: 13px; }
    .s-key { min-width: 200px; color: #666; font-variant: all-small-caps; letter-spacing: .5px; }
    .s-val { font-family: ui-monospace, SFMono-Regular, monospace; word-break: break-all; }
    .controls { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; max-width: 900px; margin-top: 12px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .controls label { font-size: 13px; color: #444; display: flex; align-items: center; gap: 8px; }
    .controls input[type=range] { width: 260px; }
    .controls .tval { font-family: ui-monospace, monospace; font-weight: 600; min-width: 3em; text-align: right; }
    .slices { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }
    .slice { background: #fff; border: 2px solid #e0e0e0; border-radius: 6px; padding: 8px; }
    .slice .imgwrap { position: relative; width: 100%; aspect-ratio: 1 / 1; overflow: hidden; }
    .slice img.layer { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; opacity: 0; }
    .slice img.layer.active { opacity: 1; }
    .slice img.gt-layer { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; opacity: 1; pointer-events: none; }
    body.hide-gt .slice img.gt-layer { opacity: 0; }
    .slice .meta { font-size: 12px; margin-top: 6px; font-family: ui-monospace, monospace; }
    .slice .idx { font-weight: 600; margin-bottom: 4px; }
    .slice .row { display: flex; align-items: center; gap: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot.prostate { background: #f9a825; }
    .dot.target { background: #c62828; }
    .dot.gt-prostate { background: #00ff00; border: 1px solid #333; }
    .dot.gt-target { background: #00c8ff; border: 1px solid #333; }
    .legend { font-size: 12px; color: #555; margin: 8px 0 0; }
    .banner { max-width: 900px; margin: 8px 0 12px; padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.45; border: 1px solid; }
    .banner.warn { background: #fff4e5; border-color: #f9a825; color: #6b3b00; }
    .banner.ok { background: #e8f5e9; border-color: #66bb6a; color: #1b5e20; }
    .banner .icon { font-weight: 700; margin-right: 6px; }
    """

    slider_disabled = "disabled" if len(target_ts) == 1 else ""
    target_ts_json = json.dumps(target_ts)
    banner_json = json.dumps(banner_by_ti)
    script = (
        "<script>\n"
        f"const TARGET_TS = {target_ts_json};\n"
        f"const DEFAULT_IDX = {default_idx};\n"
        f"const BANNER_BY_TI = {banner_json};\n"
        "function applyThreshold(ti) {\n"
        "  const t = TARGET_TS[ti];\n"
        "  const tval = document.getElementById('tval');\n"
        "  if (tval) tval.textContent = t.toFixed(2);\n"
        "  const banner = document.getElementById('banner');\n"
        "  if (banner) {\n"
        "    const info = BANNER_BY_TI[ti];\n"
        "    banner.className = 'banner ' + info.level;\n"
        "    const icon = info.level === 'warn' ? '\\u26A0' : '\\u2713';\n"
        "    banner.innerHTML = \"<span class='icon'>\" + icon + \"</span>\" + info.text;\n"
        "  }\n"
        "  document.querySelectorAll('.slice').forEach(card => {\n"
        "    const tmax = parseFloat(card.dataset.tmax);\n"
        "    const phit = card.dataset.phit === '1';\n"
        "    const thit = tmax >= t;\n"
        "    card.style.borderColor = thit ? '#c62828' : (phit ? '#f9a825' : '#e0e0e0');\n"
        "    const thitEl = card.querySelector('.thit');\n"
        "    if (thitEl) thitEl.textContent = thit ? '\\u2713' : '';\n"
        "    card.querySelectorAll('img.layer').forEach(img => {\n"
        "      img.classList.toggle('active', parseInt(img.dataset.ti, 10) === ti);\n"
        "    });\n"
        "  });\n"
        "}\n"
        "(function init() {\n"
        "  const s = document.getElementById('tslider');\n"
        "  if (s) {\n"
        "    const handler = e => applyThreshold(parseInt(e.target.value, 10));\n"
        "    s.addEventListener('input', handler);\n"
        "    s.addEventListener('change', handler);\n"
        "  }\n"
        "  const gt = document.getElementById('gt-toggle');\n"
        "  if (gt) {\n"
        "    gt.addEventListener('change', e => document.body.classList.toggle('hide-gt', !e.target.checked));\n"
        "  }\n"
        "  applyThreshold(DEFAULT_IDX);\n"
        "})();\n"
        "</script>"
    )

    gt_toggle_html = (
        "<label><input id='gt-toggle' type='checkbox' checked /> show ground truth</label>"
        if has_any_gt else ""
    )
    controls_html = (
        "<div class='controls'>"
        f"<label>target threshold: "
        f"<input id='tslider' type='range' min='0' max='{len(target_ts) - 1}' step='1' "
        f"value='{default_idx}' {slider_disabled} />"
        f"<span id='tval' class='tval'></span></label>"
        f"{gt_toggle_html}"
        f"<span style='font-size:12px; color:#888;'>"
        f"prostate fixed at {prostate_threshold:.2f}. "
        f"sweep values: {', '.join(f'{t:.2f}' for t in target_ts)}"
        f"</span>"
        "</div>"
    )

    initial_banner = banner_by_ti[default_idx]
    initial_level = initial_banner["level"]
    initial_icon = "⚠" if initial_level == "warn" else "✓"
    banner_html = (
        f"<div id='banner' class='banner {initial_level}'>"
        f"<span class='icon'>{initial_icon}</span>"
        f"{html_lib.escape(initial_banner['text'])}"
        f"</div>"
    )

    body = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Segmentation report — {html_lib.escape(case_id)}</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Segmentation report — {html_lib.escape(case_id)}</h1>"
        f"{banner_html}"
        f"<div class='summary'>{summary_html}</div>"
        f"{controls_html}"
        f"<p class='legend'>Yellow = prostate (≥ prostate_threshold), red = target (≥ slider target threshold). "
        f"Card border: red if target detected, yellow if only prostate detected.</p>"
        f"<h2>Slices</h2>"
        f"<div class='slices'>{''.join(slice_cards)}</div>"
        f"{script}"
        f"</body></html>"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")

    return {
        "report_path": str(report_path),
        "num_slices": int(num_slices),
        "prostate_slices": prostate_slices,
        "target_slices": default_target_slices,
        "target_slices_by_threshold": {str(t): v for t, v in target_slices_by_threshold.items()},
        "target_pixels_by_threshold": {str(t): v for t, v in target_pixels_by_threshold.items()},
        "prostate_threshold": float(prostate_threshold),
        "target_thresholds": target_ts,
        "default_target_threshold": float(default_target_threshold),
    }
