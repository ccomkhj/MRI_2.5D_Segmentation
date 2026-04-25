"""HTML viewer for GT masks over T2 — column 1 plain T2, column 2 T2 + mask overlay.

For every case in a dataset root (e.g. ``data/aligned_v2`` or ``data/aligned_v3``),
picks the most-massive lesion slice (or most-massive prostate slice if lesion is empty)
and embeds a base64 PNG pair as a single row. When ``--compare-with`` is given,
renders the second root in a third column on the same row so v2 vs v3 can be
eyeballed at a glance.

Reuses helpers from ``mri.diagnostics.report``.

Usage::

    # Single dataset
    uv run python -m tools.dataset.visualize_masks --root data/aligned_v3 \
        --out runs/v3_masks.html

    # Side-by-side v2 vs v3
    uv run python -m tools.dataset.visualize_masks \
        --root data/aligned_v2 --compare-with data/aligned_v3 \
        --out runs/v2_vs_v3_masks.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# Make the package import path work whether this is run as a module or a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

from mri.diagnostics.report import (
    _alpha_blend,
    _base_rgb,
    _png_b64_from_array,
)


_GT_LESION_RGB = (0, 200, 255)  # cyan (matches mri/inference/html_report.py convention)
_GT_PROSTATE_RGB = (255, 255, 0)  # yellow


class PngSink:
    """Where the panel PNGs live.

    Two modes:
      - ``embed=True``  → return base64 ``data:`` URIs (single self-contained HTML).
      - ``embed=False`` → write each PNG into ``asset_dir`` and return a relative path.
        The HTML stays small and ``<img loading="lazy">`` lets the browser decode only
        thumbnails near the viewport — works at cohort scale where embedding doesn't.
    """

    def __init__(self, embed: bool, asset_dir: Path | None, html_dir: Path) -> None:
        self.embed = embed
        self.asset_dir = asset_dir
        self.html_dir = html_dir
        self._counter = 0
        if not embed:
            assert asset_dir is not None
            asset_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rgb: np.ndarray, key: str) -> str:
        """Return either a data URI (embed=True) or a relative-to-HTML path."""
        if self.embed:
            return "data:image/png;base64," + _png_b64_from_array(rgb)
        path = self.asset_dir / f"{key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
        # Use a path relative to the HTML's directory so the file is portable.
        try:
            rel = path.relative_to(self.html_dir)
        except ValueError:
            rel = path  # different drive — fall back to absolute
        return str(rel)


def _load_volume(d: Path) -> np.ndarray | None:
    """Load a directory of indexed PNGs as a (Z, H, W) uint8 volume."""
    if not d.exists():
        return None
    files = sorted(d.glob("*.png"))
    if not files:
        return None
    return np.stack([np.array(Image.open(f).convert("L")) for f in files])


def _pick_anchor_slice(lesion: np.ndarray, prostate: np.ndarray) -> int | None:
    """Pick the most informative z-slice: most lesion mass, fall back to most prostate."""
    per_slice_lesion = (lesion > 127).reshape(lesion.shape[0], -1).sum(axis=1)
    if per_slice_lesion.max() > 0:
        return int(np.argmax(per_slice_lesion))
    per_slice_prostate = (prostate > 127).reshape(prostate.shape[0], -1).sum(axis=1)
    if per_slice_prostate.max() > 0:
        return int(np.argmax(per_slice_prostate))
    return None


def _pick_slice_strip(
    lesion: np.ndarray,
    prostate: np.ndarray,
    n_slices: int,
    n_strip: int,
) -> list[int]:
    """Pick ``n_strip`` z-slices spanning the lesion (or prostate) extent.

    Strategy: find the z-range where the relevant mask is non-empty, then sample
    ``n_strip`` evenly-spaced slices within that range. Falls back to evenly-spaced
    slices across the whole volume when no mask is present.
    """
    if n_strip <= 0:
        return []

    per_slice_lesion = (lesion > 127).reshape(lesion.shape[0], -1).sum(axis=1)
    nonzero = np.where(per_slice_lesion > 0)[0]
    if nonzero.size == 0:
        per_slice_prostate = (prostate > 127).reshape(prostate.shape[0], -1).sum(axis=1)
        nonzero = np.where(per_slice_prostate > 0)[0]

    if nonzero.size == 0:
        # Both masks empty — sample evenly across the volume.
        return [int(round(i * (n_slices - 1) / max(1, n_strip - 1))) for i in range(n_strip)]

    z_lo, z_hi = int(nonzero.min()), int(nonzero.max())
    if z_lo == z_hi:
        # Single non-empty slice — pad with neighbours so the strip has texture.
        pad = (n_strip - 1) // 2
        z_lo = max(0, z_lo - pad)
        z_hi = min(n_slices - 1, z_hi + (n_strip - 1 - pad))
    if n_strip == 1:
        return [int((z_lo + z_hi) // 2)]
    return [
        int(round(z_lo + i * (z_hi - z_lo) / (n_strip - 1)))
        for i in range(n_strip)
    ]


def _t2_plain_rgb(t2_slice: np.ndarray) -> np.ndarray:
    return _base_rgb(t2_slice, t2_slice.shape)


def _t2_with_mask_rgb(t2_slice: np.ndarray, prostate_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    base = _base_rgb(t2_slice, t2_slice.shape)
    out = _alpha_blend(base, _GT_PROSTATE_RGB, (prostate_mask > 127).astype(np.float32) * 0.25)
    out = _alpha_blend(out, _GT_LESION_RGB, (lesion_mask > 127).astype(np.float32) * 0.55)
    return out


def _leakage_ratio(prostate: np.ndarray, lesion: np.ndarray) -> float | None:
    """Fraction of lesion voxels falling outside the prostate. None if either mask empty."""
    pb = prostate > 127
    lb = lesion > 127
    if not lb.any():
        return None
    if not pb.any():
        return 1.0  # lesion exists but no prostate mask — definitionally fully outside
    outside = int(np.logical_and(lb, np.logical_not(pb)).sum())
    return outside / int(lb.sum())


def _format_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _gather_case_ids(root: Path) -> list[str]:
    """Discover case directories as 'class<n>/case_<id>' relative paths."""
    cases: list[str] = []
    for class_dir in sorted(root.glob("class*")):
        if not class_dir.is_dir():
            continue
        for case_dir in sorted(class_dir.iterdir()):
            if case_dir.is_dir() and case_dir.name.startswith("case_"):
                cases.append(f"{class_dir.name}/{case_dir.name}")
    return cases


def _modality_panels(
    case_dir: Path,
    modality: str,
    slice_idxs: list[int],
    prostate: np.ndarray,
    lesion: np.ndarray,
    label: str,
    placeholder_shape: tuple[int, int],
    sink: PngSink,
    case_id: str,
) -> tuple[list[dict], list[dict]]:
    """Return (plain_panels, overlay_panels) for one modality at the given slices."""
    vol = _load_volume(case_dir / modality)
    plain: list[dict] = []
    overlay: list[dict] = []
    safe_case = case_id.replace("/", "_")
    if vol is None:
        h, w = placeholder_shape
        blank_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for z in slice_idxs:
            blank_src = sink.write(blank_arr, f"{safe_case}/{label}_{modality}_blank_z{z}")
            plain.append({"title": f"{modality.upper()} z={z} (missing)", "src": blank_src})
            overlay.append({"title": f"{modality.upper()}+GT[{label}] z={z} (missing)", "src": blank_src})
        return plain, overlay
    for z in slice_idxs:
        cz = min(z, vol.shape[0] - 1)
        plain_src = sink.write(_t2_plain_rgb(vol[cz]), f"{safe_case}/{label}_{modality}_plain_z{cz}")
        plain.append({"title": f"{modality.upper()} z={cz}", "src": plain_src})
        if prostate.shape == vol.shape and lesion.shape == vol.shape:
            overlay_arr = _t2_with_mask_rgb(vol[cz], prostate[cz], lesion[cz])
            overlay_src = sink.write(overlay_arr, f"{safe_case}/{label}_{modality}_overlay_z{cz}")
            overlay.append({"title": f"{modality.upper()}+GT[{label}] z={cz}", "src": overlay_src})
        else:
            fallback_src = sink.write(_t2_plain_rgb(vol[cz]), f"{safe_case}/{label}_{modality}_fallback_z{cz}")
            overlay.append({
                "title": f"{modality.upper()}+GT[{label}] z={cz} (shape mismatch)",
                "src": fallback_src,
            })
    return plain, overlay


def _render_case_row(
    case_id: str,
    root: Path,
    compare_root: Path | None,
    n_strip: int = 5,
    modalities: tuple[str, ...] = ("t2", "adc", "calc"),
    sink: PngSink | None = None,
) -> dict | None:
    """Build the per-case dict consumed by the HTML renderer.

    Returns None when the primary root has no T2 directory for this case.
    """
    case_dir = root / case_id
    t2 = _load_volume(case_dir / "t2")
    if t2 is None:
        return None

    prostate = _load_volume(case_dir / "mask_prostate")
    lesion = _load_volume(case_dir / "mask_target1")
    has_prostate_dir = prostate is not None
    has_lesion_dir = lesion is not None
    if prostate is None:
        prostate = np.zeros_like(t2, dtype=np.uint8)
    if lesion is None:
        lesion = np.zeros_like(t2, dtype=np.uint8)

    # Strip picker uses the union of v2 and v3 lesion masks so cases where the two
    # converters placed the mask at different z still show both.
    strip_lesion = lesion.copy()
    strip_prostate = prostate.copy()
    compare_prostate = compare_lesion = None
    if compare_root is not None:
        c_les = _load_volume(compare_root / case_id / "mask_target1")
        c_pro = _load_volume(compare_root / case_id / "mask_prostate")
        if c_les is not None and c_les.shape == lesion.shape:
            strip_lesion = np.maximum(strip_lesion, c_les)
            compare_lesion = c_les
        if c_pro is not None and c_pro.shape == prostate.shape:
            strip_prostate = np.maximum(strip_prostate, c_pro)
            compare_prostate = c_pro

    slice_idxs = _pick_slice_strip(strip_lesion, strip_prostate, t2.shape[0], n_strip)
    placeholder_shape = (t2.shape[1], t2.shape[2])

    # Default to embedding when called without a sink (preserves the function's old API).
    if sink is None:
        sink = PngSink(embed=True, asset_dir=None, html_dir=Path.cwd())

    # Build per-modality panel sets for the primary root and (optionally) the compare root.
    primary_label = root.name
    panels_by_modality: dict[str, dict] = {}
    for modality in modalities:
        plain, overlay = _modality_panels(
            case_dir, modality, slice_idxs, prostate, lesion, primary_label, placeholder_shape,
            sink, case_id,
        )
        panels_by_modality[modality] = {"plain": plain, "overlay_primary": overlay}

    if compare_root is not None:
        compare_label = compare_root.name
        c_case_dir = compare_root / case_id
        c_prostate_full = compare_prostate
        c_lesion_full = compare_lesion
        if c_prostate_full is None:
            c_prostate_full = _load_volume(c_case_dir / "mask_prostate")
        if c_lesion_full is None:
            c_lesion_full = _load_volume(c_case_dir / "mask_target1")
        c_t2 = _load_volume(c_case_dir / "t2")
        if c_t2 is None:
            h, w = placeholder_shape
            blank_arr = np.zeros((h, w, 3), dtype=np.uint8)
            safe_case = case_id.replace("/", "_")
            for modality in modalities:
                panels_by_modality[modality]["overlay_compare"] = [
                    {
                        "title": f"{modality.upper()}+GT[{compare_label}] z={z} (missing)",
                        "src": sink.write(blank_arr, f"{safe_case}/{compare_label}_{modality}_blank_z{z}"),
                    }
                    for z in slice_idxs
                ]
            row_compare_leak = "—"
        else:
            if c_prostate_full is None:
                c_prostate_full = np.zeros_like(c_t2, dtype=np.uint8)
            if c_lesion_full is None:
                c_lesion_full = np.zeros_like(c_t2, dtype=np.uint8)
            row_compare_leak = _format_pct(_leakage_ratio(c_prostate_full, c_lesion_full))
            for modality in modalities:
                _, overlay_c = _modality_panels(
                    c_case_dir, modality, slice_idxs, c_prostate_full, c_lesion_full,
                    compare_label, placeholder_shape, sink, case_id,
                )
                panels_by_modality[modality]["overlay_compare"] = overlay_c

    row = {
        "case_id": case_id,
        "slice_idxs": slice_idxs,
        "n_slices": int(t2.shape[0]),
        "lesion_voxels": int((lesion > 127).sum()),
        "prostate_voxels": int((prostate > 127).sum()),
        "has_prostate_dir": has_prostate_dir,
        "has_lesion_dir": has_lesion_dir,
        "leak_full": _format_pct(_leakage_ratio(prostate, lesion)),
        "modalities": list(modalities),
        "panels_by_modality": panels_by_modality,
    }
    if compare_root is not None:
        row["compare_leak_full"] = row_compare_leak
    return row


_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 1.5rem auto; max-width: 1900px; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 1rem; }}
  .case {{ border: 1px solid #ddd; border-radius: 4px; padding: 0.6rem 0.8rem; margin-bottom: 1.2rem; }}
  .case-header {{ font-family: monospace; font-size: 0.95rem; margin-bottom: 0.4rem; }}
  .case-stats {{ color: #555; font-size: 0.85rem; margin-bottom: 0.4rem; }}
  .modality-block {{ margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px dashed #eee; }}
  .modality-block:last-child {{ border-bottom: 0; padding-bottom: 0; }}
  .strip-row {{ display: flex; gap: 0.3rem; margin-bottom: 0.3rem; align-items: flex-start; }}
  .strip-label {{ font-size: 0.8rem; color: #555; min-width: 6rem; padding-top: 0.5rem; font-family: monospace; }}
  .strip-row img {{ width: 200px; display: block; }}
  .panel {{ display: flex; flex-direction: column; align-items: center; }}
  .panel .caption {{ font-size: 0.75rem; color: #777; margin-top: 0.1rem; }}
  .legend {{ background: #fbf6e8; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem; }}
  .leak-bad {{ color: #b00; font-weight: bold; }}
  .warning {{ color: #b00; font-weight: bold; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{subtitle}</p>
<div class="legend">
  Yellow = GT prostate mask · Cyan = GT lesion mask. <b>Each modality (T2, ADC, CALC) gets
  a 3-row block of {n_strip} slices:</b> plain modality, then modality + GT from
  <code>--root</code>, then modality + GT from <code>--compare-with</code> (when given).
  Row labels show the dataset name in square brackets, e.g. <code>T2 + GT [aligned_v3]</code>.
  Slice indices are shared across modalities since ADC/CALC are resampled onto the T2 grid by
  the alignment step. The strip is picked from the union of both roots' lesion masks so cases
  where the two converters placed the mask at different z still show both. <b>Leakage</b> =
  fraction of lesion voxels falling outside the prostate mask (anatomically should be 0);
  displayed per-root in the case header.
</div>
"""


def _render_case_block(r: dict, compare: bool, root_label: str, compare_label: str | None) -> str:
    leak_class = ' class="leak-bad"' if r["leak_full"] not in ("—", "0.0%") else ""

    warnings: list[str] = []
    if not r["has_prostate_dir"]:
        warnings.append('<span class="warning">no mask_prostate/ dir</span>')
    if not r["has_lesion_dir"]:
        warnings.append('<span class="warning">no mask_target1/ dir</span>')
    if r["lesion_voxels"] == 0 and r["has_lesion_dir"]:
        warnings.append('<span class="warning">empty lesion mask</span>')
    warnings_html = (" · " + " · ".join(warnings)) if warnings else ""

    header = (
        f'<div class="case-header">{r["case_id"]} · '
        f'slices {min(r["slice_idxs"])}–{max(r["slice_idxs"])}/{r["n_slices"]} · '
        f'lesion={r["lesion_voxels"]}vox · prostate={r["prostate_voxels"]}vox · '
        f'leak[{root_label}]<span{leak_class}>={r["leak_full"]}</span>'
        + (f' · leak[{compare_label}]={r.get("compare_leak_full", "—")}' if compare else "")
        + warnings_html
        + "</div>"
    )

    # One block per modality: plain → primary overlay → compare overlay (if any).
    blocks: list[str] = []
    for modality in r["modalities"]:
        panels = r["panels_by_modality"][modality]
        modality_label = modality.upper()
        sub_rows: list[str] = []
        sub_rows.append(_strip_row(modality_label, panels["plain"]))
        sub_rows.append(_strip_row(f"{modality_label} + GT [{root_label}]", panels["overlay_primary"]))
        if compare and "overlay_compare" in panels:
            sub_rows.append(_strip_row(f"{modality_label} + GT [{compare_label}]", panels["overlay_compare"]))
        blocks.append(f'<div class="modality-block">{"".join(sub_rows)}</div>')

    return f'<div class="case">{header}{"".join(blocks)}</div>'


def _strip_row(label: str, panels: list[dict]) -> str:
    cells = [
        f'<div class="strip-label">{label}</div>'
    ]
    for p in panels:
        cells.append(
            f'<div class="panel">'
            f'<img src="{p["src"]}" alt="{p["title"]}" loading="lazy">'
            f'<div class="caption">{p["title"]}</div>'
            f'</div>'
        )
    return f'<div class="strip-row">{"".join(cells)}</div>'


def _render_html(
    rows: list[dict],
    title: str,
    subtitle: str,
    compare: bool,
    n_strip: int,
    root_label: str,
    compare_label: str | None,
) -> str:
    head = _HTML_HEADER.format(title=title, subtitle=subtitle, n_strip=n_strip)
    blocks = [_render_case_block(r, compare, root_label, compare_label) for r in rows]
    return head + "\n".join(blocks) + "</body></html>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 2-column T2 / T2-with-mask HTML viewer over a dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Aligned dataset root, e.g. data/aligned_v3")
    parser.add_argument("--compare-with", type=Path, default=None, help="Optional second root to render in a third column")
    parser.add_argument("--out", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N cases (0 = all)")
    parser.add_argument("--slices-per-case", type=int, default=5,
                        help="Number of z-slices to render per case (default: 5)")
    parser.add_argument("--modalities", type=str, default="t2,adc,calc",
                        help="Comma-separated modality directory names to render (default: t2,adc,calc)")
    parser.add_argument("--external-assets", action="store_true",
                        help="Write per-panel PNGs to a sidecar directory instead of base64-embedding "
                             "them in the HTML. Recommended for large cohorts — the HTML stays small "
                             "and lazy-loading lets the browser only decode thumbnails near the viewport.")
    parser.add_argument("--asset-dir", type=Path, default=None,
                        help="When --external-assets is set, where to write the PNGs. "
                             "Default: <out>_assets/ next to the HTML.")
    args = parser.parse_args(argv)

    root: Path = args.root
    compare: Path | None = args.compare_with
    if not root.is_dir():
        print(f"error: --root not a directory: {root}", file=sys.stderr)
        return 2
    if compare is not None and not compare.is_dir():
        print(f"error: --compare-with not a directory: {compare}", file=sys.stderr)
        return 2

    case_ids = _gather_case_ids(root)
    if args.limit > 0:
        case_ids = case_ids[: args.limit]
    print(f"[viz] {len(case_ids)} cases under {root}", file=sys.stderr)

    modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    if not modalities:
        print(f"error: --modalities must list at least one modality", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.external_assets:
        asset_dir = args.asset_dir or (args.out.parent / f"{args.out.stem}_assets")
        sink = PngSink(embed=False, asset_dir=asset_dir, html_dir=args.out.parent)
        print(f"[viz] writing PNGs to {asset_dir}", file=sys.stderr)
    else:
        sink = PngSink(embed=True, asset_dir=None, html_dir=args.out.parent)

    rows: list[dict] = []
    for cid in case_ids:
        try:
            row = _render_case_row(cid, root, compare, n_strip=args.slices_per_case, modalities=modalities, sink=sink)
        except Exception as exc:
            print(f"[viz] skipped {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)

    root_label = root.name
    compare_label = compare.name if compare is not None else None
    title = f"GT mask viewer: {root_label}" + (f" vs {compare_label}" if compare_label else "")
    subtitle = (
        f"Row 2 = {root_label} ({root})"
        + (f" · Row 3 = {compare_label} ({compare})" if compare_label else "")
        + f" · Cases rendered: {len(rows)}"
    )
    html = _render_html(
        rows, title=title, subtitle=subtitle,
        compare=compare is not None, n_strip=args.slices_per_case,
        root_label=root_label, compare_label=compare_label,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[viz] wrote {args.out} ({args.out.stat().st_size} bytes, {len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
