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


def _t2_plain_png(t2_slice: np.ndarray) -> str:
    return _png_b64_from_array(_base_rgb(t2_slice, t2_slice.shape))


def _t2_with_mask_png(t2_slice: np.ndarray, prostate_mask: np.ndarray, lesion_mask: np.ndarray) -> str:
    base = _base_rgb(t2_slice, t2_slice.shape)
    out = _alpha_blend(base, _GT_PROSTATE_RGB, (prostate_mask > 127).astype(np.float32) * 0.25)
    out = _alpha_blend(out, _GT_LESION_RGB, (lesion_mask > 127).astype(np.float32) * 0.55)
    return _png_b64_from_array(out)


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


def _render_case_row(
    case_id: str,
    root: Path,
    compare_root: Path | None,
    n_strip: int = 5,
) -> dict | None:
    """Render one row for the HTML; returns None if the case has no T2 at all."""
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

    # If we're comparing, pick the strip from the union of lesion masks across
    # both roots so the row covers v2's lesion z AND v3's lesion z.
    strip_lesion = lesion.copy()
    strip_prostate = prostate.copy()
    if compare_root is not None:
        c_les = _load_volume(compare_root / case_id / "mask_target1")
        c_pro = _load_volume(compare_root / case_id / "mask_prostate")
        if c_les is not None and c_les.shape == lesion.shape:
            strip_lesion = np.maximum(strip_lesion, c_les)
        if c_pro is not None and c_pro.shape == prostate.shape:
            strip_prostate = np.maximum(strip_prostate, c_pro)

    slice_idxs = _pick_slice_strip(strip_lesion, strip_prostate, t2.shape[0], n_strip)
    leak_full = _leakage_ratio(prostate, lesion)

    panels = []
    for z in slice_idxs:
        panels.append({"title": f"T2 z={z}", "png_b64": _t2_plain_png(t2[z]), "kind": "t2"})
        panels.append({
            "title": f"T2+GT z={z}",
            "png_b64": _t2_with_mask_png(t2[z], prostate[z], lesion[z]),
            "kind": "overlay",
        })

    row = {
        "case_id": case_id,
        "slice_idxs": slice_idxs,
        "n_slices": int(t2.shape[0]),
        "lesion_voxels": int((lesion > 127).sum()),
        "prostate_voxels": int((prostate > 127).sum()),
        "has_prostate_dir": has_prostate_dir,
        "has_lesion_dir": has_lesion_dir,
        "leak_full": _format_pct(leak_full),
        "panels": panels,
    }

    if compare_root is not None:
        compare_dir = compare_root / case_id
        c_t2 = _load_volume(compare_dir / "t2")
        c_prostate = _load_volume(compare_dir / "mask_prostate")
        c_lesion = _load_volume(compare_dir / "mask_target1")
        if c_t2 is None:
            row["compare_leak_full"] = "—"
            row["compare_panels"] = [
                {"title": "compare: missing", "png_b64": _png_b64_from_array(np.zeros((t2.shape[1], t2.shape[2], 3), dtype=np.uint8))}
                for _ in slice_idxs
            ]
        else:
            if c_prostate is None:
                c_prostate = np.zeros_like(c_t2, dtype=np.uint8)
            if c_lesion is None:
                c_lesion = np.zeros_like(c_t2, dtype=np.uint8)
            row["compare_leak_full"] = _format_pct(_leakage_ratio(c_prostate, c_lesion))
            row["compare_panels"] = [
                {
                    "title": f"compare T2+GT z={min(z, c_t2.shape[0] - 1)}",
                    "png_b64": _t2_with_mask_png(
                        c_t2[min(z, c_t2.shape[0] - 1)],
                        c_prostate[min(z, c_t2.shape[0] - 1)],
                        c_lesion[min(z, c_t2.shape[0] - 1)],
                    ),
                }
                for z in slice_idxs
            ]
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
  Yellow = GT prostate mask · Cyan = GT lesion mask. <b>Strip</b>: {n_strip} z-slices spanning
  the lesion z-range (or prostate z-range if no lesion). When comparing two roots, the
  strip is picked from the union of both roots' lesion masks so cases where v2 vs v3
  placed the mask at different z still show both. <b>Leakage</b> = fraction of lesion
  voxels falling outside the prostate mask (anatomically should be 0).
</div>
"""


def _render_case_block(r: dict, compare: bool) -> str:
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
        f'leak<span{leak_class}>={r["leak_full"]}</span>'
        + (f' · compare leak={r.get("compare_leak_full", "—")}' if compare else "")
        + warnings_html
        + "</div>"
    )

    # Group panels into rows: row 1 = T2 plain, row 2 = T2+GT primary, row 3 = T2+GT compare (if any).
    n_strip = len(r["slice_idxs"])
    t2_panels = [r["panels"][i * 2] for i in range(n_strip)]
    overlay_panels = [r["panels"][i * 2 + 1] for i in range(n_strip)]

    rows_html: list[str] = []
    rows_html.append(_strip_row("T2", t2_panels))
    rows_html.append(_strip_row("T2 + GT", overlay_panels))
    if compare and "compare_panels" in r:
        rows_html.append(_strip_row("T2 + GT (compare)", r["compare_panels"]))

    return f'<div class="case">{header}{"".join(rows_html)}</div>'


def _strip_row(label: str, panels: list[dict]) -> str:
    cells = [
        f'<div class="strip-label">{label}</div>'
    ]
    for p in panels:
        cells.append(
            f'<div class="panel">'
            f'<img src="data:image/png;base64,{p["png_b64"]}" alt="{p["title"]}">'
            f'<div class="caption">{p["title"]}</div>'
            f'</div>'
        )
    return f'<div class="strip-row">{"".join(cells)}</div>'


def _render_html(rows: list[dict], title: str, subtitle: str, compare: bool, n_strip: int) -> str:
    head = _HTML_HEADER.format(title=title, subtitle=subtitle, n_strip=n_strip)
    blocks = [_render_case_block(r, compare) for r in rows]
    return head + "\n".join(blocks) + "</body></html>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 2-column T2 / T2-with-mask HTML viewer over a dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Aligned dataset root, e.g. data/aligned_v3")
    parser.add_argument("--compare-with", type=Path, default=None, help="Optional second root to render in a third column")
    parser.add_argument("--out", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N cases (0 = all)")
    parser.add_argument("--slices-per-case", type=int, default=5,
                        help="Number of z-slices to render per case (default: 5)")
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

    rows: list[dict] = []
    for cid in case_ids:
        try:
            row = _render_case_row(cid, root, compare, n_strip=args.slices_per_case)
        except Exception as exc:
            print(f"[viz] skipped {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)

    title = f"GT mask viewer: {root.name}" + (f" vs {compare.name}" if compare else "")
    subtitle = f"Root: {root}" + (f" · Compare: {compare}" if compare else "") + f" · Cases rendered: {len(rows)}"
    html = _render_html(
        rows, title=title, subtitle=subtitle,
        compare=compare is not None, n_strip=args.slices_per_case,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[viz] wrote {args.out} ({args.out.stat().st_size} bytes, {len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
