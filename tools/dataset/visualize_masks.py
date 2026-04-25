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


def _render_case_row(case_id: str, root: Path, compare_root: Path | None) -> dict | None:
    """Render one row for the HTML; returns None if the case has no T2 and no masks."""
    case_dir = root / case_id
    t2 = _load_volume(case_dir / "t2")
    if t2 is None:
        return None

    prostate = _load_volume(case_dir / "mask_prostate")
    lesion = _load_volume(case_dir / "mask_target1")
    if prostate is None:
        prostate = np.zeros_like(t2, dtype=np.uint8)
    if lesion is None:
        lesion = np.zeros_like(t2, dtype=np.uint8)

    z = _pick_anchor_slice(lesion, prostate)
    if z is None:
        # No mask anywhere — pick middle slice so we at least see T2.
        z = t2.shape[0] // 2

    leak = _leakage_ratio(prostate[z], lesion[z]) if z is not None else None
    leak_full = _leakage_ratio(prostate, lesion)

    row = {
        "case_id": case_id,
        "z": int(z),
        "n_slices": int(t2.shape[0]),
        "lesion_voxels": int((lesion > 127).sum()),
        "prostate_voxels": int((prostate > 127).sum()),
        "leak_full": _format_pct(leak_full),
        "leak_slice": _format_pct(leak),
        "panels": [
            {"title": "T2", "png_b64": _t2_plain_png(t2[z])},
            {"title": "T2 + GT", "png_b64": _t2_with_mask_png(t2[z], prostate[z], lesion[z])},
        ],
    }

    if compare_root is not None:
        compare_dir = compare_root / case_id
        c_t2 = _load_volume(compare_dir / "t2")
        c_prostate = _load_volume(compare_dir / "mask_prostate")
        c_lesion = _load_volume(compare_dir / "mask_target1")
        if c_t2 is not None:
            if c_prostate is None:
                c_prostate = np.zeros_like(c_t2, dtype=np.uint8)
            if c_lesion is None:
                c_lesion = np.zeros_like(c_t2, dtype=np.uint8)
            # Use the same z if it fits, else clamp.
            cz = min(z, c_t2.shape[0] - 1)
            row["panels"].append({
                "title": f"T2 + GT (compare, z={cz})",
                "png_b64": _t2_with_mask_png(c_t2[cz], c_prostate[cz], c_lesion[cz]),
            })
            row["compare_leak_full"] = _format_pct(_leakage_ratio(c_prostate, c_lesion))
        else:
            row["panels"].append({"title": "compare: missing", "png_b64": _png_b64_from_array(np.zeros((t2.shape[1], t2.shape[2], 3), dtype=np.uint8))})
            row["compare_leak_full"] = "—"
    return row


_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 1.5rem auto; max-width: 1400px; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 0.4rem; vertical-align: top; }}
  th {{ background: #f4f4f4; text-align: left; }}
  .case-cell {{ font-family: monospace; font-size: 0.8rem; }}
  img {{ max-width: 320px; display: block; }}
  .legend {{ background: #fbf6e8; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.9rem; }}
  .leak-bad {{ color: #b00; font-weight: bold; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{subtitle}</p>
<div class="legend">
  Yellow = GT prostate mask · Cyan = GT lesion mask. <b>Slice picker</b>: most-massive
  lesion slice; falls back to most-massive prostate slice if lesion is empty; falls
  back to the middle T2 slice if both masks are empty. <b>Leakage</b> = fraction of
  lesion voxels falling outside the GT prostate mask (anatomically should be 0).
</div>
"""


def _render_html(rows: list[dict], title: str, subtitle: str, compare: bool) -> str:
    head = _HTML_HEADER.format(title=title, subtitle=subtitle)
    headers = (
        "<tr><th>case</th><th>slice</th><th>vol (l/p)</th><th>leak</th>"
        + ("<th>leak (compare)</th>" if compare else "")
        + "<th>T2</th><th>T2 + GT</th>"
        + ("<th>T2 + GT (compare)</th>" if compare else "")
        + "</tr>"
    )
    body_rows: list[str] = [headers]
    for r in rows:
        leak_class = ' class="leak-bad"' if r["leak_full"] not in ("—", "0.0%") else ""
        cells = [
            f'<td class="case-cell">{r["case_id"]}</td>',
            f'<td>{r["z"]}/{r["n_slices"]}</td>',
            f'<td>{r["lesion_voxels"]}/{r["prostate_voxels"]}</td>',
            f'<td{leak_class}>{r["leak_full"]}</td>',
        ]
        if compare:
            cells.append(f'<td>{r.get("compare_leak_full", "—")}</td>')
        for panel in r["panels"]:
            cells.append(
                f'<td><img src="data:image/png;base64,{panel["png_b64"]}" alt="{panel["title"]}"></td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return head + "<table>" + "\n".join(body_rows) + "</table></body></html>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 2-column T2 / T2-with-mask HTML viewer over a dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Aligned dataset root, e.g. data/aligned_v3")
    parser.add_argument("--compare-with", type=Path, default=None, help="Optional second root to render in a third column")
    parser.add_argument("--out", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N cases (0 = all)")
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
            row = _render_case_row(cid, root, compare)
        except Exception as exc:
            print(f"[viz] skipped {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)

    title = f"GT mask viewer: {root.name}" + (f" vs {compare.name}" if compare else "")
    subtitle = f"Root: {root}" + (f" · Compare: {compare}" if compare else "") + f" · Cases rendered: {len(rows)}"
    html = _render_html(rows, title=title, subtitle=subtitle, compare=compare is not None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[viz] wrote {args.out} ({args.out.stat().st_size} bytes, {len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
