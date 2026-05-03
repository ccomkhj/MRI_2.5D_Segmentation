"""Plotly-based 3D visualization of postprocessed segmentation results.

`build_case_figure` produces a single rotatable scene with three kinds of
isosurface traces:

- GT gland (pale yellow, low opacity) — anatomical context.
- One trace per GT lesion 3D component, color-coded by detection verdict
  (green if detected, gray if missed).
- Postprocessed predicted lesion (red).

Each trace is toggleable via the legend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class ComponentSpec:
    """One GT lesion 3D component to render."""
    mask: np.ndarray
    lesion_id: int
    detected: bool


_GLAND_COLOR = "#f4e285"
_PRED_COLOR = "#d6334b"
_DETECTED_COLOR = "#3fa34d"
_MISSED_COLOR = "#8a8a8a"


def _downsample(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    return arr[::k, ::k, ::k]


def _isosurface_from_mask(
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    opacity: float,
    showlegend: bool = True,
) -> go.Isosurface | None:
    """Wrap a binary mask into a Plotly Isosurface trace.

    Returns None if the mask has no foreground voxels (Plotly rejects empty
    isosurfaces).
    """
    if not mask.any():
        return None
    Z, Y, X = mask.shape
    z_idx, y_idx, x_idx = np.mgrid[0:Z, 0:Y, 0:X]
    return go.Isosurface(
        x=x_idx.flatten(),
        y=y_idx.flatten(),
        z=z_idx.flatten(),
        value=mask.astype(np.float32).flatten(),
        isomin=0.5,
        isomax=1.0,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
    )


def build_case_figure(
    *,
    gt_gland: np.ndarray,
    gt_lesion_components: list[ComponentSpec],
    pred_lesion: np.ndarray,
    downsample: int = 1,
) -> go.Figure:
    """Build the per-case 3D Plotly figure (no I/O)."""
    fig = go.Figure()

    gland_trace = _isosurface_from_mask(
        _downsample(gt_gland, downsample),
        name="GT gland", color=_GLAND_COLOR, opacity=0.15,
    )
    if gland_trace is not None:
        fig.add_trace(gland_trace)

    for comp in gt_lesion_components:
        suffix = "detected" if comp.detected else "missed"
        color = _DETECTED_COLOR if comp.detected else _MISSED_COLOR
        trace = _isosurface_from_mask(
            _downsample(comp.mask, downsample),
            name=f"GT lesion {comp.lesion_id} ({suffix})",
            color=color, opacity=0.55,
        )
        if trace is not None:
            fig.add_trace(trace)

    pred_trace = _isosurface_from_mask(
        _downsample(pred_lesion, downsample),
        name="Predicted lesion", color=_PRED_COLOR, opacity=0.45,
    )
    if pred_trace is not None:
        fig.add_trace(pred_trace)

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


@dataclass(frozen=True)
class CaseSummary:
    """One row in the visuals/index.html gallery."""
    case_id: str
    case_kind: str
    n_gt_lesions: int
    n_detected_lesions: int
    lesion_recall: float | None
    negative_correct: bool | None


def _format_header(meta: dict) -> str:
    """Render a small HTML header bar above the figure."""
    fields = []
    for key in ("case_id", "class_label", "n_gt_lesions",
                "n_detected_lesions", "lesion_recall", "negative_correct"):
        if key in meta and meta[key] is not None:
            fields.append(f"<b>{key}:</b> {meta[key]}")
    return "<div style='font-family:sans-serif;padding:8px;'>" + " &nbsp; ".join(fields) + "</div>"


def write_case_html(
    fig: go.Figure,
    path: Path,
    *,
    header_meta: dict,
    use_cdn: bool = False,
) -> None:
    """Write a self-contained per-case HTML.

    Args:
      fig: figure produced by ``build_case_figure``.
      path: destination ``.html`` path.
      header_meta: rendered as a small header bar above the figure. Any of
          ``case_id``, ``class_label``, ``n_gt_lesions``, ``n_detected_lesions``,
          ``lesion_recall``, ``negative_correct`` keys are surfaced.
      use_cdn: when True, the Plotly JS is loaded from a CDN at view time
          (smaller files, requires network); otherwise it's inlined.
    """
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if use_cdn else "inline",
    )
    header = _format_header(header_meta)
    case_id = header_meta.get("case_id", "case")
    full = (
        "<!doctype html><html><head>"
        f"<title>{case_id}</title>"
        "<meta charset='utf-8'></head><body>"
        f"{header}{plot_div}"
        "</body></html>"
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full)


def write_index_html(summaries: list[CaseSummary], path: Path) -> None:
    """Write a gallery linking to per-case HTMLs."""
    rows = []
    for s in summaries:
        cells = [
            f'<a href="{s.case_id}.html">{s.case_id}</a>',
            s.case_kind,
            str(s.n_gt_lesions),
            str(s.n_detected_lesions),
            "" if s.lesion_recall is None else f"{s.lesion_recall:.3f}",
            "" if s.negative_correct is None else str(s.negative_correct),
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    header = (
        "<tr>"
        "<th>case_id</th><th>case_kind</th><th>n_gt_lesions</th>"
        "<th>n_detected_lesions</th><th>lesion_recall</th>"
        "<th>negative_correct</th>"
        "</tr>"
    )
    html = (
        "<!doctype html><html><head>"
        "<title>evaluation index</title>"
        "<meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;}"
        "table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
        "</style></head><body>"
        f"<h2>Per-case evaluation visuals</h2>"
        f"<table>{header}{''.join(rows)}</table>"
        "</body></html>"
    )
    Path(path).write_text(html)
