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
