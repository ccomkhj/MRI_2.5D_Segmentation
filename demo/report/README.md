# Demo report bundle

Self-contained HTML output produced by the clinician inference flow
(`mri.service.pipeline.run_aligned_segmentation`) on case
`class3/case_0125` using the production segmentation checkpoint at
`checkpoints/default/`.

## How to view

```
open demo/report/report.html
```

Click **"View 3D ↗"** in the controls strip (or open
`demo/report/predictions/class3/case_0125/visual_3d.html` directly) to
see the rotatable Plotly figure of the postprocessed lesion volume.

## What's in the bundle

- `report.html` — the clinician-facing 2D report. Per-slice T2 with
  yellow prostate + red target overlays, with sliders for slice and
  target threshold. Off-prostate target voxels are masked away (rules 1
  + 2 from `mri.diagnostics.postprocess.apply_postprocess`).
- `predictions/class3/case_0125/visual_3d.html` — interactive 3D
  Plotly Isosurface of the postprocessed lesion. Built with
  `downsample=2` and Plotly served via CDN to keep the file ~5 MB
  (instead of ~50 MB inline). **Requires internet access** to render.
- `predictions/class3/case_0125/postprocess_meta.json` — the
  thresholds applied + voxel counts before/after the gland constraint.

## Regenerating

```
uv run python -c "
from mri.service.pipeline import run_aligned_segmentation
run_aligned_segmentation(
    metadata_path='data/aligned_v2/metadata.json',
    case_id='class3/case_0125',
    checkpoint_path='checkpoints/default',
    output_dir='/tmp/clinician_smoke',
    open_browser=False,
)
"
```

That writes the full bundle (with the 3D HTML inlined, ~75 MB) under
`/tmp/clinician_smoke/`. To match this demo's smaller 3D file size,
re-run `mri.diagnostics.visualization.build_case_figure` with
`downsample=2` and `write_case_html(..., use_cdn=True)`.
