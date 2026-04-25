# v3 mask conversion — results vs v2

**Date:** 2026-04-25
**Pipeline:** `tcia-handler` `feat/aligned-mask-rasterizer` branch (commits `623d9b2`, `6cea049`).

## What was run

1. `process_overlay_aligned.py --parquet-dir data/splitted_info --output-dir data/processed_seg_v3` — DICOM-aware mesh rasterization. 235 overlay cases matched, 540 STL files processed, 9,196 PNG slices written, 0 T2 series missing.
2. `mapping.py --all --seg-dir data/processed_seg_v3 --output data/aligned_v3` — multi-modal alignment using v3 masks. 194 cases aligned, 0 failed.

## Cohort-level audit (lesion-outside-prostate)

| split | metric | v2 | v3 | Δ |
|---|---|---:|---:|---:|
| train | leakage (any) | 57 | 58 | +1 |
| train | leakage ≥ 10% | 20 | 15 | **−5** |
| train | leakage ≥ 50% | 4 | 1 | **−3** |
| val | leakage (any) | 15 | 14 | −1 |
| val | leakage ≥ 10% | 3 | 1 | **−2** |
| val | leakage ≥ 50% | 2 | 1 | **−1** |
| val | 100% leakage | 1 | 0 | **−1** |
| test | leakage (any) | 16 | 15 | −1 |
| test | leakage ≥ 10% | 5 | 4 | **−1** |
| test | leakage ≥ 50% | 4 | 3 | **−1** |
| **total** | leakage ≥ 10% | **28** | **20** | **−8** |
| **total** | leakage ≥ 50% | **10** | **5** | **−5** |
| **total** | 100% leakage | **1** | **0** | **−1** |
| total | no mask at all | 10 | 10 | ±0 |

Net positive: half the catastrophic cases (≥ 50% leakage) are fixed, the single 100% case is fixed, and the long tail of small-leak cases shifted toward zero. The "no mask at all" count is unchanged (10 cases) — that's a separate STL-load failure mode.

## Per-case spot check (10 worst v2 offenders)

| case | v2 leak | v3 leak | verdict |
|---|---:|---:|---|
| `class2/case_0066` | 100.0% | **0.0%** | fixed |
| `class3/case_0347` | 95.8% | **0.0%** | fixed |
| `class3/case_0215` | 86.5% | 8.4% | mostly fixed |
| `class4/case_0340` | 83.8% | 69.7% | partial improvement |
| `class1/case_0927` | 82.7% | **82.7%** | not fixed |
| `class2/case_0093` | 71.0% | 5.8% | mostly fixed |
| `class4/case_0454` | 66.5% | 66.9% | not fixed |
| `class3/case_0166` | 56.1% | 28.9% | partial improvement |
| `class2/case_0121` | 55.3% | 55.3% | not fixed |
| `class3/case_0164` | 51.8% | 51.9% | not fixed |

## What v3 actually fixed

The mesh-local-coordinates bug — when the STL mesh's bounding box was placed at the wrong patient-space anchor by `mesh.voxelized()`, slice-paired against T2 by index. The fixed cases (`0066`, `0347`, `0215`, `0093`) are textbook examples of this: v3 moves the mask to the correct z-range and the leakage drops.

## What v3 did NOT fix

Four of the ten worst cases (`0927`, `0454`, `0121`, `0164`) show essentially identical leakage between v2 and v3. The visualisation (`runs/v2_vs_v3_masks.html`) shows the masks at near-identical positions in both versions. This means **the residual leakage is not from the mesh-local-coordinates bug**.

Possible causes for the residual, in order of likelihood:

1. **STL coordinate frame mismatch.** The STL's vertices may not be in DICOM patient-LPS. 3D Slicer's STL export defaults can produce RAS or LPS depending on which scene transformation is applied. If the STL is in a different frame than the DICOM the new rasterizer is referencing, the mask is placed at a transformed position relative to the prostate.
2. **Wrong T2 series matched to the overlay.** Each overlay's series UID is matched to a DICOM directory via the parquet metadata. Some overlays may map to a T2 series that has different geometry than the one the STL was actually drawn against.
3. **Anatomical reality.** A small fraction of the residual leakage is from boundary-voxel effects at the edge of the prostate where the lesion mask legitimately sits at the gland-capsule interface. This produces a few percent of leakage that cannot be eliminated by any rasterizer.

## What to do next

### Short-term: ship v3 for the cases it fixed

For the 8 cases that moved from ≥ 10% leakage to < 10%, retraining on v3 should produce measurably better Dice. The 4 stubborn cases will continue to mislabel during training but at the same level as v2, so they're not net-worse.

### Medium-term: investigate the residual

Pick one stubborn case (e.g. `class1/case_0927`) and:

1. Open the STL in 3D Slicer alongside the matched T2 DICOM. Visually confirm whether the mesh sits inside the prostate when viewed in patient-LPS space.
2. If the mesh is *not* inside the prostate in the source viewer, the STL itself is in a different frame — fix is to apply the right LPS↔RAS flip during ingest.
3. If the mesh *is* inside the prostate in the source viewer, the parquet→T2 mapping is wrong for this case — fix is to re-confirm the series_uid.

### Visualization

`tools/dataset/visualize_masks.py` produces an HTML viewer with column 1 = T2, column 2 = T2 + GT mask overlay (yellow prostate, cyan lesion). Pass `--compare-with` for a third column showing a second dataset root.

```bash
# v2 alone
uv run python -m tools.dataset.visualize_masks --root /path/to/aligned_v2 --out runs/v2.html

# v3 alone
uv run python -m tools.dataset.visualize_masks --root /path/to/aligned_v3 --out runs/v3.html

# v2 vs v3 side-by-side (third column = v3)
uv run python -m tools.dataset.visualize_masks \
    --root /path/to/aligned_v2 \
    --compare-with /path/to/aligned_v3 \
    --out runs/v2_vs_v3.html
```

The HTML embeds PNGs as base64 (~60 MB for 194 cases) so it's a single self-contained file you can open without a server.

The slice picker is GT-mass-anchored on the *primary* root, so for cases where v2 and v3 placed the mask at different z values, the comparison column may appear empty — that's not a rendering bug, it's evidence that v3 moved the mask. The cohort-level audit numbers above are the right summary for those cases.

## Files

- `tools/dataset/visualize_masks.py` (cancer_detector) — the HTML viewer.
- `runs/v2_vs_v3_masks.html` (cancer_detector) — current 194-case v2-vs-v3 comparison artifact.
- `tcia-handler/data/processed_seg_v3/` — new DICOM-aligned mask source.
- `tcia-handler/data/aligned_v3/` — multi-modal aligned dataset built on v3 masks.
- `tcia-handler/data/processed_seg/` and `data/aligned_v2` — unchanged, retained for comparison.
