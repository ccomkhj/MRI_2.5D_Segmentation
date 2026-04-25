# GT mask conversion audit

**Date:** 2026-04-25
**Scope:** `data/aligned_v2/` (the dataset all current segmentation runs train and validate on)
**Audited by:** running automated checks on every case in `data/splits/2026-03-08.yaml`

## Bottom line

**About 45% of the dataset has biologically impossible GT masks** (lesion voxels outside the prostate mask). About 5% have no masks at all despite being class > 0. The bugs are in the STL-to-PNG conversion step, not in training. No amount of model tuning can outrun this.

| split | total cases | no mask at all | lesion-outside-prostate (any) | ≥ 10% leakage | ≥ 50% leakage | 100% leakage |
|---|---:|---:|---:|---:|---:|---:|
| train | 134 | 8 | 57 (43%) | 20 | 4 | 0 |
| val | 27 | 2 | 15 (56%) | 3 | 2 | 1 |
| test | 33 | 0 | 16 (48%) | 5 | 4 | 0 |
| **total** | **194** | **10** | **88 (45%)** | **28** | **10** | **1** |

Per the dataset's own design, the lesion mask must be a subset of the prostate mask — the prostate is one organ, lesions live inside it. Any lesion voxel outside the prostate mask is, by construction, evidence of a conversion error. The audit measures the fraction of lesion voxels that fall outside the GT prostate mask per case (`leakage`).

## Worst-affected cases

| split | case | leakage | likely cause |
|---|---|---:|---|
| val | `class2/case_0066` | 100.0% | catastrophic spatial shift — lesion at z=0–3 (top of volume) while prostate at z=0–32, no (x,y) overlap |
| test | `class3/case_0347` | 95.8% | catastrophic shift |
| train | `class3/case_0215` | 86.5% | catastrophic shift |
| test | `class4/case_0340` | 83.8% | catastrophic shift |
| test | `class1/case_0927` | 82.7% | catastrophic shift |
| train | `class2/case_0093` | 71.0% | severe shift |
| train | `class4/case_0454` | 66.5% | severe shift |
| train | `class3/case_0166` | 56.1% | severe shift |
| test | `class2/case_0121` | 55.3% | severe shift |
| val | `class3/case_0164` | 51.8% | bottom slice of lesion drops below prostate |

## Cases with no masks at all (10)

These cases are class 1–4 (so by definition have a biopsy-confirmed lesion) but have empty `mask_prostate/` and `mask_target1/` directories on disk. Training picks them up and silently treats them as negative cases. Inference flagged them as `class_mask_inconsistent`.

```
class3/case_0198, class3/case_0318  (val)
+ 8 more in train (run the audit script for the full list)
```

## Root cause

The conversion pipeline has two scripts:

- `tools/preprocessing/process_overlay_to_masks.py` (488 lines) — voxelizes STL meshes in mesh-local coordinates without any DICOM alignment.
- `tools/preprocessing/process_overlay_aligned.py` (567 lines) — DICOM-aware version that reads `ImagePositionPatient`, `PixelSpacing`, `ImageOrientationPatient`, transforms mesh vertices into image-voxel space, and rasterizes.

The directory name `aligned_v2/` suggests the aligned version is what was used. That script has the right structure but at least one identifiable bug at `tools/preprocessing/process_overlay_aligned.py:283-294`:

```python
# Get voxel grid bounds
grid_origin = voxelized.transform[:3, 3]

# Create output volume
volume = np.zeros(dimensions[::-1], dtype=np.uint8)  # [slices, rows, cols]

# Map voxel grid to volume
# This is a simplified approach - may need refinement       <-- comment in source
x_min, y_min, z_min = np.maximum(np.floor(grid_origin).astype(int), 0)
x_max = min(x_min + voxel_grid.shape[0], dimensions[0])
y_max = min(y_min + voxel_grid.shape[1], dimensions[1])
z_max = min(z_min + voxel_grid.shape[2], dimensions[2])

# Copy voxel data to volume
x_end = min(voxel_grid.shape[0], x_max - x_min)
y_end = min(voxel_grid.shape[1], y_max - y_min)
z_end = min(voxel_grid.shape[2], z_max - z_min)

if x_end > 0 and y_end > 0 and z_end > 0:
    volume[z_min:z_max, y_min:y_max, x_min:x_max] = \
        voxel_grid[:x_end, :y_end, :z_end].transpose(2, 1, 0)
```

The bug: when `grid_origin[i]` is negative (i.e. the mesh extends to voxel coordinates beyond the image's origin), the destination index is clipped to `0` via `np.maximum(..., 0)` but the source slice `voxel_grid[:x_end, ...]` is **not** cropped by the corresponding negative offset. The mask is silently shifted by `|grid_origin[i]|` voxels into the image — a translation bug, not a clipping bug.

For meshes that fit comfortably inside the image bounds (positive `grid_origin`), the rasterization is correct. For meshes whose bounding box extends to or past the image origin, the mask is shifted relative to the prostate by the truncated negative offset. This explains the *bimodal* leakage distribution — most cases are fine or have small leakage, a tail are catastrophically misaligned.

A second smaller suspect: `voxel_grid[:x_end, :y_end, :z_end].transpose(2, 1, 0)` assumes trimesh's voxel matrix is in `[x, y, z]` order. Trimesh in fact uses `[i, j, k]` ordering matching its own coordinate system; the transposition that produces a `[slices, rows, cols]` output should be reviewed against trimesh's actual convention for the project's data orientation.

The aligned script's own author left the comment `# This is a simplified approach - may need refinement` on line 283. It was never refined.

## Why this didn't get caught earlier

- The training pipeline silently treats absent mask files as zero (`mri/data/datasets/segmentation.py:_load_mask`), so the 10 fully-missing cases just look like "easy negatives."
- Aggregate metrics (Dice, precision, recall) average across cases; a 45% systematic problem still produces non-zero numbers, just bad ones — looks like "the model is hard to train" rather than "the labels are wrong."
- Dice on the misaligned cases is *measurable*: the model can sometimes still find the GT lesion in spite of the shift, especially when the shift is small. The model gets penalised at training time for not predicting where the (shifted) GT says, *and* at validation time for the same. The labelling error is internally consistent with itself; only when you cross-reference against the prostate mask (which is mostly correctly aligned, so the leakage check works) does it become visible.
- The `high_confidence_disagreement` audit heuristic in the diagnostics tool flagged ~70% of val cases. That signal was largely correct — but the fix it implies (cascade gland → lesion) doesn't address the actual root cause (the lesion mesh is anchored at the wrong spatial origin, not just outside the gland).

## Implication for the segmentation leader

The current segmentation leader's `val/threshold_sweep_target_best_dice = 0.40` and `val/precision_target = 0.26` numbers are upper-bound estimates. **The true performance against correctly-aligned labels is unknown.** The 56% of val cases with leakage have lesion targets the model is being asked to learn that don't correspond to lesions in the image. Better-than-current modeling decisions cannot be made from these metrics.

## Recommended next steps, in order

### 1. Fix the rasterization bug (highest priority, probably 30–60 minutes of work)

In `tools/preprocessing/process_overlay_aligned.py:rasterize_mesh_to_slices`, the negative-origin case must crop the source voxel grid by the negative offset, not just clip the destination index. Concretely:

```python
src_x0 = max(0, -int(np.floor(grid_origin[0])))
src_y0 = max(0, -int(np.floor(grid_origin[1])))
src_z0 = max(0, -int(np.floor(grid_origin[2])))
x_min  = max(0, int(np.floor(grid_origin[0])))
# ... and copy voxel_grid[src_x0:..., src_y0:..., src_z0:...] into volume[z_min:..., y_min:..., x_min:...]
```

Plus an explicit unit test on a synthetic mesh whose bounding box extends to negative voxel coordinates, asserting the resulting volume has the mesh placed at the correct image-voxel position.

### 2. Investigate the 10 "no mask at all" cases

These are a separate failure mode. Possible causes:
- The STL mesh file was missing for these cases
- The mesh failed to load (corrupt STL)
- The mesh's bounding box was entirely outside the image

The conversion script should fail loudly (or write a sentinel) rather than silently producing nothing. Once root cause is identified, fix and re-convert.

### 3. Re-convert the dataset and re-run the diagnosis

After (1) and (2), re-run the conversion on every case, then re-run `python -m mri.cli.diagnose` on the same checkpoint. The leakage rate should drop from 45% to near 0%. If it doesn't, there's a second bug.

### 4. Re-train the segmentation leader on the cleaned dataset

Same recipe as the current leader, no architectural changes. Compare the new val Dice / precision against the current leader's. The improvement is the size of the labeling error correction — separating "labels were wrong" from "model is wrong."

### 5. Only after step 4: revisit modeling fixes

The diagnostic tool's recommendations (gland cascade, harder negatives, etc.) become meaningful only after the labels are correct. Until then, any modeling change is being evaluated against partly-wrong ground truth.

## Reproducing this audit

```bash
uv run python <<'PY'
import yaml, numpy as np
from pathlib import Path
from PIL import Image
splits = yaml.safe_load(Path("data/splits/2026-03-08.yaml").read_text())
data_root = Path("data/aligned_v2")
def load(d):
    files = sorted(d.glob("*.png"))
    return np.stack([np.array(Image.open(f).convert("L")) for f in files]) if files else None
for split, cases in splits.items():
    leak = 0; nomask = 0
    for cid in cases:
        p = load(data_root / cid / "mask_prostate")
        l = load(data_root / cid / "mask_target1")
        if p is None or l is None:
            nomask += 1; continue
        pb, lb = p > 127, l > 127
        if lb.any() and pb.any():
            outside = int(np.logical_and(lb, np.logical_not(pb)).sum())
            if outside > 0: leak += 1
    print(f"{split}: total={len(cases)}, no_mask={nomask}, leakage={leak}")
PY
```

## See also

- [`diagnostic.md`](diagnostic.md) — the diagnostic CLI that surfaced this.
- [`diagnosing-segmentation.md`](diagnosing-segmentation.md) — runbook (the "label noise" branch of step 4 now applies).
- `tools/preprocessing/process_overlay_aligned.py` — the buggy converter.
