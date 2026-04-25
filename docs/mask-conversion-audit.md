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

`aligned_v2/` is produced by **`tcia-handler`**, not `cancer_detector`. The pipeline (`tcia-handler/service/preprocess.py:step_5_process_overlays`) shells out to `tcia-handler/tools/preprocessing/process_overlay_to_masks.py`, then `tcia-handler/service/mapping.py:CaseAligner.align_case` copies the resulting masks alongside resampled T2/ADC/CALC into `aligned_v2/`. There is **no DICOM-aware mask path** in this pipeline. Three failures compound:

### Failure 1 — mesh is voxelized in mesh-local space

`tcia-handler/tools/preprocessing/process_overlay_to_masks.py:187`:

```python
def mesh_to_voxel_grid(mesh, voxel_size=0.5):
    voxelized = mesh.voxelized(pitch=voxel_size)
    voxel_grid = voxelized.matrix
    origin = voxelized.transform[:3, 3]   # CAPTURED but never returned to disk
    return voxel_grid, origin
```

Two problems:

- `pitch=0.5` is hardcoded — independent of the T2's `PixelSpacing`. T2 in-plane is ~0.5 mm so this happens to match in (x,y); T2 z-spacing is ~3 mm, so the mesh produces ~6× more z-slices than T2 needs.
- The mesh's bounding-box position in patient space (the `origin` that `voxelized.transform` carries) is captured into a local variable and **never used** by anything downstream.

The voxel grid is in patient-LPS-relative-to-mesh-bounding-box-corner. Slice index 0 of this grid is wherever the mesh's bounding box starts in physical space, not slice 0 of the T2 series.

### Failure 2 — slices are written indexed by mesh-local z

`tcia-handler/tools/preprocessing/process_overlay_to_masks.py:226-244`:

```python
for i in range(num_slices):
    slice_data = voxel_grid[:, :, i]
    if not slice_data.any(): continue
    img.save(output_dir / f"{i:04d}.png")
```

`i` here is the mesh-voxel-grid Z index. The file `0000.png` is the bottom of the mesh's bounding box, not slice 0 of the patient anatomy.

### Failure 3 — `align_case` pairs masks with T2 by index, not by physical position

`tcia-handler/service/mapping.py:603-714`. ADC and CALC are correctly resampled with SITK:

```python
adc_resampled = self.resample_to_reference(adc_volume, t2_volume)
self.save_sitk_as_pngs(adc_resampled, output_dir / "adc")
```

For masks, however (lines 668-703):

```python
for i in range(mapping.t2.num_slices):
    src_mask = struct_dir / f"{i:04d}.png"
    if src_mask.exists():
        mask = np.array(Image.open(src_mask))
        if mask.shape != (t2_volume.GetSize()[1], t2_volume.GetSize()[0]):
            mask = ... resize NEAREST ...
        Image.fromarray(mask).save(dst_mask)
    else:
        Image.fromarray(np.zeros(...)).save(dst_mask)
```

There is no `resample_to_reference` call for masks. They're just copied (with optional in-plane resize). T2 slice 0 is paired with mesh-voxel-grid slice 0; T2 slice 1 with mesh-voxel-grid slice 1; etc. Two completely independent coordinate systems are silently treated as identical.

### Why the leakage is bimodal

When the mesh's bounding-box origin happens to coincide with T2 slice 0 in physical space *and* the mesh's z-extent matches the T2 z-extent for that anatomy, the index pairing happens to align — and the mask looks correct. When either fails, the mask is rotated/shifted by an arbitrary amount. That's exactly what the data shows: most cases small/no leakage, a tail catastrophic.

### Note on the apparently-fixed copy in cancer_detector

`cancer_detector/tools/preprocessing/process_overlay_aligned.py` is a DICOM-aware rewrite that reads `ImagePositionPatient` / `ImageOrientationPatient` / `PixelSpacing` and transforms mesh vertices into T2 voxel space. **It was never wired into the production pipeline** — `tcia-handler/service/preprocess.py:225` shells out to the non-aligned script. The aligned variant also has its own bug (the negative-origin clipping at lines 283-294 that this audit originally identified) and is incomplete (`# This is a simplified approach - may need refinement` on line 283). Either fix and deploy that script, or write a fresh one — but do not assume that script as-is is correct.

## Why this didn't get caught earlier

- The training pipeline silently treats absent mask files as zero (`mri/data/datasets/segmentation.py:_load_mask`), so the 10 fully-missing cases just look like "easy negatives."
- Aggregate metrics (Dice, precision, recall) average across cases; a 45% systematic problem still produces non-zero numbers, just bad ones — looks like "the model is hard to train" rather than "the labels are wrong."
- Dice on the misaligned cases is *measurable*: the model can sometimes still find the GT lesion in spite of the shift, especially when the shift is small. The model gets penalised at training time for not predicting where the (shifted) GT says, *and* at validation time for the same. The labelling error is internally consistent with itself; only when you cross-reference against the prostate mask (which is mostly correctly aligned, so the leakage check works) does it become visible.
- The `high_confidence_disagreement` audit heuristic in the diagnostics tool flagged ~70% of val cases. That signal was largely correct — but the fix it implies (cascade gland → lesion) doesn't address the actual root cause (the lesion mesh is anchored at the wrong spatial origin, not just outside the gland).

## Implication for the segmentation leader

The current segmentation leader's `val/threshold_sweep_target_best_dice = 0.40` and `val/precision_target = 0.26` numbers are upper-bound estimates. **The true performance against correctly-aligned labels is unknown.** The 56% of val cases with leakage have lesion targets the model is being asked to learn that don't correspond to lesions in the image. Better-than-current modeling decisions cannot be made from these metrics.

## Recommended next steps, in order

### 1. Replace the mask path in `tcia-handler` with a DICOM-aware rasterizer (highest priority)

Two viable approaches, in increasing order of work:

**Option A — DICOM-aware mesh rasterization (preferred).** Add a new `process_overlay_aligned.py` to `tcia-handler/tools/preprocessing/` that, for each STL mesh:

1. Loads the T2 reference DICOM geometry (origin, spacing, direction).
2. Transforms mesh vertices from physical (LPS) into T2 voxel space using `voxel = inv(direction) @ (physical - origin) / spacing`.
3. Voxelizes / rasterizes the transformed mesh **directly into a `(num_t2_slices, H, W)` volume at T2 voxel resolution** — no separate mesh-local grid, no later resampling.
4. Saves PNGs `0000.png` through `(num_t2_slices-1):04d.png`, one per T2 slice (zeros for empty slices, so `align_case` doesn't have to fill in).

Wire `tcia-handler/service/preprocess.py:step_5_process_overlays` to call this script. Drop the mask-copy block in `align_case` (or leave it as a no-op since the masks are already in T2 grid).

**Option B — Make `align_case` actually resample masks.** Keep `process_overlay_to_masks.py` as-is for the rasterization, but in `align_case`:

1. Load the mesh-local PNG series with proper `sitk.SetSpacing` (use the 0.5 mm pitch from the converter) and `sitk.SetOrigin` (use the `voxelized.transform[:3, 3]` — but this requires `process_overlay_to_masks.py` to also persist the origin alongside the PNGs, e.g. as a `geometry.json` next to them).
2. Call `self.resample_to_reference(mask_volume, t2_volume, interpolator=sitk.sitkNearestNeighbor)` exactly the way ADC and CALC are handled.
3. Save the resampled volume as PNGs.

Option B reuses more of the existing structure but requires a sidecar geometry file that doesn't exist today. Option A is cleaner and self-contained.

Either option needs an explicit unit test: a synthetic mesh whose bounding box is offset from the image origin in patient space, rasterized through the new pipeline, then checked against an expected volume that places the mesh at the correct image-voxel position (not the corner).

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
