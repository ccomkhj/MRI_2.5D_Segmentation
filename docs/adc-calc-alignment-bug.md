# ADC / CALC alignment bug in `tcia-handler`

**Date:** 2026-04-25
**Affects:** every case in `data/aligned_v2` and `data/aligned_v3` produced by `tcia-handler/service/mapping.py`.

## Symptom

For many cases, the ADC and CALC volumes in `aligned_v?/<case>/` have non-empty content on a different z-range than T2. Specifically, the prostate (visible in T2) lands on slice indices where ADC and CALC are entirely zero. Multi-modal training/inference is silently consuming all-zero ADC/CALC channels at the slices that matter.

Example — `class3/case_0125` in `aligned_v3`:

| z range | T2 | ADC | CALC | prostate mask |
|---|---|---|---|---|
| 0–16 | non-empty | non-empty | non-empty | empty |
| 17–31 | non-empty | non-empty | non-empty | growing (~17 → ~3970 voxels) |
| **32–47** | **non-empty** | **all zeros** | **all zeros** | **peak (4310 voxels at z=35)** |
| 48–59 | non-empty | all zeros | all zeros | empty |

The same case in `aligned_v2` has 48/60 ADC slices populated vs 32/60 in `aligned_v3`. Same `mapping.py` code, different output. The non-determinism is the key symptom.

## Root cause

`tcia-handler/service/mapping.py:extract_slice_locations_from_dicom`:

```python
dcm_files = list(dicom_dir.rglob("*.dcm"))   # filesystem-dependent ordering
for dcm_file in dcm_files:
    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
    if hasattr(ds, 'ImagePositionPatient'):
        z = float(ds.ImagePositionPatient[2])
    slice_locations.append(z)
    if not meta:                          # ← only set on the FIRST file
        meta["Origin"] = [float(x) for x in ds.ImagePositionPatient]
```

The function sorts the z-positions later for the slice-location list, but `meta["Origin"]` is captured from whichever DICOM file the OS returns first in `rglob`. That ordering is filesystem-dependent and non-deterministic across runs.

Downstream in `CaseAligner.align_case`:

```python
t2_volume = self.load_png_series_as_sitk(
    t2_images_dir, mapping.t2.spacing, mapping.t2.origin
)
adc_volume = self.load_png_series_as_sitk(
    adc_images_dir, mapping.adc.spacing, mapping.adc.origin
)
adc_resampled = self.resample_to_reference(adc_volume, t2_volume)
```

`load_png_series_as_sitk` reads `sorted(images_dir.glob("*.png"))` — i.e., PNG slice 0 is the lowest-z slice (the converter wrote them in z-order). It then sets `sitk_img.SetOrigin(origin)` — but the supplied origin is the position of an arbitrary slice, **not** the position of voxel (0,0,0). SITK now believes T2's voxel (0,0,0) is at the wrong z.

Both T2 and ADC have this defect, with different arbitrary origins (different `rglob` orderings). When `resample_to_reference(adc, t2)` runs, SITK transforms physical coordinates between two wrong frames. The transform happens to be approximately correct by some constant offset, but the offset is different per run, and slices at the edges of the offset window are clipped to zero.

A second, related issue: `load_png_series_as_sitk` does not call `SetDirection`, so direction is left as identity. If the source DICOMs have non-axial `ImageOrientationPatient` (oblique acquisitions are common), the resample is wrong even with a correct origin.

## What this means for v2 vs v3

The mesh-rasterization fix that produced `aligned_v3` (separate audit at [`mask-conversion-audit.md`](mask-conversion-audit.md)) is real and helps ~5 of the worst v2 cases. But the ADC/CALC alignment bug is **independent** of that fix and affects both v2 and v3. The v3 ADC/CALC happen to be slightly worse on `case_0125` because that run got a less favorable `rglob` ordering — not because of any code change.

## The multi-modal viewer is fine

`tools/dataset/visualize_masks.py` correctly overlays the GT mask onto whatever pixel data is in the modality volume. The blank ADC and CALC panels seen in `runs/v2_vs_v3_modalities_full.html` are an honest rendering of empty data on disk, not a viewer bug.

## Proposed fix

In `tcia-handler/service/mapping.py:extract_slice_locations_from_dicom`:

1. Build a list of `(z, dcm_path)` tuples first.
2. Sort by `z`.
3. Pull metadata (origin, pixel spacing, image orientation, rows, columns) from the **lowest-z** file. Origin must be the lowest-z slice's `ImagePositionPatient` because `load_png_series_as_sitk` reads PNGs in z-order (lowest first).

In `tcia-handler/service/mapping.py:CaseAligner.load_png_series_as_sitk`:

1. Accept a `direction` parameter.
2. Call `sitk_img.SetDirection(direction)` so resampling honours `ImageOrientationPatient`.

In `MappingPipeline.build_case_mapping`:

1. Read `ImageOrientationPatient` per series and store on `SeriesInfo` alongside spacing/origin.
2. Pass it to `load_png_series_as_sitk`.

After the fix:

- Re-run `mapping.py --all --seg-dir data/processed_seg_v3 --output data/aligned_v3` to regenerate the aligned dataset.
- Re-audit ADC/CALC non-empty slice coverage. Expect the per-prostate-slice ADC coverage to rise to >95% (limited by genuine z-extent differences between T2 and DWI acquisitions, which are real and unavoidable).

## Why it didn't bite earlier

- The original mesh-mask bug was so dominant that mask correctness alone explained low Dice. Nobody looked at ADC/CALC content carefully.
- The model still trained — empty ADC just looks like a strong negative signal at those slices, which the loss function tolerated.
- Aggregate metrics never caught it: T2's contribution to the model carried enough signal that Dice/precision moved a bit even with two of three input channels effectively dark for half the prostate.
- Visualization with the per-modality viewer is what made it visible.
