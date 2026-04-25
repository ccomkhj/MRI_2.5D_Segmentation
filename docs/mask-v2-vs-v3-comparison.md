# Comparing aligned_v2 vs aligned_v3 mask conversions

## Why two parallel lineages

The original mask conversion (`tcia-handler/tools/preprocessing/process_overlay_to_masks.py`) had a structural alignment bug — see [`mask-conversion-audit.md`](mask-conversion-audit.md). The fix replaces the converter with a DICOM-aware rasterizer (`process_overlay_aligned.py`). To allow non-destructive verification before committing, the new pipeline writes to:

- `tcia-handler/data/processed_seg_v3/` (new aligned mask source)
- `tcia-handler/data/aligned_v3/` (the merged dataset the segmentation pipeline consumes)

The legacy `data/processed_seg/` and `data/aligned_v2/` are left in place so you can run side-by-side audits before deciding to switch.

## Running the v3 conversion

In `tcia-handler/`:

```bash
# 1. Rasterize meshes into T2-aligned masks (writes to data/processed_seg_v3/).
uv run python service/preprocess.py --step process_overlays
# or directly:
uv run python tools/preprocessing/process_overlay_aligned.py --output-dir data/processed_seg_v3

# 2. Run the alignment step pointing at the v3 mask source and writing to aligned_v3.
uv run python service/mapping.py --all \
    --seg-dir data/processed_seg_v3 \
    --output data/aligned_v3
```

After step 2, `tcia-handler/data/aligned_v3/<class>/<case>/{t2,adc,calc,mask_prostate,mask_target1}/` mirrors the layout of `aligned_v2/`.

## Audit: leakage rate v2 vs v3

The same audit script that produced the original 45% finding works on either lineage. The check is "does the lesion mask fall inside the prostate mask?" — biologically required, and converter-bug-sensitive.

```python
# Run this once for v2 and once for v3 with --root data/aligned_v2 / data/aligned_v3.
import yaml, numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

def audit(root: Path, splits_path: Path) -> dict:
    splits = yaml.safe_load(splits_path.read_text())
    by_split: dict = {}
    for split, cases in splits.items():
        no_mask = leak = 0
        leak_ratios = []
        for cid in cases:
            case_dir = root / cid
            try:
                p = np.stack([np.array(Image.open(f).convert("L"))
                              for f in sorted((case_dir / "mask_prostate").glob("*.png"))])
                l = np.stack([np.array(Image.open(f).convert("L"))
                              for f in sorted((case_dir / "mask_target1").glob("*.png"))])
            except (FileNotFoundError, ValueError):
                no_mask += 1
                continue
            pb, lb = p > 127, l > 127
            if not lb.any() or not pb.any():
                no_mask += 1
                continue
            outside = int(np.logical_and(lb, np.logical_not(pb)).sum())
            if outside > 0:
                leak += 1
                leak_ratios.append(outside / int(lb.sum()))
        by_split[split] = {
            "n": len(cases),
            "no_mask": no_mask,
            "leakage": leak,
            "leak_ge_10pct": sum(1 for r in leak_ratios if r >= 0.10),
            "leak_ge_50pct": sum(1 for r in leak_ratios if r >= 0.50),
        }
    return by_split

splits_path = Path("data/splits/2026-03-08.yaml")
v2 = audit(Path("data/aligned_v2"), splits_path)
v3 = audit(Path("data/aligned_v3"), splits_path)

print(f"{'split':>5} {'metric':<14} {'v2':>5} {'v3':>5} {'Δ':>6}")
for split in v2:
    for metric in ("no_mask", "leakage", "leak_ge_10pct", "leak_ge_50pct"):
        a, b = v2[split][metric], v3[split][metric]
        print(f"{split:>5} {metric:<14} {a:>5} {b:>5} {b - a:>+6}")
```

A successful v3 conversion drops `leakage` from ~45% to near zero across all splits. If `no_mask` rises in v3, the new converter is failing to find or load some STL meshes — investigate before proceeding.

## Side-by-side per-case visualisation

To eyeball a specific case, point the diagnostics CLI at each lineage in turn:

```bash
# Run v2 (already cached from the original diagnosis).
uv run python -m mri.cli.diagnose checkpoints/default

# Switch the data symlink to v3 and force re-dump.
rm data/aligned_v2  # if it was a symlink
ln -s /Users/huijokim/personal/tcia-handler/data/aligned_v3 data/aligned_v2
uv run python -m mri.cli.diagnose checkpoints/default --force
```

For a more apples-to-apples comparison without symlink swapping, copy `checkpoints/default/diagnostic/report.html` to a v2 / v3 backup before running each pass.

## Decision after the audit

If v3 leakage is near zero **and** v3 `no_mask` count is no higher than v2's:

1. Repoint `cancer_detector/data/aligned_v2` (the symlink) at `tcia-handler/data/aligned_v3`, or rename `aligned_v2 → aligned_v2_legacy` and `aligned_v3 → aligned_v2` to avoid touching downstream config.
2. Re-run the segmentation leader's training recipe on the cleaned data; compare val Dice / precision against the legacy run.
3. Re-run `mri/cli/diagnose` on the new checkpoint. The audit-queue size and `fp_outside_ratio` should drop materially.

If v3 leakage is non-trivial, the rasterizer has a remaining bug. The `tests/test_mesh_rasterization.py` covers the bugs anticipated; real DICOMs may surface unanticipated ones — investigate with the per-case visualisation before committing.

## Why not overwrite v2 directly

- v2 is the artifact every existing checkpoint was trained on; preserving it lets us measure the *delta* attributable to the labeling fix.
- If v3 turns out to have its own bugs, v2 is the safe rollback.
- Comparing v2 vs v3 metrics is itself a diagnostic — a v3 that produces wildly different lesion volumes (not just shifted positions) hints at the rasterization being wrong in some other way.
