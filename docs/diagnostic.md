# Segmentation diagnostics

Post-hoc error attribution + label audit for a finished segmentation run.

## Usage

```bash
uv run python -m mri.cli.diagnose path/to/run_dir
```

Optional flags:

- `--split val` — dataloader split key (default `val`).
- `--force` — re-run inference even if `diagnostic/predictions/<case_id>/prob.npz` is cached.
- `--include-low-priority` — include priority-3 audit findings in the HTML report.
- `--device cuda` — override torch device (default `cpu`).

## Outputs

Written to `<run_dir>/diagnostic/`:

- `predictions/<case_id>/{prob.npz, gt.npz, meta.json}` — per-case raw artifacts.
- `metrics_by_case.csv` — per-case lesion-channel metrics, including FP-inside vs FP-outside-gland counts.
- `metrics_by_class.csv` — same metrics aggregated by 0–4 class label.
- `label_audit.csv` — flagged cases ranked by priority (1 high → 3 low).
- `report.html` — single-file report tying it all together.

## Heuristics

The audit surfaces — never auto-excludes — cases that match conservative noise patterns:

| Flag | Priority | Meaning |
|---|---|---|
| `class_mask_inconsistent`     | 1 | Class label disagrees with mask presence (class>0 + empty mask, or class=0 + non-empty mask). |
| `high_confidence_disagreement`| 1 | Pred lesion prob ≥ 0.8 over a 3D component ≥ 50 voxels inside GT-gland with no GT-lesion overlap. |
| `tiny_gt_island`              | 2 | GT lesion has a 3D component < 10 voxels. |
| `erratic_slice_consistency`   | 2 | GT lesion has a z-gap ≥ 2 slices between non-empty slices. |
| `class_severity_mismatch`     | 2 | Pred lesion mass is an outlier within its 0–4 class bucket. |
| `gt_volume_outlier`           | 3 | GT lesion volume is in the top or bottom 5% of non-empty cases. |

Defaults are hardcoded in `mri/diagnostics/audit.py:AUDIT_DEFAULTS`. Promote to config only if a follow-up actually changes them.

## Design

Spec: [`docs/superpowers/specs/2026-04-25-segmentation-error-analysis-design.md`](superpowers/specs/2026-04-25-segmentation-error-analysis-design.md).
