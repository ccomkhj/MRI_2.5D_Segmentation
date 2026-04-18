# Current Segmentation Leader

> **Snapshot, not source of truth.** This file captures a point-in-time leaderboard from [`checkpoints/reports/best_jobs.html`](../checkpoints/reports/best_jobs.html). Refresh after every meaningful sweep — do not link to specific run names from evergreen docs.

## Snapshot — 2026-04-04

Based on `checkpoints/reports/best_jobs.html` generated on `2026-04-04`:

- `81` segmentation jobs were included in the leaderboard.
- Best `val/precision_target`: `seg-auto-seg-apr04-autopilot-debug-w03-r01-simple-s5-prec-md1-w2-cons`
  - `val/precision_target = 0.2603` at epoch `65`
  - `val/threshold_sweep_target_best_dice = 0.3216`
  - Recipe: `SimpleUNet`, `stack_depth=5`, positive-only pool, geometric augmentation, gentle modality dropout, light target weighting, conservative `OneCycle`
  - Config: `checkpoints/autopilot/seg-apr04-autopilot-debug/configs/wave03/01-simple-s5-prec-md1-w2-cons.yaml`
- Stronger threshold-sweep Dice leader: `seg-auto-seg-apr04-autopilot-debug-w02-r11-dynu-s5-prec-md0-w1-cons`
  - `val/threshold_sweep_target_best_dice = 0.4025`
  - Raw `val/precision_target = 0.1030`

For downstream experiments, the natural first checkpoint to try is the precision leader; the Dice leader is a useful comparison if the ROI looks too tight or too loose.

## How to refresh

Regenerate `checkpoints/reports/best_jobs.html`, then update the snapshot above (date, run names, metric values, recipe summary).
