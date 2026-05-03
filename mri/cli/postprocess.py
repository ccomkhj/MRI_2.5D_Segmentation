"""CLI entry point for gland-constrained segmentation postprocessing.

Usage::

    python -m mri.cli.postprocess <run_dir> [--split val] [--force] \\
        [--lesion-threshold FLOAT] [--gland-threshold FLOAT] [--device DEV]

Reads ``<run_dir>/diagnostic/predictions/<case>/prob.npz`` (running
``mri.diagnostics.dump.dump_predictions`` first if the cache is missing) and
writes
``<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from mri.cli.diagnose import (
    resolve_run_dir,
    _build_model_and_dataloader,
    _create_segmentation_model,
    _load_checkpoint,
)
from mri.diagnostics.dump import dump_predictions
from mri.diagnostics.postprocess import apply_postprocess
from mri.training.trainer import resolve_device


def resolve_postprocess_thresholds(
    cfg: dict,
    *,
    lesion_arg: float | None,
    gland_arg: float | None,
) -> tuple[float, float]:
    metrics_cfg = cfg.get("metrics") or {}
    cfg_threshold = metrics_cfg.get("segmentation_threshold")
    if cfg_threshold is None and (lesion_arg is None or gland_arg is None):
        warnings.warn(
            "No metrics.segmentation_threshold in resolved_config — "
            "falling back to 0.5 for missing flag(s).",
            stacklevel=2,
        )
    lesion = float(lesion_arg) if lesion_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    gland = float(gland_arg) if gland_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    return lesion, gland


def _process_case(
    *,
    case_dir: Path,
    output_dir: Path,
    lesion_threshold: float,
    gland_threshold: float,
    case_id: str | None = None,
) -> bool:
    """Postprocess one cached case. Returns True if written, False if skipped.

    ``case_id`` defaults to ``case_dir.name`` for backward compatibility with
    flat layouts; pass the full relative case path for nested layouts
    (e.g. ``class3/case_0310``).
    """
    if case_id is None:
        case_id = case_dir.name
    prob_path = case_dir / "prob.npz"
    if not prob_path.exists():
        warnings.warn(
            f"[postprocess] {case_id}: prob.npz missing, skipping.",
            stacklevel=2,
        )
        return False
    arrays = np.load(prob_path)
    lesion_prob = arrays["lesion"]
    gland_prob = arrays["gland"]

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob,
        lesion_threshold=lesion_threshold,
        gland_threshold=gland_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "lesion_mask.npz", mask=lesion_mask)
    np.savez_compressed(output_dir / "gland_mask.npz", mask=gland_mask)
    (output_dir / "meta.json").write_text(json.dumps({
        "case_id": case_id,
        "lesion_threshold": lesion_threshold,
        "gland_threshold": gland_threshold,
        "gland_present": gland_present,
        "lesion_voxels_raw": int((lesion_prob >= lesion_threshold).sum()),
        "lesion_voxels_post": int(lesion_mask.sum()),
        "gland_voxels": int(gland_mask.sum()),
    }, indent=2))
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gland-constrain + no-gland-suppress postprocessing "
                    "to a finished run's segmentation predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lesion-threshold", type=float, default=None)
    parser.add_argument("--gland-threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    cfg = yaml.safe_load(paths.resolved_config.read_text()) or {}
    lesion_threshold, gland_threshold = resolve_postprocess_thresholds(
        cfg, lesion_arg=args.lesion_threshold, gland_arg=args.gland_threshold,
    )

    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    postprocessed_dir = diag_root / "postprocessed"

    if not predictions_dir.exists():
        loader, num_slices_per_case = _build_model_and_dataloader(cfg, args.split)
        model = _create_segmentation_model(
            cfg["model"]["name"], **(cfg["model"].get("params") or {}),
        )
        device = resolve_device(args.device)
        _load_checkpoint(model, paths.checkpoint, device)
        dump_summary = dump_predictions(
            model=model,
            dataloader=loader,
            device=device,
            output_dir=predictions_dir,
            num_slices_per_case=num_slices_per_case,
            force=args.force,
            lesion_threshold=lesion_threshold,
            gland_threshold=gland_threshold,
        )
        print(f"[postprocess] dump: {dump_summary}")

    # Case dirs are wherever a prob.npz lives — use rglob so nested case_ids
    # (e.g. "class3/case_0310") work the same as flat case_ids.
    prob_paths = sorted(predictions_dir.rglob("prob.npz"))
    n_written = 0
    for prob_path in prob_paths:
        case_dir = prob_path.parent
        case_id = case_dir.relative_to(predictions_dir).as_posix()
        out_dir = postprocessed_dir / case_id
        if out_dir.exists() and not args.force:
            print(f"[postprocess] {case_id}: cached, skipping (use --force to regenerate).")
            continue
        if out_dir.exists() and args.force:
            for f in out_dir.iterdir():
                f.unlink()
        if _process_case(
            case_dir=case_dir, output_dir=out_dir,
            lesion_threshold=lesion_threshold, gland_threshold=gland_threshold,
            case_id=case_id,
        ):
            n_written += 1

    print(f"[postprocess] wrote {n_written} case(s) to {postprocessed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
