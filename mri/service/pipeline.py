"""End-to-end wrapper: DICOM zip → preprocessing → segmentation → HTML report."""

from __future__ import annotations

import html as html_lib
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import json
import yaml

import torch

from mri.config.loader import load_config
from mri.cli.infer import _build_dataloader, _load_checkpoint
from mri.inference.segmentation import run_segmentation_inference
from mri.inference.html_report import generate_html_report
from mri.models import create_segmentation_model
from mri.training.trainer import resolve_device


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Sibling-clone candidates, tried only if `import dicom_mapper` fails. The expected
# layout is `<parent>/cancer_detector/` + `<parent>/dicom_mapper/`; `tcia-handler`
# is kept for backwards-compat with the pre-rename local checkout.
_DICOM_MAPPER_CANDIDATES = (
    _REPO_ROOT.parent / "dicom_mapper",
    _REPO_ROOT.parent / "tcia-handler",
)


def _import_dicom_mapper():
    try:
        import dicom_mapper  # noqa: F401
        return
    except ImportError:
        pass
    for root in _DICOM_MAPPER_CANDIDATES:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import dicom_mapper  # noqa: F401
    except ImportError as exc:
        hint = _DICOM_MAPPER_CANDIDATES[0]
        raise ImportError(
            "dicom_mapper not importable. Install it first, e.g.\n"
            f"  git clone git@github.com:ccomkhj/dicom_mapper.git {hint}\n"
            f"  uv pip install -e {hint}"
        ) from exc


def _import_metadata_helpers():
    _import_dicom_mapper()
    import dicom_mapper

    tools_dir = Path(dicom_mapper.__file__).resolve().parent.parent / "tools"
    if tools_dir.exists() and str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import generate_training_metadata as gtm  # type: ignore
    return gtm


def _resolve_checkpoint(
    checkpoint_path: Union[str, Path],
    resolved_config_path: Optional[Union[str, Path]],
) -> Dict[str, Path]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        run_dir = checkpoint_path
        best = sorted(run_dir.glob("*_best.pt"))
        last = sorted(run_dir.glob("*_last.pt"))
        if best:
            ckpt_file = best[0]
        elif last:
            ckpt_file = last[0]
        else:
            raise FileNotFoundError(f"No *_best.pt or *_last.pt found under {run_dir}")
        cfg_path = (
            Path(resolved_config_path) if resolved_config_path else run_dir / "resolved_config.yaml"
        )
    else:
        ckpt_file = checkpoint_path
        if resolved_config_path is None:
            default_cfg = ckpt_file.parent / "resolved_config.yaml"
            if not default_cfg.exists():
                raise ValueError(
                    "resolved_config_path is required when checkpoint_path points to a .pt file "
                    "and no resolved_config.yaml is sibling of the checkpoint"
                )
            cfg_path = default_cfg
        else:
            cfg_path = Path(resolved_config_path)
        run_dir = ckpt_file.parent

    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found: {cfg_path}")
    return {"run_dir": run_dir, "ckpt_file": ckpt_file, "config_path": cfg_path}


def _preprocess_dicom_zip(zip_path: Path, staging_dir: Path, group_name: str) -> str:
    _import_dicom_mapper()
    from dicom_mapper.cli.pipeline import _process_single_vendor_zip
    from dicom_mapper.io.export import PNGExporter
    from dicom_mapper.io.vendor import case_name_from_zip
    from dicom_mapper.processing.resampling import VolumeResampler

    case_name = case_name_from_zip(zip_path)
    staging_dir.mkdir(parents=True, exist_ok=True)
    resampler = VolumeResampler()
    exporter = PNGExporter()
    ok = _process_single_vendor_zip(
        zip_path=zip_path,
        case_name=case_name,
        output_root=staging_dir,
        group_name=group_name,
        resampler=resampler,
        exporter=exporter,
    )
    if not ok:
        raise RuntimeError(f"DICOM preprocessing failed for {zip_path}")
    return case_name


def _build_metadata(staging_dir: Path, group_name: str, case_name: str, t2_context_size: int = 5) -> Path:
    gtm = _import_metadata_helpers()

    case_dir = staging_dir / group_name / case_name
    case_id = f"{group_name}/{case_name}"
    modalities = ["t2", "adc", "calc"]
    masks = ["mask_prostate", "mask_target1"]

    case_info, samples = gtm.process_case(
        case_dir=case_dir,
        case_id=case_id,
        class_num=0,
        modalities=modalities,
        masks=masks,
        t2_context_size=t2_context_size,
        positive_value=255,
    )
    if case_info is None:
        raise RuntimeError(f"process_case produced no info for {case_dir}")

    global_stats = gtm.compute_global_stats(
        data_dir=staging_dir,
        case_ids=[case_id],
        modalities=modalities,
        sample_ratio=1.0,
    )

    metadata = {
        "version": "1.0",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "input_size": [256, 256],
            "t2_context_window": t2_context_size,
            "boundary_padding": "edge_replicate",
            "modalities": modalities,
            "masks": masks,
            "mask_positive_value": 255,
        },
        "global_stats": global_stats,
        "cases": {case_id: case_info},
        "samples": samples,
    }

    metadata_path = staging_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def _write_split(staging_dir: Path, case_id: str) -> Path:
    split_path = staging_dir / "split.yaml"
    split_path.write_text(yaml.safe_dump({"train": [], "val": [], "test": [case_id]}))
    return split_path


_DEFAULT_TARGET_SWEEP = (0.1, 0.2, 0.3, 0.5, 0.7)


def _read_best_target_threshold(ckpt_run_dir: Path) -> Optional[float]:
    summary_path = ckpt_run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return None
    best = (
        data.get("summary", {})
        .get("best_val_metrics", {})
        .get("threshold_sweep_target_best_threshold")
    )
    return float(best) if best is not None else None


def run_dicom_segmentation(
    zip_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    resolved_config_path: Optional[Union[str, Path]] = None,
    prostate_threshold: Optional[float] = None,
    target_thresholds: Optional[Sequence[float]] = None,
    default_target_threshold: Optional[float] = None,
    threshold: Optional[float] = None,
    group_name: str = "case_auto",
    open_browser: bool = False,
) -> Dict[str, Any]:
    """Run the full DICOM → segmentation → HTML report pipeline.

    Parameters
    ----------
    zip_path : vendor DICOM zip file.
    checkpoint_path : a training run directory (auto-picks `*_best.pt` + `resolved_config.yaml`)
        or a direct `.pt` file (then `resolved_config_path` must be provided, or there must
        be a `resolved_config.yaml` sibling).
    output_dir : where the pipeline writes `_aligned/`, `predictions/`, and `report.html`.
    resolved_config_path : optional path to a resolved config yaml if not colocated with the ckpt.
    prostate_threshold : probability threshold for the prostate mask. Defaults to
        `metrics.segmentation_threshold` from the checkpoint config (typically 0.5).
    target_thresholds : sweep of target thresholds precomputed into the report. Defaults to
        (0.1, 0.2, 0.3, 0.5, 0.7). The HTML slider flips between these values.
    default_target_threshold : initial slider position and the threshold used to derive the
        returned `target_slices` list and the batch index summary. Defaults to the best
        target threshold recorded in the training `run_summary.json`, else 0.1.
    threshold : legacy single-threshold override. If provided, it is applied to both classes
        and the target sweep collapses to this one value (slider disabled).
    group_name : subdirectory name under staging/ used to scope the auto-preprocessed case.
    open_browser : if True, open ``report.html`` in the default browser after inference.
    """

    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    ckpt = _resolve_checkpoint(checkpoint_path, resolved_config_path)

    staging_dir = output_dir / "_aligned"
    case_name = _preprocess_dicom_zip(zip_path, staging_dir, group_name)
    case_id = f"{group_name}/{case_name}"

    metadata_path = _build_metadata(staging_dir, group_name, case_name)
    split_path = _write_split(staging_dir, case_id)

    cfg = load_config(ckpt["config_path"])
    cfg.setdefault("data", {})
    cfg["data"]["metadata"] = str(metadata_path)
    cfg["data"]["split_file"] = str(split_path)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["require_positive"] = False
    cfg["data"]["require_complete"] = False

    cfg.setdefault("inference", {})
    predictions_dir = output_dir / "predictions"
    cfg["inference"]["checkpoint"] = str(ckpt["ckpt_file"])
    cfg["inference"]["output_dir"] = str(predictions_dir)
    cfg["inference"]["device"] = cfg["inference"].get("device", "auto")
    cfg["inference"]["batch_size"] = cfg["inference"].get("batch_size", 4)

    cfg_threshold = float(cfg.get("metrics", {}).get("segmentation_threshold", 0.5))
    if threshold is not None:
        prostate_threshold = float(threshold)
        target_thresholds = (float(threshold),)
        default_target_threshold = float(threshold)
    else:
        if prostate_threshold is None:
            prostate_threshold = cfg_threshold
        prostate_threshold = float(prostate_threshold)
        if target_thresholds is None:
            target_thresholds = _DEFAULT_TARGET_SWEEP
        if default_target_threshold is None:
            best = _read_best_target_threshold(ckpt["run_dir"])
            default_target_threshold = best if best is not None else min(target_thresholds)

    device = resolve_device(cfg["inference"]["device"])
    dataloader = _build_dataloader(cfg, "segmentation", "test")

    model = create_segmentation_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
    _load_checkpoint(model, ckpt["ckpt_file"], device)

    summary = run_segmentation_inference(
        model=model,
        dataloader=dataloader,
        metadata_path=cfg["data"]["metadata"],
        output_dir=cfg["inference"]["output_dir"],
        device=device,
        threshold=prostate_threshold,
    )

    case_out = predictions_dir / case_id
    report_path = output_dir / "report.html"
    report_info = generate_html_report(
        case_output_dir=case_out,
        case_id=case_id,
        metadata_root=staging_dir,
        report_path=report_path,
        prostate_threshold=prostate_threshold,
        target_thresholds=target_thresholds,
        default_target_threshold=default_target_threshold,
        source_zip=zip_path,
        checkpoint_path=ckpt["ckpt_file"],
    )

    if open_browser:
        webbrowser.open(Path(report_info["report_path"]).resolve().as_uri())

    return {
        "case_id": case_id,
        "output_dir": str(output_dir),
        "staging_dir": str(staging_dir),
        "predictions_dir": str(predictions_dir),
        "overlays_dir": str(case_out / "overlays"),
        "html_report_path": report_info["report_path"],
        "prostate_slices": report_info["prostate_slices"],
        "target_slices": report_info["target_slices"],
        "target_slices_by_threshold": report_info["target_slices_by_threshold"],
        "target_pixels_by_threshold": report_info["target_pixels_by_threshold"],
        "num_slices": report_info["num_slices"],
        "prostate_threshold": report_info["prostate_threshold"],
        "target_thresholds": report_info["target_thresholds"],
        "default_target_threshold": report_info["default_target_threshold"],
        "checkpoint": str(ckpt["ckpt_file"]),
        "summary": summary,
    }


def _collect_zip_paths(zip_input: Union[str, Path, Iterable[Union[str, Path]]]) -> List[Path]:
    if isinstance(zip_input, (str, Path)):
        p = Path(zip_input)
        if p.is_dir():
            zips = sorted(p.glob("*.zip"))
            if not zips:
                raise FileNotFoundError(f"No .zip files found under {p}")
            return zips
        return [p]
    return [Path(z) for z in zip_input]


def _write_batch_index(
    output_dir: Path,
    results: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    checkpoint_path: Path,
) -> Path:
    def _fmt_range(indices: List[int]) -> str:
        if not indices:
            return "—"
        s = sorted(indices)
        groups, start, prev = [], s[0], s[0]
        for i in s[1:]:
            if i == prev + 1:
                prev = i
                continue
            groups.append((start, prev))
            start = prev = i
        groups.append((start, prev))
        return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in groups)

    rows = []
    for r in results:
        report_rel = Path(r["html_report_path"]).relative_to(output_dir)
        rows.append(
            "<tr>"
            f"<td><a href='{html_lib.escape(str(report_rel))}'>{html_lib.escape(r['case_id'])}</a></td>"
            f"<td>{r['num_slices']}</td>"
            f"<td>{len(r['prostate_slices'])} ({_fmt_range(r['prostate_slices'])})</td>"
            f"<td>{len(r['target_slices'])} ({_fmt_range(r['target_slices'])})</td>"
            f"<td>{r['prostate_threshold']:.2f}</td>"
            f"<td>{r['default_target_threshold']:.2f}</td>"
            "</tr>"
        )
    for e in errors:
        ident = e.get("zip") or e.get("case_id") or "<unknown>"
        rows.append(
            "<tr class='err'>"
            f"<td>{html_lib.escape(str(ident))}</td>"
            f"<td colspan='5'>ERROR: {html_lib.escape(e['error'])}</td>"
            "</tr>"
        )

    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #222; background: #f7f7f8; }
    h1 { font-size: 20px; margin: 0 0 8px; }
    .meta { font-size: 13px; color: #555; margin-bottom: 16px; }
    table { border-collapse: collapse; background: #fff; width: 100%; max-width: 1200px; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }
    th { background: #fafafa; color: #666; font-variant: all-small-caps; letter-spacing: .5px; }
    tr.err td { color: #c62828; font-family: ui-monospace, monospace; }
    a { color: #1565c0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    """
    body = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Batch segmentation report</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Batch segmentation report</h1>"
        f"<div class='meta'>checkpoint: <code>{html_lib.escape(str(checkpoint_path))}</code><br>"
        f"cases: {len(results)} OK, {len(errors)} failed<br>"
        f"generated: {datetime.now().isoformat(timespec='seconds')}</div>"
        "<table><thead><tr>"
        "<th>case</th><th>slices</th><th>prostate-positive</th><th>target-positive</th>"
        "<th>prostate thr.</th><th>target thr. (default)</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></body></html>"
    )

    index_path = output_dir / "index.html"
    index_path.write_text(body, encoding="utf-8")
    return index_path


def run_dicom_segmentation_batch(
    zip_paths: Union[str, Path, Iterable[Union[str, Path]]],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    resolved_config_path: Optional[Union[str, Path]] = None,
    prostate_threshold: Optional[float] = None,
    target_thresholds: Optional[Sequence[float]] = None,
    default_target_threshold: Optional[float] = None,
    threshold: Optional[float] = None,
    group_name: str = "case_auto",
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    """Run `run_dicom_segmentation` over multiple DICOM zips.

    Parameters
    ----------
    zip_paths : list/iterable of zip paths, a single zip path, or a directory containing *.zip.
    checkpoint_path, resolved_config_path, prostate_threshold, target_thresholds,
    default_target_threshold, threshold, group_name :
        same as :func:`run_dicom_segmentation`. The threshold kwargs are applied to every case.
    output_dir : each case writes into `output_dir/<case_stem>/`; an `index.html` linking to all
        per-case reports is written at the top level.
    continue_on_error : if True (default), a failed case is captured in `errors` and the batch
        keeps going. If False, the first exception propagates.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zips = _collect_zip_paths(zip_paths)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    ckpt = _resolve_checkpoint(checkpoint_path, resolved_config_path)

    for zip_path in zips:
        case_stem = zip_path.stem.lower().replace(" ", "_").replace(",", ".")
        case_output_dir = output_dir / case_stem
        try:
            r = run_dicom_segmentation(
                zip_path=zip_path,
                checkpoint_path=ckpt["ckpt_file"],
                output_dir=case_output_dir,
                resolved_config_path=ckpt["config_path"],
                prostate_threshold=prostate_threshold,
                target_thresholds=target_thresholds,
                default_target_threshold=default_target_threshold,
                threshold=threshold,
                group_name=group_name,
            )
            results.append(r)
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append({"zip": str(zip_path), "error": f"{type(exc).__name__}: {exc}"})

    index_path = _write_batch_index(output_dir, results, errors, ckpt["ckpt_file"])

    return {
        "output_dir": str(output_dir),
        "index_html": str(index_path),
        "num_ok": len(results),
        "num_failed": len(errors),
        "results": results,
        "errors": errors,
    }


def _resolve_aligned_case_ids(
    case_selector: Union[str, Iterable[str]],
    metadata_path: Path,
) -> List[str]:
    meta = json.loads(metadata_path.read_text())
    all_cases = list(meta.get("cases", {}).keys())
    if isinstance(case_selector, str):
        if case_selector in all_cases:
            return [case_selector]
        prefix = case_selector.rstrip("/") + "/"
        matches = [c for c in all_cases if c.startswith(prefix) or c == case_selector]
        if not matches:
            raise ValueError(
                f"No cases found for selector {case_selector!r} in {metadata_path}. "
                f"Known cases ({len(all_cases)}): {all_cases[:5]}..."
            )
        return matches
    selector_list = list(case_selector)
    missing = [c for c in selector_list if c not in all_cases]
    if missing:
        raise ValueError(f"Unknown case_ids in selector: {missing[:5]}")
    return selector_list


def run_aligned_segmentation(
    case_id: str,
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    metadata_path: Union[str, Path] = "data/aligned_v2/metadata.json",
    resolved_config_path: Optional[Union[str, Path]] = None,
    prostate_threshold: Optional[float] = None,
    target_thresholds: Optional[Sequence[float]] = None,
    default_target_threshold: Optional[float] = None,
    threshold: Optional[float] = None,
    open_browser: bool = False,
) -> Dict[str, Any]:
    """Run segmentation inference + HTML report on a single aligned_v2 case.

    Skips DICOM preprocessing — the case must already exist under ``metadata_path``'s
    directory with the usual `t2/`, `adc/`, `calc/`, `mask_prostate/`, `mask_target1/`
    layout. Ground-truth masks, when present, are drawn as coloured contours on the
    report overlays (green = prostate, cyan = target) with a toggle in the controls.
    """

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    metadata_root = metadata_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _resolve_checkpoint(checkpoint_path, resolved_config_path)

    split_path = output_dir / "split.yaml"
    split_path.write_text(yaml.safe_dump({"train": [], "val": [], "test": [case_id]}))

    cfg = load_config(ckpt["config_path"])
    cfg.setdefault("data", {})
    cfg["data"]["metadata"] = str(metadata_path)
    cfg["data"]["split_file"] = str(split_path)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["require_positive"] = False
    cfg["data"]["require_complete"] = False

    cfg.setdefault("inference", {})
    predictions_dir = output_dir / "predictions"
    cfg["inference"]["checkpoint"] = str(ckpt["ckpt_file"])
    cfg["inference"]["output_dir"] = str(predictions_dir)
    cfg["inference"]["device"] = cfg["inference"].get("device", "auto")
    cfg["inference"]["batch_size"] = cfg["inference"].get("batch_size", 4)

    cfg_threshold = float(cfg.get("metrics", {}).get("segmentation_threshold", 0.5))
    if threshold is not None:
        prostate_threshold = float(threshold)
        target_thresholds = (float(threshold),)
        default_target_threshold = float(threshold)
    else:
        if prostate_threshold is None:
            prostate_threshold = cfg_threshold
        prostate_threshold = float(prostate_threshold)
        if target_thresholds is None:
            target_thresholds = _DEFAULT_TARGET_SWEEP
        if default_target_threshold is None:
            best = _read_best_target_threshold(ckpt["run_dir"])
            default_target_threshold = best if best is not None else min(target_thresholds)

    device = resolve_device(cfg["inference"]["device"])
    dataloader = _build_dataloader(cfg, "segmentation", "test")

    model = create_segmentation_model(cfg["model"]["name"], **cfg["model"].get("params", {}))
    _load_checkpoint(model, ckpt["ckpt_file"], device)

    summary = run_segmentation_inference(
        model=model,
        dataloader=dataloader,
        metadata_path=cfg["data"]["metadata"],
        output_dir=cfg["inference"]["output_dir"],
        device=device,
        threshold=prostate_threshold,
    )

    case_out = predictions_dir / case_id
    report_path = output_dir / "report.html"
    report_info = generate_html_report(
        case_output_dir=case_out,
        case_id=case_id,
        metadata_root=metadata_root,
        report_path=report_path,
        prostate_threshold=prostate_threshold,
        target_thresholds=target_thresholds,
        default_target_threshold=default_target_threshold,
        checkpoint_path=ckpt["ckpt_file"],
    )

    if open_browser:
        webbrowser.open(Path(report_info["report_path"]).resolve().as_uri())

    return {
        "case_id": case_id,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "predictions_dir": str(predictions_dir),
        "overlays_dir": str(case_out / "overlays"),
        "html_report_path": report_info["report_path"],
        "prostate_slices": report_info["prostate_slices"],
        "target_slices": report_info["target_slices"],
        "target_slices_by_threshold": report_info["target_slices_by_threshold"],
        "num_slices": report_info["num_slices"],
        "prostate_threshold": report_info["prostate_threshold"],
        "target_thresholds": report_info["target_thresholds"],
        "default_target_threshold": report_info["default_target_threshold"],
        "checkpoint": str(ckpt["ckpt_file"]),
        "summary": summary,
    }


def run_aligned_segmentation_batch(
    case_selector: Union[str, Iterable[str]],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    metadata_path: Union[str, Path] = "data/aligned_v2/metadata.json",
    resolved_config_path: Optional[Union[str, Path]] = None,
    prostate_threshold: Optional[float] = None,
    target_thresholds: Optional[Sequence[float]] = None,
    default_target_threshold: Optional[float] = None,
    threshold: Optional[float] = None,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    """Run :func:`run_aligned_segmentation` over many aligned_v2 cases.

    ``case_selector`` accepts a case_id string, a case_id prefix (e.g. ``"class4"`` →
    every case whose id starts with ``"class4/"``), or an iterable of case_ids.
    """

    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _resolve_aligned_case_ids(case_selector, metadata_path)
    ckpt = _resolve_checkpoint(checkpoint_path, resolved_config_path)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for case_id in case_ids:
        case_stem = case_id.replace("/", "_")
        case_output_dir = output_dir / case_stem
        try:
            r = run_aligned_segmentation(
                case_id=case_id,
                checkpoint_path=ckpt["ckpt_file"],
                output_dir=case_output_dir,
                metadata_path=metadata_path,
                resolved_config_path=ckpt["config_path"],
                prostate_threshold=prostate_threshold,
                target_thresholds=target_thresholds,
                default_target_threshold=default_target_threshold,
                threshold=threshold,
            )
            results.append(r)
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})

    index_path = _write_batch_index(output_dir, results, errors, ckpt["ckpt_file"])

    return {
        "output_dir": str(output_dir),
        "index_html": str(index_path),
        "num_ok": len(results),
        "num_failed": len(errors),
        "results": results,
        "errors": errors,
    }
