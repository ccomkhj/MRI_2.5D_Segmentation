"""Gradio web UI for DICOM zip → metadata preview → segmentation → HTML report.

Launch with::

    python -m mri.service.ui

Environment variables:

- ``MRI_DEFAULT_CHECKPOINT`` — path to the run directory (or ``.pt`` file) the UI
  pre-fills into the "Advanced" checkpoint field. Falls back to a bundled default.
- ``MRI_UI_OUTPUT_ROOT`` — directory where per-run outputs are written. Defaults to
  ``<repo>/runs/ui/``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr
import pydicom

from mri.service.pipeline import run_dicom_segmentation, _import_dicom_mapper


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUTO_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "default"
_CHECKPOINT_URL = os.environ.get(
    "MRI_CHECKPOINT_URL",
    "https://github.com/ccomkhj/cancer_detector/releases/latest/download/default-checkpoint.zip",
)
_DEFAULT_CHECKPOINT = os.environ.get("MRI_DEFAULT_CHECKPOINT", str(_AUTO_CHECKPOINT_DIR))
_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("MRI_UI_OUTPUT_ROOT", str(_REPO_ROOT / "runs" / "ui"))
)


def _ensure_default_checkpoint() -> None:
    """Download and extract the default checkpoint on first launch.

    Noop if ``_AUTO_CHECKPOINT_DIR`` already contains a ``*_best.pt`` file, or if
    the user has overridden the default via ``MRI_DEFAULT_CHECKPOINT``.
    """
    if os.environ.get("MRI_DEFAULT_CHECKPOINT"):
        return
    if _AUTO_CHECKPOINT_DIR.exists() and any(_AUTO_CHECKPOINT_DIR.glob("*_best.pt")):
        return

    print(f"[ui] downloading default checkpoint from {_CHECKPOINT_URL}")
    _AUTO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = Path(tempfile.mkstemp(prefix="mri_ckpt_", suffix=".zip")[1])
    try:
        urllib.request.urlretrieve(_CHECKPOINT_URL, tmp_zip)
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(_AUTO_CHECKPOINT_DIR)
    finally:
        tmp_zip.unlink(missing_ok=True)

    # If the release zip wrapped everything in a single top-level folder, flatten it.
    entries = list(_AUTO_CHECKPOINT_DIR.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(_AUTO_CHECKPOINT_DIR / item.name))
        inner.rmdir()

    best = list(_AUTO_CHECKPOINT_DIR.glob("*_best.pt"))
    cfg = _AUTO_CHECKPOINT_DIR / "resolved_config.yaml"
    if not best or not cfg.exists():
        raise RuntimeError(
            f"Downloaded zip did not contain the expected files "
            f"(*_best.pt + resolved_config.yaml) under {_AUTO_CHECKPOINT_DIR}"
        )
    print(f"[ui] default checkpoint ready at {_AUTO_CHECKPOINT_DIR}")

_SERIES_TABLE_HEADERS = [
    "Series#",
    "Role",
    "Image type",
    "Slices",
    "Matrix",
    "TE (ms)",
    "TR (ms)",
]


def _read_patient_info(sample_dcm: Path) -> Dict[str, str]:
    """Read a small set of patient/study fields from one DICOM file."""
    try:
        ds = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read DICOM header: {exc}"}
    return {
        "Patient ID": str(getattr(ds, "PatientID", "—")),
        "Sex": str(getattr(ds, "PatientSex", "—")),
        "Age": str(getattr(ds, "PatientAge", "—")),
        "Study date": str(getattr(ds, "StudyDate", "—")),
        "Study description": str(getattr(ds, "StudyDescription", "—")),
        "Manufacturer": str(getattr(ds, "Manufacturer", "—")),
        "Model": str(getattr(ds, "ManufacturerModelName", "—")),
        "Field strength (T)": str(getattr(ds, "MagneticFieldStrength", "—")),
    }


def _patient_info_markdown(info: Dict[str, str]) -> str:
    if "error" in info:
        return f"> {info['error']}"
    rows = [f"- **{k}:** {v}" for k, v in info.items()]
    return "### Patient & study\n" + "\n".join(rows)


def _preview_metadata(
    zip_file: Optional[str],
) -> Tuple[Any, Any, Any, Any, Any]:
    """Extract zip, parse DICOMDIR, return UI updates for the preview section.

    Returns a 5-tuple matching outputs: (patient_md, series_table, classification_md,
    confirm_row_update, state).
    """
    if not zip_file:
        return (
            gr.update(value="Please upload a DICOM zip first.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )

    zip_path = Path(zip_file)

    try:
        _import_dicom_mapper()
        from dicom_mapper.io.vendor import (
            classify_series,
            extract_zip,
            find_dicomdir,
            get_series_info,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(value=f"Could not load dicom-mapper: `{exc}`", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )

    preview_dir = Path(tempfile.mkdtemp(prefix="mri_preview_"))
    try:
        extract_zip(zip_path, preview_dir)
        dicomdir = find_dicomdir(preview_dir)
        if dicomdir is None:
            return (
                gr.update(
                    value="No `DICOMDIR` found in the zip. Is this a vendor DICOM export?",
                    visible=True,
                ),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
            )
        series_list = get_series_info(dicomdir)
        classification = classify_series(series_list)
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(value=f"Could not parse DICOM zip: `{exc}`", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )

    role_by_uid: Dict[str, str] = {}
    for role, s in classification.items():
        if s:
            role_by_uid[s["uid"]] = role.upper()

    rows: List[List[Any]] = []
    for s in sorted(series_list, key=lambda x: x.get("series_number") or 0):
        role = role_by_uid.get(s["uid"], "—")
        rows.append(
            [
                s.get("series_number", "—"),
                role,
                " / ".join(s.get("image_type", [])) or "—",
                s.get("num_images", 0),
                f"{s.get('rows', 0)}×{s.get('columns', 0)}",
                f"{s.get('echo_time', 0):.0f}",
                f"{s.get('repetition_time', 0):.0f}",
            ]
        )

    patient_info = {}
    for s in series_list:
        dcm_dir = Path(s.get("dicom_dir", ""))
        if dcm_dir.exists():
            dcms = list(dcm_dir.glob("*"))
            if dcms:
                patient_info = _read_patient_info(dcms[0])
                break

    classification_bits = []
    for role in ("t2", "adc", "calc"):
        s = classification.get(role)
        if s:
            classification_bits.append(
                f"- **{role.upper()}** — Series #{s['series_number']} "
                f"({s['num_images']} slices, {s['rows']}×{s['columns']})"
            )
        else:
            classification_bits.append(f"- **{role.upper()}** — *not detected*")

    missing_required = [r for r in ("t2", "adc") if classification.get(r) is None]
    warning = (
        f"\n\n⚠️ **Missing required modality:** {', '.join(m.upper() for m in missing_required)}. "
        "Segmentation will likely fail — check the uploaded series."
        if missing_required
        else ""
    )

    classification_md = (
        "### Detected modalities\n" + "\n".join(classification_bits) + warning
    )

    state = {
        "zip_path": str(zip_path),
        "num_series": len(series_list),
        "has_missing": bool(missing_required),
    }

    return (
        gr.update(value=_patient_info_markdown(patient_info), visible=True),
        gr.update(value=rows, visible=True),
        gr.update(value=classification_md, visible=True),
        gr.update(visible=True),
        state,
    )


def _run_segmentation(
    state: Optional[Dict[str, Any]],
    checkpoint_path: str,
) -> Iterator[Tuple[str, Optional[str]]]:
    """Run the pipeline once the clinician has reviewed the metadata preview."""
    if not state or not state.get("zip_path"):
        yield "No preview available. Upload a zip and click *Analyze* first.", None
        return

    zip_path = Path(state["zip_path"])
    ckpt = Path((checkpoint_path or _DEFAULT_CHECKPOINT).strip()).expanduser()
    if not ckpt.exists():
        yield f"Checkpoint not found: `{ckpt}`", None
        return

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = _DEFAULT_OUTPUT_ROOT / f"{zip_path.stem}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    yield (
        f"Running segmentation on **{zip_path.name}**…\n\n"
        f"Output directory: `{output_dir}`\n\n"
        "This takes 1–3 minutes on a laptop. The report will open in a new tab "
        "when finished.",
        None,
    )

    try:
        result = run_dicom_segmentation(
            zip_path=zip_path,
            checkpoint_path=ckpt,
            output_dir=output_dir,
            open_browser=True,
        )
    except Exception as exc:  # noqa: BLE001
        yield f"Pipeline failed: `{type(exc).__name__}: {exc}`", None
        return

    report_path = result["html_report_path"]
    n_slices = result["num_slices"]
    n_prostate = len(result["prostate_slices"])
    n_target = len(result["target_slices"])
    prostate_thr = result["prostate_threshold"]
    target_thr = result["default_target_threshold"]

    msg = (
        f"**Done** — case `{result['case_id']}`\n\n"
        f"- Slices analyzed: **{n_slices}**\n"
        f"- Prostate-positive slices: **{n_prostate}** (threshold {prostate_thr:.2f})\n"
        f"- Target-positive slices: **{n_target}** (threshold {target_thr:.2f})\n\n"
        f"Report should have opened in a new tab. File: `{report_path}`"
    )
    yield msg, report_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Prostate MRI Segmentation") as app:
        gr.Markdown(
            "# Prostate MRI Segmentation\n"
            "1. Upload a vendor DICOM **.zip** and click *Analyze*.\n"
            "2. Review the detected series.\n"
            "3. Click *Run segmentation* to generate the annotated report."
        )
        zip_input = gr.File(
            label="DICOM zip",
            file_types=[".zip"],
            type="filepath",
        )
        with gr.Accordion("Advanced", open=False):
            ckpt_input = gr.Textbox(
                label="Checkpoint (run directory or .pt file)",
                value=_DEFAULT_CHECKPOINT,
                lines=1,
            )
        analyze_btn = gr.Button("Analyze", variant="primary")

        patient_md = gr.Markdown(visible=False)
        series_table = gr.Dataframe(
            headers=_SERIES_TABLE_HEADERS,
            interactive=False,
            visible=False,
            label="Series",
            wrap=True,
        )
        classification_md = gr.Markdown(visible=False)

        with gr.Row(visible=False) as confirm_row:
            run_btn = gr.Button("Run segmentation", variant="primary")

        status = gr.Markdown()
        report_file = gr.File(label="Report (download)")

        preview_state = gr.State()

        analyze_btn.click(
            fn=_preview_metadata,
            inputs=[zip_input],
            outputs=[
                patient_md,
                series_table,
                classification_md,
                confirm_row,
                preview_state,
            ],
        )
        run_btn.click(
            fn=_run_segmentation,
            inputs=[preview_state, ckpt_input],
            outputs=[status, report_file],
        )
    return app


def main() -> None:
    _DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_default_checkpoint()
    except Exception as exc:  # noqa: BLE001 - UI should still launch without the download
        print(
            f"[ui] WARNING: could not fetch default checkpoint ({exc}). "
            "You can still analyze cases by pasting a checkpoint path into the Advanced field."
        )
    app = build_app()
    app.launch(
        inbrowser=True,
        allowed_paths=[str(_DEFAULT_OUTPUT_ROOT)],
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
