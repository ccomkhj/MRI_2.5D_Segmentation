from __future__ import annotations

import csv
import json
from pathlib import Path

from mri.experiments.latest_jobs_report import (
    extract_report_row,
    generate_latest_jobs_report,
    load_training_run_manifests,
    render_latest_jobs_html,
    select_latest_job_rows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(
    run_dir: Path,
    *,
    run_name: str,
    task: str = "classification",
    status: str = "completed",
    created_at: str = "2026-03-29T10:00:00+00:00",
    finished_at: str | None = "2026-03-29T11:00:00+00:00",
    summary: dict | None = None,
    history_csv: str | None = None,
    run_type: str = "train",
    job_id: str | None = "123",
) -> dict:
    return {
        "run_name": run_name,
        "run_type": run_type,
        "task": task,
        "status": status,
        "created_at": created_at,
        "finished_at": finished_at,
        "config": {"source_path": str(run_dir / "config.yaml")},
        "artifacts": {
            "run_dir": str(run_dir),
            "history_csv": history_csv,
            "summary_json": str(run_dir / "run_summary.json"),
            "best_checkpoint": str(run_dir / f"{run_name}_best.pt"),
        },
        "environment": {"slurm": {"job_id": job_id}},
        "summary": summary,
    }


def test_select_latest_job_rows_sorts_by_best_precision_target_then_time(tmp_path: Path):
    root = tmp_path / "checkpoints"
    low_dir = root / "low"
    high_dir = root / "high"
    no_metric_old_dir = root / "no-metric-old"
    no_metric_new_dir = root / "no-metric-new"

    def segmentation_summary(best_precision_target: float) -> dict:
        return {
            "primary_metric_name": "precision_target",
            "best_metric": best_precision_target,
            "best_epoch": 3,
            "best_val_metrics": {
                "precision_target": best_precision_target,
                "dice_target": best_precision_target / 2,
                "recall_target": best_precision_target / 3,
            },
            "final_val_metrics": {
                "precision_target": best_precision_target - 0.01,
                "dice_target": best_precision_target / 2,
                "recall_target": best_precision_target / 3,
            },
        }

    classification_summary = {
        "primary_metric_name": "macro_f1",
        "best_metric": 0.4,
        "best_epoch": 3,
        "best_val_metrics": {"acc": 0.6, "macro_f1": 0.4},
        "final_val_metrics": {"acc": 0.55, "macro_f1": 0.35},
    }

    _write_json(
        low_dir / "run_manifest.json",
        _manifest(
            low_dir,
            run_name="low",
            task="segmentation",
            created_at="2026-03-29T10:00:00+00:00",
            finished_at="2026-03-29T11:00:00+00:00",
            summary=segmentation_summary(0.22),
        ),
    )
    _write_json(
        high_dir / "run_manifest.json",
        _manifest(
            high_dir,
            run_name="high",
            task="segmentation",
            created_at="2026-03-29T09:00:00+00:00",
            finished_at="2026-03-29T10:00:00+00:00",
            summary=segmentation_summary(0.40),
        ),
    )
    _write_json(
        no_metric_old_dir / "run_manifest.json",
        _manifest(
            no_metric_old_dir,
            run_name="no-metric-old",
            created_at="2026-03-29T12:00:00+00:00",
            finished_at="2026-03-29T12:30:00+00:00",
            summary=classification_summary,
        ),
    )
    _write_json(
        no_metric_new_dir / "run_manifest.json",
        _manifest(
            no_metric_new_dir,
            run_name="no-metric-new",
            created_at="2026-03-29T12:45:00+00:00",
            finished_at="2026-03-29T13:00:00+00:00",
            summary=classification_summary,
        ),
    )

    manifests = load_training_run_manifests(root)
    rows, skipped_count = select_latest_job_rows(manifests, latest_n=4)

    assert skipped_count == 0
    assert [row["run_name"] for row in rows] == ["high", "low", "no-metric-new", "no-metric-old"]


def test_filters_inference_and_metricless_runs(tmp_path: Path):
    root = tmp_path / "checkpoints"
    valid_dir = root / "valid-train"
    metricless_dir = root / "metricless"
    inference_dir = root / "inference"

    _write_json(
        valid_dir / "run_manifest.json",
        _manifest(
            valid_dir,
            run_name="valid-train",
            summary={
                "primary_metric_name": "macro_f1",
                "best_metric": 0.3,
                "best_epoch": 2,
                "best_val_metrics": {"acc": 0.5, "macro_f1": 0.3},
                "final_val_metrics": {"acc": 0.45, "macro_f1": 0.25},
            },
        ),
    )
    _write_json(
        metricless_dir / "run_manifest.json",
        _manifest(
            metricless_dir,
            run_name="metricless",
            summary={
                "primary_metric_name": "macro_f1",
                "best_metric": 0.3,
                "best_epoch": 2,
                "best_val_metrics": {"loss": 1.2},
                "final_val_metrics": {"loss": 1.0},
            },
        ),
    )
    _write_json(
        inference_dir / "sample_inference_manifest.json",
        _manifest(
            inference_dir,
            run_name="inference-job",
            run_type="inference",
            summary={
                "primary_metric_name": "macro_f1",
                "best_metric": 0.3,
                "best_epoch": 2,
                "best_val_metrics": {"acc": 0.5, "macro_f1": 0.3},
                "final_val_metrics": {"acc": 0.45, "macro_f1": 0.25},
            },
        ),
    )

    manifests = load_training_run_manifests(root)
    rows, skipped_count = select_latest_job_rows(manifests, latest_n=10)

    assert [manifest["run_name"] for manifest in manifests] == ["metricless", "valid-train"]
    assert [row["run_name"] for row in rows] == ["valid-train"]
    assert skipped_count == 1


def test_extract_report_row_uses_csv_fallback(tmp_path: Path):
    run_dir = tmp_path / "checkpoints" / "csv-fallback"
    history_path = run_dir / "metrics_history.csv"
    _write_csv(
        history_path,
        [
            {
                "epoch": 1,
                "primary_metric": 0.22,
                "val/acc": 0.50,
                "val/macro_f1": 0.22,
            },
            {
                "epoch": 2,
                "primary_metric": 0.37,
                "val/acc": 0.63,
                "val/macro_f1": 0.37,
            },
            {
                "epoch": 3,
                "primary_metric": 0.31,
                "val/acc": 0.58,
                "val/macro_f1": 0.31,
            },
        ],
    )

    manifest = _manifest(
        run_dir,
        run_name="csv-fallback",
        summary=None,
        history_csv=str(history_path),
    )

    row = extract_report_row(manifest)

    assert row is not None
    assert row["summary_source"] == "csv"
    assert row["best_epoch"] == 2
    assert row["best_metric"] == 0.37
    assert row["best_acc"] == 0.63
    assert row["final_acc"] == 0.58
    assert row["best_macro_f1"] == 0.37
    assert row["final_macro_f1"] == 0.31


def test_rendered_html_contains_task_specific_metrics_and_links(tmp_path: Path):
    root = tmp_path / "checkpoints"
    cls_dir = root / "cls-run"
    seg_dir = root / "seg-run"

    classification_row = extract_report_row(
        _manifest(
            cls_dir,
            run_name="classification-run",
            task="classification",
            summary={
                "primary_metric_name": "macro_f1",
                "best_metric": 0.44,
                "best_epoch": 4,
                "best_val_metrics": {"acc": 0.71, "macro_f1": 0.44},
                "final_val_metrics": {"acc": 0.69, "macro_f1": 0.41},
            },
        )
    )
    segmentation_row = extract_report_row(
        _manifest(
            seg_dir,
            run_name="segmentation-run",
            task="segmentation",
            summary={
                "primary_metric_name": "precision_target",
                "best_metric": 0.22,
                "best_epoch": 7,
                "best_val_metrics": {
                    "precision_target": 0.22,
                    "dice_target": 0.15,
                    "recall_target": 0.11,
                },
                "final_val_metrics": {
                    "precision_target": 0.20,
                    "dice_target": 0.13,
                    "recall_target": 0.10,
                },
            },
        )
    )

    assert classification_row is not None
    assert segmentation_row is not None

    html = render_latest_jobs_html(
        [classification_row, segmentation_row],
        root=root,
        latest_n=10,
        skipped_count=2,
        generated_at="2026-04-03T08:00:00+00:00",
    )

    assert "Included jobs" in html
    assert "Skipped jobs" in html
    assert "classification-run" in html
    assert "segmentation-run" in html
    assert "Top n: 10 | Sort rule: best val/precision_target descending, fallback to finished_at then created_at" in html
    assert "Best val/precision_target" in html
    assert "Best val/acc" in html
    assert "Best val/macro_f1" in html
    assert "Best val/dice_target" in html
    assert "Best val/recall_target" in html
    assert "file://" in html


def test_generate_latest_jobs_report_writes_html(tmp_path: Path):
    root = tmp_path / "checkpoints"
    run_dir = root / "classification-run"
    _write_json(
        run_dir / "run_manifest.json",
        _manifest(
            run_dir,
            run_name="classification-run",
            summary={
                "primary_metric_name": "macro_f1",
                "best_metric": 0.44,
                "best_epoch": 4,
                "best_val_metrics": {"acc": 0.71, "macro_f1": 0.44},
                "final_val_metrics": {"acc": 0.69, "macro_f1": 0.41},
            },
        ),
    )

    output_path = generate_latest_jobs_report(root, latest_n=10)

    assert output_path == root / "reports" / "latest_jobs.html"
    assert output_path.exists()
    html = output_path.read_text()
    assert "Latest Training Jobs Report" in html
    assert "classification-run" in html
    assert "run dir" in html
