"""Static HTML reporting for the latest training jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Sequence
import csv

from mri.experiments.runtime import load_run_manifests, utc_now_iso


TASK_METRICS: dict[str, tuple[str, ...]] = {
    "classification": ("acc", "macro_f1"),
    "segmentation": ("precision_target", "dice_target", "recall_target"),
}


def load_training_run_manifests(root: Path) -> list[dict[str, Any]]:
    """Load training manifests under a report root."""
    return [manifest for manifest in load_run_manifests(root) if manifest.get("run_type") == "train"]


def _parse_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    number = _parse_float(value)
    if number is None:
        return None
    return int(number)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _extract_validation_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if not key.startswith("val/"):
            continue
        parsed = _parse_float(value)
        if parsed is not None:
            metrics[key.removeprefix("val/")] = parsed
    return metrics


def _guess_primary_metric_name(task: str | None, row: dict[str, Any]) -> str:
    fieldnames = set(row)
    if task == "classification" and "val/macro_f1" in fieldnames:
        return "macro_f1"
    if task == "segmentation" and "val/precision_target" in fieldnames:
        return "precision_target"
    if "val/dice" in fieldnames:
        return "dice"
    if "val/acc" in fieldnames:
        return "acc"
    return "primary_metric"


def _summary_from_history_csv(history_csv: Path, task: str | None) -> dict[str, Any] | None:
    if not history_csv.exists():
        return None

    with history_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    valid_rows: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        primary_metric = _parse_float(row.get("primary_metric"))
        if primary_metric is not None:
            valid_rows.append((primary_metric, row))

    if not valid_rows:
        return None

    best_metric, best_row = max(valid_rows, key=lambda item: item[0])
    final_row = rows[-1]

    return {
        "best_epoch": _parse_int(best_row.get("epoch")),
        "best_metric": best_metric,
        "best_val_metrics": _extract_validation_metrics(best_row),
        "final_val_metrics": _extract_validation_metrics(final_row),
        "primary_metric_name": _guess_primary_metric_name(task, best_row),
    }


def _has_usable_metrics(task: str | None, best_val_metrics: dict[str, Any], final_val_metrics: dict[str, Any]) -> bool:
    metric_names = TASK_METRICS.get(task or "")
    if metric_names is None:
        return bool(best_val_metrics or final_val_metrics)
    return any(
        _parse_float(best_val_metrics.get(name)) is not None or _parse_float(final_val_metrics.get(name)) is not None
        for name in metric_names
    )


def extract_report_row(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Extract one task-aware report row from a training manifest."""
    summary = manifest.get("summary")
    summary_source = "manifest"
    artifacts = manifest.get("artifacts", {})
    task = manifest.get("task")

    if not summary:
        history_csv = artifacts.get("history_csv")
        if history_csv:
            summary = _summary_from_history_csv(Path(history_csv), task)
        summary_source = "csv"

    if not summary:
        return None

    best_val_metrics = summary.get("best_val_metrics", {}) or {}
    final_val_metrics = summary.get("final_val_metrics", {}) or {}
    if not _has_usable_metrics(task, best_val_metrics, final_val_metrics):
        return None

    environment = manifest.get("environment", {})
    slurm = environment.get("slurm", {})
    sort_timestamp = manifest.get("finished_at") or manifest.get("created_at")

    row = {
        "run_name": manifest.get("run_name"),
        "run_type": manifest.get("run_type"),
        "task": task,
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "finished_at": manifest.get("finished_at"),
        "sort_timestamp": sort_timestamp,
        "slurm_job_id": slurm.get("job_id"),
        "primary_metric_name": summary.get("primary_metric_name"),
        "best_metric": _parse_float(summary.get("best_metric")),
        "best_epoch": _parse_int(summary.get("best_epoch")),
        "config_path": manifest.get("config", {}).get("source_path"),
        "run_dir": artifacts.get("run_dir"),
        "history_csv": artifacts.get("history_csv"),
        "summary_json": artifacts.get("summary_json"),
        "best_checkpoint": artifacts.get("best_checkpoint"),
        "best_val_metrics": best_val_metrics,
        "final_val_metrics": final_val_metrics,
        "summary_source": summary_source,
    }

    for metric_name in ("acc", "macro_f1", "precision_target", "dice_target", "recall_target"):
        row[f"best_{metric_name}"] = _parse_float(best_val_metrics.get(metric_name))
        row[f"final_{metric_name}"] = _parse_float(final_val_metrics.get(metric_name))
    return row


def _timestamp_sort_value(value: Any) -> float:
    timestamp = _parse_timestamp(value)
    if timestamp is None:
        return float("-inf")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.timestamp()


def _row_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    return _timestamp_sort_value(row.get("sort_timestamp")), str(row.get("run_name") or "")


def select_latest_job_rows(manifests: Sequence[dict[str, Any]], latest_n: int) -> tuple[list[dict[str, Any]], int]:
    """Extract, filter, and sort report rows by best val/precision_target."""
    rows: list[dict[str, Any]] = []
    skipped_count = 0

    for manifest in manifests:
        row = extract_report_row(manifest)
        if row is None:
            skipped_count += 1
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("best_precision_target") is not None,
            row.get("best_precision_target") if row.get("best_precision_target") is not None else float("-inf"),
            *_row_sort_key(row),
        ),
        reverse=True,
    )
    return rows[:latest_n], skipped_count


def _format_metric(value: float | None) -> str:
    if value is None:
        return '<span class="muted">&mdash;</span>'
    return f"{value:.4f}"


def _format_text(value: Any) -> str:
    if value is None or value == "":
        return '<span class="muted">&mdash;</span>'
    return escape(str(value))


def _path_href(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = path.resolve()
    try:
        return path.as_uri()
    except ValueError:
        return None


def _render_link(label: str, path_value: str | None) -> str:
    href = _path_href(path_value)
    if href is None:
        return f'<span class="muted">{escape(label)}</span>'
    return f'<a href="{escape(href)}">{escape(label)}</a>'


def _render_link_group(row: dict[str, Any]) -> str:
    links = [
        _render_link("run dir", row.get("run_dir")),
        _render_link("history", row.get("history_csv")),
        _render_link("summary", row.get("summary_json")),
        _render_link("checkpoint", row.get("best_checkpoint")),
    ]
    return '<div class="link-group">' + "".join(f"<span>{link}</span>" for link in links) + "</div>"


def _render_table(headers: Sequence[str], rows: Sequence[str], empty_message: str, colspan: int) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    if rows:
        body_html = "".join(rows)
    else:
        body_html = (
            f'<tr><td colspan="{colspan}" class="empty-row">{escape(empty_message)}</td></tr>'
        )
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def _overview_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            "<tr>"
            f"<td>{_render_link(str(row.get('run_name') or 'run'), row.get('run_dir'))}</td>"
            f"<td>{_format_text(row.get('task'))}</td>"
            f"<td>{_format_text(row.get('status'))}</td>"
            f"<td>{_format_text(row.get('slurm_job_id'))}</td>"
            f"<td>{_format_text(row.get('finished_at'))}</td>"
            f"<td>{_format_text(row.get('created_at'))}</td>"
            f"<td>{_format_text(row.get('primary_metric_name'))}</td>"
            f"<td>{_format_metric(row.get('best_metric'))}</td>"
            f"<td>{_format_text(row.get('best_epoch'))}</td>"
            f"<td>{_format_metric(row.get('best_precision_target'))}</td>"
            f"<td>{_render_link('config', row.get('config_path'))}</td>"
            f"<td>{_render_link_group(row)}</td>"
            "</tr>"
        )
    return rendered_rows


def _classification_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    rendered_rows = []
    for row in rows:
        if row.get("task") != "classification":
            continue
        rendered_rows.append(
            "<tr>"
            f"<td>{_render_link(str(row.get('run_name') or 'run'), row.get('run_dir'))}</td>"
            f"<td>{_format_text(row.get('status'))}</td>"
            f"<td>{_format_text(row.get('finished_at'))}</td>"
            f"<td>{_format_text(row.get('best_epoch'))}</td>"
            f"<td>{_format_metric(row.get('best_acc'))}</td>"
            f"<td>{_format_metric(row.get('final_acc'))}</td>"
            f"<td>{_format_metric(row.get('best_macro_f1'))}</td>"
            f"<td>{_format_metric(row.get('final_macro_f1'))}</td>"
            "</tr>"
        )
    return rendered_rows


def _segmentation_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    rendered_rows = []
    for row in rows:
        if row.get("task") != "segmentation":
            continue
        rendered_rows.append(
            "<tr>"
            f"<td>{_render_link(str(row.get('run_name') or 'run'), row.get('run_dir'))}</td>"
            f"<td>{_format_text(row.get('status'))}</td>"
            f"<td>{_format_text(row.get('finished_at'))}</td>"
            f"<td>{_format_text(row.get('best_epoch'))}</td>"
            f"<td>{_format_metric(row.get('best_precision_target'))}</td>"
            f"<td>{_format_metric(row.get('final_precision_target'))}</td>"
            f"<td>{_format_metric(row.get('best_dice_target'))}</td>"
            f"<td>{_format_metric(row.get('final_dice_target'))}</td>"
            f"<td>{_format_metric(row.get('best_recall_target'))}</td>"
            f"<td>{_format_metric(row.get('final_recall_target'))}</td>"
            "</tr>"
        )
    return rendered_rows


def _stat_card(label: str, value: int) -> str:
    return (
        '<div class="stat-card">'
        f'<div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{value}</div>'
        "</div>"
    )


def render_latest_jobs_html(
    rows: Sequence[dict[str, Any]],
    *,
    root: Path,
    latest_n: int,
    skipped_count: int,
    generated_at: str | None = None,
) -> str:
    """Render a self-contained HTML report."""
    generated_at = generated_at or utc_now_iso()
    classification_count = sum(1 for row in rows if row.get("task") == "classification")
    segmentation_count = sum(1 for row in rows if row.get("task") == "segmentation")
    completed_count = sum(1 for row in rows if row.get("status") == "completed")
    running_count = sum(1 for row in rows if row.get("status") == "running")

    stats_html = "".join(
        [
            _stat_card("Included jobs", len(rows)),
            _stat_card("Skipped jobs", skipped_count),
            _stat_card("Completed jobs", completed_count),
            _stat_card("Running jobs", running_count),
            _stat_card("Classification jobs", classification_count),
            _stat_card("Segmentation jobs", segmentation_count),
        ]
    )

    overview_table = _render_table(
        headers=(
            "Run",
            "Task",
            "Status",
            "SLURM job",
            "Finished at",
            "Created at",
            "Primary metric",
            "Best metric",
            "Best epoch",
            "Best val/precision_target",
            "Config",
            "Artifacts",
        ),
        rows=_overview_rows(rows),
        empty_message="No training jobs with usable validation metrics were found.",
        colspan=12,
    )
    classification_table = _render_table(
        headers=(
            "Run",
            "Status",
            "Finished at",
            "Best epoch",
            "Best val/acc",
            "Final val/acc",
            "Best val/macro_f1",
            "Final val/macro_f1",
        ),
        rows=_classification_rows(rows),
        empty_message="No classification jobs matched the latest selection.",
        colspan=8,
    )
    segmentation_table = _render_table(
        headers=(
            "Run",
            "Status",
            "Finished at",
            "Best epoch",
            "Best val/precision_target",
            "Final val/precision_target",
            "Best val/dice_target",
            "Final val/dice_target",
            "Best val/recall_target",
            "Final val/recall_target",
        ),
        rows=_segmentation_rows(rows),
        empty_message="No segmentation jobs matched the latest selection.",
        colspan=10,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Latest Training Jobs Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #edf3ef;
      --panel: #ffffff;
      --panel-alt: #f7faf8;
      --text: #173228;
      --muted: #5f7a6e;
      --accent: #1d6b50;
      --accent-soft: #d9ece3;
      --border: #cfe0d8;
      --shadow: 0 14px 40px rgba(18, 63, 47, 0.12);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(29, 107, 80, 0.18), transparent 34%),
        linear-gradient(180deg, #f4f8f5 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.5;
    }}

    .container {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}

    .hero {{
      background: linear-gradient(135deg, #184f3b 0%, #1d6b50 55%, #2a8b67 100%);
      color: #ffffff;
      border-radius: 24px;
      padding: 32px;
      box-shadow: var(--shadow);
      margin-bottom: 28px;
    }}

    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2.2rem;
      line-height: 1.15;
    }}

    .hero p {{
      margin: 6px 0;
      color: rgba(255, 255, 255, 0.9);
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}

    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }}

    .stat-label {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 8px;
    }}

    .stat-value {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--accent);
    }}

    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}

    .section h2 {{
      margin: 0 0 16px;
      font-size: 1.35rem;
    }}

    .section p {{
      margin: 0 0 16px;
      color: var(--muted);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 16px;
      background: var(--panel-alt);
    }}

    th, td {{
      text-align: left;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}

    th {{
      background: var(--accent-soft);
      color: var(--text);
      font-size: 0.92rem;
    }}

    tbody tr:hover {{
      background: #f0f7f3;
    }}

    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    .muted {{
      color: var(--muted);
    }}

    .link-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .empty-row {{
      text-align: center;
      color: var(--muted);
      background: #fbfdfc;
    }}

    .footer {{
      color: var(--muted);
      font-size: 0.92rem;
      text-align: center;
      padding: 10px 0 30px;
    }}

    @media (max-width: 900px) {{
      .container {{
        padding: 18px;
      }}

      .hero, .section {{
        padding: 20px;
        border-radius: 18px;
      }}

      table, thead, tbody, tr, th, td {{
        display: block;
      }}

      thead {{
        display: none;
      }}

      tr {{
        border-bottom: 1px solid var(--border);
        padding: 8px 0;
      }}

      td {{
        padding: 8px 0;
        border-bottom: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Latest Training Jobs Report</h1>
      <p>Generated at: {escape(generated_at)}</p>
      <p>Report root: {escape(str(root.resolve()))}</p>
      <p>Top n: {latest_n} | Sort rule: best val/precision_target descending, fallback to finished_at then created_at</p>
    </section>

    <section class="stats-grid">
      {stats_html}
    </section>

    <section class="section">
      <h2>Ranked Jobs</h2>
      <p>Overview of the selected training jobs ranked by best val/precision_target.</p>
      {overview_table}
    </section>

    <section class="section">
      <h2>Classification Metrics</h2>
      <p>Validation accuracy and macro-F1 for included classification jobs after precision_target ranking.</p>
      {classification_table}
    </section>

    <section class="section">
      <h2>Segmentation Metrics</h2>
      <p>Validation precision_target, dice_target, and recall_target for included segmentation jobs.</p>
      {segmentation_table}
    </section>

    <div class="footer">Static offline HTML report generated from local run manifests.</div>
  </div>
</body>
</html>
"""


def generate_latest_jobs_report(
    root: Path,
    *,
    output_path: Path | None = None,
    latest_n: int = 10,
) -> Path:
    """Generate the latest-jobs HTML report and return the output path."""
    manifests = load_training_run_manifests(root)
    rows, skipped_count = select_latest_job_rows(manifests, latest_n=latest_n)
    html = render_latest_jobs_html(
        rows,
        root=root,
        latest_n=latest_n,
        skipped_count=skipped_count,
        generated_at=utc_now_iso(),
    )

    destination = output_path or (root / "reports" / "latest_jobs.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)
    return destination


__all__ = [
    "extract_report_row",
    "generate_latest_jobs_report",
    "load_training_run_manifests",
    "render_latest_jobs_html",
    "select_latest_job_rows",
]
