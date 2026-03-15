"""Classification inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
except Exception:  # pragma: no cover
    precision_recall_fscore_support = None
    roc_auc_score = None


def _macro_f1(preds: List[int], targets: List[int], num_classes: int) -> float:
    f1s = []
    for class_idx in range(num_classes):
        tp = sum(1 for pred, target in zip(preds, targets) if pred == class_idx and target == class_idx)
        fp = sum(1 for pred, target in zip(preds, targets) if pred == class_idx and target != class_idx)
        fn = sum(1 for pred, target in zip(preds, targets) if pred != class_idx and target == class_idx)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


def _confusion_matrix(preds: List[int], targets: List[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(targets, preds):
        matrix[target][pred] += 1
    return matrix


def _expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> Dict[str, object]:
    if confidences.size == 0:
        return {"num_bins": num_bins, "ece": None, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins = []
    for idx in range(num_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "start": float(lower),
                    "end": float(upper),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += abs(bin_acc - bin_conf) * (count / confidences.size)
        bins.append(
            {
                "start": float(lower),
                "end": float(upper),
                "count": count,
                "mean_confidence": bin_conf,
                "accuracy": bin_acc,
            }
        )
    return {"num_bins": num_bins, "ece": float(ece), "bins": bins}


def _multiclass_brier_score(targets: np.ndarray, probs: np.ndarray) -> float | None:
    if targets.size == 0 or probs.size == 0:
        return None
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _compute_auroc(targets: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, object]:
    if roc_auc_score is None or targets.size == 0 or probs.size == 0:
        return {
            "auroc_macro_ovr": None,
            "auroc_weighted_ovr": None,
            "per_class_auroc": {str(label): None for label in range(num_classes)},
        }

    per_class: Dict[str, float | None] = {}
    for label in range(num_classes):
        binary_targets = (targets == label).astype(int)
        if binary_targets.min() == binary_targets.max():
            per_class[str(label)] = None
            continue
        per_class[str(label)] = float(roc_auc_score(binary_targets, probs[:, label]))

    try:
        auroc_macro_ovr = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auroc_macro_ovr = None
    try:
        auroc_weighted_ovr = float(roc_auc_score(targets, probs, multi_class="ovr", average="weighted"))
    except ValueError:
        auroc_weighted_ovr = None

    return {
        "auroc_macro_ovr": auroc_macro_ovr,
        "auroc_weighted_ovr": auroc_weighted_ovr,
        "per_class_auroc": per_class,
    }


def run_classification_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_csv: str | Path | None = None,
) -> Dict[str, List[Dict] | Dict[str, object]]:
    model = model.to(device)
    model.eval()

    results: List[Dict] = []
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1]
            metas = batch[2]
            if isinstance(metas, dict):
                meta_list = [
                    {k: metas[k][i] for k in metas}
                    for i in range(len(metas[next(iter(metas))]))
                ]
            else:
                meta_list = metas

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            num_classes = probs.shape[1]

            for i, m in enumerate(meta_list):
                row = {
                    "case_id": m["case_id"],
                    "pred": int(preds[i].item()),
                    "prob": float(probs[i, preds[i]].item()),
                    "target": int(labels[i].item()),
                }
                for class_idx in range(num_classes):
                    row[f"prob_{class_idx}"] = float(probs[i, class_idx].item())
                results.append(row)
                all_preds.append(row["pred"])
                all_targets.append(row["target"])
                all_probs.append([row[f"prob_{class_idx}"] for class_idx in range(num_classes)])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            fieldnames = list(results[0].keys()) if results else ["case_id", "pred", "prob", "target"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary: Dict[str, object] = {
        "num_cases": len(results),
    }
    if all_targets:
        targets_np = np.asarray(all_targets, dtype=np.int64)
        preds_np = np.asarray(all_preds, dtype=np.int64)
        probs_np = np.asarray(all_probs, dtype=np.float64)
        confidences = probs_np.max(axis=1)
        correct = (preds_np == targets_np).astype(np.float64)
        accuracy = sum(int(pred == target) for pred, target in zip(all_preds, all_targets)) / max(1, len(all_targets))
        confusion = _confusion_matrix(all_preds, all_targets, num_classes)
        per_class_metrics = []
        if precision_recall_fscore_support is not None:
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets,
                all_preds,
                labels=list(range(num_classes)),
                zero_division=0,
            )
            for label in range(num_classes):
                per_class_metrics.append(
                    {
                        "label": label,
                        "precision": float(precision[label]),
                        "recall": float(recall[label]),
                        "f1": float(f1[label]),
                        "support": int(support[label]),
                    }
                )
        else:
            for label in range(num_classes):
                row_sum = sum(confusion[label])
                col_sum = sum(confusion[row][label] for row in range(num_classes))
                tp = confusion[label][label]
                precision_val = tp / (col_sum + 1e-8)
                recall_val = tp / (row_sum + 1e-8)
                f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
                per_class_metrics.append(
                    {
                        "label": label,
                        "precision": float(precision_val),
                        "recall": float(recall_val),
                        "f1": float(f1_val),
                        "support": int(row_sum),
                    }
                )

        calibration = _expected_calibration_error(confidences, correct)
        auroc = _compute_auroc(targets_np, probs_np, num_classes)
        macro_f1 = _macro_f1(all_preds, all_targets, num_classes)
        summary.update(
            {
                "accuracy": float(accuracy),
                "macro_f1": macro_f1,
                "confusion_matrix": confusion,
                "per_class_metrics": per_class_metrics,
                "class_labels": list(range(num_classes)),
                "mean_confidence": float(confidences.mean()),
                "expected_calibration_error": calibration["ece"],
                "calibration": calibration,
                "brier_score": _multiclass_brier_score(targets_np, probs_np),
                "primary_metric_name": "macro_f1",
                "best_metric": macro_f1,
                **auroc,
            }
        )

    return {
        "rows": results,
        "summary": summary,
    }
