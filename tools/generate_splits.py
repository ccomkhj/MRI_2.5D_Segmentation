#!/usr/bin/env python3
"""Generate train/val/test split YAML files from metadata.json."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mri.data.metadata import load_metadata
from mri.data.index_builders import classification_label_from_case_info


def _split_list(items: List[str], ratios: List[float], rng: random.Random) -> Dict[str, List[str]]:
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def _case_label(case_info: Dict[str, object], label_space: str) -> int:
    if label_space == "original":
        return int(case_info.get("class", 0))
    if label_space == "downstream_5class":
        return classification_label_from_case_info(case_info)
    raise ValueError(f"Unsupported label_space: {label_space}")


def build_splits(
    meta_path: str | Path,
    ratios: List[float],
    seed: int,
    stratify: bool,
    label_space: str = "downstream_5class",
) -> Dict[str, List[str]]:
    meta = load_metadata(meta_path)
    rng = random.Random(seed)

    if not stratify:
        return _split_list(list(meta.cases.keys()), ratios, rng)

    # Stratify by the requested label space so downstream classification stays balanced.
    by_class: Dict[int, List[str]] = {}
    for case_id, case_info in meta.cases.items():
        cls = _case_label(case_info, label_space)
        by_class.setdefault(cls, []).append(case_id)

    splits = {"train": [], "val": [], "test": []}
    for case_ids in by_class.values():
        part = _split_list(case_ids, ratios, rng)
        for k in splits:
            splits[k].extend(part[k])

    # Shuffle final splits to remove class ordering
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def summarize_splits(
    meta_path: str | Path,
    splits: Dict[str, List[str]],
    label_space: str = "downstream_5class",
) -> Dict[str, object]:
    meta = load_metadata(meta_path)

    split_summaries: Dict[str, Dict[str, object]] = {}
    labels = set()
    for split_name in ("train", "val", "test"):
        cases = list(splits.get(split_name, []))
        histogram: Dict[str, int] = {}
        for case_id in cases:
            case_info = meta.cases[case_id]
            label = _case_label(case_info, label_space)
            labels.add(label)
            key = str(label)
            histogram[key] = histogram.get(key, 0) + 1

        split_summaries[split_name] = {
            "num_cases": len(cases),
            "label_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        }

    return {
        "schema_version": 1,
        "metadata": str(Path(meta_path)),
        "label_space": label_space,
        "num_cases": len(meta.cases),
        "labels": sorted(labels),
        "splits": split_summaries,
    }


def write_split_artifacts(
    *,
    splits: Dict[str, List[str]],
    summary: Dict[str, object],
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(splits, f, sort_keys=False)

    if summary_path is None:
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path, summary_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate YAML splits from metadata.json")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output YAML path")
    parser.add_argument("--ratios", default="0.7,0.15,0.15", help="Comma-separated ratios train,val,test")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--no-stratify", action="store_true", help="Disable class stratification")
    parser.add_argument(
        "--label-space",
        choices=["original", "downstream_5class"],
        default="downstream_5class",
        help="Label space used for stratification and split summaries.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional JSON path for split summary and label histograms. Defaults to <output>_summary.json.",
    )
    args = parser.parse_args(argv)

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must be three numbers summing to 1.0")

    splits = build_splits(
        args.metadata,
        ratios,
        args.seed,
        stratify=not args.no_stratify,
        label_space=args.label_space,
    )
    summary = summarize_splits(args.metadata, splits, label_space=args.label_space)

    out_path, summary_path = write_split_artifacts(
        splits=splits,
        summary=summary,
        output_path=args.output,
        summary_path=args.summary_output,
    )

    print(f"Wrote splits to {out_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Label space: {args.label_space}")
    for k in ("train", "val", "test"):
        split_summary = summary["splits"][k]
        print(f"  {k}: {split_summary['num_cases']} cases | labels={split_summary['label_histogram']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
