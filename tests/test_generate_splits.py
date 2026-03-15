from __future__ import annotations

from tools.generate_splits import build_splits, summarize_splits


def test_build_splits_downstream_5class_keeps_no_target_cases_in_all_splits(fake_aligned_dataset):
    metadata_path = fake_aligned_dataset / "metadata.json"

    splits = build_splits(
        metadata_path,
        ratios=[0.5, 0.25, 0.25],
        seed=1337,
        stratify=True,
        label_space="downstream_5class",
    )
    summary = summarize_splits(metadata_path, splits, label_space="downstream_5class")

    assert summary["splits"]["train"]["label_histogram"]["0"] == 2
    assert summary["splits"]["val"]["label_histogram"]["0"] == 1
    assert summary["splits"]["test"]["label_histogram"]["0"] == 1


def test_summarize_splits_can_report_original_label_space(fake_aligned_dataset):
    metadata_path = fake_aligned_dataset / "metadata.json"
    splits = {
        "train": ["class3/case_0005", "class4/case_0006"],
        "val": [],
        "test": [],
    }

    summary = summarize_splits(metadata_path, splits, label_space="original")

    assert summary["splits"]["train"]["label_histogram"] == {"3": 1, "4": 1}
