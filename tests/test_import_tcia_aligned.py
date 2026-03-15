from __future__ import annotations

from tools.dataset.import_tcia_aligned import sync_aligned_dataset, validate_aligned_dataset


def test_sync_aligned_dataset_copy_creates_matching_destination(fake_aligned_dataset, tmp_path):
    dest_root = tmp_path / "copied_aligned_v2"

    result = sync_aligned_dataset(fake_aligned_dataset, dest_root, mode="copy", validate_files=True)

    assert result["action"] == "create"
    assert dest_root.is_dir()
    assert validate_aligned_dataset(dest_root, validate_files=True)["num_cases"] == 12


def test_sync_aligned_dataset_link_reuses_existing_matching_destination(fake_aligned_dataset, tmp_path):
    dest_root = tmp_path / "linked_aligned_v2"

    first = sync_aligned_dataset(fake_aligned_dataset, dest_root, mode="link", validate_files=False)
    second = sync_aligned_dataset(fake_aligned_dataset, dest_root, mode="link", validate_files=False)

    assert first["action"] == "create"
    assert second["action"] == "reuse"
    assert dest_root.is_symlink()
