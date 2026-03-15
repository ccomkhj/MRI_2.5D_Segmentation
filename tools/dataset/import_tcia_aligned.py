#!/usr/bin/env python3
"""Import or sync aligned TCIA data into this repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mri.data.metadata import load_metadata


DEFAULT_SOURCE = Path("/Users/huijokim/personal/tcia-handler/data/aligned_v2")
DEFAULT_DEST = Path("data/aligned_v2")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _metadata_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    raise FileNotFoundError(path)


def _expected_case_dirs(case_info: Dict[str, Any]) -> List[str]:
    required = ["t2"]
    if case_info.get("has_adc", False):
        required.append("adc")
    if case_info.get("has_calc", False):
        required.append("calc")
    if case_info.get("slices_with_prostate"):
        required.append("mask_prostate")
    if case_info.get("slices_with_target"):
        required.append("mask_target1")
    return required


def _sample_required_files(sample: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    files = sample.get("files", {})
    yield "t2", files.get("t2", f"{sample['slice_idx']:04d}.png")
    if sample.get("has_adc", False):
        yield "adc", files.get("adc", f"{sample['slice_idx']:04d}.png")
    if sample.get("has_calc", False):
        yield "calc", files.get("calc", f"{sample['slice_idx']:04d}.png")
    if sample.get("has_prostate", False):
        yield "mask_prostate", files.get("mask_prostate", f"{sample['slice_idx']:04d}.png")
    if sample.get("has_target", False):
        yield "mask_target1", files.get("mask_target1", f"{sample['slice_idx']:04d}.png")


def validate_aligned_dataset(
    dataset_root: str | Path,
    *,
    validate_files: bool = False,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata.json"
    meta = load_metadata(metadata_path)

    missing_cases: List[str] = []
    missing_dirs: List[str] = []
    missing_files: List[str] = []

    for case_id, case_info in meta.cases.items():
        case_dir = dataset_root / case_id
        if not case_dir.is_dir():
            missing_cases.append(case_id)
            continue

        for dir_name in _expected_case_dirs(case_info):
            if not (case_dir / dir_name).is_dir():
                missing_dirs.append(f"{case_id}/{dir_name}")

    if validate_files:
        for sample in meta.samples:
            case_dir = dataset_root / sample["case_id"]
            for dir_name, filename in _sample_required_files(sample):
                if not (case_dir / dir_name / filename).exists():
                    missing_files.append(f"{sample['case_id']}/{dir_name}/{filename}")

    if missing_cases or missing_dirs or missing_files:
        parts = []
        if missing_cases:
            preview = ", ".join(missing_cases[:5])
            suffix = "" if len(missing_cases) <= 5 else f", ... (+{len(missing_cases) - 5} more)"
            parts.append(f"missing case directories: {preview}{suffix}")
        if missing_dirs:
            preview = ", ".join(missing_dirs[:5])
            suffix = "" if len(missing_dirs) <= 5 else f", ... (+{len(missing_dirs) - 5} more)"
            parts.append(f"missing modality/mask directories: {preview}{suffix}")
        if missing_files:
            preview = ", ".join(missing_files[:5])
            suffix = "" if len(missing_files) <= 5 else f", ... (+{len(missing_files) - 5} more)"
            parts.append(f"missing sample files: {preview}{suffix}")
        raise FileNotFoundError(f"Aligned dataset validation failed for {dataset_root}: " + "; ".join(parts))

    return {
        "dataset_root": str(dataset_root.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _metadata_sha256(metadata_path),
        "num_cases": len(meta.cases),
        "num_samples": len(meta.samples),
        "case_ids": sorted(meta.cases.keys()),
        "validate_files": validate_files,
    }


def _datasets_match(source_summary: Dict[str, Any], dest_summary: Dict[str, Any]) -> bool:
    return (
        source_summary["metadata_sha256"] == dest_summary["metadata_sha256"]
        and source_summary["num_cases"] == dest_summary["num_cases"]
        and source_summary["num_samples"] == dest_summary["num_samples"]
    )


def write_import_manifest(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def default_import_manifest_path(dest_root: str | Path) -> Path:
    dest_root = Path(dest_root)
    return dest_root.parent / f"{dest_root.name}_import_manifest.json"


def sync_aligned_dataset(
    source_root: str | Path,
    dest_root: str | Path,
    *,
    mode: str = "copy",
    force: bool = False,
    validate_files: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    source_root = Path(source_root)
    dest_root = Path(dest_root)

    if mode not in {"copy", "link"}:
        raise ValueError(f"Unsupported mode: {mode}")

    source_summary = validate_aligned_dataset(source_root, validate_files=validate_files)
    action = "reuse"
    existing_dest_summary: Dict[str, Any] | None = None

    if dest_root.exists():
        try:
            existing_dest_summary = validate_aligned_dataset(dest_root, validate_files=validate_files)
        except FileNotFoundError:
            if not force:
                raise

        if dest_root.resolve() == source_root.resolve():
            action = "reuse"
        elif existing_dest_summary and _datasets_match(source_summary, existing_dest_summary):
            action = "reuse"
        elif force:
            action = "replace"
            if not dry_run:
                _remove_path(dest_root)
        else:
            raise FileExistsError(
                f"Destination {dest_root} already exists and does not match {source_root}. "
                "Use --force to replace it."
            )
    else:
        action = "create"

    if action in {"create", "replace"} and not dry_run:
        dest_root.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copytree(source_root, dest_root, symlinks=False)
        else:
            dest_root.symlink_to(source_root.resolve(), target_is_directory=True)

    dest_summary: Dict[str, Any] | None = None
    if dest_root.exists():
        dest_summary = validate_aligned_dataset(dest_root, validate_files=validate_files)

    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "source_root": str(source_root.resolve()),
        "dest_root": str(dest_root.resolve()) if dest_root.exists() else str(dest_root),
        "mode": mode,
        "force": force,
        "dry_run": dry_run,
        "action": action,
        "source": source_summary,
        "destination": dest_summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import aligned TCIA data into this repository")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source aligned_v2 directory")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory inside this repo")
    parser.add_argument("--mode", choices=["copy", "link"], default="copy", help="Copy files or create a symlink")
    parser.add_argument("--force", action="store_true", help="Replace a mismatched destination")
    parser.add_argument("--validate-files", action="store_true", help="Validate sample-level files from metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without changing destination data")
    parser.add_argument(
        "--manifest-output",
        help="Optional JSON manifest path. Defaults to <dest>/import_manifest.json.",
    )
    args = parser.parse_args(argv)

    result = sync_aligned_dataset(
        args.source,
        args.dest,
        mode=args.mode,
        force=args.force,
        validate_files=args.validate_files,
        dry_run=args.dry_run,
    )

    manifest_output = Path(args.manifest_output) if args.manifest_output else default_import_manifest_path(args.dest)
    manifest_path = write_import_manifest(manifest_output, result)

    print(f"Source:      {result['source_root']}")
    print(f"Destination: {result['dest_root']}")
    print(f"Mode:        {result['mode']}")
    print(f"Action:      {result['action']}")
    print(f"Dry run:     {result['dry_run']}")
    if result["destination"] is not None:
        print(f"Cases:       {result['destination']['num_cases']}")
        print(f"Samples:     {result['destination']['num_samples']}")
    else:
        print(f"Cases:       {result['source']['num_cases']}")
        print(f"Samples:     {result['source']['num_samples']}")
    print(f"Manifest:    {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
