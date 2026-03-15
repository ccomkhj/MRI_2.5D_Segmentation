#!/usr/bin/env python3
"""
Validate 2.5D pipeline setup without requiring PyTorch.

This script checks:
1. Manifest files exist
2. Data structure is correct
3. Required files are accessible
4. Basic statistics about the dataset

Usage:
    python tools/validate_2d5_setup.py
"""

import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict


def check_manifest(manifest_path: Path) -> dict:
    """
    Check a manifest file and return statistics.

    Args:
        manifest_path: Path to manifest CSV

    Returns:
        Dictionary with statistics
    """
    if not manifest_path.exists():
        return {"exists": False}

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        return {"exists": True, "error": str(e)}

    stats = {
        "exists": True,
        "num_rows": len(df),
        "num_cases": df["case_id"].nunique(),
        "num_series": df["series_uid"].nunique(),
    }

    # Check for masks
    if "mask_path" in df.columns:
        has_masks = df["mask_path"].notna() & (df["mask_path"] != "")
        stats["num_with_masks"] = has_masks.sum()
        stats["mask_coverage"] = f"{has_masks.sum() / len(df) * 100:.1f}%"

    # Group by series to get slice counts
    series_slice_counts = []
    for (case_id, series_uid), group in df.groupby(["case_id", "series_uid"]):
        series_slice_counts.append(len(group))

    if series_slice_counts:
        stats["min_slices_per_series"] = min(series_slice_counts)
        stats["max_slices_per_series"] = max(series_slice_counts)
        stats["avg_slices_per_series"] = sum(series_slice_counts) / len(
            series_slice_counts
        )

    # Check a few image paths
    sample_paths = df["image_path"].head(5).tolist()
    stats["sample_images_exist"] = sum(1 for p in sample_paths if Path(p).exists())
    stats["sample_images_total"] = len(sample_paths)

    return stats


def check_2d5_compatibility(manifest_path: Path, stack_depth: int = 5) -> dict:
    """
    Check how many valid 2.5D samples can be created.

    Args:
        manifest_path: Path to manifest CSV
        stack_depth: Stack depth for 2.5D

    Returns:
        Dictionary with compatibility info
    """
    if not manifest_path.exists():
        return {"compatible": False, "reason": "Manifest not found"}

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        return {"compatible": False, "reason": f"Error reading manifest: {e}"}

    df["slice_idx"] = df["slice_idx"].astype(int)

    # Count valid samples per series
    valid_samples = 0
    total_series = 0
    series_with_enough_slices = 0

    half_depth = stack_depth // 2

    for (case_id, series_uid), group in df.groupby(["case_id", "series_uid"]):
        total_series += 1
        num_slices = len(group)

        # All slices can be used with padding
        # But let's count those with full context (no padding needed)
        full_context_slices = max(0, num_slices - 2 * half_depth)

        if num_slices >= stack_depth:
            series_with_enough_slices += 1

        valid_samples += num_slices  # All slices are valid with padding

    return {
        "compatible": True,
        "total_series": total_series,
        "series_with_full_context": series_with_enough_slices,
        "total_valid_samples": valid_samples,
        "stack_depth": stack_depth,
    }


def main():
    print("=" * 80)
    print("2.5D Pipeline Setup Validation")
    print("=" * 80)

    # Check project structure
    project_root = Path.cwd()
    print(f"\nProject root: {project_root}")

    # Check for processed data directory
    processed_dir = project_root / "data" / "processed"
    if not processed_dir.exists():
        print(f"\n✗ Error: Processed data directory not found: {processed_dir}")
        print("  Please run: python tools/dicom_converter.py --all")
        return 1

    print(f"✓ Processed data directory exists: {processed_dir}")

    # Check for manifest files
    manifest_files = {
        "class1": processed_dir / "class1" / "manifest.csv",
        "class2": processed_dir / "class2" / "manifest.csv",
        "class3": processed_dir / "class3" / "manifest.csv",
        "class4": processed_dir / "class4" / "manifest.csv",
        "combined": processed_dir / "manifest_all.csv",
    }

    print("\n" + "=" * 80)
    print("Manifest Files")
    print("=" * 80)

    found_manifests = []
    all_stats = {}

    for name, path in manifest_files.items():
        stats = check_manifest(path)
        all_stats[name] = stats

        if stats["exists"]:
            found_manifests.append((name, path))
            print(f"\n✓ {name}: {path}")
            if "error" in stats:
                print(f"  ✗ Error: {stats['error']}")
            else:
                print(f"  Rows: {stats['num_rows']}")
                print(f"  Cases: {stats['num_cases']}")
                print(f"  Series: {stats['num_series']}")
                if "num_with_masks" in stats:
                    print(
                        f"  With masks: {stats['num_with_masks']} ({stats['mask_coverage']})"
                    )
                if "avg_slices_per_series" in stats:
                    print(
                        f"  Slices per series: {stats['min_slices_per_series']}-"
                        f"{stats['max_slices_per_series']} "
                        f"(avg: {stats['avg_slices_per_series']:.1f})"
                    )
                print(
                    f"  Sample images exist: {stats['sample_images_exist']}/{stats['sample_images_total']}"
                )
        else:
            print(f"\n✗ {name}: Not found at {path}")

    if not found_manifests:
        print("\n✗ No manifest files found!")
        print("  Please run: python tools/dicom_converter.py --all")
        return 1

    # Check 2.5D compatibility
    print("\n" + "=" * 80)
    print("2.5D Compatibility Check")
    print("=" * 80)

    for stack_depth in [3, 5, 7]:
        print(f"\nStack depth = {stack_depth}:")

        for name, path in found_manifests:
            if "error" in all_stats[name]:
                continue

            compat = check_2d5_compatibility(path, stack_depth)

            if compat["compatible"]:
                print(f"  {name}:")
                print(f"    Total series: {compat['total_series']}")
                print(
                    f"    Series with full context: {compat['series_with_full_context']}"
                )
                print(f"    Total valid samples: {compat['total_valid_samples']}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_rows = sum(
        s.get("num_rows", 0) for s in all_stats.values() if s.get("exists")
    )
    total_with_masks = sum(
        s.get("num_with_masks", 0) for s in all_stats.values() if s.get("exists")
    )

    print(f"\nTotal slices: {total_rows}")
    print(f"Total with masks: {total_with_masks}")
    if total_rows > 0:
        print(f"Mask coverage: {total_with_masks / total_rows * 100:.1f}%")

    # Check Python packages
    print("\n" + "=" * 80)
    print("Python Package Check")
    print("=" * 80)

    packages = {
        "pandas": "Required for data loading",
        "numpy": "Required for array operations",
        "pillow": "Required for image loading",
        "torch": "Required for PyTorch models (2.5D pipeline)",
        "monai": "Required for MONAI SegResNet",
        "segmentation_models_pytorch": "Required for SMP ResUNet",
    }

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package}: Installed")
        except ImportError:
            if package in ["torch", "monai", "segmentation_models_pytorch"]:
                print(f"○ {package}: Not installed (optional for 2.5D pipeline)")
                print(f"  Install with: pip install -r requirements_2d5.txt")
            else:
                print(f"✗ {package}: Not installed (required)")
                print(f"  {description}")

    # Next steps
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)

    if total_rows == 0:
        print("\n1. Process DICOM data:")
        print("   python tools/dicom_converter.py --all")
    else:
        print("\n1. ✓ Data is ready!")

    print("\n2. Install 2.5D pipeline dependencies:")
    print("   pip install -r requirements_2d5.txt")

    print("\n3. Run a modular segmentation smoke train:")
    print("   python mri/cli/train.py --config mri/config/task/segmentation.yaml --epochs 1 --batch_size 2")

    print("\n4. Read the project documentation:")
    print("   README.md")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
