#!/usr/bin/env python3
"""
Data Backup Script for Cloud Deployment

This script creates a compressed backup of all necessary training data
for deployment on cloud machines.

Usage:
    # Create backup (includes processed data + masks)
    python tools/data_backup.py

    # Create backup with checkpoints
    python tools/data_backup.py --include-checkpoints

    # Create backup with Aim logs
    python tools/data_backup.py --include-aim

    # Full backup (everything)
    python tools/data_backup.py --full

    # Custom output location
    python tools/data_backup.py --output /path/to/backup.zip

Workflow:
    1. Run this script locally to create backup
    2. Git clone repo on cloud machine
    3. Transfer backup.zip to cloud machine (scp/rsync)
    4. Unzip on cloud machine: python tools/deployment/data_restore.py backup.zip
"""

import argparse
import zipfile
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json


def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_file_list(base_path, patterns=None, exclude_patterns=None):
    """
    Get list of files to backup

    Args:
        base_path: Base directory to search
        patterns: List of glob patterns to include
        exclude_patterns: List of patterns to exclude

    Returns:
        List of Path objects
    """
    files = []
    base = Path(base_path)

    if not base.exists():
        return files

    if patterns is None:
        # Get all files recursively
        for root, dirs, filenames in os.walk(base):
            # Filter out excluded directories
            if exclude_patterns:
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(
                        Path(root) / d == Path(base) / ex for ex in exclude_patterns
                    )
                ]

            for filename in filenames:
                file_path = Path(root) / filename
                # Check exclusions
                if exclude_patterns:
                    if any(
                        str(ex) in str(file_path.relative_to(base))
                        for ex in exclude_patterns
                    ):
                        continue
                files.append(file_path)
    else:
        # Use specific patterns
        for pattern in patterns:
            files.extend(base.glob(pattern))

    return files


def create_backup(args):
    """Create backup archive"""

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("\n" + "=" * 80)
    print("MRI Data Backup Creator")
    print("=" * 80 + "\n")

    # Define what to backup
    backup_items = {
        "essential": {
            "data/processed/": "Processed PNG images",
            "data/processed_seg/": "Segmentation masks",
            "data/tcia/": "TCIA manifests",
            "data/splitted_images/": "Image metadata (parquet)",
        },
        "checkpoints": {
            "checkpoints/": "Trained model checkpoints",
        },
        "aim": {
            ".aim/": "Experiment tracking data",
        },
        "optional": {
            "data/splitted_info/": "Enriched metadata (optional)",
            "validation_results/": "Validation visualizations (optional)",
        },
    }

    # Determine what to include
    items_to_backup = {}
    items_to_backup.update(backup_items["essential"])

    if args.include_checkpoints or args.full:
        items_to_backup.update(backup_items["checkpoints"])

    if args.include_aim or args.full:
        items_to_backup.update(backup_items["aim"])

    if args.full:
        items_to_backup.update(backup_items["optional"])

    # Calculate sizes
    print("Calculating sizes...\n")
    sizes = {}
    total_size = 0

    for path, desc in items_to_backup.items():
        if Path(path).exists():
            size = get_dir_size(path)
            sizes[path] = size
            total_size += size
            status = "✓" if size > 0 else "⚠"
            print(f"{status} {path:40s} {format_size(size):>12s} - {desc}")
        else:
            print(f"✗ {path:40s} {'NOT FOUND':>12s} - {desc}")
            sizes[path] = 0

    print(f"\nTotal size to backup: {format_size(total_size)}")

    if total_size == 0:
        print("\n⚠️  No data found to backup!")
        return

    # Confirm
    if not args.yes:
        response = input(f"\nProceed with backup? [y/N]: ")
        if response.lower() != "y":
            print("Backup cancelled.")
            return

    # Create backup
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mri_data_backup_{timestamp}.zip"

    output_path = Path(output_file)

    print(f"\nCreating backup: {output_path}")
    print("This may take several minutes...\n")

    # Collect all files
    all_files = []
    for path in items_to_backup.keys():
        if Path(path).exists():
            files = get_file_list(path)
            all_files.extend(files)

    # Create zip with progress bar
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(all_files, desc="Archiving files", unit="file"):
            arcname = str(file_path)
            zipf.write(file_path, arcname=arcname)

    # Create metadata file
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_files": len(all_files),
        "total_size_bytes": total_size,
        "total_size_human": format_size(total_size),
        "included_directories": list(items_to_backup.keys()),
        "backup_type": "full" if args.full else "essential",
        "includes_checkpoints": args.include_checkpoints or args.full,
        "includes_aim": args.include_aim or args.full,
    }

    metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Final stats
    backup_size = output_path.stat().st_size
    compression_ratio = (1 - backup_size / total_size) * 100 if total_size > 0 else 0

    print("\n" + "=" * 80)
    print("Backup Complete!")
    print("=" * 80)
    print(f"\nBackup file: {output_path}")
    print(f"Metadata file: {metadata_file}")
    print(f"Original size: {format_size(total_size)}")
    print(f"Compressed size: {format_size(backup_size)}")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print(f"Total files: {len(all_files)}")

    print("\n" + "=" * 80)
    print("Transfer Instructions")
    print("=" * 80)
    print("\n1. Git clone on cloud machine:")
    print("   git clone <your-repo-url>")
    print("   cd <repo-name>")

    print("\n2. Transfer backup from local to cloud:")
    print(f"   # From local machine:")
    print(f"   scp {output_path} user@cloud:/path/to/repo/")

    print("\n3. Extract on cloud machine:")
    print(f"   # On cloud machine:")
    print(f"   cd /path/to/repo")
    print(f"   python tools/deployment/data_restore.py {output_path.name}")

    print("\n4. Verify data integrity:")
    print("   python service/validate_data.py")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create backup of MRI training data for cloud deployment"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zip file path (default: mri_data_backup_<timestamp>.zip)",
    )

    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include trained model checkpoints",
    )

    parser.add_argument(
        "--include-aim",
        action="store_true",
        help="Include Aim experiment tracking data",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full backup (includes checkpoints, Aim, and optional data)",
    )

    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    create_backup(args)


if __name__ == "__main__":
    main()
