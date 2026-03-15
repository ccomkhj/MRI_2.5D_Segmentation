#!/usr/bin/env python3
"""
Data Restore Script for Cloud Deployment

This script extracts and verifies data backup on cloud machines.

Usage:
    # Restore from backup
    python tools/deployment/data_restore.py mri_data_backup_20240101_120000.zip

    # Dry run (show what would be extracted)
    python tools/deployment/data_restore.py backup.zip --dry-run

    # Force overwrite existing files
    python tools/deployment/data_restore.py backup.zip --force
"""

import argparse
import zipfile
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def load_metadata(backup_path):
    """Load metadata file if available"""
    metadata_file = backup_path.parent / f"{backup_path.stem}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def restore_backup(args):
    """Restore backup archive"""
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("\n" + "=" * 80)
    print("MRI Data Restore Tool")
    print("=" * 80 + "\n")
    
    backup_path = Path(args.backup_file)
    
    if not backup_path.exists():
        print(f"❌ Error: Backup file not found: {backup_path}")
        return
    
    # Load metadata if available
    metadata = load_metadata(backup_path)
    
    print(f"Backup file: {backup_path}")
    print(f"Size: {format_size(backup_path.stat().st_size)}")
    
    if metadata:
        print(f"Created: {metadata.get('created_at', 'Unknown')}")
        print(f"Type: {metadata.get('backup_type', 'Unknown')}")
        print(f"Files: {metadata.get('total_files', 'Unknown')}")
        print(f"Original size: {metadata.get('total_size_human', 'Unknown')}")
    
    # Open zip and list contents
    print("\nAnalyzing backup contents...")
    
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        file_list = zipf.namelist()
        total_files = len(file_list)
        
        # Categorize files
        categories = {}
        for file_path in file_list:
            top_dir = file_path.split('/')[0] if '/' in file_path else file_path
            if top_dir not in categories:
                categories[top_dir] = []
            categories[top_dir].append(file_path)
        
        print(f"\nBackup contains {total_files} files in {len(categories)} directories:\n")
        
        for category, files in sorted(categories.items()):
            print(f"  {category:40s} {len(files):>6} files")
    
    # Check for conflicts
    conflicts = []
    for file_path in file_list:
        target = Path(file_path)
        if target.exists():
            conflicts.append(file_path)
    
    if conflicts and not args.force:
        print(f"\n⚠️  Warning: {len(conflicts)} files already exist")
        print(f"   Use --force to overwrite, or --dry-run to preview")
        
        if not args.dry_run:
            response = input("\nOverwrite existing files? [y/N]: ")
            if response.lower() != 'y':
                print("Restore cancelled.")
                return
    
    # Dry run
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - No files will be extracted")
        print("=" * 80)
        print(f"\nWould extract {total_files} files to current directory")
        if conflicts:
            print(f"Would overwrite {len(conflicts)} existing files")
        return
    
    # Extract files
    print("\n" + "=" * 80)
    print("Extracting backup...")
    print("=" * 80 + "\n")
    
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        for file_path in tqdm(file_list, desc="Extracting", unit="file"):
            zipf.extract(file_path, path=".")
    
    # Verify extraction
    print("\nVerifying extraction...")
    
    verified = 0
    failed = []
    
    for file_path in file_list:
        target = Path(file_path)
        if target.exists():
            verified += 1
        else:
            failed.append(file_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("Restore Complete!")
    print("=" * 80)
    print(f"\nExtracted: {verified}/{total_files} files")
    
    if failed:
        print(f"Failed: {len(failed)} files")
        print("\nFailed files:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    else:
        print("✓ All files extracted successfully")
    
    # Next steps
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("\n1. Verify data integrity:")
    print("   python service/validate_data.py")
    
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n3. Start training:")
    print("   python mri/cli/train.py --config mri/config/task/segmentation.yaml")
    
    print("\n4. Monitor training:")
    print("   aim up")
    
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Restore MRI training data backup on cloud machine"
    )
    
    parser.add_argument(
        "backup_file",
        type=str,
        help="Path to backup zip file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually extracting"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files without prompting"
    )
    
    args = parser.parse_args()
    
    restore_backup(args)


if __name__ == "__main__":
    main()
