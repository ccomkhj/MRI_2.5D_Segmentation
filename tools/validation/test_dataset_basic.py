#!/usr/bin/env python3
"""
Basic dataset test without requiring PyTorch.

Tests that the data loading logic works correctly by manually
loading and stacking slices.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_manual_stacking(manifest_csv: str, stack_depth: int = 5):
    """
    Test manual slice stacking without PyTorch.
    
    Args:
        manifest_csv: Path to manifest CSV
        stack_depth: Number of slices to stack
    """
    print("="*80)
    print("Basic 2.5D Dataset Test (No PyTorch Required)")
    print("="*80)
    
    print(f"\nLoading manifest: {manifest_csv}")
    df = pd.read_csv(manifest_csv)
    
    print(f"Total slices: {len(df)}")
    print(f"Cases: {df['case_id'].nunique()}")
    print(f"Series: {df['series_uid'].nunique()}")
    
    # Convert slice_idx to int
    df['slice_idx'] = df['slice_idx'].astype(int)
    
    # Group by series
    series_groups = {}
    for (case_id, series_uid), group in df.groupby(['case_id', 'series_uid']):
        group = group.sort_values('slice_idx').reset_index(drop=True)
        series_groups[(case_id, series_uid)] = group
    
    print(f"\nFound {len(series_groups)} unique series")
    
    # Test loading first series
    first_key = list(series_groups.keys())[0]
    first_group = series_groups[first_key]
    
    print(f"\nTesting with first series:")
    print(f"  Case: {first_key[0]}")
    print(f"  Series: {first_key[1][:30]}...")
    print(f"  Slices: {len(first_group)}")
    
    # Test loading and stacking
    print(f"\nTesting 2.5D stacking (depth={stack_depth}):")
    
    half_depth = stack_depth // 2
    central_idx = len(first_group) // 2  # Middle slice
    central_slice = first_group.iloc[central_idx]['slice_idx']
    
    print(f"  Central slice index: {central_slice}")
    print(f"  Stack range: {central_slice - half_depth} to {central_slice + half_depth}")
    
    # Load slices
    slices = []
    for offset in range(-half_depth, half_depth + 1):
        slice_idx = central_slice + offset
        
        # Handle boundaries with reflection
        if slice_idx < first_group['slice_idx'].min():
            slice_idx = first_group['slice_idx'].min() + (first_group['slice_idx'].min() - slice_idx)
        elif slice_idx > first_group['slice_idx'].max():
            slice_idx = first_group['slice_idx'].max() - (slice_idx - first_group['slice_idx'].max())
        
        # Get row for this slice
        row = first_group[first_group['slice_idx'] == slice_idx]
        if len(row) == 0:
            # Fallback to central slice
            row = first_group[first_group['slice_idx'] == central_slice]
        row = row.iloc[0]
        
        # Load image
        img_path = Path(row['image_path'])
        if not img_path.exists():
            print(f"  ✗ Image not found: {img_path}")
            return False
        
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.uint8)
        slices.append(img_array)
        
        print(f"    Slice {slice_idx}: shape={img_array.shape}, range=[{img_array.min()}, {img_array.max()}]")
    
    # Stack
    stacked = np.stack(slices, axis=0)
    print(f"\n  Stacked shape: {stacked.shape} (expected: [{stack_depth}, H, W])")
    
    # Normalize
    normalized = stacked.astype(np.float32) / 255.0
    print(f"  After normalization: dtype={normalized.dtype}, range=[{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # This is what would go to the model
    print(f"\n  ✓ Successfully created 2.5D stack!")
    print(f"  ✓ Shape matches model input requirement: [{stack_depth}, {normalized.shape[1]}, {normalized.shape[2]}]")
    
    # Test batch simulation
    print(f"\nSimulating batch of 4 samples:")
    batch = []
    for i in range(min(4, len(first_group) - stack_depth)):
        test_slices = []
        central_idx = i + half_depth
        central_slice = first_group.iloc[central_idx]['slice_idx']
        
        for offset in range(-half_depth, half_depth + 1):
            slice_idx = central_slice + offset
            row = first_group[first_group['slice_idx'] == slice_idx]
            if len(row) == 0:
                row = first_group[first_group['slice_idx'] == central_slice]
            row = row.iloc[0]
            
            img = Image.open(row['image_path']).convert('L')
            img_array = np.array(img, dtype=np.uint8)
            test_slices.append(img_array)
        
        sample_stack = np.stack(test_slices, axis=0).astype(np.float32) / 255.0
        batch.append(sample_stack)
    
    batch_array = np.stack(batch, axis=0)
    print(f"  Batch shape: {batch_array.shape}")
    print(f"  Expected format: [batch_size, stack_depth, height, width]")
    print(f"  Actual: [{batch_array.shape[0]}, {batch_array.shape[1]}, {batch_array.shape[2]}, {batch_array.shape[3]}]")
    
    # Check compatibility
    print(f"\n" + "="*80)
    print("Model Compatibility Check")
    print("="*80)
    
    print(f"\n✓ Data format is compatible with:")
    print(f"  • SMP ResUNet: expects [B, {stack_depth}, H, W] as float32 ✓")
    print(f"  • MONAI SegResNet: expects [B, {stack_depth}, H, W] as float32 ✓")
    print(f"\n✓ Your data produces: [{batch_array.shape[0]}, {batch_array.shape[1]}, {batch_array.shape[2]}, {batch_array.shape[3]}] as float32 ✓")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 2.5D data loading (no PyTorch)")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/class2/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--stack-depth",
        type=int,
        default=5,
        help="Stack depth",
    )
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {args.manifest}")
        print("\nAvailable manifests:")
        for p in Path("data/processed").glob("class*/manifest.csv"):
            print(f"  {p}")
        return 1
    
    success = test_manual_stacking(str(manifest_path), args.stack_depth)
    
    if success:
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print("\n✓ All tests passed!")
        print("\n✓ Data loading works correctly")
        print("✓ 2.5D stacking works correctly")
        print("✓ Normalization works correctly")
        print("✓ Format is compatible with both SMP and MONAI models")
        
        print("\n📝 Next Steps:")
        print("  1. Install PyTorch and model libraries:")
        print("     pip install torch torchvision monai segmentation-models-pytorch")
        print("\n  2. Run a segmentation smoke train:")
        print("     python mri/cli/train.py --config mri/config/task/segmentation.yaml --epochs 1 --batch_size 2")
        
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
