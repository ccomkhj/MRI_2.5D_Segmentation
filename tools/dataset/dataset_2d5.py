#!/usr/bin/env python3
"""
PyTorch Dataset for 2.5D MRI Segmentation.

This module provides a Dataset class that loads PNG slices from the processed
data directory and stacks them to create 2.5D input tensors suitable for
models like SMP ResUNet and MONAI SegResNet.

Usage:
    from tools.dataset.dataset_2d5 import MRI25DDataset
    from torch.utils.data import DataLoader
    
    dataset = MRI25DDataset(
        manifest_csv="data/processed/manifest_all.csv",
        stack_depth=5,
        normalize_method="scale"
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for images, masks in dataloader:
        # images: [batch_size, stack_depth, height, width]
        # masks: [batch_size, 1, height, width]
        outputs = model(images)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRI25DDataset(Dataset):
    """
    PyTorch Dataset for 2.5D MRI segmentation.
    
    Loads PNG slices from manifest CSV and stacks consecutive slices
    to create multi-channel 2D input suitable for 2.5D segmentation models.
    
    Args:
        manifest_csv: Path to manifest CSV file with columns:
                      case_id, series_uid, slice_idx, image_path, mask_path, etc.
        stack_depth: Number of consecutive slices to stack (k in 2.5D approach).
                     Should be odd for symmetric context around central slice.
        image_size: Target image size (H, W). If None, uses original size.
        transform: Optional transform to apply to images and masks.
        normalize_method: Normalization method - "scale" (0-1) or "zscore".
        has_masks: Whether to load masks. If False, returns None for masks.
        filter_by_class: Optional list of class numbers to include.
        padding_mode: How to handle edges - "reflect", "replicate", or "edge".
    """
    
    def __init__(
        self,
        manifest_csv: str,
        stack_depth: int = 5,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        normalize_method: str = "scale",
        has_masks: bool = True,
        filter_by_class: Optional[List[int]] = None,
        padding_mode: str = "reflect",
    ):
        self.manifest_csv = Path(manifest_csv)
        self.stack_depth = stack_depth
        self.image_size = image_size
        self.transform = transform
        self.normalize_method = normalize_method
        self.has_masks = has_masks
        self.padding_mode = padding_mode
        
        # Stack depth should be odd for symmetric context
        if stack_depth % 2 == 0:
            logger.warning(
                f"stack_depth={stack_depth} is even. Recommend odd values "
                f"for symmetric context around central slice."
            )
        
        self.half_depth = stack_depth // 2
        
        # Load manifest
        logger.info(f"Loading manifest from {manifest_csv}")
        self.manifest = pd.read_csv(manifest_csv)
        
        # Filter by class if specified
        if filter_by_class is not None:
            if 'class' in self.manifest.columns:
                before = len(self.manifest)
                self.manifest = self.manifest[
                    self.manifest['class'].isin(filter_by_class)
                ]
                logger.info(
                    f"Filtered by class {filter_by_class}: "
                    f"{before} → {len(self.manifest)} rows"
                )
        
        # Convert slice_idx to int
        self.manifest['slice_idx'] = self.manifest['slice_idx'].astype(int)
        
        # Group by case and series to know the slice range for each volume
        self.series_groups = self._build_series_index()
        
        # Build valid sample indices
        self.valid_samples = self._build_valid_samples()
        
        logger.info(f"Dataset initialized with {len(self.valid_samples)} valid samples")
        logger.info(f"Stack depth: {stack_depth}, Normalize: {normalize_method}")
    
    def _build_series_index(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Build index of all series and their slices.
        
        Returns:
            Dictionary mapping (case_id, series_uid) to DataFrame of slices.
        """
        series_groups = {}
        
        for (case_id, series_uid), group in self.manifest.groupby(
            ['case_id', 'series_uid']
        ):
            # Sort by slice index
            group = group.sort_values('slice_idx').reset_index(drop=True)
            series_groups[(case_id, series_uid)] = group
        
        logger.info(f"Found {len(series_groups)} unique series")
        return series_groups
    
    def _build_valid_samples(self) -> List[Dict]:
        """
        Build list of valid samples where we can extract a full stack.
        
        Each sample is a dict with:
            - case_id: Case ID
            - series_uid: Series UID
            - central_slice_idx: Index of central slice
            - slice_range: List of slice indices to load for the stack
            - central_row_idx: Index in the manifest DataFrame
        """
        valid_samples = []
        
        for (case_id, series_uid), group in self.series_groups.items():
            min_slice = group['slice_idx'].min()
            max_slice = group['slice_idx'].max()
            
            # For each potential central slice
            for idx, row in group.iterrows():
                central_slice = row['slice_idx']
                
                # Calculate the required slice range
                slice_range = list(range(
                    central_slice - self.half_depth,
                    central_slice + self.half_depth + 1
                ))
                
                # Check if we have all required slices
                # We'll handle padding later, but let's ensure we're not too far from edges
                # For now, accept all slices and handle padding in __getitem__
                
                valid_samples.append({
                    'case_id': case_id,
                    'series_uid': series_uid,
                    'central_slice_idx': central_slice,
                    'slice_range': slice_range,
                    'central_row_idx': idx,
                    'min_slice': min_slice,
                    'max_slice': max_slice,
                })
        
        return valid_samples
    
    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.valid_samples)
    
    def _get_slice_path(
        self, case_id: str, series_uid: str, slice_idx: int, is_mask: bool = False
    ) -> Optional[Path]:
        """
        Get the file path for a specific slice.
        
        Args:
            case_id: Case ID
            series_uid: Series UID
            slice_idx: Slice index
            is_mask: Whether to get mask path (True) or image path (False)
        
        Returns:
            Path to the file, or None if not found
        """
        group = self.series_groups.get((case_id, series_uid))
        if group is None:
            return None
        
        # Find row with matching slice_idx
        row = group[group['slice_idx'] == slice_idx]
        if len(row) == 0:
            return None
        
        row = row.iloc[0]
        
        if is_mask:
            mask_path = row['mask_path']
            if pd.isna(mask_path) or mask_path == '':
                return None
            # Handle multiple mask paths (separated by |)
            if '|' in str(mask_path):
                mask_path = mask_path.split('|')[0]  # Take first mask
            return Path(mask_path)
        else:
            return Path(row['image_path'])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load image from path as numpy array.
        
        Args:
            path: Path to image file
        
        Returns:
            Numpy array of shape [H, W], dtype uint8
        """
        img = Image.open(path).convert('L')  # Grayscale
        
        # Resize if needed
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        return np.array(img, dtype=np.uint8)
    
    def _pad_slice_index(
        self, slice_idx: int, min_slice: int, max_slice: int
    ) -> int:
        """
        Handle boundary cases by padding slice indices.
        
        Args:
            slice_idx: Requested slice index
            min_slice: Minimum available slice index
            max_slice: Maximum available slice index
        
        Returns:
            Valid slice index within bounds
        """
        if self.padding_mode == "reflect":
            if slice_idx < min_slice:
                # Reflect: if min=0 and request -1, return 1
                return min_slice + (min_slice - slice_idx)
            elif slice_idx > max_slice:
                # Reflect: if max=10 and request 11, return 9
                return max_slice - (slice_idx - max_slice)
        elif self.padding_mode == "replicate" or self.padding_mode == "edge":
            # Clamp to boundaries
            return np.clip(slice_idx, min_slice, max_slice)
        
        return slice_idx
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize array to float32.
        
        Args:
            arr: Input array [k, H, W], dtype uint8
        
        Returns:
            Normalized array [k, H, W], dtype float32
        """
        arr = arr.astype(np.float32)
        
        if self.normalize_method == "scale":
            # Scale to [0, 1]
            arr = arr / 255.0
        elif self.normalize_method == "zscore":
            # Z-score normalization
            mean = arr.mean()
            std = arr.std()
            if std > 0:
                arr = (arr - mean) / std
            else:
                arr = arr - mean
        else:
            raise ValueError(
                f"Unknown normalize_method: {self.normalize_method}. "
                f"Use 'scale' or 'zscore'"
            )
        
        return arr
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image_tensor, mask_tensor):
                - image_tensor: [stack_depth, H, W], dtype float32
                - mask_tensor: [1, H, W], dtype float32, or None if no mask
        """
        sample = self.valid_samples[idx]
        
        case_id = sample['case_id']
        series_uid = sample['series_uid']
        slice_range = sample['slice_range']
        min_slice = sample['min_slice']
        max_slice = sample['max_slice']
        central_slice_idx = sample['central_slice_idx']
        
        # Load stack of slices
        slices = []
        for slice_idx in slice_range:
            # Handle padding at boundaries
            padded_idx = self._pad_slice_index(slice_idx, min_slice, max_slice)
            
            img_path = self._get_slice_path(case_id, series_uid, padded_idx, is_mask=False)
            if img_path is None or not img_path.exists():
                # Fallback: use central slice
                img_path = self._get_slice_path(
                    case_id, series_uid, central_slice_idx, is_mask=False
                )
            
            img = self._load_image(img_path)
            slices.append(img)
        
        # Stack to [k, H, W]
        image_stack = np.stack(slices, axis=0)
        
        # Normalize
        image_stack = self._normalize(image_stack)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_stack)
        
        # Load mask for central slice if available
        mask_tensor = None
        if self.has_masks:
            mask_path = self._get_slice_path(
                case_id, series_uid, central_slice_idx, is_mask=True
            )
            
            if mask_path is not None and mask_path.exists():
                mask = self._load_image(mask_path)
                # Normalize mask to [0, 1]
                mask = (mask > 0).astype(np.float32)
                # Add channel dimension: [H, W] -> [1, H, W]
                mask = mask[np.newaxis, ...]
                mask_tensor = torch.from_numpy(mask)
        
        # Apply transforms if provided
        if self.transform is not None:
            # Transforms should accept (image, mask) and return (image, mask)
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        
        return image_tensor, mask_tensor
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get metadata about a sample without loading the images.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with sample metadata
        """
        return self.valid_samples[idx].copy()


def collate_fn_with_none(batch):
    """
    Custom collate function that handles None values in masks.
    
    If all masks are None, returns None for the mask batch.
    Otherwise, filters out None masks and returns only valid samples.
    """
    images = []
    masks = []
    
    for img, mask in batch:
        images.append(img)
        masks.append(mask)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Handle masks
    if all(m is None for m in masks):
        # All masks are None
        masks = None
    else:
        # Filter out None masks and stack
        valid_masks = [m for m in masks if m is not None]
        if valid_masks:
            masks = torch.stack(valid_masks, dim=0)
        else:
            masks = None
    
    return images, masks


def create_dataloader(
    manifest_csv: str,
    batch_size: int = 8,
    stack_depth: int = 5,
    image_size: Optional[Tuple[int, int]] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize_method: str = "scale",
    filter_by_class: Optional[List[int]] = None,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for 2.5D MRI segmentation.
    
    Args:
        manifest_csv: Path to manifest CSV file
        batch_size: Batch size
        stack_depth: Number of slices to stack
        image_size: Target image size (H, W) or None for original size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        normalize_method: Normalization method ("scale" or "zscore")
        filter_by_class: Optional list of class numbers to include
        **dataset_kwargs: Additional arguments for MRI25DDataset
    
    Returns:
        DataLoader object
    """
    dataset = MRI25DDataset(
        manifest_csv=manifest_csv,
        stack_depth=stack_depth,
        image_size=image_size,
        normalize_method=normalize_method,
        filter_by_class=filter_by_class,
        **dataset_kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_with_none,  # Use custom collate function
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    manifest_csv = "data/processed/manifest_all.csv"
    
    if not Path(manifest_csv).exists():
        print(f"Error: Manifest file not found: {manifest_csv}")
        print("Please run dicom_converter.py first to generate the manifest.")
        sys.exit(1)
    
    print("="*80)
    print("Testing MRI25DDataset")
    print("="*80)
    
    # Create dataset
    dataset = MRI25DDataset(
        manifest_csv=manifest_csv,
        stack_depth=5,
        image_size=(256, 256),
        normalize_method="scale",
        has_masks=True,
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    
    # Test loading a few samples
    print("\nTesting sample loading:")
    for i in range(min(3, len(dataset))):
        image, mask = dataset[i]
        info = dataset.get_sample_info(i)
        
        print(f"\nSample {i}:")
        print(f"  Case: {info['case_id']}, Series: {info['series_uid'][:20]}...")
        print(f"  Central slice: {info['central_slice_idx']}")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        if mask is not None:
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"  Mask coverage: {mask.sum() / mask.numel() * 100:.2f}%")
        else:
            print(f"  Mask: None (no labels)")
    
    # Test DataLoader
    print("\n" + "="*80)
    print("Testing DataLoader")
    print("="*80)
    
    dataloader = create_dataloader(
        manifest_csv=manifest_csv,
        batch_size=4,
        stack_depth=5,
        image_size=(256, 256),
        shuffle=False,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Load one batch
    images, masks = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape} (batch_size, stack_depth, H, W)")
    print(f"  Masks: {masks.shape if masks is not None else 'None'}")
    
    print("\n✓ All tests passed!")
