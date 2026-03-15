#!/usr/bin/env python3
"""
Data augmentation transforms for 2.5D MRI segmentation.

Provides both basic PyTorch transforms and MONAI-based transforms
for medical image augmentation.

Usage:
    from tools.dataset.transforms_2d5 import get_train_transforms, get_val_transforms
    
    train_transforms = get_train_transforms(image_size=(256, 256))
    dataset = MRI25DDataset(..., transform=train_transforms)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable
import random


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        if random.random() < self.prob:
            # Flip along width dimension (last dimension)
            image = torch.flip(image, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
        return image, mask


class RandomVerticalFlip:
    """Randomly flip image and mask vertically."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        if random.random() < self.prob:
            # Flip along height dimension (second to last)
            image = torch.flip(image, dims=[-2])
            if mask is not None:
                mask = torch.flip(mask, dims=[-2])
        return image, mask


class RandomRotate90:
    """Randomly rotate image and mask by 90 degrees."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        if random.random() < self.prob:
            # Random number of 90-degree rotations (1, 2, or 3)
            k = random.randint(1, 3)
            # Rotate in the H-W plane (last two dimensions)
            image = torch.rot90(image, k=k, dims=[-2, -1])
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=[-2, -1])
        return image, mask


class RandomIntensityScale:
    """Randomly scale intensity values."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        image = image * scale
        # Clamp to valid range if using scale normalization
        image = torch.clamp(image, 0.0, 1.0)
        return image, mask


class RandomIntensityShift:
    """Randomly shift intensity values."""
    
    def __init__(self, shift_range: Tuple[float, float] = (-0.1, 0.1)):
        self.shift_range = shift_range
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        shift = random.uniform(self.shift_range[0], self.shift_range[1])
        image = image + shift
        # Clamp to valid range if using scale normalization
        image = torch.clamp(image, 0.0, 1.0)
        return image, mask


class RandomGaussianNoise:
    """Add random Gaussian noise."""
    
    def __init__(self, noise_std: float = 0.01, prob: float = 0.5):
        self.noise_std = noise_std
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple:
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)
        return image, mask


class ToTensor:
    """Convert numpy arrays to tensors (if not already)."""
    
    def __call__(self, image, mask=None):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        return image, mask


def get_train_transforms(
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
    intensity_scale: bool = True,
    intensity_shift: bool = True,
    gaussian_noise: bool = True,
    noise_std: float = 0.01,
) -> Compose:
    """
    Get training augmentation transforms.
    
    Args:
        horizontal_flip_prob: Probability of horizontal flip
        vertical_flip_prob: Probability of vertical flip
        rotate90_prob: Probability of 90-degree rotation
        intensity_scale: Whether to apply intensity scaling
        intensity_shift: Whether to apply intensity shift
        gaussian_noise: Whether to add Gaussian noise
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Compose object with transforms
    """
    transforms = []
    
    # Geometric transforms
    if horizontal_flip_prob > 0:
        transforms.append(RandomHorizontalFlip(prob=horizontal_flip_prob))
    
    if vertical_flip_prob > 0:
        transforms.append(RandomVerticalFlip(prob=vertical_flip_prob))
    
    if rotate90_prob > 0:
        transforms.append(RandomRotate90(prob=rotate90_prob))
    
    # Intensity transforms
    if intensity_scale:
        transforms.append(RandomIntensityScale(scale_range=(0.9, 1.1)))
    
    if intensity_shift:
        transforms.append(RandomIntensityShift(shift_range=(-0.1, 0.1)))
    
    if gaussian_noise:
        transforms.append(RandomGaussianNoise(noise_std=noise_std, prob=0.5))
    
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """
    Get validation transforms (no augmentation).
    
    Returns:
        Compose object with no-op transforms
    """
    return Compose([])


# ============================================================================
# MONAI-based transforms (optional, if MONAI is available)
# ============================================================================

try:
    from monai.transforms import (
        Compose as MonaiCompose,
        RandFlipd,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandGaussianNoised,
    )
    
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


def get_monai_transforms(is_training: bool = True):
    """
    Get MONAI-based transforms (if MONAI is available).
    
    Args:
        is_training: Whether to apply training augmentations
    
    Returns:
        MONAI Compose object or None if MONAI not available
    """
    if not HAS_MONAI:
        print("Warning: MONAI not available. Use get_train_transforms() instead.")
        return None
    
    if is_training:
        return MonaiCompose([
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.5),
        ])
    else:
        return MonaiCompose([])


if __name__ == "__main__":
    print("Testing transforms...")
    
    # Create dummy data
    image = torch.rand(5, 256, 256)  # [stack_depth, H, W]
    mask = torch.randint(0, 2, (1, 256, 256)).float()  # [1, H, W]
    
    print(f"Original image shape: {image.shape}")
    print(f"Original mask shape: {mask.shape}")
    
    # Test transforms
    transforms = get_train_transforms()
    
    print("\nApplying transforms...")
    for i in range(3):
        aug_image, aug_mask = transforms(image.clone(), mask.clone())
        print(f"  Sample {i}: image {aug_image.shape}, mask {aug_mask.shape}")
        print(f"    Image range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
        print(f"    Mask unique: {aug_mask.unique().tolist()}")
    
    print("\n✓ Transforms test passed!")
