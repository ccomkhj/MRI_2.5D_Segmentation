"""Built-in transforms for 2.5D segmentation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Callable

import numpy as np

from .registry import register_transform


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if random.random() < self.prob:
            image = np.flip(image, axis=-1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-1).copy()
        return image, mask


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if random.random() < self.prob:
            image = np.flip(image, axis=-2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-2).copy()
        return image, mask


class RandomRotate90:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if random.random() < self.prob:
            k = random.randint(1, 3)
            image = np.rot90(image, k=k, axes=(-2, -1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(-2, -1)).copy()
        return image, mask


class RandomIntensityScale:
    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        scale = random.uniform(*self.scale_range)
        image = np.clip(image * scale, 0.0, 255.0).astype(np.float32, copy=False)
        return image, mask


class RandomIntensityShift:
    def __init__(self, shift_range: tuple[float, float] = (-16.0, 16.0)) -> None:
        self.shift_range = (float(shift_range[0]), float(shift_range[1]))

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        shift = random.uniform(*self.shift_range)
        image = np.clip(image + shift, 0.0, 255.0).astype(np.float32, copy=False)
        return image, mask


class RandomGaussianNoise:
    def __init__(self, noise_std: float = 4.0, prob: float = 0.5) -> None:
        self.noise_std = float(noise_std)
        self.prob = float(prob)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if random.random() < self.prob:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 255.0).astype(np.float32, copy=False)
        return image, mask


@register_transform("segmentation_2d5_geometric")
def build_segmentation_2d5_geometric(
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
) -> Compose:
    transforms = []
    if horizontal_flip_prob > 0:
        transforms.append(RandomHorizontalFlip(prob=horizontal_flip_prob))
    if vertical_flip_prob > 0:
        transforms.append(RandomVerticalFlip(prob=vertical_flip_prob))
    if rotate90_prob > 0:
        transforms.append(RandomRotate90(prob=rotate90_prob))
    return Compose(transforms)


@register_transform("segmentation_2d5_basic")
def build_segmentation_2d5_basic(
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
    intensity_scale: bool = True,
    intensity_shift: bool = True,
    gaussian_noise: bool = True,
    scale_range: Sequence[float] = (0.9, 1.1),
    shift_range: Sequence[float] = (-16.0, 16.0),
    noise_std: float = 4.0,
    noise_prob: float = 0.5,
) -> Compose:
    transforms = list(
        build_segmentation_2d5_geometric(
            horizontal_flip_prob=horizontal_flip_prob,
            vertical_flip_prob=vertical_flip_prob,
            rotate90_prob=rotate90_prob,
        ).transforms
    )
    if intensity_scale:
        transforms.append(RandomIntensityScale(scale_range=(float(scale_range[0]), float(scale_range[1]))))
    if intensity_shift:
        transforms.append(RandomIntensityShift(shift_range=(float(shift_range[0]), float(shift_range[1]))))
    if gaussian_noise:
        transforms.append(RandomGaussianNoise(noise_std=noise_std, prob=noise_prob))
    return Compose(transforms)
