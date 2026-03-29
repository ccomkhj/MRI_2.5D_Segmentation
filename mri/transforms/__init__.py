"""Transform registries and built-in transform presets."""

from .registry import get_transform, register_transform

# Import built-ins for side-effect registration.
from . import segmentation_2d5  # noqa: F401

__all__ = ["get_transform", "register_transform"]
