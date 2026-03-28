"""Task implementations."""

from __future__ import annotations

__all__ = ["ClassificationTask", "SegmentationTask"]


def __getattr__(name: str):
    if name == "ClassificationTask":
        from .classification import ClassificationTask

        return ClassificationTask
    if name == "SegmentationTask":
        from .segmentation import SegmentationTask

        return SegmentationTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
