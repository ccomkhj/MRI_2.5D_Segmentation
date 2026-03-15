"""MONAI segmentation model builders."""

from __future__ import annotations

from typing import Any

from mri.models.registry import filter_model_kwargs, register_segmentation_model

try:
    from monai.networks.nets import DynUNet, SegResNet, UNet, VNet
except Exception:  # pragma: no cover - optional dependency
    DynUNet = SegResNet = UNet = VNet = None

try:
    from .simple_unet import SimpleUNet
except Exception:  # pragma: no cover
    SimpleUNet = None


@register_segmentation_model("simple_unet")
def build_simple_unet(**kwargs: Any):
    if SimpleUNet is None:
        raise ImportError("SimpleUNet not available")
    return SimpleUNet(**filter_model_kwargs(SimpleUNet, kwargs, "simple_unet"))


@register_segmentation_model("dynunet")
def build_dynunet(**kwargs: Any):
    if DynUNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'dynunet'.")
    return DynUNet(**filter_model_kwargs(DynUNet, kwargs, "dynunet"))


@register_segmentation_model("segresnet")
def build_segresnet(**kwargs: Any):
    if SegResNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'segresnet'.")
    return SegResNet(**filter_model_kwargs(SegResNet, kwargs, "segresnet"))


@register_segmentation_model("unet")
def build_unet(**kwargs: Any):
    if UNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'unet'.")
    return UNet(**filter_model_kwargs(UNet, kwargs, "unet"))


@register_segmentation_model("vnet")
def build_vnet(**kwargs: Any):
    if VNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'vnet'.")
    return VNet(**filter_model_kwargs(VNet, kwargs, "vnet"))
