"""MONAI classification model builders (3D)."""

from __future__ import annotations

from typing import Any

from mri.models.registry import filter_model_kwargs, register_classification_model

try:
    from monai.networks.nets import ViT
except Exception:  # pragma: no cover - optional dependency
    ViT = None

try:
    from monai.networks.nets import DenseNet
except Exception:  # pragma: no cover - optional dependency
    DenseNet = None

try:
    from monai.networks.nets import EfficientNetBN
except Exception:  # pragma: no cover - optional dependency
    EfficientNetBN = None

try:
    from monai.networks.nets import SwinTransformer
except Exception:  # pragma: no cover - optional dependency
    SwinTransformer = None

try:
    from monai.networks.nets import resnet as monai_resnet
except Exception:  # pragma: no cover - optional dependency
    monai_resnet = None


@register_classification_model("swin")
def build_swin(**kwargs: Any):
    if SwinTransformer is None:
        raise ImportError(
            "MONAI SwinTransformer is not available in this environment/version. "
            "Use vit, densenet121, efficientnetb7, or resnet101 instead."
        )
    return SwinTransformer(**filter_model_kwargs(SwinTransformer, kwargs, "swin"))


@register_classification_model("vit")
def build_vit(**kwargs: Any):
    if ViT is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'vit'.")
    return ViT(**filter_model_kwargs(ViT, kwargs, "vit"))


@register_classification_model("resnet101")
def build_resnet101(**kwargs: Any):
    if monai_resnet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'resnet101'.")
    if hasattr(monai_resnet, "resnet101"):
        fn = monai_resnet.resnet101
        target = getattr(monai_resnet, "ResNet", fn)
        return fn(**filter_model_kwargs(target, kwargs, "resnet101"))
    raise AttributeError("MONAI resnet101 not available")


@register_classification_model("resnext101")
def build_resnext101(**kwargs: Any):
    if monai_resnet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'resnext101'.")
    fn = getattr(monai_resnet, "resnext101_32x8d", None) or getattr(monai_resnet, "resnext101", None)
    if fn is None:
        raise AttributeError("MONAI resnext101 not available")
    target = getattr(monai_resnet, "ResNet", fn)
    return fn(**filter_model_kwargs(target, kwargs, "resnext101"))


@register_classification_model("densenet121")
def build_densenet121(**kwargs: Any):
    if DenseNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'densenet121'.")
    if "out_channels" not in kwargs and "num_classes" in kwargs:
        kwargs["out_channels"] = kwargs["num_classes"]
    return DenseNet(**filter_model_kwargs(DenseNet, kwargs, "densenet121"))


@register_classification_model("efficientnetb7")
def build_efficientnetb7(**kwargs: Any):
    if EfficientNetBN is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'efficientnetb7'.")
    kwargs.setdefault("model_name", "efficientnet-b7")
    return EfficientNetBN(**filter_model_kwargs(EfficientNetBN, kwargs, "efficientnetb7"))
