"""MONAI segmentation model builders."""

from __future__ import annotations

from typing import Any
import warnings

from mri.models.registry import filter_model_kwargs, register_segmentation_model

try:
    from monai.networks.nets import DynUNet, SegResNet, UNet, VNet
except Exception:  # pragma: no cover - optional dependency
    DynUNet = SegResNet = UNet = VNet = None

try:
    from .simple_unet import SimpleUNet
except Exception:  # pragma: no cover
    SimpleUNet = None

from .calibration import LogitCalibrationWrapper


def _extract_calibration_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    calibration_keys = {
        "logit_temperature_init",
        "learn_logit_temperature",
        "logit_bias_init",
        "learn_logit_bias",
    }
    return {key: kwargs.pop(key) for key in list(kwargs) if key in calibration_keys}


def _wrap_with_logit_calibration(model, *, model_name: str, kwargs: dict[str, Any], calibration_cfg: dict[str, Any]):
    if not calibration_cfg:
        return model

    out_channels = kwargs.get("out_channels")
    if out_channels is None:
        raise ValueError(f"Model '{model_name}' requires out_channels to enable logit calibration")

    learn_temperature = bool(calibration_cfg.get("learn_logit_temperature", False))
    learn_bias = bool(calibration_cfg.get("learn_logit_bias", False))
    temperature_init = float(calibration_cfg.get("logit_temperature_init", 1.0))
    bias_init = calibration_cfg.get("logit_bias_init", 0.0)

    if not learn_temperature and temperature_init == 1.0 and not learn_bias and bias_init == 0.0:
        warnings.warn(
            f"Calibration params for model '{model_name}' do not change logits; returning base model",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    return LogitCalibrationWrapper(
        model,
        out_channels=int(out_channels),
        logit_temperature_init=temperature_init,
        learn_logit_temperature=learn_temperature,
        logit_bias_init=bias_init,
        learn_logit_bias=learn_bias,
    )


@register_segmentation_model("simple_unet")
def build_simple_unet(**kwargs: Any):
    if SimpleUNet is None:
        raise ImportError("SimpleUNet not available")
    build_kwargs = dict(kwargs)
    calibration_cfg = _extract_calibration_kwargs(build_kwargs)
    model = SimpleUNet(**filter_model_kwargs(SimpleUNet, build_kwargs, "simple_unet"))
    return _wrap_with_logit_calibration(model, model_name="simple_unet", kwargs=build_kwargs, calibration_cfg=calibration_cfg)


@register_segmentation_model("dynunet")
def build_dynunet(**kwargs: Any):
    if DynUNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'dynunet'.")
    build_kwargs = dict(kwargs)
    calibration_cfg = _extract_calibration_kwargs(build_kwargs)
    model = DynUNet(**filter_model_kwargs(DynUNet, build_kwargs, "dynunet"))
    return _wrap_with_logit_calibration(model, model_name="dynunet", kwargs=build_kwargs, calibration_cfg=calibration_cfg)


@register_segmentation_model("segresnet")
def build_segresnet(**kwargs: Any):
    if SegResNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'segresnet'.")
    build_kwargs = dict(kwargs)
    calibration_cfg = _extract_calibration_kwargs(build_kwargs)
    model = SegResNet(**filter_model_kwargs(SegResNet, build_kwargs, "segresnet"))
    return _wrap_with_logit_calibration(model, model_name="segresnet", kwargs=build_kwargs, calibration_cfg=calibration_cfg)


@register_segmentation_model("unet")
def build_unet(**kwargs: Any):
    if UNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'unet'.")
    build_kwargs = dict(kwargs)
    calibration_cfg = _extract_calibration_kwargs(build_kwargs)
    model = UNet(**filter_model_kwargs(UNet, build_kwargs, "unet"))
    return _wrap_with_logit_calibration(model, model_name="unet", kwargs=build_kwargs, calibration_cfg=calibration_cfg)


@register_segmentation_model("vnet")
def build_vnet(**kwargs: Any):
    if VNet is None:
        raise ImportError("MONAI not installed. Install dependencies from requirements.txt to use 'vnet'.")
    build_kwargs = dict(kwargs)
    calibration_cfg = _extract_calibration_kwargs(build_kwargs)
    model = VNet(**filter_model_kwargs(VNet, build_kwargs, "vnet"))
    return _wrap_with_logit_calibration(model, model_name="vnet", kwargs=build_kwargs, calibration_cfg=calibration_cfg)
