"""Model registry for segmentation and classification models."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict
import warnings

SEGMENTATION_MODELS: Dict[str, Callable] = {}
CLASSIFICATION_MODELS: Dict[str, Callable] = {}


def register_segmentation_model(name: str):
    def decorator(fn: Callable):
        SEGMENTATION_MODELS[name] = fn
        return fn
    return decorator


def register_classification_model(name: str):
    def decorator(fn: Callable):
        CLASSIFICATION_MODELS[name] = fn
        return fn
    return decorator


def filter_model_kwargs(factory: Callable[..., Any], kwargs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Drop unsupported kwargs so layered config overrides can switch model families safely."""

    target = factory.__init__ if inspect.isclass(factory) else factory
    signature = inspect.signature(target)
    params = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return dict(kwargs)

    supported = {
        param.name
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    supported.discard("self")

    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(key for key in kwargs if key not in supported)
    if ignored:
        warnings.warn(
            f"Ignoring unsupported params for model '{model_name}': {', '.join(ignored)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return filtered


def create_segmentation_model(name: str, **kwargs):
    if name not in SEGMENTATION_MODELS:
        raise KeyError(f"Unknown segmentation model: {name}")
    return SEGMENTATION_MODELS[name](**kwargs)


def create_classification_model(name: str, **kwargs):
    if name not in CLASSIFICATION_MODELS:
        raise KeyError(f"Unknown classification model: {name}")
    return CLASSIFICATION_MODELS[name](**kwargs)
