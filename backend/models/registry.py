import gc
import importlib

import torch

from ..device import DEVICE

# name -> (module, class). Imported lazily so one broken model install doesn't break the rest.
MODELS = {
    "depth_anything": ("depth_anything", "DepthAnythingV2"),
}

_loaded: dict = {}


def cls(name: str):
    module, cls_name = MODELS[name]
    return getattr(importlib.import_module(f".{module}", __package__), cls_name)


def get(name: str):
    if name not in _loaded:
        _loaded[name] = cls(name)(DEVICE)
    return _loaded[name]


def status() -> list[dict]:
    return [
        {"name": n, "loaded": n in _loaded, "device": DEVICE, "scale_kind": cls(n).scale_kind, "version": cls(n).version}
        for n in MODELS
    ]


def unload(name: str | None = None) -> None:
    for n in [name] if name else list(_loaded):
        _loaded.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
