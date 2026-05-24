from __future__ import annotations

import importlib


_REGISTRY: dict[str, str] = {}


def register(id: str, entry_point: str, **_kwargs) -> None:
    _REGISTRY[id] = entry_point


def make(id: str, **kwargs):
    if id not in _REGISTRY:
        raise KeyError(f"Environment '{id}' is not registered")
    module_name, attr_name = _REGISTRY[id].split(":", 1)
    module = importlib.import_module(module_name)
    env_cls = getattr(module, attr_name)
    return env_cls(**kwargs)
