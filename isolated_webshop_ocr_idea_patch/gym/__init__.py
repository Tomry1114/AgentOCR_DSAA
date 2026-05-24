from __future__ import annotations

from .envs.registration import make, register


class Env:
    metadata = {}

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        return None


class _Spaces:
    pass


spaces = _Spaces()

__all__ = ["Env", "make", "register", "spaces"]
