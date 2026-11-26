"""Name -> component registries. Every swappable piece of the pipeline
(transfer function, forward model, bias, likelihood, integrator, selection,
Gibbs block) is looked up here by the name given in the config, so new
components are added by registering them, without touching the pipeline."""
from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any = None) -> Callable | Any:
        """Use as ``REG.register("name", obj)`` or as a decorator."""
        if obj is not None:
            self._items[name] = obj
            return obj

        def deco(o):
            self._items[name] = o
            return o

        return deco

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} '{name}'; available: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
