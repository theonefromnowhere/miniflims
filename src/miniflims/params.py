"""Nuisance-parameter specifications shared by bias models and likelihoods."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    init: float
    lower: float
    upper: float
    width: float  # initial slice-sampler bracket width
    sample: bool = True
    prior: str = "uniform"  # "uniform" | "normal" (both truncated to bounds)
    prior_mean: float = 0.0
    prior_std: float = 1.0

    def __post_init__(self):
        if not self.lower < self.upper:
            raise ValueError(f"{self.name}: need lower < upper")
        if not self.lower <= self.init <= self.upper:
            raise ValueError(
                f"{self.name}: init {self.init} outside [{self.lower}, {self.upper}]"
            )
        if self.width <= 0:
            raise ValueError(f"{self.name}: width must be positive")
        if self.prior not in ("uniform", "normal"):
            raise ValueError(f"{self.name}: unknown prior '{self.prior}'")
        if self.prior_std <= 0:
            raise ValueError(f"{self.name}: prior_std must be positive")


def apply_override(spec: ParamSpec, override) -> ParamSpec:
    """Merge the non-null fields of a config ``ParamOverride`` into a spec."""
    if override is None:
        return spec
    changes = {
        f.name: getattr(override, f.name)
        for f in dataclasses.fields(override)
        if getattr(override, f.name) is not None
    }
    return dataclasses.replace(spec, **changes)


def resolve_specs(defaults, overrides: dict, owner: str) -> list[ParamSpec]:
    names = {s.name for s in defaults}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(
            f"{owner}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(names)}"
        )
    return [apply_override(s, overrides.get(s.name)) for s in defaults]
