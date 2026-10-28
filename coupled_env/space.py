"""Config space: ordered, named knobs. Levels are listed least-privilege first, so
index 0 of every knob is the most locked-down setting (the "lockdown" config) and the
last index is the most exposed. That ordering is a readability/analysis convenience
(e.g. a lockdown baseline); the environment never assumes the knobs are totally ordered
in any semantic sense -- all preconditions use explicit value sets, not thresholds.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Mapping, Sequence


class Knob:
    def __init__(self, name: str, levels: Sequence[str]):
        if len(levels) != len(set(levels)):
            raise ValueError(f"knob {name!r} has duplicate levels: {levels}")
        self.name = name
        self.levels = list(levels)

    def __len__(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        return f"Knob({self.name}: {'|'.join(self.levels)})"


class ConfigSpace:
    def __init__(self, knobs: Sequence[Knob]):
        names = [k.name for k in knobs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate knob names: {names}")
        self.knobs = list(knobs)

    @property
    def names(self) -> list[str]:
        return [k.name for k in self.knobs]

    def size(self) -> int:
        n = 1
        for k in self.knobs:
            n *= len(k)
        return n

    def validate_config(self, cfg: Mapping[str, str]) -> None:
        for k in self.knobs:
            if k.name not in cfg:
                raise KeyError(f"config missing knob {k.name!r}")
            if cfg[k.name] not in k.levels:
                raise ValueError(f"knob {k.name!r} has illegal level {cfg[k.name]!r}")

    def to_vector(self, cfg: Mapping[str, str]) -> list[int]:
        return [k.levels.index(cfg[k.name]) for k in self.knobs]

    def to_dict(self, vector: Sequence[int]) -> dict[str, str]:
        return {k.name: k.levels[vector[i]] for i, k in enumerate(self.knobs)}

    def enumerate(self) -> Iterator[dict[str, str]]:
        """Yield every config in the space as a dict. Only use on modest spaces."""
        for combo in product(*[k.levels for k in self.knobs]):
            yield {k.name: combo[i] for i, k in enumerate(self.knobs)}

    def lockdown(self) -> dict[str, str]:
        """The most locked-down config (index 0 of every knob)."""
        return {k.name: k.levels[0] for k in self.knobs}

    def wide_open(self) -> dict[str, str]:
        """The most exposed config (last index of every knob)."""
        return {k.name: k.levels[-1] for k in self.knobs}
