"""A tiny predicate DSL so channel/task preconditions are declarative and inspectable.

A Predicate is a callable `config_dict -> bool` that also knows how to describe itself
and report which knobs it reads. Build them from `eq`, `in_`, combine with `&`, `|`, `~`:

    vault_open = eq("filesystem", "bind_rw") & in_("capabilities", {"chown", "broad"})

Because preconditions are data (not buried in if-ladders) we can print a scenario, and --
crucially -- enumerate the config space and know *exactly* which channels any config
leaves open. That exactness is the ground truth for both scoring and the validator.
"""

from __future__ import annotations

from typing import Iterable, Mapping


class Predicate:
    """Boolean function of a config dict, with introspection."""

    def __call__(self, cfg: Mapping[str, str]) -> bool:
        raise NotImplementedError

    def knobs(self) -> set[str]:
        """Knobs this predicate reads -- used to check a coupling spans >= 2 knobs."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    # combinators
    def __and__(self, other: "Predicate") -> "Predicate":
        return And(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return Or(self, other)

    def __invert__(self) -> "Predicate":
        return Not(self)

    def __repr__(self) -> str:
        return self.describe()


class Eq(Predicate):
    def __init__(self, knob: str, value: str):
        self.knob = knob
        self.value = value

    def __call__(self, cfg):
        return cfg[self.knob] == self.value

    def knobs(self):
        return {self.knob}

    def describe(self):
        return f"{self.knob}={self.value}"


class In(Predicate):
    def __init__(self, knob: str, values: Iterable[str]):
        self.knob = knob
        self.values = frozenset(values)

    def __call__(self, cfg):
        return cfg[self.knob] in self.values

    def knobs(self):
        return {self.knob}

    def describe(self):
        return f"{self.knob} in {{{','.join(sorted(self.values))}}}"


class And(Predicate):
    def __init__(self, *parts: Predicate):
        self.parts = parts

    def __call__(self, cfg):
        return all(p(cfg) for p in self.parts)

    def knobs(self):
        out: set[str] = set()
        for p in self.parts:
            out |= p.knobs()
        return out

    def describe(self):
        return "(" + " AND ".join(p.describe() for p in self.parts) + ")"


class Or(Predicate):
    def __init__(self, *parts: Predicate):
        self.parts = parts

    def __call__(self, cfg):
        return any(p(cfg) for p in self.parts)

    def knobs(self):
        out: set[str] = set()
        for p in self.parts:
            out |= p.knobs()
        return out

    def describe(self):
        return "(" + " OR ".join(p.describe() for p in self.parts) + ")"


class Not(Predicate):
    def __init__(self, inner: Predicate):
        self.inner = inner

    def __call__(self, cfg):
        return not self.inner(cfg)

    def knobs(self):
        return self.inner.knobs()

    def describe(self):
        return f"NOT {self.inner.describe()}"


class Const(Predicate):
    def __init__(self, value: bool):
        self.value = value

    def __call__(self, cfg):
        return self.value

    def knobs(self):
        return set()

    def describe(self):
        return str(self.value)


# convenience constructors
def eq(knob: str, value: str) -> Predicate:
    return Eq(knob, value)


def in_(knob: str, values: Iterable[str]) -> Predicate:
    return In(knob, values)


def const(value: bool) -> Predicate:
    return Const(value)
