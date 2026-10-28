"""Scenario = a config space + benign tasks + breach channels, all coupled by predicates.

- A **Task** is benign work the agent should complete. `requires` is the predicate the
  config must satisfy for the task to be *doable at all* (e.g. needs a writable dir and
  network). Its `weight` sets how much of the utility it is worth. `steps` documents the
  multi-step nature for the real (LLM) runner; the oracle only reads `requires`.
- A **Channel** is a misconfiguration-exposure leak path. `exposes_when` is the predicate
  over the config that leaves the channel's secret reachable. Families group them.

Coupling lives entirely in the predicates: when a Task.requires and a Channel.exposes_when
reference overlapping knobs, satisfying the task and closing the channel compete.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from coupled_env.predicates import Predicate
from coupled_env.space import ConfigSpace


class Task:
    def __init__(self, id: str, description: str, requires: Predicate,
                 weight: float = 1.0, steps: Sequence[str] = ()):
        self.id = id
        self.description = description
        self.requires = requires
        self.weight = weight
        self.steps = list(steps)

    def satisfiable(self, cfg: Mapping[str, str]) -> bool:
        return self.requires(cfg)


class Channel:
    def __init__(self, id: str, family: str, exposes_when: Predicate, note: str = "",
                 severity: float = 1.0):
        self.id = id
        self.family = family
        self.exposes_when = exposes_when
        self.note = note
        self.severity = severity   # weight in graded containment (higher = worse to leak)

    def is_open(self, cfg: Mapping[str, str]) -> bool:
        return self.exposes_when(cfg)


class Scenario:
    def __init__(self, id: str, space: ConfigSpace, tasks: Sequence[Task],
                 channels: Sequence[Channel], description: str = ""):
        self.id = id
        self.space = space
        self.tasks = list(tasks)
        self.channels = list(channels)
        self.description = description
        self._check_predicate_knobs()

    def _check_predicate_knobs(self) -> None:
        """Fail fast if any predicate references a knob the space doesn't define."""
        valid = set(self.space.names)
        for t in self.tasks:
            bad = t.requires.knobs() - valid
            if bad:
                raise ValueError(f"task {t.id!r} references unknown knob(s) {bad}")
        for c in self.channels:
            bad = c.exposes_when.knobs() - valid
            if bad:
                raise ValueError(f"channel {c.id!r} references unknown knob(s) {bad}")

    # --- evaluation (ground truth from the predicates) ---
    def open_channels(self, cfg: Mapping[str, str]) -> list[Channel]:
        return [c for c in self.channels if c.is_open(cfg)]

    def satisfied_tasks(self, cfg: Mapping[str, str]) -> list[Task]:
        return [t for t in self.tasks if t.satisfiable(cfg)]

    @property
    def total_weight(self) -> float:
        return sum(t.weight for t in self.tasks) or 1.0
