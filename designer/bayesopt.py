"""Black-box-optimization designer baseline.

Because the LLM policy is frozen, the designer optimizes a fixed noisy function
`config -> reward`. For a small discrete space this is exactly where BayesOpt /
bandits shine, and they are the honest bar a learned (RL) designer must beat.

This class is *contextual*: it keeps separate optimizer state per context key
(the task-suite id), matching the v0 "condition on the task suite" design.

Backend selection, degrading gracefully so v0 needs no heavy wheels:
  * If `scikit-optimize` is importable, use a GP/forest `Optimizer` per context.
  * Otherwise fall back to a dependency-free UCB1 bandit over the enumerated
    space (valid because the space is small and enumerable).

Both backends expose the same `propose` / `update` contract.
"""

from __future__ import annotations

import math
import random
from typing import Any

from config.space import ConfigSpace, DEFAULT_SPACE
from designer.interface import Designer, DesignerResult

try:  # optional dependency
    from skopt import Optimizer as _SkOptimizer  # type: ignore
    from skopt.space import Integer as _Integer  # type: ignore
    _HAVE_SKOPT = True
except Exception:  # pragma: no cover - exercised only when skopt is absent
    _HAVE_SKOPT = False


class _UCBBandit:
    """UCB1 over the enumerated discrete space (dependency-free fallback)."""

    def __init__(self, space: ConfigSpace, seed: int | None):
        self.arms = space.enumerate()
        self.rng = random.Random(seed)
        self.counts = [0] * len(self.arms)
        self.means = [0.0] * len(self.arms)
        self.total = 0
        self._index = {tuple(v): i for i, v in enumerate(self.arms)}

    def propose(self) -> list[int]:
        # pull every arm once before using the UCB rule
        for i, c in enumerate(self.counts):
            if c == 0:
                return list(self.arms[i])
        logt = math.log(self.total + 1)
        best_i, best_score = 0, -math.inf
        for i in range(len(self.arms)):
            bonus = math.sqrt(2 * logt / self.counts[i])
            score = self.means[i] + bonus
            if score > best_score:
                best_i, best_score = i, score
        return list(self.arms[best_i])

    def update(self, vector: list[int], reward: float) -> None:
        i = self._index[tuple(vector)]
        self.counts[i] += 1
        self.total += 1
        # reward is in [-1, 1]; map to [0, 1] so UCB bonuses are well-scaled
        r = (reward + 1.0) / 2.0
        self.means[i] += (r - self.means[i]) / self.counts[i]


# skopt surrogate models worth comparing on this discrete/ordinal space.
# GP treats the integer knobs as metric/ordinal; RF/ET (tree ensembles) handle
# the categorical-ish knobs (capabilities, network) more naturally.
SURROGATES = ("GP", "RF", "ET", "GBRT")


class _SkoptBackend:
    """One skopt Optimizer over integer dims (minimizes -> we feed negative reward)."""

    def __init__(self, space: ConfigSpace, seed: int | None, surrogate: str = "GP"):
        if surrogate not in SURROGATES:
            raise ValueError(f"unknown surrogate {surrogate!r}; choices: {SURROGATES}")
        dims = [_Integer(0, len(d.levels) - 1, name=d.name) for d in space.dimensions]
        self.opt = _SkOptimizer(dims, random_state=seed, base_estimator=surrogate)
        self._pending: list[int] | None = None

    def propose(self) -> list[int]:
        self._pending = [int(x) for x in self.opt.ask()]
        return list(self._pending)

    def update(self, vector: list[int], reward: float) -> None:
        self.opt.tell([int(x) for x in vector], -float(reward))  # minimize negative reward


class BayesOptDesigner(Designer):
    def __init__(self, space: ConfigSpace = DEFAULT_SPACE, seed: int | None = None, surrogate: str = "GP"):
        self.space = space
        self.seed = seed
        self.surrogate = surrogate
        # name records the actual backend so comparison runs are self-describing
        self.name = f"bayesopt_{surrogate.lower()}" if _HAVE_SKOPT else "ucb_fallback"
        self._backends: dict[str, Any] = {}
        self._best: dict[str, tuple[float, list[int]]] = {}

    @staticmethod
    def _key(context: dict[str, Any]) -> str:
        return str(context.get("suite", "_"))

    def _backend(self, key: str) -> Any:
        if key not in self._backends:
            self._backends[key] = (
                _SkoptBackend(self.space, self.seed, self.surrogate)
                if _HAVE_SKOPT else _UCBBandit(self.space, self.seed)
            )
        return self._backends[key]

    def propose(self, context: dict[str, Any]) -> list[int]:
        return self._backend(self._key(context)).propose()

    def update(self, result: DesignerResult) -> None:
        if not result.usable:
            return  # don't feed INVALID episodes into the optimizer
        key = self._key(result.context)
        self._backend(key).update(result.vector, result.reward)
        best = self._best.get(key)
        if best is None or result.reward > best[0]:
            self._best[key] = (result.reward, list(result.vector))

    def best_for(self, context: dict[str, Any]) -> list[int] | None:
        found = self._best.get(self._key(context))
        return list(found[1]) if found else None
