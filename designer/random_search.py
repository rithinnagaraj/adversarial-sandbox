"""Random-search designer -- the baseline every learned designer must beat.

It ignores feedback (pure exploration) but keeps a record of the best config seen
per context, which is a surprisingly strong baseline for a small discrete space
and the honest bar for claiming a learned designer "works".
"""

from __future__ import annotations

import random
from typing import Any

from config.space import ConfigSpace, DEFAULT_SPACE
from designer.interface import Designer, DesignerResult


class RandomSearchDesigner(Designer):
    name = "random_search"

    def __init__(self, space: ConfigSpace = DEFAULT_SPACE, seed: int | None = None):
        self.space = space
        self.rng = random.Random(seed)
        # best (reward, vector) seen per context key
        self._best: dict[str, tuple[float, list[int]]] = {}

    @staticmethod
    def _key(context: dict[str, Any]) -> str:
        return str(context.get("suite", "_"))

    def propose(self, context: dict[str, Any]) -> list[int]:
        return self.space.sample(self.rng)

    def update(self, result: DesignerResult) -> None:
        if not result.usable:
            return
        key = self._key(result.context)
        best = self._best.get(key)
        if best is None or result.reward > best[0]:
            self._best[key] = (result.reward, list(result.vector))

    def best_for(self, context: dict[str, Any]) -> list[int] | None:
        found = self._best.get(self._key(context))
        return list(found[1]) if found else None
