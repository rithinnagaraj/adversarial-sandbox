"""The designer (adversary) interface and shared result record.

A designer proposes a container configuration and is told, afterwards, how the
episode went. v0 designers condition on the task-suite id (the `context`); the
v1 RL designer additionally conditions on partial-trajectory handles -- both fit
the same `propose(context) -> vector` / `update(result)` contract.

`DesignerResult` is the compact feedback the designer learns from; it is derived
from the full `Episode` record but kept small so designers don't depend on the
whole trajectory schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from episode.outcomes import Episode, Outcome


@dataclass
class DesignerResult:
    vector: list[int]
    context: dict[str, Any]
    reward: float
    outcome: Outcome
    usable: bool                 # False for INVALID episodes
    episode_id: str

    @classmethod
    def from_episode(cls, episode: Episode, reward: float, usable: bool) -> "DesignerResult":
        return cls(
            vector=list(episode.config_vector),
            context=dict(episode.context),
            reward=reward,
            outcome=episode.outcome,
            usable=usable,
            episode_id=episode.episode_id,
        )


class Designer(ABC):
    """Proposes configs and learns from results."""

    name: str = "designer"

    @abstractmethod
    def propose(self, context: dict[str, Any]) -> list[int]:
        """Return a config vector for the given context (e.g. {'suite': 'mixed_v0'})."""
        raise NotImplementedError

    @abstractmethod
    def update(self, result: DesignerResult) -> None:
        """Incorporate feedback from one episode."""
        raise NotImplementedError
