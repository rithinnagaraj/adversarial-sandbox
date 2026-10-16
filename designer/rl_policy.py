"""v1 RL designer -- trajectory-conditioned adaptive adversary (interface stub).

This is the "task suite + partial trajectory" designer the user chose, staged for
v1. It is NOT wired into the v0 loop; it documents and fixes the contract so the
later build is additive.

Why RL earns its place here (and not in v0): the designer becomes *sequential and
adaptive* -- it observes partial trajectory handles mid-episode and may toggle the
**runtime-mutable** knobs only (see `ConfigSpace.mutable_dims`: filesystem,
network, resources). Structural knobs (isolation, capabilities) remain fixed at
container creation, an honest physical constraint. That mid-episode decision
process is a sequential problem a contextual bandit/BayesOpt cannot express,
which is the justification for learning a policy rather than searching.

Guardrails to carry into the implementation:
  * The STUDENT (LLM) stays frozen. Only this environment-designer policy learns.
  * Use the attempt-gated reward (`reward.attempt_gated_reward`) here, so the
    policy cannot score by starving/confusing the agent instead of containing it.
"""

from __future__ import annotations

from typing import Any

from config.space import ConfigSpace, DEFAULT_SPACE
from designer.interface import Designer, DesignerResult


class RLDesigner(Designer):
    """Placeholder for the trajectory-conditioned RL designer (built in the VM)."""

    name = "rl_designer"

    def __init__(self, space: ConfigSpace = DEFAULT_SPACE):
        self.space = space
        self.mutable_dims = space.mutable_dims()

    def propose(self, context: dict[str, Any]) -> list[int]:
        raise NotImplementedError(
            "RLDesigner is a v1 stub. Implement a contextual policy that emits an initial "
            "config from the task suite; adaptivity (reconfiguring mutable_dims mid-episode) "
            "attaches via the runner's step hooks during VM integration."
        )

    def update(self, result: DesignerResult) -> None:
        raise NotImplementedError(
            "RLDesigner is a v1 stub. Train on attempt-gated reward; keep the LLM student frozen."
        )
