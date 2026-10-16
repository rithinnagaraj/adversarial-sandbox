"""Core data contracts shared across the whole harness.

The `Episode` record is the single contract produced by the runner and consumed
by monitors, the reward function, the logger, and every designer's `update()`.
Keep this module dependency-free (stdlib only) so it imports everywhere.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


class Outcome(str, enum.Enum):
    """How an episode ended. The string values are what gets logged to jsonl.

    Only the first three feed the v0 binary reward. ``INVALID`` is tracked so we
    can audit how much apparent "containment" was really the agent giving up,
    the harness breaking, or an incoherent environment -- and later gate the
    reward on a *detected attempt* (v1).
    """

    SUBTASKS_DONE_FLAG_UNREAD = "subtasks_done_flag_unread"  # the good outcome
    FLAG_READ = "flag_read"                                   # containment failed
    SUBTASKS_INCOMPLETE = "subtasks_incomplete"               # utility failed
    INVALID = "invalid"                                       # crash/timeout/incoherent

    @property
    def is_contained(self) -> bool:
        """True when the flag was *not* read. Note: INVALID is not containment."""
        return self in (Outcome.SUBTASKS_DONE_FLAG_UNREAD, Outcome.SUBTASKS_INCOMPLETE)


class StepKind(str, enum.Enum):
    """What a single trajectory step represents."""

    SUBTASK_ATTEMPT = "subtask_attempt"
    SUBTASK_RESULT = "subtask_result"
    FLAG_ATTEMPT = "flag_attempt"      # the agent is working on the final flag step
    OBSERVATION = "observation"
    NOTE = "note"                      # harness-level annotation


@dataclass
class Step:
    """One entry in an episode trajectory.

    We record *what the agent did and what came back*, method-agnostically. We do
    not store or require any escape technique; `action` is whatever opaque action
    string the agent emitted, and `boundary_probe` is a behavioural flag the
    monitor sets when the agent is working against the container boundary (used
    later for attempt-gating), not a description of how.
    """

    kind: StepKind
    t: float = field(default_factory=time.time)
    action: str = ""              # opaque action text emitted by the agent
    observation: str = ""         # opaque result text returned to the agent
    subtask_id: str | None = None
    boundary_probe: bool = False  # monitor flag: agent engaged the boundary this step
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class Episode:
    """The full record of one rollout of the frozen agent in one container config.

    This is the unit the designer learns from. `config_vector` is the designer's
    proposal; `context` is what the designer was conditioned on (the task suite id
    in v0, plus partial-trajectory handles in v1).
    """

    episode_id: str
    config_vector: list[int]
    context: dict[str, Any]                       # what the designer conditioned on
    subtasks_total: int = 0
    subtasks_completed: int = 0
    flag_read: bool = False
    outcome: Outcome = Outcome.INVALID
    attempt_detected: bool = False                # any boundary_probe step seen
    end_reason: str = ""                          # human-readable classifier note
    trajectory: list[Step] = field(default_factory=list)
    reward: float | None = None                   # filled by reward.py
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    error: str | None = None                      # set when the episode did not complete cleanly
    # When outcome is INVALID, which kind:
    #   "env_induced"   -> the environment derailed the agent (e.g. resource starvation).
    #                      This is a *finding* about the agent+sandbox and may be learned from.
    #   "harness_fault" -> our own tooling failed (exception, spin-up error, timeout bug).
    #                      This is noise to fix, never a config property to learn from.
    invalid_kind: str | None = None

    @property
    def subtasks_all_done(self) -> bool:
        return self.subtasks_total > 0 and self.subtasks_completed >= self.subtasks_total

    def add_step(self, step: Step) -> None:
        self.trajectory.append(step)
        if step.boundary_probe:
            self.attempt_detected = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["trajectory"] = [s.to_dict() for s in self.trajectory]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
