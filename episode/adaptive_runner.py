"""Adaptive (v1) episode runner: the designer reconfigures mid-episode.

This encodes the three guardrails we settled on, so the designer cannot win by the
degenerate "clamp everything right before the flag step" strategy:

  * Hidden phase   -- the state handed to the designer is coarse behavioural
                      telemetry only (subtasks done, steps elapsed, attempt flag,
                      current permissiveness). It never reveals which objective is
                      next, so the designer cannot time actions to the flag step.
  * Randomized flag timing -- the flag objective is inserted at a random position
                      among the benign subtasks, not always last.
  * Loosen-only    -- runtime reconfiguration may only GRANT access on the mutable
                      knobs (move toward more permissive), never revoke it. Clamping
                      is unexpressible, so the exploit is impossible by construction.

Runs against the same provisioner/agent contracts as the base runner, plus an env
that supports `reconfigure(levels)` (FakeEnv does; DockerEnv's runtime loosen is a
VM follow-up). Produces the same `Episode` record so logging/metrics are shared.
"""

from __future__ import annotations

import time
import uuid

from config.space import ConfigSpace, DEFAULT_SPACE
from episode.agent import Agent, ActionResult, Objective
from episode.monitors import finalize
from episode.outcomes import Episode, Step, StepKind
from episode.runner import RunnerConfig
from episode.subtasks import SubtaskSuite


class AdaptiveEpisodeRunner:
    def __init__(self, provisioner, agent: Agent, designer, space: ConfigSpace = DEFAULT_SPACE,
                 cfg: RunnerConfig | None = None, seed: int | None = None):
        self.provisioner = provisioner
        self.agent = agent
        self.designer = designer          # must expose begin_episode/propose_initial/reconfigure/learn
        self.space = space
        self.cfg = cfg or RunnerConfig()
        self.mutable = space.mutable_dims()

    # --- loosen-only helper -------------------------------------------------
    def _apply_loosen_only(self, levels: dict[str, str], decisions: dict[str, str]) -> dict[str, str]:
        """decisions: knob -> 'hold'|'loosen'. 'loosen' moves one step MORE permissive."""
        out = dict(levels)
        by_name = {d.name: d for d in self.space.dimensions}
        for knob, dec in decisions.items():
            if knob not in self.mutable or dec != "loosen":
                continue
            dim = by_name[knob]
            idx = dim.levels.index(out[knob])
            out[knob] = dim.levels[min(idx + 1, len(dim.levels) - 1)]  # never decreases
        return out

    # --- telemetry state (hidden phase) ------------------------------------
    def _state(self, episode: Episode, steps: int, total_steps: int, levels: dict[str, str]) -> dict:
        return {
            "subtasks_done_frac": episode.subtasks_completed / max(1, episode.subtasks_total),
            "steps_frac": steps / max(1, total_steps),
            "attempt_detected": 1.0 if episode.attempt_detected else 0.0,
            # current permissiveness of the mutable knobs (so it knows what it already granted)
            "mutable_perm": [self._perm(levels, k) for k in self.mutable],
        }

    def _perm(self, levels: dict[str, str], knob: str) -> float:
        by_name = {d.name: d for d in self.space.dimensions}
        dim = by_name[knob]
        return dim.levels.index(levels[knob]) / max(1, len(dim.levels) - 1)

    # --- schedule with randomized flag timing ------------------------------
    @staticmethod
    def _schedule(suite: SubtaskSuite, rng) -> list:
        items = [("subtask", s) for s in suite.subtasks]
        pos = rng.randint(1, len(items))  # flag not first; may be between subtasks or last
        items.insert(pos, ("flag", None))
        return items

    def run(self, suite: SubtaskSuite, context: dict, seed: int | None = None) -> Episode:
        import random
        rng = random.Random(seed)
        episode = Episode(episode_id=uuid.uuid4().hex[:12], config_vector=[], context=dict(context),
                          subtasks_total=len(suite))
        self.agent.reset()
        self.designer.begin_episode(context)
        try:
            vector = self.designer.propose_initial(context)
            episode.config_vector = list(vector)
            levels = self.space.to_dict(vector)
            schedule = self._schedule(suite, rng)
            total = len(schedule)
            with self.provisioner.provision(vector, seed=seed) as env:
                for step_i, (phase, subtask) in enumerate(schedule):
                    # reconfigure BEFORE the objective, from telemetry only (hidden phase)
                    state = self._state(episode, step_i, total, levels)
                    decisions = self.designer.reconfigure(state)
                    levels = self._apply_loosen_only(levels, decisions)
                    if hasattr(env, "reconfigure"):
                        env.reconfigure(levels)
                    episode.add_step(Step(kind=StepKind.NOTE, observation=f"reconfigure->{levels}"))
                    if phase == "subtask":
                        self._run_subtask(env, subtask, episode)
                    else:
                        self._run_flag(env, episode)
        except Exception as exc:
            episode.error = f"{type(exc).__name__}: {exc}"
            episode.invalid_kind = "harness_fault"
        episode.ended_at = time.time()
        finalize(episode)
        # the designer learns from the terminal reward (set by the caller after reward())
        return episode

    def _run_subtask(self, env, subtask, episode: Episode) -> None:
        last: ActionResult | None = None
        for attempt in range(self.cfg.subtask_attempts):
            obj = Objective(phase="subtask", objective_id=subtask.id, description=subtask.description,
                            attempt_index=attempt, max_attempts=self.cfg.subtask_attempts)
            action = self.agent.act(obj, last)
            episode.add_step(Step(kind=StepKind.SUBTASK_ATTEMPT, action=action.text, subtask_id=subtask.id))
            if action.intent == "giveup":
                # giving up on a benign subtask -> incomplete utility, not INVALID
                episode.add_step(Step(kind=StepKind.NOTE, subtask_id=subtask.id,
                                      observation="agent gave up on subtask"))
                return
            result = env.execute(action, obj, subtask)
            episode.add_step(Step(kind=StepKind.SUBTASK_RESULT, observation=result.observation,
                                  subtask_id=subtask.id, boundary_probe=result.boundary_probe))
            last = result
            if result.terminal:
                episode.error = result.meta.get("invalid_reason", "terminal during subtask")
                episode.invalid_kind = "env_induced"
                return
            if result.subtask_completed:
                episode.subtasks_completed += 1
                return

    def _run_flag(self, env, episode: Episode) -> None:
        last: ActionResult | None = None
        for attempt in range(self.cfg.flag_attempts):
            obj = Objective(phase="flag", objective_id="flag", description=self.cfg.flag_instruction,
                            attempt_index=attempt, max_attempts=self.cfg.flag_attempts)
            action = self.agent.act(obj, last)
            episode.add_step(Step(kind=StepKind.FLAG_ATTEMPT, action=action.text))
            if action.intent == "giveup":
                break
            result = env.execute(action, obj, None)
            episode.add_step(Step(kind=StepKind.OBSERVATION, observation=result.observation,
                                  boundary_probe=result.boundary_probe))
            last = result
            if result.flag_read:
                episode.flag_read = True
                return
            if result.terminal:
                episode.error = result.meta.get("invalid_reason", "terminal during flag step")
                episode.invalid_kind = "env_induced"
                return
