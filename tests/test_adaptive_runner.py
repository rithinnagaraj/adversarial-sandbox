"""Tests for the adaptive runner's guarantees (host-safe: no numpy, a stub designer)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.provisioner import FakeProvisioner
from config.space import DEFAULT_SPACE
from episode.adaptive_runner import AdaptiveEpisodeRunner
from episode.agent import FakeAgent
from episode.subtasks import get_suite


class _StubDesigner:
    """Starts restrictive and always asks to loosen; records what it was shown."""
    name = "stub"

    def __init__(self):
        self.states_seen = []

    def begin_episode(self, context):
        pass

    def propose_initial(self, context):
        return [0, 0, 0, 0, 0, 1, 0]  # mutable knobs start low so loosening is visible

    def reconfigure(self, state):
        self.states_seen.append(state)
        return {"filesystem": "loosen", "network": "loosen", "resources": "loosen"}

    def learn(self, episode_return):
        pass


def _runner(designer):
    return AdaptiveEpisodeRunner(FakeProvisioner(DEFAULT_SPACE), FakeAgent(), designer, DEFAULT_SPACE)


def test_loosen_only_never_decreases_a_knob():
    r = _runner(_StubDesigner())
    levels = {"isolation": "gvisor", "capabilities": "minimal", "filesystem": "project",
              "network": "allowlist", "env_injection": "minimal", "resources": "moderate", "rootfs": "readonly"}
    # even if asked to loosen, indices only increase (toward permissive), never drop
    out = r._apply_loosen_only(levels, {"filesystem": "loosen", "network": "loosen"})
    assert out["filesystem"] == "shared_host" and out["network"] == "open"
    # an immutable knob is never touched even if named
    out2 = r._apply_loosen_only(levels, {"isolation": "loosen"})
    assert out2["isolation"] == "gvisor"


def test_flag_is_never_scheduled_first():
    rng = random.Random(0)
    suite = get_suite("mixed_v0")
    for _ in range(50):
        sched = AdaptiveEpisodeRunner._schedule(suite, rng)
        assert sched[0][0] == "subtask"            # flag never first
        assert sum(1 for p, _ in sched if p == "flag") == 1


def test_designer_state_hides_phase():
    d = _StubDesigner()
    _runner(d).run(get_suite("mixed_v0"), {"suite": "mixed_v0"}, seed=1)
    assert d.states_seen, "reconfigure should have been called"
    for s in d.states_seen:
        # telemetry only -- nothing revealing which objective is next
        assert set(s) == {"subtasks_done_frac", "steps_frac", "attempt_detected", "mutable_perm"}


def test_episode_runs_and_is_classified():
    ep = _runner(_StubDesigner()).run(get_suite("mixed_v0"), {"suite": "mixed_v0"}, seed=2)
    assert ep.outcome is not None and ep.subtasks_total == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_adaptive_runner: all passed")
