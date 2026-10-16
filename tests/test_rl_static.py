"""Tests for StaticRLDesigner (needs numpy -> run in the WSL venv)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from designer.interface import DesignerResult
from designer.rl_static import StaticRLDesigner
from episode.outcomes import Outcome


def _result(vector, reward, suite="mixed_v0"):
    return DesignerResult(vector=vector, context={"suite": suite}, reward=reward,
                          outcome=Outcome.SUBTASKS_DONE_FLAG_UNREAD, usable=True, episode_id="t")


def test_positive_reward_raises_action_probability():
    d = StaticRLDesigner(seed=0)
    ctx = {"suite": "mixed_v0"}
    feats = d.phi(ctx)
    a = [0, 0, 0, 0, 0, 0, 0]
    p_before = d._probs(feats)[0][a[0]]
    for _ in range(40):
        d.update(_result(a, reward=1.0))
    p_after = d._probs(feats)[0][a[0]]
    assert p_after > p_before  # rewarded action became more likely


def test_negative_reward_lowers_action_probability():
    d = StaticRLDesigner(seed=0)
    ctx = {"suite": "mixed_v0"}
    feats = d.phi(ctx)
    a = [2, 3, 3, 3, 2, 3, 1]  # each index valid for its knob's level count
    p_before = d._probs(feats)[0][a[0]]
    for _ in range(40):
        d.update(_result(a, reward=-1.0))
    assert d._probs(feats)[0][a[0]] < p_before


def test_contexts_can_learn_different_preferences():
    # each suite is rewarded for a DIFFERENT level on knob 0; the policy should
    # end up preferring the matching level in each context (a shared bias term
    # means some transfer, but the suite-specific feature must dominate)
    d = StaticRLDesigner(seed=0)
    for _ in range(80):
        d.update(_result([0, 1, 1, 1, 1, 1, 1], reward=1.0, suite="mixed_v0"))
        d.update(_result([2, 1, 1, 1, 1, 1, 1], reward=1.0, suite="light_v0"))
    assert int(d._probs(d.phi({"suite": "mixed_v0"}))[0].argmax()) == 0
    assert int(d._probs(d.phi({"suite": "light_v0"}))[0].argmax()) == 2


def test_unusable_results_are_ignored():
    d = StaticRLDesigner(seed=0)
    r = _result([0, 0, 0, 0, 0, 0, 0], reward=1.0)
    r.usable = False
    before = [W.copy() for W in d.W]
    d.update(r)
    assert all((b == a).all() for b, a in zip(before, d.W))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_rl_static: all passed")
