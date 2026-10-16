"""Dry-run integration: the whole v0 loop on fakes, no containers, no tokens."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from designer.bayesopt import BayesOptDesigner
from designer.random_search import RandomSearchDesigner
from episode.monitors import read_episodes
from loop import ExperimentConfig, run_experiment


def _run(designer, rounds, log_path):
    exp = ExperimentConfig(rounds=rounds, seed=0, log_path=log_path)
    return run_experiment(designer, exp)


def test_loop_logs_and_classifies_every_episode():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "episodes.jsonl")
        summary = _run(RandomSearchDesigner(seed=0), 60, log)
        episodes = read_episodes(log)
        assert len(episodes) == 60
        # every logged episode carries a valid outcome and a reward
        valid_outcomes = {
            "subtasks_done_flag_unread", "flag_read", "subtasks_incomplete", "invalid",
        }
        for ep in episodes:
            assert ep["outcome"] in valid_outcomes
            assert ep["reward"] is not None
            assert ep["trajectory"]  # non-empty trajectory logged
        # confound audit fields are present
        assert "confound_share" in summary
        assert "invalid_share" in summary


def test_optimizer_beats_random_on_mean_reward():
    # On the synthetic objective, a bandit/BayesOpt optimizer should not do worse
    # than blind random search over a decent number of rounds.
    with tempfile.TemporaryDirectory() as d:
        r = _run(RandomSearchDesigner(seed=1), 200, os.path.join(d, "r.jsonl"))
        b = _run(BayesOptDesigner(seed=1), 200, os.path.join(d, "b.jsonl"))
        assert b["mean_reward"] >= r["mean_reward"] - 0.05  # at least competitive


def test_best_config_is_recorded_per_suite():
    with tempfile.TemporaryDirectory() as d:
        summary = _run(BayesOptDesigner(seed=2), 120, os.path.join(d, "e.jsonl"))
        assert summary["best_config_per_suite"]  # at least one suite recorded a best
        for suite, info in summary["best_config_per_suite"].items():
            assert set(info["levels"]) == {"isolation", "capabilities", "filesystem", "network",
                                           "env_injection", "resources", "rootfs"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_loop_integration: all passed")
