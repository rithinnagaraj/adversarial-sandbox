"""Tests for the scaled generated scenario and scale-free analysis (stdlib only)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coupled_env import analysis, oracle, validate
from coupled_env.scenarios.datacenter import build_datacenter, build_datacenter_small


def test_datacenter_hits_targets():
    sc = build_datacenter()
    assert sc.space.size() > 1_000_000_000          # > 1e9 configs
    assert len(sc.channels) >= 100                   # 100+ breach paths
    assert all(len(c.exposes_when.knobs()) >= 2 for c in sc.channels)  # all coupled


def test_graded_containment_is_non_binary():
    sc = build_datacenter()
    # a config that opens exactly one channel is partially -- not totally -- contained
    cfg = sc.space.lockdown()
    assert oracle.containment_score(sc, cfg) == 1.0  # nothing open
    cfg_one = dict(cfg, proc="visible", capabilities="broad")  # opens proc_snoop (+? check)
    opens = sc.open_channels(cfg_one)
    assert len(opens) >= 1
    score = oracle.containment_score(sc, cfg_one)
    assert 0.0 < score < 1.0                          # strictly between -> graded, not binary


def test_validator_passes_at_scale():
    sc = build_datacenter()
    checks = validate.run(sc)
    failed = [c.name for c in checks if not c.ok]
    assert not failed, f"validator failures: {failed}"


def test_scalefree_forced_tradeoff_matches_whole_space_truth():
    # The rigorous ground truth for 'forced_tradeoff' is a whole-space property:
    # the task can NEVER be done while strictly contained. On the small instance we can
    # brute-force that and confirm the scale-free verdict agrees exactly.
    # (This equivalence holds because lockdown is the safest setting of the non-task knobs,
    #  so if the task leaks even then, every fuller config leaks too.)
    sc = build_datacenter_small()
    labels = {t: info["label"] for t, info in analysis.task_feasibility(sc).items()}
    configs = list(sc.space.enumerate())
    for t in sc.tasks:
        safe_achievable = any(t.satisfiable(c) and oracle.contained(sc, c) for c in configs)
        is_forced = labels[t.id] == "forced_tradeoff"
        assert is_forced == (not safe_achievable), (
            f"{t.id}: scale-free forced={is_forced} but whole-space safe_achievable={safe_achievable}")
        # tasks labelled free/needs_care must genuinely be safe-achievable somewhere
        if labels[t.id] in ("free", "needs_care"):
            assert safe_achievable


def test_forced_and_compensation_both_present():
    sc = build_datacenter()
    feas = analysis.task_feasibility(sc)
    labels = {info["label"] for info in feas.values()}
    assert "forced_tradeoff" in labels
    assert "needs_care" in labels


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("test_datacenter: all passed")
