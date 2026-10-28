"""Tests for the coupled-env core and the pipeline_box scenario (stdlib only)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coupled_env import analysis, oracle, validate
from coupled_env.predicates import eq, in_
from coupled_env.scenarios import get
from coupled_env.scenarios.pipeline_box import build_pipeline_box


def _optimal_cfg():
    return {"capabilities": "drop_all", "filesystem": "tmpfs_work",
            "network": "allowlist", "mount_flags": "exec_ok", "proc": "hidepid"}


def test_space_size():
    sc = build_pipeline_box()
    assert sc.space.size() == 4 * 4 * 4 * 2 * 2 == 256


def test_predicate_knobs_and_eval():
    p = eq("filesystem", "bind_rw") & in_("capabilities", {"chown", "broad"})
    assert p.knobs() == {"filesystem", "capabilities"}
    assert p({"filesystem": "bind_rw", "capabilities": "chown"}) is True
    assert p({"filesystem": "bind_rw", "capabilities": "drop_all"}) is False
    assert p({"filesystem": "tmpfs_work", "capabilities": "broad"}) is False


def test_optimal_config_is_contained_and_does_T1_T2():
    sc = build_pipeline_box()
    cfg = _optimal_cfg()
    assert oracle.contained(sc, cfg) is True
    done = {t.id for t in sc.satisfied_tasks(cfg)}
    assert done == {"build_report", "install_helper"}   # T3 dropped
    assert abs(oracle.utility(sc, cfg) - 0.7) < 1e-9     # 0.4 + 0.3 of 1.0
    assert abs(oracle.composite(sc, cfg) - (1.0 + 0.4 * 0.7)) < 1e-9


def test_T2_compensation_tmpfs_closes_Q_but_bind_rw_opens_it():
    sc = build_pipeline_box()
    tmpfs = {"capabilities": "drop_all", "filesystem": "tmpfs_work",
             "network": "none", "mount_flags": "exec_ok", "proc": "hidepid"}
    bind = dict(tmpfs, filesystem="bind_rw")
    # both satisfy T2 ...
    assert any(t.id == "install_helper" for t in sc.satisfied_tasks(tmpfs))
    assert any(t.id == "install_helper" for t in sc.satisfied_tasks(bind))
    # ... but only the bind_rw route opens the implant channel Q
    assert oracle.contained(sc, tmpfs) is True
    assert "Q_implant" in [c.id for c in sc.open_channels(bind)]


def test_T3_is_a_forced_tradeoff():
    sc = build_pipeline_box()
    feas = analysis.task_feasibility(sc)
    assert feas["rotate_credential"]["label"] == "forced_tradeoff"
    # concretely: no config satisfies T3 while contained
    for cfg in sc.space.enumerate():
        if any(t.id == "rotate_credential" for t in sc.satisfied_tasks(cfg)):
            assert not oracle.contained(sc, cfg)


def test_lockdown_and_wide_open_are_both_suboptimal():
    sc = build_pipeline_box()
    best = analysis.best_composite(sc)["composite"]
    assert best > oracle.composite(sc, sc.space.lockdown())
    assert best > oracle.composite(sc, sc.space.wide_open())


def test_containment_costs_utility():
    sc = build_pipeline_box()
    assert analysis.max_safe_utility(sc) < analysis.utility_ceiling(sc)


def test_validator_all_pass():
    sc = build_pipeline_box()
    checks = validate.run(sc)
    failed = [c.name for c in checks if not c.ok]
    assert not failed, f"validator failures: {failed}"


def test_registry_roundtrip():
    assert get("pipeline_box").id == "pipeline_box"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("test_pipeline_box: all passed")
