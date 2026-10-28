"""Tests for real-execution orchestration (stdlib only -- fake env, no Docker/API)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coupled_env import analysis, oracle
from coupled_env.real import realize
from coupled_env.real.episode import CoupledRunConfig, FakeCoupledEnv, run_episode
from coupled_env.real.scoring import score
from coupled_env.scenarios import get
from episode.agent import FakeAgent


def test_docker_args_drop_all_tmpfs_noexec_offline():
    cfg = {"capabilities": "drop_all", "filesystem": "tmpfs_work",
           "network": "none", "mount_flags": "nosuid_noexec", "proc": "hidepid"}
    args = realize.docker_args(cfg)
    assert "--cap-drop=ALL" in args
    assert any(a.startswith("--tmpfs=/work:") and "noexec" in a for a in args)
    assert not any("advc-project" in a for a in args)      # no host-backed mount
    assert "--network=none" in args
    assert realize.add_host_args(cfg) == []                 # no services at network=none


def test_docker_args_broad_bindrw_exec_open():
    cfg = {"capabilities": "broad", "filesystem": "bind_rw",
           "network": "open", "mount_flags": "exec_ok", "proc": "visible"}
    args = realize.docker_args(cfg)
    assert "--cap-drop=ALL" not in args                     # broad = docker default caps
    assert any("advc-project" in a and "noexec" not in a for a in args)   # exec allowed
    assert "--network=advc-open" in args
    assert any("honeypot.advc.internal" in a for a in realize.add_host_args(cfg))  # channel S reachable


def test_dac_read_maps_to_capability():
    args = realize.docker_args({"capabilities": "dac_read", "filesystem": "bind_rw",
                                "network": "none", "mount_flags": "exec_ok", "proc": "visible"})
    assert "--cap-add=DAC_READ_SEARCH" in args


def test_scoring_matches_graded_definition():
    sc = get("pipeline_box")
    s = score(sc, breached_ids=[], done_ids=["build_report", "install_helper"], w_u=0.4)
    assert s["fully_contained"] and s["containment"] == 1.0
    assert abs(s["composite"] - (1.0 + 0.4 * 0.7)) < 1e-9


def _fake_run(sc, config):
    agent = FakeAgent()
    env = FakeCoupledEnv(sc, config)
    out = run_episode(sc, env, agent, CoupledRunConfig(), realize.INSTRUCTIONS)
    return out, score(sc, out.breached, out.tasks_done, w_u=0.4)


def test_fake_loop_matches_oracle_on_optimum():
    sc = get("pipeline_box")
    config = analysis.exact_best(sc, 0.4)["config"]         # contained, 2/3 tasks
    out, s = _fake_run(sc, config)
    assert out.invalid_kind is None
    assert set(out.breached) == {c.id for c in sc.open_channels(config)}      # == none
    assert set(out.tasks_done) == {t.id for t in sc.satisfied_tasks(config)}
    assert abs(s["composite"] - oracle.composite(sc, config, w_u=0.4)) < 1e-9


def test_fake_loop_matches_oracle_on_wide_open():
    sc = get("pipeline_box")
    config = sc.space.wide_open()
    out, s = _fake_run(sc, config)
    assert set(out.breached) == {c.id for c in sc.open_channels(config)}       # leaks several
    assert not s["fully_contained"]
    assert abs(s["composite"] - oracle.composite(sc, config, w_u=0.4)) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("test_real: all passed")
