"""Tests for the learnable designers (numpy -> run in the WSL venv).

    ~/advc-venv/bin/python coupled_env/tests/test_designers.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coupled_env import analysis, oracle
from coupled_env.designers import FactorizedDesigner, MLPDesigner
from coupled_env.scenarios import get


def _train(designer, sc, rounds, batch, w_u=0.4, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        rollouts = [designer.propose() for _ in range(batch)]
        rewards = [oracle.composite(sc, cfg, w_u=w_u) for cfg, _ in rollouts]
        designer.update_batch(rollouts, rewards)
    return designer.recommend()


def test_mlp_recommend_is_valid_config():
    sc = get("pipeline_box")
    d = MLPDesigner(sc, hidden=(32, 32), seed=0)
    cfg = d.recommend()
    sc.space.validate_config(cfg)              # no illegal levels
    assert set(cfg) == set(sc.space.names)


def test_mlp_learns_pipeline_box_optimum():
    sc = get("pipeline_box")
    exact = analysis.exact_best(sc, 0.4)["composite"]   # 1.28
    d = MLPDesigner(sc, hidden=(64, 64), lr=5e-3, seed=0)
    rec = _train(d, sc, rounds=250, batch=64)
    got = oracle.composite(sc, rec, w_u=0.4)
    assert got >= exact - 1e-6, f"MLP reached {got}, exact optimum {exact}"
    assert oracle.contained(sc, rec)


def test_factorized_also_learns_pipeline_box():
    # pipeline_box has a single-config optimum, so even the factorized policy can reach it.
    sc = get("pipeline_box")
    exact = analysis.exact_best(sc, 0.4)["composite"]
    d = FactorizedDesigner(sc, lr=0.2, seed=0)
    rec = _train(d, sc, rounds=250, batch=64)
    assert oracle.composite(sc, rec, w_u=0.4) >= exact - 1e-6


def test_mlp_beats_lockdown_on_datacenter_small():
    sc = get("datacenter_small")
    lock = oracle.composite(sc, sc.space.lockdown(), 0.4)
    d = MLPDesigner(sc, hidden=(64, 64), lr=5e-3, seed=0)
    rec = _train(d, sc, rounds=250, batch=64)
    assert oracle.composite(sc, rec, w_u=0.4) > lock, "designer should beat doing nothing"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("test_designers: all passed")
