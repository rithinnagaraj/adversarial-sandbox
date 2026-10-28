"""Analysis of a scenario's tradeoff structure.

The hard constraint: at >1e9 configs we cannot enumerate the whole space. The trick that
keeps analysis EXACT anyway is **locality** -- a channel's/task's predicate reads only a
handful of knobs, so questions about it ("is this task ever safe-achievable?", "is this
channel openable?") can be answered by enumerating just those few knobs while holding every
other knob at its safest (lockdown) value. That sub-enumeration is tiny regardless of the
full space size. Only the *global optimum* truly needs the whole space; for that we fall
back to sampling (and label it an estimate), because finding the global optimum in a huge
space is the designer's job, not the validator's.
"""

from __future__ import annotations

import random
from itertools import product
from typing import Iterable, Mapping

from coupled_env import oracle
from coupled_env.scenario import Scenario, Task

EXACT_LIMIT = 500_000   # brute-force the whole space only below this


def local_enumerate(scenario: Scenario, subset: Iterable[str], base: Mapping[str, str]):
    """Yield configs that vary only `subset` (over their levels), all other knobs fixed at
    `base`. Exact and cheap because subset is small, whatever the full space size."""
    subset = set(subset)
    knobs = [k for k in scenario.space.knobs if k.name in subset]
    for combo in product(*[k.levels for k in knobs]):
        cfg = dict(base)
        for i, k in enumerate(knobs):
            cfg[k.name] = combo[i]
        yield cfg


def channel_openable_closable(scenario: Scenario, channel) -> tuple[bool, bool]:
    """Exact: can this channel be opened, and can it be closed? Enumerate only its knobs."""
    lock = scenario.space.lockdown()
    knobs = channel.exposes_when.knobs()
    opens = closes = False
    for cfg in local_enumerate(scenario, knobs, lock):
        if channel.is_open(cfg):
            opens = True
        else:
            closes = True
        if opens and closes:
            break
    return opens, closes


def task_feasibility(scenario: Scenario) -> dict[str, dict]:
    """Per task, EXACT at any scale: is it satisfiable, and can it be done while strictly
    contained? Enumerate only the task's own knobs with all others at their safest value --
    which is the best case for containment, so the verdict is exact.

      forced_tradeoff : satisfiable, but never while contained -> must be sacrificed
      needs_care      : safe-achievable, but some satisfying configs leak -> pick the combo
      free            : every satisfying config is contained
    """
    lock = scenario.space.lockdown()
    out: dict[str, dict] = {}
    for t in scenario.tasks:
        sat = [cfg for cfg in local_enumerate(scenario, t.requires.knobs(), lock)
               if t.satisfiable(cfg)]
        safe = [cfg for cfg in sat if oracle.contained(scenario, cfg)]
        if not sat:
            label = "impossible"
        elif not safe:
            label = "forced_tradeoff"
        elif len(safe) < len(sat):
            label = "needs_care"
        else:
            label = "free"
        out[t.id] = {"label": label, "weight": t.weight,
                     "n_local_satisfying": len(sat), "n_local_safe": len(safe)}
    return out


def _labels(scenario: Scenario) -> dict[str, list[str]]:
    feas = task_feasibility(scenario)
    buckets: dict[str, list[str]] = {"free": [], "needs_care": [],
                                     "forced_tradeoff": [], "impossible": []}
    for tid, info in feas.items():
        buckets[info["label"]].append(tid)
    return buckets


def max_safe_utility(scenario: Scenario) -> float:
    """Highest utility reachable while STRICTLY contained. Brute force -- small spaces only."""
    best = 0.0
    for cfg in scenario.space.enumerate():
        if oracle.contained(scenario, cfg):
            best = max(best, oracle.utility(scenario, cfg))
    return best


def utility_ceiling(scenario: Scenario) -> float:
    """Highest utility reachable ignoring containment. Brute force -- small spaces only."""
    return max(oracle.utility(scenario, cfg) for cfg in scenario.space.enumerate())


# --- global-optimum facts: exact if small, else sampled estimate ---

def sampled_best(scenario: Scenario, n: int, w_u: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    best = None
    for _ in range(n):
        cfg = {k.name: rng.choice(k.levels) for k in scenario.space.knobs}
        comp = oracle.composite(scenario, cfg, w_u=w_u)
        if best is None or comp > best["composite"]:
            best = {"config": cfg, "composite": comp,
                    "contained": oracle.contained(scenario, cfg),
                    "utility": oracle.utility(scenario, cfg)}
    return best


def exact_best(scenario: Scenario, w_u: float) -> dict:
    best = None
    for cfg in scenario.space.enumerate():
        comp = oracle.composite(scenario, cfg, w_u=w_u)
        if best is None or comp > best["composite"]:
            best = {"config": cfg, "composite": comp,
                    "contained": oracle.contained(scenario, cfg),
                    "utility": oracle.utility(scenario, cfg)}
    return best


def best_composite(scenario: Scenario, w_u: float = 0.4, samples: int = 200_000) -> dict:
    if scenario.space.size() <= EXACT_LIMIT:
        return dict(exact_best(scenario, w_u), exact=True)
    return dict(sampled_best(scenario, samples, w_u), exact=False)


def summary(scenario: Scenario, w_u: float = 0.4, samples: int = 200_000) -> dict:
    size = scenario.space.size()
    lock = scenario.space.lockdown()
    wide = scenario.space.wide_open()
    best = best_composite(scenario, w_u, samples)
    buckets = _labels(scenario)
    return {
        "space_size": size,
        "n_channels": len(scenario.channels),
        "n_tasks": len(scenario.tasks),
        "best": {k: best[k] for k in ("composite", "contained", "utility")},
        "best_config": best["config"],
        "best_is_exact": best["exact"],
        "lockdown": {"composite": round(oracle.composite(scenario, lock, w_u=w_u), 4),
                     "utility": round(oracle.utility(scenario, lock), 4),
                     "containment": round(oracle.containment_score(scenario, lock), 4)},
        "wide_open": {"composite": round(oracle.composite(scenario, wide, w_u=w_u), 4),
                      "utility": round(oracle.utility(scenario, wide), 4),
                      "containment": round(oracle.containment_score(scenario, wide), 4)},
        "task_labels": {k: v for k, v in buckets.items() if v},
        "task_feasibility": task_feasibility(scenario),
    }
