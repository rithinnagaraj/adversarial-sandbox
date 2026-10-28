"""Scenario self-tests: is the scenario a genuinely interesting *coupled* problem?

Every check here is EXACT and scale-free -- it never enumerates the full space. It uses
local enumeration (over just the knobs a predicate reads) and comparisons of specific
configs, plus a couple of logical implications:

  * "exists a safe-achievable task (weight>0)" => a contained config does useful work
    => best composite > lockdown (which does nothing). So lockdown is not optimal.
  * "exists a forced-tradeoff task (weight>0)" => the utility ceiling (which can include
    that task) strictly exceeds the max utility reachable while contained (which cannot).
    So containment genuinely costs utility.

That means these pass/fail verdicts hold even for a >1e9-config scenario.

(The *real-container* validator -- confirming each channel's token is actually reachable
in an opening config and unreachable in a closing one -- is a separate Docker-backed check;
see README. This module validates the declared structure.)
"""

from __future__ import annotations

from coupled_env import analysis, oracle
from coupled_env.scenario import Scenario


class Check:
    def __init__(self, name: str, ok: bool, detail: str):
        self.name = name
        self.ok = ok
        self.detail = detail


def run(scenario: Scenario, w_u: float = 0.4, max_channels_listed: int = 8) -> list[Check]:
    checks: list[Check] = []

    # 1. every channel is both openable and closable (exact, per-channel local enumeration)
    not_openable, not_closable = [], []
    for c in scenario.channels:
        opens, closes = analysis.channel_openable_closable(scenario, c)
        if not opens:
            not_openable.append(c.id)
        if not closes:
            not_closable.append(c.id)
    checks.append(Check(
        "every channel is openable and closable",
        not not_openable and not not_closable,
        f"{len(scenario.channels)} channels; "
        f"never-openable={not_openable or 'none'}; never-closable={not_closable or 'none'}",
    ))

    # 2. at least one channel is a genuine multi-knob conjunction (coupling, not a single knob)
    multi = [c for c in scenario.channels if len(c.exposes_when.knobs()) >= 2]
    checks.append(Check(
        "has multi-knob couplings",
        len(multi) >= 1,
        f"{len(multi)}/{len(scenario.channels)} channels depend on >=2 knobs"
        + (f" (e.g. {[c.id for c in multi[:max_channels_listed]]})" if multi else ""),
    ))

    feas = analysis.task_feasibility(scenario)
    forced = [t for t, i in feas.items() if i["label"] == "forced_tradeoff" and i["weight"] > 0]
    needs_care = [t for t, i in feas.items() if i["label"] == "needs_care"]
    safe_achievable = [t for t, i in feas.items()
                       if i["label"] in ("free", "needs_care") and i["weight"] > 0]

    # 3. at least one forced tradeoff
    checks.append(Check(
        "has a forced tradeoff",
        len(forced) >= 1,
        f"tasks doable only by opening a channel: {forced[:max_channels_listed] or 'none'}"
        f" ({len(forced)} total)",
    ))

    # 4. at least one compensation/routing choice
    checks.append(Check(
        "has a compensation/routing choice",
        len(needs_care) >= 1,
        f"tasks safe only with the right knob combo: {needs_care[:max_channels_listed] or 'none'}"
        f" ({len(needs_care)} total)",
    ))

    # 5. lockdown is not optimal  (implied by a safe-achievable task with weight>0)
    lock = scenario.space.lockdown()
    lock_comp = oracle.composite(scenario, lock, w_u=w_u)
    checks.append(Check(
        "lockdown is not optimal",
        len(safe_achievable) >= 1,
        f"lockdown composite={lock_comp:.3f}; a contained config can also do "
        f"{len(safe_achievable)} task(s) -> beats lockdown",
    ))

    # 6. wide-open is not optimal  (lockdown already beats grab-everything)
    wide = scenario.space.wide_open()
    wide_comp = oracle.composite(scenario, wide, w_u=w_u)
    checks.append(Check(
        "wide-open is not optimal",
        lock_comp > wide_comp + 1e-9,
        f"wide-open composite={wide_comp:.3f} < lockdown {lock_comp:.3f} (something beats wide-open)",
    ))

    # 7. containment costs utility  (implied by a forced-tradeoff task with weight>0)
    checks.append(Check(
        "containment costs utility",
        len(forced) >= 1,
        f"{len(forced)} task(s) add utility only by breaking containment",
    ))

    return checks


def all_ok(checks: list[Check]) -> bool:
    return all(c.ok for c in checks)
