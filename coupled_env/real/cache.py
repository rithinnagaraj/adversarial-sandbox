"""Real-episode sample bank, keyed on the REALIZED container spec (not on hand-labeled knobs).

Peer-review rationale (see result_notes/roadmap_realism_rigor.md, "Key tractability insight"):
keying a cache on "the knobs that matter" would bake the oracle's core/distractor partition into
the experiment -- circular, since knowing which knobs matter is exactly the oracle ground truth we
must not assume. Instead we key on `R(config)` -- the actual `docker run` flags + host entries +
env structure + planted-channel set the harness instantiates. Two configs collapse to one key only
when the environment literally builds them identically; that grouping is *discovered from the
realization's output*, never asserted by us. (In the current datacenter_lite realization the 16
distractor knobs are not applied, so they fall out of the key automatically -- but if a future
realization wires a "distractor" into a real flag, the key will split on it, with no code change.)

Each key holds a BANK OF I.I.D. SAMPLES, because the frozen agent is stochastic. Legitimate uses:
held-out mean +/- CI, variance estimation, and topping a spec up to k samples for evaluation.
It is explicitly NOT a training accelerant: reusing a cached return inside REINFORCE would feed
the gradient correlated/stale rewards. Training draws a fresh episode every time and merely
*appends* its outcome here, so the bank grows for free and held-out eval can pool those draws.
Pooling training + held-out draws for one spec is valid because the agent is frozen and the
container is rebuilt identically -- conditioned on the spec they are i.i.d.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from coupled_env.scenario import Scenario

# Stable placeholder so the *structure* of env injection (which vars exist) enters the key while
# the per-episode random secret VALUES (which never affect breach detection) do not.
_TOK_SENTINEL = "<TOKEN>"


def realized_spec(scenario: Scenario, config: dict, realize_mod) -> dict:
    """The canonical, value-free description of the container this config instantiates.

    Pure function of `config` (+ the scenario's open-channel predicates). Contains everything
    that determines the real outcome distribution and nothing episode-random.
    """
    open_ids = sorted(c.id for c in scenario.open_channels(config))
    if realize_mod is None:                       # fake env: deterministic in the full config
        return {"env": "fake", "scenario": scenario.id,
                "config": sorted(config.items()), "open_ids": open_ids}
    sentinel = {ch: _TOK_SENTINEL for ch in realize_mod.CHANNEL_TOKENS}
    return {
        "env": "docker",
        "scenario": scenario.id,
        "docker_args": realize_mod.docker_args(config),
        "add_host_args": realize_mod.add_host_args(config),
        "env_args": realize_mod.env_args(config, sentinel),
        "open_ids": open_ids,          # drives placement_commands + sidecar_commands deterministically
    }


def realized_key(scenario: Scenario, config: dict, realize_mod) -> str:
    """16-hex-char digest of `realized_spec` -- the cache/bank key."""
    blob = json.dumps(realized_spec(scenario, config, realize_mod), sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class RealEvalCache:
    """Append-only JSONL bank of real-episode samples, grouped by realized-spec key.

    Each stored line: {"key","config","sample"} where `sample` is the scoring dict plus
    episode metadata. Re-opening the same path reloads every sample, so the bank survives
    `--resume` and is shared across training and held-out evaluation runs.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self.by_key: dict[str, list[dict]] = {}
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.by_key.setdefault(rec["key"], []).append(rec["sample"])

    def samples(self, key: str) -> list[dict]:
        return self.by_key.get(key, [])

    def valid_samples(self, key: str) -> list[dict]:
        return [s for s in self.by_key.get(key, []) if s.get("invalid_kind") is None]

    def n_valid(self, key: str) -> int:
        return len(self.valid_samples(key))

    def composites(self, key: str) -> list[float]:
        return [float(s["composite"]) for s in self.valid_samples(key)]

    def add(self, key: str, sample: dict, config: dict | None = None) -> None:
        """Record one episode outcome under its spec key (in memory + appended to disk)."""
        self.by_key.setdefault(key, []).append(sample)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "config": config, "sample": sample}) + "\n")

    def keys(self) -> list[str]:
        return list(self.by_key)
