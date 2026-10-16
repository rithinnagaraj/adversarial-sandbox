"""Static vs adaptive designer ablation (simulation).

Both arms run through the SAME adaptive runner -- same randomized flag timing, same
hidden-phase telemetry, same reward -- so the only difference is whether runtime
reconfiguration is used:

  * static   -- proposes one config at creation and always HOLDS (never reconfigures).
                This is v0's one-config-per-trajectory policy, as a learnable net.
  * adaptive -- the trajectory-conditioned AdaptiveRLDesigner (loosen-only).

Since "hold" is an available adaptive action, static is a special case of adaptive.
The question this answers: does adaptivity earn its extra sample cost at a given
budget? Reports reward + components per arm, plus how often the adaptive arm actually
grants access (if ~0, adaptivity isn't buying anything).

    ~/advc-venv/bin/python experiments/ablation.py --rounds 400 --repeats 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.provisioner import FakeProvisioner
from config.space import DEFAULT_SPACE
from designer.rl_adaptive import AdaptiveRLDesigner
from designer.rl_static import StaticRLDesigner
from episode.adaptive_runner import AdaptiveEpisodeRunner
from episode.agent import FakeAgent
from episode.outcomes import Outcome
from episode.runner import RunnerConfig
from episode.subtasks import get_suite
from reward import is_contained, is_invalid, make_scheme, utility_fraction


class StaticAsAdaptive:
    """Adapts StaticRLDesigner to the adaptive interface: propose once, always hold."""

    name = "static"

    def __init__(self, **kw):
        self.inner = StaticRLDesigner(**kw)
        self._ctx = None
        self._vec = None

    def begin_episode(self, context):
        self._ctx = context

    def propose_initial(self, context):
        self._vec = self.inner.propose(context)
        return self._vec

    def reconfigure(self, state):
        return {}  # always hold -> never reconfigures

    def learn(self, episode_return):
        from designer.interface import DesignerResult
        self.inner.update(DesignerResult(vector=self._vec, context=self._ctx, reward=episode_return,
                                         outcome=Outcome.SUBTASKS_DONE_FLAG_UNREAD, usable=True, episode_id="a"))


def run_arm(designer, rounds, repeats, world, reward, seed):
    prov = FakeProvisioner(DEFAULT_SPACE, world=world)
    agent = FakeAgent()
    runner = AdaptiveEpisodeRunner(prov, agent, designer, DEFAULT_SPACE,
                                   RunnerConfig(subtask_attempts=2, flag_attempts=3), seed=seed)
    scheme = make_scheme(reward, w_c=1.0, w_u=0.2, w_i=0.5)
    rewards, contained, observed, util, invalid, loosens = [], 0, 0, 0.0, 0, 0
    ep_idx = 0
    for i in range(rounds):
        suite = get_suite("mixed_v0")
        reps = []
        for j in range(repeats):
            ep = runner.run(suite, {"suite": "mixed_v0", "round": i}, seed=seed + 1000 * ep_idx + j)
            r = scheme.reward(ep)
            reps.append(r)
            rewards.append(r)
            util += utility_fraction(ep)
            if not is_invalid(ep):
                observed += 1
                contained += int(is_contained(ep))
            else:
                invalid += 1
            loosens += sum(1 for s in ep.trajectory if s.observation.startswith("reconfigure->"))
            ep_idx += 1
        designer.learn(sum(reps) / len(reps))
    n = max(1, len(rewards))
    return {
        "arm": designer.name,
        "mean_reward": round(sum(rewards) / n, 4),
        "containment_rate": round(contained / observed, 4) if observed else None,
        "utility_mean": round(util / n, 4),
        "invalid_share": round(invalid / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--world", default="isolation_gated")
    ap.add_argument("--reward", default="composite")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = [
        StaticAsAdaptive(space=DEFAULT_SPACE, suites=("mixed_v0", "light_v0"), seed=args.seed),
        AdaptiveRLDesigner(space=DEFAULT_SPACE, suites=("mixed_v0", "light_v0"), seed=args.seed),
    ]
    print(f"ablation: world={args.world} reward={args.reward} rounds={args.rounds} repeats={args.repeats}")
    rows = [run_arm(a, args.rounds, args.repeats, args.world, args.reward, args.seed) for a in arms]
    cols = ["arm", "mean_reward", "containment_rate", "utility_mean", "invalid_share"]
    print("  ".join(f"{c:>16s}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>16s}" for c in cols))


if __name__ == "__main__":
    main()
