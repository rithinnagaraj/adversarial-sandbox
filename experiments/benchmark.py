"""Phase B: compare designers that generalize, at a fixed episode budget.

Runs, on the FULL 243-config space with one fixed reward scheme:
  * random              -- the honest baseline
  * bayesopt GP         -- GP surrogate (treats knobs as ordinal/metric)
  * bayesopt RF / ET    -- tree surrogates (handle categorical knobs better)
  * ucb (optional)      -- tabular, no-generalization reference

Each designer gets the SAME budget (rounds x repeats episodes), so the comparison
is apples-to-apples in the unit that will be expensive for real: LLM episodes.
Repeats average out the Bernoulli noise in containment before the designer learns.

Outputs:
  * a printed comparison table (sample efficiency + final components)
  * logs/phase_b_results.json  -- full summaries incl. learning curves (for Phase C)

Run (inside the skopt-enabled env, e.g. WSL):
    ~/advc-venv/bin/python experiments/benchmark.py --rounds 120 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.space import DEFAULT_SPACE
from designer.bayesopt import BayesOptDesigner, _HAVE_SKOPT
from designer.random_search import RandomSearchDesigner
from loop import ExperimentConfig, run_experiment


def _make(spec: str, seed: int):
    """spec is 'random', 'ucb', or 'bayesopt:<SURROGATE>'."""
    if spec == "random":
        return RandomSearchDesigner(space=DEFAULT_SPACE, seed=seed)
    if spec == "ucb":  # force the tabular fallback regardless of skopt availability
        from designer.bayesopt import _UCBBandit

        class _Tabular(RandomSearchDesigner):  # thin adapter around the bandit
            name = "ucb_tabular"

            def __init__(self, seed):
                super().__init__(space=DEFAULT_SPACE, seed=seed)
                self._b = {}

            def _bk(self, ctx):
                k = str(ctx.get("suite", "_"))
                self._b.setdefault(k, _UCBBandit(DEFAULT_SPACE, seed))
                return self._b[k]

            def propose(self, context):
                return self._bk(context).propose()

            def update(self, result):
                if result.usable:
                    self._bk(result.context).update(result.vector, result.reward)

        return _Tabular(seed)
    if spec.startswith("bayesopt:"):
        return BayesOptDesigner(space=DEFAULT_SPACE, seed=seed, surrogate=spec.split(":", 1)[1])
    if spec == "rl":
        from designer.rl_static import StaticRLDesigner
        return StaticRLDesigner(space=DEFAULT_SPACE, seed=seed)
    raise ValueError(f"unknown designer spec {spec!r}")


def run(specs, rounds, repeats, reward, seed, outdir, world):
    os.makedirs(outdir, exist_ok=True)
    results = []
    for spec in specs:
        designer = _make(spec, seed)
        exp = ExperimentConfig(
            rounds=rounds, repeats=repeats, seed=seed, reward=reward, world=world,
            reward_kwargs={} if reward != "composite" else {"w_c": 1.0, "w_u": 0.2, "w_i": 0.5},
            log_path=os.path.join(outdir, f"ep_{world}_{designer.name}.jsonl"),
        )
        t0 = time.time()
        summary = run_experiment(designer, exp)
        summary["wall_seconds"] = round(time.time() - t0, 1)
        summary["spec"] = spec
        results.append(summary)
        print(f"  [{world}] ran {designer.name:16s} in {summary['wall_seconds']:6.1f}s")
    return results


def _knife_edge(levels: dict) -> bool:
    """Does the best config actually preserve utility (all mixed_v0 subtasks feasible)?"""
    return (levels.get("filesystem") != "none" and levels.get("network") != "none"
            and levels.get("resources") != "tight" and levels.get("capabilities") != "minimal")


def _fmt_table(results, world):
    cols = ["designer", "episodes", "mean_rwd", "contain", "utility", "invalid",
            "best@mixed", "best_is"]
    print(f"\n--- world = {world} ---")
    print("  ".join(f"{c:>11s}" for c in cols))
    print("  ".join("-" * 11 for _ in cols))
    for r in results:
        b = r["best_config_per_suite"].get("mixed_v0", {})
        best = b.get("reward", None)
        kind = "n/a" if not b else ("KNIFE" if _knife_edge(b.get("levels", {})) else "lockdown")
        row = [
            r["designer"], r["episodes"], r["mean_reward"], r["containment_rate"],
            r["utility_mean"], r["invalid_share"],
            "n/a" if best is None else best, kind,
        ]
        print("  ".join(f"{str(v):>11s}" for v in row))
    print("(best_is = whether the top-reward mixed_v0 config actually preserves utility)")


def main():
    ap = argparse.ArgumentParser(description="Phase B designer comparison.")
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reward", default="composite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="logs/phase_b")
    ap.add_argument("--include-ucb", action="store_true", help="also run the tabular no-generalization reference")
    ap.add_argument("--worlds", nargs="+", default=["isolation_gated", "entangled"],
                    help="escape models to validate across (knife-edge exists vs not)")
    args = ap.parse_args()

    specs = ["random"]
    if _HAVE_SKOPT:
        specs += ["bayesopt:GP", "bayesopt:RF", "bayesopt:ET"]
    else:
        print("WARNING: scikit-optimize not importable here; only random/rl (+ucb) will run.")
    specs.append("rl")  # the learnable designer
    if args.include_ucb:
        specs.append("ucb")

    print(f"Phase B: {len(specs)} designers x {args.rounds} rounds x {args.repeats} repeats "
          f"= {args.rounds * args.repeats} episodes each, reward={args.reward}, worlds={args.worlds}")
    all_results = {}
    for world in args.worlds:
        all_results[world] = run(specs, args.rounds, args.repeats, args.reward, args.seed, args.outdir, world)

    out = os.path.join(args.outdir, "phase_b_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results_by_world": all_results}, fh, indent=2)
    for world, results in all_results.items():
        _fmt_table(results, world)
    print(f"\nfull results (incl. learning curves): {out}")


if __name__ == "__main__":
    main()
