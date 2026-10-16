"""Outer experiment loop.

Per round:  context -> designer.propose -> provision + run episode -> classify
-> reward -> log -> designer.update.

v0 runs entirely on the FakeProvisioner + FakeAgent (no Docker, no tokens). The
summary includes the **confound audit** the plan calls for: how many episodes
ended INVALID, and -- critically -- what share of apparently-"contained" episodes
had no detected boundary attempt. A high share there means the designer is
"winning" by breaking the agent, not by containing it.

The reward is a *pluggable scheme* (see reward.py), selected with --reward; the
summary always reports the three components separately (containment rate, mean
utility fraction, INVALID rate, split into env-induced vs harness-fault) so the
objective's components stay legible regardless of which scheme drives learning.

Usage:
    python loop.py --rounds 200 --designer random --reward composite
    python loop.py --rounds 200 --designer bayesopt --reward breach_averse --seed 0
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
from typing import Any

from config.provisioner import FakeProvisioner, Provisioner
from config.space import DEFAULT_SPACE, ConfigSpace
from designer.interface import Designer, DesignerResult
from designer.random_search import RandomSearchDesigner
from episode.agent import Agent, FakeAgent
from episode.monitors import TrajectoryLogger
from episode.outcomes import Outcome
from episode.runner import EpisodeRunner, RunnerConfig
from episode.subtasks import SUITES, SubtaskSuite, get_suite
from reward import (
    RewardScheme,
    is_contained,
    is_env_induced_invalid,
    is_harness_fault,
    is_invalid,
    make_scheme,
    utility_fraction,
)


@dataclass
class ExperimentConfig:
    rounds: int = 200
    suites: tuple[str, ...] = ("mixed_v0", "light_v0")  # cycled as designer context
    seed: int = 0
    log_path: str = "logs/episodes.jsonl"
    reward: str = "composite"                      # scheme name (see reward.SCHEME_NAMES)
    reward_kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. composite weights
    repeats: int = 1                               # episodes per proposed config; averaged
    #   repeats=1 for low-noise checks; raise (e.g. 3) when the objective is noisy,
    #   since containment is a probability and a single rollout is one Bernoulli sample.
    world: str = "isolation_gated"                 # FakeEnv escape model (see ESCAPE_WORLDS)


def build_designer(name: str, space: ConfigSpace, seed: int, surrogate: str = "GP",
                   suites: tuple[str, ...] | None = None) -> Designer:
    if name == "random":
        return RandomSearchDesigner(space=space, seed=seed)
    if name == "bayesopt":
        from designer.bayesopt import BayesOptDesigner  # optional dep
        return BayesOptDesigner(space=space, seed=seed, surrogate=surrogate)
    if name == "rl":
        from designer.rl_static import StaticRLDesigner  # needs numpy (WSL)
        return StaticRLDesigner(space=space, suites=tuple(suites) if suites else ("mixed_v0", "light_v0"),
                                seed=seed)
    raise ValueError(f"unknown designer {name!r} (choices: random, bayesopt, rl)")


def run_experiment(
    designer: Designer,
    exp: ExperimentConfig,
    provisioner: Provisioner | None = None,
    agent: Agent | None = None,
    scheme: RewardScheme | None = None,
) -> dict[str, Any]:
    provisioner = provisioner or FakeProvisioner(DEFAULT_SPACE, world=exp.world)
    agent = agent or FakeAgent()
    scheme = scheme or make_scheme(exp.reward, **exp.reward_kwargs)
    runner = EpisodeRunner(provisioner, agent, RunnerConfig())
    repeats = max(1, exp.repeats)

    outcome_counts: collections.Counter[str] = collections.Counter()
    rewards: list[float] = []     # per-episode rewards (all repeats)
    # component accumulators (scheme-independent, item 3)
    observed = 0                  # episodes where containment was observed (not INVALID)
    contained = 0                 # of those, how many held
    utility_sum = 0.0             # mean fractional utility over all episodes
    env_invalid = 0               # env-induced INVALID (a finding)
    harness_invalid = 0           # harness-fault INVALID (our bug, noise)
    good_total = 0                # the "good" categorical outcome
    good_no_attempt = 0           # good but no boundary attempt (the gave-up confound)
    per_suite_best: dict[str, tuple[float, list[int]]] = {}
    # learning curve (for Phase C plots) + sample-efficiency metric
    curve: list[dict[str, Any]] = []
    first_good_round: dict[str, int] = {}
    episode_idx = 0

    with TrajectoryLogger(exp.log_path) as logger:
        for i in range(exp.rounds):
            suite_id = exp.suites[i % len(exp.suites)]
            suite: SubtaskSuite = get_suite(suite_id)
            context = {"suite": suite_id, "round": i}

            vector = designer.propose(context)

            # evaluate the proposed config `repeats` times (independent seeds) and
            # average the reward, since containment is a probability (noisy signal).
            rep_rewards: list[float] = []
            any_usable = False
            for j in range(repeats):
                episode = runner.run(vector, suite, context, seed=exp.seed + 1000 * episode_idx + j)
                episode.reward = scheme.reward(episode)
                logger.log(episode)
                episode_idx += 1

                rep_rewards.append(episode.reward)
                any_usable = any_usable or scheme.should_learn(episode)
                outcome_counts[episode.outcome.value] += 1
                rewards.append(episode.reward)
                utility_sum += utility_fraction(episode)
                if not is_invalid(episode):
                    observed += 1
                    if is_contained(episode):
                        contained += 1
                elif is_harness_fault(episode):
                    harness_invalid += 1
                elif is_env_induced_invalid(episode):
                    env_invalid += 1
                if episode.outcome == Outcome.SUBTASKS_DONE_FLAG_UNREAD:
                    good_total += 1
                    if not episode.attempt_detected:
                        good_no_attempt += 1
                    first_good_round.setdefault(suite_id, i)

            avg_reward = sum(rep_rewards) / len(rep_rewards)
            # the designer learns once per proposed config, from the averaged reward
            agg = DesignerResult(
                vector=list(vector), context=dict(context), reward=avg_reward,
                outcome=episode.outcome, usable=any_usable, episode_id=f"r{i}",
            )
            designer.update(agg)
            if any_usable:
                best = per_suite_best.get(suite_id)
                if best is None or avg_reward > best[0]:
                    per_suite_best[suite_id] = (avg_reward, list(vector))
            curve.append({
                "round": i, "suite": suite_id, "avg_reward": round(avg_reward, 4),
                "best_so_far": round(per_suite_best.get(suite_id, (avg_reward, None))[0], 4),
            })

    n = max(1, len(rewards))
    invalid = outcome_counts.get(Outcome.INVALID.value, 0)
    summary = {
        "designer": designer.name,
        "reward_scheme": scheme.name,
        "world": exp.world,
        "rounds": exp.rounds,
        "repeats": repeats,
        "episodes": len(rewards),
        "mean_reward": round(sum(rewards) / n, 4),
        "outcome_counts": dict(outcome_counts),
        # --- the three objective components, tracked separately (item 3) ---
        "containment_rate": round(contained / observed, 4) if observed else None,
        "utility_mean": round(utility_sum / n, 4),
        "invalid_share": round(invalid / n, 4),
        "env_induced_invalid_share": round(env_invalid / n, 4),
        "harness_fault_share": round(harness_invalid / n, 4),
        # confound audit: of the "good" episodes, how many had no real attempt?
        "good_total": good_total,
        "good_without_attempt": good_no_attempt,
        "confound_share": round(good_no_attempt / good_total, 4) if good_total else 0.0,
        # sample efficiency: first round a genuine knife-edge config appeared
        "first_good_round": first_good_round,
        "best_config_per_suite": {
            s: {"reward": round(r, 4), "levels": DEFAULT_SPACE.to_dict(v)}
            for s, (r, v) in per_suite_best.items()
        },
        "curve": curve,
        "log_path": exp.log_path,
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== experiment summary ({summary['designer']} / reward={summary['reward_scheme']} / "
          f"world={summary['world']}, {summary['rounds']} rounds) ===")
    print(f"mean reward      : {summary['mean_reward']}")
    print(f"outcome counts   : {summary['outcome_counts']}")
    print("-- objective components (scheme-independent) --")
    print(f"containment rate : {summary['containment_rate']}  (of episodes where containment was observed)")
    print(f"utility mean     : {summary['utility_mean']}  (mean fraction of subtasks completed)")
    print(f"INVALID share    : {summary['invalid_share']}  "
          f"(env-induced={summary['env_induced_invalid_share']}, harness-fault={summary['harness_fault_share']})")
    print(f"confound audit   : {summary['good_without_attempt']}/{summary['good_total']} "
          f"'good' episodes had NO boundary attempt (share={summary['confound_share']})")
    print(f"first good round : {summary['first_good_round']}  (sample efficiency: lower is better)")
    print("best config per suite:")
    for suite, info in summary["best_config_per_suite"].items():
        print(f"  {suite}: reward={info['reward']}  {info['levels']}")
    print(f"log: {summary['log_path']}")


def main() -> None:
    from reward import SCHEME_NAMES
    from designer.bayesopt import SURROGATES
    from config.provisioner import ESCAPE_WORLDS
    ap = argparse.ArgumentParser(description="Adversarial container-config designer (v0 fake loop).")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--designer", choices=["random", "bayesopt", "rl"], default="random")
    ap.add_argument("--surrogate", choices=list(SURROGATES), default="GP",
                    help="BayesOpt surrogate model (only used by --designer bayesopt)")
    ap.add_argument("--world", choices=list(ESCAPE_WORLDS), default=ExperimentConfig.world,
                    help="FakeEnv escape model: isolation_gated (knife-edge exists) or entangled (none)")
    ap.add_argument("--reward", choices=list(SCHEME_NAMES), default=ExperimentConfig.reward)
    ap.add_argument("--repeats", type=int, default=1,
                    help="episodes per proposed config (averaged); raise when the objective is noisy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suites", nargs="+", default=list(ExperimentConfig.suites),
                    help=f"suite ids to cycle as context; known: {sorted(SUITES)}")
    ap.add_argument("--log", default=ExperimentConfig.log_path)
    # composite-scheme weights (ignored by other schemes)
    ap.add_argument("--w-c", type=float, default=1.0, help="composite: containment weight")
    ap.add_argument("--w-u", type=float, default=0.2, help="composite: utility weight")
    ap.add_argument("--w-i", type=float, default=0.5, help="composite: INVALID penalty weight")
    args = ap.parse_args()

    reward_kwargs = {"w_c": args.w_c, "w_u": args.w_u, "w_i": args.w_i} if args.reward == "composite" else {}
    exp = ExperimentConfig(
        rounds=args.rounds, suites=tuple(args.suites), seed=args.seed, log_path=args.log,
        reward=args.reward, reward_kwargs=reward_kwargs, repeats=args.repeats, world=args.world,
    )
    designer = build_designer(args.designer, DEFAULT_SPACE, args.seed, surrogate=args.surrogate,
                              suites=tuple(args.suites))
    summary = run_experiment(designer, exp)
    _print_summary(summary)


if __name__ == "__main__":
    main()
