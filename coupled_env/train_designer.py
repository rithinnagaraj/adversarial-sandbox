"""Train a learnable designer against a scenario's oracle (numpy -> WSL venv).

    ~/advc-venv/bin/python -m coupled_env.train_designer --scenario pipeline_box --designer mlp
    ~/advc-venv/bin/python -m coupled_env.train_designer --scenario datacenter  --designer mlp \
        --rounds 400 --batch 64

No Docker / no LLM -- the oracle gives the reward, so this trains in seconds-to-minutes and
lets us develop the policy before any real-execution cost. Reports the learned
recommendation vs. the exact baselines and vs. random search on the same episode budget.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from coupled_env import analysis, oracle
from coupled_env.designers import make_designer
from coupled_env.scenarios import get


def main():
    ap = argparse.ArgumentParser(description="Train a designer on a coupled-env scenario.")
    ap.add_argument("--scenario", default="pipeline_box")
    ap.add_argument("--designer", default="mlp", choices=["mlp", "factorized"])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--w-u", dest="w_u", type=float, default=0.4)
    ap.add_argument("--hidden", default="128,128", help="MLP hidden sizes, e.g. 256,256")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sc = get(args.scenario)
    kw = {"seed": args.seed}
    if args.designer == "mlp":
        kw["hidden"] = tuple(int(x) for x in args.hidden.split(","))
    if args.lr is not None:
        kw["lr"] = args.lr
    if args.entropy is not None:
        kw["entropy"] = args.entropy
    designer = make_designer(args.designer, sc, **kw)

    episodes = args.rounds * args.batch
    print(f"train {args.designer} on {sc.id} | {sc.space.size():.2e} configs, "
          f"{len(sc.channels)} channels | {args.rounds} rounds x {args.batch} = {episodes} episodes "
          f"| w_u={args.w_u}")

    best_seen = {"composite": -1e9}
    t0 = time.time()
    for r in range(args.rounds):
        rollouts = [designer.propose() for _ in range(args.batch)]
        rewards = [oracle.composite(sc, cfg, w_u=args.w_u) for cfg, _ in rollouts]
        designer.update_batch(rollouts, rewards)
        # track the best config the policy actually sampled
        bi = int(np.argmax(rewards))
        if rewards[bi] > best_seen["composite"]:
            cfg = rollouts[bi][0]
            best_seen = {"composite": rewards[bi], "cfg": cfg,
                         "contained": oracle.contained(sc, cfg),
                         "utility": oracle.utility(sc, cfg)}
        if r % max(1, args.rounds // 15) == 0 or r == args.rounds - 1:
            rec = designer.recommend()
            rc = oracle.composite(sc, rec, w_u=args.w_u)
            print(f"  round {r:4d}  batch_mean={np.mean(rewards):+.3f}  "
                  f"recommend_composite={rc:+.3f}  best_seen={best_seen['composite']:+.3f}")

    rec = designer.recommend()
    rec_out = oracle.outcome(sc, rec, w_u=args.w_u)
    lock = oracle.composite(sc, sc.space.lockdown(), w_u=args.w_u)
    wide = oracle.composite(sc, sc.space.wide_open(), w_u=args.w_u)
    rand = analysis.sampled_best(sc, episodes, args.w_u, seed=args.seed + 7)

    print(f"\n=== result ({time.time() - t0:.1f}s) ===")
    print(f"LEARNED recommendation: composite={rec_out['composite']}  "
          f"contained={rec_out['contained']}  containment={rec_out['containment_score']}  "
          f"utility={rec_out['utility']}")
    print(f"  tasks done: {len(rec_out['satisfied_tasks'])}/{len(sc.tasks)}   "
          f"channels open: {len(rec_out['open_channels'])}/{len(sc.channels)}")
    print(f"best config the policy sampled: composite={best_seen['composite']:.3f}  "
          f"contained={best_seen.get('contained')}  utility={best_seen.get('utility', 0):.3f}")
    print(f"\nbaselines (w_u={args.w_u}):")
    print(f"  lockdown        composite={lock:.3f}")
    print(f"  wide_open       composite={wide:.3f}")
    print(f"  random ({episodes} draws) composite={rand['composite']:.3f}  contained={rand['contained']}")
    if sc.space.size() <= analysis.EXACT_LIMIT:
        ex = analysis.exact_best(sc, args.w_u)
        print(f"  EXACT optimum   composite={ex['composite']:.3f}  contained={ex['contained']}  "
              f"utility={ex['utility']:.3f}")


if __name__ == "__main__":
    main()
