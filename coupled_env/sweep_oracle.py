"""Phase 1 (rigor): multi-seed ORACLE training sweep -> convergence/regret distribution.

The oracle scorer is analytic (no containers, instant), so running the whole designer training
across many seeds is essentially free. This turns the single-trajectory "it converged to the
optimum" into a distribution: mean +/- CI of the final composite, the regret to the best known
config, and the fraction of seeds that reach it. Real episodes are NOT touched here (that budget
is spent in holdout_eval).

    ~/advc-venv/bin/python -m coupled_env.sweep_oracle --scenario datacenter_lite \
        --seeds 10 --rounds 150 --batch 64 --out logs/dclite_oracle_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

from coupled_env import analysis, oracle
from coupled_env.designers import make_designer
from coupled_env.scenarios import get
from coupled_env.stats import fmt_ci, mean_ci
from coupled_env.train_designer import oracle_train


def run_sweep(sc, seeds, rounds, batch, w_u, hidden, lr, entropy, designer_kind="mlp"):
    rows = []
    for seed in seeds:
        kw = {"seed": seed, "hidden": hidden}
        if lr is not None:
            kw["lr"] = lr
        if entropy is not None:
            kw["entropy"] = entropy
        d = make_designer(designer_kind, sc, **kw)
        t0 = time.time()
        oracle_train(d, sc, rounds, batch, w_u, tag=f"seed{seed}", log_every=1)
        rec = oracle.outcome(sc, d.recommend(), w_u=w_u)
        rows.append({"seed": seed, "composite": rec["composite"], "contained": rec["contained"],
                     "utility": rec["utility"], "n_tasks": len(rec["satisfied_tasks"]),
                     "n_open": len(rec["open_channels"]), "config_vector": sc.space.to_vector(rec["config"]),
                     "wall_s": round(time.time() - t0, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Multi-seed oracle training sweep (Phase 1 rigor).")
    ap.add_argument("--scenario", default="datacenter_lite")
    ap.add_argument("--designer", default="mlp", choices=["mlp", "factorized"])
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (0..seeds-1)")
    ap.add_argument("--seed-list", default=None, help="explicit comma-separated seeds (overrides --seeds)")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--w-u", dest="w_u", type=float, default=0.4)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-3, help="within this of ref optimum counts as 'optimal'")
    ap.add_argument("--out", default=None, help="JSONL output; default logs/<scenario>_oracle_sweep.jsonl")
    args = ap.parse_args()

    sc = get(args.scenario)
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    hidden = tuple(int(x) for x in args.hidden.split(","))
    out = args.out or f"logs/{sc.id}_oracle_sweep.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"oracle sweep: {sc.id} | {len(seeds)} seeds x {args.rounds} rounds x {args.batch} | w_u={args.w_u}")
    rows = run_sweep(sc, seeds, args.rounds, args.batch, args.w_u, hidden, args.lr, args.entropy,
                     args.designer)

    # Reference optimum: the analytic best (exact for small spaces, sampled estimate for huge
    # ones) OR the best any seed found -- whichever is higher, so regret is never spuriously
    # negative when the sampled estimate undershoots the true optimum.
    best_est = analysis.best_composite(sc, args.w_u)
    ref_opt = max(best_est["composite"], max(r["composite"] for r in rows))
    for r in rows:
        r["regret"] = round(ref_opt - r["composite"], 4)
        r["optimal"] = r["regret"] <= args.eps

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"scenario": sc.id, "ref_opt": ref_opt,
                             "ref_opt_exact": best_est.get("exact", False),
                             "rounds": args.rounds, "batch": args.batch, "w_u": args.w_u,
                             "eps": args.eps}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    comps = [r["composite"] for r in rows]
    regs = [r["regret"] for r in rows]
    n_opt = sum(r["optimal"] for r in rows)
    n_contained = sum(r["contained"] for r in rows)
    print(f"\n=== oracle sweep summary ({len(rows)} seeds) ===")
    print(f"ref optimum: {ref_opt:.4f} ({'exact' if best_est.get('exact') else 'sampled estimate'})")
    print(f"composite:   {fmt_ci(mean_ci(comps))}   range [{min(comps):.3f}, {max(comps):.3f}]")
    print(f"regret:      {fmt_ci(mean_ci(regs))}")
    print(f"reached optimum (regret<= {args.eps}): {n_opt}/{len(rows)} seeds")
    print(f"strictly contained recommendation:     {n_contained}/{len(rows)} seeds")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
