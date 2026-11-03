"""Phase 1 (rigor): held-out REAL evaluation of named configs with mean +/- CI.

Runs each of a set of configs (the learned recommendation + baselines) for k real episodes,
pooling with any i.i.d. samples already in the bank (valid because the agent is frozen and the
container is rebuilt identically given the realized spec). Reports, per config: composite mean
+/- CI, containment, utility, per-channel breach rate, fully-contained rate, the oracle score,
and the behaviour-only sim-to-real gap (oracle - real mean). Ends with a variance decomposition
(within-config = agent stochasticity, between-config = how separated the configs are).

    # host dry-run against the fake env (no Docker/LLM) -- exercises the whole path:
    python -m coupled_env.real.holdout_eval --scenario datacenter_lite --fake --k 5 \
        --configs lockdown,wide_open,best

    # real, on the VM (GLM via JarvisLabs), incl. the trained policy's recommendation:
    ADVCONTAINER_EXPERIMENT_ENV=1 JARVISLABS_API_KEY=... \
    ~/advc-venv/bin/python -m coupled_env.real.holdout_eval --scenario datacenter_lite \
        --provider jarvis --model glm-5.3 --k 5 \
        --configs learned,lockdown,wide_open --checkpoint logs/dclite_glm_ckpt.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from coupled_env import analysis, oracle
from coupled_env.real import realize as realize_pkg
from coupled_env.real.cache import RealEvalCache, realized_key
from coupled_env.real.episode import CoupledRunConfig
from coupled_env.real.runner import make_agent, outcome_sample, run_one
from coupled_env.scenarios import get
from coupled_env.stats import fmt_ci, mean_ci


def _random_config(sc, seed):
    rng = random.Random(seed)
    return {k.name: rng.choice(k.levels) for k in sc.space.knobs}


def _learned_config(sc, checkpoint, hidden):
    from coupled_env.designers import make_designer
    d = make_designer("mlp", sc, hidden=hidden, seed=0)
    d.load(checkpoint)
    return d.recommend()


def resolve_configs(sc, names, checkpoint, hidden, w_u, seed):
    out = {}
    for name in names:
        if name == "lockdown":
            out[name] = sc.space.lockdown()
        elif name == "wide_open":
            out[name] = sc.space.wide_open()
        elif name == "random":
            out[name] = _random_config(sc, seed)
        elif name == "best":
            out[name] = analysis.best_composite(sc, w_u)["config"]
        elif name == "learned":
            if not checkpoint or not os.path.exists(checkpoint):
                sys.exit(f"--configs includes 'learned' but checkpoint {checkpoint!r} not found")
            out[name] = _learned_config(sc, checkpoint, hidden)
        else:
            sys.exit(f"unknown config name {name!r} (use learned/lockdown/wide_open/random/best)")
    return out


def evaluate_config(sc, name, cfg, realize_mod, agent, runcfg, instructions, w_u, fake,
                    k, cache, log_path, seed):
    key = realized_key(sc, cfg, realize_mod)
    have = cache.n_valid(key)
    need = max(0, k - have)
    for _ in range(need):
        scored, out, wall = run_one(sc, cfg, realize_mod, agent, runcfg, instructions, w_u, fake)
        cache.add(key, outcome_sample(out, scored, wall), config=sc.space.to_vector(cfg))
        if out.invalid_kind:                              # surface the reason instead of silently dropping
            print(f"  ! invalid episode ({out.invalid_kind}): {(out.error or '')[:200]}")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"config_name": name, "seed": seed, "key": key,
                                 "composite": scored["composite"], "breached": scored["breached"],
                                 "tasks_done": scored["tasks_done"],
                                 "invalid_kind": out.invalid_kind, "error": out.error}) + "\n")
    samples = cache.valid_samples(key)
    comps = [s["composite"] for s in samples]
    ci = mean_ci(comps)
    n = len(samples)
    breach_rate = {}
    if n:
        for ch in sc.channels:
            breach_rate[ch.id] = round(sum(ch.id in s["breached"] for s in samples) / n, 3)
    contained_rate = round(sum(not s["breached"] for s in samples) / n, 3) if n else float("nan")
    orc = oracle.composite(sc, cfg, w_u=w_u)
    return {"name": name, "config_vector": sc.space.to_vector(cfg), "key": key,
            "ci": ci, "n": n, "comps": comps,
            "containment_mean": round(sum(s["containment"] for s in samples) / n, 4) if n else None,
            "utility_mean": round(sum(s["utility"] for s in samples) / n, 4) if n else None,
            "contained_rate": contained_rate, "breach_rate": breach_rate,
            "oracle_composite": round(orc, 4), "gap_oracle_minus_real": round(orc - ci["mean"], 4) if n else None}


def main():
    ap = argparse.ArgumentParser(description="Held-out real evaluation with mean +/- CI (Phase 1).")
    ap.add_argument("--scenario", default="datacenter_lite")
    ap.add_argument("--provider", default="jarvis",
                    choices=["bedrock", "bedrock-openai", "jarvis", "sarvam", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--configs", default="learned,lockdown,wide_open",
                    help="comma list from: learned,lockdown,wide_open,random,best")
    ap.add_argument("--checkpoint", default="logs/dclite_glm_ckpt.npz")
    ap.add_argument("--hidden", default="128,128", help="MLP hidden dims (must match training)")
    ap.add_argument("--k", type=int, default=5, help="target valid episodes per config (pooled with bank)")
    ap.add_argument("--w-u", dest="w_u", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0, help="seed for the 'random' config")
    ap.add_argument("--fake", action="store_true", help="oracle-backed dry run (no Docker/API)")
    ap.add_argument("--cache", default=None, help="sample bank JSONL; default logs/<scenario>_bank.jsonl")
    ap.add_argument("--log", default=None, help="held-out episode log; default logs/<scenario>_holdout.jsonl")
    args = ap.parse_args()

    sc = get(args.scenario)
    names = [s.strip() for s in args.configs.split(",") if s.strip()]
    hidden = tuple(int(x) for x in args.hidden.split(","))
    cache_path = args.cache or f"logs/{sc.id}_bank.jsonl"
    log_path = args.log or f"logs/{sc.id}_holdout.jsonl"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    realize_mod = None
    if not args.fake:
        realize_mod = realize_pkg.get_realization(sc.id)
        if os.environ.get("ADVCONTAINER_EXPERIMENT_ENV") != "1":
            sys.exit("refusing real run: set ADVCONTAINER_EXPERIMENT_ENV=1 (VM only), or use --fake.")
        # fail fast (before any episode) if the provider's API key is missing -- otherwise every
        # episode harness-faults on its first model call and silently yields 0 samples.
        key_env = {"jarvis": "JARVISLABS_API_KEY", "sarvam": "SARVAM_API_KEY",
                   "openrouter": "OPENROUTER_API_KEY", "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
                   "bedrock-openai": "AWS_BEARER_TOKEN_BEDROCK"}.get(args.provider)
        if key_env and not os.environ.get(key_env):
            sys.exit(f"refusing real run: ${key_env} is not set in this shell. Load your key file "
                     f"first (e.g. `set -a; . ~/.jarvis.env; set +a`); do not paste the key on the "
                     f"command line.")
        model = args.model or {"sarvam": "glm5.3", "openrouter": "z-ai/glm-5.3",
                               "jarvis": "glm-5.3"}.get(args.provider)
        if not model:
            sys.exit("--model is required for a real run.")
    else:
        model = None
    agent, _ = make_agent(args.provider, model, args.fake)

    instructions = realize_mod.INSTRUCTIONS if realize_mod else {}
    runcfg = CoupledRunConfig()
    cache = RealEvalCache(cache_path)
    configs = resolve_configs(sc, names, args.checkpoint, hidden, args.w_u, args.seed)

    print(f"held-out eval: {sc.id} | {'FAKE' if args.fake else args.provider} | k={args.k} | "
          f"configs={names} | bank={cache_path}")
    results = []
    for name, cfg in configs.items():
        r = evaluate_config(sc, name, cfg, realize_mod, agent, runcfg, instructions, args.w_u,
                            args.fake, args.k, cache, log_path, args.seed)
        results.append(r)
        print(f"\n[{name}] vector={r['config_vector']}")
        print(f"  composite (real):  {fmt_ci(r['ci'])}")
        print(f"  oracle composite:  {r['oracle_composite']}   gap(oracle-real)={r['gap_oracle_minus_real']}")
        print(f"  containment={r['containment_mean']}  utility={r['utility_mean']}  "
              f"fully-contained rate={r['contained_rate']}")
        leaks = {c: rate for c, rate in (r["breach_rate"] or {}).items() if rate > 0}
        print(f"  per-channel breach rate (>0): {leaks or 'none'}")

    # variance decomposition: within-config (agent stochasticity) vs between-config (separation)
    within = [r["ci"]["sd"] for r in results if r["n"] >= 2]
    means = [r["ci"]["mean"] for r in results if r["n"] >= 1]
    print("\n=== variance decomposition ===")
    if within:
        print(f"within-config sd (agent stochasticity), mean over configs: "
              f"{sum(within) / len(within):.4f}")
    if len(means) >= 2:
        print(f"between-config sd of means (separation): {mean_ci(means)['sd']:.4f}")
    print(f"bank: {cache_path}   log: {log_path}")


if __name__ == "__main__":
    main()
