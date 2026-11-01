"""Train the MLP designer against a scenario -- oracle (fast sim) or real containers.

Oracle-only (fast, free; develop the policy):
    ~/advc-venv/bin/python -m coupled_env.train_designer --scenario datacenter_lite \
        --designer mlp --rounds 400 --batch 64

Sim-to-real: oracle PRE-TRAIN to initialize the MLP, then FINE-TUNE on real-container +
real-LLM episodes at a bounded budget. Reward comes from the real executor (graded
containment). Checkpoints every round so a crash resumes mid-run.
    # host dry-run of the real loop (no Docker/boto3): swaps in the oracle-backed fake env
    ~/advc-venv/bin/python -m coupled_env.train_designer --scenario datacenter_lite --real \
        --fake-real --real-episodes 20 --pretrain-rounds 30
    # the real thing on the VM (Claude via Bedrock AIP):
    AWS_PROFILE=advc AWS_REGION=... ADVCONTAINER_EXPERIMENT_ENV=1 \
    ~/advc-venv/bin/python -m coupled_env.train_designer --scenario datacenter_lite --real \
        --provider bedrock --model "<AIP_ARN>" --real-episodes 300 --real-batch 4 \
        --checkpoint logs/dclite_ckpt.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from coupled_env import analysis, oracle
from coupled_env.designers import make_designer
from coupled_env.scenarios import get


def oracle_train(designer, sc, rounds, batch, w_u, tag="oracle", log_every=15):
    for r in range(rounds):
        rollouts = [designer.propose() for _ in range(batch)]
        rewards = [oracle.composite(sc, cfg, w_u=w_u) for cfg, _ in rollouts]
        designer.update_batch(rollouts, rewards)
        if r % max(1, rounds // log_every) == 0 or r == rounds - 1:
            rc = oracle.composite(sc, designer.recommend(), w_u=w_u)
            print(f"  [{tag}] round {r:4d}  batch_mean={np.mean(rewards):+.3f}  recommend={rc:+.3f}")


def _uses_shared_service(sc, cfg, realize_mod) -> bool:
    """True if this config's placement execs a SHARED singleton container (the honeypot).
    Such episodes must be serialized under parallelism (the token write would race)."""
    if realize_mod is None:
        return False
    open_ids = {c.id for c in sc.open_channels(cfg)}
    sentinel = {ch: "X" for ch in realize_mod.CHANNEL_TOKENS}
    return any(c and c[0] == "exec" for c in realize_mod.placement_commands(sentinel, open_ids))


def _episode_record(sc, cfg, scored, out, seed, episode_idx):
    """Ordered per-episode training-log line (for convergence/aggregation)."""
    return {"seed": seed, "episode": episode_idx,
            "config_vector": sc.space.to_vector(cfg),
            "composite": scored["composite"], "containment": scored["containment"],
            "utility": scored["utility"], "breached": scored["breached"],
            "tasks_done": scored["tasks_done"], "invalid_kind": out.invalid_kind,
            "episode_id": out.episode_id}


def real_finetune(designer, sc, args, start_done=0):
    from coupled_env.real import realize as realize_pkg
    from coupled_env.real.cache import RealEvalCache, realized_key
    from coupled_env.real.episode import CoupledRunConfig
    from coupled_env.real.runner import make_agent, outcome_sample, run_one

    fake = args.fake_real
    realize_mod = None if fake else realize_pkg.get_realization(sc.id)
    instructions = {} if fake else realize_mod.INSTRUCTIONS
    model = None
    if not fake:
        model = args.model or {"sarvam": "glm5.3", "openrouter": "z-ai/glm-5.3",
                               "jarvis": "glm-5.3"}.get(args.provider)
        if not model:
            sys.exit("--model is required for a real run (for --provider bedrock pass your AIP ARN).")
    shared_agent, _ = make_agent(args.provider, model, fake)   # reused in the serial path
    runcfg = CoupledRunConfig()
    parallel = max(1, args.parallel)
    honeypot_lock = threading.Lock()

    cache = RealEvalCache(args.cache)
    ep_log = args.episode_log
    os.makedirs(os.path.dirname(ep_log) or ".", exist_ok=True)
    meta_path = args.checkpoint + ".json"

    def evaluate(cfg, episode_idx):
        """One episode -> (composite, outcome). Fresh isolated namespace + agent when parallel;
        always appends the outcome to the sample bank + the ordered episode log."""
        if parallel > 1:
            ns = f"ep{episode_idx}"
            agent, _ = make_agent(args.provider, model, fake)
        else:
            ns, agent = "", shared_agent
        if _uses_shared_service(sc, cfg, realize_mod):
            with honeypot_lock:
                scored, out, wall = run_one(sc, cfg, realize_mod, agent, runcfg,
                                            instructions, args.w_u, fake, namespace=ns)
        else:
            scored, out, wall = run_one(sc, cfg, realize_mod, agent, runcfg,
                                        instructions, args.w_u, fake, namespace=ns)
        key = realized_key(sc, cfg, realize_mod)
        cache.add(key, outcome_sample(out, scored, wall), config=sc.space.to_vector(cfg))
        with open(ep_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_episode_record(sc, cfg, scored, out, args.seed, episode_idx)) + "\n")
        return scored["composite"], out

    done = start_done
    invalids = 0
    t0 = time.time()
    while done < args.real_episodes:
        batch = min(args.real_batch, args.real_episodes - done)
        rollouts = [designer.propose() for _ in range(batch)]
        idxs = list(range(done + 1, done + 1 + batch))
        if parallel > 1 and not fake and batch > 1:
            with ThreadPoolExecutor(max_workers=min(parallel, batch)) as ex:
                results = list(ex.map(lambda a: evaluate(a[0][0], a[1]),
                                      zip(rollouts, idxs)))
        else:
            results = [evaluate(cfg, i) for (cfg, _info), i in zip(rollouts, idxs)]
        rewards = [rew for rew, _ in results]
        invalids += sum(out.invalid_kind is not None for _, out in results)
        done += batch
        designer.update_batch(rollouts, rewards)
        designer.save(args.checkpoint)                       # checkpoint EVERY round
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"scenario": sc.id, "pretrain_done": True, "real_episodes_done": done,
                       "seed": args.seed}, fh)
        rc = oracle.composite(sc, designer.recommend(), w_u=args.w_u)
        print(f"  [real] ep {done:4d}/{args.real_episodes}  batch_mean={np.mean(rewards):+.3f}  "
              f"recommend(oracle)={rc:+.3f}  invalids={invalids}  par={parallel}  ckpt={args.checkpoint}")
    print(f"  real fine-tune wall: {time.time() - t0:.0f}s, {invalids} invalid episodes")
    print(f"  sample bank: {args.cache}   episode log: {ep_log}")


def main():
    ap = argparse.ArgumentParser(description="Train the MLP designer (oracle or real).")
    ap.add_argument("--scenario", default="pipeline_box")
    ap.add_argument("--designer", default="mlp", choices=["mlp", "factorized"])
    ap.add_argument("--rounds", type=int, default=300, help="oracle rounds (oracle-only mode)")
    ap.add_argument("--batch", type=int, default=64, help="oracle batch size")
    ap.add_argument("--w-u", dest="w_u", type=float, default=0.4)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # --- real sim-to-real mode ---
    ap.add_argument("--real", action="store_true", help="oracle-pretrain then real-episode fine-tune")
    ap.add_argument("--real-episodes", dest="real_episodes", type=int, default=300)
    ap.add_argument("--real-batch", dest="real_batch", type=int, default=4,
                    help="proposals evaluated per update during real fine-tune")
    ap.add_argument("--pretrain-rounds", dest="pretrain_rounds", type=int, default=150)
    ap.add_argument("--provider", default="bedrock",
                    choices=["bedrock", "bedrock-openai", "jarvis", "sarvam", "openrouter"])
    ap.add_argument("--model", default=None,
                    help="model id; bedrock=AIP ARN, jarvis=GLM model slug")
    ap.add_argument("--checkpoint", default="logs/designer_ckpt.npz")
    ap.add_argument("--resume", action="store_true", help="resume from --checkpoint (skip pretrain if done)")
    ap.add_argument("--fake-real", dest="fake_real", action="store_true",
                    help="use the oracle-backed fake env for the 'real' loop (host-testable, no Docker/LLM)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent real episodes per batch (>1 uses per-episode isolated volumes; "
                         "honeypot-opening configs are auto-serialized). 1 = validated serial pipeline.")
    ap.add_argument("--cache", default=None,
                    help="real-episode sample bank (JSONL, keyed on realized spec); "
                         "default <checkpoint-stem>_bank.jsonl")
    ap.add_argument("--episode-log", dest="episode_log", default=None,
                    help="ordered per-episode training log (JSONL); default <checkpoint-stem>_episodes.jsonl")
    args = ap.parse_args()
    stem = args.checkpoint.rsplit(".", 1)[0]
    if args.cache is None:
        args.cache = stem + "_bank.jsonl"
    if args.episode_log is None:
        args.episode_log = stem + "_episodes.jsonl"

    sc = get(args.scenario)
    kind = "mlp" if args.real else args.designer      # real mode needs the MLP (checkpoint + rollout cache)
    kw = {"seed": args.seed}
    if kind == "mlp":
        kw["hidden"] = tuple(int(x) for x in args.hidden.split(","))
    if args.lr is not None:
        kw["lr"] = args.lr
    if args.entropy is not None:
        kw["entropy"] = args.entropy
    designer = make_designer(kind, sc, **kw)

    print(f"train {kind} on {sc.id} | {sc.space.size():.2e} configs, {len(sc.channels)} channels, "
          f"{len(sc.tasks)} tasks | w_u={args.w_u}" + ("  [REAL]" if args.real else ""))

    # ---------------- real sim-to-real mode ----------------
    if args.real:
        if not args.fake_real and os.environ.get("ADVCONTAINER_EXPERIMENT_ENV") != "1":
            sys.exit("refusing real run: set ADVCONTAINER_EXPERIMENT_ENV=1 (or use --fake-real).")
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        start_done, pretrained = 0, False
        if args.resume and os.path.exists(args.checkpoint):
            designer.load(args.checkpoint)
            if os.path.exists(args.checkpoint + ".json"):
                meta = json.load(open(args.checkpoint + ".json", encoding="utf-8"))
                start_done = meta.get("real_episodes_done", 0)
                pretrained = meta.get("pretrain_done", False)
            print(f"resumed from {args.checkpoint}: real_episodes_done={start_done}, pretrained={pretrained}")
        if not pretrained:
            print(f"oracle pre-train: {args.pretrain_rounds} rounds x {args.batch} (fast/free)")
            oracle_train(designer, sc, args.pretrain_rounds, args.batch, args.w_u, tag="pretrain")
            designer.save(args.checkpoint)
            with open(args.checkpoint + ".json", "w", encoding="utf-8") as fh:
                json.dump({"scenario": sc.id, "pretrain_done": True, "real_episodes_done": 0}, fh)
        print(f"real fine-tune: budget {args.real_episodes} episodes, batch {args.real_batch}, "
              f"provider={'FAKE' if args.fake_real else args.provider}")
        real_finetune(designer, sc, args, start_done=start_done)
        rec_out = oracle.outcome(sc, designer.recommend(), w_u=args.w_u)
        print(f"\n=== result ===")
        print(f"LEARNED recommendation (oracle-scored): composite={rec_out['composite']} "
              f"contained={rec_out['contained']} utility={rec_out['utility']} "
              f"tasks {len(rec_out['satisfied_tasks'])}/{len(sc.tasks)} "
              f"channels_open {len(rec_out['open_channels'])}/{len(sc.channels)}")
        print(f"checkpoint: {args.checkpoint}")
        return

    # ---------------- oracle-only mode (unchanged) ----------------
    episodes = args.rounds * args.batch
    best_seen = {"composite": -1e9}
    t0 = time.time()
    oracle_train(designer, sc, args.rounds, args.batch, args.w_u)
    rec_out = oracle.outcome(sc, designer.recommend(), w_u=args.w_u)
    lock = oracle.composite(sc, sc.space.lockdown(), w_u=args.w_u)
    wide = oracle.composite(sc, sc.space.wide_open(), w_u=args.w_u)
    rand = analysis.sampled_best(sc, episodes, args.w_u, seed=args.seed + 7)
    print(f"\n=== result ({time.time() - t0:.1f}s) ===")
    print(f"LEARNED recommendation: composite={rec_out['composite']}  contained={rec_out['contained']}  "
          f"utility={rec_out['utility']}  tasks {len(rec_out['satisfied_tasks'])}/{len(sc.tasks)}  "
          f"channels_open {len(rec_out['open_channels'])}/{len(sc.channels)}")
    print(f"baselines: lockdown={lock:.3f}  wide_open={wide:.3f}  "
          f"random({episodes})={rand['composite']:.3f}(contained={rand['contained']})")
    if sc.space.size() <= analysis.EXACT_LIMIT:
        ex = analysis.exact_best(sc, args.w_u)
        print(f"  EXACT optimum composite={ex['composite']:.3f} contained={ex['contained']}")


if __name__ == "__main__":
    main()
