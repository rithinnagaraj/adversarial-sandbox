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
import secrets
import sys
import time

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


def _fresh_tokens(realize_mod):
    return {ch: f"FLAG-{ch}-{secrets.token_hex(8)}" for ch in realize_mod.CHANNEL_TOKENS}


def _real_reward(sc, cfg, realize_mod, agent, runcfg, instructions, w_u, fake):
    """Run ONE real (or fake) episode for `cfg`; return its graded composite + outcome."""
    from coupled_env.real.episode import CoupledDockerEnv, FakeCoupledEnv, run_episode
    from coupled_env.real.scoring import score
    env = (FakeCoupledEnv(sc, cfg) if fake
           else CoupledDockerEnv(sc, cfg, realize_mod, _fresh_tokens(realize_mod)))
    out = run_episode(sc, env, agent, runcfg, instructions)
    return score(sc, out.breached, out.tasks_done, w_u=w_u)["composite"], out


def real_finetune(designer, sc, args, start_done=0):
    from coupled_env.real import realize as realize_pkg
    from coupled_env.real.episode import CoupledRunConfig
    from episode.agent import ApiAgent, FakeAgent, make_completer

    fake = args.fake_real
    realize_mod = None if fake else realize_pkg.get_realization(sc.id)
    instructions = {} if fake else realize_mod.INSTRUCTIONS
    if fake:
        agent = FakeAgent()
    else:
        model = args.model or {"sarvam": "glm5.3", "openrouter": "z-ai/glm-5.3"}.get(args.provider)
        if not model:
            sys.exit("--model is required for a real run (for --provider bedrock pass your AIP ARN).")
        agent = ApiAgent(complete=make_completer(args.provider, model), max_context_chars=16000)
    runcfg = CoupledRunConfig()
    meta_path = args.checkpoint + ".json"
    done = start_done
    invalids = 0
    t0 = time.time()
    while done < args.real_episodes:
        batch = min(args.real_batch, args.real_episodes - done)
        rollouts = [designer.propose() for _ in range(batch)]
        rewards = []
        for cfg, _info in rollouts:
            rew, out = _real_reward(sc, cfg, realize_mod, agent, runcfg, instructions, args.w_u, fake)
            rewards.append(rew)
            invalids += int(out.invalid_kind is not None)
            done += 1
        designer.update_batch(rollouts, rewards)
        designer.save(args.checkpoint)                       # checkpoint EVERY round
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"scenario": sc.id, "pretrain_done": True, "real_episodes_done": done}, fh)
        rc = oracle.composite(sc, designer.recommend(), w_u=args.w_u)
        print(f"  [real] ep {done:4d}/{args.real_episodes}  batch_mean={np.mean(rewards):+.3f}  "
              f"recommend(oracle)={rc:+.3f}  invalids={invalids}  ckpt={args.checkpoint}")
    print(f"  real fine-tune wall: {time.time() - t0:.0f}s, {invalids} invalid episodes")


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
    args = ap.parse_args()

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
