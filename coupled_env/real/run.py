"""Run a config on real containers (or a fake env), scoring with graded containment.

    # host dry-run (no Docker, no API) -- verifies orchestration against the oracle:
    python -m coupled_env.real.run --fake

    # real run on the VM (Sarvam GLM), the exact-optimal config, 3 repeats:
    ADVCONTAINER_EXPERIMENT_ENV=1 SARVAM_API_KEY=... python -m coupled_env.real.run \
        --provider sarvam --model glm5.3 --config best --repeats 3

    # or OpenRouter, a specific config vector:
    ADVCONTAINER_EXPERIMENT_ENV=1 OPENROUTER_API_KEY=... python -m coupled_env.real.run \
        --provider openrouter --model z-ai/glm-5.3 --vector 0,1,2,1,0

Real Docker execution is implemented for the pilot-sized `pipeline_box` scenario only.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

from coupled_env import analysis
from coupled_env.real import realize
from coupled_env.real.episode import (CoupledDockerEnv, CoupledRunConfig, FakeCoupledEnv,
                                      run_episode)
from coupled_env.real.scoring import score
from coupled_env.scenarios import get
from episode.agent import ApiAgent, FakeAgent, make_completer


def _fresh_tokens(realize_mod) -> dict[str, str]:
    return {ch: f"FLAG-{ch}-{secrets.token_hex(8)}" for ch in realize_mod.CHANNEL_TOKENS}


def _persist_trace(outcome, rep, log_path):
    """Always store the full per-turn trace (incl. reasoning) of a validation run as JSONL."""
    path = log_path.rsplit(".", 1)[0] + "_trace.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        for turn in outcome.trajectory:
            fh.write(json.dumps({"episode_id": outcome.episode_id, "rep": rep, **turn}) + "\n")
    return path


def _dump_trace(outcome, rep):
    """Print a readable per-turn transcript (only when --trace)."""
    print(f"\n===== episode {outcome.episode_id} rep {rep} "
          f"(error={outcome.error}, invalid={outcome.invalid_kind}) =====")
    for t in outcome.trajectory:
        print(f"[{t['phase']}{'/' + t['task'] if t['task'] else ''} a{t['attempt']}] "
              f"intent={t['intent']} stop_reason={t.get('stop_reason')}")
        if t.get("reasoning"):
            print(f"    reasoning: {t['reasoning'][:800]!r}")
        if t.get("reply"):
            print(f"    reply: {t['reply']!r}")
        if t.get("command"):
            print(f"    cmd  : {t['command']!r}")
        if t.get("observation"):
            print(f"    out  : {t['observation']!r}")
        if t.get("breached_now"):
            print(f"    BREACHED: {t['breached_now']}")


def _pick_config(scenario, args):
    if args.vector:
        return scenario.space.to_dict([int(x) for x in args.vector.split(",")])
    if args.config == "lockdown":
        return scenario.space.lockdown()
    if args.config == "wide_open":
        return scenario.space.wide_open()
    return analysis.best_composite(scenario, args.w_u)["config"]  # exact if small, else sampled


def main():
    ap = argparse.ArgumentParser(description="Run a coupled_env config on real containers.")
    ap.add_argument("--scenario", default="pipeline_box")
    ap.add_argument("--provider", default="bedrock",
                    choices=["bedrock", "bedrock-openai", "jarvis", "sarvam", "openrouter"])
    ap.add_argument("--model", default=None,
                    help="model id; for bedrock pass your AIP ARN, for jarvis the GLM model slug. "
                         "defaults: glm5.3 (sarvam) / z-ai/glm-5.3 (openrouter)")
    ap.add_argument("--config", default="best", choices=["best", "lockdown", "wide_open"])
    ap.add_argument("--vector", default=None, help="explicit config, e.g. 0,1,2,1,0 (overrides --config)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--w-u", dest="w_u", type=float, default=0.4)
    ap.add_argument("--fake", action="store_true", help="oracle-backed dry run: no Docker, no API")
    ap.add_argument("--trace", action="store_true",
                    help="print + save each turn (agent reply, command, stop_reason, output)")
    ap.add_argument("--log", default="logs/coupled_real.jsonl")
    args = ap.parse_args()

    scenario = get(args.scenario)
    config = _pick_config(scenario, args)

    realize_mod = None
    if not args.fake:
        try:
            realize_mod = realize.get_realization(args.scenario)
        except KeyError as e:
            sys.exit(str(e) + " -- use --fake for oracle-backed scenarios.")
        if os.environ.get("ADVCONTAINER_EXPERIMENT_ENV") != "1":
            sys.exit("refusing: set ADVCONTAINER_EXPERIMENT_ENV=1 (VM only).")
        model = args.model or {"sarvam": "glm5.3", "openrouter": "z-ai/glm-5.3",
                               "jarvis": "glm-5.3"}.get(args.provider)
        if not model:
            sys.exit("--model is required here: for --provider bedrock pass your AIP ARN "
                     "(and run `aws sso login` + export AWS_PROFILE/AWS_REGION first).")
        completer = make_completer(args.provider, model)
        agent = ApiAgent(complete=completer, max_context_chars=16000)
    else:
        agent = FakeAgent()
        completer = None

    cfg = CoupledRunConfig()
    print(f"run {args.scenario} | config={config} | {'FAKE' if args.fake else args.provider} "
          f"| repeats={args.repeats} | w_u={args.w_u}")

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    rows, composites = [], []
    with open(args.log, "a", encoding="utf-8") as fh:
        for i in range(args.repeats):
            env = (FakeCoupledEnv(scenario, config) if args.fake
                   else CoupledDockerEnv(scenario, config, realize_mod,
                                         _fresh_tokens(realize_mod)))
            t0 = time.time()
            instructions = realize_mod.INSTRUCTIONS if realize_mod else {}
            outcome = run_episode(scenario, env, agent, cfg, instructions)
            s = score(scenario, outcome.breached, outcome.tasks_done, w_u=args.w_u)
            row = {"scenario": args.scenario, "config": config, "repeat": i,
                   "provider": (None if args.fake else args.provider),
                   "serving": ({k: v for k, v in (getattr(completer, "last_meta", {}) or {}).items()
                                if k != "reasoning"} if completer else {}),
                   "error": outcome.error, "invalid_kind": outcome.invalid_kind,
                   "wall_s": round(time.time() - t0, 1), **s}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            rows.append(row); composites.append(s["composite"])
            print(f"  rep {i}: composite={s['composite']:.3f} contained={s['fully_contained']} "
                  f"tasks={s['tasks_done']} breached={s['breached']} "
                  f"{'ERR='+outcome.error if outcome.error else ''}")
            tpath = _persist_trace(outcome, i, args.log)    # always store reasoning trace
            if args.trace:
                _dump_trace(outcome, i)
    print(f"reasoning/turn trace: {tpath}")

    ok = [c for c, r in zip(composites, rows) if r["invalid_kind"] is None]
    print(f"\nmean composite (valid eps): {round(sum(ok)/len(ok), 4) if ok else 'n/a'}   log: {args.log}")
    if args.fake:
        from coupled_env import oracle
        print(f"oracle composite for this config: {round(oracle.composite(scenario, config, w_u=args.w_u), 4)} "
              f"(fake run should match)")


if __name__ == "__main__":
    main()
