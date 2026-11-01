"""Phase 0/1: aggregate the JSONL logs into summary tables (stdlib only -- runs anywhere).

Three log kinds, auto-detected or forced with --kind:
  oracle   : multi-seed oracle sweep (coupled_env.sweep_oracle)        -> regret distribution
  finetune : ordered per-episode real training log (train_designer)    -> convergence curve
  bank     : realized-spec sample bank (coupled_env.real.cache)        -> per-spec mean +/- CI

    python -m coupled_env.aggregate logs/dclite_oracle_sweep.jsonl
    python -m coupled_env.aggregate logs/dclite_glm_ckpt_episodes.jsonl --kind finetune
    python -m coupled_env.aggregate logs/datacenter_lite_bank.jsonl --window 25
"""

from __future__ import annotations

import argparse
import json

from coupled_env.stats import fmt_ci, mean_ci


def _read(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detect(rows):
    if not rows:
        return "empty"
    if any("regret" in r for r in rows) or (rows and "ref_opt" in rows[0]):
        return "oracle"
    if rows and "key" in rows[0] and "sample" in rows[0]:
        return "bank"
    if any("episode" in r for r in rows):
        return "finetune"
    return "unknown"


def summarize_oracle(rows):
    header = rows[0] if "ref_opt" in rows[0] else {}
    data = [r for r in rows if "regret" in r]
    comps = [r["composite"] for r in data]
    regs = [r["regret"] for r in data]
    n_opt = sum(r.get("optimal") for r in data)
    n_cont = sum(r.get("contained") for r in data)
    print(f"oracle sweep: {len(data)} seeds")
    if header:
        print(f"ref optimum: {header['ref_opt']:.4f} "
              f"({'exact' if header.get('ref_opt_exact') else 'sampled estimate'})")
    print(f"composite:   {fmt_ci(mean_ci(comps))}   range [{min(comps):.3f}, {max(comps):.3f}]")
    print(f"regret:      {fmt_ci(mean_ci(regs))}")
    print(f"reached optimum: {n_opt}/{len(data)} seeds   strictly contained: {n_cont}/{len(data)}")


def summarize_finetune(rows, window):
    data = [r for r in rows if "composite" in r]
    data.sort(key=lambda r: r.get("episode", 0))
    comps = [r["composite"] for r in data]
    invalid = sum(r.get("invalid_kind") is not None for r in data)
    print(f"fine-tune episodes: {len(data)}   invalid: {invalid}")
    # windowed moving average (convergence trace)
    print(f"\nconvergence (moving mean of composite, window={window}):")
    step = max(1, len(data) // 12)
    for end in range(window, len(data) + 1, step):
        w = comps[end - window:end]
        print(f"  ep {end:4d}: mean={sum(w) / len(w):+.3f}")
    if comps:
        tail = comps[-window:]
        print(f"\nfinal window ({len(tail)} eps): {fmt_ci(mean_ci(tail))}")


def summarize_bank(rows, top):
    by_key = {}
    cfg_of = {}
    for r in rows:
        s = r["sample"]
        if s.get("invalid_kind") is not None:
            continue
        by_key.setdefault(r["key"], []).append(s["composite"])
        cfg_of[r["key"]] = r.get("config")
    print(f"sample bank: {len(by_key)} distinct realized specs, "
          f"{sum(len(v) for v in by_key.values())} valid samples")
    ranked = sorted(by_key.items(), key=lambda kv: -mean_ci(kv[1])["mean"])
    print(f"\ntop {min(top, len(ranked))} specs by mean composite:")
    for key, comps in ranked[:top]:
        print(f"  {key}  {fmt_ci(mean_ci(comps))}  vector={cfg_of.get(key)}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate coupled_env JSONL logs into tables.")
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["oracle", "finetune", "bank"], default=None)
    ap.add_argument("--window", type=int, default=25, help="moving-average window (finetune)")
    ap.add_argument("--top", type=int, default=10, help="top-N specs to show (bank)")
    args = ap.parse_args()

    rows = _read(args.path)
    kind = args.kind or _detect(rows)
    print(f"# {args.path}  (kind={kind})\n")
    if kind == "oracle":
        summarize_oracle(rows)
    elif kind == "finetune":
        summarize_finetune(rows, args.window)
    elif kind == "bank":
        summarize_bank(rows, args.top)
    else:
        print(f"cannot summarize (detected kind={kind!r}); pass --kind explicitly.")


if __name__ == "__main__":
    main()
