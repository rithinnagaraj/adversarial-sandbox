"""CLI for inspecting, validating and analysing coupled-env scenarios.

    python -m coupled_env.cli list
    python -m coupled_env.cli describe pipeline_box
    python -m coupled_env.cli validate pipeline_box
    python -m coupled_env.cli analyze  pipeline_box [--w-u 0.4]

All commands are pure-Python (no Docker/API/numpy) and fast on modest scenarios.
"""

from __future__ import annotations

import argparse
import sys

from coupled_env import analysis, scenarios, validate


def _fmt_cfg(cfg: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in cfg.items())


def _fmt_size(n: int) -> str:
    return f"{n:,}" if n < 1_000_000_000 else f"{n:.3e} ({n:,})"


def cmd_list(_args) -> int:
    print("available scenarios:")
    for name in scenarios.names():
        sc = scenarios.get(name)
        print(f"  {name:16s} {_fmt_size(sc.space.size()):>22} configs  "
              f"{len(sc.tasks)} tasks  {len(sc.channels)} channels")
    return 0


def cmd_describe(args) -> int:
    sc = scenarios.get(args.scenario)
    print(f"# scenario: {sc.id}")
    print(sc.description)
    print(f"\nconfig space: {sc.space.size():,} configs")
    for k in sc.space.knobs:
        print(f"  {k.name:12s}: {' | '.join(k.levels)}")
    print("\ntasks (requirement predicate):")
    for t in sc.tasks:
        print(f"  [{t.weight:>3}] {t.id:18s} {t.requires.describe()}")
        print(f"        {t.description}")
    print("\nchannels (exposed when):")
    for c in sc.channels:
        print(f"  {c.id:14s} ({c.family:16s}) {c.exposes_when.describe()}")
        print(f"        {c.note}")
    return 0


def cmd_validate(args) -> int:
    sc = scenarios.get(args.scenario)
    checks = validate.run(sc, w_u=args.w_u)
    print(f"# validating scenario: {sc.id}\n")
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"  [{mark}] {c.name}")
        print(f"         {c.detail}")
    ok = validate.all_ok(checks)
    print(f"\noverall: {'ALL PASS -- scenario is a non-trivial coupled problem' if ok else 'FAILURES above'}")
    return 0 if ok else 1


def cmd_analyze(args) -> int:
    sc = scenarios.get(args.scenario)
    s = analysis.summary(sc, w_u=args.w_u, samples=args.samples)
    kind = "exact (brute force)" if s["best_is_exact"] else f"sampled ({args.samples:,} draws)"
    print(f"# analysis: {sc.id}   (w_u={args.w_u})")
    print(f"space: {_fmt_size(s['space_size'])} configs   "
          f"{s['n_channels']} channels   {s['n_tasks']} tasks\n")
    b = s["best"]
    print(f"best config found [{kind}] -- composite {b['composite']:.3f} "
          f"(containment-weighted, not all-or-nothing):")
    print(f"    {_fmt_cfg(s['best_config'])}")
    print(f"    contained={b['contained']}  utility={b['utility']:.3f}")
    print(f"\nbaselines (exact):")
    print(f"    lockdown  : composite={s['lockdown']['composite']}  "
          f"utility={s['lockdown']['utility']}  containment={s['lockdown']['containment']}")
    print(f"    wide_open : composite={s['wide_open']['composite']}  "
          f"utility={s['wide_open']['utility']}  containment={s['wide_open']['containment']}")
    print(f"\ntask verdicts (exact, scale-free):")
    for label in ("forced_tradeoff", "needs_care", "free", "impossible"):
        ids = s["task_labels"].get(label)
        if ids:
            shown = ids[:12]
            more = f"  (+{len(ids) - len(shown)} more)" if len(ids) > len(shown) else ""
            print(f"    {label:16s} ({len(ids):>2}): {', '.join(shown)}{more}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coupled_env.cli", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list scenarios").set_defaults(func=cmd_list)

    for name, fn in [("describe", cmd_describe), ("validate", cmd_validate), ("analyze", cmd_analyze)]:
        p = sub.add_parser(name, help=f"{name} a scenario")
        p.add_argument("scenario")
        p.add_argument("--w-u", dest="w_u", type=float, default=0.4,
                       help="utility weight in the composite reward (default 0.4)")
        if name == "analyze":
            p.add_argument("--samples", type=int, default=200_000,
                           help="random draws for the best-config estimate on huge spaces")
        p.set_defaults(func=fn)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
