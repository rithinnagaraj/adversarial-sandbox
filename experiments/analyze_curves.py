"""Phase B/C analysis: read phase_b_results.json and compare designers *honestly*.

Aggregate means (mean reward, overall containment rate) are confounded by WHERE a
designer spends its episodes: a tabular method still enumerating the restrictive
corner of the space shows high containment without having *learned* anything. The
fair question is instead: how fast does each designer's BEST-SO-FAR config climb
toward the optimum? That is what the learning curve answers.

Prints, per suite, each designer's best-so-far reward at round checkpoints, plus
the round at which it first reached >= target fraction of the best reward anyone
found (a simple-regret / sample-efficiency view). Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def best_so_far_by_suite(curve: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Reconstruct best-so-far per suite from the per-round curve."""
    out: dict[str, list[tuple[int, float]]] = {}
    running: dict[str, float] = {}
    for pt in curve:
        s = pt["suite"]
        running[s] = max(running.get(s, float("-inf")), pt["avg_reward"])
        out.setdefault(s, []).append((pt["round"], running[s]))
    return out


def at_round(series: list[tuple[int, float]], r: int) -> float | None:
    val = None
    for rd, v in series:
        if rd <= r:
            val = v
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="logs/phase_b/phase_b_results.json")
    ap.add_argument("--suite", default="mixed_v0")
    ap.add_argument("--world", default="isolation_gated")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[20, 40, 60, 80, 100, 119])
    ap.add_argument("--target", type=float, default=0.9, help="fraction of global-best to 'reach'")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)
    # support both the per-world layout and a flat list
    if "results_by_world" in data:
        results = data["results_by_world"].get(args.world)
        if results is None:
            raise SystemExit(f"no results for world {args.world!r}; have {list(data['results_by_world'])}")
        print(f"[world={args.world}]")
    else:
        results = data["results"]

    # global best reward anyone achieved on this suite (the yardstick)
    global_best = max(
        (r["best_config_per_suite"].get(args.suite, {}).get("reward", float("-inf")) for r in results),
        default=float("-inf"),
    )
    threshold = args.target * global_best
    print(f"suite={args.suite}  global_best={global_best:.3f}  target={args.target:.0%} -> {threshold:.3f}\n")

    header = ["designer"] + [f"r{c}" for c in args.checkpoints] + [f"reach@{args.target:.0%}"]
    print("  ".join(f"{h:>13s}" for h in header))
    print("  ".join("-" * 13 for _ in header))
    for r in results:
        series = best_so_far_by_suite(r.get("curve", [])).get(args.suite, [])
        row = [r["designer"]]
        for c in args.checkpoints:
            v = at_round(series, c)
            row.append("n/a" if v is None else f"{v:.3f}")
        reach = next((rd for rd, v in series if v >= threshold), None)
        row.append("never" if reach is None else str(reach))
        print("  ".join(f"{str(v):>13s}" for v in row))
    print("\nreach@ = first round best-so-far hit the target (lower is better, = more sample-efficient)")


if __name__ == "__main__":
    main()
