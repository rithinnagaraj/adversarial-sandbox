"""Tiny stdlib-only statistics for reporting (no numpy/scipy dependency).

Used by the aggregation + held-out-eval tooling so results tables can be produced on any
host (including ones without the training stack installed).
"""

from __future__ import annotations

import math
from typing import Sequence

# t_{0.975, dof} for a two-sided 95% CI. Small-sample honest (k=5 -> dof=4 -> 2.776),
# falling back to the normal 1.96 for large dof. Keeps us off scipy.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060,
        30: 2.042}


def _t95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if dof in _T95:
        return _T95[dof]
    keys = sorted(_T95)
    if dof < keys[-1]:                      # nearest tabulated dof at/above
        return _T95[min(k for k in keys if k >= dof)]
    return 1.96                             # large-sample normal approximation


def mean_ci(xs: Sequence[float], conf: float = 0.95) -> dict:
    """Mean with a two-sided CI (t-based, 95% only). Returns nan bounds for n<2.

    `sd` is the sample standard deviation (the spread of individual episodes — i.e. the
    agent-stochasticity term in a variance decomposition); `sem` is sd/sqrt(n) (uncertainty
    of the mean). `half` is the CI half-width.
    """
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "sem": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "half": float("nan")}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "sem": 0.0,
                "lo": mean, "hi": mean, "half": 0.0}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    half = _t95(n - 1) * sem if conf == 0.95 else 1.96 * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "lo": mean - half, "hi": mean + half, "half": half}


def fmt_ci(ci: dict, prec: int = 3) -> str:
    if ci["n"] == 0:
        return "n/a (0 samples)"
    if ci["n"] == 1:
        return f"{ci['mean']:.{prec}f} (n=1)"
    return f"{ci['mean']:.{prec}f} ± {ci['half']:.{prec}f} (n={ci['n']})"
