"""Static RL designer: a learnable contextual policy (the swap for search).

This is the third v0->v1 swap: replace the search/BayesOpt designer with a small
*learnable network*. It keeps the v0 formulation exactly -- one config per episode,
conditioned on the task suite -- so it drops into the same loop and is directly
comparable to random / BayesOpt.

Policy: a linear-softmax contextual policy. For context features phi(s), each of the
5 knobs d has weights W_d (features x levels_d); a level is sampled from
softmax(W_d . phi). Trained with REINFORCE and a per-context reward baseline:

    theta_d += lr * (r - b(s)) * grad_theta_d log pi_d(a_d | s)   + entropy bonus

The network is deliberately tiny (a few hundred weights) -- the expensive thing is
the episode, not the update, so this trains fine on CPU. numpy is imported lazily so
the stdlib-only host path and tests are unaffected (runs in the WSL venv).
"""

from __future__ import annotations

from typing import Any

from config.space import ConfigSpace, DEFAULT_SPACE
from designer.interface import Designer, DesignerResult

try:
    import numpy as _np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


class _FeatureMap:
    """Context -> feature vector. One-hot over a suite vocabulary + bias.

    Kept small and swappable; richer features (per-knob requirement profile) would
    let the policy generalize to unseen suites, a later enhancement.
    """

    def __init__(self, suites: tuple[str, ...]):
        self.suites = list(suites)

    @property
    def dim(self) -> int:
        return len(self.suites) + 1  # + bias

    def __call__(self, context: dict[str, Any]):
        v = _np.zeros(self.dim, dtype=float)
        v[0] = 1.0  # bias
        suite = str(context.get("suite", ""))
        if suite in self.suites:
            v[1 + self.suites.index(suite)] = 1.0
        return v


class StaticRLDesigner(Designer):
    name = "rl_static"

    def __init__(
        self,
        space: ConfigSpace = DEFAULT_SPACE,
        suites: tuple[str, ...] = ("mixed_v0", "light_v0"),
        lr: float = 0.2,
        entropy: float = 0.01,
        baseline_decay: float = 0.9,
        seed: int | None = None,
    ):
        if not _HAVE_NUMPY:
            raise RuntimeError("StaticRLDesigner needs numpy (run in the WSL venv, not the host).")
        self.space = space
        self.lr = lr
        self.entropy = entropy
        self.baseline_decay = baseline_decay
        self.rng = _np.random.default_rng(seed)
        self.phi = _FeatureMap(suites)
        # one weight matrix per knob: (features x levels_d)
        self.W = [_np.zeros((self.phi.dim, len(d.levels))) for d in space.dimensions]
        self._baseline: dict[str, float] = {}
        self._best: dict[str, tuple[float, list[int]]] = {}

    # --- policy math ---
    def _probs(self, feats):
        out = []
        for W in self.W:
            logits = feats @ W
            logits -= logits.max()
            e = _np.exp(logits)
            out.append(e / e.sum())
        return out

    def propose(self, context: dict[str, Any]) -> list[int]:
        feats = self.phi(context)
        probs = self._probs(feats)
        return [int(self.rng.choice(len(p), p=p)) for p in probs]

    def recommend(self, context: dict[str, Any]) -> list[int]:
        """The designer's current best guess (greedy argmax), separate from exploring."""
        feats = self.phi(context)
        return [int(p.argmax()) for p in self._probs(feats)]

    def update(self, result: DesignerResult) -> None:
        if not result.usable:
            return
        key = str(result.context.get("suite", "_"))
        feats = self.phi(result.context)
        probs = self._probs(feats)
        b = self._baseline.get(key, 0.0)
        advantage = result.reward - b
        for d, (W, p, a) in enumerate(zip(self.W, probs, result.vector)):
            onehot = _np.zeros_like(p)
            onehot[a] = 1.0
            # REINFORCE gradient of log pi plus an entropy bonus, both as outer products
            grad_logp = _np.outer(feats, onehot - p)
            grad_entropy = _np.outer(feats, -p * (_np.log(p + 1e-12) + 1.0))
            self.W[d] += self.lr * (advantage * grad_logp + self.entropy * grad_entropy)
        # update the per-context reward baseline (variance reduction)
        self._baseline[key] = self.baseline_decay * b + (1 - self.baseline_decay) * result.reward
        best = self._best.get(key)
        if best is None or result.reward > best[0]:
            self._best[key] = (result.reward, list(result.vector))

    def best_for(self, context: dict[str, Any]) -> list[int] | None:
        found = self._best.get(str(context.get("suite", "_")))
        return list(found[1]) if found else None
