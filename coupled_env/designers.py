"""Learnable designers for the coupled environment (numpy -> run in the WSL venv).

Two policies, trained with REINFORCE against the oracle reward:

- `FactorizedDesigner`: the v1-style baseline. One independent softmax per knob, no
  cross-knob conditioning. It can represent a single best value per knob but NOT a
  correlation like "set mntI=rw for a task, then compensate elsewhere".

- `MLPDesigner`: the upgrade. The config is built **autoregressively** -- one knob at a
  time, each decision conditioned (through a shared multi-layer MLP trunk) on the context
  and the knobs already chosen. A per-knob output head emits that knob's distribution; a
  value head provides a learned baseline (actor-critic). Because it is autoregressive it
  can represent any joint config distribution, so it can learn the knob *correlations* the
  coupled environment rewards. This is the "slightly bigger model".

Both optimise E[composite] where composite = containment + w_u*utility from oracle.py.
Training is pure simulation against the oracle (no Docker / no LLM), so it is fast.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from coupled_env import oracle
from coupled_env.scenario import Scenario


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class _Adam:
    """Minimal Adam over a list of named numpy arrays."""

    def __init__(self, params: dict, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class FactorizedDesigner:
    """Independent per-knob softmax (the baseline that cannot represent correlations)."""

    name = "factorized"

    def __init__(self, scenario: Scenario, lr=0.1, entropy=0.01, baseline_decay=0.9, seed=0):
        self.sc = scenario
        self.knobs = scenario.space.knobs
        self.lr, self.entropy, self.decay = lr, entropy, baseline_decay
        self.rng = np.random.default_rng(seed)
        self.logits = [np.zeros(len(k)) for k in self.knobs]
        self.baseline = 0.0

    def _probs(self):
        return [_softmax(z) for z in self.logits]

    def propose(self):
        probs = self._probs()
        idx = [int(self.rng.choice(len(p), p=p)) for p in probs]
        cfg = {k.name: k.levels[idx[i]] for i, k in enumerate(self.knobs)}
        return cfg, idx

    def recommend(self):
        idx = [int(np.argmax(z)) for z in self.logits]
        return {k.name: k.levels[idx[i]] for i, k in enumerate(self.knobs)}

    def update_batch(self, rollouts, rewards):
        r = np.array(rewards)
        adv = r - self.baseline
        self.baseline = self.decay * self.baseline + (1 - self.decay) * float(r.mean())
        probs = self._probs()
        grads = [np.zeros_like(z) for z in self.logits]
        for b, (_, idx) in enumerate(rollouts):
            for i, p in enumerate(probs):
                onehot = np.zeros_like(p)
                onehot[idx[i]] = 1.0
                # ascend adv*logprob + entropy  ->  grad on logits
                logp = np.log(p + 1e-12)
                H = -(p * logp).sum()
                grads[i] += adv[b] * (onehot - p) - self.entropy * (p * (logp + H))
        for i in range(len(self.logits)):
            self.logits[i] += self.lr * grads[i] / len(rollouts)


class MLPDesigner:
    """Autoregressive policy: shared MLP trunk -> per-knob head, plus a value head."""

    name = "mlp"

    def __init__(self, scenario: Scenario, hidden=(128, 128), lr=3e-3,
                 entropy=0.02, value_coef=0.5, seed=0):
        self.sc = scenario
        self.knobs = scenario.space.knobs
        self.K = len(self.knobs)
        self.Ls = [len(k) for k in self.knobs]
        self.entropy, self.value_coef = entropy, value_coef
        self.rng = np.random.default_rng(seed)

        # input: per knob a block of (L_i + 1): one-hot of chosen level, or the "undecided" slot
        self.blocks = [L + 1 for L in self.Ls]
        self.offs = np.cumsum([0] + self.blocks)
        self.D = int(self.offs[-1])
        H1, H2 = hidden

        def he(shape):  # He init
            return self.rng.standard_normal(shape) * np.sqrt(2.0 / shape[-1])

        self.params = {
            "W1": he((H1, self.D)), "b1": np.zeros(H1),
            "W2": he((H2, H1)), "b2": np.zeros(H2),
            "Wv": np.zeros(H2), "bv": np.zeros(1),
        }
        for i, L in enumerate(self.Ls):
            self.params[f"Wh{i}"] = he((L, H2))
            self.params[f"bh{i}"] = np.zeros(L)
        self.opt = _Adam(self.params, lr=lr)

    # --- input encoding ---
    def _input(self, chosen: Sequence[int], step: int) -> np.ndarray:
        x = np.zeros(self.D)
        for j in range(self.K):
            base = self.offs[j]
            if j < step:
                x[base + chosen[j]] = 1.0       # decided -> one-hot
            else:
                x[base + self.Ls[j]] = 1.0      # undecided marker
        return x

    # --- forward through trunk; returns activations for backprop ---
    def _trunk(self, x):
        z1 = self.params["W1"] @ x + self.params["b1"]
        a1 = _relu(z1)
        z2 = self.params["W2"] @ a1 + self.params["b2"]
        a2 = _relu(z2)
        return {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def _rollout(self, greedy=False):
        chosen: list[int] = []
        steps = []
        for t in range(self.K):
            x = self._input(chosen, t)
            c = self._trunk(x)
            logits = self.params[f"Wh{t}"] @ c["a2"] + self.params[f"bh{t}"]
            p = _softmax(logits)
            v = float(self.params["Wv"] @ c["a2"] + self.params["bv"][0])
            a = int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p))
            chosen.append(a)
            steps.append({"cache": c, "p": p, "a": a, "v": v})
        cfg = {k.name: k.levels[chosen[i]] for i, k in enumerate(self.knobs)}
        return cfg, chosen, steps

    def propose(self):
        cfg, chosen, steps = self._rollout(greedy=False)
        return cfg, (chosen, steps)

    def recommend(self):
        cfg, _, _ = self._rollout(greedy=True)
        return cfg

    def _zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.params.items()}

    def update_batch(self, rollouts, rewards):
        grads = self._zero_grads()
        R = np.array(rewards)
        for b, (_, (chosen, steps)) in enumerate(rollouts):
            for t, st in enumerate(steps):
                c, p, a, v = st["cache"], st["p"], st["a"], st["v"]
                adv = R[b] - v
                # policy + entropy gradient on logits_t
                onehot = np.zeros_like(p)
                onehot[a] = 1.0
                logp = np.log(p + 1e-12)
                H = -(p * logp).sum()
                dlogits = adv * (p - onehot) + self.entropy * (p * (logp + H))
                # value gradient
                dv = self.value_coef * (v - R[b])
                # head backprop
                grads[f"Wh{t}"] += np.outer(dlogits, c["a2"])
                grads[f"bh{t}"] += dlogits
                da2 = self.params[f"Wh{t}"].T @ dlogits
                grads["Wv"] += dv * c["a2"]
                grads["bv"] += dv
                da2 = da2 + self.params["Wv"] * dv
                # trunk backprop
                dz2 = da2 * (c["z2"] > 0)
                grads["W2"] += np.outer(dz2, c["a1"])
                grads["b2"] += dz2
                da1 = self.params["W2"].T @ dz2
                dz1 = da1 * (c["z1"] > 0)
                grads["W1"] += np.outer(dz1, c["x"])
                grads["b1"] += dz1
        n = len(rollouts)
        for k in grads:
            grads[k] /= n
        self.opt.step(grads)


def make_designer(kind: str, scenario: Scenario, **kw):
    if kind == "mlp":
        return MLPDesigner(scenario, **kw)
    if kind == "factorized":
        return FactorizedDesigner(scenario, **kw)
    raise ValueError(f"unknown designer {kind!r}")
