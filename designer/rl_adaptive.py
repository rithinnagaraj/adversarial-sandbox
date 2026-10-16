"""Adaptive RL designer: trajectory-conditioned policy (the v1 adversary).

Two policy heads, both linear-softmax, trained together with REINFORCE on the
episode's terminal reward (a per-context baseline reduces variance):

  * initial head -- at container creation, picks all 5 knobs from context features
    (same form as the static designer).
  * reconfigure head -- at each mid-episode decision point, for each RUNTIME-MUTABLE
    knob it picks {hold, loosen} from behavioural-telemetry features. The runner
    enforces loosen-only (grant, never revoke), so the clamp exploit is impossible;
    this head only decides *when to grant* more access.

The designer never sees the phase or which objective is next -- only telemetry -- so
it cannot special-case the flag step. numpy, lazily required (WSL).
"""

from __future__ import annotations

from typing import Any

from config.space import ConfigSpace, DEFAULT_SPACE

try:
    import numpy as _np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

_RECONFIG_ACTIONS = ("hold", "loosen")


class AdaptiveRLDesigner:
    name = "rl_adaptive"

    def __init__(self, space: ConfigSpace = DEFAULT_SPACE,
                 suites: tuple[str, ...] = ("mixed_v0", "light_v0"),
                 lr: float = 0.1, entropy: float = 0.01, baseline_decay: float = 0.9,
                 seed: int | None = None):
        if not _HAVE_NUMPY:
            raise RuntimeError("AdaptiveRLDesigner needs numpy (run in the WSL venv).")
        self.space = space
        self.lr, self.entropy, self.baseline_decay = lr, entropy, baseline_decay
        self.rng = _np.random.default_rng(seed)
        self.suites = list(suites)
        self.mutable = space.mutable_dims()

        self._ctx_dim = len(suites) + 1                      # initial-head features
        self._tel_dim = 3 + len(self.mutable)                # telemetry features
        self.Winit = [_np.zeros((self._ctx_dim, len(d.levels))) for d in space.dimensions]
        self.Wrec = {k: _np.zeros((self._tel_dim, len(_RECONFIG_ACTIONS))) for k in self.mutable}
        self._baseline = 0.0
        # per-episode log of (which_head, key, feats, probs, action_index)
        self._episode: list[tuple] = []

    # --- features ---
    def _ctx_feats(self, context):
        v = _np.zeros(self._ctx_dim); v[0] = 1.0
        s = str(context.get("suite", ""))
        if s in self.suites:
            v[1 + self.suites.index(s)] = 1.0
        return v

    def _tel_feats(self, state):
        return _np.array([state["subtasks_done_frac"], state["steps_frac"],
                          state["attempt_detected"], *state["mutable_perm"]], dtype=float)

    @staticmethod
    def _softmax(feats, W):
        z = feats @ W
        z -= z.max()
        e = _np.exp(z)
        return e / e.sum()

    # --- episode lifecycle ---
    def begin_episode(self, context: dict[str, Any]) -> None:
        self._episode = []
        self._ctx = context

    def propose_initial(self, context: dict[str, Any]) -> list[int]:
        feats = self._ctx_feats(context)
        vector = []
        for d, W in enumerate(self.Winit):
            p = self._softmax(feats, W)
            a = int(self.rng.choice(len(p), p=p))
            self._episode.append(("init", d, feats, p, a))
            vector.append(a)
        return vector

    def reconfigure(self, state: dict) -> dict[str, str]:
        feats = self._tel_feats(state)
        decisions = {}
        for k in self.mutable:
            p = self._softmax(feats, self.Wrec[k])
            a = int(self.rng.choice(len(p), p=p))
            self._episode.append(("rec", k, feats, p, a))
            decisions[k] = _RECONFIG_ACTIONS[a]
        return decisions

    def learn(self, episode_return: float) -> None:
        """REINFORCE over every decision in the episode, using the terminal return."""
        adv = episode_return - self._baseline
        for head, key, feats, p, a in self._episode:
            onehot = _np.zeros_like(p); onehot[a] = 1.0
            grad_logp = _np.outer(feats, onehot - p)
            grad_ent = _np.outer(feats, -p * (_np.log(p + 1e-12) + 1.0))
            W = self.Winit[key] if head == "init" else self.Wrec[key]
            W += self.lr * (adv * grad_logp + self.entropy * grad_ent)
        self._baseline = self.baseline_decay * self._baseline + (1 - self.baseline_decay) * episode_return
        self._episode = []
