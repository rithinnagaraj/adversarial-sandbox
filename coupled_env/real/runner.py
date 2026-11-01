"""Shared single-episode runner for real (or fake) runs.

Centralizes token minting, agent creation, and "one config -> scored outcome + bank record",
so train_designer's real fine-tune, the held-out evaluator, and run.py all mint episodes the
same way. Stdlib + the real/* modules only (no numpy), so it imports on any host.
"""

from __future__ import annotations

import secrets
import time

from coupled_env.real.episode import (CoupledDockerEnv, CoupledRunConfig, FakeCoupledEnv,
                                       run_episode)
from coupled_env.real.scoring import score


def fresh_tokens(realize_mod) -> dict[str, str]:
    return {ch: f"FLAG-{ch}-{secrets.token_hex(8)}" for ch in realize_mod.CHANNEL_TOKENS}


def make_agent(provider: str, model: str, fake: bool, max_context_chars: int = 16000):
    """Return (agent, completer). A FRESH agent per call -- parallel workers must not share
    one (ApiAgent carries per-episode conversation state)."""
    from episode.agent import ApiAgent, FakeAgent, make_completer
    if fake:
        return FakeAgent(), None
    completer = make_completer(provider, model)
    return ApiAgent(complete=completer, max_context_chars=max_context_chars), completer


def outcome_sample(outcome, scored: dict, wall_s: float) -> dict:
    """The per-episode record stored in the RealEvalCache bank."""
    return {**scored, "invalid_kind": outcome.invalid_kind, "error": outcome.error,
            "episode_id": outcome.episode_id, "wall_s": round(wall_s, 1)}


def run_one(scenario, config, realize_mod, agent, runcfg, instructions, w_u,
            fake: bool, namespace: str = ""):
    """Run ONE episode for `config`; return (scored_dict, outcome, wall_seconds).

    `namespace` isolates this episode's host-backed volumes (parallel runs); "" uses the
    shared volumes exactly as the validated serial pipeline does.
    """
    env = (FakeCoupledEnv(scenario, config) if fake
           else CoupledDockerEnv(scenario, config, realize_mod, fresh_tokens(realize_mod),
                                 namespace=namespace))
    t0 = time.time()
    out = run_episode(scenario, env, agent, runcfg, instructions)
    scored = score(scenario, out.breached, out.tasks_done, w_u=w_u)
    return scored, out, time.time() - t0
