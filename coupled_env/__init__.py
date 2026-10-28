"""coupled_env: a richer, *interconnected* container-hardening environment.

Unlike the v1 space (config/space.py) where each knob gates an independent leak
channel, here breach channels and benign tasks are declared as **predicates over
multiple knobs**, so closing a leak and finishing a task compete for the same
settings. That coupling is the whole point: it creates genuine tradeoffs (some
leaks are avoidable only with a specific knob *combination*; others are impossible
to avoid without sacrificing a task).

This package is pure-Python and dependency-free: it models the environment and
scores any config with an *oracle* (a worst-case perfect agent that breaches every
open channel and finishes every satisfiable task). That lets us analyse a scenario's
tradeoff structure and validate it is non-trivial WITHOUT spending containers or API
calls. Real-container execution (a frozen LLM agent in Docker) plugs in later by
replacing the oracle with a real rollout and reusing the exact same predicates for
scoring -- see README.md ("Plugging in real execution").
"""

from coupled_env.space import ConfigSpace, Knob
from coupled_env.scenario import Channel, Scenario, Task
from coupled_env import predicates

__all__ = ["ConfigSpace", "Knob", "Channel", "Scenario", "Task", "predicates"]
