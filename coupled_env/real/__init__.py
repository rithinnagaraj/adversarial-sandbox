"""Real-container execution for coupled_env scenarios (pilot-sized ones).

The oracle in coupled_env/oracle.py is a fast *simulation* used to train the designer.
This package runs a **real frozen LLM agent in a real Docker container** instead, scoring
with the SAME graded-containment semantics. It reuses the v1 machinery that was debugged
across the A-D pilots (config/docker_env.py's `_docker`, episode/agent.py's ApiAgent).

Scope: real execution targets *pilot-sized* scenarios (pipeline_box: 5 knobs / 3 tasks /
4 channels). The `datacenter` scenario (100 channels / 1e31 configs) is simulation-only --
you would never place 100 tokens and run a 32-task episode in one container.

Pieces:
  realize.py  -- config dict -> Docker args; per-channel token placement; per-task checkers
  scoring.py  -- graded containment + utility from a finished episode
  episode.py  -- the episode loop; FakeCoupledEnv (oracle-backed, host-testable) + CoupledDockerEnv
  run.py      -- CLI driver (fake or docker; OpenRouter or Sarvam)
"""
