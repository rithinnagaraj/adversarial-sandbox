# Roadmap — making the setting more realistic, challenging, and statistically rigorous

Next-steps plan after the v3 GLM-5.3 sim-to-real result (designer converged to the oracle
optimum: composite 1.32, 0/8 leaks + 4/5 tasks). Goal: push toward a result that is both
rigorous and credibly realistic, **while staying feasible on one GCP VM**.

## The binding constraint
Real episodes are expensive (~125 s each; 300 episodes ≈ 10.4 h, single-threaded). Every realism
and rigor idea is bought against that budget, so the plan leads with the levers that stretch the
budget, then spends the savings.

## Key tractability insight (corrected — do NOT key on "core knobs")
A first draft proposed caching real outcomes keyed on the **8 core knobs**. That is circular: the
core/distractor partition *is* oracle ground truth, so keying on it bakes privileged knowledge into
the experimental plumbing. In a realistic setting we would not know which knobs cause the leak.

**Correct cache:** key on the **realized container spec** — the deterministic function
`R(config) → {docker flags, planted-token set, env, sidecars}` the harness actually instantiates.
Two configs collapse to one key only if the environment literally builds them identically; that
grouping is *discovered from R's output*, never asserted by us. It is a sound, leak-free
memoization of the environment we already run. Caveats that keep it honest:
- Under a stochastic agent, each entry is a **bank of i.i.d. samples**, not a fixed value.
  Legitimate for held-out mean ± CI and variance estimation.
- **Not** a free training accelerant: reusing a cached return when REINFORCE re-proposes a config
  injects correlated/stale reward and biases the gradient + variance. So the cache buys
  *evaluation* rigor; any training-time reuse needs explicit justification. (This walks back the
  earlier "makes multi-seed cheap" claim — multi-seed cost is carried by the free oracle, not the cache.)

**Per-episode isolation + parallelism.** Today `advc-project`/`advc-shared` volumes and
container/network names are shared/fixed → forces serial runs. Make them per-episode unique → run N
containers at once (each is already `--cpus=1 --memory=1g`), cutting wall-clock by the VM's vCPU
count. Pure infra, no faithfulness risk.

## Threats to validity (must stand against strict peer review)
Governing principle: **oracle ground truth (the knob partition, the channel predicates) may inform
analysis and efficiency, but must never be load-bearing in the environment or the learner.**

1. **The 1.53×10⁹ search space is cosmetic.** The 16 distractors are *provably* zero-effect (the
   realization never applies them), so the effective space is ~4,608 and "the designer ignores
   distractors" is trivial (zero gradient, not realistic nuisance). Fix (Phase 2): either (a) drop
   the cosmetic distractors and report the honest effective dimensionality — reviewers prefer a
   small honest number to an inflated one — or **(b, preferred)** make some nuisance knobs genuinely
   couple into channels with *a-priori-unknown* effect (e.g. `userns` really changes a mapped-root
   read, `pids_limit` really gates a fork-based sidecar), so "discover which knobs matter" becomes a
   real learning problem and the realized-spec cache becomes legitimately dynamic.
2. **Reachability == predicate.** Tokens are planted iff the oracle predicate holds, so the real
   container can never surprise the oracle on *reachability* — only on agent *behavior*. The measured
   "sim-to-real gap" is therefore **behavior-only by construction**; predicate-defined reachability is
   an imposed assumption, not a validated one. State this as scope, do not sell it as validating the oracle.
3. **Stochastic caching correlation** (see cache caveats above): eval-only sample bank, not training reuse.
4. **Hand-chosen reward/severities/task set.** The "optimum = 1.32" depends on `w_u=0.4` and the
   specific severities; robustness to these is untested. Add a small sensitivity sweep (free, oracle-side).

---

## Phase 0 — Instrumentation & tractability (do first; unblocks everything)
- Real-eval cache (`coupled_env/real/`, keyed on the **realized container spec** `R(config)` — not
  hand-labeled core knobs; sample bank per spec; persists to `logs/`, survives `--resume`). See
  "Key tractability insight (corrected)" for why.
- Per-episode unique volume/network/container names → concurrent episode scheduler in `episode.py`.
- Seed plumbing through `train_designer.py`; structured per-repeat logging (seed, config, repeat idx,
  per-channel breach, per-task done).
- Aggregation script: mean ± CI, regret vs oracle optimum, convergence curves.

## Phase 1 — Statistical rigor (mostly free, cheap real tail)
- **Oracle, many seeds (free):** ≥10-seed pretrain → distribution of final regret and
  "fraction of proposals that are optimal." Turns "it converged" into "it reliably converges,
  regret X ± Y."
- **Real held-out eval:** run the learned recommendation *and each baseline*
  (lockdown / wide_open / random / learned) k=5× via the cache; report mean ± CI. Retires the
  "1 repeat, indicative not exact" caveat.
- **Variance decomposition:** separate policy stochasticity from agent stochasticity (the cache
  gives the within-config spread for free).
- A few real fine-tune seeds *if* Phase 0 parallelism makes it affordable; otherwise keep
  real-seeding to the held-out eval and let the oracle carry the seed count.

## Phase 2 — Realism of channels & tasks (faithfulness work; each change is a finding)
- **Promote proxy channels to real reachability gates.** `proc_snoop`/`env_proc` are currently
  placement-gated tokens → make them genuine: run the secret-holding sidecar in a **separate PID
  namespace unless `proc=visible`**, so `/proc` reachability is physically enforced (also replaces
  the `hidepid` flag the daemon rejected with something real).
- **Add 1–2 genuinely independent real channels** (no-exploit rule intact): a mock
  **metadata-service SSRF** reachable only at `network=open`, and/or a benign **egress/DNS exfil**
  channel — so the tradeoff is not all routed through filesystem/caps.
- **Harder multi-step tasks + a real service backend** (replace the static-JSON stub with a small
  app that logs requests and has a behaviorally-distinct honeypot endpoint). Add a task whose
  *legitimate* path and *breach* path genuinely diverge (e.g. read a secret via the service token,
  not the vault), sharpening the utility-vs-containment tension.
- **Honest boundary:** `seccomp_bypass` can't be a *real* exploit without crossing the no-exploit
  line. Reframe as a benign syscall-gated info leak, or keep it explicitly labeled as the one
  abstraction — documented, not faked. Re-run `validate` after each change.

## Phase 3 — Model lineup & generalization (gated on account access)
- Add **DeepSeek / Qwen** as additional acting agents (retry a frontier model; Opus's refusal
  stays a legitimate datapoint).
- **Per-model fine-tune:** pretrain once (free), copy the checkpoint, fine-tune per agent —
  "which agent must the designer work hardest to contain?"
- **Transfer matrix:** designer trained against agent A, evaluated containing agent B
  (the headline generalization claim).

## Phase 4 — The case
- **Quantify the sim-to-real gap** (oracle-predicted vs real composite per config) as a
  first-class result — the methodological contribution that scales beyond this scenario.
- **Learned vs expert-hand-hardened baseline:** does the RL designer match/beat a security
  engineer's config while keeping *more* utility?

## What this buys (four defensible claims)
1. Reliable near-optimal convergence with CIs (Phase 1).
2. Containment under *real* agent behavior with genuinely-realized channels (Phase 2).
3. Generalization across attacker capability (Phase 3).
4. A quantified sim-to-real method (Phase 4).

## Implementation status
- **Phase 0 — DONE (host-validated).** `coupled_env/real/cache.py` (sample bank keyed on
  `realized_spec(scenario, config, realize_mod)`); per-episode volume isolation via a `namespace`
  arg threaded through `CoupledDockerEnv` + both realize modules (`ns=""` is byte-identical to the
  serial pipeline, unit-asserted); `train_designer.py` now writes the bank + an ordered episode log
  and supports `--parallel N` (per-episode isolated volumes; honeypot-opening configs auto-serialized
  via a lock); `coupled_env/aggregate.py` summarizer; `coupled_env/stats.py` (stdlib t-based CI).
  6 new unit tests (34 pass total), incl. the anti-circularity key test (distractors collapse, a
  core knob splits) and a fake-env held-out-eval test.
- **Phase 1 — DONE (host-validated on the fake/oracle path).** `coupled_env/sweep_oracle.py`
  (multi-seed oracle sweep → regret distribution, fraction optimal); `coupled_env/real/holdout_eval.py`
  (k-sample held-out eval of learned + baselines, pooled via the bank, mean ± CI, per-channel breach
  rate, oracle gap, variance decomposition).
- **Phase 2 Step 1 — IMPLEMENTED (host-tested; VM validation in progress).** The `/proc` channels
  are now PHYSICALLY gated, replacing placement gating. `_CAPS["broad"]` gains `SYS_PTRACE`;
  `realize_datacenter_lite.docker_args` emits `--pid=container:<holder>` iff `proc=visible` (so the
  agent shares the sidecar's PID namespace only then); new `proc_holder_spec` runs a per-episode
  PID-namespace sidecar; `episode.py` starts/tears it down; env_proc/proc_snoop removed from the
  in-container `sidecar_commands`. The realized-spec key now splits on proc visibility (the bank is
  legitimately dynamic for that knob). `coupled_env/real/validate_proc.py` proves the truth table on
  the VM with a deterministic probe (no LLM); `diag_proc.py` is the throwaway diagnostic used below.
  - **VM finding (runc, Ubuntu 22.04, yama ptrace_scope=1):** a cross-process `/proc/<pid>/environ`
    read requires `CAP_SYS_PTRACE` **even same-uid** — so `env_proc` as originally specified (an
    *environment* leak with no privilege) is NOT physically realizable. Corrected mechanism maps the
    two channels onto two different real kernel rules: **env_proc → secret in a process's argv, read
    via world-readable `/proc/<pid>/cmdline` (no cap)**; **proc_snoop → secret in a process's
    `/proc/<pid>/environ` (needs `SYS_PTRACE`=broad)**. Both held by root; the capability gate is the
    cmdline-vs-environ distinction, which is the actual kernel behaviour.
  - **Run on the VM:** `ADVCONTAINER_EXPERIMENT_ENV=1 ~/advc-venv/bin/python -m coupled_env.real.validate_proc`
    → expect ALL PASS. NOTE: this changes the realized spec of every `proc=visible` / `caps=broad`
    config, so pre-Phase-2 bank/eval numbers for those configs are not comparable and should be
    re-collected.
- **Not yet validated on real Docker:** `--parallel N` (needs ~N cpus + N GB RAM on the VM; the
  dynamic per-episode volumes and honeypot serialization are coded but unrun against Docker — do one
  short validation run before trusting it for the full budget). The default `--parallel 1` reproduces
  the validated v3 pipeline exactly.

## Recommended order
Phase 0 now (pure infra, no faithfulness risk) → Phase 1 (cheap, high-credibility) → Phases 2 and 3
in parallel as access/appetite allow → Phase 4 writeup.

## Open inputs that affect sequencing
- **VM size:** how many vCPUs? Sets how much Phase 0 parallelism helps (and whether real
  multi-seed training is affordable).
- **Model access:** is anything beyond GLM-5.3 reachable on JarvisLabs yet? If yes, Phase 3 can
  start right after Phase 0; if no, slot Phase 2 ahead of it.
