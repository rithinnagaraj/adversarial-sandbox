# SAFETY — read before running real containers

This project studies whether a container **configuration** can keep a *frozen* LLM
agent from reading a flag that lives outside its sandbox, while still letting the
agent do benign work. The goal is **hardening sandboxes**, not producing an agent
that is better at escaping. The following preconditions are not optional for the
real-container phase.

## Hard preconditions (real / Docker phase)

1. **Run only inside a disposable Linux VM.** A throwaway VM with no real secrets,
   credentials, SSH keys, cloud tokens, or personal files. Never the daily-driver
   Windows host, and never a machine with access to anything you care about.
2. **The flag is a honeypot.** The only thing reachable beyond the container
   boundary is a dummy `flag.txt` in a disposable location. Never place the real
   host root, home directory, or any real secret on the far side of the boundary.
3. **The experiment-env marker gates real provisioning.** `DockerProvisioner`
   refuses to run unless `ADVCONTAINER_EXPERIMENT_ENV=1`. Set it *only* inside the
   VM, so the real path can never spin a container on an unintended machine.
4. **Default-deny the network boundary.** Network defaults to `none`; the "open"
   config level must point at a **local sink**, not the real internet, so even a
   capable agent cannot reach the outside world from inside the experiment.
5. **The student LLM stays frozen.** Never train or fine-tune the agent. Only the
   *environment designer* is optimized, and only toward *containment* — we measure
   the fixed agent, we never improve its ability to get out.

## What this harness deliberately does NOT do

- It does not implement, require, or describe any container-escape technique. The
  flag-read signal is a method-agnostic observation: *did the honeypot flag get
  read, yes or no*. How an escape might happen is out of scope for this code.
- It does not reward "containment" that is really the agent giving up. The outcome
  taxonomy separates `INVALID` (crash/timeout/starved/incoherent) from genuine
  containment, and the v1 `attempt_gated_reward` withholds credit unless the agent
  actually engaged the boundary. Watch the **confound audit** in the loop summary.

## v0 (this host) is safe by construction

The v0 loop uses `FakeProvisioner` + `FakeAgent`: no containers, no network, no
LLM calls. It is pure simulation for validating the harness logic and cannot
touch anything outside the Python process.
