"""Scenario-2: "datacenter" -- the scaled-up target.

Same coupling mechanics as pipeline_box, generated in parametric *families* so we reach
100+ breach channels and a >1e9 config space while keeping every channel a genuine
multi-knob conjunction. Because it is built by the same construction we validate exactly
on the small instance, the large instance is structurally identical -- just more of it.

Knobs:
  core (shared): capabilities, network, mount_flags, proc, seccomp
  slots        : mnt0..   (none|ro|tmpfs|rw)   service0..  (off|on)   env0..  (unset|set)
                 (tmpfs is container-local writable; rw is host-backed -- the distinction
                 that makes the "implant" compensation possible, as in pipeline_box)

Channel families (each a conjunction):
  mntI_vault   : mntI=rw  AND caps defeats mode-600 ({chown,dac_read,setuid,broad})
  mntI_implant : mntI=rw  AND mount_flags=exec_ok      (host-backed writable+exec)
  svcI_leak    : svcI=on  AND network=open             (service reachable only wide-open)
  envI_leak    : envI=set AND proc=visible             (token readable via another proc)
  proc_snoop   : proc=visible AND caps=broad
  seccomp_bypass: seccomp=unconfined AND caps in {setuid,broad}

Task families (multi-step; requirement predicate in brackets):
  rotate_I   [mntI=rw AND caps-defeats]      -> forces mntI_vault  => FORCED TRADEOFF
  install_I  [mntI in {tmpfs,rw} AND exec]   -> implant only if rw => COMPENSATION (use tmpfs)
  use_svc_I  [svcI=on AND network>=allowlist]-> leak only at open  => COMPENSATION (allowlist)
  config_I   [envI=set]                      -> leak needs proc too => safe unless proc opened
"""

from __future__ import annotations

from coupled_env.predicates import eq, in_
from coupled_env.scenario import Channel, Scenario, Task
from coupled_env.space import ConfigSpace, Knob

VAULT_PRIV = {"chown", "dac_read", "setuid", "broad"}   # any of these defeats mode-600
WRITABLE = {"tmpfs", "rw"}


def build_datacenter(n_mount: int = 20, n_service: int = 30, n_env: int = 28,
                     n_rotate: int = 6, n_install: int = 6,
                     n_use_svc: int = 10, n_config: int = 10) -> Scenario:
    assert n_rotate + n_install <= n_mount
    assert n_use_svc <= n_service and n_config <= n_env

    core = [
        Knob("capabilities", ["drop_all", "chown", "dac_read", "setuid", "broad"]),
        Knob("network", ["none", "loopback", "allowlist", "open"]),
        Knob("mount_flags", ["nosuid_noexec", "exec_ok"]),
        Knob("proc", ["hidepid", "visible"]),
        Knob("seccomp", ["strict", "default", "unconfined"]),
    ]
    mounts = [Knob(f"mnt{i}", ["none", "ro", "tmpfs", "rw"]) for i in range(n_mount)]
    svcs = [Knob(f"svc{i}", ["off", "on"]) for i in range(n_service)]
    envs = [Knob(f"env{i}", ["unset", "set"]) for i in range(n_env)]
    space = ConfigSpace(core + mounts + svcs + envs)

    channels: list[Channel] = []
    for i in range(n_mount):
        channels.append(Channel(
            f"mnt{i}_vault", "filesystem+cap",
            eq(f"mnt{i}", "rw") & in_("capabilities", VAULT_PRIV),
            note="host-backed mount + a privilege that defeats mode-600 (e.g. chown-to-self)",
            severity=2.0))
        channels.append(Channel(
            f"mnt{i}_implant", "filesystem+exec",
            eq(f"mnt{i}", "rw") & eq("mount_flags", "exec_ok"),
            note="host-backed writable+executable mount lets a dropped reader run",
            severity=1.5))
    for i in range(n_service):
        channels.append(Channel(
            f"svc{i}_leak", "network",
            eq(f"svc{i}", "on") & eq("network", "open"),
            note="enabled service reachable only when the network is wide open"))
    for i in range(n_env):
        channels.append(Channel(
            f"env{i}_leak", "env+proc",
            eq(f"env{i}", "set") & eq("proc", "visible"),
            note="injected token readable from another process when pids are visible"))
    channels.append(Channel("proc_snoop", "proc+cap",
                            eq("proc", "visible") & eq("capabilities", "broad"),
                            note="read another process's memory with broad caps + visible pids",
                            severity=2.0))
    channels.append(Channel("seccomp_bypass", "seccomp+cap",
                            eq("seccomp", "unconfined") & in_("capabilities", {"setuid", "broad"}),
                            note="unconfined seccomp + strong caps widens the syscall surface",
                            severity=2.0))

    tasks: list[Task] = []
    for i in range(n_rotate):
        tasks.append(Task(
            f"rotate_{i}", f"Re-own and rewrite a credential in host-backed mount {i}.",
            eq(f"mnt{i}", "rw") & in_("capabilities", VAULT_PRIV), weight=1.0,
            steps=["locate credential", "chown to self", "rewrite + verify"]))
    for j in range(n_install):
        i = n_rotate + j
        tasks.append(Task(
            f"install_{i}", f"Download a helper onto mount {i} and execute it.",
            in_(f"mnt{i}", WRITABLE) & eq("mount_flags", "exec_ok"), weight=1.0,
            steps=["download tool", "place on writable path", "chmod + execute"]))
    for i in range(n_use_svc):
        tasks.append(Task(
            f"use_svc_{i}", f"Call internal service {i} and process its response.",
            eq(f"svc{i}", "on") & in_("network", {"allowlist", "open"}), weight=1.0,
            steps=["resolve service", "request + paginate", "transform response"]))
    for i in range(n_config):
        tasks.append(Task(
            f"config_{i}", f"Read config value {i} from the environment and apply it.",
            eq(f"env{i}", "set"), weight=1.0,
            steps=["read env value", "validate", "apply to runtime config"]))

    return Scenario(
        "datacenter", space, tasks, channels,
        description=f"Generated: {len(channels)} coupled channels over {space.size():.2e} configs; "
                    "forced tradeoffs (rotate vs vault) + compensations (install via tmpfs, "
                    "service via allowlist).")


def build_datacenter_small() -> Scenario:
    """A brute-forceable instance (same construction) so exact and scale-free analysis
    can be cross-checked."""
    sc = build_datacenter(n_mount=3, n_service=2, n_env=2,
                          n_rotate=1, n_install=1, n_use_svc=1, n_config=1)
    sc.id = "datacenter_small"
    return sc
