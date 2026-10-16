"""Tests for the pure config-vector -> ContainerSpec mapping (no Docker needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.container_spec import PROJECT_DIR, SHARED_DIR, WORKSPACE_DIR, spec_for
from config.space import DEFAULT_SPACE


def _vec(**knobs):
    base = {"isolation": "gvisor", "capabilities": "drop_all", "filesystem": "none",
            "network": "none", "env_injection": "none", "resources": "moderate", "rootfs": "readonly"}
    base.update(knobs)
    return DEFAULT_SPACE.from_dict(base)


def test_strong_config_is_locked_down():
    s = spec_for(_vec())
    assert s.runtime == "runsc" and s.seccomp == "default" and s.no_new_privileges
    assert s.cap_drop == ["ALL"] and s.read_only_rootfs and s.mounts == [] and s.network == "none"


def test_weak_config_is_permissive_but_not_an_escape_primitive():
    s = spec_for(_vec(isolation="standard_runc", capabilities="broad", filesystem="shared_host",
                      network="open", env_injection="full", resources="high", rootfs="writable"))
    assert s.runtime == "runc" and s.seccomp == "unconfined"
    assert s.cap_drop == [] and not s.read_only_rootfs
    assert s.mounts == [WORKSPACE_DIR, PROJECT_DIR, SHARED_DIR] and s.network == "open"
    s.assert_safe()


def test_mounts_grow_with_filesystem_scope():
    assert spec_for(_vec(filesystem="none")).mounts == []
    assert spec_for(_vec(filesystem="scratch")).mounts == [WORKSPACE_DIR]
    assert spec_for(_vec(filesystem="project")).mounts == [WORKSPACE_DIR, PROJECT_DIR]
    assert spec_for(_vec(filesystem="shared_host")).mounts == [WORKSPACE_DIR, PROJECT_DIR, SHARED_DIR]


def test_moderate_is_chown_only_broad_keeps_default_set():
    # moderate must NOT include SETUID/DAC -- otherwise channel D leaks below 'broad'
    assert spec_for(_vec(capabilities="moderate")).cap_add == ["CHOWN"]
    assert "SETUID" not in spec_for(_vec(capabilities="moderate")).cap_add
    assert spec_for(_vec(capabilities="broad")).cap_drop == []   # default caps incl DAC_OVERRIDE + SETUID


def test_every_config_passes_the_safety_invariant():
    for vector in DEFAULT_SPACE.enumerate():
        spec_for(vector).assert_safe()


def test_docker_args_render():
    args = spec_for(_vec(network="allowlist")).to_docker_args()
    assert "--runtime=runsc" in args and "--network=advc-allowlist" in args
    assert any(a.startswith("--memory=") for a in args)


def test_resource_tiers_differ():
    assert spec_for(_vec(resources="tight")).cpus < spec_for(_vec(resources="high")).cpus
    assert spec_for(_vec(resources="tight")).timeout_s < spec_for(_vec(resources="high")).timeout_s


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_container_spec: all passed")
