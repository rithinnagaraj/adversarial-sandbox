"""Tests for the two escape-model worlds (does a knife-edge exist?)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.provisioner import FakeEnv
from config.space import DEFAULT_SPACE


def _escape(levels: dict, world: str) -> float:
    env = FakeEnv(levels, DEFAULT_SPACE, random.Random(0), world=world)
    return env._escape_prob()


# a config that grants everything the benign subtasks need (full utility)
OPEN = {"isolation": "gvisor", "capabilities": "broad", "filesystem": "shared_host",
        "network": "open", "env_injection": "full", "resources": "high", "rootfs": "writable"}
# the same permissive surface but with the weakest isolation
OPEN_WEAK_ISO = {**OPEN, "isolation": "standard_runc"}


def test_isolation_gated_has_a_knife_edge():
    # strong isolation contains even with everything else open -> low escape
    assert _escape(OPEN, "isolation_gated") < 0.1
    # weakening ONLY isolation makes that same permissive config leak badly
    assert _escape(OPEN_WEAK_ISO, "isolation_gated") > 0.6
    # so strong isolation genuinely buys containment here (a knife-edge exists)
    assert _escape(OPEN, "isolation_gated") < _escape(OPEN_WEAK_ISO, "isolation_gated")


def test_entangled_has_no_knife_edge():
    # in the entangled world, strong isolation CANNOT offset open surface:
    # a full-utility config still has high escape -> lockdown is the only way to contain
    assert _escape(OPEN, "entangled") > 0.5


def test_worlds_agree_under_full_lockdown():
    lockdown = {"isolation": "gvisor", "capabilities": "drop_all", "filesystem": "none",
                "network": "none", "env_injection": "none", "resources": "moderate", "rootfs": "readonly"}
    # with nothing open and strongest isolation, both worlds are near the base rate
    assert _escape(lockdown, "isolation_gated") < 0.1
    assert _escape(lockdown, "entangled") < 0.1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_worlds: all passed")
