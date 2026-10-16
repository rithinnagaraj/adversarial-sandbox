"""Structural tests for the real (VM) pieces -- no Docker/network needed."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.provisioner import DockerProvisioner
from episode.real_subtasks import REAL_MIXED, real_suite
from episode.subtasks import SUITE_MIXED


def test_real_subtasks_mirror_v0_ids_and_requirements():
    v0 = {s.id: s for s in SUITE_MIXED.subtasks}
    real = {r.id: r for r in REAL_MIXED}
    assert set(v0) == set(real)  # same subtasks
    for sid, r in real.items():
        assert r.requirement_note == v0[sid].requirement_note  # same knob requirement
        assert r.instruction and r.check_cmd  # both specified


def test_add_host_entries_match_network_policy():
    # channel A: the honeypot is resolvable ONLY on the open network; ref-data follows egress
    from config.docker_env import _add_host_args
    assert _add_host_args("none") == []
    assert _add_host_args("loopback") == []
    allow = _add_host_args("allowlist")
    assert any("refdata.advc.internal" in h for h in allow)
    assert not any("honeypot" in h for h in allow)          # restricted config can't even name it
    opn = _add_host_args("open")
    assert any("refdata.advc.internal" in h for h in opn)
    assert any("honeypot.advc.internal" in h for h in opn)  # open config can reach the honeypot


def test_real_suite_lookup():
    assert real_suite("mixed_v0") == REAL_MIXED
    try:
        real_suite("nope")
        assert False
    except KeyError:
        pass


def test_docker_provisioner_build_spec_is_pure_and_safe():
    # build_spec must work without Docker / without the env guard (7-knob vectors)
    prov = DockerProvisioner()
    for vector in ([0, 0, 0, 0, 0, 1, 0], [2, 3, 3, 3, 2, 3, 1]):
        prov.build_spec(vector).assert_safe()


def test_docker_provisioner_provision_is_guarded(monkeypatch=None):
    # without the experiment-env marker, provision must refuse
    os.environ.pop("ADVCONTAINER_EXPERIMENT_ENV", None)
    try:
        DockerProvisioner().provision([0, 0, 0, 0, 0, 1, 0])
        assert False, "expected refusal without the env marker"
    except RuntimeError as e:
        assert "refuses to run" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_real_pieces: all passed")
