"""Unit tests for the config design space (pure logic, no containers)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.space import DEFAULT_SPACE, ConfigSpace


def test_size_matches_level_product():
    space = DEFAULT_SPACE
    expected = 1
    for c in space.level_counts:
        expected *= c
    assert space.size() == expected
    assert len(space.enumerate()) == expected


def test_vector_dict_roundtrip():
    space = DEFAULT_SPACE
    for vector in space.enumerate():
        mapping = space.to_dict(vector)
        assert space.from_dict(mapping) == vector


def test_is_valid_rejects_bad_vectors():
    space = DEFAULT_SPACE
    assert space.is_valid([0] * space.n_dims)
    assert not space.is_valid([0] * (space.n_dims - 1))      # wrong length
    bad = [len(d.levels) for d in space.dimensions]          # all out of range
    assert not space.is_valid(bad)


def test_mutable_dims_are_the_runtime_knobs():
    assert set(DEFAULT_SPACE.mutable_dims()) == {"filesystem", "network", "env_injection", "resources"}


def test_restrictiveness_bounds():
    space = DEFAULT_SPACE
    most_restrictive = [0] * space.n_dims
    least_restrictive = [len(d.levels) - 1 for d in space.dimensions]
    assert space.restrictiveness(most_restrictive) == 0.0
    assert space.restrictiveness(least_restrictive) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("test_space: all passed")
