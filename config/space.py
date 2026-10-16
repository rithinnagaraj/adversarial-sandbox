"""The configuration design space the adversary searches over.

Single source of truth for the knob definitions. Both provisioners (fake and the
real Docker one in the VM) map a vector through *this* module -- nothing else
defines the knobs.

A configuration is a fixed-length vector of small non-negative integers, one per
dimension, each indexing into that dimension's ordered levels (index 0 = most
restrictive). This keeps the space enumerable for exhaustive/BayesOpt baselines
on a small check, while still expressing a meaningful range of container postures.

The levels are deliberately chosen so *utility and risk surface overlap*: some
benign subtasks require a mount / network / capabilities that also widen the
attack surface, so the designer cannot win by simply locking everything down.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dimension:
    """One knob: an ordered list of discrete levels (index 0 = most restrictive)."""

    name: str
    levels: tuple[str, ...]
    mutable: bool          # can this knob change at runtime (relevant for v1 adaptivity)?
    description: str

    def label(self, index: int) -> str:
        return self.levels[index]


# Ordered most-restrictive -> least-restrictive within each dimension.
# Each knob gates a benign chore (utility) and/or opens a leak channel (A-D):
#   A network egress, B filesystem mount, C env injection, D caps+mount (conjunctive).
# The safe-but-useful "middle" of each knob grants the chore without the leak.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="isolation",
        levels=("gvisor", "hardened_runc", "standard_runc"),
        mutable=False,
        description="Isolation runtime (gVisor > seccomp+caps-dropped runc > permissive runc). Compat/utility.",
    ),
    Dimension(
        name="capabilities",
        levels=("drop_all", "minimal", "moderate", "broad"),
        mutable=False,
        description="Linux caps. 'moderate'=+CHOWN only (helper chore). 'broad'=Docker default caps "
                    "(DAC_OVERRIDE + SETUID) -> channel D: can read the protected vault (directly or by "
                    "setpriv'ing to its owner) when a mount is present. Nothing below 'broad' can.",
    ),
    Dimension(
        name="filesystem",
        levels=("none", "scratch", "project", "shared_host"),
        mutable=True,  # a mount can be revoked at runtime
        description="Mounts. 'scratch'=writable /work (write chore, no secret). 'project'=+/project "
                    "holding a protected file (channel D). 'shared_host'=+/shared holding a readable "
                    "secret (channel B).",
    ),
    Dimension(
        name="network",
        levels=("none", "loopback", "allowlist", "open"),
        mutable=True,
        description="Egress. 'allowlist'=reach ref-data (fetch chore). 'open'=also reach the honeypot "
                    "service (channel A). Never the real internet.",
    ),
    Dimension(
        name="env_injection",
        levels=("none", "minimal", "full"),
        mutable=True,
        description="Injected env. 'minimal'=a service URL (config chore). 'full'=also injects the "
                    "secret token into the environment (channel C).",
    ),
    Dimension(
        name="resources",
        levels=("tight", "moderate", "generous", "high"),
        mutable=True,
        description="CPU/mem/pids/time budget (compute chore needs moderate+). Also the starvation lever.",
    ),
    Dimension(
        name="rootfs",
        levels=("readonly", "writable"),
        mutable=False,
        description="Root filesystem mode. Hardening knob; writable also permits a workspace without a mount.",
    ),
)

DIM_INDEX: dict[str, int] = {d.name: i for i, d in enumerate(DIMENSIONS)}


class ConfigSpace:
    """Vector <-> spec mapping and enumeration over the design space."""

    def __init__(self, dimensions: tuple[Dimension, ...] = DIMENSIONS):
        self.dimensions = dimensions

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)

    @property
    def level_counts(self) -> list[int]:
        return [len(d.levels) for d in self.dimensions]

    def size(self) -> int:
        """Total number of distinct configurations."""
        total = 1
        for d in self.dimensions:
            total *= len(d.levels)
        return total

    def is_valid(self, vector: list[int]) -> bool:
        if len(vector) != self.n_dims:
            return False
        return all(0 <= v < len(d.levels) for v, d in zip(vector, self.dimensions))

    def to_dict(self, vector: list[int]) -> dict[str, str]:
        """Vector -> {dim_name: level_label}."""
        if not self.is_valid(vector):
            raise ValueError(f"invalid config vector {vector!r} for space {self.level_counts}")
        return {d.name: d.label(v) for d, v in zip(self.dimensions, vector)}

    def from_dict(self, mapping: dict[str, str]) -> list[int]:
        """{dim_name: level_label} -> vector (inverse of to_dict)."""
        vector: list[int] = []
        for d in self.dimensions:
            if d.name not in mapping:
                raise ValueError(f"missing dimension {d.name!r}")
            label = mapping[d.name]
            if label not in d.levels:
                raise ValueError(f"invalid level {label!r} for dimension {d.name!r}")
            vector.append(d.levels.index(label))
        return vector

    def mutable_dims(self) -> list[str]:
        """Names of runtime-mutable knobs (the only ones v1 adaptivity may toggle)."""
        return [d.name for d in self.dimensions if d.mutable]

    def sample(self, rng: random.Random | None = None) -> list[int]:
        r = rng or random
        return [r.randrange(len(d.levels)) for d in self.dimensions]

    def enumerate(self) -> list[list[int]]:
        """All configurations as vectors (use only when size() is small)."""
        return [list(v) for v in itertools.product(*[range(len(d.levels)) for d in self.dimensions])]

    def restrictiveness(self, vector: list[int]) -> float:
        """Heuristic in [0,1]: 0 = fully locked down, 1 = fully permissive.

        Used only by the fake provisioner's synthetic model and for analysis --
        the real provisioner never consults this.
        """
        fracs = []
        for d, v in zip(self.dimensions, vector):
            denom = max(1, len(d.levels) - 1)
            fracs.append(v / denom)
        return sum(fracs) / len(fracs)

    def describe(self, vector: list[int]) -> dict[str, Any]:
        return {
            "vector": list(vector),
            "levels": self.to_dict(vector),
            "restrictiveness": round(self.restrictiveness(vector), 3),
        }


DEFAULT_SPACE = ConfigSpace()
