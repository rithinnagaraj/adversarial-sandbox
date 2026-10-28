"""Scenario registry: name -> builder. Add new scenarios here."""

from __future__ import annotations

from typing import Callable

from coupled_env.scenario import Scenario
from coupled_env.scenarios.datacenter import build_datacenter, build_datacenter_small
from coupled_env.scenarios.datacenter_lite import build_datacenter_lite
from coupled_env.scenarios.pipeline_box import build_pipeline_box

REGISTRY: dict[str, Callable[[], Scenario]] = {
    "pipeline_box": build_pipeline_box,
    "datacenter": build_datacenter,
    "datacenter_small": build_datacenter_small,
    "datacenter_lite": build_datacenter_lite,
}


def get(name: str) -> Scenario:
    if name not in REGISTRY:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def names() -> list[str]:
    return sorted(REGISTRY)
