"""Tracker factory (mirrors ``cpp/src/tracking/factory.cpp``)."""

from __future__ import annotations

from .base import Tracker
from .config import CONFIG_TYPES, TrackerConfig, validate
from .trackers.botsort import BoTSORT, BoTSORTConfig

# Config class -> tracker class. Add one entry per tracker.
_REGISTRY: dict[type, type[Tracker]] = {
    BoTSORTConfig: BoTSORT,
}


def available_trackers() -> list[str]:
    return [cls.TYPE for cls in CONFIG_TYPES]


def create_tracker(config: TrackerConfig) -> Tracker:
    validate(config)
    return _REGISTRY[type(config)](config)
