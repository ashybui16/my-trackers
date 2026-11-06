"""Tracker config selection (mirrors ``cpp/include/mot/tracking/config.hpp``).

``tracker_config_from_mapping`` reads the ``tracker`` section of
``configs/config.yaml`` with the same rules as ``readTrackerConfig`` in
``cpp/src/pipeline/config.cpp``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Union

from .trackers.botsort.config import BoTSORTConfig

# Add each new tracker's config class here (and to the factory registry).
TrackerConfig = Union[BoTSORTConfig]
CONFIG_TYPES: tuple[type, ...] = (BoTSORTConfig,)


def tracker_type(config: TrackerConfig) -> str:
    return _check_config(config).TYPE


def default_tracker_config(type_name: str) -> TrackerConfig:
    for cls in CONFIG_TYPES:
        if cls.TYPE == type_name:
            return cls()
    known = ", ".join(cls.TYPE for cls in CONFIG_TYPES)
    raise ValueError(f"unknown tracker type '{type_name}' (available: {known})")


def validate(config: TrackerConfig) -> None:
    _check_config(config).validate()


def tracker_config_from_mapping(values: Mapping[str, Any]) -> TrackerConfig:
    """Build a config from a ``tracker`` mapping.

    Accepts ``{type: <name>, <name>: {...}}`` or the legacy flat form with
    BoT-SORT keys at the top level. Unknown keys raise ``ValueError``.
    """

    if not isinstance(values, Mapping):
        raise ValueError("tracker must be a mapping")
    if "type" not in values:
        return _from_section(BoTSORTConfig, values, "tracker")
    type_name = values["type"]
    if not isinstance(type_name, str):
        raise ValueError("tracker.type must be a string")
    cls = type(default_tracker_config(type_name))
    _check_keys(values, {"type", type_name}, "tracker")
    section = values.get(type_name)
    if section is None:
        return cls()
    if not isinstance(section, Mapping):
        raise ValueError(f"tracker.{type_name} must be a mapping")
    return _from_section(cls, section, f"tracker.{type_name}")


def _check_config(config: Any) -> TrackerConfig:
    if not isinstance(config, CONFIG_TYPES):
        raise TypeError(f"unsupported tracker config: {type(config).__name__}")
    return config


def _check_keys(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {section}: {', '.join(unknown)}")


def _from_section(cls: type, values: Mapping[str, Any], section: str) -> TrackerConfig:
    _check_keys(values, {field.name for field in fields(cls)}, section)
    return cls(**values)
