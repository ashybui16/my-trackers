"""Standalone detector-agnostic multi-object tracking package.

The package deliberately exposes only normalized detections and track outputs.
Detector framework objects (including Ultralytics ``Results``) never cross this
boundary. Layout mirrors the C++ library in ``cpp/src/tracking``; see README.md.
"""

from .base import Tracker
from .config import (
    TrackerConfig,
    default_tracker_config,
    tracker_config_from_mapping,
    tracker_type,
    validate,
)
from .factory import available_trackers, create_tracker
from .trackers.botsort import BoTSORT, BoTSORTConfig
from .types import Detection, TrackOutput

__all__ = [
    "BoTSORT",
    "BoTSORTConfig",
    "Detection",
    "TrackOutput",
    "Tracker",
    "TrackerConfig",
    "available_trackers",
    "create_tracker",
    "default_tracker_config",
    "tracker_config_from_mapping",
    "tracker_type",
    "validate",
]
__version__ = "0.1.0"
