"""Standalone detector-agnostic BoT-SORT tracking package.

The package deliberately exposes only normalized detections and track outputs.
Detector framework objects (including Ultralytics ``Results``) never cross this
boundary.
"""

from .config import BoTSORTConfig
from .tracker import BoTSORT
from .types import Detection, TrackOutput

__all__ = ["BoTSORT", "BoTSORTConfig", "Detection", "TrackOutput"]
__version__ = "0.1.0"
