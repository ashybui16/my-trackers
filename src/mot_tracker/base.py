"""Tracker interface (mirrors ``mot::ITracker`` in ``cpp/include/mot/tracking/tracker.hpp``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .types import Detection, TrackOutput


class Tracker(ABC):
    """Common interface implemented by every tracker in the package."""

    @abstractmethod
    def update(self, detections: Iterable[Detection], frame: np.ndarray | None = None) -> list[TrackOutput]:
        """Consume one frame of detections and return the active tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all tracks and restart the ID counter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The tracker type, equal to its config's ``TYPE``."""
