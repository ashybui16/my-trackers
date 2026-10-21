"""Public value types exchanged with the tracker."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable


def _as_box(value: Iterable[Real], *, name: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain four numeric values") from exc
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain only finite values")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return values


def _as_score(value: Real) -> float:
    if not isinstance(value, Real):
        raise TypeError("score must be a real number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and in [0, 1]")
    return score


def _as_class_id(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("class_id must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalized axis-aligned detector output.

    ``bbox_xyxy`` is always ``(x1, y1, x2, y2)`` in image coordinates.
    """

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_xyxy", _as_box(self.bbox_xyxy, name="bbox_xyxy"))
        object.__setattr__(self, "score", _as_score(self.score))
        object.__setattr__(self, "class_id", _as_class_id(self.class_id))


@dataclass(frozen=True, slots=True)
class TrackOutput:
    """One active track returned for the current frame."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, Integral):
            raise TypeError("track_id must be an integer")
        if int(self.track_id) <= 0:
            raise ValueError("track_id must be positive")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "bbox_xyxy", _as_box(self.bbox_xyxy, name="bbox_xyxy"))
        object.__setattr__(self, "score", _as_score(self.score))
        object.__setattr__(self, "class_id", _as_class_id(self.class_id))
