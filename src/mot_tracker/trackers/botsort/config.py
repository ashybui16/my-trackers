"""BoT-SORT configuration (mirrors ``mot::BoTSORTConfig``)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Mapping


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class BoTSORTConfig:
    """Supported baseline BoT-SORT settings.

    The configuration intentionally has no ReID, detector, Torch or framework
    fields.  Those concerns belong to the caller.
    """

    TYPE: ClassVar[str] = "botsort"

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    gmc_downscale: int = 2

    def __post_init__(self) -> None:
        for key, value in self._normalized().items():
            object.__setattr__(self, key, value)

    def validate(self) -> None:
        """Raise ``TypeError``/``ValueError`` if any field is invalid."""

        self._normalized()

    def _normalized(self) -> dict[str, Any]:
        high = _probability(self.track_high_thresh, "track_high_thresh")
        low = _probability(self.track_low_thresh, "track_low_thresh")
        new = _probability(self.new_track_thresh, "new_track_thresh")
        match = _probability(self.match_thresh, "match_thresh")
        if not low < high:
            raise ValueError("track_low_thresh must be smaller than track_high_thresh")
        if new < high:
            raise ValueError("new_track_thresh must be at least track_high_thresh")
        if isinstance(self.track_buffer, bool) or not isinstance(self.track_buffer, Integral):
            raise TypeError("track_buffer must be an integer")
        if int(self.track_buffer) < 0:
            raise ValueError("track_buffer must be non-negative")
        if not isinstance(self.fuse_score, bool):
            raise TypeError("fuse_score must be boolean")
        if self.gmc_method not in {"sparseOptFlow", "none"}:
            raise ValueError("gmc_method must be 'sparseOptFlow' or 'none'")
        if isinstance(self.gmc_downscale, bool) or not isinstance(self.gmc_downscale, Integral):
            raise TypeError("gmc_downscale must be an integer")
        if int(self.gmc_downscale) < 1:
            raise ValueError("gmc_downscale must be at least 1")
        return {
            "track_high_thresh": high,
            "track_low_thresh": low,
            "new_track_thresh": new,
            "track_buffer": int(self.track_buffer),
            "match_thresh": match,
            "gmc_downscale": int(self.gmc_downscale),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BoTSORTConfig":
        """Build a config from a YAML-like mapping.

        Unknown keys are intentionally ignored so legacy Ultralytics ReID
        settings cannot leak into the standalone tracker.
        """

        supported = {
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "gmc_method",
            "gmc_downscale",
        }
        return cls(**{key: values[key] for key in supported if key in values})
