"""Internal track state and lifecycle primitives."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from .geometry import warp_xywh_state, xywh_to_xyxy, xyxy_to_xywh
from .kalman_filter import KalmanFilterXYWH
from .types import Detection


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3

    # Compatibility spellings used by common BoT-SORT implementations.
    New = NEW
    Tracked = TRACKED
    Lost = LOST
    Removed = REMOVED


class Track:
    """A single XYWH Kalman track used internally by :class:`BoTSORT`."""

    def __init__(self, detection: Detection, detection_index: int = -1) -> None:
        self._tlwh = np.asarray(
            (
                detection.bbox_xyxy[0],
                detection.bbox_xyxy[1],
                detection.bbox_xyxy[2] - detection.bbox_xyxy[0],
                detection.bbox_xyxy[3] - detection.bbox_xyxy[1],
            ),
            dtype=np.float64,
        )
        self.kalman_filter: KalmanFilterXYWH | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.NEW
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.cls = self.class_id
        self.detection_index = int(detection_index)
        self.idx = self.detection_index
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @property
    def age(self) -> int:
        return self.frame_id - self.start_frame

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        value = self.mean[:4].copy()
        value[:2] -= value[2:] / 2.0
        return value

    @property
    def xywh(self) -> np.ndarray:
        return xyxy_to_xywh(self.xyxy)

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return np.asarray(
                (self._tlwh[0], self._tlwh[1], self._tlwh[0] + self._tlwh[2], self._tlwh[1] + self._tlwh[3]),
                dtype=np.float64,
            )
        return xywh_to_xyxy(self.mean[:4])

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.xyxy)

    @property
    def result(self) -> list[float]:
        return [*self.xyxy.tolist(), float(self.track_id), float(self.score), float(self.class_id), float(self.idx)]

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int, track_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = int(track_id)
        self.mean, self.covariance = kalman_filter.initiate(xyxy_to_xywh(self.xyxy))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        # Baseline BoT-SORT emits tracks created on the first frame directly;
        # later first sightings become confirmed after one successful match.
        self.is_activated = frame_id == 1
        self.frame_id = int(frame_id)
        self.start_frame = int(frame_id)

    def predict(self) -> None:
        if self.mean is None or self.covariance is None or self.kalman_filter is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[6:8] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    def apply_warp(self, warp: np.ndarray) -> None:
        if self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = warp_xywh_state(self.mean, self.covariance, warp)

    def update(self, detection_track: "Track", frame_id: int) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            raise RuntimeError("track must be activated before update")
        self.frame_id = int(frame_id)
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            xyxy_to_xywh(detection_track.xyxy),
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = float(detection_track.score)
        self.class_id = int(detection_track.class_id)
        self.cls = self.class_id
        self.detection_index = int(detection_track.detection_index)
        self.idx = self.detection_index

    def re_activate(self, detection_track: "Track", frame_id: int, new_id: bool = False, track_id: int | None = None) -> None:
        self.update(detection_track, frame_id)
        if new_id:
            if track_id is None:
                raise ValueError("track_id is required when assigning a new ID")
            self.track_id = int(track_id)
        self.tracklet_len = 0

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, state={self.state.name}, frames={self.start_frame}-{self.end_frame})"
