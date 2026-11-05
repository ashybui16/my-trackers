"""Standalone baseline BoT-SORT orchestration."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ...base import Tracker
from ...common.association import associate, apply_gmc, joint_tracks, remove_duplicate_tracks, subtract_tracks
from ...common.gmc import GMC
from ...common.kalman_filter import KalmanFilterXYWH
from ...common.track import Track, TrackState
from ...types import Detection, TrackOutput
from .config import BoTSORTConfig


class BoTSORT(Tracker):
    """Detector-agnostic baseline BoT-SORT tracker."""

    def __init__(self, config: BoTSORTConfig | None = None) -> None:
        if config is not None and not isinstance(config, BoTSORTConfig):
            raise TypeError("config must be a BoTSORTConfig")
        self.config = config or BoTSORTConfig()
        self.gmc = GMC(self.config.gmc_method, self.config.gmc_downscale)
        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []
        self.frame_id = 0
        self._next_track_id = 1
        self.kalman_filter = KalmanFilterXYWH()

    @property
    def name(self) -> str:
        return BoTSORTConfig.TYPE

    # Internal names used by the reference implementation are exposed as
    # read/write aliases for debugging and lifecycle fixtures only.
    @property
    def tracked_stracks(self) -> list[Track]:
        return self.tracked_tracks

    @tracked_stracks.setter
    def tracked_stracks(self, value: list[Track]) -> None:
        self.tracked_tracks = value

    @property
    def lost_stracks(self) -> list[Track]:
        return self.lost_tracks

    @lost_stracks.setter
    def lost_stracks(self, value: list[Track]) -> None:
        self.lost_tracks = value

    @property
    def removed_stracks(self) -> list[Track]:
        return self.removed_tracks

    @removed_stracks.setter
    def removed_stracks(self, value: list[Track]) -> None:
        self.removed_tracks = value

    def reset(self) -> None:
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self._next_track_id = 1
        self.kalman_filter = KalmanFilterXYWH()
        self.gmc.reset()

    def update(self, detections: Iterable[Detection], frame: np.ndarray | None = None) -> list[TrackOutput]:
        if detections is None:
            raise TypeError("detections must be an iterable of Detection")
        try:
            values = list(detections)
        except TypeError as exc:
            raise TypeError("detections must be an iterable of Detection") from exc
        if any(not isinstance(item, Detection) for item in values):
            raise TypeError("detections must contain only Detection objects")
        if self.config.gmc_method == "sparseOptFlow" and frame is None:
            raise ValueError("frame is required when gmc_method='sparseOptFlow'")
        if frame is not None and not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")

        self.frame_id += 1
        activated: list[Track] = []
        refound: list[Track] = []
        newly_lost: list[Track] = []
        newly_removed: list[Track] = []

        high_detections = [
            Track(item, index)
            for index, item in enumerate(values)
            if item.score >= self.config.track_high_thresh
        ]
        low_detections = [
            Track(item, index)
            for index, item in enumerate(values)
            if self.config.track_low_thresh < item.score < self.config.track_high_thresh
        ]

        unconfirmed = [item for item in self.tracked_tracks if not item.is_activated]
        confirmed = [item for item in self.tracked_tracks if item.is_activated]
        track_pool = joint_tracks(confirmed, self.lost_tracks)
        for item in track_pool:
            item.predict()
        if self.config.gmc_method == "sparseOptFlow":
            warp = self.gmc.apply(frame)  # type: ignore[arg-type]
            apply_gmc(track_pool, warp)
            apply_gmc(unconfirmed, warp)

        unmatched_tracks, unmatched_detections = self._first_association(
            track_pool, high_detections, activated, refound
        )
        unmatched_tracks, unmatched_detections = self._second_association(
            track_pool,
            unmatched_tracks,
            unmatched_detections,
            low_detections,
            activated,
            refound,
            newly_lost,
        )
        remaining_detections, remaining_high = self._unconfirmed_association(
            unconfirmed,
            unmatched_detections,
            high_detections,
            activated,
            newly_removed,
        )
        self._init_new_tracks(remaining_detections, remaining_high, activated)

        for item in self.lost_tracks:
            if self.frame_id - item.end_frame > self.config.track_buffer:
                item.mark_removed()
                newly_removed.append(item)

        self._merge_pools(activated, refound, newly_lost, newly_removed)
        return self._current_outputs()

    def _first_association(
        self,
        track_pool: list[Track],
        detections: list[Track],
        activated: list[Track],
        refound: list[Track],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        matches, unmatched_tracks, unmatched_detections = associate(
            track_pool,
            detections,
            match_thresh=self.config.match_thresh,
            use_score_fusion=self.config.fuse_score,
        )
        for track_index, detection_index in matches.tolist():
            track = track_pool[int(track_index)]
            detection = detections[int(detection_index)]
            if track.state == TrackState.TRACKED:
                track.update(detection, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(detection, self.frame_id)
                refound.append(track)
        return unmatched_tracks, unmatched_detections

    def _second_association(
        self,
        track_pool: list[Track],
        unmatched_pool_indices: tuple[int, ...],
        unmatched_high_detection_indices: tuple[int, ...],
        detections: list[Track],
        activated: list[Track],
        refound: list[Track],
        newly_lost: list[Track],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        remaining_active = [
            track_pool[index]
            for index in unmatched_pool_indices
            if track_pool[index].state == TrackState.TRACKED
        ]
        if remaining_active and detections:
            matches, unmatched_active, _ = associate(
                remaining_active,
                detections,
                match_thresh=0.5,
                use_score_fusion=False,
            )
            for track_index, detection_index in matches.tolist():
                remaining_active[int(track_index)].update(detections[int(detection_index)], self.frame_id)
                activated.append(remaining_active[int(track_index)])
        else:
            unmatched_active = tuple(range(len(remaining_active)))
        for index in unmatched_active:
            track = remaining_active[int(index)]
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)
        # The second-stage detections are not consumed by the unconfirmed/new
        # track path; low detections are only a recovery mechanism.
        return tuple(unmatched_pool_indices), tuple(unmatched_high_detection_indices)

    def _unconfirmed_association(
        self,
        unconfirmed: list[Track],
        unmatched_detection_indices: tuple[int, ...],
        detections: list[Track],
        activated: list[Track],
        newly_removed: list[Track],
    ) -> tuple[list[int], list[Track]]:
        remaining = [detections[index] for index in unmatched_detection_indices]
        if not unconfirmed:
            return list(range(len(remaining))), remaining
        matches, unmatched_unconfirmed, unmatched_detections = associate(
            unconfirmed,
            remaining,
            match_thresh=0.7,
            use_score_fusion=self.config.fuse_score,
        )
        for track_index, detection_index in matches.tolist():
            track = unconfirmed[int(track_index)]
            track.update(remaining[int(detection_index)], self.frame_id)
            activated.append(track)
        for index in unmatched_unconfirmed:
            track = unconfirmed[int(index)]
            track.mark_removed()
            newly_removed.append(track)
        return list(unmatched_detections), remaining

    def _init_new_tracks(self, indices: Iterable[int], detections: list[Track], activated: list[Track]) -> None:
        for index in indices:
            track = detections[int(index)]
            if track.score < self.config.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, self._next_track_id)
            self._next_track_id += 1
            activated.append(track)

    def _merge_pools(
        self,
        activated: list[Track],
        refound: list[Track],
        newly_lost: list[Track],
        newly_removed: list[Track],
    ) -> None:
        tracked = [item for item in self.tracked_tracks if item.state == TrackState.TRACKED]
        tracked = joint_tracks(tracked, activated)
        tracked = joint_tracks(tracked, refound)
        lost = subtract_tracks(self.lost_tracks, tracked)
        lost.extend(newly_lost)
        # Use the previous history before appending this frame's removals.
        # A newly expired track stays in the lost pool for one more association.
        lost = subtract_tracks(lost, self.removed_tracks)
        tracked, lost, _ = remove_duplicate_tracks(tracked, lost)
        # Duplicate suppression only drops pool entries, without marking REMOVED.
        self.tracked_tracks = tracked
        self.lost_tracks = lost
        self.removed_tracks.extend(newly_removed)
        if len(self.removed_tracks) > 1000:
            self.removed_tracks = self.removed_tracks[-1000:]

    def _current_outputs(self) -> list[TrackOutput]:
        outputs: list[TrackOutput] = []
        for item in self.tracked_tracks:
            if item.state != TrackState.TRACKED or not item.is_activated:
                continue
            outputs.append(
                TrackOutput(
                    track_id=item.track_id,
                    bbox_xyxy=item.bbox_xyxy,
                    score=item.score,
                    class_id=item.class_id,
                )
            )
        return outputs


__all__ = ["BoTSORT"]
