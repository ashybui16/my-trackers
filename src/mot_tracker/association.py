"""Association and persistent track-pool helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .assignment import linear_assignment
from .geometry import fuse_score, iou_distance
from .track import Track, TrackState


def _boxes(items: Iterable[Track]) -> np.ndarray:
    values = [item.xyxy for item in items]
    return np.asarray(values, dtype=np.float64).reshape(-1, 4) if values else np.empty((0, 4), dtype=np.float64)


def track_distance(tracks: list[Track], detections: list[Track], *, use_score_fusion: bool) -> np.ndarray:
    distances = iou_distance(_boxes(tracks), _boxes(detections))
    if use_score_fusion and distances.size:
        distances = fuse_score(distances, [item.score for item in detections])
    return distances


def associate(
    tracks: list[Track],
    detections: list[Track],
    *,
    match_thresh: float,
    use_score_fusion: bool,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    distances = track_distance(tracks, detections, use_score_fusion=use_score_fusion)
    return linear_assignment(distances, match_thresh)


def joint_tracks(first: list[Track], second: list[Track]) -> list[Track]:
    ids = {item.track_id for item in first}
    return first + [item for item in second if item.track_id not in ids]


def subtract_tracks(first: list[Track], second: list[Track]) -> list[Track]:
    ids = {item.track_id for item in second}
    return [item for item in first if item.track_id not in ids]


def remove_duplicate_tracks(
    tracked: list[Track], lost: list[Track], *, iou_threshold: float = 0.85
) -> tuple[list[Track], list[Track], list[Track]]:
    """Remove duplicate active/lost tracks, preferring the longer-lived one."""

    if not tracked or not lost:
        return tracked, lost, []
    distances = iou_distance(_boxes(tracked), _boxes(lost))
    duplicate_pairs = np.argwhere(distances < 1.0 - iou_threshold)
    drop_tracked: set[int] = set()
    drop_lost: set[int] = set()
    removed: list[Track] = []
    for tracked_index, lost_index in duplicate_pairs.tolist():
        active = tracked[tracked_index]
        stale = lost[lost_index]
        active_life = active.frame_id - active.start_frame
        stale_life = stale.frame_id - stale.start_frame
        if active_life > stale_life:
            drop_lost.add(lost_index)
            removed.append(stale)
        else:
            drop_tracked.add(tracked_index)
            removed.append(active)
    return (
        [item for index, item in enumerate(tracked) if index not in drop_tracked],
        [item for index, item in enumerate(lost) if index not in drop_lost],
        removed,
    )


def apply_gmc(tracks: Iterable[Track], warp: np.ndarray) -> None:
    for track in tracks:
        track.apply_warp(warp)


def remove_duplicate_stracks(tracked: list[Track], lost: list[Track], *, dup_thresh: float = 0.15) -> tuple[list[Track], list[Track]]:
    """Two-return-value compatibility helper using an IoU-distance threshold."""

    active, stale, _ = remove_duplicate_tracks(tracked, lost, iou_threshold=1.0 - dup_thresh)
    return active, stale


joint_stracks = joint_tracks
sub_stracks = subtract_tracks


__all__ = [
    "associate",
    "apply_gmc",
    "joint_tracks",
    "remove_duplicate_tracks",
    "remove_duplicate_stracks",
    "subtract_tracks",
    "joint_stracks",
    "sub_stracks",
    "track_distance",
]
