"""Contract every tracker must satisfy (port of cpp/tests/tracking/test_tracker_contract.cpp)."""

from __future__ import annotations

import numpy as np
import pytest

from mot_tracker import (
    BoTSORTConfig,
    Detection,
    Tracker,
    TrackOutput,
    available_trackers,
    create_tracker,
    default_tracker_config,
    tracker_config_from_mapping,
    tracker_type,
    validate,
)

IMAGE = np.full((720, 1280, 3), (40, 80, 120), dtype=np.uint8)


def frame_detections(frame: int) -> list[Detection]:
    dx = 2.0 * frame
    return [
        Detection((100 + dx, 100, 140 + dx, 180), 0.95, 0),
        Detection((400 + dx, 300, 460 + dx, 340), 0.90, 3),
    ]


def run_sequence(tracker: Tracker, frames: int) -> list[list[TrackOutput]]:
    return [tracker.update(frame_detections(frame), IMAGE) for frame in range(frames)]


def test_registry_is_not_empty() -> None:
    assert available_trackers()


def test_unknown_tracker_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tracker type 'does-not-exist'"):
        default_tracker_config("does-not-exist")


@pytest.mark.parametrize("type_name", available_trackers())
def test_tracker_contract(type_name: str) -> None:
    config = default_tracker_config(type_name)
    assert tracker_type(config) == type_name
    tracker = create_tracker(config)
    assert isinstance(tracker, Tracker)
    assert tracker.name == type_name

    assert tracker.update([], IMAGE) == []
    tracker.reset()

    first = run_sequence(tracker, 12)
    last = first[-1]
    assert len(last) == 2, "two steady objects are tracked"
    id_by_class: dict[int, int] = {}
    for outputs in first:
        for track in outputs:
            assert id_by_class.setdefault(track.class_id, track.track_id) == track.track_id
            assert track.track_id > 0
    assert last[0].track_id != last[1].track_id

    tracker.reset()
    again = run_sequence(tracker, 12)
    expected = run_sequence(create_tracker(config), 12)
    assert again == expected, "reset matches a fresh tracker"


def test_create_tracker_validates_config() -> None:
    with pytest.raises(TypeError):
        create_tracker(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate({"type": "botsort"})  # type: ignore[arg-type]


def test_config_from_nested_mapping() -> None:
    config = tracker_config_from_mapping(
        {"type": "botsort", "botsort": {"track_buffer": 45, "gmc_method": "none"}}
    )
    assert config == BoTSORTConfig(track_buffer=45, gmc_method="none")
    assert tracker_config_from_mapping({"type": "botsort"}) == BoTSORTConfig()


def test_config_from_legacy_flat_mapping() -> None:
    assert tracker_config_from_mapping({"match_thresh": 0.7}) == BoTSORTConfig(match_thresh=0.7)


@pytest.mark.parametrize(
    "values",
    [
        {"type": "botsort", "botsort": {"with_reid": True}},
        {"type": "botsort", "bytetrack": {}},
        {"type": "nope"},
        {"with_reid": True},
    ],
)
def test_config_from_mapping_rejects_unknown_keys(values: dict) -> None:
    with pytest.raises(ValueError):
        tracker_config_from_mapping(values)
