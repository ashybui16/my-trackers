from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools.visdrone_mot import (
    DetectionCache,
    FingerprintMismatch,
    MOTObject,
    build_cache_fingerprint,
    combine_stats,
    default_tunable_config,
    detect_sequence,
    evaluate_pair_stats,
    load_detection_cache,
    metrics_from_stats,
    reset_tracker_for_sequence,
    save_detection_cache,
    serialize_visdrone_rows,
    suggest_tracker_params,
    tracker_config_fingerprint,
    validate_cache_fingerprint,
    validate_prediction_rows,
    visdrone_to_yolo_class,
    yolo_to_visdrone_class,
)


def test_detector_reads_file_sequence_in_single_frame_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = []
    for index in range(3):
        frame = tmp_path / f"{index:07d}.jpg"
        frame.write_bytes(b"unused by fake model")
        frames.append(frame)

    class FakeLoader:
        def __init__(self, paths, batch):
            self.paths = paths
            self.bs = batch
            self.nf = len(paths)

    ultralytics = ModuleType("ultralytics")
    ultralytics.__path__ = []
    data = ModuleType("ultralytics.data")
    data.__path__ = []
    loaders = ModuleType("ultralytics.data.loaders")
    loaders.LoadImagesAndVideos = FakeLoader
    loaders.SourceTypes = SimpleNamespace
    ultralytics.data = data
    data.loaders = loaders
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.data", data)
    monkeypatch.setitem(sys.modules, "ultralytics.data.loaders", loaders)

    class FakeModel:
        def predict(self, *, source, **kwargs):
            assert source.bs == 1
            assert source.nf == len(frames)
            assert source.paths == [str(path) for path in frames]
            assert kwargs["batch"] == 1
            return iter(SimpleNamespace(boxes=None) for _ in frames)

    frame_indices, offsets, xyxy, confidence, classes = detect_sequence(FakeModel(), frames, device="cpu")

    assert frame_indices.tolist() == [1, 2, 3]
    assert offsets.tolist() == [0, 0, 0, 0]
    assert xyxy.shape == (0, 4)
    assert confidence.shape == (0,)
    assert classes.shape == (0,)


def test_class_mapping_is_bijective() -> None:
    assert [yolo_to_visdrone_class(value) for value in range(10)] == list(range(1, 11))
    assert [visdrone_to_yolo_class(value) for value in range(1, 11)] == list(range(10))
    with pytest.raises(ValueError):
        yolo_to_visdrone_class(10)
    with pytest.raises(ValueError):
        visdrone_to_yolo_class(0)


def test_serialization_rejects_duplicate_track_ids() -> None:
    rows = [(1, 1, 0, 0, 10, 10, 0.9, 1, -1, -1), (1, 1, 20, 20, 10, 10, 0.8, 2, -1, -1)]
    with pytest.raises(Exception):
        serialize_visdrone_rows(rows, 1)
    assert serialize_visdrone_rows([rows[0]], 1).endswith("-1,-1\n")


def test_cache_roundtrip_and_fingerprint(tmp_path: Path) -> None:
    frame = tmp_path / "0000001.jpg"
    frame.write_bytes(b"frame")
    sequences = {"seq": [frame]}
    fingerprint = build_cache_fingerprint(
        split="val", sequences=sequences, weights_sha256="weights", inference={"conf": 0.01}
    )
    cache_dir = tmp_path / "cache"
    cache = DetectionCache(
        "seq",
        np.array([1]),
        np.array([0, 1]),
        np.array([[1, 2, 4, 6]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([0], dtype=np.int64),
        fingerprint,
    )
    save_detection_cache(cache_dir, cache)
    (cache_dir / "metadata.json").write_text(json.dumps(fingerprint))
    loaded = load_detection_cache(cache_dir, "seq")
    assert loaded.detections_for(0)[0].tolist() == [[1, 2, 4, 6]]
    validate_cache_fingerprint(cache_dir, fingerprint)
    frame.write_bytes(b"changed")
    changed = build_cache_fingerprint(
        split="val", sequences=sequences, weights_sha256="weights", inference={"conf": 0.01}
    )
    with pytest.raises(FingerprintMismatch):
        validate_cache_fingerprint(cache_dir, changed)


def test_perfect_fixture_is_100_percent() -> None:
    ground_truth = [
        MOTObject(1, 1, (0, 0, 10, 10), 1),
        MOTObject(2, 1, (1, 0, 10, 10), 1),
    ]
    predictions = [
        MOTObject(1, 7, (0, 0, 10, 10), 1, 1.0),
        MOTObject(2, 7, (1, 0, 10, 10), 1, 1.0),
    ]
    metrics = metrics_from_stats(evaluate_pair_stats(ground_truth, predictions))
    assert metrics["HOTA"] == pytest.approx(100.0)
    assert metrics["MOTA"] == pytest.approx(100.0)
    assert metrics["IDF1"] == pytest.approx(100.0)


def test_id_switch_reduces_association_and_identity() -> None:
    ground_truth = [MOTObject(1, 1, (0, 0, 10, 10), 1), MOTObject(2, 1, (0, 0, 10, 10), 1)]
    predictions = [MOTObject(1, 7, (0, 0, 10, 10), 1, 1.0), MOTObject(2, 8, (0, 0, 10, 10), 1, 1.0)]
    metrics = metrics_from_stats(evaluate_pair_stats(ground_truth, predictions))
    assert metrics["IDSW"] == 1
    assert metrics["AssA"] < 100.0
    assert metrics["IDF1"] < 100.0


def test_ignore_region_does_not_count_as_false_positive() -> None:
    ground_truth = [MOTObject(1, 0, (0, 0, 20, 20), 0, ignore=True)]
    predictions = [MOTObject(1, 7, (2, 2, 10, 10), 1, 1.0)]
    metrics = metrics_from_stats(evaluate_pair_stats(ground_truth, predictions))
    assert metrics["FP"] == 0
    assert metrics["GT"] == 0


def test_search_space_is_constrained() -> None:
    class FakeTrial:
        def suggest_float(self, name, low, high):
            return {"track_high_thresh": 0.30, "track_low_thresh": 0.10, "new_track_thresh": 0.40, "match_thresh": 0.80}[name]

        def suggest_categorical(self, name, choices):
            if name == "gmc_method":
                assert choices == ["sparseOptFlow"]
            return choices[0]

    params = suggest_tracker_params(FakeTrial())
    assert 0.15 <= params["track_high_thresh"] <= 0.60
    assert 0.01 <= params["track_low_thresh"] < params["track_high_thresh"]
    assert params["track_high_thresh"] <= params["new_track_thresh"] <= 0.80
    assert params["track_buffer"] in (15, 30, 45, 60, 90)
    assert 0.60 <= params["match_thresh"] <= 0.95
    assert params["gmc_method"] == "sparseOptFlow"


def test_tracker_reset_is_called_at_sequence_boundary() -> None:
    class FakeTracker:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls += 1

    tracker = FakeTracker()
    assert reset_tracker_for_sequence(tracker) is tracker
    assert tracker.calls == 1


def test_config_fingerprint_changes_with_tuned_parameter() -> None:
    config = default_tunable_config()
    changed = dict(config)
    changed["track_buffer"] = 60
    assert tracker_config_fingerprint(config) != tracker_config_fingerprint(changed)
