from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import lap
import numpy as np
import pytest

from mot_tracker import BoTSORT, BoTSORTConfig, Detection, TrackOutput
from mot_tracker.common.assignment import linear_assignment
from mot_tracker.common.geometry import iou_distance, iou_matrix, xywh_to_xyxy
from mot_tracker.common.kalman_filter import KalmanFilterXYWH
from mot_tracker.common.track import Track, TrackState


def detection(x: float = 0.0, *, score: float = 0.9, class_id: int = 0) -> Detection:
    return Detection((x, 0.0, x + 10.0, 10.0), score, class_id)


def test_public_types_validate_and_normalize() -> None:
    value = Detection([0, 1, 4, 5], np.float64(0.75), np.int64(3))
    assert value.bbox_xyxy == (0.0, 1.0, 4.0, 5.0)
    assert value.score == 0.75
    assert value.class_id == 3
    with pytest.raises(ValueError):
        Detection((0, 0, 0, 1), 0.5, 0)
    with pytest.raises(ValueError):
        Detection((0, 0, 1, 1), 1.1, 0)
    with pytest.raises(TypeError):
        Detection((0, 0, 1, 1), 0.5, True)
    assert TrackOutput(1, (0, 0, 1, 1), 0.5, 0).track_id == 1


def test_geometry_and_assignment_cover_empty_rectangular_and_tie_cases() -> None:
    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]).tolist() == [[1.0, 0.0]]
    matches, unmatched_rows, unmatched_cols = linear_assignment(
        np.asarray([[0.1, 0.9, 0.9], [0.9, 0.1, 0.9]]), thresh=0.5
    )
    assert matches.tolist() == [[0, 0], [1, 1]]
    assert unmatched_rows == ()
    assert unmatched_cols == (2,)
    tie_matches, _, _ = linear_assignment(np.zeros((2, 2)), thresh=0.1)
    assert tie_matches.tolist() == [[0, 1], [1, 0]]


def test_kalman_xywh_initiate_predict_multi_predict_and_update() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "kalman_xywh.json").read_text())
    kalman = KalmanFilterXYWH()
    mean, covariance = kalman.initiate(np.asarray(fixture["measurement"]))
    np.testing.assert_allclose(mean, fixture["initiate_mean"], atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), fixture["initiate_covariance_diag"], atol=1e-12)
    predicted_mean, predicted_covariance = kalman.predict(mean, covariance)
    multi_mean, multi_covariance = kalman.multi_predict(np.stack([mean, mean]), np.stack([covariance, covariance]))
    np.testing.assert_allclose(multi_mean[0], predicted_mean, atol=1e-12)
    np.testing.assert_allclose(multi_covariance[0], predicted_covariance, atol=1e-12)
    updated_mean, updated_covariance = kalman.update(predicted_mean, predicted_covariance, [10.5, 20.2, 4.1, 7.9])
    assert np.all(np.isfinite(updated_mean))
    np.testing.assert_allclose(updated_covariance, updated_covariance.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(updated_covariance)) > 0


def test_lifecycle_lost_reactivate_expire_empty_frame_and_reset() -> None:
    config = BoTSORTConfig(track_buffer=0, gmc_method="none")
    tracker = BoTSORT(config)
    assert tracker.update([detection()])[0].track_id == 1
    assert tracker.update([]) == []
    assert tracker.removed_tracks == []
    assert tracker.update([]) == []
    assert tracker.removed_tracks[0].track_id == 1
    expired = tracker.removed_tracks[0]
    assert tracker.lost_tracks == [expired]
    assert expired.state == TrackState.REMOVED
    recovered = tracker.update([detection(class_id=2)])[0]
    assert recovered.track_id == 1
    assert recovered.class_id == 2
    assert tracker.lost_tracks == []
    # The historical entry is the same object, as in Ultralytics/C++.
    assert tracker.removed_tracks == [expired]
    assert expired.state == TrackState.TRACKED
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert tracker.lost_tracks == []
    # Once it leaves the pool, a new first sighting needs confirmation.
    assert tracker.update([detection(class_id=2)]) == []
    assert tracker.update([detection(class_id=2)])[0].track_id == 2
    tracker.reset()
    assert tracker.frame_id == 0
    assert tracker.update([detection()])[0].track_id == 1


def test_reactivation_keeps_id_and_latest_score_and_class() -> None:
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none"))
    first = tracker.update([detection(class_id=1)])[0]
    assert tracker.update([]) == []
    recovered = tracker.update([detection(score=0.65, class_id=4)])[0]
    assert recovered.track_id == first.track_id
    assert recovered.score == pytest.approx(0.65)
    assert recovered.class_id == 4


def test_each_tracker_has_its_own_id_counter() -> None:
    config = BoTSORTConfig(gmc_method="none")
    first = BoTSORT(config)
    second = BoTSORT(config)
    assert first.update([detection()])[0].track_id == 1
    assert second.update([detection()])[0].track_id == 1


def test_sparse_gmc_requires_frame_and_none_does_not() -> None:
    with pytest.raises(ValueError):
        BoTSORT().update([detection()])
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none"))
    assert tracker.update([detection()])[0].track_id == 1


def test_package_does_not_import_ultralytics() -> None:
    assert "ultralytics" not in sys.modules


def test_sparse_gmc_synthetic_translation_and_reset() -> None:
    cv2 = pytest.importorskip("cv2")
    from mot_tracker.common.gmc import GMC, IDENTITY_WARP

    frame = np.zeros((160, 200), dtype=np.uint8)
    for x in range(15, 190, 20):
        cv2.circle(frame, (x, 40 + (x % 50)), 4, 255, -1)
    shifted = cv2.warpAffine(frame, np.asarray([[1, 0, 4], [0, 1, 3]], dtype=np.float32), (200, 160))
    gmc = GMC("sparseOptFlow", downscale=1)
    np.testing.assert_allclose(gmc.apply(frame), IDENTITY_WARP, atol=1e-12)
    warp = gmc.apply(shifted)
    assert warp[0, 2] == pytest.approx(4.0, abs=1.0)
    assert warp[1, 2] == pytest.approx(3.0, abs=1.0)
    gmc.reset()
    np.testing.assert_allclose(gmc.apply(frame), IDENTITY_WARP, atol=1e-12)


@pytest.mark.parametrize(
    "first,second,overlap,distance",
    [
        ((0, 0, 1, 1), (0, 0, 1, 1), 0.9999998807907104, 1.1920928955078125e-7),
        ((0, 0, 0.001, 0.001), (0, 0, 0.001, 0.001), 0.9090909361839294, 0.09090906381607056),
        ((0, 0, 2, 2), (1, 0, 3, 2), 0.3333333432674408, 0.6666666269302368),
        ((50, 25, 30, 35), (30, 25, 50, 35), 0.0, 1.0),
        ((0, 0, 0, 0), (0, 0, 0, 0), 0.0, 1.0),
        ((10000000.1, 0, 10000001.1, 1), (10000000.4, 0, 10000001.4, 1),
         0.9999998807907104, 1.1920928955078125e-7),
    ],
)
def test_iou_matches_ultralytics_and_cpp_golden(first, second, overlap, distance) -> None:
    # Golden values shared with cpp/tests/tracking/test_botsort.cpp.
    overlaps = iou_matrix([first], [second])
    distances = iou_distance([first], [second])
    assert overlaps.dtype == distances.dtype == np.float32
    assert float(overlaps[0, 0]) == overlap
    assert float(distances[0, 0]) == distance


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0), (2, 3), (3, 2), (9, 11)])
def test_assignment_matches_lapjv_for_rectangular_costs(shape) -> None:
    cost = np.random.default_rng(42).random(shape)
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)
    assert matches.shape[1] == 2
    if not cost.size:
        assert matches.shape == (0, 2)
        assert unmatched_rows == tuple(range(shape[0]))
        assert unmatched_cols == tuple(range(shape[1]))
        return
    _, rows, cols = lap.lapjv(cost, extend_cost=True, cost_limit=0.5)
    assert matches.tolist() == [[i, int(j)] for i, j in enumerate(rows) if j >= 0]
    assert unmatched_rows == tuple(np.flatnonzero(rows < 0))
    assert unmatched_cols == tuple(np.flatnonzero(cols < 0))


def test_assignment_cost_limit_is_part_of_optimization() -> None:
    # Filtering a full matching afterwards would return two expensive matches.
    cost = np.asarray([[0.0, 0.8], [0.8, 1.7]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 1.0)
    assert matches.tolist() == [[0, 0]]
    assert unmatched_rows == unmatched_cols == (1,)
    matches, unmatched_rows, unmatched_cols = linear_assignment(np.ones((2, 3)), 0.5)
    assert matches.shape == (0, 2)
    assert unmatched_rows == (0, 1)
    assert unmatched_cols == (0, 1, 2)


def test_missing_lap_raises_import_error_without_fallback() -> None:
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         f"import sys; sys.path.insert(0, {source_dir!r}); sys.modules['lap'] = None; import mot_tracker"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "lap" in result.stderr


def test_expired_track_leaves_pool_on_following_frame_without_match() -> None:
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none", track_buffer=0))
    tracker.update([detection()])
    tracker.update([])
    tracker.update([])
    expired = tracker.lost_tracks[0]
    assert expired.state == TrackState.REMOVED
    assert tracker.removed_tracks == [expired]
    tracker.update([])
    assert tracker.lost_tracks == []
    # Append each frame's removals without deduplicating history entries.
    assert tracker.removed_tracks == [expired, expired]
    assert tracker.update([detection()]) == []
    assert tracker.update([detection()])[0].track_id == 2


def test_removed_history_keeps_latest_1000_entries_in_append_order() -> None:
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none", track_buffer=0))
    for index in range(1010):
        tracker.update([detection(20.0 * index)])
        for _ in range(3):
            tracker.update([])
        assert len(tracker.removed_tracks) <= 1000
    assert [track.track_id for track in tracker.removed_tracks] == list(range(11, 1011))
    assert tracker.lost_tracks == []
    tracker.reset()
    assert tracker.removed_tracks == []


@pytest.mark.parametrize("active_frame,lost_frame,drop_active", [(3, 1, False), (1, 3, True), (2, 2, True)])
def test_duplicate_suppression_only_drops_pool_entries(active_frame, lost_frame, drop_active) -> None:
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none"))
    active, stale = Track(detection()), Track(detection())
    active.activate(tracker.kalman_filter, 1, 1)
    stale.activate(tracker.kalman_filter, 1, 2)
    active.frame_id = active_frame
    stale.frame_id = lost_frame
    stale.mark_lost()
    tracker.tracked_tracks = [active]
    tracker.lost_tracks = [stale]
    tracker._merge_pools([], [], [], [])
    assert tracker.tracked_tracks == ([] if drop_active else [active])
    assert tracker.lost_tracks == ([stale] if drop_active else [])
    assert active.state == TrackState.TRACKED
    assert stale.state == TrackState.LOST
    assert tracker.removed_tracks == []


def test_signed_gmc_bbox_predict_update_and_output_match_golden() -> None:
    kalman = KalmanFilterXYWH()
    track = Track(Detection((30, 25, 50, 35), 0.9, 0))
    track.activate(kalman, 1, 1)
    angle = 2.2
    track.apply_warp(np.asarray([[np.cos(angle), -np.sin(angle), 5], [np.sin(angle), np.cos(angle), 7]]))
    expected_box = [-32.86744361315013, 16.542364183204057, -52.722429996452945, 26.8272810870424]
    np.testing.assert_allclose(track.xyxy, expected_box, rtol=0, atol=1e-12)
    np.testing.assert_allclose(track.xywh, track.mean[:4], rtol=0, atol=0)
    np.testing.assert_allclose(xywh_to_xyxy(track.xywh), expected_box, rtol=0, atol=1e-12)
    assert track.xywh[2] < 0
    assert TrackOutput(1, track.bbox_xyxy, 0.9, 0).bbox_xyxy == track.bbox_xyxy
    with pytest.raises(ValueError):
        Detection(track.bbox_xyxy, 0.9, 0)
    track.predict()
    assert track.mean[2] == pytest.approx(-19.854986383302816, abs=1e-12)
    track.update(Track(Detection((39.5, 29.5, 40.5, 30.5), 0.8, 2)), 2)
    np.testing.assert_allclose(
        track.mean,
        [18.502678114060394, 27.062568670904014, -3.20462825092617, 1.0464663753364114,
         11.179857593517506, 0.6854816591811217, 3.49599153973328, -2.5820180205521126],
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.diag(track.covariance),
        [0.7401845468830346, 0.24617419319732825, 0.7401845468830346, 0.24617419319732825,
         0.6662386392927906, 0.8718378512325102, 0.6662386392927906, 0.8718378512325102],
        rtol=0, atol=1e-10,
    )
    assert track.mean[2] < 0


def test_signed_gmc_track_becomes_lost_and_new_detection_gets_new_id() -> None:
    tracker = BoTSORT(BoTSORTConfig(gmc_method="none", track_buffer=1))
    observed = Detection((30, 25, 50, 35), 0.9, 0)
    assert tracker.update([observed])[0].track_id == 1
    cosine, sine = np.cos(2.2), np.sin(2.2)
    warp = np.asarray([[cosine, -sine, 40 - 40 * cosine + 30 * sine],
                       [sine, cosine, 30 - 40 * sine - 30 * cosine]])
    tracker.tracked_tracks[0].apply_warp(warp)
    assert tracker._current_outputs()[0].bbox_xyxy[0] > tracker._current_outputs()[0].bbox_xyxy[2]
    assert tracker.update([observed]) == []
    assert len(tracker.lost_tracks) == 1
    stale = tracker.lost_tracks[0]
    assert stale.track_id == 1 and stale.state == TrackState.LOST
    assert stale.xyxy[0] > stale.xyxy[2]
    assert tracker.update([observed])[0].track_id == 2


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_geometry_and_track_output_reject_nonfinite_coordinates(invalid) -> None:
    with pytest.raises(ValueError):
        iou_matrix([[0, 0, invalid, 1]], [[0, 0, 1, 1]])
    with pytest.raises(ValueError):
        xywh_to_xyxy([0, 0, invalid, 1])
    with pytest.raises(ValueError):
        TrackOutput(1, (0, 0, invalid, 1), 0.9, 0)


def _box_detection(class_id: int, score: float = 0.9, class_scores=None) -> Detection:
    return Detection((0.0, 0.0, 10.0, 10.0), score, class_id, class_scores)


def test_class_ema_resists_single_flip_and_alpha_zero_follows_latest() -> None:
    for alpha, expected in ((0.9, 4), (0.0, 3)):
        tracker = BoTSORT(BoTSORTConfig(gmc_method="none", class_ema_alpha=alpha))
        for frame in range(6):
            outputs = tracker.update([_box_detection(4 if frame < 5 else 3)])
        assert [item.class_id for item in outputs] == [expected]


def test_class_ema_blends_full_detector_scores() -> None:
    kalman = KalmanFilterXYWH()
    track = Track(_box_detection(1, 0.6, (0.2, 0.6, 0.1)), class_ema_alpha=0.5)
    track.activate(kalman, 1, 1)
    track.update(Track(_box_detection(0, 0.8, (0.8, 0.1, 0.0)), class_ema_alpha=0.5), 2)
    np.testing.assert_allclose(track.class_vector, [0.5, 0.35, 0.05], atol=1e-12)
    assert track.class_id == track.cls == 0


def test_class_ema_config_and_scores_validation() -> None:
    assert BoTSORTConfig.from_mapping({"class_ema_alpha": 0.9}).class_ema_alpha == 0.9
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError):
            BoTSORTConfig(class_ema_alpha=bad)
    with pytest.raises(ValueError):
        _box_detection(0, 0.5, (1.5,))
