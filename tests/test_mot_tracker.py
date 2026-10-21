from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from mot_tracker import BoTSORT, BoTSORTConfig, Detection, TrackOutput
from mot_tracker.assignment import linear_assignment
from mot_tracker.geometry import iou_matrix
from mot_tracker.kalman_filter import KalmanFilterXYWH


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
    assert tie_matches.tolist() == [[0, 0], [1, 1]]


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
    # A new first sighting after removal is unconfirmed until the next match.
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
    from mot_tracker.gmc import GMC, IDENTITY_WARP

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
