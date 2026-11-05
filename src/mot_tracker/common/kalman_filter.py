"""Constant-velocity Kalman filter in XYWH measurement space."""

from __future__ import annotations

import numpy as np


class KalmanFilterXYWH:
    """Eight-dimensional ``(cx, cy, w, h, vx, vy, vw, vh)`` Kalman filter."""

    def __init__(self) -> None:
        self._motion_mat = np.eye(8, dtype=np.float64)
        self._motion_mat[:4, 4:] += np.eye(4, dtype=np.float64)
        self._update_mat = np.eye(4, 8, dtype=np.float64)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(measurement, dtype=np.float64)
        if value.shape != (4,) or not np.all(np.isfinite(value)) or np.any(value[2:] <= 0):
            raise ValueError("measurement must be finite (cx, cy, width, height) with positive size")
        mean = np.r_[value, np.zeros(4, dtype=np.float64)]
        std = np.asarray(
            (
                2 * self._std_weight_position * value[2],
                2 * self._std_weight_position * value[3],
                2 * self._std_weight_position * value[2],
                2 * self._std_weight_position * value[3],
                10 * self._std_weight_velocity * value[2],
                10 * self._std_weight_velocity * value[3],
                10 * self._std_weight_velocity * value[2],
                10 * self._std_weight_velocity * value[3],
            ),
            dtype=np.float64,
        )
        return mean, np.diag(std * std)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_value, covariance_value = self._state(mean, covariance)
        std_pos = self._std_weight_position * mean_value[[2, 3, 2, 3]]
        std_vel = self._std_weight_velocity * mean_value[[2, 3, 2, 3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        predicted_mean = self._motion_mat @ mean_value
        predicted_covariance = self._motion_mat @ covariance_value @ self._motion_mat.T + motion_cov
        return predicted_mean, predicted_covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray(mean, dtype=np.float64)
        covariances = np.asarray(covariance, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != 8 or covariances.shape != (len(means), 8, 8):
            raise ValueError("mean must have shape (N, 8) and covariance shape (N, 8, 8)")
        if len(means) == 0:
            return means.copy(), covariances.copy()
        std_pos = self._std_weight_position * means[:, [2, 3, 2, 3]]
        std_vel = self._std_weight_velocity * means[:, [2, 3, 2, 3]]
        motion_cov = np.zeros((len(means), 8, 8), dtype=np.float64)
        motion_cov[:, np.arange(8), np.arange(8)] = np.square(np.concatenate((std_pos, std_vel), axis=1))
        predicted_means = means @ self._motion_mat.T
        predicted_covariances = self._motion_mat @ covariances @ self._motion_mat.T + motion_cov
        return predicted_means, predicted_covariances

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_value, covariance_value = self._state(mean, covariance)
        std = self._std_weight_position * mean_value[[2, 3, 2, 3]]
        innovation_covariance = np.diag(np.square(std))
        if confidence is not None:
            confidence_value = float(confidence)
            if not np.isfinite(confidence_value):
                raise ValueError("confidence must be finite")
            innovation_covariance *= max(1.0 - confidence_value, 0.05)
        return mean_value[:4].copy(), covariance_value[:4, :4] + innovation_covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_value, covariance_value = self._state(mean, covariance)
        measurement_value = np.asarray(measurement, dtype=np.float64)
        if measurement_value.shape != (4,) or not np.all(np.isfinite(measurement_value)) or np.any(measurement_value[2:] <= 0):
            raise ValueError("measurement must be finite (cx, cy, width, height) with positive size")
        projected_mean, projected_covariance = self.project(mean_value, covariance_value, confidence)
        kalman_gain = np.linalg.solve(projected_covariance, covariance_value[:, :4].T).T
        innovation = measurement_value - projected_mean
        updated_mean = mean_value + kalman_gain @ innovation
        updated_covariance = covariance_value - kalman_gain @ projected_covariance @ kalman_gain.T
        # Roundoff can make an otherwise symmetric covariance very slightly
        # asymmetric; preserving symmetry makes golden numeric fixtures stable.
        updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
        return updated_mean, updated_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        projected_mean, projected_covariance = self.project(mean, covariance)
        values = np.asarray(measurements, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("measurements must have shape (N, 4)")
        if only_position:
            projected_mean = projected_mean[:2]
            projected_covariance = projected_covariance[:2, :2]
            values = values[:, :2]
        difference = values - projected_mean
        if metric == "gaussian":
            return np.sum(difference * difference, axis=1)
        if metric != "maha":
            raise ValueError("metric must be 'maha' or 'gaussian'")
        factor = np.linalg.cholesky(projected_covariance)
        solved = np.linalg.solve(factor, difference.T)
        return np.sum(solved * solved, axis=0)

    @staticmethod
    def _state(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_value = np.asarray(mean, dtype=np.float64)
        covariance_value = np.asarray(covariance, dtype=np.float64)
        if mean_value.shape != (8,) or covariance_value.shape != (8, 8):
            raise ValueError("mean must have shape (8,) and covariance shape (8, 8)")
        if not np.all(np.isfinite(mean_value)) or not np.all(np.isfinite(covariance_value)):
            raise ValueError("mean and covariance must be finite")
        return mean_value, covariance_value
