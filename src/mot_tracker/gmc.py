"""Global motion compensation using sparse optical flow."""

from __future__ import annotations

import warnings

import numpy as np


IDENTITY_WARP = np.eye(2, 3, dtype=np.float64)


class GMC:
    """Stateful sparse-optical-flow GMC implementation.

    Only ``sparseOptFlow`` and ``none`` are supported by v1.  A failed OpenCV
    operation is isolated to the current frame and always advances the frame
    reference, allowing the next frame to recover.
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        if method not in {"sparseOptFlow", "none"}:
            raise ValueError("method must be 'sparseOptFlow' or 'none'")
        if isinstance(downscale, bool) or int(downscale) < 1:
            raise ValueError("downscale must be a positive integer")
        self.method = method
        self.downscale = int(downscale)
        self.prev_frame: np.ndarray | None = None
        self.prev_keypoints: np.ndarray | None = None
        self.initialized_first_frame = False
        self._cv2 = None
        self.feature_params = {
            "maxCorners": 400,
            "qualityLevel": 0.01,
            "minDistance": 0,
            "blockSize": 3,
            "useHarrisDetector": False,
            "k": 0.04,
        }

    # Names retained for easy inspection and parity with common GMC code.
    @property
    def prevFrame(self) -> np.ndarray | None:
        return self.prev_frame

    @property
    def prevKeyPoints(self) -> np.ndarray | None:
        return self.prev_keypoints

    @property
    def initializedFirstFrame(self) -> bool:
        return self.initialized_first_frame

    def _opencv(self):
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("sparseOptFlow GMC requires opencv-python") from exc
            self._cv2 = cv2
        return self._cv2

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        value = np.asarray(frame)
        if value.ndim not in {2, 3} or value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("frame must be a non-empty grayscale or BGR image")
        if value.ndim == 3 and value.shape[2] not in {1, 3}:
            raise ValueError("frame must have one grayscale or three BGR channels")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("frame must have a numeric dtype")
        if not np.all(np.isfinite(value)):
            raise ValueError("frame must contain finite values")
        return value

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._opencv()
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.dtype not in (np.uint8, np.float32):
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.downscale > 1:
            height, width = gray.shape[:2]
            gray = cv2.resize(gray, (max(1, width // self.downscale), max(1, height // self.downscale)))
        return gray

    def _remember(self, gray: np.ndarray, keypoints: np.ndarray | None) -> None:
        self.prev_frame = np.array(gray, copy=True)
        self.prev_keypoints = None if keypoints is None else np.array(keypoints, copy=True)
        self.initialized_first_frame = True

    def apply(self, frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        del detections  # v1 does not mask detector boxes for sparse optical flow.
        value = self._validate_frame(frame)
        if self.method == "none":
            return IDENTITY_WARP.copy()
        gray: np.ndarray | None = None
        keypoints: np.ndarray | None = None
        try:
            gray = self._gray(value)
            cv2 = self._opencv()
            keypoints = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            if not self.initialized_first_frame:
                self._remember(gray, keypoints)
                return IDENTITY_WARP.copy()
            if self.prev_keypoints is None or len(self.prev_keypoints) == 0:
                self._remember(gray, keypoints)
                self._warn("GMC has no previous features; using identity warp")
                return IDENTITY_WARP.copy()
            if keypoints is None or len(keypoints) == 0:
                self._remember(gray, keypoints)
                self._warn("GMC found no features; using identity warp")
                return IDENTITY_WARP.copy()
            matched, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, self.prev_keypoints, None)
            if matched is None or status is None:
                self._remember(gray, keypoints)
                self._warn("GMC optical flow returned no matches; using identity warp")
                return IDENTITY_WARP.copy()
            good = status.reshape(-1).astype(bool)
            previous_points = self.prev_keypoints.reshape(-1, 2)[good]
            current_points = matched.reshape(-1, 2)[good]
            warp = IDENTITY_WARP.copy()
            if len(previous_points) >= 5:
                estimated, _ = cv2.estimateAffinePartial2D(previous_points, current_points, cv2.RANSAC)
                if estimated is not None and estimated.shape == (2, 3) and np.all(np.isfinite(estimated)):
                    warp = np.asarray(estimated, dtype=np.float64)
                    if self.downscale > 1:
                        warp[:, 2] *= self.downscale
                else:
                    self._warn("GMC affine estimation failed; using identity warp")
            else:
                self._warn("GMC found too few optical-flow matches; using identity warp")
            self._remember(gray, keypoints)
            return warp
        except Exception as exc:
            # Reference state is updated even when OpenCV itself fails. Use a
            # cheap raw-frame reference when grayscale conversion was the part
            # that failed, so a later frame can still attempt recovery.
            self.prev_frame = np.array(gray if gray is not None else value, copy=True)
            self.prev_keypoints = None if keypoints is None else np.array(keypoints, copy=True)
            self.initialized_first_frame = True
            self._warn(f"GMC failed ({exc}); using identity warp")
            return IDENTITY_WARP.copy()

    @staticmethod
    def _warn(message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def reset_params(self) -> None:
        self.prev_frame = None
        self.prev_keypoints = None
        self.initialized_first_frame = False

    def reset(self) -> None:
        self.reset_params()


SparseOptFlowGMC = GMC

__all__ = ["GMC", "IDENTITY_WARP", "SparseOptFlowGMC"]
