"""Adapters from the repository's detector/cache formats to ``mot_tracker``.

This module is intentionally separate from the baseline tools.  It may reuse
their dataset, cache and serialization helpers, but the tracking path below
only passes normalized ``Detection`` objects into the standalone package.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mot_tracker import BoTSORT, BoTSORTConfig, Detection, TrackOutput

from tools import visdrone_mot as baseline


SUPPORTED_GMC_METHODS = ("sparseOptFlow", "none")


class StandaloneAdapterError(RuntimeError):
    """An error caused by an invalid adapter input or output."""


def standalone_config(values: Mapping[str, Any]) -> BoTSORTConfig:
    """Map legacy tracker YAML fields to the standalone public config."""

    if values.get("tracker_type") not in {None, "botsort"}:
        raise StandaloneAdapterError("Only tracker_type=botsort is supported")
    if bool(values.get("with_reid", False)):
        raise StandaloneAdapterError("The standalone v1 tracker does not support ReID")
    method = values.get("gmc_method", "sparseOptFlow")
    if method not in SUPPORTED_GMC_METHODS:
        raise StandaloneAdapterError(f"gmc_method must be one of {SUPPORTED_GMC_METHODS}")
    try:
        return BoTSORTConfig.from_mapping({**values, "gmc_method": method})
    except (TypeError, ValueError) as exc:
        raise StandaloneAdapterError(f"Invalid BoT-SORT config: {exc}") from exc


def standalone_config_from_yaml(path: Path) -> BoTSORTConfig:
    try:
        values = baseline._load_yaml(path)
    except Exception as exc:
        raise StandaloneAdapterError(f"Cannot read tracker config: {path}") from exc
    if not isinstance(values, Mapping):
        raise StandaloneAdapterError(f"Tracker config must be a mapping: {path}")
    return standalone_config(values)


def _track_row(track: TrackOutput, frame_index: int) -> tuple[Any, ...]:
    x1, y1, x2, y2 = (float(value) for value in track.bbox_xyxy)
    width, height = x2 - x1, y2 - y1
    if not all(math.isfinite(value) for value in (x1, y1, width, height, track.score)):
        raise StandaloneAdapterError("Tracker returned a non-finite output")
    if width <= 0 or height <= 0:
        raise StandaloneAdapterError("Tracker returned a non-positive bounding box")
    if not 0 <= track.class_id < baseline.NUM_CLASSES:
        raise StandaloneAdapterError(f"Tracker returned invalid YOLO class {track.class_id}")
    return (
        int(frame_index),
        int(track.track_id),
        x1,
        y1,
        width,
        height,
        float(track.score),
        baseline.yolo_to_visdrone_class(track.class_id),
        -1,
        -1,
    )


def track_cached_sequence(
    *,
    cache: baseline.DetectionCache,
    sequence_dir: Path,
    config: BoTSORTConfig,
    output_path: Path,
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> list[tuple[Any, ...]]:
    """Replay one detector cache through the standalone tracker."""

    if output_path.exists() and not overwrite:
        raise StandaloneAdapterError(f"Prediction already exists; pass --overwrite: {output_path}")
    try:
        import cv2
    except ImportError as exc:
        raise StandaloneAdapterError("opencv-python is required for standalone tracking") from exc

    frames = baseline.sequence_frames(sequence_dir)
    if frame_limit is not None:
        if frame_limit <= 0:
            raise StandaloneAdapterError("frame_limit must be positive")
        frames = frames[:frame_limit]
    if len(frames) != cache.num_frames:
        raise StandaloneAdapterError(f"Cache frame count does not match sequence {cache.sequence}")

    tracker = BoTSORT(config)
    rows: list[tuple[Any, ...]] = []
    for position, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise StandaloneAdapterError(f"OpenCV could not read frame: {frame_path}")
        xyxy, confidence, classes = cache.detections_for(position)
        try:
            detections = [
                Detection(tuple(float(value) for value in box), float(score), int(class_id))
                for box, score, class_id in zip(xyxy, confidence, classes)
            ]
            outputs = tracker.update(detections, image)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise StandaloneAdapterError(f"Tracking failed at {cache.sequence} frame {position + 1}: {exc}") from exc
        rows.extend(_track_row(track, position + 1) for track in outputs)

    baseline.validate_prediction_rows(rows, len(frames))
    baseline.atomic_write_text(output_path, baseline.serialize_visdrone_rows(rows, len(frames)))
    return rows


def track_cached_split(
    *,
    data_root: Path,
    split: str,
    cache_dir: Path,
    prediction_dir: Path,
    config: BoTSORTConfig,
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> dict[str, Path]:
    """Replay all sequences in a detector cache with one tracker per sequence."""

    metadata = baseline.load_cache_metadata(cache_dir)
    if metadata.get("split") != split:
        raise StandaloneAdapterError(f"Cache split is {metadata.get('split')!r}, requested {split!r}")
    if metadata.get("frame_limit") != frame_limit:
        raise StandaloneAdapterError(
            f"Cache frame_limit is {metadata.get('frame_limit')!r}, requested {frame_limit!r}"
        )
    sequences = baseline.split_sequences(data_root, split)
    selected = {
        name: (frames[:frame_limit] if frame_limit is not None else frames)
        for name, frames in sequences.items()
    }
    if metadata.get("manifest") != baseline.split_manifest(selected):
        raise StandaloneAdapterError("Cache frame manifest does not match the current dataset")

    prediction_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for sequence, frames in selected.items():
        cache = baseline.load_detection_cache(cache_dir, sequence)
        if len(frames) != cache.num_frames:
            raise StandaloneAdapterError(f"Cache and data differ for {sequence}")
        output_path = prediction_dir / f"{sequence}.txt"
        track_cached_sequence(
            cache=cache,
            sequence_dir=data_root / split / "sequences" / sequence,
            config=config,
            output_path=output_path,
            overwrite=overwrite,
            frame_limit=frame_limit,
        )
        outputs[sequence] = output_path
    return outputs
