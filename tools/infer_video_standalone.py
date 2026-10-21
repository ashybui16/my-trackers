#!/usr/bin/env python3
"""YOLO ``predict`` plus standalone BoT-SORT video inference.

The historical ``tools/infer_video.py`` is left untouched.  This CLI is the
detector/tracker-separated implementation from PLAN1.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
VENDORED_ULTRALYTICS = PROJECT_ROOT / "third-party" / "my-ultralytics"
for path in (PROJECT_ROOT, SOURCE_ROOT, VENDORED_ULTRALYTICS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mot_tracker import BoTSORT, Detection, TrackOutput
from tools.mot_tracker_adapter import standalone_config_from_yaml


DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "26leaf1.pt"
DEFAULT_TRACKER = VENDORED_ULTRALYTICS / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "video_inference_standalone"


class InferenceError(RuntimeError):
    """An error that can be fixed by changing an input or option."""


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def detections_from_result(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    values = [boxes.xyxy, boxes.conf, boxes.cls]
    arrays = [value.detach().cpu().numpy() if hasattr(value, "detach") else value for value in values]
    xyxy = np.asarray(arrays[0], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(arrays[1], dtype=np.float64).reshape(-1)
    classes = np.asarray(arrays[2], dtype=np.int64).reshape(-1)
    if len(xyxy) != len(scores) or len(xyxy) != len(classes):
        raise InferenceError("Ultralytics returned inconsistent detection arrays")
    try:
        return [
            Detection(tuple(float(value) for value in box), float(score), int(class_id))
            for box, score, class_id in zip(xyxy, scores, classes)
        ]
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"Invalid detector output: {exc}") from exc


def prediction_rows(tracks: Sequence[TrackOutput], frame_index: int) -> list[str]:
    rows: list[str] = []
    for track in tracks:
        x1, y1, x2, y2 = (float(value) for value in track.bbox_xyxy)
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in (x1, y1, width, height, track.score)):
            raise InferenceError(f"Invalid track output at frame {frame_index}")
        rows.append(
            f"{frame_index},{track.track_id},{x1:.3f},{y1:.3f},{width:.3f},{height:.3f},"
            f"{track.score:.6f},{track.class_id + 1},-1,-1\n"
        )
    return rows


def render_tracks(frame: np.ndarray, tracks: Sequence[TrackOutput]) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox_xyxy)
        color = ((37 * track.track_id) % 255, (67 * track.track_id) % 255, (101 * track.track_id) % 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{track.track_id}:{track.class_id + 1} {track.score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def video_fps(video: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise InferenceError(f"OpenCV cannot open input video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if math.isfinite(fps) and fps > 0 else 30.0


def run_video(
    video: Path,
    *,
    weights: Path,
    tracker: Path,
    output_dir: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    overwrite: bool,
) -> tuple[Path, Path, int, int]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise InferenceError("Run this CLI in the detector environment with OpenCV and Ultralytics installed") from exc
    video = video.expanduser().resolve()
    weights = weights.expanduser().resolve()
    tracker = tracker.expanduser().resolve()
    for path, label in ((video, "video"), (weights, "weights"), (tracker, "tracker config")):
        if not path.is_file():
            raise InferenceError(f"Missing {label}: {path}")
    if imgsz <= 0 or max_det <= 0:
        raise InferenceError("--imgsz and --max-det must be positive")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise InferenceError("--conf and --iou must be in [0, 1]")
    try:
        standalone_config = standalone_config_from_yaml(tracker)
    except Exception as exc:
        raise InferenceError(f"Invalid tracker config: {exc}") from exc

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{video.stem}_predictions.txt"
    annotated_path = output_dir / f"{video.stem}_annotated.mp4"
    existing = [path for path in (predictions_path, annotated_path) if path.exists()]
    if existing and not overwrite:
        raise InferenceError("Output already exists; pass --overwrite")

    model = YOLO(str(weights))
    tracker_instance = BoTSORT(standalone_config)
    results = model.predict(
        source=str(video),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=resolve_device(device),
        save=False,
        verbose=False,
    )
    fps = video_fps(video)
    text_handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{video.stem}.", suffix=".tmp.txt", dir=output_dir, delete=False
    )
    temp_predictions = Path(text_handle.name)
    video_handle = tempfile.NamedTemporaryFile(prefix=f".{video.stem}.", suffix=".tmp.mp4", dir=output_dir, delete=False)
    temp_video = Path(video_handle.name)
    video_handle.close()
    writer = None
    frame_count = 0
    row_count = 0
    try:
        for frame_count, result in enumerate(results, start=1):
            frame = getattr(result, "orig_img", None)
            if frame is None:
                raise InferenceError("Ultralytics did not return a decoded source frame")
            tracks = tracker_instance.update(detections_from_result(result), frame)
            annotated = render_tracks(frame, tracks)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise InferenceError(f"OpenCV cannot create output video: {annotated_path}")
            writer.write(annotated)
            rows = prediction_rows(tracks, frame_count)
            text_handle.writelines(rows)
            row_count += len(rows)
        if frame_count == 0:
            raise InferenceError(f"No frames were decoded from: {video}")
        text_handle.flush()
        os.fsync(text_handle.fileno())
        text_handle.close()
        if writer is not None:
            writer.release()
            writer = None
        os.replace(temp_predictions, predictions_path)
        os.replace(temp_video, annotated_path)
    except Exception:
        text_handle.close()
        if writer is not None:
            writer.release()
        temp_predictions.unlink(missing_ok=True)
        temp_video.unlink(missing_ok=True)
        raise
    return predictions_path, annotated_path, frame_count, row_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictions, annotated, frames, rows = run_video(
            args.video,
            weights=args.weights,
            tracker=args.tracker,
            output_dir=args.output_dir,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            overwrite=args.overwrite,
        )
    except (InferenceError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Processed {frames} frames and wrote {rows} tracked objects")
    print(f"Predictions: {predictions}")
    print(f"Annotated video: {annotated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
