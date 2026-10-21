#!/usr/bin/env python3
"""Run YOLO + BoT-SORT on a video and save MOT predictions and annotations."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ULTRALYTICS = PROJECT_ROOT / "third-party" / "my-ultralytics"
if str(VENDORED_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "26leaf1.pt"
DEFAULT_TRACKER = VENDORED_ULTRALYTICS / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "video_inference"


class InferenceError(RuntimeError):
    """An error that can be fixed by changing an input or CLI option."""


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def prediction_rows(result: Any, frame_index: int) -> list[str]:
    """Convert one tracked Ultralytics result to VisDrone-MOT text rows."""
    boxes = result.boxes
    if boxes is None or not boxes.is_track or boxes.id is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    track_ids = boxes.id.detach().cpu().numpy()
    rows = []
    for coords, score, class_id, track_id in zip(xyxy, confidence, classes, track_ids):
        x1, y1, x2, y2 = (float(value) for value in coords)
        width, height = x2 - x1, y2 - y1
        values = (x1, y1, width, height, float(score))
        if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in values):
            raise InferenceError(f"Invalid bounding box returned at frame {frame_index}")
        # VisDrone uses one-based class IDs while YOLO uses zero-based IDs.
        rows.append(
            f"{frame_index},{int(round(float(track_id)))},{x1:.3f},{y1:.3f},{width:.3f},{height:.3f},"
            f"{float(score):.6f},{int(round(float(class_id))) + 1},-1,-1\n"
        )
    return rows


def video_fps(video: Path) -> float:
    """Read input FPS, falling back to 30 when the container has no valid value."""
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
    """Run tracking and return prediction path, video path, frames and rows."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise InferenceError("Run this script in the 'my-trackers' conda environment") from exc

    video = video.expanduser().resolve()
    weights = weights.expanduser().resolve()
    tracker = tracker.expanduser().resolve()
    for path, label in ((video, "video"), (weights, "weights"), (tracker, "tracker config")):
        if not path.is_file():
            raise InferenceError(f"Missing {label}: {path}")
    if imgsz <= 0 or max_det <= 0:
        raise InferenceError("--imgsz and --max-det must be positive")
    if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
        raise InferenceError("--conf and --iou must be in [0, 1]")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{video.stem}_predictions.txt"
    annotated_path = output_dir / f"{video.stem}_annotated.mp4"
    existing = [path for path in (predictions_path, annotated_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise InferenceError(f"Output already exists: {names}. Pass --overwrite to replace it")

    fps = video_fps(video)
    resolved_device = resolve_device(device)
    model = YOLO(str(weights))
    results = model.track(
        source=str(video),
        tracker=str(tracker),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=resolved_device,
        save=False,
        verbose=False,
    )

    txt_handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{video.stem}.", suffix=".tmp.txt", dir=output_dir, delete=False
    )
    temp_predictions = Path(txt_handle.name)
    video_handle = tempfile.NamedTemporaryFile(
        prefix=f".{video.stem}.", suffix=".tmp.mp4", dir=output_dir, delete=False
    )
    temp_video = Path(video_handle.name)
    video_handle.close()
    writer = None
    frame_count = 0
    row_count = 0
    try:
        for frame_count, result in enumerate(results, start=1):
            annotated = result.plot(color_mode="instance")
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise InferenceError(f"OpenCV cannot create output video: {annotated_path}")
            writer.write(annotated)
            rows = prediction_rows(result, frame_count)
            txt_handle.writelines(rows)
            row_count += len(rows)

        if frame_count == 0:
            raise InferenceError(f"No frames were decoded from: {video}")
        txt_handle.flush()
        os.fsync(txt_handle.fileno())
        txt_handle.close()
        if writer is not None:
            writer.release()
            writer = None
        os.replace(temp_predictions, predictions_path)
        os.replace(temp_video, annotated_path)
    except Exception:
        txt_handle.close()
        if writer is not None:
            writer.release()
        temp_predictions.unlink(missing_ok=True)
        temp_video.unlink(missing_ok=True)
        raise

    return predictions_path, annotated_path, frame_count, row_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO checkpoint")
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER, help="BoT-SORT YAML config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
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
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Processed {frames} frames and wrote {rows} tracked objects")
    print(f"Predictions: {predictions}")
    print(f"Annotated video: {annotated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
