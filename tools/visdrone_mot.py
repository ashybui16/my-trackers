#!/usr/bin/env python3
"""Reproducible VisDrone-MOT detector-cache, BoT-SORT and evaluation CLI.

The project deliberately keeps this tool outside the vendored Ultralytics tree. It
uses the vendored tracker through its public Python classes, while all dataset,
cache, serialization and evaluation policy lives here.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ULTRALYTICS = PROJECT_ROOT / "third-party" / "my-ultralytics"
if str(VENDORED_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "VisDroneMOT"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "26leaf1.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "visdrone_mot" / "26leaf1_botsort"
DEFAULT_TRACKER_CONFIG = VENDORED_ULTRALYTICS / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"

ULTRALYTICS_VERSION = "8.4.157"
SCHEMA_VERSION = 1
NUM_CLASSES = 10
YOLO_CLASSES = tuple(range(NUM_CLASSES))
VISDRONE_CLASSES = tuple(range(1, NUM_CLASSES + 1))
GMC_METHODS = ("sparseOptFlow",)
TRACK_BUFFER_VALUES = (15, 30, 45, 60, 90)
HOTA_THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))
INFERENCE_SETTINGS = {"imgsz": 640, "conf": 0.01, "max_det": 300, "classes": list(YOLO_CLASSES)}


class PipelineError(RuntimeError):
    """A user-actionable pipeline error."""


class FingerprintMismatch(PipelineError):
    """Raised when an existing cache was made for different inputs/settings."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def ultralytics_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(VENDORED_ULTRALYTICS), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except OSError:
        return None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def yolo_to_visdrone_class(class_id: int) -> int:
    class_id = int(class_id)
    if class_id not in YOLO_CLASSES:
        raise ValueError(f"YOLO class must be in 0..9, got {class_id}")
    return class_id + 1


def visdrone_to_yolo_class(class_id: int) -> int:
    class_id = int(class_id)
    if class_id not in VISDRONE_CLASSES:
        raise ValueError(f"VisDrone class must be in 1..10, got {class_id}")
    return class_id - 1


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        yaml = None
    try:
        text = path.read_text()
        if yaml is not None:
            value = yaml.safe_load(text)
        else:
            # The vendored BoT-SORT config is a flat mapping. Keep the CLI's
            # config/fingerprint tests usable in a minimal environment while
            # PyYAML remains the normal parser in the pinned env.
            value = {}
            for raw_line in text.splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, raw_value = [part.strip() for part in line.split(":", 1)]
                lowered = raw_value.lower()
                if lowered in {"true", "false"}:
                    parsed: Any = lowered == "true"
                else:
                    try:
                        parsed = json.loads(raw_value)
                    except json.JSONDecodeError:
                        parsed = raw_value.strip("'\"")
                value[key] = parsed
    except OSError as exc:
        raise PipelineError(f"Cannot read tracker config: {path}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Tracker config must be a YAML mapping: {path}")
    return dict(value)


def _dump_yaml(value: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        return yaml.safe_dump(dict(value), sort_keys=False)
    lines = []
    for key, item in value.items():
        if isinstance(item, bool):
            rendered = "true" if item else "false"
        else:
            rendered = str(item)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def load_tracker_config(path: Path = DEFAULT_TRACKER_CONFIG) -> dict[str, Any]:
    config = _load_yaml(path)
    required = (
        "tracker_type",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
        "gmc_method",
        "proximity_thresh",
        "appearance_thresh",
        "with_reid",
        "model",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise PipelineError(f"Tracker config is missing keys: {', '.join(missing)}")
    return config


def validate_tracker_config(config: Mapping[str, Any], *, tuned: bool = False) -> dict[str, Any]:
    value = dict(config)
    if value.get("tracker_type") != "botsort":
        raise PipelineError("Only tracker_type=botsort is supported by this pipeline")
    try:
        high = float(value["track_high_thresh"])
        low = float(value["track_low_thresh"])
        new = float(value["new_track_thresh"])
        buffer = int(value["track_buffer"])
        match = float(value["match_thresh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError("BoT-SORT threshold and buffer values must be numeric") from exc
    if not 0.15 <= high <= 0.60:
        raise PipelineError(f"track_high_thresh must be in [0.15, 0.60], got {high}")
    if low < 0.01 or low >= high:
        raise PipelineError(f"track_low_thresh must be >= 0.01 and < high ({high}), got {low}")
    if new < high or new > 0.80:
        raise PipelineError(f"new_track_thresh must be in [high, 0.80], got {new}")
    if buffer not in TRACK_BUFFER_VALUES:
        raise PipelineError(f"track_buffer must be one of {TRACK_BUFFER_VALUES}, got {buffer}")
    if not 0.60 <= match <= 0.95:
        raise PipelineError(f"match_thresh must be in [0.60, 0.95], got {match}")
    if bool(value.get("with_reid")):
        raise PipelineError("The tuning plan requires with_reid=False")
    if value.get("gmc_method") not in GMC_METHODS:
        raise PipelineError(f"gmc_method must be one of {GMC_METHODS}")
    if not isinstance(value.get("fuse_score"), (bool, np.bool_)):
        raise PipelineError("fuse_score must be boolean")
    if tuned:
        value.update(
            {
                "track_high_thresh": high,
                "track_low_thresh": low,
                "new_track_thresh": new,
                "track_buffer": buffer,
                "match_thresh": match,
                "fuse_score": bool(value["fuse_score"]),
                "with_reid": False,
            }
        )
    return value


def tracker_config_fingerprint(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(config)).encode("utf-8"))


def write_tracker_config(path: Path, config: Mapping[str, Any]) -> None:
    atomic_write_text(path, _dump_yaml(config))


def sequence_frames(sequence_dir: Path) -> list[Path]:
    if not sequence_dir.is_dir():
        raise PipelineError(f"Sequence directory does not exist: {sequence_dir}")
    frames = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    frames.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not frames:
        raise PipelineError(f"No image frames found in {sequence_dir}")
    indices = [int(p.stem) for p in frames if p.stem.isdigit()]
    if len(indices) != len(frames) or indices != list(range(1, len(frames) + 1)):
        raise PipelineError(
            f"Frames must be numbered continuously from 1 in {sequence_dir}; "
            f"found {frames[0].name}..{frames[-1].name}"
        )
    return frames


def split_sequences(data_root: Path, split: str) -> dict[str, list[Path]]:
    sequences_root = data_root / split / "sequences"
    if not sequences_root.is_dir():
        raise PipelineError(f"Missing split directory: {sequences_root}")
    result = {p.name: sequence_frames(p) for p in sorted(sequences_root.iterdir()) if p.is_dir()}
    if not result:
        raise PipelineError(f"No sequences found in {sequences_root}")
    return result


def frame_manifest(frames: Sequence[Path]) -> dict[str, Any]:
    entries = []
    digest = hashlib.sha256()
    for frame in frames:
        stat = frame.stat()
        entry = {"name": frame.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        entries.append(entry)
        digest.update(canonical_json(entry).encode("utf-8"))
        digest.update(b"\n")
    return {"count": len(entries), "sha256": digest.hexdigest(), "frames": entries}


def split_manifest(sequences: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    return {name: frame_manifest(frames) for name, frames in sorted(sequences.items())}


def build_cache_fingerprint(
    *,
    split: str,
    sequences: Mapping[str, Sequence[Path]],
    weights_sha256: str,
    inference: Mapping[str, Any],
    frame_limit: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": weights_sha256,
        "ultralytics_version": ULTRALYTICS_VERSION,
        "ultralytics_commit": ultralytics_commit(),
        "split": split,
        "frame_limit": frame_limit,
        "inference": dict(inference),
        "manifest": split_manifest(sequences),
    }
    payload["fingerprint"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "metadata.json", cache_dir / "detections"


@dataclass
class DetectionCache:
    sequence: str
    frame_indices: np.ndarray
    offsets: np.ndarray
    xyxy: np.ndarray
    confidence: np.ndarray
    classes: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_frames(self) -> int:
        return int(len(self.frame_indices))

    def detections_for(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        return self.xyxy[start:end], self.confidence[start:end], self.classes[start:end]


def save_detection_cache(cache_dir: Path, cache: DetectionCache) -> None:
    _, detections_dir = cache_paths(cache_dir)
    detections_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(
        detections_dir / f"{cache.sequence}.npz",
        {
            "frame_indices": np.asarray(cache.frame_indices, dtype=np.int64),
            "offsets": np.asarray(cache.offsets, dtype=np.int64),
            "xyxy": np.asarray(cache.xyxy, dtype=np.float32),
            "confidence": np.asarray(cache.confidence, dtype=np.float32),
            "classes": np.asarray(cache.classes, dtype=np.int64),
        },
    )


def load_detection_cache(cache_dir: Path, sequence: str) -> DetectionCache:
    metadata_path, detections_dir = cache_paths(cache_dir)
    if not metadata_path.is_file():
        raise PipelineError(f"Cache metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read cache metadata: {metadata_path}") from exc
    path = detections_dir / f"{sequence}.npz"
    if not path.is_file():
        raise PipelineError(f"Detection cache for {sequence} is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as values:
            frame_indices = np.asarray(values["frame_indices"], dtype=np.int64)
            offsets = np.asarray(values["offsets"], dtype=np.int64)
            xyxy = np.asarray(values["xyxy"], dtype=np.float32)
            confidence = np.asarray(values["confidence"], dtype=np.float32)
            classes = np.asarray(values["classes"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise PipelineError(f"Invalid detection cache file: {path}") from exc
    if len(offsets) != len(frame_indices) + 1 or offsets[0] != 0 or offsets[-1] != len(xyxy):
        raise PipelineError(f"Invalid cache offsets for {sequence}")
    if frame_indices.tolist() != list(range(1, len(frame_indices) + 1)):
        raise PipelineError(f"Cache frame indices are not continuous for {sequence}")
    if np.any(np.diff(offsets) < 0):
        raise PipelineError(f"Cache offsets are not monotonic for {sequence}")
    if xyxy.ndim != 2 or xyxy.shape[1:] != (4,) or len(confidence) != len(xyxy) or len(classes) != len(xyxy):
        raise PipelineError(f"Invalid cache array shapes for {sequence}")
    if not np.all(np.isfinite(xyxy)) or not np.all(np.isfinite(confidence)):
        raise PipelineError(f"Non-finite detector output in cache for {sequence}")
    if np.any((classes < 0) | (classes >= NUM_CLASSES)):
        raise PipelineError(f"Invalid YOLO class in cache for {sequence}")
    return DetectionCache(sequence, frame_indices, offsets, xyxy, confidence, classes, metadata)


def load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    path, _ = cache_paths(cache_dir)
    if not path.is_file():
        raise PipelineError(f"Cache metadata is missing: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read cache metadata: {path}") from exc


def validate_cache_fingerprint(cache_dir: Path, expected: Mapping[str, Any]) -> None:
    actual = load_cache_metadata(cache_dir)
    if actual.get("fingerprint") != expected.get("fingerprint"):
        raise FingerprintMismatch(
            f"Cache fingerprint mismatch for {cache_dir}: expected {expected.get('fingerprint')}, "
            f"found {actual.get('fingerprint')}. Use --overwrite to rebuild the cache."
        )
    for sequence in expected.get("manifest", {}):
        load_detection_cache(cache_dir, sequence)


def _model_names_are_ten_classes(names: Any) -> bool:
    if isinstance(names, Mapping):
        try:
            keys = {int(key) for key in names}
        except (TypeError, ValueError):
            return False
        return keys == set(YOLO_CLASSES)
    return isinstance(names, (list, tuple)) and len(names) == NUM_CLASSES


def preflight(
    weights: Path,
    *,
    device: str,
    require_optuna: bool = False,
    require_trackeval: bool = False,
) -> tuple[Any, str]:
    """Load the checkpoint and validate import path, versions and class mapping."""
    if not weights.is_file():
        raise PipelineError(f"Checkpoint does not exist: {weights}")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise PipelineError("PyTorch and vendored Ultralytics are required; activate my-trackers") from exc
    if getattr(torch, "__version__", None) != "2.4.1+cu118":
        raise PipelineError(f"Expected Torch 2.4.1+cu118, got {getattr(torch, '__version__', None)}")
    import_path = Path(ultralytics.__file__).resolve()
    expected_prefix = (VENDORED_ULTRALYTICS / "ultralytics").resolve()
    if expected_prefix not in import_path.parents:
        raise PipelineError(f"Ultralytics import path is not vendored: {import_path}")
    if getattr(ultralytics, "__version__", None) != ULTRALYTICS_VERSION:
        raise PipelineError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, got {getattr(ultralytics, '__version__', None)}")
    dependency_errors = []
    try:
        import numpy

        if numpy.__version__ != "1.26.4":
            dependency_errors.append(f"numpy==1.26.4 required, found {numpy.__version__}")
    except ImportError:
        dependency_errors.append("numpy is missing")
    try:
        import cv2

        if not cv2.__version__.startswith("4.11.0"):
            dependency_errors.append(f"opencv-python==4.11.0.86 required, found {cv2.__version__}")
    except ImportError:
        dependency_errors.append("opencv-python is missing")
    for module_name in ("scipy", "lap"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            dependency_errors.append(f"{module_name} is missing")
    try:
        import torchvision

        if getattr(torchvision, "__version__", None) != "0.19.1+cu118":
            dependency_errors.append(
                f"torchvision==0.19.1+cu118 required, found {getattr(torchvision, '__version__', None)}"
            )
    except ImportError:
        dependency_errors.append("torchvision is missing")
    if require_optuna:
        try:
            importlib.import_module("optuna")
        except ImportError:
            dependency_errors.append("optuna>=4,<5 is missing")
    if require_trackeval:
        try:
            importlib.import_module("trackeval")
        except ImportError:
            dependency_errors.append("trackeval is missing")
    if dependency_errors:
        raise PipelineError("Dependency preflight failed: " + "; ".join(dependency_errors))
    resolved_device = resolve_device(device)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise PipelineError(f"Requested {resolved_device}, but CUDA is not available")
    model = YOLO(str(weights))
    if not _model_names_are_ten_classes(getattr(model, "names", None)):
        raise PipelineError(f"Checkpoint class mapping must contain exactly 0..9; got {getattr(model, 'names', None)}")
    return model, resolved_device


def _result_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    values = [boxes.xyxy, boxes.conf, boxes.cls]
    values = [value.detach().cpu().numpy() if hasattr(value, "detach") else value for value in values]
    xyxy = np.asarray(values[0], dtype=np.float32).reshape(-1, 4)
    confidence = np.asarray(values[1], dtype=np.float32).reshape(-1)
    classes = np.asarray(values[2], dtype=np.int64).reshape(-1)
    if len(xyxy) != len(confidence) or len(xyxy) != len(classes):
        raise PipelineError("Ultralytics returned inconsistent detection array lengths")
    if np.any((classes < 0) | (classes >= NUM_CLASSES)):
        raise PipelineError("Checkpoint produced a class outside YOLO classes 0..9")
    return xyxy, confidence, classes


def detect_sequence(
    model: Any,
    frames: Sequence[Path],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        # A plain list of paths is treated by Ultralytics as an in-memory image
        # source: every file is decoded up front and the whole sequence becomes
        # one batch.  Use the file-backed loader so long sequences are actually
        # inferred one frame at a time.  ``stream=True`` only streams returned
        # results and does not change the input batch size.
        from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

        source = LoadImagesAndVideos([str(path) for path in frames], batch=1)
        source.source_type = SourceTypes()
        results = model.predict(
            source=source,
            stream=True,
            batch=1,
            imgsz=INFERENCE_SETTINGS["imgsz"],
            conf=INFERENCE_SETTINGS["conf"],
            max_det=INFERENCE_SETTINGS["max_det"],
            classes=INFERENCE_SETTINGS["classes"],
            device=device,
            save=False,
            verbose=False,
        )
        xyxy_parts, confidence_parts, class_parts = [], [], []
        offsets = [0]
        for result in results:
            xyxy, confidence, classes = _result_arrays(result)
            xyxy_parts.append(xyxy)
            confidence_parts.append(confidence)
            class_parts.append(classes)
            offsets.append(offsets[-1] + len(xyxy))
    except Exception as exc:
        raise PipelineError(f"Detector inference failed: {exc}") from exc
    if len(offsets) != len(frames) + 1:
        raise PipelineError(f"Detector returned {len(offsets) - 1} frames for {len(frames)} input frames")
    return (
        np.arange(1, len(frames) + 1, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(xyxy_parts) if xyxy_parts else np.empty((0, 4), dtype=np.float32),
        np.concatenate(confidence_parts) if confidence_parts else np.empty(0, dtype=np.float32),
        np.concatenate(class_parts) if class_parts else np.empty(0, dtype=np.int64),
    )


def cache_split(
    *,
    data_root: Path,
    split: str,
    weights: Path,
    cache_dir: Path,
    device: str,
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    sequences = split_sequences(data_root, split)
    if frame_limit is not None:
        if frame_limit <= 0:
            raise PipelineError("--max-frames must be positive")
        sequences = {name: list(frames[:frame_limit]) for name, frames in sequences.items()}
    fingerprint = build_cache_fingerprint(
        split=split,
        sequences=sequences,
        weights_sha256=sha256_file(weights),
        inference=INFERENCE_SETTINGS,
        frame_limit=frame_limit,
    )
    metadata_path, _ = cache_paths(cache_dir)
    if metadata_path.exists() and not overwrite:
        validate_cache_fingerprint(cache_dir, fingerprint)
        return load_cache_metadata(cache_dir)
    if cache_dir.exists() and not overwrite:
        raise PipelineError(f"Cache already exists; pass --overwrite to replace it: {cache_dir}")
    model, resolved_device = preflight(weights, device=device)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=cache_dir.parent))
    try:
        for sequence, frames in sequences.items():
            print(f"[cache] {split}/{sequence}: {len(frames)} frames", flush=True)
            frame_indices, offsets, xyxy, confidence, classes = detect_sequence(model, frames, device=resolved_device)
            save_detection_cache(
                temp_dir,
                DetectionCache(sequence, frame_indices, offsets, xyxy, confidence, classes, fingerprint),
            )
        metadata = {**fingerprint, "cache_dir": str(cache_dir), "device": resolved_device, "created_unix": time.time()}
        atomic_write_json(temp_dir / "metadata.json", metadata)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(temp_dir, cache_dir)
        return metadata
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def expected_cache_fingerprint(
    *, data_root: Path, split: str, weights: Path, frame_limit: int | None = None
) -> dict[str, Any]:
    sequences = split_sequences(data_root, split)
    if frame_limit is not None:
        sequences = {name: list(frames[:frame_limit]) for name, frames in sequences.items()}
    return build_cache_fingerprint(
        split=split,
        sequences=sequences,
        weights_sha256=sha256_file(weights),
        inference=INFERENCE_SETTINGS,
        frame_limit=frame_limit,
    )


class DetectionBatch:
    """Minimal Results-like object consumed by Ultralytics' BOTSORT class."""

    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray, classes: np.ndarray):
        self._xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        if len(self._xyxy) != len(self.conf) or len(self._xyxy) != len(self.cls):
            raise ValueError("Detection arrays must have equal lengths")

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        value = self._xyxy.copy()
        value[:, 2] = self._xyxy[:, 2] - self._xyxy[:, 0]
        value[:, 3] = self._xyxy[:, 3] - self._xyxy[:, 1]
        value[:, 0] = (self._xyxy[:, 0] + self._xyxy[:, 2]) / 2
        value[:, 1] = (self._xyxy[:, 1] + self._xyxy[:, 3]) / 2
        return value

    def __len__(self) -> int:
        return len(self._xyxy)

    def __getitem__(self, index: Any) -> "DetectionBatch":
        return DetectionBatch(self._xyxy[index], self.conf[index], self.cls[index])


def tracker_args(config: Mapping[str, Any], device: str = "cpu") -> SimpleNamespace:
    value = validate_tracker_config(config, tuned=True)
    value["device"] = device
    return SimpleNamespace(**value)


def reset_tracker_for_sequence(tracker: Any) -> Any:
    """Reset tracker state at an explicit sequence boundary and return it."""
    tracker.reset()
    return tracker


def _prediction_row_from_track(track: Sequence[float], frame: int) -> tuple[Any, ...]:
    if len(track) < 8:
        raise PipelineError(f"BoT-SORT returned an invalid track row with {len(track)} values")
    x1, y1, x2, y2, track_id, score, cls, _ = [float(v) for v in track[:8]]
    width, height = x2 - x1, y2 - y1
    if not all(math.isfinite(v) for v in (x1, y1, width, height, track_id, score, cls)):
        raise PipelineError("BoT-SORT returned a non-finite prediction")
    if width <= 0 or height <= 0:
        raise PipelineError("BoT-SORT returned a non-positive bounding box")
    if int(round(cls)) not in YOLO_CLASSES:
        raise PipelineError(f"BoT-SORT returned an invalid YOLO class: {cls}")
    return (
        int(frame), int(round(track_id)), x1, y1, width, height, score,
        yolo_to_visdrone_class(int(round(cls))), -1, -1,
    )


def validate_prediction_rows(rows: Sequence[Sequence[Any]], num_frames: int) -> None:
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if len(row) != 10:
            raise PipelineError(f"VisDrone prediction rows must have 10 fields, got {len(row)}")
        frame, track_id = int(row[0]), int(row[1])
        if not 1 <= frame <= num_frames:
            raise PipelineError(f"Prediction frame {frame} is outside 1..{num_frames}")
        if track_id <= 0:
            raise PipelineError(f"Prediction track ID must be positive, got {track_id}")
        if (frame, track_id) in seen:
            raise PipelineError(f"Duplicate track ID {track_id} in frame {frame}")
        seen.add((frame, track_id))
        x, y, width, height, confidence = [float(row[i]) for i in (2, 3, 4, 5, 6)]
        if not all(math.isfinite(v) for v in (x, y, width, height, confidence)):
            raise PipelineError("Prediction contains a non-finite value")
        if width <= 0 or height <= 0:
            raise PipelineError("Prediction boxes must have positive width and height")
        if not 0.0 <= confidence <= 1.0:
            raise PipelineError(f"Prediction confidence must be in [0, 1], got {confidence}")
        if int(row[7]) not in VISDRONE_CLASSES:
            raise PipelineError(f"Prediction class must be in 1..10, got {row[7]}")


def serialize_visdrone_rows(rows: Sequence[Sequence[Any]], num_frames: int) -> str:
    validate_prediction_rows(rows, num_frames)
    lines = []
    for row in rows:
        frame, track_id = int(row[0]), int(row[1])
        x, y, width, height, confidence = [float(row[i]) for i in (2, 3, 4, 5, 6)]
        lines.append(
            f"{frame},{track_id},{x:.3f},{y:.3f},{width:.3f},{height:.3f},{confidence:.6f},{int(row[7])},-1,-1"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def track_sequence(
    *,
    cache: DetectionCache,
    sequence_dir: Path,
    config: Mapping[str, Any],
    output_path: Path,
    device: str = "cpu",
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> list[tuple[Any, ...]]:
    try:
        import cv2
        from ultralytics.trackers.bot_sort import BOTSORT
    except ImportError as exc:
        raise PipelineError("opencv-python, lap and vendored Ultralytics are required for tracking") from exc
    if output_path.exists() and not overwrite:
        raise PipelineError(f"Prediction already exists; pass --overwrite to replace it: {output_path}")
    frames = sequence_frames(sequence_dir)
    if frame_limit is not None:
        frames = frames[:frame_limit]
    if len(frames) != cache.num_frames:
        raise FingerprintMismatch(f"Cache frame count for {cache.sequence} and data differ")
    tracker = BOTSORT(args=tracker_args(config, device=device))
    reset_tracker_for_sequence(tracker)  # explicit boundary; also resets the global ID allocator
    rows: list[tuple[Any, ...]] = []
    for position, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError(f"OpenCV could not read frame: {frame_path}")
        xyxy, confidence, classes = cache.detections_for(position)
        for track in tracker.update(DetectionBatch(xyxy, confidence, classes), image):
            rows.append(_prediction_row_from_track(track, position + 1))
    validate_prediction_rows(rows, len(frames))
    atomic_write_text(output_path, serialize_visdrone_rows(rows, len(frames)))
    return rows


def track_split(
    *,
    data_root: Path,
    split: str,
    cache_dir: Path,
    prediction_dir: Path,
    config_path: Path | None = None,
    config: Mapping[str, Any] | None = None,
    device: str = "cpu",
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> dict[str, Path]:
    if config is not None and config_path is not None:
        raise PipelineError("Pass either config or config_path, not both")
    config = validate_tracker_config(config or load_tracker_config(config_path or DEFAULT_TRACKER_CONFIG), tuned=True)
    metadata = load_cache_metadata(cache_dir)
    if metadata.get("split") != split:
        raise FingerprintMismatch(f"Cache split is {metadata.get('split')!r}, requested {split!r}")
    sequences = split_sequences(data_root, split)
    if metadata.get("frame_limit") != frame_limit:
        raise FingerprintMismatch(
            f"Cache frame_limit is {metadata.get('frame_limit')!r}, requested {frame_limit!r}"
        )
    manifest_sequences = {
        name: (frames[:frame_limit] if frame_limit is not None else frames)
        for name, frames in sequences.items()
    }
    if metadata.get("manifest") != split_manifest(manifest_sequences):
        raise FingerprintMismatch("Cache frame manifest does not match the current dataset")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sequence, frames in sequences.items():
        cache = load_detection_cache(cache_dir, sequence)
        selected_frames = frames[:frame_limit] if frame_limit else frames
        if len(selected_frames) != cache.num_frames:
            raise FingerprintMismatch(f"Cache and data frame manifests differ for {sequence}")
        path = prediction_dir / f"{sequence}.txt"
        print(f"[track] {split}/{sequence}: {len(selected_frames)} frames", flush=True)
        track_sequence(
            cache=cache,
            sequence_dir=data_root / split / "sequences" / sequence,
            config=config,
            output_path=path,
            device=device,
            overwrite=overwrite,
            frame_limit=frame_limit,
        )
        outputs[sequence] = path
    return outputs


# ---------------------------------------------------------------------------
# VisDrone parsing and a deterministic evaluator fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MOTObject:
    frame: int
    track_id: int
    bbox: tuple[float, float, float, float]
    cls: int
    confidence: float = 1.0
    ignore: bool = False


def _parse_csv_row(line: str, path: Path, line_number: int) -> list[str]:
    values = [part.strip() for part in line.split(",")]
    if len(values) != 10:
        raise PipelineError(f"{path}:{line_number}: expected 10 comma-separated fields, got {len(values)}")
    return values


def read_ground_truth(path: Path) -> list[MOTObject]:
    if not path.is_file():
        raise PipelineError(f"Ground-truth annotation is missing: {path}")
    result = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise PipelineError(f"Cannot read annotation: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        values = _parse_csv_row(line, path, line_number)
        try:
            frame, track_id = int(values[0]), int(values[1])
            x, y, width, height = (float(values[i]) for i in (2, 3, 4, 5))
            mark, cls = int(values[6]), int(values[7])
        except ValueError as exc:
            raise PipelineError(f"{path}:{line_number}: invalid numeric value") from exc
        if frame < 1 or track_id < 0 or width <= 0 or height <= 0:
            raise PipelineError(f"{path}:{line_number}: invalid frame, id or box")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise PipelineError(f"{path}:{line_number}: non-finite box")
        result.append(MOTObject(frame, track_id, (x, y, width, height), cls, 1.0, mark != 1 or cls in (0, 11)))
    return result


def read_prediction_file(path: Path, num_frames: int) -> list[MOTObject]:
    if not path.is_file():
        raise PipelineError(f"Prediction file is missing: {path}")
    result = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise PipelineError(f"Cannot read prediction file: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        values = _parse_csv_row(line, path, line_number)
        try:
            frame, track_id = int(values[0]), int(values[1])
            x, y, width, height, confidence = (float(values[i]) for i in (2, 3, 4, 5, 6))
            cls = int(values[7])
        except ValueError as exc:
            raise PipelineError(f"{path}:{line_number}: invalid numeric value") from exc
        row = (frame, track_id, x, y, width, height, confidence, cls, -1, -1)
        validate_prediction_rows([row], num_frames)
        result.append(MOTObject(frame, track_id, (x, y, width, height), cls, confidence, False))
    # Validate duplicate IDs across the complete file, not only one row at a time.
    validate_prediction_rows(
        [
            (item.frame, item.track_id, *item.bbox, item.confidence, item.cls, -1, -1)
            for item in result
        ],
        num_frames,
    )
    return result


def _xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def box_ioa(box: Sequence[float], region: Sequence[float]) -> float:
    """Intersection over the candidate box, used for VisDrone ignore regions."""
    bx1, by1, bx2, by2 = _xyxy(box)
    rx1, ry1, rx2, ry2 = _xyxy(region)
    intersection = max(0.0, min(bx2, rx2) - max(bx1, rx1)) * max(0.0, min(by2, ry2) - max(by1, ry1))
    area = box[2] * box[3]
    return intersection / area if area > 0 else 0.0


def _linear_assignment(similarity: np.ndarray) -> list[tuple[int, int]]:
    if similarity.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(1.0 - similarity)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        # The pinned environment has scipy. This small fallback keeps fixture tests
        # usable in a minimal Python environment and is exact enough for its tiny
        # matrices.
        candidates = sorted(
            ((float(similarity[row, col]), row, col) for row in range(similarity.shape[0]) for col in range(similarity.shape[1])),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        result = []
        for _, row, col in candidates:
            if row not in used_rows and col not in used_cols:
                result.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        return result


def _group_by_frame(objects: Sequence[MOTObject]) -> dict[int, list[MOTObject]]:
    groups: dict[int, list[MOTObject]] = defaultdict(list)
    for item in objects:
        groups[item.frame].append(item)
    return groups


def _match_frame(
    ground_truth: Sequence[MOTObject], predictions: Sequence[MOTObject], threshold: float
) -> tuple[list[tuple[MOTObject, MOTObject, float]], list[MOTObject], list[MOTObject]]:
    valid_gt = [item for item in ground_truth if not item.ignore and item.cls in VISDRONE_CLASSES]
    candidates = [item for item in predictions if item.cls in VISDRONE_CLASSES]
    if not valid_gt or not candidates:
        return [], valid_gt, candidates
    similarity = np.asarray(
        [[box_iou(gt.bbox, pred.bbox) if gt.cls == pred.cls else 0.0 for pred in candidates] for gt in valid_gt],
        dtype=np.float64,
    )
    matches = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for gt_index, pred_index in _linear_assignment(similarity):
        value = float(similarity[gt_index, pred_index])
        if value >= threshold:
            matches.append((valid_gt[gt_index], candidates[pred_index], value))
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    unmatched_gt = [item for index, item in enumerate(valid_gt) if index not in matched_gt]
    unmatched_pred = [item for index, item in enumerate(candidates) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def _is_ignored_prediction(prediction: MOTObject, ignored_regions: Sequence[MOTObject]) -> bool:
    return any(box_ioa(prediction.bbox, region.bbox) >= 0.5 for region in ignored_regions)


def _filter_objects(objects: Sequence[MOTObject], cls: int | None) -> list[MOTObject]:
    if cls is None:
        return list(objects)
    return [item for item in objects if item.cls == cls or item.ignore]


def _threshold_stats(
    ground_truth: Sequence[MOTObject], predictions: Sequence[MOTObject], threshold: float
) -> dict[str, Any]:
    gt_by_frame = _group_by_frame(ground_truth)
    pred_by_frame = _group_by_frame(predictions)
    frame_ids = sorted(set(gt_by_frame) | set(pred_by_frame))
    matches: list[tuple[MOTObject, MOTObject, float]] = []
    effective_predictions: list[MOTObject] = []
    gt_match_by_id: dict[tuple[int, int], list[tuple[int, int | None]]] = defaultdict(list)
    for frame in frame_ids:
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        frame_matches, unmatched_gt, unmatched_pred = _match_frame(gt_frame, pred_frame, threshold)
        matches.extend(frame_matches)
        ignored = [item for item in gt_frame if item.ignore]
        matched_pred_ids = {id(prediction) for _, prediction, _ in frame_matches}
        for prediction in pred_frame:
            if id(prediction) in matched_pred_ids or not _is_ignored_prediction(prediction, ignored):
                effective_predictions.append(prediction)
        for item in gt_frame:
            if item.ignore or item.cls not in VISDRONE_CLASSES:
                continue
            matched_prediction = next((prediction for gt, prediction, _ in frame_matches if gt is item), None)
            gt_match_by_id[(item.cls, item.track_id)].append(
                (frame, matched_prediction.track_id if matched_prediction is not None else None)
            )
        # ``unmatched_gt`` is intentionally not used directly: the complete GT
        # timeline above is needed for fragmentation and identity switches.
        _ = unmatched_gt, unmatched_pred

    gt_counts: dict[tuple[int, int], int] = defaultdict(int)
    pred_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = defaultdict(int)
    for item in ground_truth:
        if not item.ignore and item.cls in VISDRONE_CLASSES:
            gt_counts[(item.cls, item.track_id)] += 1
    for item in effective_predictions:
        pred_counts[(item.cls, item.track_id)] += 1
    for gt, prediction, _ in matches:
        pair_counts[((gt.cls, gt.track_id), (prediction.cls, prediction.track_id))] += 1
    association_sum = 0.0
    for gt, prediction, _ in matches:
        gt_key = (gt.cls, gt.track_id)
        pred_key = (prediction.cls, prediction.track_id)
        pair_count = pair_counts[(gt_key, pred_key)]
        denominator = gt_counts[gt_key] + pred_counts[pred_key] - pair_count
        association_sum += pair_count / denominator if denominator > 0 else 0.0
    loc_sum = float(sum(iou for _, _, iou in matches))
    tp = len(matches)
    gt_detections = sum(gt_counts.values())
    pred_detections = sum(pred_counts.values())
    fp = pred_detections - tp
    fn = gt_detections - tp
    id_switches = 0
    fragmentation = 0
    for timeline in gt_match_by_id.values():
        previous_frame = None
        previous_prediction = None
        had_match = False
        in_gap = False
        for frame, prediction_id in timeline:
            if prediction_id is None:
                if had_match:
                    in_gap = True
                continue
            if previous_prediction is not None and prediction_id != previous_prediction:
                if previous_frame is None or frame == previous_frame + 1:
                    id_switches += 1
            if had_match and in_gap:
                fragmentation += 1
            had_match = True
            in_gap = False
            previous_frame, previous_prediction = frame, prediction_id
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": gt_detections,
        "pred": pred_detections,
        "association_sum": association_sum,
        "loc_sum": loc_sum,
        "matches": matches,
        "pair_counts": dict(pair_counts),
        "gt_counts": dict(gt_counts),
        "pred_counts": dict(pred_counts),
        "idsw": id_switches,
        "fragmentation": fragmentation,
    }


def _max_identity_matches(pair_counts: Mapping[tuple[Any, Any], int]) -> int:
    if not pair_counts:
        return 0
    gt_keys = sorted({key[0] for key in pair_counts})
    pred_keys = sorted({key[1] for key in pair_counts})
    matrix = np.asarray(
        [[pair_counts.get((gt_key, pred_key), 0) for pred_key in pred_keys] for gt_key in gt_keys],
        dtype=np.float64,
    )
    return int(sum(matrix[row, col] for row, col in _linear_assignment(matrix)))


def evaluate_pair_stats(ground_truth: Sequence[MOTObject], predictions: Sequence[MOTObject]) -> dict[str, Any]:
    hota = {}
    for threshold in HOTA_THRESHOLDS:
        hota[f"{threshold:.2f}"] = _threshold_stats(ground_truth, predictions, threshold)
    clear = hota["0.50"]
    idtp = _max_identity_matches(clear["pair_counts"])
    identity = {
        "gt": clear["gt"],
        "pred": clear["pred"],
        "idtp": idtp,
        "idfn": clear["gt"] - idtp,
        "idfp": clear["pred"] - idtp,
    }
    count = {
        "gt_detections": clear["gt"],
        "pred_detections": clear["pred"],
        "gt_ids": len(clear["gt_counts"]),
        "pred_ids": len(clear["pred_counts"]),
    }
    return {"hota": hota, "clear": clear, "identity": identity, "count": count}


def combine_stats(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    if left is None:
        return copy.deepcopy(right)
    for threshold in left["hota"]:
        for key in ("tp", "fp", "fn", "gt", "pred", "association_sum", "loc_sum", "idsw", "fragmentation"):
            left["hota"][threshold][key] += right["hota"][threshold][key]
    for key in ("tp", "fp", "fn", "gt", "pred", "association_sum", "loc_sum", "idsw", "fragmentation"):
        left["clear"][key] += right["clear"][key]
    for key in ("gt", "pred", "idtp", "idfn", "idfp"):
        left["identity"][key] += right["identity"][key]
    for key in ("gt_detections", "pred_detections", "gt_ids", "pred_ids"):
        left["count"][key] += right["count"][key]
    return left


def metrics_from_stats(stats: Mapping[str, Any]) -> dict[str, float | int]:
    hota_values, deta_values, assa_values, loca_values = [], [], [], []
    for value in stats["hota"].values():
        denominator = value["tp"] + value["fp"] + value["fn"]
        deta = value["tp"] / denominator if denominator else 0.0
        assa = value["association_sum"] / value["tp"] if value["tp"] else 0.0
        hota_values.append(math.sqrt(deta * assa))
        deta_values.append(deta)
        assa_values.append(assa)
        loca_values.append(value["loc_sum"] / value["tp"] if value["tp"] else 0.0)
    hota = stats["hota"]["0.50"]
    clear = stats["clear"]
    identity = stats["identity"]
    gt = int(clear["gt"])
    pred = int(clear["pred"])
    mota = 1.0 - (clear["fp"] + clear["fn"] + clear["idsw"]) / gt if gt else 0.0
    motp = clear["loc_sum"] / clear["tp"] if clear["tp"] else 0.0
    idf1_denominator = 2 * identity["idtp"] + identity["idfn"] + identity["idfp"]
    idf1 = 2 * identity["idtp"] / idf1_denominator if idf1_denominator else 0.0
    idp = identity["idtp"] / (identity["idtp"] + identity["idfp"]) if identity["idtp"] + identity["idfp"] else 0.0
    idr = identity["idtp"] / (identity["idtp"] + identity["idfn"]) if identity["idtp"] + identity["idfn"] else 0.0
    precision = clear["tp"] / (clear["tp"] + clear["fp"]) if clear["tp"] + clear["fp"] else 0.0
    recall = clear["tp"] / (clear["tp"] + clear["fn"]) if clear["tp"] + clear["fn"] else 0.0
    return {
        "HOTA": 100.0 * float(np.mean(hota_values)),
        "DetA": 100.0 * float(np.mean(deta_values)),
        "AssA": 100.0 * float(np.mean(assa_values)),
        "LocA": 100.0 * float(np.mean(loca_values)),
        "MOTA": 100.0 * mota,
        "MOTP": 100.0 * motp,
        "IDF1": 100.0 * idf1,
        "IDP": 100.0 * idp,
        "IDR": 100.0 * idr,
        "Precision": 100.0 * precision,
        "Recall": 100.0 * recall,
        "TP": int(clear["tp"]),
        "FP": int(clear["fp"]),
        "FN": int(clear["fn"]),
        "IDSW": int(clear["idsw"]),
        "Fragmentation": int(clear["fragmentation"]),
        "GT": gt,
        "Pred": pred,
        "NumGTIDs": int(stats["count"]["gt_ids"]),
        "NumPredIDs": int(stats["count"]["pred_ids"]),
    }


def _stats_for_class(objects: Sequence[MOTObject], cls: int) -> list[MOTObject]:
    return [item for item in objects if item.cls == cls or item.ignore]


def evaluate_native(
    *,
    data_root: Path,
    split: str,
    predictions_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    sequences = split_sequences(data_root, split)
    sequence_reports: dict[str, Any] = {}
    sequence_stats: dict[str, dict[str, Any]] = {}
    class_stats: dict[int, dict[str, Any] | None] = {cls: None for cls in VISDRONE_CLASSES}
    overall_stats: dict[str, Any] | None = None
    for sequence, frames in sequences.items():
        gt_path = data_root / split / "annotations" / f"{sequence}.txt"
        gt = read_ground_truth(gt_path)
        pred = read_prediction_file(predictions_dir / f"{sequence}.txt", len(frames))
        stats = evaluate_pair_stats(gt, pred)
        sequence_stats[sequence] = stats
        overall_stats = combine_stats(overall_stats, stats)
        per_class = {}
        for cls in VISDRONE_CLASSES:
            class_result = evaluate_pair_stats(_stats_for_class(gt, cls), _stats_for_class(pred, cls))
            class_stats[cls] = combine_stats(class_stats[cls], class_result)
            per_class[str(cls)] = metrics_from_stats(class_result)
        sequence_reports[sequence] = {
            "num_frames": len(frames),
            "metrics": metrics_from_stats(stats),
            "classes": per_class,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "backend": "native",
        "split": split,
        "predictions_dir": str(predictions_dir),
        "sequences": sequence_reports,
        "classes": {str(cls): metrics_from_stats(class_stats[cls]) for cls in VISDRONE_CLASSES},
        "summary": metrics_from_stats(overall_stats or evaluate_pair_stats([], [])),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / f"{split}_report.json", report)
        _write_metrics_csv(output_dir / f"{split}_metrics.csv", report)
    return report


def _write_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    metric_names = [
        "HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR",
        "Precision", "Recall", "TP", "FP", "FN", "IDSW", "Fragmentation", "GT", "Pred",
        "NumGTIDs", "NumPredIDs",
    ]
    lines = []
    header = ["scope", "name", "class", *metric_names]
    lines.append(header)
    for sequence, value in report.get("sequences", {}).items():
        lines.append(["sequence", sequence, "all", *[value["metrics"].get(name, "") for name in metric_names]])
        for cls, metrics in value.get("classes", {}).items():
            lines.append(["sequence_class", sequence, cls, *[metrics.get(name, "") for name in metric_names]])
    for cls, metrics in report.get("classes", {}).items():
        lines.append(["class", "all", cls, *[metrics.get(name, "") for name in metric_names]])
    lines.append(["summary", "all", "all", *[report["summary"].get(name, "") for name in metric_names]])
    from io import StringIO

    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerows(lines)
    atomic_write_text(path, stream.getvalue())


# ---------------------------------------------------------------------------
# Optuna tuning and the locked test handoff
# ---------------------------------------------------------------------------


def default_tunable_config() -> dict[str, Any]:
    return validate_tracker_config(load_tracker_config(DEFAULT_TRACKER_CONFIG), tuned=True)


def suggest_tracker_params(trial: Any) -> dict[str, Any]:
    """Suggest the BoT-SORT tuning search space.

    ``track_low_thresh`` and ``new_track_thresh`` use the sampled high threshold
    as their bound, so invalid configurations cannot reach the tracker.
    """
    high = float(trial.suggest_float("track_high_thresh", 0.15, 0.60))
    low_upper = max(0.010001, high - 0.000001)
    low = float(trial.suggest_float("track_low_thresh", 0.01, low_upper))
    new = float(trial.suggest_float("new_track_thresh", high, 0.80))
    return {
        "track_high_thresh": high,
        "track_low_thresh": low,
        "new_track_thresh": new,
        "track_buffer": int(trial.suggest_categorical("track_buffer", list(TRACK_BUFFER_VALUES))),
        "match_thresh": float(trial.suggest_float("match_thresh", 0.60, 0.95)),
        "fuse_score": bool(trial.suggest_categorical("fuse_score", [True, False])),
        "gmc_method": str(trial.suggest_categorical("gmc_method", list(GMC_METHODS))),
    }


def config_from_params(params: Mapping[str, Any], base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(base or default_tunable_config())
    config.update(dict(params))
    config["with_reid"] = False
    return validate_tracker_config(config, tuned=True)


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name, float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def is_better_trial(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    for name in ("HOTA", "IDF1", "MOTA"):
        difference = _metric_value(candidate, name) - _metric_value(incumbent, name)
        if abs(difference) > 1e-6:
            return difference > 0
    return int(candidate.get("trial", 0)) < int(incumbent.get("trial", 0))


def _trial_record(trial: Any) -> dict[str, Any]:
    metrics = dict(trial.user_attrs.get("metrics", {}))
    return {
        "trial": int(trial.number),
        "state": str(trial.state),
        "value": trial.value,
        "metrics": metrics,
        "params": dict(trial.params),
        "metrics_path": trial.user_attrs.get("metrics_path"),
    }


def _write_trials_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "trial", "state", "value", "HOTA", "IDF1", "MOTA", "track_high_thresh", "track_low_thresh",
        "new_track_thresh", "track_buffer", "match_thresh", "fuse_score", "gmc_method", "metrics_path",
    ]
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        metrics = record.get("metrics", {})
        params = record.get("params", {})
        writer.writerow(
            {
                "trial": record.get("trial"),
                "state": record.get("state"),
                "value": record.get("value"),
                "HOTA": metrics.get("HOTA"),
                "IDF1": metrics.get("IDF1"),
                "MOTA": metrics.get("MOTA"),
                **{key: params.get(key) for key in fields[6:13]},
                "metrics_path": record.get("metrics_path"),
            }
        )
    atomic_write_text(path, stream.getvalue())


def tune_val(
    *,
    data_root: Path,
    weights: Path,
    output_root: Path,
    device: str,
    n_trials: int = 300,
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    if n_trials != 300:
        # A smaller number is useful for a smoke run, but the default and the
        # recorded production handoff must be exactly the planned 300 trials.
        print(f"[tune] development run: requested {n_trials} trials (production plan is 300)", flush=True)
    try:
        import optuna
    except ImportError as exc:
        raise PipelineError("optuna>=4,<5 is required for tuning") from exc
    cache_dir = output_root / "cache" / "val"
    cache_split(
        data_root=data_root,
        split="val",
        weights=weights,
        cache_dir=cache_dir,
        device=device,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    cache_metadata = load_cache_metadata(cache_dir)
    tuning_dir = output_root / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    # Keep the fixed-GMC search separate from studies created with other GMC choices.
    study_path = tuning_dir / "study_sparse_optflow.sqlite3"
    if overwrite and study_path.exists():
        study_path.unlink()
    storage = f"sqlite:///{study_path.resolve()}"
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="visdrone_botsort_val",
        storage=storage,
        load_if_exists=not overwrite,
        direction="maximize",
        sampler=sampler,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.80,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
            }
        )
    trial_root = tuning_dir / "trials_sparse_optflow"
    trial_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        params = suggest_tracker_params(trial)
        config = config_from_params(params)
        trial_dir = trial_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "botsort.yaml"
        write_tracker_config(config_path, config)
        prediction_dir = trial_dir / "predictions"
        track_split(
            data_root=data_root,
            split="val",
            cache_dir=cache_dir,
            prediction_dir=prediction_dir,
            config_path=config_path,
            device=resolve_device(device),
            overwrite=True,
            frame_limit=frame_limit,
        )
        if importlib.util.find_spec("trackeval") is not None:
            trackeval_dir = trial_dir / ".trackeval"
            try:
                raw_trackeval = run_trackeval_adapter(
                    data_root=data_root,
                    split="val",
                    predictions_dir=prediction_dir,
                    work_dir=trackeval_dir,
                )
                atomic_write_json(trial_dir / "trackeval_raw.json", raw_trackeval)
                metrics = summarize_trackeval_raw(raw_trackeval)
            except PipelineError:
                # A development environment without a working TrackEval install
                # can still exercise Optuna; the production requirements make
                # this fallback unnecessary.
                report = evaluate_native(
                    data_root=data_root, split="val", predictions_dir=prediction_dir, output_dir=trial_dir
                )
                metrics = report["summary"]
            finally:
                if trackeval_dir.exists():
                    shutil.rmtree(trackeval_dir)
        else:
            report = evaluate_native(data_root=data_root, split="val", predictions_dir=prediction_dir, output_dir=trial_dir)
            metrics = report["summary"]
        metrics_path = trial_dir / "metrics.json"
        atomic_write_json(metrics_path, metrics)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("config_fingerprint", tracker_config_fingerprint(config))
        return float(metrics["HOTA"])

    finished_trials = sum(
        trial.state.name in {"COMPLETE", "PRUNED", "FAIL"} for trial in study.trials
    )
    remaining = max(0, n_trials - finished_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    records = [_trial_record(trial) for trial in study.trials]
    if len(records) != n_trials:
        raise PipelineError(f"Expected exactly {n_trials} total trials, found {len(records)}")
    best: dict[str, Any] | None = None
    for record in records:
        if str(record.get("state")) != "TrialState.COMPLETE":
            continue
        candidate = {"trial": record["trial"], **record.get("metrics", {})}
        if is_better_trial(candidate, best):
            best = candidate
    if best is None:
        raise PipelineError("Optuna completed without a successful trial")
    best_record = next(record for record in records if record["trial"] == best["trial"])
    best_config = config_from_params(best_record["params"])
    best_config_path = tuning_dir / "best_botsort.yaml"
    write_tracker_config(best_config_path, best_config)
    best_trial = {
        "schema_version": SCHEMA_VERSION,
        "locked": True,
        "source_split": "val",
        "trial": best["trial"],
        "params": best_record["params"],
        "metrics": best_record["metrics"],
        "config": best_config,
        "config_fingerprint": tracker_config_fingerprint(best_config),
        "cache_fingerprint": cache_metadata.get("fingerprint"),
        "study_path": str(study_path),
        "n_trials": len(records),
        "created_unix": time.time(),
    }
    atomic_write_json(tuning_dir / "best_trial.json", best_trial)
    _write_trials_csv(tuning_dir / "trials.csv", records)
    val_report = {
        "schema_version": SCHEMA_VERSION,
        "split": "val",
        "objective": "HOTA",
        "tie_break": ["IDF1", "MOTA", "trial"],
        "best_trial": best_trial,
        "trials": records,
    }
    atomic_write_json(tuning_dir / "val_report.json", val_report)
    print(f"[tune] best trial={best['trial']} HOTA={best['HOTA']:.6f} IDF1={best['IDF1']:.6f} MOTA={best['MOTA']:.6f}")
    return best_trial


def validate_locked_test_config(config_path: Path, metadata_path: Path, cache_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not config_path.is_file() or not metadata_path.is_file():
        raise PipelineError("test requires tuning/best_botsort.yaml and tuning/best_trial.json")
    config = validate_tracker_config(load_tracker_config(config_path), tuned=True)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read locked tuning metadata: {metadata_path}") from exc
    if metadata.get("locked") is not True or metadata.get("source_split") != "val":
        raise PipelineError("best_trial.json is not a locked val tuning result")
    if metadata.get("config_fingerprint") != tracker_config_fingerprint(config):
        raise PipelineError("best_botsort.yaml does not match best_trial.json")
    if cache_metadata is not None and metadata.get("cache_fingerprint") != cache_metadata.get("fingerprint"):
        raise FingerprintMismatch("Val tuning metadata does not match the current val cache")
    return config


def test_locked(
    *,
    data_root: Path,
    weights: Path,
    output_root: Path,
    device: str,
    best_config_path: Path,
    best_trial_path: Path,
    overwrite: bool = False,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    tuning_cache = output_root / "cache" / "val"
    tuning_metadata = load_cache_metadata(tuning_cache)
    validate_cache_fingerprint(
        tuning_cache,
        expected_cache_fingerprint(
            data_root=data_root,
            split="val",
            weights=weights,
            frame_limit=tuning_metadata.get("frame_limit"),
        ),
    )
    config = validate_locked_test_config(best_config_path, best_trial_path, tuning_metadata)
    test_cache = output_root / "cache" / "test"
    cache_split(
        data_root=data_root,
        split="test",
        weights=weights,
        cache_dir=test_cache,
        device=device,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    prediction_dir = output_root / "test" / "predictions"
    track_split(
        data_root=data_root,
        split="test",
        cache_dir=test_cache,
        prediction_dir=prediction_dir,
        config=config,
        device=resolve_device(device),
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    report = evaluate_predictions(
        data_root=data_root,
        split="test",
        predictions_dir=prediction_dir,
        output_dir=output_root / "test",
        evaluator="auto",
    )
    report["locked_config_fingerprint"] = tracker_config_fingerprint(config)
    report["val_cache_fingerprint"] = tuning_metadata.get("fingerprint")
    report["test_cache_fingerprint"] = load_cache_metadata(test_cache).get("fingerprint")
    atomic_write_json(output_root / "test" / "test_report.json", report)
    print(f"[test] HOTA={report['summary']['HOTA']:.6f} IDF1={report['summary']['IDF1']:.6f} MOTA={report['summary']['MOTA']:.6f}")
    return report


def _write_trackeval_fixture(
    *,
    data_root: Path,
    split: str,
    predictions_dir: Path,
    root: Path,
    cls: int,
) -> tuple[Path, Path, str]:
    """Create a MOTChallenge-shaped, single-class view for TrackEval.

    TrackEval's MOTChallenge adapter is intentionally single-class. VisDrone has
    ten classes, so the adapter is invoked once per class. Ignore annotations are
    represented as class 0 and valid objects as class 1, matching the adapter's
    pre-processing contract.
    """
    gt_root = root / "gt"
    tracker_root = root / "trackers"
    tracker_name = "visdrone"
    seqmap = root / "seqmap.txt"
    sequences = split_sequences(data_root, split)
    root.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(sequences) + "\n")
    for sequence, frames in sequences.items():
        gt_dir = gt_root / sequence / "gt"
        tracker_dir = tracker_root / tracker_name / "data"
        gt_dir.mkdir(parents=True, exist_ok=True)
        tracker_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            first_image = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
            if first_image is None:
                raise ValueError("cannot read first frame")
            height, width = first_image.shape[:2]
        except (ImportError, ValueError) as exc:
            raise PipelineError(f"Cannot read dimensions for TrackEval sequence {sequence}") from exc
        (gt_root / sequence / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={sequence}\n"
            "imDir=img1\n"
            "frameRate=30\n"
            f"seqLength={len(frames)}\n"
            f"imWidth={width}\n"
            f"imHeight={height}\n"
            "imExt=.jpg\n"
        )
        gt_objects = read_ground_truth(data_root / split / "annotations" / f"{sequence}.txt")
        gt_lines = []
        for item in gt_objects:
            is_valid = not item.ignore and item.cls == cls
            # MOTChallenge's adapter knows class 8 as ``distractor``. Mapping
            # all non-target VisDrone classes to that valid class avoids its
            # invalid-class guard while preserving distractor suppression.
            gt_class = 1 if is_valid else 8
            gt_lines.append(
                f"{item.frame},{item.track_id},{item.bbox[0]:.6f},{item.bbox[1]:.6f},{item.bbox[2]:.6f},"
                f"{item.bbox[3]:.6f},{0 if item.ignore else 1},{gt_class},1"
            )
        (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""))
        pred_objects = read_prediction_file(predictions_dir / f"{sequence}.txt", len(frames))
        pred_lines = [
            f"{item.frame},{item.track_id},{item.bbox[0]:.6f},{item.bbox[1]:.6f},{item.bbox[2]:.6f},"
            f"{item.bbox[3]:.6f},{item.confidence:.6f},1,1"
            for item in pred_objects
            if item.cls == cls
        ]
        (tracker_dir / f"{sequence}.txt").write_text("\n".join(pred_lines) + ("\n" if pred_lines else ""))
    return gt_root, tracker_root, tracker_name


def run_trackeval_adapter(
    *,
    data_root: Path,
    split: str,
    predictions_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run the pinned TrackEval package on class-specific MOT views.

    The native report remains the authoritative VisDrone report because it keeps
    the ten-class and IoA-ignore semantics explicit. This adapter still executes
    the requested TrackEval HOTA/CLEAR/Identity/Count stack and stores its raw
    result for auditability.
    """
    # The pinned TrackEval commit predates NumPy 1.24 and still references the
    # removed aliases. Keep the compatibility shim local to this adapter rather
    # than relaxing the project's required NumPy pin.
    for alias, builtin in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if alias not in np.__dict__:
            setattr(np, alias, builtin)
    try:
        import trackeval
    except ImportError as exc:
        raise PipelineError("trackeval is not installed") from exc
    try:
        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update(
            {
                "USE_PARALLEL": False,
                "PRINT_CONFIG": False,
                "PRINT_RESULTS": False,
                "PRINT_ONLY_COMBINED": True,
                "OUTPUT_SUMMARY": False,
                "OUTPUT_DETAILED": False,
                "TIME_PROGRESS": False,
            }
        )
        evaluator = trackeval.Evaluator(evaluator_config)
        raw_by_class = {}
        for cls in VISDRONE_CLASSES:
            class_root = work_dir / f"class_{cls}"
            gt_root, tracker_root, tracker_name = _write_trackeval_fixture(
                data_root=data_root,
                split=split,
                predictions_dir=predictions_dir,
                root=class_root,
                cls=cls,
            )
            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            dataset_config.update(
                {
                    "GT_FOLDER": str(gt_root),
                    "TRACKERS_FOLDER": str(tracker_root),
                    "TRACKERS_TO_EVAL": [tracker_name],
                    "SPLIT_TO_EVAL": split,
                    "SEQMAP_FILE": str(class_root / "seqmap.txt"),
                    "SKIP_SPLIT_FOL": True,
                    "DO_PREPROC": True,
                    "PRINT_CONFIG": False,
                    "CLASSES_TO_EVAL": ["pedestrian"],
                }
            )
            dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
            metrics_config = {"PRINT_CONFIG": False, "THRESHOLD": 0.5}
            metrics = [
                trackeval.metrics.HOTA(metrics_config),
                trackeval.metrics.CLEAR(metrics_config),
                trackeval.metrics.Identity(metrics_config),
            ]
            raw_by_class[str(cls)] = evaluator.evaluate([dataset], metrics)
        return {"trackeval_version": getattr(trackeval, "__version__", None), "classes": raw_by_class}
    except Exception as exc:
        raise PipelineError(f"TrackEval adapter failed: {exc}") from exc


def summarize_trackeval_raw(raw: Mapping[str, Any]) -> dict[str, float | int]:
    """Combine the ten single-class TrackEval runs detection-wise.

    ``MotChallenge2DBox`` always adds Count and combines sequences internally.
    Since VisDrone has ten classes, the adapter runs that dataset once per class;
    this function applies TrackEval's detection-averaged combination rule across
    those ten combined-class results.
    """
    hota_tp = hota_fn = hota_fp = None
    assa_weighted = None
    loca_weighted = None
    clear_tp = clear_fn = clear_fp = clear_idsw = clear_frag = clear_motp_sum = 0
    identity_tp = identity_fn = identity_fp = 0
    gt_dets = pred_dets = gt_ids = pred_ids = 0
    for class_payload in raw.get("classes", {}).values():
        if not isinstance(class_payload, (list, tuple)) or not class_payload:
            continue
        evaluator_result = class_payload[0]
        dataset_result = evaluator_result.get("MotChallenge2DBox", {})
        tracker_result = next(iter(dataset_result.values()))
        combined = tracker_result["COMBINED_SEQ"]["pedestrian"]
        hota_block = combined["HOTA"]
        tp = np.asarray(hota_block["HOTA_TP"], dtype=np.int64)
        fn = np.asarray(hota_block["HOTA_FN"], dtype=np.int64)
        fp = np.asarray(hota_block["HOTA_FP"], dtype=np.int64)
        assa = np.asarray(hota_block["AssA"], dtype=np.float64)
        loca = np.asarray(hota_block["LocA"], dtype=np.float64)
        hota_tp = tp.copy() if hota_tp is None else hota_tp + tp
        hota_fn = fn.copy() if hota_fn is None else hota_fn + fn
        hota_fp = fp.copy() if hota_fp is None else hota_fp + fp
        assa_weighted = assa * tp if assa_weighted is None else assa_weighted + assa * tp
        loca_weighted = loca * tp if loca_weighted is None else loca_weighted + loca * tp
        clear = combined["CLEAR"]
        identity = combined["Identity"]
        count = combined["Count"]
        clear_tp += int(clear["CLR_TP"])
        clear_fn += int(clear["CLR_FN"])
        clear_fp += int(clear["CLR_FP"])
        clear_idsw += int(clear["IDSW"])
        clear_frag += int(clear["Frag"])
        clear_motp_sum += float(clear["MOTP_sum"])
        identity_tp += int(identity["IDTP"])
        identity_fn += int(identity["IDFN"])
        identity_fp += int(identity["IDFP"])
        gt_dets += int(count["GT_Dets"])
        pred_dets += int(count["Dets"])
        gt_ids += int(count["GT_IDs"])
        pred_ids += int(count["IDs"])
    if hota_tp is None:
        return {"HOTA": 0.0, "IDF1": 0.0, "MOTA": 0.0, "IDP": 0.0, "IDR": 0.0}
    assa_combined = assa_weighted / np.maximum(1, hota_tp)
    loca_combined = loca_weighted / np.maximum(1, hota_tp)
    deta_combined = hota_tp / np.maximum(1, hota_tp + hota_fn + hota_fp)
    hota_combined = np.sqrt(deta_combined * assa_combined)
    idf1_denominator = 2 * identity_tp + identity_fn + identity_fp
    idf1 = 2 * identity_tp / idf1_denominator if idf1_denominator else 0.0
    idp = identity_tp / (identity_tp + identity_fp) if identity_tp + identity_fp else 0.0
    idr = identity_tp / (identity_tp + identity_fn) if identity_tp + identity_fn else 0.0
    mota = 1.0 - (clear_fn + clear_fp + clear_idsw) / gt_dets if gt_dets else 0.0
    motp = clear_motp_sum / clear_tp if clear_tp else 0.0
    precision = clear_tp / (clear_tp + clear_fp) if clear_tp + clear_fp else 0.0
    recall = clear_tp / (clear_tp + clear_fn) if clear_tp + clear_fn else 0.0
    return {
        "HOTA": 100.0 * float(np.mean(hota_combined)),
        "DetA": 100.0 * float(np.mean(deta_combined)),
        "AssA": 100.0 * float(np.mean(assa_combined)),
        "LocA": 100.0 * float(np.mean(loca_combined)),
        "MOTA": 100.0 * mota,
        "MOTP": 100.0 * motp,
        "IDF1": 100.0 * idf1,
        "IDP": 100.0 * idp,
        "IDR": 100.0 * idr,
        "Precision": 100.0 * precision,
        "Recall": 100.0 * recall,
        "TP": clear_tp,
        "FP": clear_fp,
        "FN": clear_fn,
        "IDSW": clear_idsw,
        "Fragmentation": clear_frag,
        "GT": gt_dets,
        "Pred": pred_dets,
        "NumGTIDs": gt_ids,
        "NumPredIDs": pred_ids,
    }


def evaluate_predictions(
    *,
    data_root: Path,
    split: str,
    predictions_dir: Path,
    output_dir: Path,
    evaluator: str = "auto",
) -> dict[str, Any]:
    if evaluator not in {"auto", "trackeval", "native"}:
        raise PipelineError(f"Unknown evaluator: {evaluator}")
    if evaluator == "trackeval" or (evaluator == "auto" and importlib.util.find_spec("trackeval") is not None):
        work_dir = output_dir / ".trackeval"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            trackeval_raw = run_trackeval_adapter(
                data_root=data_root,
                split=split,
                predictions_dir=predictions_dir,
                work_dir=work_dir,
            )
        except PipelineError:
            if evaluator == "trackeval":
                raise
            trackeval_raw = {"status": "unavailable_or_failed"}
        else:
            atomic_write_json(output_dir / f"{split}_trackeval_raw.json", trackeval_raw)
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)
    else:
        trackeval_raw = {"status": "not_installed"}
    report = evaluate_native(data_root=data_root, split=split, predictions_dir=predictions_dir, output_dir=output_dir)
    report["trackeval"] = trackeval_raw
    if trackeval_raw.get("classes"):
        report["native_summary"] = report["summary"]
        report["trackeval_summary"] = summarize_trackeval_raw(trackeval_raw)
        report["summary"] = report["trackeval_summary"]
    report["backend"] = "trackeval+native" if trackeval_raw.get("classes") else "native"
    _write_metrics_csv(output_dir / f"{split}_metrics.csv", report)
    atomic_write_json(output_dir / f"{split}_report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto", help="auto, cpu or a CUDA device such as cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="replace explicitly targeted outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="run detector inference once and write a fingerprinted cache")
    _common_arguments(cache)
    cache.add_argument("--split", choices=("val", "test"), required=True)
    cache.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    cache.add_argument("--max-frames", type=int, default=None, help="development smoke limit; changes the fingerprint")

    track = subparsers.add_parser("track", help="run BoT-SORT from a detector cache")
    _common_arguments(track)
    track.add_argument("--split", choices=("val", "test"), required=True)
    track.add_argument("--cache-dir", type=Path, default=None)
    track.add_argument("--prediction-dir", type=Path, default=None)
    track.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    track.add_argument("--max-frames", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate", help="evaluate existing VisDrone predictions")
    evaluate.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    evaluate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--predictions-dir", type=Path, required=True)
    evaluate.add_argument("--evaluator", choices=("auto", "trackeval", "native"), default="auto")

    tune = subparsers.add_parser("tune", help="run the 300-trial val Optuna study")
    _common_arguments(tune)
    tune.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    tune.add_argument("--n-trials", type=int, default=300, help="300 for the planned run; smaller values are smoke runs")
    tune.add_argument("--max-frames", type=int, default=None, help="development smoke limit; changes the cache fingerprint")

    test = subparsers.add_parser("test", help="run the locked best val configuration on test")
    _common_arguments(test)
    test.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    test.add_argument("--best-config", type=Path, default=None)
    test.add_argument("--best-trial", type=Path, default=None)
    test.add_argument("--max-frames", type=int, default=None, help="development smoke limit; changes the test cache fingerprint")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cache":
            metadata = cache_split(
                data_root=args.data_root,
                split=args.split,
                weights=args.weights,
                cache_dir=args.output_root / "cache" / args.split,
                device=args.device,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
            print(json.dumps({"cache": str(args.output_root / "cache" / args.split), "fingerprint": metadata["fingerprint"]}, indent=2))
        elif args.command == "track":
            cache_dir = args.cache_dir or args.output_root / "cache" / args.split
            prediction_dir = args.prediction_dir or args.output_root / "tracks" / args.split
            track_split(
                data_root=args.data_root,
                split=args.split,
                cache_dir=cache_dir,
                prediction_dir=prediction_dir,
                config_path=args.tracker_config,
                device=resolve_device(args.device),
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
        elif args.command == "evaluate":
            output_dir = args.output_root / ("test" if args.split == "test" else "evaluation")
            report = evaluate_predictions(
                data_root=args.data_root,
                split=args.split,
                predictions_dir=args.predictions_dir,
                output_dir=output_dir,
                evaluator=args.evaluator,
            )
            print(json.dumps(report["summary"], indent=2))
        elif args.command == "tune":
            tune_val(
                data_root=args.data_root,
                weights=args.weights,
                output_root=args.output_root,
                device=args.device,
                n_trials=args.n_trials,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
        elif args.command == "test":
            best_config = args.best_config or args.output_root / "tuning" / "best_botsort.yaml"
            best_trial = args.best_trial or args.output_root / "tuning" / "best_trial.json"
            test_locked(
                data_root=args.data_root,
                weights=args.weights,
                output_root=args.output_root,
                device=args.device,
                best_config_path=best_config,
                best_trial_path=best_trial,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
        return 0
    except (PipelineError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
