#!/usr/bin/env python3
"""VisDrone pipeline entry point backed by the standalone ``mot_tracker``.

The original ``tools/visdrone_mot.py`` remains the historical baseline.  This
entry point reuses its detector-cache/evaluator helpers while routing every
tracking operation through ``Detection`` -> ``BoTSORT`` -> ``TrackOutput``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mot_tracker import BoTSORTConfig

from tools import visdrone_mot as baseline
from tools.mot_tracker_adapter import (
    StandaloneAdapterError,
    standalone_config,
    standalone_config_from_yaml,
    track_cached_split,
)


DEFAULT_OUTPUT_ROOT = baseline.DEFAULT_OUTPUT_ROOT
DEFAULT_TRACKER_CONFIG = baseline.DEFAULT_TRACKER_CONFIG


def _config_mapping(config: BoTSORTConfig) -> dict[str, Any]:
    return {"tracker_type": "botsort", **asdict(config)}


def _write_standalone_config(path: Path, config: BoTSORTConfig) -> None:
    baseline.write_tracker_config(path, _config_mapping(config))


def _cache(
    *, data_root: Path, split: str, weights: Path, output_root: Path, device: str, overwrite: bool, frame_limit: int | None
) -> dict[str, Any]:
    return baseline.cache_split(
        data_root=data_root,
        split=split,
        weights=weights,
        cache_dir=output_root / "cache" / split,
        device=device,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )


def track(
    *, data_root: Path, split: str, output_root: Path, tracker_config: Path, overwrite: bool, frame_limit: int | None
) -> dict[str, Path]:
    config = standalone_config_from_yaml(tracker_config)
    return track_cached_split(
        data_root=data_root,
        split=split,
        cache_dir=output_root / "cache" / split,
        prediction_dir=output_root / "standalone" / split / "predictions",
        config=config,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )


def evaluate(
    *, data_root: Path, split: str, output_root: Path, predictions_dir: Path, evaluator: str
) -> dict[str, Any]:
    return baseline.evaluate_predictions(
        data_root=data_root,
        split=split,
        predictions_dir=predictions_dir,
        output_dir=output_root / "standalone" / split / "evaluation",
        evaluator=evaluator,
    )


def suggest_standalone_params(trial: Any) -> dict[str, Any]:
    high = float(trial.suggest_float("track_high_thresh", 0.15, 0.60))
    low = float(trial.suggest_float("track_low_thresh", 0.01, max(0.010001, high - 0.000001)))
    new = float(trial.suggest_float("new_track_thresh", high, 0.80))
    return {
        "track_high_thresh": high,
        "track_low_thresh": low,
        "new_track_thresh": new,
        "track_buffer": int(trial.suggest_categorical("track_buffer", [15, 30, 45, 60, 90])),
        "match_thresh": float(trial.suggest_float("match_thresh", 0.60, 0.95)),
        "fuse_score": bool(trial.suggest_categorical("fuse_score", [True, False])),
        "gmc_method": str(trial.suggest_categorical("gmc_method", ["sparseOptFlow", "none"])),
    }


def tune(
    *,
    data_root: Path,
    weights: Path,
    output_root: Path,
    device: str,
    n_trials: int,
    overwrite: bool,
    frame_limit: int | None,
) -> dict[str, Any]:
    """Run the plan's Optuna search while using the standalone tracker."""

    try:
        import optuna
    except ImportError as exc:
        raise StandaloneAdapterError("optuna>=4,<5 is required for tuning") from exc
    if n_trials <= 0:
        raise StandaloneAdapterError("n_trials must be positive")
    cache_dir = output_root / "cache" / "val"
    _cache(
        data_root=data_root,
        split="val",
        weights=weights,
        output_root=output_root,
        device=device,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    tuning_dir = output_root / "standalone" / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    study_path = tuning_dir / "study.sqlite3"
    if overwrite and study_path.exists():
        study_path.unlink()
    study = optuna.create_study(
        study_name="visdrone_botsort_standalone_val",
        storage=f"sqlite:///{study_path.resolve()}",
        load_if_exists=not overwrite,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    if not study.trials:
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
    trial_root = tuning_dir / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        params = suggest_standalone_params(trial)
        config = standalone_config(params)
        trial_dir = trial_root / f"trial_{trial.number:03d}"
        prediction_dir = trial_dir / "predictions"
        track_cached_split(
            data_root=data_root,
            split="val",
            cache_dir=cache_dir,
            prediction_dir=prediction_dir,
            config=config,
            overwrite=True,
            frame_limit=frame_limit,
        )
        report = baseline.evaluate_predictions(
            data_root=data_root,
            split="val",
            predictions_dir=prediction_dir,
            output_dir=trial_dir,
            evaluator="auto",
        )
        metrics = report["summary"]
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("metrics_path", str(trial_dir / "val_report.json"))
        baseline.atomic_write_json(trial_dir / "config.json", _config_mapping(config))
        return float(metrics["HOTA"])

    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    records = [baseline._trial_record(trial) for trial in study.trials]
    if len(records) != n_trials:
        raise StandaloneAdapterError(f"Expected {n_trials} trials, found {len(records)}")
    best_record: Mapping[str, Any] | None = None
    for record in records:
        if str(record.get("state")) != "TrialState.COMPLETE":
            continue
        candidate = {"trial": record["trial"], **record.get("metrics", {})}
        incumbent = None if best_record is None else {"trial": best_record["trial"], **best_record.get("metrics", {})}
        if baseline.is_better_trial(candidate, incumbent):
            best_record = record
    if best_record is None:
        raise StandaloneAdapterError("No completed standalone tuning trial")
    best_config = standalone_config(best_record["params"])
    best_config_path = tuning_dir / "best_botsort.yaml"
    _write_standalone_config(best_config_path, best_config)
    best_trial = {
        "schema_version": baseline.SCHEMA_VERSION,
        "locked": True,
        "source_split": "val",
        "trial": best_record["trial"],
        "params": best_record["params"],
        "metrics": best_record.get("metrics", {}),
        "config": _config_mapping(best_config),
        "config_fingerprint": baseline.tracker_config_fingerprint(_config_mapping(best_config)),
        "n_trials": len(records),
        "created_unix": time.time(),
    }
    baseline.atomic_write_json(tuning_dir / "best_trial.json", best_trial)
    baseline._write_trials_csv(tuning_dir / "trials.csv", records)
    baseline.atomic_write_json(tuning_dir / "val_report.json", {"best_trial": best_trial, "trials": records})
    return best_trial


def test_locked(
    *, data_root: Path, weights: Path, output_root: Path, device: str, best_config: Path, best_trial: Path, overwrite: bool, frame_limit: int | None
) -> dict[str, Any]:
    if not best_config.is_file() or not best_trial.is_file():
        raise StandaloneAdapterError("missing locked standalone best_botsort.yaml/best_trial.json")
    metadata = json.loads(best_trial.read_text())
    if not metadata.get("locked"):
        raise StandaloneAdapterError("best_trial.json is not locked")
    config = standalone_config_from_yaml(best_config)
    if metadata.get("source_split") not in {None, "val"}:
        raise StandaloneAdapterError("locked standalone config must originate from val tuning")
    expected_fingerprint = metadata.get("config_fingerprint")
    if expected_fingerprint:
        actual_fingerprint = baseline.tracker_config_fingerprint(_config_mapping(config))
        if actual_fingerprint != expected_fingerprint:
            raise StandaloneAdapterError("best_botsort.yaml does not match best_trial.json")
    _cache(
        data_root=data_root,
        split="test",
        weights=weights,
        output_root=output_root,
        device=device,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    prediction_dir = output_root / "standalone" / "test" / "predictions"
    track_cached_split(
        data_root=data_root,
        split="test",
        cache_dir=output_root / "cache" / "test",
        prediction_dir=prediction_dir,
        config=config,
        overwrite=overwrite,
        frame_limit=frame_limit,
    )
    return evaluate(
        data_root=data_root,
        split="test",
        output_root=output_root,
        predictions_dir=prediction_dir,
        evaluator="auto",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-root", type=Path, default=baseline.DEFAULT_DATA_ROOT)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--max-frames", type=int, default=None)

    cache = subparsers.add_parser("cache")
    common(cache)
    cache.add_argument("--split", choices=("val", "test"), required=True)
    cache.add_argument("--weights", type=Path, default=baseline.DEFAULT_WEIGHTS)
    cache.add_argument("--device", default="auto")

    tracking = subparsers.add_parser("track")
    common(tracking)
    tracking.add_argument("--split", choices=("val", "test"), required=True)
    tracking.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--data-root", type=Path, default=baseline.DEFAULT_DATA_ROOT)
    evaluation.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--predictions-dir", type=Path, required=True)
    evaluation.add_argument("--evaluator", choices=("auto", "trackeval", "native"), default="auto")

    tuning = subparsers.add_parser("tune")
    common(tuning)
    tuning.add_argument("--weights", type=Path, default=baseline.DEFAULT_WEIGHTS)
    tuning.add_argument("--device", default="auto")
    tuning.add_argument("--n-trials", type=int, default=300)

    testing = subparsers.add_parser("test")
    common(testing)
    testing.add_argument("--weights", type=Path, default=baseline.DEFAULT_WEIGHTS)
    testing.add_argument("--device", default="auto")
    testing.add_argument("--best-config", type=Path, default=None)
    testing.add_argument("--best-trial", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cache":
            metadata = _cache(
                data_root=args.data_root,
                split=args.split,
                weights=args.weights,
                output_root=args.output_root,
                device=args.device,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
            print(json.dumps({"cache": str(args.output_root / "cache" / args.split), "fingerprint": metadata["fingerprint"]}, indent=2))
        elif args.command == "track":
            outputs = track(
                data_root=args.data_root,
                split=args.split,
                output_root=args.output_root,
                tracker_config=args.tracker_config,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
            print(json.dumps({"predictions": {name: str(path) for name, path in outputs.items()}}, indent=2))
        elif args.command == "evaluate":
            report = evaluate(
                data_root=args.data_root,
                split=args.split,
                output_root=args.output_root,
                predictions_dir=args.predictions_dir,
                evaluator=args.evaluator,
            )
            print(json.dumps(report["summary"], indent=2))
        elif args.command == "tune":
            print(json.dumps(tune(data_root=args.data_root, weights=args.weights, output_root=args.output_root, device=args.device, n_trials=args.n_trials, overwrite=args.overwrite, frame_limit=args.max_frames), indent=2))
        else:
            print(json.dumps(test_locked(data_root=args.data_root, weights=args.weights, output_root=args.output_root, device=args.device, best_config=args.best_config or args.output_root / "standalone" / "tuning" / "best_botsort.yaml", best_trial=args.best_trial or args.output_root / "standalone" / "tuning" / "best_trial.json", overwrite=args.overwrite, frame_limit=args.max_frames)["summary"], indent=2))
        return 0
    except (StandaloneAdapterError, baseline.PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
