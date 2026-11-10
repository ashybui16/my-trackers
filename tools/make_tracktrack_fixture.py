"""Run Ultralytics TRACKTRACK on seeded synthetic detections; write inputs + expected outputs as JSON.

Usage (Ultralytics environment): python tools/make_tracktrack_fixture.py cpp/tests/fixtures/tracktrack_parity.json
"""
import json
import sys

import numpy as np
from ultralytics.cfg import ROOT
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace


class Dets:
    """Minimal NumPy Results stand-in: conf, cls, xywh and boolean indexing."""

    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Dets(self.xywh[mask], self.conf[mask], self.cls[mask])


def xyxy_from_xywh(xywh):
    """Same float64 path as TTSTrack: tlwh = xywh - wh/2, xyxy = tlwh + wh."""
    b = xywh.astype(np.float64)
    x1, y1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    return np.stack([x1, y1, x1 + b[:, 2], y1 + b[:, 3]], axis=1)


def sequence(seed, frames, n_objects, with_recovered):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([50, 50], [900, 600], (n_objects, 2))
    vel = rng.uniform(-4, 4, (n_objects, 2))
    size = rng.uniform([20, 30], [60, 90], (n_objects, 2))
    cls = rng.integers(0, 3, n_objects)
    gaps = {i: set(range(s, s + rng.integers(2, 12))) for i, s in enumerate(rng.integers(5, frames, n_objects))}
    out = []
    for f in range(frames):
        pos += vel
        rows, rec = [], []
        for i in range(n_objects):
            if f in gaps[i] or rng.random() < 0.05:
                continue
            c = pos[i] + rng.normal(0, 1.5, 2)
            wh = size[i] * rng.uniform(0.95, 1.05, 2)
            score = rng.uniform(0.15, 1.0)
            rows.append([*c, *wh, score, cls[i] if rng.random() > 0.05 else (cls[i] + 1) % 3])
            if rng.random() < 0.05:  # near-duplicate box (exercises TAI NMS)
                rows.append([*(c + rng.normal(0, 3, 2)), *wh, rng.uniform(0.7, 1.0), cls[i]])
            if with_recovered and rng.random() < 0.15:  # stand-in for a box tight NMS dropped
                rec.append([*(c + rng.normal(0, 6, 2)), *(wh * 0.9), rng.uniform(0.5, 1.0), cls[i]])
        if rng.random() < 0.3:  # clutter
            rows.append([*rng.uniform([50, 50], [900, 600]), 30, 40, rng.uniform(0.1, 0.9), 0])
        out.append((np.array(rows, dtype=np.float32).reshape(-1, 6), np.array(rec, dtype=np.float32).reshape(-1, 6)))
    return out


def run(cfg_overrides, seed, frames, n_objects, with_recovered):
    cfg = {**YAML.load(ROOT / "cfg/trackers/tracktrack.yaml"), "gmc_method": "none", **cfg_overrides}
    tracker = TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))
    result_frames = []
    for dets, rec in sequence(seed, frames, n_objects, with_recovered):
        xywh = dets[:, :4]
        dets_del = (rec[:, :4], rec[:, 4], rec[:, 5]) if len(rec) else None
        tracks = tracker.update(Dets(xywh, dets[:, 4], dets[:, 5]), img=None, dets_del=dets_del)
        tracks = np.asarray(tracks, dtype=np.float64).reshape(-1, 8)
        det_xyxy = xyxy_from_xywh(xywh)
        rec_xyxy = xyxy_from_xywh(rec[:, :4]) if len(rec) else np.zeros((0, 4))
        result_frames.append({
            # x1 y1 x2 y2 score class  (score is the float32 value, widened)
            "dets": np.column_stack([det_xyxy, dets[:, 4:6].astype(np.float64)]).ravel().tolist(),
            "recovered": np.column_stack([rec_xyxy, rec[:, 4:6].astype(np.float64)]).ravel().tolist(),
            # id x1 y1 x2 y2 score class
            "expected": np.column_stack([tracks[:, 4], tracks[:, :4], tracks[:, 5:7]]).ravel().tolist(),
        })
    keys = ["track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh",
            "lost_match_thr", "conf_weight", "angle_weight", "penalty_p", "penalty_q", "reduce_step", "tai_thr",
            "min_track_len"]
    return {"config": {k: cfg[k] for k in keys}, "frames": result_frames}


runs = [
    run({}, seed=1, frames=40, n_objects=10, with_recovered=False),
    run({"lost_match_thr": 0.9, "track_buffer": 10}, seed=2, frames=40, n_objects=10, with_recovered=True),
    run({"min_track_len": 1, "track_high_thresh": 0.5, "track_low_thresh": 0.2, "new_track_thresh": 0.6},
        seed=3, frames=40, n_objects=10, with_recovered=True),
]
with open(sys.argv[1], "w") as f:
    json.dump({"runs": runs}, f, separators=(",", ":"))
print({i: sum(len(fr["expected"]) // 7 for fr in r["frames"]) for i, r in enumerate(runs)})
