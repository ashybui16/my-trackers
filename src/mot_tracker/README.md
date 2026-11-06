# Trackers (Python)

This package mirrors the C++ library in `cpp/src/tracking` module for module, so
a tracker prototyped here can be ported by translation. Python is also the
numerical oracle: C++ tests compare against values produced here.

Every tracker implements `mot_tracker.Tracker` (`base.py`, ≙ `mot::ITracker`):

```python
def update(self, detections: Iterable[Detection], frame: np.ndarray | None = None) -> list[TrackOutput]: ...
def reset(self) -> None: ...
@property
def name(self) -> str: ...
```

Applications pick a tracker by config and call `create_tracker(config)`.

## Layout

| Python | C++ | Contents |
|---|---|---|
| `types.py` | `include/mot/core/types.hpp` | `Detection`, `TrackOutput` |
| `base.py` | `include/mot/tracking/tracker.hpp` | `Tracker` interface |
| `config.py` | `include/mot/tracking/config.hpp`, `src/pipeline/config.cpp` | `TrackerConfig`, `tracker_type`, `default_tracker_config`, `validate`, `tracker_config_from_mapping` |
| `factory.py` | `src/tracking/factory.cpp` | `create_tracker`, `available_trackers` |
| `common/` | `src/tracking/common/` | Kalman filter, linear assignment, IoU/geometry, association helpers, GMC, `Track` |
| `trackers/botsort/` | `src/tracking/botsort/`, `include/mot/tracking/botsort_config.hpp` | BoT-SORT (Ultralytics-compatible) |

## Adding a tracker (example: ByteTrack)

1. **Config.** Add `trackers/bytetrack/config.py` with a frozen dataclass
   `ByteTrackConfig` that has `TYPE: ClassVar[str] = "bytetrack"` and a `validate()` method.
   Use the same field names as the future C++ struct.
2. **Implementation.** Add `trackers/bytetrack/tracker.py` with
   `class ByteTrack(Tracker)`, using what you need from `common/`, and export
   both classes from `trackers/bytetrack/__init__.py`.
3. **Register.** In `config.py`, add the class to `TrackerConfig` and `CONFIG_TYPES`.
   In `factory.py`, add `ByteTrackConfig: ByteTrack` to `_REGISTRY`.
4. **Tests.** `tests/test_tracker_contract.py` picks up the new tracker automatically.
   Put tracker-specific tests in `tests/test_bytetrack.py`.

YAML (`configs/config.yaml`) then selects it with `tracker: {type: bytetrack, bytetrack: {...}}`.

## Porting to C++

Once the tracker is worth keeping, save its per-frame outputs on a few sequences as
fixtures under `tests/fixtures/`. Then follow `cpp/src/tracking/README.md` and add a
C++ test that checks against those fixtures. Keep new shared helpers in `common/` on
both sides so the modules stay one-to-one.
