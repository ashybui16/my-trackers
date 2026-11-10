# Trackers

Every tracker implements `mot::ITracker` (`include/mot/tracking/tracker.hpp`):

```cpp
std::vector<TrackOutput> update(const std::vector<Detection>& detections, const cv::Mat& frame);
void reset();
std::string_view name() const;
// Optional: trackers that use recovered (loose-NMS) detections override both.
std::vector<TrackOutput> updateWithRecovered(const std::vector<Detection>& detections,
                                             const std::vector<Detection>& recovered, const cv::Mat& frame);
bool usesRecoveredDetections() const;
```

Applications never name a tracker class directly. They pick one by config
(`tracker.type` in YAML, `--tracker` in `mot_video`) and call
`mot::createTracker(config)`.

## Layout

| Path | Contents |
|---|---|
| `common/` | Building blocks shared by SORT-family trackers: Kalman filter, linear assignment, IoU/geometry, association helpers, GMC, `Track`. `track_pool.hpp` has pool bookkeeping templated on the track type, for trackers with their own track class. Private headers, not installed. |
| `botsort/` | BoT-SORT (Ultralytics-compatible). |
| `tracktrack/` | TrackTrack (Ultralytics-compatible, no ReID). Uses recovered detections via `updateWithRecovered`. |
| `factory.cpp` | `createTracker`, `defaultTrackerConfig`, `trackerType`, `availableTrackers`. |

## Adding a tracker (example: ByteTrack)

1. **Config.** Add `include/mot/tracking/bytetrack_config.hpp` with a
   `struct ByteTrackConfig { static constexpr std::string_view kType{"bytetrack"}; ...; void validate() const; };`
   and implement `validate()` in `bytetrack/bytetrack_config.cpp`.
2. **Register the config.** In `include/mot/tracking/config.hpp`, include the header and
   extend the variant: `using TrackerConfig = std::variant<BoTSORTConfig, ByteTrackConfig>;`
3. **Implementation.** Add `bytetrack/bytetrack.{hpp,cpp}` with
   `class ByteTrack final : public ITracker`. Reuse what you need from `common/`.
4. **Factory.** In `factory.cpp`, add
   `std::unique_ptr<ITracker> makeTracker(const ByteTrackConfig& c) { return std::make_unique<ByteTrack>(c); }`
5. **YAML.** In `src/pipeline/config.cpp`, add a
   `readTracker(const cv::FileNode&, const std::string& section, ByteTrackConfig&)` overload.
6. **Build.** List the new `.cpp` files in `src/tracking/CMakeLists.txt`.
7. **Tests.** `tests/tracking/test_tracker_contract.cpp` and `tests/pipeline/test_pipeline.cpp`
   pick up the new tracker automatically. Add tracker-specific tests as
   `tests/tracking/test_bytetrack.cpp`.

Steps 2, 4 and 5 rely on `std::visit`, so if you forget an overload the build
fails instead of the tracker breaking at runtime.

Then select it in YAML:

```yaml
tracker:
  type: bytetrack
  bytetrack: { ... }
```
