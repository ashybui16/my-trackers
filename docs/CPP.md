# C++ TensorRT video pipeline

PLAN2 adds a C++17 implementation under `cpp/`:

- `mot_tracker_cpp`: detector-agnostic BoT-SORT. GMC state transforms and
  removed-track bookkeeping follow the vendored Ultralytics 8.4.157 implementation;
  assignment continues to use the existing C++ Hungarian solver.
- `mot_inference_cpp`: RAII TensorRT detector and synchronous `MotPipeline`.
- `mot_video`: video input, annotated MP4 output and VisDrone/MOT text output.

Implementation files are grouped under `cpp/src/detector/`, `cpp/src/tracker/`,
`cpp/src/pipeline/` and `cpp/src/apps/`. Tests are grouped under
`cpp/tests/detector/` and `cpp/tests/tracker/`. Public headers live under
`cpp/include/mot_detector/`, `cpp/include/mot_tracker/` and
`cpp/include/mot_pipeline/`. The postprocess header is private and lives in
`cpp/internal/detector/`. Headers under `cpp/include/mot_inference/` forward
the old include paths to the new locations.

GMC applies the same affine rotation block to `(x, y)`, `(w, h)` and both
velocity pairs, with translation applied only to the center. This can produce
signed sizes for large rotations. Finite signed sizes and inverted corners are
preserved in Track::bbox(), prediction, Kalman updates and TrackOutput, without
substituting a previous box or repairing dimensions. Input detections still require
positive width and height. Invalid/non-finite warp rollback remains in place.
Tracker IoU follows Ultralytics matching.iou_distance/bbox_ioa: float32
coordinates and arithmetic, intersection dimensions clipped at zero, signed box
areas, and denominator union + 1e-7. IoU distance is also subtracted in float32.
The raw-detector NMS implementation is separate and unchanged.
Tracks newly marked removed stay in the lost pool until the next frame's merge,
so they can recover their ID if matched in that frame. Removed history retains
the latest 1000 entries, including repeated references as in Ultralytics.

## Build

The target is Linux x86_64 with OpenCV 4, CUDA and TensorRT 8.6.x. Configure
from the repository root or from `cpp/`:

```bash
cmake -S . -B /tmp/my-trackers-cpp-build \
  -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DMOT_BUILD_TESTS=ON
cmake --build /tmp/my-trackers-cpp-build -j
ctest --test-dir /tmp/my-trackers-cpp-build --output-on-failure
```

The tracker test is CPU-only. To enable the separate TensorRT smoke test,
configure with `-DMOT_BUILD_GPU_TESTS=ON`, set `MOT_ENGINE_PATH` to a
GPU-compatible engine, then run `ctest -L gpu`; the test is skipped when the
environment variable is absent.

`TENSORRT_ROOT` must contain `include/NvInfer.h` and the matching
`libnvinfer.so`. `CUDAToolkit_ROOT` may be set to the CUDA toolkit used by
TensorRT. The serialized engine is not portable across arbitrary GPU or
TensorRT-version changes; a deserialize failure means the engine must be
rebuilt for the target environment.

## Run

```bash
/tmp/my-trackers-cpp-build/mot_video data/input.ts \
  --engine weights/26leaf1.engine \
  --output-dir runs/video_inference_cpp \
  --conf 0.01 --iou 0.7 --max-det 300 \
  --gmc-method sparseOptFlow
```

The CLI writes `<stem>_annotated.mp4` and `<stem>_predictions.txt`. The video
keeps the decoded input dimensions and FPS and uses `mp4v`. Prediction rows
are `frame,id,x,y,width,height,confidence,class,-1,-1`; frame and class IDs
are one-based. Incomplete inference only leaves temporary dot files, which
are removed on failure. Pass `--overwrite` to replace existing outputs.

All tracker flags are exposed by the CLI: `--track-high-thresh`,
`--track-low-thresh`, `--new-track-thresh`, `--track-buffer`,
`--match-thresh`, `--fuse-score`/`--no-fuse-score`, `--gmc-method` and
`--gmc-downscale`.

## Embedding a camera/frame caller

For detection only, call the detector directly:

```cpp
#include "mot_detector/detector.hpp"

mot_inference::TensorRTDetector detector("weights/26leaf1.engine");
std::vector<mot_tracker::Detection> detections = detector.infer(bgr_frame);
```

For detection and tracking, use the existing pipeline:

```cpp
#include "mot_pipeline/pipeline.hpp"

mot_inference::PipelineConfig config;
config.detector.conf_threshold = 0.01;
config.tracker.gmc_method = "sparseOptFlow";
mot_inference::MotPipeline pipeline("weights/26leaf1.engine", config);

// Each call is synchronous and owns state for this stream.
std::vector<mot_tracker::TrackOutput> tracks = pipeline.processFrame(bgr_frame);
// Start a new stream without reloading the TensorRT engine.
pipeline.reset();
```

`MotPipeline` and `BoTSORT` instances are stateful and must not be called
concurrently. Use one instance per camera stream. Camera capture, queues,
backpressure and evaluator integration remain outside v1.
