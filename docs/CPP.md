# C++ TensorRT video pipeline

PLAN2 adds a C++17 implementation under `cpp/`:

- `mot_tracker_cpp`: detector-agnostic BoT-SORT with the Python package's
  Kalman/lifecycle/association/GMC behavior.
- `mot_inference_cpp`: RAII TensorRT detector and synchronous `MotPipeline`.
- `mot_video`: video input, annotated MP4 output and VisDrone/MOT text output.

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

```cpp
#include "mot_inference/pipeline.hpp"

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
