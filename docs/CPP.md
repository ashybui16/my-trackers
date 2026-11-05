# C++ video pipeline (TensorRT / ONNX Runtime)

PLAN2 adds a C++17 multi-object-tracking library under `cpp/`, organised so new
trackers can be added and so another application can consume it as a CMake package.

| Target | Contents | Needs CUDA |
|---|---|---|
| `mot::core` | `Detection`, `TrackOutput`, `BBox`, `DetectorConfig` | no |
| `mot::tracking` | `ITracker`, `TrackerConfig`, `createTracker()`, all trackers (BoT-SORT today) | no |
| `mot::pipeline` | `Pipeline` (any `IDetector` + any `ITracker`), strict YAML `PipelineConfig` | no |
| `mot::visualization` | `Annotator` for tracks and detections | no |
| `mot::tensorrt` | `TensorRTDetector` (implements `IDetector`) | yes |
| `mot::onnxruntime` | `OnnxRuntimeDetector` (implements `IDetector`, CPU execution provider) | no |
| `mot::mot` | umbrella over everything that was built | |

`mot_video` (in `cpp/apps/`) is the video CLI: video in, annotated MP4 and
VisDrone/MOT text out.

```
cpp/
├── include/mot/{core,detection,tracking,pipeline,tensorrt,onnxruntime,visualization}/  # public API
├── src/<module>/            # implementation + private headers (never installed)
│   ├── detection/yolo/      # letterbox preprocess + YOLO decode/NMS shared by both detectors
│   └── tracking/{common,botsort}/   # shared SORT building blocks, one folder per tracker
├── apps/mot_video/          # CLI
├── tests/<module>/          # CTest suites (tracker contract test covers every tracker)
├── benchmarks/
└── cmake/                   # helpers, FindTensorRT, FindOnnxRuntime, package config
```

All code is in namespace `mot`. To add a tracker, follow
[`cpp/src/tracking/README.md`](../cpp/src/tracking/README.md).

BoT-SORT: GMC state transforms and removed-track bookkeeping follow the vendored
Ultralytics 8.4.157 implementation; assignment uses the C++ Hungarian solver.

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

## Detector contract

Both detectors share the same contract and the same preprocessing and decode/NMS code
(`cpp/src/detection/yolo/`), so they produce the same detections up to numerical precision.

The detector accepts non-empty 2D BGR frames of type `CV_8UC3`. Frame sizes may
vary; letterboxing maps each frame to the engine's fixed `[1,3,H,W]` input,
which must match `DetectorConfig`. Both bindings must use linear FP32 device
storage, with one input and one raw YOLO output `[1,4+class_count,N]`, where
`N` is fixed by the engine. Dynamic shapes, other image types, FP16 bindings,
candidate-first output and embedded NMS output are rejected.

The current `weights/26leaf1.engine` has input `images` `[1,3,640,640]` and
output `output0` `[1,14,34000]`, both FP32. Building with `--fp16` enables FP16
computation internally; it does not require FP16 input/output bindings.
Tensor buffers on CPU/GPU and the letterbox canvas are allocated once during
construction and reused by `detect()`.

`OnnxRuntimeDetector` loads a `.onnx` model exported with `dynamic=False` and
`nms=False` (Ultralytics `end2end=False`), e.g. `weights/26c1.onnx`: input
`images` `[1,3,640,640]` and output `output0` `[1,14,34000]`, both FP32. It runs on
the ONNX Runtime CPU execution provider with full graph optimization; the fourth
constructor argument `intra_op_threads` (default 0 = ONNX Runtime's choice) sets the
thread count. Input/output tensors are bound to preallocated buffers, so
`detect()` does not allocate tensors per frame.

## Build

The target is Linux x86_64 with OpenCV 4, plus CUDA and TensorRT 8.6.x for the
TensorRT detector and/or the ONNX Runtime C/C++ SDK for the ONNX detector. Get the
SDK from the official prebuilt release (the Python `onnxruntime` wheel has no C++ headers):

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
  | sudo tar xz -C /opt   # → /opt/onnxruntime-linux-x64-1.23.2
```

Configure from the repository root or from `cpp/`:

```bash
cmake -S . -B /tmp/my-trackers-cpp-build \
  -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.23.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/my-trackers-cpp-build -j
ctest --test-dir /tmp/my-trackers-cpp-build --output-on-failure
```

`MOT_WITH_TENSORRT` and `MOT_WITH_ONNXRUNTIME` (both ON by default) are switched
off with a configure warning when their dependency is not found. `ONNXRUNTIME_ROOT`
(CMake or environment variable) is optional when the SDK is extracted as
`/opt/onnxruntime-linux-x64-<version>`. Use `-DMOT_WITH_TENSORRT=OFF` with ONNX Runtime
for a GPU-free build that still has a detector and `mot_video`; with both backends off
only the libraries and CPU tests are built (no `mot_video`). An installed `mot_video`
needs `libonnxruntime.so` on the loader path (e.g. `LD_LIBRARY_PATH`), as for TensorRT. Apps and tests are built by default only when this repository is
the top-level project. The tracking, tracker-contract, pipeline, configuration,
annotator, postprocess and performance-regression tests are CPU-only. With ONNX
Runtime, `mot_inference_onnx` (labels `onnxruntime;cpu`) runs `weights/26c1.onnx`
(override with `MOT_ONNX_MODEL_PATH`; optional `MOT_VIDEO_PATH` frame source) and is
skipped when the model is missing. To enable the separate TensorRT smoke test,
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
/tmp/my-trackers-cpp-build/cpp/apps/mot_video/mot_video data/input.ts \
  --config weights/config.yaml \
  --output-dir runs/video_inference_cpp
```

The CLI writes `<stem>_annotated.mp4` and `<stem>_predictions.txt`. The video
keeps the decoded input dimensions and FPS and uses `mp4v`. Prediction rows
are `frame,id,x,y,width,height,confidence,class,-1,-1`; frame and class IDs
are one-based. Incomplete inference only leaves temporary dot files, which
are removed on failure. Pass `--overwrite` to replace existing outputs.

`weights/config.yaml` stores the model path, class names and all detector/tracker
parameters. The tracker is selected by `tracker.type`; its parameters live in a
section of the same name:

```yaml
detector:
  model_path: "26leaf1.engine"   # .engine → TensorRT, .onnx → ONNX Runtime; optional for custom IDetectors
  class_names: [pedestrian, people, bicycle, ...]  # index = class_id
  # class_count: 10  # optional; defaults to the number of class_names
tracker:
  type: botsort
  botsort: { track_high_thresh: 0.40, track_buffer: 90, ... }
```

The older flat form (BoT-SORT keys directly under `tracker:` with no `type`) is
still accepted and read as BoT-SORT.
Runtime parameter values come from the selected YAML; benchmark comparisons must use the same file.
Run from the repository root, pass `--config /path/to/config.yaml`, or set
`MOT_CONFIG_PATH`. Priority is `--config`, then `MOT_CONFIG_PATH`, then
`weights/config.yaml`. CLI parameter flags override YAML regardless of argument
order. The detector backend is chosen from the model extension (`.onnx` → ONNX Runtime,
`.engine`/`.trt`/`.plan` → TensorRT); `mot_video --help` lists the backends built in.
`engine_path` is still accepted as the legacy name of `model_path` (setting both is an
error). Relative YAML model paths resolve against the config directory; a `--model`
(alias `--engine`) override resolves against the working directory. `--class-count`
overrides the number of detector classes. `--tracker TYPE` selects another tracker
(its defaults are used unless the YAML type matches). Both `detector` and `tracker` mappings
are required; omitted fields use C++ defaults. Unknown/duplicate keys, wrong types
and invalid settings fail with a message containing the config path.

`detector.class_names` may also be a path to an Ultralytics dataset YAML
(`names: {0: ..., 1: ...}`), resolved against the config directory.

`Annotator` is independent of inference. It takes the class names (from
`detector.class_names` or `mot::loadClassNames()`), uses a color per class, and
draws class names with track IDs. Missing class names fall back to `Class <id>`.
The CLI uses it for annotated video output.

All BoT-SORT flags are exposed by the CLI (they are rejected for other trackers): `--track-high-thresh`,
`--track-low-thresh`, `--new-track-thresh`, `--track-buffer`,
`--match-thresh`, `--fuse-score`/`--no-fuse-score`, `--gmc-method` and
`--gmc-downscale`.

## Using the library from another application

Either vendor the folder or install it:

```cmake
# Option A: source tree (only the libraries are built)
set(MOT_WITH_TENSORRT ON)            # OFF if you bring your own detector
set(MOT_WITH_ONNXRUNTIME ON)         # CPU ONNX detector; needs ONNXRUNTIME_ROOT
add_subdirectory(path/to/my-trackers/cpp mot)
# Option B: cmake --install <build> --prefix <prefix>, then
find_package(mot CONFIG REQUIRED)    # with CMAKE_PREFIX_PATH=<prefix>

target_link_libraries(my_app PRIVATE mot::mot)   # or individual mot::* targets
```

Detection and tracking from a YAML file:

```cpp
#include <mot/pipeline/config.hpp>
#include <mot/pipeline/pipeline.hpp>
#include <mot/tensorrt/detector.hpp>      // or <mot/onnxruntime/detector.hpp>
#include <mot/tracking/factory.hpp>

auto config = mot::PipelineConfig::fromFile("weights/config.yaml");
mot::Pipeline pipeline(
    std::make_unique<mot::TensorRTDetector>(config.model_path, config.detector, config.class_names),
    // or std::make_unique<mot::OnnxRuntimeDetector>(config.model_path, config.detector, config.class_names)
    mot::createTracker(config.tracker));

// Each call is synchronous and owns state for this stream.
std::vector<mot::TrackOutput> tracks = pipeline.process(bgr_frame);
// Start a new stream without reloading the engine.
pipeline.reset();
// Switch algorithm, keeping the loaded detector.
pipeline.setTracker(mot::createTracker(mot::defaultTrackerConfig("botsort")));
```

Any detector can be used by implementing `mot::IDetector`
(`detect(const cv::Mat&)` and `classNames()`), and trackers can be used on their own:

```cpp
auto tracker = mot::createTracker(mot::BoTSORTConfig{});
auto tracks = tracker->update(detections, bgr_frame);
```

Rendering (link `mot::visualization`):

```cpp
#include <mot/visualization/annotator.hpp>

mot::Annotator annotator(config.class_names);
annotator.draw(annotated, tracks);
// For detection-only callers: annotator.drawDetections(annotated, detections);
```

`Pipeline` and tracker instances are stateful and must not be called
concurrently. Use one instance per camera stream. Camera capture, queues,
backpressure and evaluator integration remain outside the library.

## Performance benchmarks

Optimizations preserve the synchronous processing order and public detector/tracker/pipeline/annotator APIs.
Tracker assignment still uses the same padded Hungarian problem and tie rules. NMS keeps its class-aware
ordering, thresholds, 30000-candidate cap and clipping order. No runtime configuration values are changed.
Scratch buffers are instance-owned. GMC uses separate current/history images; detector transfers use pinned
host memory with one startup warning and pageable fallback if pinned allocation runs out of memory.

Build optional benchmarks (off by default):

```bash
cmake -S . -B /tmp/my-trackers-perf -DCMAKE_BUILD_TYPE=Release \
  -DMOT_BUILD_BENCHMARKS=ON -DMOT_BUILD_GPU_TESTS=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/my-trackers-perf -j4
/tmp/my-trackers-perf/cpp/benchmarks/mot_benchmark_cpu --config weights/config.yaml > cpu.csv
/tmp/my-trackers-perf/cpp/benchmarks/mot_benchmark_video --config weights/config.yaml \
  --video data/input.ts --output-dir /tmp/mot-benchmark-video > video.csv
```

The CPU target also builds with `-DMOT_WITH_TENSORRT=OFF`. Both commands default to 100 warm-up iterations,
1000 measured samples and 3 independent repeats; override with `--warmup`, `--frames`, and `--repeats`.
The CPU target measures rectangular/square assignment, raw NMS (including above-cap overlap cases), trackers at three object counts, and GMC.
Use `--only WORKLOAD`, e.g. `--only gmc_640x360` or `--only nms_overlap_34000`, for targeted CPU reruns.
Its tracker workloads intentionally disable GMC to isolate association/Kalman; the separate GMC workload
uses the configured method/downscale. Synthetic detections are not a tracking-accuracy benchmark.

The video target processes the first warm-up + measured frames from the start on each repeat with new detector,
tracker and GMC state. It measures decode, detector (including transfers/pre/postprocessing), tracker,
annotation, video writes, detection+tracking and combined per-frame latency. It annotates in place in both
baseline and optimized measurements and writes `benchmark_1.mp4` etc. in the output directory (overwritten on
rerun); it does not write MOT prediction text. Writer finalization time is reported separately on stderr.
Each CSV row includes sample count, mean, median, nearest-rank p95, throughput and a CPU workload checksum.
Short video inputs produce an explicit insufficient-samples warning; zero measured frames fail.

Run before/after benchmarks sequentially, with the same Release toolchain, hardware, engine, YAML and video.
Avoid concurrent builds or benchmarks when collecting comparison numbers. Do not interpret input video FPS
metadata as processing throughput, or the detection+tracking stage as complete application throughput.

See [CPP_PERFORMANCE.md](CPP_PERFORMANCE.md) for this change's measurements and validation limits.
