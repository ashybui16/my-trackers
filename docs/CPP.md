# C++ TensorRT video pipeline

PLAN2 adds a C++17 implementation under `cpp/`:

- `mot_tracker_cpp`: detector-agnostic BoT-SORT. GMC state transforms and
  removed-track bookkeeping follow the vendored Ultralytics 8.4.157 implementation;
  assignment continues to use the existing C++ Hungarian solver.
- `mot_inference_cpp`: RAII TensorRT detector and synchronous `MotPipeline`.
- `mot_pipeline_config_cpp`: strict YAML configuration loader, usable without CUDA.
- `mot_annotator_cpp`: independent `MotAnnotator` for tracks and detections, usable without CUDA.
- `mot_video`: video input, annotated MP4 output and VisDrone/MOT text output.

Implementation files are grouped under `cpp/src/detector/`, `cpp/src/tracker/`,
`cpp/src/pipeline/`, `cpp/src/annotator/` and `cpp/src/apps/`. Tests are grouped
under `cpp/tests/detector/`, `cpp/tests/tracker/`, `cpp/tests/pipeline/` and
`cpp/tests/annotator/`. Public headers live under `cpp/include/mot_detector/`,
`cpp/include/mot_tracker/`, `cpp/include/mot_pipeline/` and
`cpp/include/mot_annotator/`. The postprocess header is private and lives in
`cpp/internal/detector/`.

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
construction and reused by `infer()`.

## Build

The target is Linux x86_64 with OpenCV 4, CUDA and TensorRT 8.6.x. Configure
from the repository root or from `cpp/`:

```bash
cmake -S . -B /tmp/my-trackers-cpp-build \
  -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DMOT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/my-trackers-cpp-build -j
ctest --test-dir /tmp/my-trackers-cpp-build --output-on-failure
```

The tracker, configuration, annotator, detector postprocess and performance-regression tests are CPU-only. To enable the separate TensorRT smoke test,
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
/tmp/my-trackers-cpp-build/cpp/mot_video data/input.ts \
  --config weights/config.yaml \
  --output-dir runs/video_inference_cpp
```

The CLI writes `<stem>_annotated.mp4` and `<stem>_predictions.txt`. The video
keeps the decoded input dimensions and FPS and uses `mp4v`. Prediction rows
are `frame,id,x,y,width,height,confidence,class,-1,-1`; frame and class IDs
are one-based. Incomplete inference only leaves temporary dot files, which
are removed on failure. Pass `--overwrite` to replace existing outputs.

`weights/config.yaml` stores the engine path and all detector/tracker parameters.
Runtime parameter values come from the selected YAML; benchmark comparisons must use the same file.
Run from the repository root, pass `--config /path/to/config.yaml`, or set
`MOT_CONFIG_PATH`. Priority is `--config`, then `MOT_CONFIG_PATH`, then
`weights/config.yaml`. CLI parameter flags override YAML regardless of argument
order. Relative YAML engine paths resolve against the config directory; an
`--engine` override resolves against the working directory. `--class-count`
overrides the number of detector classes. Both `detector` and `tracker` mappings
are required; omitted fields use C++ defaults. Unknown/duplicate keys, wrong types
and invalid settings fail with a message containing the config path.

`MotAnnotator` is independent of inference. It reads class names from
`visdrone.yaml` beside the engine, uses a color per class, and draws class names
with track IDs. Missing class names fall back to `Class <id>`. The CLI uses it
for annotated video output.

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

auto config = mot_inference::PipelineConfig::fromFile("weights/config.yaml");
mot_inference::MotPipeline pipeline(config.engine_path, config);

// Each call is synchronous and owns state for this stream.
std::vector<mot_tracker::TrackOutput> tracks = pipeline.processFrame(bgr_frame);
// Start a new stream without reloading the TensorRT engine.
pipeline.reset();
```

Use the annotator separately when rendering is needed (link `mot::annotator`):

```cpp
#include "mot_annotator/annotation.hpp"

mot_inference::MotAnnotator annotator(config.engine_path, config.detector.class_count);
cv::Mat annotated = bgr_frame.clone();
annotator.draw(annotated, tracks);
// For detection-only callers: annotator.drawDetections(annotated, detections);
```

`MotPipeline` and `BoTSORT` instances are stateful and must not be called
concurrently. Use one instance per camera stream. Camera capture, queues,
backpressure and evaluator integration remain outside v1.

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
/tmp/my-trackers-perf/cpp/mot_benchmark_cpu --config weights/config.yaml > cpu.csv
/tmp/my-trackers-perf/cpp/mot_benchmark_video --config weights/config.yaml \
  --video data/input.ts --output-dir /tmp/mot-benchmark-video > video.csv
```

The CPU target also builds with `-DMOT_BUILD_INFERENCE=OFF`. Both commands default to 100 warm-up iterations,
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
