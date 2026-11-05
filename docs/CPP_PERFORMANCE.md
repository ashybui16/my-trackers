# C++ optimization measurements — 2026-11-02

## Conditions

Intel Core i7-8750H, NVIDIA GTX 1050 Ti, driver 580.178.04, OpenCV 4.6.0,
TensorRT 8.6.1.6 and GCC 13.3.0; both builds use CMake Release. CPU/GPU clocks and
OpenCV thread count use system defaults. Results describe this machine and input, not a general FPS guarantee.

The pre-optimization working tree was copied to `/tmp/my-trackers-opt-baseline` before changing runtime code.
Both variants use the same `26leaf1.engine` and the same `weights/config.yaml`:

```yaml
# Paths are relative to this configuration file. CLI flags override these values.
detector:
  engine_path: "26leaf1.engine"
  input_width: 640
  input_height: 640
  class_count: 10
  conf_threshold: 0.20
  iou_threshold: 0.70
  max_det: 300

tracker:
  track_high_thresh: 0.40
  track_low_thresh: 0.20
  new_track_thresh: 0.70
  track_buffer: 90
  match_thresh: 0.70
  fuse_score: true
  gmc_method: "sparseOptFlow"
  gmc_downscale: 2
```

Config SHA-256: `161411dc311bf3da1bdbbb947d75e3235c977657e1ec59286650de1263ea0ca8`.
Engine SHA-256: `ddc81d1cc405371bb282da324bfefd4702418f57c12dcbfcca7cd1de8e162f30`.
The YAML, class-name file and annotator implementation were not modified.

Initial exploratory runs included concurrent work and are excluded from the comparison below.
Final comparisons ran baseline/optimized sequentially in three pairs, without concurrent builds/benchmarks.
CPU workloads use 100 warm-up + 100 measured iterations per repeat; video uses 100 warm-up + 1000 measured
frames per repeat, starting from the beginning with fresh detector/tracker/GMC state.
The video is `data/input.ts`, 1280x720 at 25 FPS. Each run processed all 1100 requested frames.

## CPU workloads

Values below are the median of three per-repeat median latencies. The reduction column compares latency,
not whole-application FPS. Tracker workloads disable GMC to isolate association/Kalman and include missed detections/LOST tracks.

| Workload | Baseline median ms | Optimized median ms | Latency reduction |
|---|---:|---:|---:|
| assignment_16x16 | 0.0816 | 0.0638 | 21.8% |
| assignment_64x96 | 11.0310 | 8.8783 | 19.5% |
| assignment_128x128 | 38.2483 | 32.5598 | 14.9% |
| nms_128 | 0.0367 | 0.0262 | 28.8% |
| nms_2048 | 2.8298 | 1.8007 | 36.4% |
| nms_34000 | 50.3242 | 31.8170 | 36.8% |
| tracker_no_gmc_16 | 0.3192 | 0.3073 | 3.7% |
| tracker_no_gmc_64 | 4.1418 | 3.4765 | 16.1% |
| tracker_no_gmc_128 | 25.4841 | 21.5554 | 15.4% |

The short CPU batch showed a GMC slowdown, so GMC was investigated separately rather than counted as a speedup.
A dedicated 100-warm-up + 1000-sample measurement alternated baseline/optimized order over three repeats:
640x360 GMC median 3.0215 -> 2.9892 ms. The 1280x720 targeted measurements were also approximately neutral. Treat GMC changes as reduced buffer
allocation/copying; there is no material measured GMC speedup and no allocator-count/RSS measurement is claimed.
Both the original short batch and follow-up raw data are retained below.

## Real video

For each column, take the median of that metric across the three repeats. Stage mean, stage p95 and throughput
are separate statistics; throughput is not computed from the median latency.

| Stage | Baseline mean ms | Optimized mean ms | Baseline p95 ms | Optimized p95 ms | Baseline FPS | Optimized FPS |
|---|---:|---:|---:|---:|---:|---:|
| decode | 1.452 | 1.378 | 2.349 | 2.182 | 688.633 | 725.680 |
| detector | 41.152 | 39.987 | 42.932 | 40.956 | 24.300 | 25.008 |
| tracker | 7.545 | 7.287 | 10.432 | 9.873 | 132.545 | 137.233 |
| annotation | 0.117 | 0.115 | 0.274 | 0.257 | 8581.226 | 8663.433 |
| write | 10.543 | 10.400 | 13.698 | 12.988 | 94.847 | 96.151 |
| detection_tracking | 48.696 | 47.197 | 52.960 | 50.195 | 20.535 | 21.188 |
| end_to_end | 60.809 | 59.092 | 67.435 | 64.176 | 16.445 | 16.923 |

Measured video throughput: 16.445 -> 16.923 FPS (2.9% increase).
The engine and output encoding dominate this sample, so synthetic CPU gains do not translate directly to the
same percentage in video throughput. Detector timing includes preprocessing, transfers, GPU execution and NMS;
it does not isolate GPU kernel latency.

This benchmark annotates in place in both variants, includes MP4 writes, excludes prediction-text output and
reports writer finalization separately. It is not a full CLI wall-clock measurement and does not measure the
CLI frame-clone saving. GPU/video results were collected before the final full-sort retention below; the change
only affects the above-30000 candidate branch and final parity validation was repeated afterwards.

## Rejected optimization and capped NMS

`partial_sort` was rejected after measuring its regression on tied-score, heavily overlapping inputs:
30001 candidates took about 2.001 ms in baseline versus 4.224 ms with partial sort.
The final code retains full sort and the exact original top-30000/tie rules, alongside cached areas and reused scratch.

Targeted measurements use 100 warm-up + 300 measured samples, three repeats:

| Candidates, all overlapping | Baseline median ms | Final median ms |
|---|---:|---:|
| 30001 | 2.001 | 1.795 |
| 34000 | 2.211 | 1.959 |
| 100000 | 6.590 | 5.875 |

Reproduce the input shape with the optional CPU target's `nms_overlap_30001`, `nms_overlap_34000` and
`nms_overlap_100000` workloads, e.g. `--only nms_overlap_34000 --frames 300`. The temporary targeted harness used
an upper-middle median for even sample counts; the shipped benchmark reports the average of the two middle values.

## Correctness validation

- All 6 tests pass with CUDA/TensorRT, including real-GPU detector/pipeline smoke tests.
- A separate `MOT_BUILD_INFERENCE=OFF` Release build and all 5 CPU tests pass.
- New regression coverage: exhaustive small assignment optima, threshold/tie cases, strided/changing-size
  preprocessing, reusable NMS scratch/candidate cap, Kalman input ownership, signed/invalid/overflow GMC rollback,
  retained GMC history and reset. Existing detector/track lifecycle goldens also pass.
- A final 250-frame trace matches baseline byte-for-byte: detections, all tracked/lost/removed state,
  means/covariances and annotation pixel hashes. This is evidence for these inputs, not a universal bitwise requirement.
- Actual baseline/optimized CLIs process the same 120-frame fixture: MOT prediction files match byte-for-byte;
  all 120 decoded output frames match; dimensions, frame count and FPS metadata match (1280x720, 25 FPS).
- Forced `cudaMallocHost` allocation failure through a temporary interposition library exercises pageable fallback;
  the real-GPU smoke/reset tests pass. One warning is emitted per detector instance.
- No public detector/tracker/pipeline/annotator API, YAML schema, engine or runtime parameter changes.

## Evidence

Raw CSV files and environment/CLI metadata are in [benchmarks/](benchmarks/).
The files named `optimized-targeted.csv` record the discarded partial-sort experiment;
`optimized-targeted-final.csv` records the retained full-sort implementation.
Temporary full traces, frame hashes and generated videos are under `/tmp/my-trackers-opt-results` on this machine;
large video/trace artifacts are not committed to the repository.
