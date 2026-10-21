# BoT-SORT Improvement Roadmap

## 1. Mục tiêu

Xây dựng một phiên bản BoT-SORT cải tiến theo lộ trình:

1. Dùng implementation hiện có để tạo **baseline đáng tin cậy**.
2. Prototype và thử nghiệm ý tưởng mới bằng **Python**.
3. Tách tracker khỏi framework để có một implementation **độc lập, modular, dễ benchmark**.
4. Viết test để đảm bảo behavior ổn định trước khi tối ưu.
5. Benchmark accuracy + runtime.
6. Port từng module sang **C++** gần như 1:1.
7. Tích hợp vào pipeline realtime.

Mục tiêu cuối cùng:

```text
Detector
   ↓
Detections
   ↓
Custom BoT-SORT
   ↓
Tracks
   ↓
Realtime C++ Pipeline
```

---

## 2. Chiến lược codebase

### 2.1. Original BoT-SORT repo

Vai trò:

- Reference implementation của paper.
- Dùng để kiểm tra behavior.
- Dùng để đối chiếu:
  - association
  - Kalman filter
  - GMC
  - ReID
  - track lifecycle
  - lost/reactivate logic

Không nên dùng làm codebase chính lâu dài vì dependency và cấu trúc tương đối cũ.

### 2.2. Ultralytics BoT-SORT

Vai trò:

- Baseline thực dụng.
- Tích hợp nhanh với YOLO.
- Dễ chạy thử trên video/dataset.
- Dễ tạo benchmark ban đầu.

Không nên sửa trực tiếp sâu trong source Ultralytics.

### 2.3. Custom implementation

Đây sẽ là codebase chính để nghiên cứu.

Luồng đề xuất:

```text
Ultralytics BoT-SORT
        ↓
baseline
        ↓
extract/simplify
        ↓
Custom Python BoT-SORT
        ↓
experiment
        ↓
benchmark
        ↓
C++ implementation
```

---

# 3. Kiến trúc repository đề xuất

```text
botsort-research/
│
├── README.md
├── PLAN.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   ├── experiment_a.yaml
│   └── experiment_b.yaml
│
├── src/
│   └── botsort/
│       ├── __init__.py
│       ├── tracker.py
│       ├── track.py
│       ├── types.py
│       ├── kalman_filter.py
│       ├── association.py
│       ├── assignment.py
│       ├── gmc.py
│       ├── reid.py
│       ├── geometry.py
│       └── config.py
│
├── detectors/
│   ├── base.py
│   └── ultralytics_yolo.py
│
├── experiments/
│   ├── run_video.py
│   ├── run_dataset.py
│   ├── benchmark_accuracy.py
│   └── benchmark_runtime.py
│
├── tests/
│   ├── test_kalman.py
│   ├── test_iou.py
│   ├── test_assignment.py
│   ├── test_track_state.py
│   ├── test_association.py
│   └── test_tracker_sequence.py
│
├── tools/
│   ├── visualize_tracks.py
│   ├── export_detections.py
│   └── compare_runs.py
│
├── results/
│
└── cpp/
    ├── CMakeLists.txt
    ├── include/
    │   └── botsort/
    │       ├── tracker.hpp
    │       ├── track.hpp
    │       ├── types.hpp
    │       ├── kalman_filter.hpp
    │       ├── association.hpp
    │       ├── assignment.hpp
    │       └── gmc.hpp
    │
    ├── src/
    │   ├── tracker.cpp
    │   ├── track.cpp
    │   ├── kalman_filter.cpp
    │   ├── association.cpp
    │   ├── assignment.cpp
    │   └── gmc.cpp
    │
    └── tests/
```

---

# 4. API mục tiêu

Ngay từ Python nên thiết kế API gần với C++.

## Detection

```python
@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    cls: int
    feature: np.ndarray | None = None
```

Convention:

```text
bbox = [x1, y1, x2, y2]
```

Phải thống nhất convention ngay từ đầu.

---

## Track

```python
@dataclass
class TrackOutput:
    track_id: int
    bbox: np.ndarray
    score: float
    cls: int
    state: str
```

---

## Tracker

```python
class BoTSORT:
    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[TrackOutput]:
        ...
```

C++ tương ứng:

```cpp
std::vector<TrackOutput> BoTSORT::update(
    const std::vector<Detection>& detections,
    const cv::Mat& frame
);
```

---

# 5. Phase 0 — Reproduce baseline

## Mục tiêu

Có một baseline đáng tin cậy trước khi thay đổi thuật toán.

## Task

- [ ] Clone/run Ultralytics.
- [ ] Chạy BoT-SORT mặc định trên video test.
- [ ] Lưu output tracking.
- [ ] Kiểm tra ID visually.
- [ ] Chạy trên dataset nhỏ.
- [ ] Ghi lại config tracker.
- [ ] Đo FPS.
- [ ] Đo accuracy nếu dataset có ground truth.

## Output

```text
results/baseline/
├── config.yaml
├── metrics.json
├── runtime.json
└── demo.mp4
```

## Exit criteria

- Baseline chạy deterministic hoặc sai khác nằm trong mức hiểu được.
- Có ít nhất một video regression test.
- Có metric baseline để so sánh về sau.

---

# 6. Phase 1 — Tách detector khỏi tracker

Tracker không được phụ thuộc trực tiếp vào YOLO.

Không nên:

```python
tracker.update(yolo_results)
```

Nên:

```python
detections = detector(frame)
tracks = tracker.update(detections, frame)
```

## Interface

```python
class Detector:
    def detect(self, frame) -> list[Detection]:
        ...
```

## Mục tiêu

BoT-SORT chỉ biết:

```text
bbox
confidence
class
optional embedding
```

Không biết Ultralytics `Results`.

## Exit criteria

- Có thể thay detector mà không sửa tracker.
- Có thể replay detection từ file mà không chạy detector.

---

# 7. Phase 2 — Custom Python BoT-SORT

Reimplement logic cần thiết thành package riêng.

Thứ tự đề xuất:

1. Geometry / IoU.
2. Hungarian assignment.
3. Kalman filter.
4. Track object.
5. Track state machine.
6. First association.
7. Second association.
8. Lost/reactivation.
9. GMC.
10. ReID.

Không rewrite tất cả cùng lúc.

---

# 8. Phase 3 — Unit testing

Đây là phase quan trọng trước khi nghiên cứu cải tiến.

## 8.1. IoU

Test:

```text
same bbox       → IoU = 1
no overlap      → IoU = 0
partial overlap → known value
```

---

## 8.2. Kalman filter

Test:

- initiate
- predict
- update
- multi-step prediction
- covariance remains valid
- compare Python baseline implementation

---

## 8.3. Hungarian assignment

Test cost matrix:

```text
simple 2x2
unmatched track
unmatched detection
threshold filtering
```

---

## 8.4. Track lifecycle

State machine:

```text
New
 ↓
Tracked
 ↓
Lost
 ↓
Tracked

Lost
 ↓
Removed
```

Test:

- new track creation
- lost track
- reactivation
- expiration
- duplicate prevention

---

## 8.5. Sequence regression test

Input:

```text
detections frame 1
detections frame 2
detections frame 3
...
```

Expected:

```text
track IDs
track states
bbox predictions
```

Đây sẽ là test cực kỳ quan trọng khi port sang C++.

---

# 9. Phase 4 — Modular association

Không hard-code toàn bộ logic association trong tracker.

Interface:

```python
class AssociationStrategy:
    def compute_cost(
        self,
        tracks,
        detections,
        context=None,
    ) -> np.ndarray:
        ...
```

Implementations có thể gồm:

```text
IoUAssociation
MotionAssociation
ReIDAssociation
IoUReIDAssociation
AdaptiveAssociation
OcclusionAwareAssociation
CustomAssociation
```

---

# 10. Các hướng cải tiến có thể thử

## 10.1. Adaptive association weight

Thay vì:

```text
cost = constant_a * IoU + constant_b * ReID
```

Dùng:

```text
cost =
    w_motion(track, detection, scene) * motion_cost
  + w_appearance(track, detection, scene) * appearance_cost
```

Weight có thể phụ thuộc:

- occlusion level
- detection confidence
- track age
- velocity uncertainty
- bbox scale
- camera motion
- ReID confidence

---

## 10.2. Occlusion-aware association

Phân biệt:

```text
visible track
partially occluded
heavily occluded
```

Khi occlusion cao:

- giảm trust IoU
- tăng trust motion / appearance
- tăng lost buffer hợp lý

---

## 10.3. Motion uncertainty

Dùng covariance từ Kalman để điều chỉnh association.

Ví dụ:

```text
high uncertainty
→ gating rộng hơn

low uncertainty
→ gating chặt hơn
```

Có thể thử Mahalanobis distance.

---

## 10.4. Adaptive track buffer

Thay vì `track_buffer` cố định:

```text
track_buffer = f(
    track_velocity,
    confidence,
    occlusion,
    object_size
)
```

---

## 10.5. Better feature management

Không chỉ lưu feature cuối.

Có thể thử:

```text
EMA feature
feature gallery
top-k recent features
confidence-weighted feature
```

---

## 10.6. Scene-aware GMC

Không phải frame nào cũng cần GMC.

Có thể:

```text
estimate camera motion
        ↓
if motion significant:
    apply GMC
else:
    skip
```

Mục tiêu:

- giảm compute
- tránh GMC noise khi camera static

---

# 11. Experiment framework

Mỗi experiment phải có config riêng.

Ví dụ:

```yaml
experiment: adaptive_association_v1

tracker:
  high_thresh: 0.6
  low_thresh: 0.1
  match_thresh: 0.8
  track_buffer: 30

association:
  type: adaptive

reid:
  enabled: true

gmc:
  enabled: true
```

Kết quả:

```text
results/
└── adaptive_association_v1/
    ├── config.yaml
    ├── metrics.json
    ├── runtime.json
    ├── log.txt
    └── videos/
```

---

# 12. Metrics

Không đánh giá tracker chỉ bằng FPS.

## Accuracy

Ưu tiên:

```text
HOTA
IDF1
MOTA
IDs
AssA
DetA
```

Quan trọng nhất khi cải tiến identity:

```text
IDF1
IDs
AssA
```

---

## Runtime

Đo riêng:

```text
Detector time
ReID time
GMC time
Association time
Kalman time
Total tracker time
Total pipeline time
```

Ví dụ:

```text
Detector      8.2 ms
ReID          2.3 ms
GMC           0.8 ms
Association   0.5 ms
Tracker       1.6 ms
Total        12.1 ms
```

Không chỉ ghi FPS tổng.

---

# 13. Profiling

Python:

```text
cProfile
line_profiler
py-spy
torch profiler
```

C++:

```text
perf
VTune
Nsight Systems
Nsight Compute
```

Mục tiêu là xác định bottleneck trước khi optimize.

---

# 14. Phase 5 — Benchmark matrix

Mỗi cải tiến cần ablation rõ ràng.

Ví dụ:

| Experiment | GMC | ReID | Adaptive Assoc | HOTA | IDF1 | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | ✓ | ✓ | ✗ | | | | |
| A | ✓ | ✗ | ✗ | | | | |
| B | ✗ | ✓ | ✗ | | | | |
| C | ✓ | ✓ | ✓ | | | | |

Không nên thay 4 thứ cùng lúc.

---

# 15. Phase 6 — Chuẩn bị port C++

Trước khi port:

- [ ] Python implementation độc lập với Ultralytics.
- [ ] Unit test coverage đủ tốt.
- [ ] Có regression sequence.
- [ ] Input/output interface ổn định.
- [ ] Config ổn định.
- [ ] Có metric baseline.
- [ ] Đã profile bottleneck.

---

# 16. Mapping Python → C++

| Python | C++ |
|---|---|
| NumPy | Eigen |
| OpenCV Python | OpenCV C++ |
| scipy LAP/Hungarian | LAPJV / custom solver |
| dataclass | struct/class |
| list | std::vector |
| numpy matrix | Eigen::Matrix |
| ONNX Runtime Python | ONNX Runtime C++ |
| TensorRT Python | TensorRT C++ |

---

# 17. C++ architecture

```text
Detector
   ↓
std::vector<Detection>
   ↓
BoTSORT::update()
   │
   ├── KalmanFilter
   ├── Association
   ├── Assignment
   ├── GMC
   └── ReID
   ↓
std::vector<TrackOutput>
```

Không để tracker biết detector implementation.

---

# 18. C++ API đề xuất

```cpp
struct Detection {
    cv::Rect2f bbox;
    float score;
    int class_id;
    std::vector<float> feature;
};

struct TrackOutput {
    int track_id;
    cv::Rect2f bbox;
    float score;
    int class_id;
};

class BoTSORT {
public:
    std::vector<TrackOutput> update(
        const std::vector<Detection>& detections,
        const cv::Mat& frame
    );
};
```

---

# 19. Cross-language regression test

Cùng một detection sequence:

```text
detections.json
```

Chạy:

```text
Python tracker
C++ tracker
```

So sánh từng frame:

```text
track id
state
bbox
matching pairs
Kalman state
```

Cho phép numerical tolerance nhỏ:

```text
abs(py - cpp) < epsilon
```

Đây là cách tốt nhất để phát hiện lỗi khi port.

---

# 20. Optimization sau khi port

Chỉ optimize sau khi behavior Python/C++ tương đương.

Thứ tự:

1. Algorithm correctness.
2. Memory allocation.
3. Data copy.
4. Matrix operations.
5. ReID inference.
6. GMC.
7. Parallelization.
8. GPU optimization.

Không optimize quá sớm.

---

# 21. Realtime considerations

Theo dõi:

```text
latency
throughput
memory
GPU utilization
CPU utilization
frame queue
```

Pipeline có thể:

```text
Capture
  ↓
Detector
  ↓
Tracker
  ↓
Output
```

Hoặc async:

```text
Capture Thread
      ↓
Detection Worker
      ↓
Tracking Worker
      ↓
Render / downstream
```

Tracker nên giữ state ở một thread cố định nếu có thể.

---

# 22. Definition of Done

## Python research version

- [ ] Không phụ thuộc vào Ultralytics tracker internals.
- [ ] Detector-independent.
- [ ] Modular association.
- [ ] Modular ReID.
- [ ] Modular GMC.
- [ ] Unit tested.
- [ ] Dataset benchmark.
- [ ] Runtime profiling.
- [ ] Reproducible experiment configs.

## C++ version

- [ ] Behavior match Python.
- [ ] Cross-language regression test pass.
- [ ] Build bằng CMake.
- [ ] OpenCV/Eigen dependencies rõ ràng.
- [ ] Không memory leak.
- [ ] Runtime đạt target.
- [ ] Tích hợp được detector production.
- [ ] Có benchmark end-to-end.

---

# 23. Recommended milestone order

```text
M0
Ultralytics baseline

 ↓

M1
Detector-independent Python tracker interface

 ↓

M2
Custom Python BoT-SORT baseline

 ↓

M3
Regression + unit tests

 ↓

M4
Research improvements

 ↓

M5
Accuracy/runtime benchmark

 ↓

M6
Freeze algorithm

 ↓

M7
C++ port

 ↓

M8
Python/C++ regression parity

 ↓

M9
Realtime optimization

 ↓

M10
Production integration
```

---

# 24. Nguyên tắc quan trọng

## Rule 1

Không cải tiến thuật toán khi baseline chưa reproduce được.

## Rule 2

Không port sang C++ trước khi Python algorithm tương đối ổn định.

## Rule 3

Không phụ thuộc detector framework trong tracker core.

## Rule 4

Không optimize trước khi profile.

## Rule 5

Mỗi cải tiến phải có ablation.

## Rule 6

Luôn giữ một regression dataset nhỏ để test nhanh.

## Rule 7

Python và C++ nên dùng cùng:

```text
bbox convention
threshold meaning
track state
association order
config names
```

---

# 25. Bước đầu tiên nên làm

Ngay bây giờ:

```text
1. Tạo repository botsort-research.

2. Chạy Ultralytics BoT-SORT baseline.

3. Lưu detections của một video thành file.

4. Tạo interface:

   Detection
   TrackOutput
   BoTSORT.update()

5. Replay detections qua tracker.

6. Viết custom IoU + assignment + Track lifecycle.

7. So sánh output với baseline.

8. Sau khi match behavior:
   bắt đầu experiment cải tiến.
```

---

# 26. Target architecture cuối cùng

```text
                    ┌──────────────┐
                    │   Detector   │
                    └──────┬───────┘
                           │
                      Detections
                           │
                    ┌──────▼───────┐
                    │   BoT-SORT   │
                    │              │
                    │ ┌──────────┐ │
                    │ │ Motion   │ │
                    │ └──────────┘ │
                    │ ┌──────────┐ │
                    │ │ ReID     │ │
                    │ └──────────┘ │
                    │ ┌──────────┐ │
                    │ │ GMC      │ │
                    │ └──────────┘ │
                    │ ┌──────────┐ │
                    │ │ Assoc.   │ │
                    │ └──────────┘ │
                    └──────┬───────┘
                           │
                         Tracks
                           │
             ┌─────────────▼─────────────┐
             │      Realtime Pipeline    │
             └───────────────────────────┘
```

---

# 27. Kết luận

Chiến lược đề xuất:

```text
Original BoT-SORT
      ↓
reference

Ultralytics BoT-SORT
      ↓
baseline

Custom Python BoT-SORT
      ↓
research + experiments

Frozen algorithm
      ↓
C++ port

C++ optimized implementation
      ↓
Realtime production pipeline
```

Không nên lựa chọn giữa "Ultralytics" và "tự implement" như hai hướng loại trừ nhau.

Hướng hiệu quả nhất là:

> dùng Ultralytics để có baseline nhanh, sau đó xây dựng custom implementation độc lập theo từng module, kiểm chứng bằng regression tests, thử nghiệm cải tiến trong Python, rồi port implementation đã ổn định sang C++.
