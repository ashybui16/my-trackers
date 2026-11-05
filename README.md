# My Trackers

`my-trackers` là dự án theo dõi đa đối tượng (Multi-Object Tracking — MOT) dựa trên **BoT-SORT**, được tách thành một package Python độc lập với detector. Tracker nhận các detection đã chuẩn hóa, quản lý vòng đời track và trả về ID ổn định theo từng frame.

Repo đồng thời cung cấp:

- package `mot_tracker` không phụ thuộc Ultralytics hoặc PyTorch ở runtime;
- CLI chạy YOLO + BoT-SORT trên video;
- pipeline cache detection, tracking, tuning và đánh giá trên VisDrone-MOT;
- bộ test cho hình học, LAP assignment, Kalman filter, GMC và track lifecycle.

> Phiên bản hiện tại là `0.1.0`, yêu cầu Python 3.10 trở lên và chỉ dành cho sử dụng nội bộ.

## Tính năng chính

- API độc lập với detector qua `Detection` và `TrackOutput`.
- Kalman filter theo trạng thái XYWH.
- Ghép IoU hai tầng cho detection có độ tin cậy cao/thấp.
- Score fusion và LAPJV assignment.
- Global Motion Compensation bằng sparse optical flow hoặc có thể tắt hoàn toàn.
- Quản lý track active, lost, reactivated và removed.
- Mỗi tracker có bộ đếm ID riêng và hỗ trợ `reset()`.
- Xuất prediction theo định dạng VisDrone-MOT và video đã vẽ bounding box.
- Tuning tham số bằng Optuna; đánh giá HOTA, IDF1 và MOTA bằng TrackEval hoặc evaluator nội bộ.

## C++ TensorRT implementation

PLAN2 adds a C++17 library (`mot::tracking`, `mot::pipeline`, `mot::tensorrt`, ...) with pluggable trackers, and the `mot_video` CLI.
Build and embedding instructions are in [docs/CPP.md](docs/CPP.md); the
Python tracker remains the numerical oracle.

## Cấu trúc repo

```text
.
├── src/mot_tracker/                  # Package BoT-SORT độc lập
│   ├── tracker.py                    # Điều phối tracking và track lifecycle
│   ├── track.py                      # Trạng thái của từng track
│   ├── kalman_filter.py              # Kalman filter XYWH
│   ├── association.py                # Ghép track với detection
│   ├── assignment.py                 # LAPJV assignment
│   ├── geometry.py                   # IoU và xử lý bounding box
│   ├── gmc.py                        # Global Motion Compensation
│   ├── config.py                     # BoTSORTConfig
│   └── types.py                      # Detection và TrackOutput
├── tools/
│   ├── infer_video_standalone.py     # YOLO predict -> mot_tracker -> video/MOT output
│   ├── visdrone_mot_standalone.py    # Pipeline VisDrone dùng tracker độc lập
│   ├── mot_tracker_adapter.py        # Adapter detector cache/config
│   ├── infer_video.py                # Baseline Ultralytics lịch sử
│   └── visdrone_mot.py               # Baseline/evaluator và cache helpers
├── tests/                            # Unit và regression tests
├── cpp/                              # C++17 tracker, TensorRT pipeline và CLI
├── docs/                             # Kế hoạch và ghi chú triển khai
├── pyproject.toml                    # Metadata/dependency của package lõi
└── requirements-mot.txt              # Dependency cho benchmark VisDrone
```

## Cài đặt

### Chỉ dùng tracker độc lập

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Package lõi sử dụng NumPy, OpenCV và `lap`; các dependency này được khai báo trong `pyproject.toml`. Solver ghép dùng `lap.lapjv(extend_cost=True, cost_limit=...)` như mặc định Ultralytics. Thiếu `lap` sẽ báo lỗi import, không dùng solver fallback.

Vòng đời track đồng bộ với Ultralytics/C++: track vừa hết hạn còn trong pool lost để thử ghép thêm một frame; lịch sử removed giữ tối đa 1000 phần tử mới nhất theo thứ tự append. Track bị loại do trùng chỉ rời pool, không đổi trạng thái hay thêm vào lịch sử removed. IoU dùng float32, diện tích có dấu và epsilon `1e-7` ở mẫu số. Bbox/state sau GMC giữ nguyên kích thước có dấu, kể cả góc đảo; `TrackOutput` chấp nhận bbox hữu hạn này, còn `Detection` đầu vào vẫn yêu cầu kích thước dương.

### Chạy đầy đủ video/VisDrone pipeline

Ngoài package lõi, pipeline cần checkout Ultralytics nội bộ tại `third-party/my-ultralytics`, PyTorch phù hợp với CPU/CUDA của máy, checkpoint detector và các dependency benchmark:

```bash
python -m pip install -e .
python -m pip install -r requirements-mot.txt
python -m pip install -e third-party/my-ultralytics
```

Các đường dẫn mặc định mà CLI sử dụng:

```text
weights/26leaf1.pt
third-party/my-ultralytics/ultralytics/cfg/trackers/botsort.yaml
data/VisDroneMOT/
```

Dataset, checkpoint, kết quả chạy và checkout Ultralytics đều không được commit vào repo.

## Sử dụng package

Ví dụ tối thiểu khi không dùng GMC:

```python
from mot_tracker import BoTSORT, BoTSORTConfig, Detection

tracker = BoTSORT(BoTSORTConfig(gmc_method="none"))

detections = [
    Detection(
        bbox_xyxy=(100, 80, 220, 260),
        score=0.92,
        class_id=0,
    )
]

tracks = tracker.update(detections)

for track in tracks:
    print(track.track_id, track.bbox_xyxy, track.score, track.class_id)
```

Nếu dùng cấu hình mặc định `gmc_method="sparseOptFlow"`, cần truyền frame OpenCV ở mỗi lần cập nhật:

```python
tracks = tracker.update(detections, frame_bgr)
```

Mỗi video hoặc camera stream nên dùng một instance `BoTSORT` riêng. Gọi `tracker.reset()` khi bắt đầu stream mới; ID sẽ bắt đầu lại từ `1`.

### Quy ước dữ liệu

- `bbox_xyxy`: `(x1, y1, x2, y2)` theo tọa độ ảnh, có chiều rộng và chiều cao dương.
- `score`: số thực hữu hạn trong `[0, 1]`.
- `class_id`: số nguyên; tracker ghép class-agnostic nhưng giữ class của detection mới nhất.
- `update()` chỉ trả về các track đang active ở frame hiện tại.
- Một danh sách detection rỗng vẫn được tính là một frame và cập nhật track lifecycle.

### Cấu hình BoT-SORT

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `track_high_thresh` | `0.25` | Ngưỡng detection cho bước ghép chính |
| `track_low_thresh` | `0.10` | Ngưỡng dưới cho bước phục hồi track |
| `new_track_thresh` | `0.25` | Điểm tối thiểu để tạo track mới |
| `track_buffer` | `30` | Số frame giữ một track ở trạng thái lost |
| `match_thresh` | `0.80` | Ngưỡng chi phí cho bước ghép chính |
| `fuse_score` | `True` | Kết hợp confidence vào chi phí association |
| `gmc_method` | `sparseOptFlow` | `sparseOptFlow` hoặc `none` |
| `gmc_downscale` | `2` | Hệ số giảm kích thước ảnh khi chạy GMC |

## Chạy tracking trên video

CLI standalone dùng `YOLO.predict()` làm detector, sau đó chuyển detection sang package `mot_tracker`:

```bash
python tools/infer_video_standalone.py data/input.mp4 \
  --weights weights/26leaf1.pt \
  --device auto \
  --output-dir runs/video_inference_standalone
```

Kết quả gồm:

- `<video>_predictions.txt`: prediction theo định dạng MOT/VisDrone;
- `<video>_annotated.mp4`: video đã vẽ track ID, class và confidence.

Dùng `--overwrite` để ghi đè output đã tồn tại. Có thể xem toàn bộ tùy chọn bằng:

```bash
python tools/infer_video_standalone.py --help
```

## Pipeline VisDrone-MOT

Dataset cần có cấu trúc sau cho mỗi split `val` hoặc `test`:

```text
data/VisDroneMOT/
├── val/
│   ├── sequences/<sequence_name>/0000001.jpg
│   └── annotations/<sequence_name>.txt
└── test/
    ├── sequences/<sequence_name>/0000001.jpg
    └── annotations/<sequence_name>.txt
```

### 1. Cache detection

Detector chỉ cần chạy một lần; cache được gắn fingerprint theo checkpoint, dữ liệu và cấu hình inference:

```bash
python tools/visdrone_mot_standalone.py cache \
  --split val \
  --weights weights/26leaf1.pt \
  --device auto
```

### 2. Chạy tracker từ cache

```bash
python tools/visdrone_mot_standalone.py track \
  --split val \
  --tracker-config third-party/my-ultralytics/ultralytics/cfg/trackers/botsort.yaml
```

Prediction mặc định được ghi vào:

```text
runs/visdrone_mot/26leaf1_botsort/standalone/val/predictions/
```

### 3. Đánh giá

```bash
python tools/visdrone_mot_standalone.py evaluate \
  --split val \
  --predictions-dir runs/visdrone_mot/26leaf1_botsort/standalone/val/predictions \
  --evaluator auto
```

`--evaluator auto` ưu tiên TrackEval khi khả dụng và có thể dùng evaluator nội bộ làm phương án thay thế.

### 4. Tuning trên validation split

```bash
python tools/visdrone_mot_standalone.py tune \
  --weights weights/26leaf1.pt \
  --device auto \
  --n-trials 300
```

Kết quả tuning gồm SQLite study, thông tin từng trial, `best_botsort.yaml` và `best_trial.json`. Có thể dùng số trial nhỏ hơn và `--max-frames` để smoke test trước khi chạy đầy đủ.

### 5. Đánh giá cấu hình đã khóa trên test split

```bash
python tools/visdrone_mot_standalone.py test \
  --weights weights/26leaf1.pt \
  --device auto
```

Lệnh này yêu cầu `best_botsort.yaml` và `best_trial.json` được tạo từ bước tuning validation. Không dùng test split để chọn tham số.

> Khi dùng `--max-frames`, hãy truyền cùng một giá trị cho các bước cache/track liên quan vì giới hạn frame là một phần của fingerprint. Thêm `--overwrite` khi chủ động muốn thay thế artifact đã có.

## Kiểm thử

```bash
python -m pytest -q
```

Các test của package lõi không import Ultralytics. Bộ test hiện bao phủ validation của public API, IoU, LAPJV, Kalman XYWH, GMC, track lifecycle, detector cache, serialization và evaluator.

## Giới hạn hiện tại

- Chưa hỗ trợ ReID hoặc appearance embedding.
- Chỉ hỗ trợ bounding box axis-aligned.
- GMC chỉ hỗ trợ `sparseOptFlow` và `none` trong package độc lập.
- Chưa có implementation C++.
- Các module bên trong như assignment, Kalman và association chưa được xem là public API ổn định.

## Tài liệu thêm

- [`docs/PLAN.md`](docs/PLAN.md): kế hoạch pipeline/benchmark ban đầu.
- [`docs/PLAN1.md`](docs/PLAN1.md): thiết kế tách BoT-SORT thành package độc lập.

## Giấy phép

Mã nguồn là proprietary và chỉ dành cho sử dụng nội bộ. Hãy thực hiện review pháp lý trước khi phân phối hoặc tích hợp vào sản phẩm.
