# Tách BoT-SORT thành package Python độc lập

## Tóm tắt

- Tạo package `mot_tracker` độc lập hoàn toàn với Ultralytics ở runtime, phát hành nội bộ/proprietary.
- Phiên bản đầu giữ hành vi baseline hiện tại: Kalman XYWH, IoU association hai tầng, track lifecycle, score fusion và GMC `sparseOptFlow`; chưa có ReID.
- Ultralytics chỉ còn vai trò detector trong các CLI hiện có. Tracker nhận `Detection` chuẩn hóa và không biết `Results`, YOLO, Torch hay dataset/evaluator.
- Chưa tạo mã C++; API và test fixture được thiết kế để port gần 1:1 trong giai đoạn sau.

## Public API và cấu trúc

- Đóng gói bằng `pyproject.toml`, source trong `src/mot_tracker/`, phiên bản đầu `0.1.0`, Python `>=3.10`.
- Public API:
  ```python
  Detection(
      bbox_xyxy: tuple[float, float, float, float],
      score: float,
      class_id: int,
  )

  TrackOutput(
      track_id: int,
      bbox_xyxy: tuple[float, float, float, float],
      score: float,
      class_id: int,
  )

  BoTSORTConfig(
      track_high_thresh=0.25,
      track_low_thresh=0.10,
      new_track_thresh=0.25,
      track_buffer=30,
      match_thresh=0.80,
      fuse_score=True,
      gmc_method="sparseOptFlow",  # hoặc "none"
      gmc_downscale=2,
  )

  tracker = BoTSORT(config)
  tracks = tracker.update(detections, frame=None)
  tracker.reset()
  ```
- `bbox_xyxy` luôn là `[x1, y1, x2, y2]`; input phải hữu hạn, kích thước dương, score trong `[0,1]`.
- `update()` chỉ trả track đã active ở frame hiện tại. Lost/removed được giữ nội bộ; ID là bộ đếm riêng của từng tracker và trở lại `1` sau `reset()`.
- Mỗi video/camera stream dùng một instance riêng. Empty detections vẫn làm tăng frame counter và chạy lifecycle.
- Khi bật `sparseOptFlow`, mỗi lần `update()` bắt buộc có frame BGR hoặc grayscale hợp lệ; GMC lỗi sẽ cảnh báo, dùng identity warp và cập nhật reference frame để phục hồi ở frame sau.
- Package chỉ phụ thuộc NumPy, OpenCV và `lapjv`; không phụ thuộc Ultralytics, Torch, SciPy, detector hay evaluator. Môi trường benchmark tiếp tục pin `numpy==1.26.4`, `opencv-python==4.11.0.86`, `lap==0.5.12`.

## Implementation

- Tách nội bộ thành các owner rõ ràng: types/config, geometry, LAP assignment, Kalman XYWH, GMC, track state/lifecycle, association và tracker orchestration. Các module này chưa trở thành plugin API công khai.
- Giữ luồng baseline:
  1. Lọc detection không hợp lệ và chia high/low confidence.
  2. Predict active/lost tracks bằng Kalman; áp dụng GMC trước association.
  3. Ghép active+lost với high detections bằng IoU, optional score fusion và LAPJV với `match_thresh`.
  4. Ghép active còn lại với low detections bằng IoU threshold `0.5`.
  5. Ghép unconfirmed tracks với high detections còn lại bằng threshold `0.7`.
  6. Tạo track mới khi score đạt `new_track_thresh`; quản lý lost/reactivation/removal và loại duplicate ở IoU `>0.85`, ưu tiên track sống lâu hơn.
- Giữ association class-agnostic như baseline để đạt numerical parity; class và score của track được cập nhật từ detection mới nhất.
- Chỉ hỗ trợ axis-aligned boxes, `sparseOptFlow` và `none`; chưa có ReID, embedding, ORB, SIFT, ECC, oriented boxes hoặc public strategy interface.
- Chuyển pipeline VisDrone sang tạo `Detection` và gọi `mot_tracker.BoTSORT`; loại `DetectionBatch` cùng import tracker nội bộ Ultralytics. Evaluator/cache/tuning vẫn nằm ngoài package, nhưng search space GMC giảm còn `sparseOptFlow|none`.
- Chuyển CLI inference video từ `YOLO.track()` sang `YOLO.predict()` → adapter detections → `mot_tracker` → render/output, chứng minh detector và tracker đã tách rời.
- Config adapter tiếp tục đọc YAML hiện có nhưng chỉ ánh xạ các field được `BoTSORTConfig` hỗ trợ; field ReID cũ không đi vào package.

## Clean-room và kiểm thử

- Thực hiện theo hai vai trò tách biệt:
  - Vai trò đặc tả chạy implementation vendored như black box và sinh golden fixtures cho geometry, LAP, Kalman, GMC và lifecycle.
  - Vai trò implementation chỉ dùng paper, đặc tả API và fixtures; không tham chiếu hoặc sao chép source AGPL.
- Golden fixtures được lưu dưới dạng JSON/NPZ, không import Ultralytics khi chạy test. Script tạo oracle là công cụ development riêng, không đóng gói cùng library.
- Test bắt buộc:
  - IoU: trùng hoàn toàn, không giao, giao một phần, input rỗng.
  - LAPJV: ma trận vuông/chữ nhật/rỗng, unmatched, cost limit và tie case cố định.
  - Kalman: initiate, predict, multi-predict, project, update, covariance và golden numeric với `atol=1e-9`.
  - GMC: frame đầu identity, dịch chuyển synthetic, thiếu feature, lỗi OpenCV và reset; tolerance `1e-3`.
  - Lifecycle: tạo/confirm, lost, reactivate giữ ID, expire, duplicate removal, empty frame, reset và nhiều tracker không dùng chung ID state.
  - API validation và yêu cầu frame khi GMC bật.
  - Smoke test replay một đoạn detection cache qua adapter; package test phải chạy được khi module `ultralytics` không tồn tại.
  - Regression cho cache/evaluate hiện có để bảo đảm việc đổi tracker không phá format VisDrone.
- Chuẩn nghiệm thu v1 là numerical parity của core và lifecycle fixtures; chưa dùng HOTA/IDF1 toàn bộ val làm release gate.
- Chạy test trong conda env `my-trackers`; base Python 3.14 hiện không phải môi trường nghiệm thu.

## Hướng phát triển sau v1

- Nhánh cải tiến tracker: thêm từng thay đổi vào module nội bộ, tạo config/ablation riêng và dùng detection cache cùng evaluator hiện có để đo HOTA, IDF1, AssA và runtime.
- Nhánh C++ chỉ bắt đầu khi API/config/lifecycle đã ổn định: port các type và module core, chạy lại cùng golden fixtures; C++ library chỉ phục vụ inference video/camera, không mang evaluator vào sản phẩm.
- Trước khi phân phối hoặc tích hợp sản phẩm, thực hiện review pháp lý để xác nhận quy trình clean-room và quyền sở hữu proprietary.
