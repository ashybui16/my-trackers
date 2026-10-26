# Port `mot_tracker` sang C++ với TensorRT video inference

## Tóm tắt

- Thêm implementation C++17 độc lập, giữ nguyên code Python làm oracle đối chiếu.
- Tạo hai thư viện tái sử dụng và một CLI:
  - `mot_tracker_cpp`: BoT-SORT thuần C++.
  - `mot_inference_cpp`: TensorRT detector và pipeline `detect → track`.
  - `mot_video`: đọc video, inference bằng `weights/26leaf1.engine`, xuất video và MOT text.
- Target v1: Linux x86_64, NVIDIA CUDA, TensorRT 8.6.x và OpenCV 4; xử lý đồng bộ từng frame để sau này caller có thể cấp frame từ camera.

## API và thay đổi chính

- Public API dùng các kiểu rõ ràng:
  - `Detection { bbox_xyxy, score, class_id }`.
  - `TrackOutput { track_id, bbox_xyxy, score, class_id }`.
  - `BoTSORTConfig` giữ các tham số Python hiện tại.
  - `BoTSORT::update(detections, frame)`, `reset()`.
  - `TensorRTDetector::infer(const cv::Mat&)`.
  - `MotPipeline::processFrame(const cv::Mat&)`, `reset()`.
- Mỗi `BoTSORT`/`MotPipeline` sở hữu state và ID riêng. Nhiều stream dùng nhiều instance; một instance không được gọi đồng thời từ nhiều thread.
- Port đầy đủ tracker hiện tại: Kalman XYWH double precision, IoU/score fusion, assignment có cost limit, high/low association, unconfirmed/lost/reactivated/removed lifecycle, duplicate removal và sparse optical-flow GMC.
- Implement assignment nội bộ và dùng OpenCV cho matrix/GMC; không thêm Eigen, LAP, YAML hoặc GoogleTest.
- Detector:
  - Load engine một lần bằng RAII, tạo execution context, CUDA stream và buffer tái sử dụng.
  - Tìm input/output theo binding direction thay vì hard-code tên; yêu cầu một input NCHW batch 1 và một output FP32/FP16.
  - Input mặc định 640×640; với dynamic engine, đặt kích thước từ `DetectorConfig` và kiểm tra optimization profile.
  - Letterbox màu 114, BGR→RGB, HWC→CHW, chuẩn hóa `[0,1]`, lưu thông tin scale/padding để đưa box về ảnh gốc.
  - Hỗ trợ raw YOLO layout `[1,14,N]` hoặc `[1,N,14]`: 4 giá trị XYWH và 10 class score; chọn class tốt nhất, confidence filter, class-aware NMS, `max_det`.
  - Đồng thời nhận `[1,N,6]` nếu engine đã nhúng postprocess. Shape/layout khác phải dừng với thông báo binding cụ thể.
  - Defaults giữ parity pipeline Python: `conf=0.01`, `iou=0.7`, `max_det=300`.
- CMake tạo target build/install được, nhận `TENSORRT_ROOT` và CUDA toolkit path; liên kết OpenCV, CUDA Runtime và `nvinfer`.
- CLI nhận video positional cùng `--engine`, output directory, detector thresholds và toàn bộ tracker flags. Kết quả:
  - `<stem>_annotated.mp4`, giữ kích thước/FPS đầu vào và dùng `mp4v`.
  - `<stem>_predictions.txt` theo MOT/VisDrone, frame và class đánh số từ 1.
  - Ghi qua file tạm rồi rename để không để lại output hoàn chỉnh giả khi inference thất bại.
- Bổ sung tài liệu build/run, engine compatibility và ví dụ nhúng `MotPipeline` với frame camera. Không thêm camera-capture CLI, VisDrone evaluator, tuning hoặc ReID trong v1.

## Kiểm thử và nghiệm thu

- CTest không phụ thuộc framework ngoài:
  - Validation kiểu/config; chuyển đổi box, IoU và score fusion.
  - Assignment rỗng, rectangular, tie và cost-limit.
  - Kalman so với fixture Python hiện có.
  - Lifecycle: first-frame activation, confirmation, lost, reactivation giữ ID, expiry, empty frame, duplicate removal và reset.
  - Sparse GMC với ảnh dịch chuyển tổng hợp; lỗi OpenCV phải dùng identity warp và vẫn cập nhật reference frame.
  - Preprocess letterbox và postprocess/NMS bằng tensor tổng hợp cho cả hai output layout.
- Tạo fixture detection sequence trung lập từ Python và chạy qua cả hai tracker: track ID/state/class phải giống tuyệt đối; box/score dùng tolerance `1e-5` khi GMC tắt.
- Trên máy NVIDIA mục tiêu:
  - Load `weights/26leaf1.engine`, xác nhận TensorRT binding, warm-up và inference thành công.
  - Chạy smoke test trên một đoạn `data/input.ts`.
  - So sánh cùng engine giữa pipeline Python và C++: detection/tracked class-ID và track-ID theo frame phải khớp; box/score sai số tối đa `1e-3`.
  - Xác nhận video mở được, số frame đầu ra bằng đầu vào, MOT text hợp lệ và không còn allocation CUDA tăng dần theo frame.
- Build phải thành công với warnings nghiêm ngặt; test tracker không cần GPU, test engine được đánh dấu riêng và chỉ chạy khi có NVIDIA driver.

## Giả định và mặc định

- `26leaf1.engine` tương thích GPU mục tiêu và TensorRT 8.6.x; serialized engine không được coi là portable giữa TensorRT/GPU khác nhau. Nếu deserialize thất bại, CLI báo cần build lại engine; việc export lại không thuộc phạm vi này.
- Engine có 10 lớp VisDrone: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus và motor.
- Association vẫn class-agnostic như Python; class/score của track lấy từ detection mới nhất.
- API xử lý đồng bộ và không tự drop frame. Queue, callback và backpressure sẽ thuộc lớp tích hợp camera sau này.
