Để đọc hiểu `cpp/`, hãy bắt đầu từ **luồng dữ liệu của một khung hình**:

`video frame → TensorRTDetector::infer() → Detection[] → BoTSORT::update() → TrackOutput[] → video + tệp kết quả`

Điểm nối hai phần nằm trong [pipeline.cpp](/home/ashybui16/code/my-trackers/cpp/src/pipeline.cpp:10): chỉ hai lời gọi `infer(frame)` và `update(detections, frame)`.

### Thứ tự đọc đề xuất

1. [types.hpp](/home/ashybui16/code/my-trackers/cpp/include/mot_tracker/types.hpp:7) và [config.hpp](/home/ashybui16/code/my-trackers/cpp/include/mot_tracker/config.hpp:7): `Detection` là hộp phát hiện đầu vào; `TrackOutput` thêm ID theo dõi. `BBox` có dạng `[x1, y1, x2, y2]`.
2. [tracker.cpp](/home/ashybui16/code/my-trackers/cpp/src/tracker.cpp:26): đây là hàm điều phối chính. Mỗi lần `update()` tương ứng một frame: chia detection theo điểm tin cậy, dự đoán vị trí track cũ, ghép cặp, tạo hoặc khôi phục track, rồi trả kết quả.
3. [track.cpp](/home/ashybui16/code/my-trackers/cpp/src/track.cpp:99): xem một track nhận ID, được cập nhật, chuyển sang `Lost`, rồi `Removed` như thế nào.
4. [association.cpp](/home/ashybui16/code/my-trackers/cpp/src/association.cpp:21), [geometry.cpp](/home/ashybui16/code/my-trackers/cpp/src/geometry.cpp:29), [assignment.cpp](/home/ashybui16/code/my-trackers/cpp/src/assignment.cpp:74): `IoU` đo mức chồng lấp; chi phí ghép cơ bản là `1 − IoU`; thuật toán gán chọn các cặp trong ngưỡng.
5. [kalman_filter.cpp](/home/ashybui16/code/my-trackers/cpp/src/kalman_filter.cpp:60) và [gmc.cpp](/home/ashybui16/code/my-trackers/cpp/src/gmc.cpp:58): lần lượt dự đoán chuyển động đối tượng và bù chuyển động camera.
6. Sau khi hiểu lõi, đọc [detector.cpp](/home/ashybui16/code/my-trackers/cpp/src/detector.cpp:411) để xem ảnh được chuẩn bị, chạy TensorRT và lọc detection; cuối cùng đọc vòng lặp video trong [mot_video.cpp](/home/ashybui16/code/my-trackers/cpp/src/mot_video.cpp:239).

**Ví dụ để bám theo khi đọc:** frame đầu có một detection điểm `0.9`: tracker tạo track ID `1`. Frame sau không có detection: track thành `Lost` và không xuất ra. Nếu detection phù hợp xuất hiện lại, `reActivate()` giữ ID `1`. Trường hợp này được kiểm tra trực tiếp trong [test_mot_tracker.cpp](/home/ashybui16/code/my-trackers/cpp/tests/test_mot_tracker.cpp:85).

Một chi tiết dễ vướng: cấu hình mặc định dùng `sparseOptFlow`, nên `BoTSORT::update()` cần cả ảnh `frame`. Muốn thử tracker chỉ với danh sách detection, đặt `gmc_method = "none"` như bài test. Xem [tracker.cpp](/home/ashybui16/code/my-trackers/cpp/src/tracker.cpp:26) và [hướng dẫn build/chạy](/home/ashybui16/code/my-trackers/docs/CPP.md).