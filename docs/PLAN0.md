# Tuning BoT-SORT trên VisDroneMOT Val và đánh giá trên Test

## Tóm tắt

- Xây dựng CLI ngoài `third-party/my-ultralytics`, chạy trong conda env `my-trackers`.
- Dùng detector `weights/26leaf1.pt`; cache detection một lần rồi chạy 300 trial BoT-SORT trên `data/VisDroneMOT/val`.
- Chọn cấu hình theo HOTA tổng hợp, khóa cấu hình thắng cuộc rồi mới inference/evaluate trên `data/VisDroneMOT/test`.
- Không render video, không sửa mã nguồn vendored và không tuning dựa trên test.

## Môi trường và dependency

- Editable install hiện đã đúng:
  - Ultralytics `8.4.157`.
  - Import từ `/home/ashybui16/code/my-trackers/third-party/my-ultralytics/ultralytics`.
  - Không cần chạy lại `pip install -e`.
- Sửa `requirements-mot.txt` để khóa:

  ```text
  numpy==1.26.4
  opencv-python==4.11.0.86
  lap==0.5.12
  scipy==1.15.3
  optuna>=4,<5
  trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a
  ```

- Cài dependency và kiểm tra môi trường:

  ```bash
  conda activate my-trackers
  python -m pip install -r requirements-mot.txt
  python -m pip check
  ```

- Hạ OpenCV từ `5.0.0.93` xuống `4.11.0.86` để tương thích NumPy 1.26.4; giữ nguyên Torch `2.4.1+cu118` và TorchVision `0.19.1+cu118`.
- Preflight của CLI phải kiểm tra import path Ultralytics, phiên bản, dependency, class mapping checkpoint và thiết bị. Hiện CUDA chưa khả dụng nên `device=auto` sẽ chọn CPU.

## CLI và dữ liệu trung gian

- Thêm `tools/visdrone_mot.py` với các subcommand:
  - `cache`: chạy detector và tạo detection cache.
  - `track`: chạy BoT-SORT từ cache với một YAML.
  - `evaluate`: đánh giá prediction đã có.
  - `tune`: cache val nếu cần, chạy Optuna và chọn cấu hình.
  - `test`: dùng cấu hình đã khóa để cache, track và evaluate test.
- Cache detector bằng `model.predict`, không dùng `model.track`:
  - `imgsz=640`, `conf=0.01`, `max_det=300`, toàn bộ 10 lớp.
  - Lưu `xyxy`, confidence, class và frame index theo sequence.
  - Fingerprint gồm SHA-256 checkpoint, phiên bản/commit Ultralytics, split, manifest frame và tham số inference.
  - Cache sai fingerprint bị từ chối hoặc tạo lại khi có `--overwrite`.
- BoT-SORT đọc detection từ cache và ảnh gốc để GMC hoạt động; tracker được khởi tạo lại ở đầu mỗi sequence.
- Output dùng `runs/visdrone_mot/26leaf1_botsort/` với các vùng riêng cho `cache/val`, `tuning`, `cache/test` và `test`.

## Tuning trên Val

- Dùng toàn bộ 7 sequence/2.846 frame val.
- Chạy đúng 300 trial Optuna TPE tuần tự, seed `42`, không pruning:
  - Trial đầu là `botsort.yaml` mặc định.
  - Mỗi trial chạy và evaluate đủ 7 sequence.
  - `with_reid=False`; không tune appearance/proximity.
- Search space:
  - `track_high_thresh`: 0,15–0,60.
  - `track_low_thresh`: tối thiểu 0,01 và luôn nhỏ hơn `track_high_thresh`.
  - `new_track_thresh`: từ `track_high_thresh` đến 0,80.
  - `track_buffer`: `15, 30, 45, 60, 90`.
  - `match_thresh`: 0,60–0,95.
  - `fuse_score`: `True/False`.
  - `gmc_method`: `sparseOptFlow`, `orb`, `sift`, `ecc`, `none`.
- Objective là HOTA detection-averaged do TrackEval tổng hợp trên 10 lớp và 7 sequence.
- Khi HOTA bằng nhau trong sai số `1e-6`, lần lượt ưu tiên IDF1, MOTA và trial có số nhỏ hơn.
- Lưu `study.sqlite3`, `trials.csv`, metric từng trial, `best_botsort.yaml`, `best_trial.json` và báo cáo val. Best trial phải ghi đầy đủ metric, tham số và fingerprint.

## Đánh giá cuối trên Test

- `test` chỉ chấp nhận `best_botsort.yaml` và metadata sinh từ bước tuning val.
- Cache detector một lần cho 17 sequence/6.635 frame test với cùng checkpoint và inference settings của val.
- Prediction được ghi atomically theo định dạng VisDrone:

  ```text
  frame,id,x,y,width,height,confidence,class,-1,-1
  ```

- Chuyển class YOLO `0..9` thành VisDrone `1..10`; kiểm tra frame liên tục, box hợp lệ và không trùng track ID trong cùng frame.
- TrackEval chạy HOTA, CLEAR, Identity và Count:
  - GT hợp lệ: `mark=1`, class `1..10`.
  - Ignore: `mark=0` hoặc class `0/11`.
  - Prediction chưa match GT hợp lệ và có IoA ≥ 0,5 với ignore region không bị tính FP.
  - Xuất metric theo sequence, lớp và tổng hợp: HOTA/DetA/AssA, MOTA/MOTP, IDF1/IDP/IDR, precision/recall, FP/FN, IDSW và fragmentation.
- Test không ghi đè kết quả cũ nếu thiếu `--overwrite`.

## Kiểm thử và nghiệm thu

- Unit test cho class mapping, cache/fingerprint, serialization VisDrone, reset tracker và ràng buộc search space.
- Fixture evaluator:
  - Prediction hoàn hảo cho HOTA/MOTA/IDF1 bằng 100%.
  - ID switch làm giảm AssA/IDF1 và tăng IDSW.
  - FP, FN, sai lớp và ignored region được xử lý đúng.
- Smoke test CPU trên vài frame val trước khi tạo full cache.
- Nghiệm thu khi:
  - `pip check` không còn xung đột dependency.
  - Tuning hoàn thành đủ 300 trial và tạo `best_botsort.yaml`.
  - Test tạo đủ 17 file prediction và báo cáo cho đủ 10 lớp/17 sequence.
  - Không tham số nào được điều chỉnh từ metric test.
  - `evaluate` có thể chạy lại từ prediction mà không inference lại.
