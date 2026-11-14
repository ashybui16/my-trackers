# BoT-SORT pipeline

BoT-SORT ("BoT-SORT: Robust Associations Multi-Pedestrian Tracking", 2022), tương thích với
Ultralytics `BOTSORT` **không có ReID**: ByteTrack + Kalman XYWH + camera motion compensation (GMC).

| File | Nội dung |
|---|---|
| [`botsort.cpp`](botsort.cpp) / [`botsort.hpp`](botsort.hpp) | Class `BoTSORT`: `update()` và các bước association. |
| [`botsort_config.cpp`](botsort_config.cpp) | `BoTSORTConfig::validate()`. Default ở [`include/mot/tracking/botsort_config.hpp`](../../../include/mot/tracking/botsort_config.hpp). |
| [`../common/track.cpp`](../common/track.cpp) | `Track`: một tracklet (Kalman state, score, class EMA, vòng đời). |
| [`../common/kalman_filter.cpp`](../common/kalman_filter.cpp) | `KalmanFilterXYWH`: state 8 chiều `(x, y, w, h, vx, vy, vw, vh)`. |
| [`../common/gmc.cpp`](../common/gmc.cpp) | `GMC`: ước lượng chuyển động camera bằng sparse optical flow hoặc keypoint học được (dpoint + matcher). |
| [`../common/association.cpp`](../common/association.cpp) | `associate()`: cost IoU (+ fuse score) rồi Hungarian. |
| [`../common/assignment.cpp`](../common/assignment.cpp) | `linearAssignment()`: Hungarian có ngưỡng cost. |
| [`../common/track_pool.hpp`](../common/track_pool.hpp) | `jointTracks`, `subtractTracks`, `applyGMC`, `removeDuplicateTracks`. |

Số dòng trong tài liệu này ứng với code tại thời điểm viết. Nếu sửa code thì cập nhật lại.

## Tổng quan một frame

Toàn bộ một frame nằm trong `BoTSORT::update()` ([botsort.cpp:24-94](botsort.cpp#L24-L94)):

```
detections
    ▼
 1. Chia detection: high | low                                    botsort.cpp:34-43
 2. Chia track: confirmed | unconfirmed ; pool = confirmed ∪ lost  botsort.cpp:50-61
 3. Kalman predict (pool)  →  GMC (pool + unconfirmed)             botsort.cpp:62-69
 4. First association:  pool × high                                botsort.cpp:71   → firstAssociation       :96-114
 5. Second association: pool Tracked còn thừa × low                botsort.cpp:72-78 → secondAssociation     :116-153
 6. Unconfirmed × high còn thừa                                    botsort.cpp:79-83 → unconfirmedAssociation:155-186
 7. Khởi tạo track mới từ high còn thừa                            botsort.cpp:84   → initNewTracks          :188-199
 8. Xoá track Lost quá track_buffer                                botsort.cpp:86-91
 9. mergePools  →  output                                          botsort.cpp:92-93 → mergePools :201-228, currentOutputs :230-241
```

Các list tạm của frame (`activated`, `refound`, `newly_lost`, `newly_removed`,
[botsort.cpp:45-48](botsort.cpp#L45-L48)) gom kết quả của từng bước. Chúng chỉ được gộp vào các pool
lâu dài (`tracked_tracks_`, `lost_tracks_`, `removed_tracks_`) ở bước 9.

Ở đầu `update()` ([botsort.cpp:26-29](botsort.cpp#L26-L29)): nếu bật GMC mà không truyền `frame` thì ném
`invalid_argument` (Ultralytics chỉ lặng lẽ bỏ qua GMC). Sau đó tăng `frame_id_`, nên frame đầu tiên là 1.

## 1. Chia detection

**Code:** [botsort.cpp:34-43](botsort.cpp#L34-L43)

Mỗi detection đủ điểm được bọc thành một `Track` chưa activate
([track.cpp:34-38](../common/track.cpp#L34-L38)). Track này giữ box gốc, score, class EMA và
`detection_index` = vị trí trong vector `detections` đầu vào. Index được lấy **trước** khi chia nhóm,
nên luôn trỏ đúng vào input gốc.

| Nhóm | Điều kiện | Dòng |
|---|---|---|
| high | `score >= track_high_thresh` (0.25) | [38-39](botsort.cpp#L38-L39) |
| low | `track_low_thresh < score < track_high_thresh` (0.10 – 0.25) | [40-41](botsort.cpp#L40-L41) |

Detection có `score <= track_low_thresh` bị bỏ ở [dòng 36](botsort.cpp#L36), không tạo `Track`.

Box có w hoặc h ≤ 0 không bao giờ tới được đây: postprocess của detector đã bỏ chúng, và constructor
`Detection` cũng validate (`validateBBox`, [src/core/types.cpp](../../core/types.cpp)).

## 2. Chia track

**Code:** [botsort.cpp:50-61](botsort.cpp#L50-L61)

- [Dòng 54-60](botsort.cpp#L54-L60): duyệt `tracked_tracks_`, track có `isActivated()` vào `confirmed`,
  còn lại vào `unconfirmed`. Track unconfirmed là track vừa sinh ở frame trước (trừ frame 1), chưa được
  xác nhận.
- [Dòng 61](botsort.cpp#L61): `track_pool = jointTracks(confirmed, lost_tracks_)`. Hàm này nối hai list
  và bỏ trùng ID, list đầu được ưu tiên ([track_pool.hpp:44-62](../common/track_pool.hpp#L44-L62)).

Track Lost nằm trong pool nên được match với detection high ở bước 4. Đây là cách BoT-SORT hồi sinh
track bị mất.

## 3. Predict rồi GMC

**Code:** [botsort.cpp:62-69](botsort.cpp#L62-L69)

1. **Predict** ([dòng 62-64](botsort.cpp#L62-L64)): gọi `Track::predict()`
   ([track.cpp:64-77](../common/track.cpp#L64-L77)) cho mọi track trong `pool`.
   - Track không ở state `Tracked` (tức là Lost) có vận tốc w/h (`mean[6]`, `mean[7]`) bị đặt về 0
     trước khi predict ([track.cpp:69-73](../common/track.cpp#L69-L73)). Nhờ vậy box của track bị
     mất lâu không phình hoặc co mãi.
   - Kalman predict ([kalman_filter.cpp:87-107](../common/kalman_filter.cpp#L87-L107)): `mean = F·mean`,
     `P = F·P·Fᵀ + Q`, với nhiễu `Q` tỉ lệ theo w/h hiện tại.
   - `unconfirmed` **không** được predict, giống Ultralytics.
2. **GMC** ([dòng 65-69](botsort.cpp#L65-L69)), chạy khi `gmc_method != "none"`:
   - `sparseOptFlow` ([gmc.cpp:165-228](../common/gmc.cpp#L165-L228)): chuyển frame sang ảnh xám và
     thu nhỏ cố định 2× ([gmc.cpp:35-60](../common/gmc.cpp#L35-L60)). Sau đó lấy tối đa 400 góc bằng
     `goodFeaturesToTrack` ([dòng 171](../common/gmc.cpp#L171)), theo dõi chúng từ frame trước bằng
     Lucas-Kanade ([dòng 193](../common/gmc.cpp#L193)).
   - `dpoint` ([gmc.cpp:112-163](../common/gmc.cpp#L112-L163)): ảnh xám được resize về input của model
     (512×512, giá trị [0, 1]). `IKeypointGmcModel` chạy dpoint lấy keypoint + descriptor của frame hiện
     tại rồi chạy matcher (mutual nearest neighbour, độ tương đồng > 0.75) với frame trước đã cache, nên
     mỗi frame chỉ chạy dpoint một lần. Điểm match được đổi từ toạ độ 512×512 về toạ độ frame gốc
     (căn tâm pixel như `cv::resize`). Backend: `OnnxRuntimeKeypointGmc` hoặc `TensorRTKeypointGmc`,
     truyền vào qua `createTracker(config, model)`.
   - Cả hai cùng ước lượng affine 4 bậc tự do bằng `estimateAffinePartial2D` + RANSAC trong
     `estimateWarp` ([gmc.cpp:85-110](../common/gmc.cpp#L85-L110)). Với `sparseOptFlow`, phần tịnh tiến
     được nhân lại với hệ số downscale. Frame đầu, ít hơn 5 điểm match hoặc ước lượng lỗi đều trả về
     ma trận đơn vị (kèm warning).
   - `applyGMC` gọi `Track::applyWarp` ([track.cpp:79-97](../common/track.cpp#L79-L97)) →
     `warpXywhState` ([geometry.cpp:92-135](../common/geometry.cpp#L92-L135)). Tâm và vận tốc tâm
     được xoay theo `R`, kích thước và vận tốc kích thước được xoay theo `|R|` để w/h không đổi dấu,
     rồi cộng phần tịnh tiến vào tâm. Covariance được biến đổi thành `T·P·Tᵀ`. Nếu kết quả không hữu
     hạn thì giữ state cũ.
   - Warp được áp cho cả `pool` lẫn `unconfirmed`.

Thứ tự **predict trước, GMC sau** giống Ultralytics BoT-SORT (TrackTrack thì ngược lại).

## 4. First association: pool × high

**Code:** `firstAssociation` [botsort.cpp:96-114](botsort.cpp#L96-L114)

- [Dòng 101](botsort.cpp#L101): `associate(track_pool, high, match_thresh, fuse_score)`
  ([association.cpp:44-59](../common/association.cpp#L44-L59)).
  - Cost ([`trackDistance`, association.cpp:21-42](../common/association.cpp#L21-L42)):
    `d = 1 - IoU`, tính bằng float32 như `bbox_ioa` của Ultralytics
    ([geometry.cpp:25-58](../common/geometry.cpp#L25-L58)). Khi `fuse_score` bật (mặc định),
    `d = 1 - (1 - d) * score_det` ([association.cpp:37](../common/association.cpp#L37)): cùng IoU thì
    detection score cao hơn sẽ rẻ hơn.
  - Gán cặp ([`linearAssignment`, assignment.cpp:74-](../common/assignment.cpp#L74)): Hungarian trên
    ma trận được đệm thêm hàng/cột giả, nên mỗi track hoặc detection có thể không được match. Chỉ cặp có
    cost ≤ `match_thresh` (0.8) mới được nhận, tương đương `lap.lapjv(extend_cost=True, cost_limit=...)`.
- [Dòng 102-112](botsort.cpp#L102-L112): với mỗi cặp match:
  - Track `Tracked` → `update()` ([track.cpp:99-115](../common/track.cpp#L99-L115)), vào `activated`.
    `update()` chạy Kalman update chuẩn ([kalman_filter.cpp:140-158](../common/kalman_filter.cpp#L140-L158)),
    rồi cập nhật score, class EMA ([class_ema.cpp:33-55](../common/class_ema.cpp#L33-L55)) và
    `detection_index`, tăng `tracklet_len` và đặt `is_activated = true`.
  - Track `Lost` → `reActivate()` ([track.cpp:117-126](../common/track.cpp#L117-L126)), giữ ID cũ,
    reset `tracklet_len`, vào `refound`.
- [Dòng 113](botsort.cpp#L113): trả về index track và detection high chưa match cho các bước sau.

## 5. Second association: Tracked còn thừa × low

**Code:** `secondAssociation` [botsort.cpp:116-153](botsort.cpp#L116-L153)

- [Dòng 124-130](botsort.cpp#L124-L130): từ các track chưa match ở bước 4, chỉ lấy track **đang
  Tracked**. Track Lost không được ghép với detection low.
- [Dòng 132-139](botsort.cpp#L132-L139): `associate(remaining_active, low, 0.5, false)`. Cost là
  `1 - IoU` thuần (không fuse score), ngưỡng cố định 0.5. Match → `update()`, vào `activated`.
- [Dòng 140-144](botsort.cpp#L140-L144): nếu không có track hoặc không có detection low thì mọi track
  còn lại đều tính là chưa match.
- [Dòng 145-151](botsort.cpp#L145-L151): track còn thừa → `markLost()`, vào `newly_lost`.
- [Dòng 152](botsort.cpp#L152): trả lại nguyên danh sách detection high chưa match từ bước 4. Bước 5
  chỉ dùng detection low, nên không làm thay đổi danh sách này.

Đây là ý tưởng của ByteTrack: detection score thấp (thường là object bị che) chỉ dùng để nối dài track
đang sống, không dùng để sinh track mới hay hồi sinh track Lost. Không fuse score vì score thấp sẽ đẩy
cost vượt ngưỡng 0.5.

## 6. Unconfirmed × high còn thừa

**Code:** `unconfirmedAssociation` [botsort.cpp:155-186](botsort.cpp#L155-L186)

- [Dòng 161-165](botsort.cpp#L161-L165): `remaining` = các detection high chưa match ở bước 4.
- [Dòng 166-173](botsort.cpp#L166-L173): không có track unconfirmed thì mọi detection trong `remaining`
  được chuyển thẳng sang bước 7.
- [Dòng 174](botsort.cpp#L174): `associate(unconfirmed, remaining, 0.7, fuse_score)`, cost IoU + fuse
  score, ngưỡng cố định 0.7.
- [Dòng 175-179](botsort.cpp#L175-L179): match → `update()`, track được activate (xác nhận ở lần
  match thứ 2), vào `activated`.
- [Dòng 180-184](botsort.cpp#L180-L184): unconfirmed không match → `markRemoved()` ngay, vào
  `newly_removed`.
- [Dòng 185](botsort.cpp#L185): trả về detection chưa match (index trong `remaining`) kèm chính
  `remaining`.

## 7. Khởi tạo track mới

**Code:** `initNewTracks` [botsort.cpp:188-199](botsort.cpp#L188-L199)

- [Dòng 193-195](botsort.cpp#L193-L195): bỏ detection có `score < new_track_thresh` (0.25).
- [Dòng 196](botsort.cpp#L196): `activate()` ([track.cpp:48-62](../common/track.cpp#L48-L62)) với ID
  mới `next_track_id_++`. Hàm này khởi tạo Kalman từ box
  ([kalman_filter.cpp:60-85](../common/kalman_filter.cpp#L60-L85)) và đặt state `Tracked`.
- [track.cpp:59](../common/track.cpp#L59): `is_activated = (frame_id == 1)`. Track sinh ở frame 1 hiển
  thị ngay. Track sinh ở các frame sau là unconfirmed tới khi match ở bước 6 của frame kế tiếp.

## 8. Hết hạn track Lost

**Code:** [botsort.cpp:86-91](botsort.cpp#L86-L91)

Track trong `lost_tracks_` có `frame_id - end_frame > track_buffer` (30) → `markRemoved()`, vào
`newly_removed`.

## 9. `mergePools` và output

**Code:** `mergePools` [botsort.cpp:201-228](botsort.cpp#L201-L228)

1. [Dòng 205-212](botsort.cpp#L205-L212): `tracked = (tracked_tracks_ còn Tracked) ∪ activated ∪ refound`,
   ưu tiên track đứng trước khi trùng ID.
2. [Dòng 213-216](botsort.cpp#L213-L216): `lost = (lost_tracks_ − tracked) ∪ newly_lost`, rồi trừ
   `removed_tracks_` **của các frame trước**. Track vừa bị Removed ở frame này vẫn có thể được match ở
   frame sau, như Ultralytics.
3. [Dòng 218-220](botsort.cpp#L218-L220): `removeDuplicateTracks`
   ([track_pool.hpp:93-](../common/track_pool.hpp#L93)): cặp tracked/lost có IoU > 0.85 → bỏ track sống
   ngắn hơn (tính theo `frame_id - start_frame`).
4. [Dòng 222-227](botsort.cpp#L222-L227): append `newly_removed` vào `removed_tracks_`, giữ tối đa 1000
   track gần nhất.

**Output:** `currentOutputs` [botsort.cpp:230-241](botsort.cpp#L230-L241) lấy các track trong
`tracked_tracks_` có state `Tracked` và `isActivated()`, trả về
`TrackOutput(track_id, bbox, score, class_id, detection_index)`. `bbox` là box của Kalman state
([track.cpp:40-42](../common/track.cpp#L40-L42)), không phải box thô của detector. Muốn lấy box thô hay
`class_scores` thì dùng `detections[detection_index]`. Mọi track output đều vừa được match ở frame này
(track không match đã chuyển sang Lost ở bước 5), nên `detection_index` luôn hợp lệ.

## Vòng đời track

```
detection
    │ activate (frame > 1)                 activate (frame 1)
    ▼                                            │
 Tracked, unconfirmed ──match (bước 6)──►  Tracked, activated  ◄──┐
    │                                            │ không match      │ match bước 4 → reActivate
    │ không match (bước 6)                       ▼ (bước 5)         │
    │                                          Lost ────────────────┘
    │                                            │ > track_buffer frame (bước 8)
    ▼                                            ▼
 Removed ◄──────────────────────────────────────┘
```

Chuyển state: `markLost()` / `markRemoved()` ([track.cpp:128-134](../common/track.cpp#L128-L134)).

## Tham số

| Tham số | Mặc định | Dùng ở |
|---|---|---|
| `track_high_thresh` | 0.25 | Chia nhóm high ([botsort.cpp:38](botsort.cpp#L38)) |
| `track_low_thresh` | 0.10 | Chia nhóm low ([botsort.cpp:36](botsort.cpp#L36), [40](botsort.cpp#L40)) |
| `new_track_thresh` | 0.25 | Score tối thiểu để sinh track, phải ≥ `track_high_thresh` ([botsort.cpp:193](botsort.cpp#L193)) |
| `track_buffer` | 30 | Số frame giữ track Lost ([botsort.cpp:87](botsort.cpp#L87)) |
| `match_thresh` | 0.80 | Ngưỡng cost của first association ([botsort.cpp:101](botsort.cpp#L101)) |
| `fuse_score` | true | Fuse score vào cost IoU ở bước 4 và 6 ([botsort.cpp:101](botsort.cpp#L101), [174](botsort.cpp#L174)) |
| `gmc_method` | `sparseOptFlow` | `sparseOptFlow`, `dpoint` hoặc `none` ([botsort.cpp:65](botsort.cpp#L65)) |
| `gmc_dpoint_model`, `gmc_matcher_model` | rỗng | Model dpoint/matcher cho `dpoint` (`.onnx` → ONNX Runtime, `.engine` → TensorRT); app nạp chúng |
| `class_ema_alpha` | 0.0 | EMA của vector class, 0 = theo detection mới nhất ([class_ema.cpp:33](../common/class_ema.cpp#L33)) |

Các giá trị cố định trong code, giống Ultralytics: ngưỡng 0.5 của second association
([botsort.cpp:133](botsort.cpp#L133)), ngưỡng 0.7 của unconfirmed ([botsort.cpp:174](botsort.cpp#L174)),
IoU dedup 0.85, removed buffer 1000 ([botsort.cpp:224](botsort.cpp#L224)) và GMC downscale 2.
Ràng buộc giữa các tham số được kiểm tra trong `validate()`
([botsort_config.cpp](botsort_config.cpp)).
