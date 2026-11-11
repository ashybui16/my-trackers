# TrackTrack pipeline

TrackTrack ("Focusing on Tracks for Online Multi-Object Tracking", CVPR 2025), bản port của
Ultralytics `TRACKTRACK` **không có ReID** (cost chỉ dựa trên chuyển động và hình học).

| File | Nội dung |
|---|---|
| [`tracktrack.cpp`](tracktrack.cpp) / [`tracktrack.hpp`](tracktrack.hpp) | Class `TrackTrack`: vòng lặp `updateWithRecovered()` và `mergePools()`. |
| [`tracktrack_detail.cpp`](tracktrack_detail.cpp) | `costMatrix`, `iterativeAssociate`, `trackAwareNms` (tách riêng để test). |
| [`tt_track.cpp`](tt_track.cpp) / [`tt_track.hpp`](tt_track.hpp) | `TTTrack`: Kalman XYWH + NSA, lịch sử box, vận tốc góc box, lịch sử score. |
| [`tracktrack_config.cpp`](tracktrack_config.cpp) | `TrackTrackConfig::validate()`. Default ở [`include/mot/tracking/tracktrack_config.hpp`](../../../include/mot/tracking/tracktrack_config.hpp). |
| [`../common/`](../common/) | Dùng chung với BoT-SORT: `KalmanFilterXYWH`, `GMC`, `track_pool.hpp`. |

Số dòng trong tài liệu này ứng với code tại thời điểm viết. Nếu sửa code thì cập nhật lại.

## Tổng quan một frame

Toàn bộ một frame nằm trong `TrackTrack::updateWithRecovered()`
([tracktrack.cpp:37-205](tracktrack.cpp#L37-L205)):

```
detections (tight NMS) ─┐
recovered  (loose NMS) ─┤
                        ▼
 1. Chia detection thành 3 nhóm: high | low | recovered           tracktrack.cpp:49-69
 2. Chia track: confirmed (Tracked) + lost → pool ; New → unconf.  tracktrack.cpp:71-77
 3. GMC (warp)  →  Kalman predict              (GMC TRƯỚC predict) tracktrack.cpp:79-86
 4. Main association: pool × (high ∪ low ∪ recovered)              tracktrack.cpp:88-116
 5. Unconfirmed × high còn thừa                                    tracktrack.cpp:118-145
 6. (tuỳ chọn) Relaxed rebind: track Lost cũ × detection còn thừa  tracktrack.cpp:147-173
 7. Track-aware initialization (TAI) cho detection còn thừa        tracktrack.cpp:175-186
 8. Xoá track Lost quá track_buffer                                tracktrack.cpp:188-193
 9. mergePools  →  output                                          tracktrack.cpp:194-204 → mergePools :207-231
```

Các list tạm của frame (`activated`, `refound`, `newly_lost`, `newly_removed`,
[tracktrack.cpp:44-47](tracktrack.cpp#L44-L47)) gom kết quả của từng bước. Chúng chỉ được gộp vào các
pool lâu dài ở bước 9.

`update()` ([tracktrack.cpp:33-35](tracktrack.cpp#L33-L35)) chính là `updateWithRecovered()` với
`recovered` rỗng. Pipeline gọi `updateWithRecovered()` khi `usesRecoveredDetections()` trả về `true`
([src/pipeline/pipeline.cpp:15-18](../../pipeline/pipeline.cpp#L15-L18)).

Ở đầu hàm ([tracktrack.cpp:40-43](tracktrack.cpp#L40-L43)): nếu bật GMC mà không truyền `frame` thì ném
`invalid_argument`. Sau đó tăng `frame_id_`, nên frame đầu tiên là 1.

## 0. Recovered detections (phía detector)

**Code:** [src/detection/yolo/postprocess.cpp:185](../../detection/yolo/postprocess.cpp#L185) (NMS kép),
[:236-262](../../detection/yolo/postprocess.cpp#L236-L262) (lọc recovered), hằng số ở
[postprocess.hpp:46-47](../../detection/yolo/postprocess.hpp#L46-L47).

Detector YOLO chạy song song hai NMS: NMS chuẩn (`iou_threshold` trong config) và một NMS lỏng
(`kLooseNmsIou = 0.95`). Box bị NMS chuẩn loại nhưng NMS lỏng giữ lại, và không gần như trùng hoàn toàn
với một box đã giữ (IoU < `kLooseNmsDedupIou = 0.97`), được trả về dưới dạng `recovered`. Đây là các
object bị che khuất hoặc đứng sát nhau mà NMS chuẩn hay xoá nhầm (tương ứng `compute_dets_del` của
Ultralytics).

## 1. Chia detection

**Code:** [tracktrack.cpp:49-69](tracktrack.cpp#L49-L69)

Mỗi detection được bọc thành một `TTTrack` chưa activate
([tt_track.cpp:27-33](tt_track.cpp#L27-L33)). `prev_score` được khởi tạo bằng chính score. Các detection
được xếp vào **một** vector `all_dets` theo thứ tự nhóm, nên mỗi nhóm chiếm một khoảng cột liên tiếp
trong cost matrix:

| Nhóm | Điều kiện | Cột trong cost matrix | Dòng |
|---|---|---|---|
| high | `score >= track_high_thresh` (0.6) | `[0, n_high)` | [52-57](tracktrack.cpp#L52-L57) |
| low | `track_low_thresh < score < track_high_thresh` (0.25 – 0.6) | `[n_high, n_high_low)` | [58-64](tracktrack.cpp#L58-L64) |
| recovered | recovered và `score > track_high_thresh` | `[n_high_low, end)` | [65-69](tracktrack.cpp#L65-L69) |

Detection có `score <= track_low_thresh` bị bỏ. `detection_index` của high/low là vị trí trong
`detections` gốc. Recovered detection có `detection_index = -1`
([dòng 67](tracktrack.cpp#L67)) vì nó không nằm trong `detections`.

## 2. Chia track

**Code:** [tracktrack.cpp:71-77](tracktrack.cpp#L71-L77)

- [Dòng 74-76](tracktrack.cpp#L74-L76): chia theo **state**, không theo `is_activated` như BoT-SORT.
  Track `Tracked` vào `confirmed`, track `New` (chưa đủ `min_track_len` hit) vào `unconfirmed`.
- Track sinh ở frame 1 được hiển thị ngay (`is_activated = true`) nhưng vẫn ở state `New`
  ([tt_track.cpp:74-76](tt_track.cpp#L74-L76)), nên vẫn rơi vào `unconfirmed`.
- [Dòng 77](tracktrack.cpp#L77): `pool = jointTracks(confirmed, lost_tracks_)`
  ([track_pool.hpp:44-62](../common/track_pool.hpp#L44-L62)).

## 3. GMC rồi predict

**Code:** [tracktrack.cpp:79-86](tracktrack.cpp#L79-L86)

Khác BoT-SORT, TrackTrack áp GMC **trước** Kalman predict:

1. **GMC** ([dòng 80-84](tracktrack.cpp#L80-L84)), chỉ chạy khi `gmc_method == "sparseOptFlow"`:
   `gmc_.apply(frame)` ([gmc.cpp:62-158](../common/gmc.cpp#L62-L158)) ước lượng ma trận affine giữa hai
   frame (sparse optical flow, ảnh thu nhỏ cố định 2×), rồi `TTTrack::applyWarp`
   ([tt_track.cpp:96-113](tt_track.cpp#L96-L113)) → `warpXywhState`
   ([geometry.cpp:92-135](../common/geometry.cpp#L92-L135)) áp warp lên mean và covariance của `pool` và
   `unconfirmed`. Nếu kết quả không hữu hạn thì giữ nguyên state cũ.
2. **Predict** ([dòng 85-86](tracktrack.cpp#L85-L86)) cho cả `pool` **và** `unconfirmed` (BoT-SORT không
   predict unconfirmed). `TTTrack::predict()` ([tt_track.cpp:81-94](tt_track.cpp#L81-L94)) đặt vận tốc
   w/h (`mean[6]`, `mean[7]`) về 0 trước khi predict nếu track không ở state `Tracked`.

## 4. Main association

**Code:** [tracktrack.cpp:88-116](tracktrack.cpp#L88-L116)

### Cost matrix

**Code:** `costMatrix` [tracktrack_detail.cpp:60-98](tracktrack_detail.cpp#L60-L98), gọi ở
[tracktrack.cpp:89](tracktrack.cpp#L89)

Với mỗi cặp (track `t`, detection `d`), tính bằng float32 như bản NumPy. Ma trận khởi tạo toàn 1
([dòng 67](tracktrack_detail.cpp#L67)).

```
IoU   = IoU(t.box, d.box)                                           tracktrack_detail.cpp:86
if IoU <= 0.1:  cost = 1                                            :87-89  (kMinAngleIou, :11)
else:
  cost  = 1 - HIoU(t, d) * IoU                                      :90     (verticalIouF :31-35)
        + conf_weight  * |ŝ_t - s_d|                                :91     (ŝ_t tính ở :84)
        + angle_weight * angle(t, d)                                :92-93  (angleDistance :38-52)
  cost  = clip(cost, 0, 1)                                          :94
```

- **HMIoU** ([`verticalIouF`, dòng 31-35](tracktrack_detail.cpp#L31-L35)): nhân IoU với IoU theo trục
  dọc, phạt mạnh hơn các box lệch về chiều cao hoặc vị trí y (người đứng gần hay xa camera).
- **Confidence distance** ([dòng 84](tracktrack_detail.cpp#L84), [91](tracktrack_detail.cpp#L91)):
  score của object bị che thường giảm dần. Score frame này được dự đoán bằng ngoại suy tuyến tính
  `ŝ_t = s_t + (s_t - s_t_prev)` rồi so với score detection.
- **Angle distance** ([`angleDistance`, dòng 38-52](tracktrack_detail.cpp#L38-L52)): với 4 góc box (LT,
  LB, RT, RB), so hướng vận tốc đơn vị của track (`velocity()`, xem mục *Cập nhật track*) với hướng từ
  box của track cách đây `kDeltaT = 3` frame ([tt_track.hpp:20](tt_track.hpp#L20)) tới box detection.
  Lấy trung bình |góc| của 4 góc, chia π, rồi nhân score detection
  ([dòng 51](tracktrack_detail.cpp#L51)). Box quá khứ lấy từ `historyBox`
  ([tt_track.cpp:51-61](tt_track.cpp#L51-L61)). Nếu không có đúng frame cần tìm thì dùng box mới nhất
  trong lịch sử.

### Penalty theo nhóm

**Code:** [tracktrack.cpp:90-97](tracktrack.cpp#L90-L97)

Cộng penalty cho các cột từ `n_high` trở đi, rồi clip lại về [0, 1]:

| Cột | Penalty |
|---|---|
| high | 0 |
| low | `penalty_p` (0.2) |
| recovered | `penalty_q` (0.4) |

Nhờ vậy cả ba nhóm được ghép trong **một** lần, nhưng detection kém tin cậy hơn chỉ thắng khi không có
ứng viên high tốt hơn. Cách này thay cho việc ghép tuần tự high → low của ByteTrack/BoT-SORT.

### Iterative assignment

**Code:** `iterativeAssociate` [tracktrack_detail.cpp:100-184](tracktrack_detail.cpp#L100-L184), gọi ở
[tracktrack.cpp:98](tracktrack.cpp#L98)

Không dùng Hungarian. Mỗi vòng lặp ([dòng 122-176](tracktrack_detail.cpp#L122-L176)):

1. [Dòng 123-146](tracktrack_detail.cpp#L123-L146): trong một lần quét theo hàng, tìm argmin theo hàng và
   theo cột trên các hàng/cột còn active. Tie-break giữ index đầu tiên (`<` chặt), như `np.argmin`.
2. [Dòng 147-157](tracktrack_detail.cpp#L147-L157): cặp `(r, c)` match nếu chúng là **mutual nearest
   neighbour** (argmin của hàng `r` là `c` và argmin của cột `c` là `r`) **và** `cost < gate`.
3. [Dòng 158](tracktrack_detail.cpp#L158): dừng nếu vòng này không match được cặp nào.
4. [Dòng 159-175](tracktrack_detail.cpp#L159-L175): ngược lại thì `gate -= reduce_step` (0.05), và loại
   các hàng/cột đã match hoặc có min ≥ gate mới. Những hàng/cột này không bao giờ match được nữa vì gate
   chỉ giảm, nên bỏ chúng đi cho nhanh mà kết quả vẫn y như bản NumPy (giải thích ở
   [dòng 101-106](tracktrack_detail.cpp#L101-L106)).

Gate khởi đầu là `match_thresh` (0.7). Gate co dần nghĩa là các cặp rõ ràng được chốt trước, còn các cặp
đang tranh chấp phải có cost ngày càng nhỏ mới được match.

### Cập nhật track

**Code:** [tracktrack.cpp:99-116](tracktrack.cpp#L99-L116)

- Match, track đang `Tracked` → `update()` ([tt_track.cpp:126-158](tt_track.cpp#L126-L158)), vào
  `activated` ([dòng 102-104](tracktrack.cpp#L102-L104)). `update()` làm các việc sau:
  - Lưu `prev_score = score` ([tt_track.cpp:129](tt_track.cpp#L129)).
  - Chạy NSA-Kalman update ([`kalmanUpdate`, tt_track.cpp:115-124](tt_track.cpp#L115-L124)): measurement
    noise được nhân với `max(1 - score, 0.05)`
    ([kalman_filter.cpp:125-130](../common/kalman_filter.cpp#L125-L130)), nên detection càng tin cậy thì
    càng kéo state mạnh.
  - Đẩy box detection vào lịch sử ([dòng 131-132](tt_track.cpp#L131-L132)). Đây là ring buffer
    `kDeltaT + 1` phần tử ([tt_track.cpp:41-49](tt_track.cpp#L41-L49)).
  - Tính lại vận tốc góc ([dòng 134-149](tt_track.cpp#L134-L149)): với dt = 1..3, cộng hướng đơn vị từ
    box quá khứ tới box hiện tại chia dt, rồi chia trung bình cho 3.
  - Cập nhật score, class EMA và `detection_index`. Track `New` chuyển thành `Tracked` khi
    `tracklet_len + 1 >= min_track_len` ([dòng 152-155](tt_track.cpp#L152-L155)).
- Match, track đang `Lost` → `reActivate()` ([tt_track.cpp:160-171](tt_track.cpp#L160-L171)), vào
  `refound` ([dòng 105-108](tracktrack.cpp#L105-L108)). `reActivate()` cũng chạy NSA-Kalman update,
  nhưng đẩy box của **Kalman state** (không phải box detection) vào lịch sử và **không** tính lại vận tốc
  góc. Nó reset `tracklet_len` và chuyển thẳng về `Tracked`.
- Track trong `pool` không match và chưa `Lost` → `markLost()`, vào `newly_lost`
  ([dòng 110-116](tracktrack.cpp#L110-L116)).

## 5. Unconfirmed × leftover high

**Code:** [tracktrack.cpp:118-145](tracktrack.cpp#L118-L145)

- [Dòng 119-122](tracktrack.cpp#L119-L122): `leftover` = các detection **high** chưa match ở bước 4
  (cột < `n_high`). Low và recovered còn thừa bị bỏ.
- [Dòng 124-126](tracktrack.cpp#L124-L126): chạy lại `costMatrix` + `iterativeAssociate`, không có
  penalty, dùng cùng `match_thresh`/`reduce_step`.
- [Dòng 127-131](tracktrack.cpp#L127-L131): match → `update()`, vào `activated`.
- [Dòng 132-136](tracktrack.cpp#L132-L136): unconfirmed không match → `markRemoved()` ngay.
- [Dòng 137-139](tracktrack.cpp#L137-L139): `leftover` chỉ còn các detection chưa match.
- [Dòng 140-145](tracktrack.cpp#L140-L145): nếu không có unconfirmed hoặc không có leftover thì mọi
  unconfirmed đều bị xoá.

## 6. Relaxed rebind (tuỳ chọn)

**Code:** [tracktrack.cpp:147-173](tracktrack.cpp#L147-L173)

Chỉ chạy khi `lost_match_thr > 0` (mặc định 0 = tắt), [dòng 148](tracktrack.cpp#L148).

- [Dòng 149-159](tracktrack.cpp#L149-L159): lấy các track trong `pool` đã `Lost` **từ frame trước**. Track
  vừa bị đánh Lost ở bước 4 (có trong `newly_lost`) không được tính.
- [Dòng 161-163](tracktrack.cpp#L161-L163): ghép với `leftover` bằng `iterativeAssociate`, gate khởi đầu là
  `lost_match_thr`.
- [Dòng 164-168](tracktrack.cpp#L164-L168): match → `reActivate()`, vào `refound`.
- [Dòng 169-171](tracktrack.cpp#L169-L171): `leftover` chỉ còn các detection chưa match.

## 7. Track-aware initialization

**Code:** [tracktrack.cpp:175-186](tracktrack.cpp#L175-L186), `trackAwareNms`
[tracktrack_detail.cpp:186-228](tracktrack_detail.cpp#L186-L228)

Thay cho quy tắc "mọi detection thừa có score cao đều sinh track mới":

1. [tracktrack.cpp:176-180](tracktrack.cpp#L176-L180): `active` = track `Tracked` hiện có cộng các track
   vừa được activate ở frame này.
2. [tracktrack_detail.cpp:194-198](tracktrack_detail.cpp#L194-L198): ứng viên là detection trong
   `leftover` có `score > new_track_thresh` (0.7).
3. [Dòng 201-214](tracktrack_detail.cpp#L201-L214): loại ứng viên có IoU > `tai_thr` (0.55) với bất kỳ
   track nào trong `active`.
4. [Dòng 217-226](tracktrack_detail.cpp#L217-L226): NMS giữa các ứng viên còn lại theo score giảm dần,
   ứng viên mạnh hơn loại ứng viên yếu hơn có IoU > `tai_thr`. Khi score bằng nhau, index lớn hơn được xét
   trước, giống `argsort()[::-1]` của NumPy với input nhỏ.
5. [tracktrack.cpp:182-186](tracktrack.cpp#L182-L186): ứng viên còn lại được `activate()`
   ([tt_track.cpp:63-79](tt_track.cpp#L63-L79)) với ID mới. Track mới ở state `New` (trừ khi
   `min_track_len <= 1`), và chỉ hiển thị ngay nếu là frame 1.

## 8. Hết hạn track Lost

**Code:** [tracktrack.cpp:188-193](tracktrack.cpp#L188-L193)

Track trong `lost_tracks_` có `frame_id - end_frame > track_buffer` (30) → `markRemoved()`.

## 9. `mergePools` và output

**Code:** `mergePools` [tracktrack.cpp:207-231](tracktrack.cpp#L207-L231), giống hệt BoT-SORT

1. [Dòng 211-216](tracktrack.cpp#L211-L216): `tracked = (tracked_tracks_ còn Tracked) ∪ activated ∪ refound`,
   ưu tiên track đứng trước khi trùng ID.
2. [Dòng 217-220](tracktrack.cpp#L217-L220): `lost = (lost_tracks_ − tracked) ∪ newly_lost`, rồi trừ
   `removed_tracks_` **của các frame trước**.
3. [Dòng 222-224](tracktrack.cpp#L222-L224): `removeDuplicateTracks`
   ([track_pool.hpp:93-](../common/track_pool.hpp#L93)): cặp tracked/lost có IoU > 0.85 → bỏ track sống
   ngắn hơn.
4. [Dòng 226-230](tracktrack.cpp#L226-L230): append `newly_removed` vào `removed_tracks_`, giữ tối đa 1000
   track gần nhất.

**Output** ([tracktrack.cpp:196-204](tracktrack.cpp#L196-L204)): các track trong `tracked_tracks_` có
`is_activated` **và** được cập nhật ở chính frame này (`frameId() == frame_id_`), trả về
`TrackOutput(track_id, bbox, score, class_id, detection_index)`. Track được match bằng một recovered
detection sẽ có `detection_index = -1`.

## Tham số

| Tham số | Mặc định | Dùng ở |
|---|---|---|
| `track_high_thresh` | 0.6 | Chia nhóm high, ngưỡng cho recovered ([tracktrack.cpp:53](tracktrack.cpp#L53), [66](tracktrack.cpp#L66)) |
| `track_low_thresh` | 0.25 | Chia nhóm low ([tracktrack.cpp:60](tracktrack.cpp#L60)) |
| `new_track_thresh` | 0.7 | TAI: score tối thiểu để sinh track ([tracktrack_detail.cpp:197](tracktrack_detail.cpp#L197)) |
| `track_buffer` | 30 | Số frame giữ track Lost ([tracktrack.cpp:189](tracktrack.cpp#L189)) |
| `match_thresh` | 0.7 | Gate ban đầu của iterative assignment ([tracktrack.cpp:98](tracktrack.cpp#L98), [126](tracktrack.cpp#L126)) |
| `reduce_step` | 0.05 | Lượng giảm gate mỗi vòng ([tracktrack_detail.cpp:159](tracktrack_detail.cpp#L159)) |
| `lost_match_thr` | 0.0 | Gate của relaxed rebind, 0 = tắt ([tracktrack.cpp:148](tracktrack.cpp#L148)) |
| `conf_weight` | 0.1 | Trọng số confidence distance ([tracktrack_detail.cpp:91](tracktrack_detail.cpp#L91)) |
| `angle_weight` | 0.05 | Trọng số angle distance ([tracktrack_detail.cpp:92-93](tracktrack_detail.cpp#L92-L93)) |
| `penalty_p` | 0.2 | Penalty cho detection low ([tracktrack.cpp:94](tracktrack.cpp#L94)) |
| `penalty_q` | 0.4 | Penalty cho detection recovered ([tracktrack.cpp:94](tracktrack.cpp#L94)) |
| `tai_thr` | 0.55 | IoU của track-aware NMS ([tracktrack_detail.cpp:200](tracktrack_detail.cpp#L200)) |
| `min_track_len` | 3 | Số hit để New → Tracked, 0/1 = xác nhận ngay ([tt_track.cpp:74](tt_track.cpp#L74), [152](tt_track.cpp#L152)) |
| `gmc_method` | `sparseOptFlow` | `sparseOptFlow` hoặc `none` ([tracktrack.cpp:80](tracktrack.cpp#L80)) |
| `class_ema_alpha` | 0.0 | EMA của vector class, 0 = theo detection mới nhất ([class_ema.cpp:33](../common/class_ema.cpp#L33)) |

Các giá trị cố định trong code: IoU tối thiểu 0.1 của cost
([tracktrack_detail.cpp:11](tracktrack_detail.cpp#L11)), `kDeltaT = 3` ([tt_track.hpp:20](tt_track.hpp#L20)),
IoU dedup 0.85, removed buffer 1000 ([tracktrack.cpp:227](tracktrack.cpp#L227)) và GMC downscale 2.

## Khác biệt chính so với BoT-SORT

| | BoT-SORT | TrackTrack |
|---|---|---|
| Thứ tự GMC / predict | predict → GMC | GMC → predict ([tracktrack.cpp:79-86](tracktrack.cpp#L79-L86)) |
| Predict unconfirmed | Không | Có ([tracktrack.cpp:86](tracktrack.cpp#L86)) |
| Chia confirmed/unconfirmed | Theo `is_activated` | Theo state `Tracked`/`New` ([tracktrack.cpp:75](tracktrack.cpp#L75)) |
| Cost | `1 - IoU` (fuse score) | HMIoU + confidence + angle, penalty theo nhóm |
| Matching | Hungarian, 2 bước tuần tự (high rồi low) | Iterative mutual-NN, 1 bước cho cả 3 nhóm |
| Recovered detections | Không | Có (loose NMS) |
| Kalman update | Chuẩn | NSA (noise theo score) |
| Khởi tạo track | Mọi high thừa có `score >= new_track_thresh` | Track-aware NMS |
| Xác nhận track | Ở lần match thứ 2 | Sau `min_track_len` hit |
| Output | Mọi track Tracked + activated | Chỉ track được cập nhật ở frame này |
