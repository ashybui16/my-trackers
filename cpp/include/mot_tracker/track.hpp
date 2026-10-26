#pragma once

#include "mot_tracker/geometry.hpp"
#include "mot_tracker/kalman_filter.hpp"
#include "mot_tracker/types.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace mot_tracker {

enum class TrackState {
    NEW = 0,
    TRACKED = 1,
    LOST = 2,
    REMOVED = 3,
    New = NEW,
    Tracked = TRACKED,
    Lost = LOST,
    Removed = REMOVED,
};

// Trạng thái và lịch sử của một đối tượng được theo dõi qua các frame.
class Track {
public:
    // Tạo track tạm từ detection; detection_index là vị trí trong danh sách đầu vào.
    explicit Track(const Detection& detection, int detection_index = -1);

    // Cấp ID và khởi tạo bộ lọc Kalman cho track mới.
    void activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id);
    // Dự đoán vị trí tiếp theo trước khi ghép detection.
    void predict();
    // Bù chuyển động camera vào trạng thái dự đoán.
    void applyWarp(const cv::Mat& warp);
    // Cập nhật track đang theo dõi từ detection đã ghép.
    void update(const Track& detection_track, int frame_id);
    // Khôi phục track bị mất; new_id cho phép cấp lại ID nếu cần.
    void reActivate(const Track& detection_track, int frame_id, bool new_id = false, int track_id = 0);
    // Chuyển trạng thái khi đối tượng vắng mặt hoặc hết thời gian chờ.
    void markLost();
    void markRemoved();

    // Trả hộp hiện tại ở dạng góc hoặc tâm; dùng hộp hợp lệ gần nhất khi cần.
    BBox bbox() const;
    std::array<double, 4> xywh() const;
    const cv::Mat& mean() const { return mean_; }
    const cv::Mat& covariance() const { return covariance_; }

    int trackId() const { return track_id_; }
    double score() const { return score_; }
    int classId() const { return class_id_; }
    int detectionIndex() const { return detection_index_; }
    bool isActivated() const { return is_activated_; }
    TrackState state() const { return state_; }
    int frameId() const { return frame_id_; }
    int startFrame() const { return start_frame_; }
    int endFrame() const { return frame_id_; }
    int age() const { return frame_id_ - start_frame_; }

private:
    // Hộp ban đầu và hộp hợp lệ gần nhất dùng khi trạng thái dự đoán lỗi.
    BBox initial_bbox_{};
    std::array<double, 4> last_valid_xywh_{};
    // Bộ lọc dùng chung do tracker sở hữu; mean_/covariance_ là trạng thái riêng.
    KalmanFilterXYWH* kalman_filter_{nullptr};
    cv::Mat mean_;
    cv::Mat covariance_;
    // ID ổn định của đối tượng và cờ xác nhận track.
    int track_id_{0};
    bool is_activated_{false};
    // Điểm, lớp và chỉ số detection gần nhất.
    TrackState state_{TrackState::New};
    double score_{0.0};
    int class_id_{0};
    int detection_index_{-1};
    // Số lần cập nhật liên tiếp, frame gần nhất và frame khởi đầu.
    int tracklet_len_{0};
    int frame_id_{0};
    int start_frame_{0};
};

using TrackPtr = std::shared_ptr<Track>;

}  // namespace mot_tracker
