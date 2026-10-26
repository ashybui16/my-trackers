#pragma once

#include "mot_tracker/config.hpp"
#include "mot_tracker/gmc.hpp"
#include "mot_tracker/track.hpp"

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace mot_tracker {

class BoTSORT {
public:
    explicit BoTSORT(BoTSORTConfig config = {});

    // Xử lý một frame, ghép detection vào track và trả các track đã xác nhận.
    std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                    const cv::Mat& frame = cv::Mat());
    // Xóa các track và đưa bộ đếm frame/ID về trạng thái ban đầu.
    void reset();

    const BoTSORTConfig& config() const { return config_; }
    int frameId() const { return frame_id_; }
    const std::vector<TrackPtr>& trackedTracks() const { return tracked_tracks_; }
    const std::vector<TrackPtr>& lostTracks() const { return lost_tracks_; }
    const std::vector<TrackPtr>& removedTracks() const { return removed_tracks_; }

private:
    // Cấu hình và bộ bù chuyển động camera dùng suốt chuỗi frame.
    BoTSORTConfig config_;
    GMC gmc_;
    // Các nhóm track đang theo dõi, tạm mất và đã loại bỏ.
    std::vector<TrackPtr> tracked_tracks_;
    std::vector<TrackPtr> lost_tracks_;
    std::vector<TrackPtr> removed_tracks_;
    // Bộ đếm frame, ID kế tiếp và bộ lọc chung cho các track.
    int frame_id_{0};
    int next_track_id_{1};
    KalmanFilterXYWH kalman_filter_;

    // Ghép lần đầu với detection có điểm cao; trả các chỉ số chưa ghép.
    std::pair<std::vector<int>, std::vector<int>> firstAssociation(
        const std::vector<TrackPtr>& track_pool,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& refound);
    // Thử ghép track còn lại với detection điểm thấp; đánh dấu track mới mất.
    std::pair<std::vector<int>, std::vector<int>> secondAssociation(
        const std::vector<TrackPtr>& track_pool,
        const std::vector<int>& unmatched_pool_indices,
        const std::vector<int>& unmatched_high_detection_indices,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& refound,
        std::vector<TrackPtr>& newly_lost);
    // Ghép track chưa xác nhận với detection điểm cao còn lại.
    std::pair<std::vector<int>, std::vector<TrackPtr>> unconfirmedAssociation(
        const std::vector<TrackPtr>& unconfirmed,
        const std::vector<int>& unmatched_detection_indices,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& newly_removed);
    // Khởi tạo track mới từ những detection chưa được sử dụng.
    void initNewTracks(const std::vector<int>& indices,
                       const std::vector<TrackPtr>& detections,
                       std::vector<TrackPtr>& activated);
    // Gộp kết quả các lượt ghép và dọn track mất/quá hạn/trùng lặp.
    void mergePools(const std::vector<TrackPtr>& activated,
                    const std::vector<TrackPtr>& refound,
                    const std::vector<TrackPtr>& newly_lost,
                    const std::vector<TrackPtr>& newly_removed);
    // Chuyển các track đang hoạt động thành kết quả trả về.
    std::vector<TrackOutput> currentOutputs() const;
};

}  // namespace mot_tracker
