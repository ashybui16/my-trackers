#pragma once

#include "mot_tracker/assignment.hpp"
#include "mot_tracker/track.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace mot_tracker {

// Tính ma trận chi phí IoU giữa track và detection; có thể kết hợp điểm tin cậy.
std::vector<std::vector<double>> trackDistance(const std::vector<TrackPtr>& tracks,
                                               const std::vector<TrackPtr>& detections,
                                               bool use_score_fusion);
// Ghép track với detection trong giới hạn chi phí match_thresh.
AssignmentResult associate(const std::vector<TrackPtr>& tracks,
                           const std::vector<TrackPtr>& detections,
                           double match_thresh,
                           bool use_score_fusion);

// Hợp hai danh sách track theo ID, giữ phần tử xuất hiện trước.
std::vector<TrackPtr> jointTracks(const std::vector<TrackPtr>& first,
                                  const std::vector<TrackPtr>& second);
// Lấy các track ở first không có ID trong second.
std::vector<TrackPtr> subtractTracks(const std::vector<TrackPtr>& first,
                                     const std::vector<TrackPtr>& second);
// Bù chuyển động camera cho trạng thái của từng track.
void applyGMC(const std::vector<TrackPtr>& tracks, const cv::Mat& warp);

// Ba nhóm track sau khi xử lý các bản sao trùng lặp.
struct DuplicateResult {
    std::vector<TrackPtr> tracked;
    std::vector<TrackPtr> lost;
    std::vector<TrackPtr> removed;
};

// Loại track trùng giữa nhóm đang theo dõi và nhóm bị mất theo ngưỡng IoU.
DuplicateResult removeDuplicateTracks(const std::vector<TrackPtr>& tracked,
                                      const std::vector<TrackPtr>& lost,
                                      double iou_threshold = 0.85);

}  // namespace mot_tracker
