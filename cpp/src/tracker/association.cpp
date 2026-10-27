#include "mot_tracker/association.hpp"

#include "mot_tracker/geometry.hpp"

#include <unordered_set>

namespace mot_tracker {
namespace {

std::vector<BBox> boxes(const std::vector<TrackPtr>& tracks) {
    // Lấy hộp hiện tại theo đúng thứ tự track để lập ma trận chi phí.
    std::vector<BBox> result;
    result.reserve(tracks.size());
    for (const auto& track : tracks) {
        result.push_back(track->bbox());
    }
    return result;
}

}  // namespace

std::vector<std::vector<double>> trackDistance(const std::vector<TrackPtr>& tracks,
                                               const std::vector<TrackPtr>& detections,
                                               bool use_score_fusion) {
    // Đổi hộp thành ma trận 1 - IoU; tùy chọn giảm chi phí cho detection đáng tin hơn.
    std::vector<std::vector<double>> distances = iouDistance(boxes(tracks), boxes(detections));
    if (use_score_fusion && !distances.empty() && !distances.front().empty()) {
        std::vector<double> scores;
        scores.reserve(detections.size());
        for (const auto& detection : detections) {
            scores.push_back(detection->score());
        }
        distances = fuseScore(distances, scores);
    }
    return distances;
}

AssignmentResult associate(const std::vector<TrackPtr>& tracks,
                           const std::vector<TrackPtr>& detections,
                           double match_thresh,
                           bool use_score_fusion) {
    // Nếu hai phía đều có phần tử, giải phép ghép với ngưỡng chi phí đã chọn.
    if (tracks.empty() || detections.empty()) {
        AssignmentResult result;
        for (std::size_t index = 0; index < tracks.size(); ++index) {
            result.unmatched_rows.push_back(static_cast<int>(index));
        }
        for (std::size_t index = 0; index < detections.size(); ++index) {
            result.unmatched_cols.push_back(static_cast<int>(index));
        }
        return result;
    }
    return linearAssignment(trackDistance(tracks, detections, use_score_fusion), match_thresh);
}

std::vector<TrackPtr> jointTracks(const std::vector<TrackPtr>& first,
                                  const std::vector<TrackPtr>& second) {
    std::unordered_set<int> ids;
    for (const auto& track : first) {
        ids.insert(track->trackId());
    }
    std::vector<TrackPtr> result = first;
    for (const auto& track : second) {
        if (ids.insert(track->trackId()).second) {
            result.push_back(track);
        }
    }
    return result;
}

std::vector<TrackPtr> subtractTracks(const std::vector<TrackPtr>& first,
                                     const std::vector<TrackPtr>& second) {
    std::unordered_set<int> ids;
    for (const auto& track : second) {
        ids.insert(track->trackId());
    }
    std::vector<TrackPtr> result;
    result.reserve(first.size());
    for (const auto& track : first) {
        if (ids.find(track->trackId()) == ids.end()) {
            result.push_back(track);
        }
    }
    return result;
}

void applyGMC(const std::vector<TrackPtr>& tracks, const cv::Mat& warp) {
    // Áp cùng biến đổi camera cho mọi track trong pool.
    for (const auto& track : tracks) {
        track->applyWarp(warp);
    }
}

DuplicateResult removeDuplicateTracks(const std::vector<TrackPtr>& tracked,
                                      const std::vector<TrackPtr>& lost,
                                      double iou_threshold) {
    // So cặp track đang theo dõi và bị mất; giữ quỹ đạo tồn tại lâu hơn khi trùng.
    if (iou_threshold < 0.0 || iou_threshold > 1.0) {
        throw std::invalid_argument("iou_threshold must be in [0, 1]");
    }
    DuplicateResult result{tracked, lost, {}};
    if (tracked.empty() || lost.empty()) {
        return result;
    }
    const auto distances = iouDistance(boxes(tracked), boxes(lost));
    std::unordered_set<int> drop_tracked;
    std::unordered_set<int> drop_lost;
    for (std::size_t tracked_index = 0; tracked_index < tracked.size(); ++tracked_index) {
        for (std::size_t lost_index = 0; lost_index < lost.size(); ++lost_index) {
            if (distances[tracked_index][lost_index] >= 1.0 - iou_threshold) {
                continue;
            }
            const auto& active = tracked[tracked_index];
            const auto& stale = lost[lost_index];
            const int active_life = active->frameId() - active->startFrame();
            const int stale_life = stale->frameId() - stale->startFrame();
            if (active_life > stale_life) {
                drop_lost.insert(static_cast<int>(lost_index));
                result.removed.push_back(stale);
            } else {
                drop_tracked.insert(static_cast<int>(tracked_index));
                result.removed.push_back(active);
            }
        }
    }
    result.tracked.clear();
    result.lost.clear();
    for (std::size_t index = 0; index < tracked.size(); ++index) {
        if (drop_tracked.find(static_cast<int>(index)) == drop_tracked.end()) {
            result.tracked.push_back(tracked[index]);
        }
    }
    for (std::size_t index = 0; index < lost.size(); ++index) {
        if (drop_lost.find(static_cast<int>(index)) == drop_lost.end()) {
            result.lost.push_back(lost[index]);
        }
    }
    return result;
}

}  // namespace mot_tracker
