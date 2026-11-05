#include "tracking/common/association.hpp"

#include "tracking/common/geometry.hpp"

#include <cmath>
#include <unordered_set>

namespace mot {
namespace {

std::vector<BBox> boxes(const std::vector<TrackPtr>& tracks) {
    std::vector<BBox> result;
    result.reserve(tracks.size());
    for (const auto& track : tracks) {
        result.push_back(track->bbox());
    }
    return result;
}

}

std::vector<std::vector<double>> trackDistance(const std::vector<TrackPtr>& tracks,
                                               const std::vector<TrackPtr>& detections,
                                               bool use_score_fusion) {
    std::vector<std::vector<double>> distances = iouDistance(boxes(tracks), boxes(detections));
    if (use_score_fusion && !distances.empty() && !distances.front().empty()) {
        std::vector<double> scores;
        scores.reserve(detections.size());
        for (const auto& detection : detections) {
            scores.push_back(detection->score());
        }
        for (double score : scores) validateScore(score);
        for (auto& row : distances) {
            for (std::size_t column = 0; column < scores.size(); ++column) {
                if (!std::isfinite(row[column])) {
                    throw std::invalid_argument("cost matrix must contain only finite values");
                }
                row[column] = 1.0 - (1.0 - row[column]) * scores[column];
            }
        }
    }
    return distances;
}

AssignmentResult associate(const std::vector<TrackPtr>& tracks,
                           const std::vector<TrackPtr>& detections,
                           double match_thresh,
                           bool use_score_fusion) {
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
    ids.reserve(first.size() + second.size());
    for (const auto& track : first) {
        ids.insert(track->trackId());
    }
    std::vector<TrackPtr> result = first;
    result.reserve(first.size() + second.size());
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
    ids.reserve(second.size());
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
    for (const auto& track : tracks) {
        track->applyWarp(warp);
    }
}

DuplicateResult removeDuplicateTracks(const std::vector<TrackPtr>& tracked,
                                      const std::vector<TrackPtr>& lost,
                                      double iou_threshold) {
    if (iou_threshold < 0.0 || iou_threshold > 1.0) {
        throw std::invalid_argument("iou_threshold must be in [0, 1]");
    }
    DuplicateResult result{tracked, lost, {}};
    if (tracked.empty() || lost.empty()) {
        return result;
    }
    const auto tracked_boxes = boxes(tracked);
    const auto lost_boxes = boxes(lost);
    std::unordered_set<int> drop_tracked;
    std::unordered_set<int> drop_lost;
    drop_tracked.reserve(tracked.size());
    drop_lost.reserve(lost.size());
    for (std::size_t tracked_index = 0; tracked_index < tracked.size(); ++tracked_index) {
        for (std::size_t lost_index = 0; lost_index < lost.size(); ++lost_index) {
            const double distance = 1.0F - static_cast<float>(iou(tracked_boxes[tracked_index], lost_boxes[lost_index]));
            if (distance >= 1.0 - iou_threshold) {
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

}
