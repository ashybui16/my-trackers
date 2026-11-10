#include "tracking/common/association.hpp"

#include "tracking/common/geometry.hpp"

#include <cmath>

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
    return jointTracks<Track>(first, second);
}

std::vector<TrackPtr> subtractTracks(const std::vector<TrackPtr>& first,
                                     const std::vector<TrackPtr>& second) {
    return subtractTracks<Track>(first, second);
}

void applyGMC(const std::vector<TrackPtr>& tracks, const cv::Mat& warp) {
    applyGMC<Track>(tracks, warp);
}

DuplicateResult removeDuplicateTracks(const std::vector<TrackPtr>& tracked,
                                      const std::vector<TrackPtr>& lost,
                                      double iou_threshold) {
    return removeDuplicateTracks<Track>(tracked, lost, iou_threshold);
}

}
