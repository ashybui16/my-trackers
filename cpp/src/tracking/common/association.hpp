#pragma once

#include "tracking/common/assignment.hpp"
#include "tracking/common/track.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace mot {

std::vector<std::vector<double>> trackDistance(const std::vector<TrackPtr>& tracks,
                                               const std::vector<TrackPtr>& detections,
                                               bool use_score_fusion);
AssignmentResult associate(const std::vector<TrackPtr>& tracks,
                           const std::vector<TrackPtr>& detections,
                           double match_thresh,
                           bool use_score_fusion);

std::vector<TrackPtr> jointTracks(const std::vector<TrackPtr>& first,
                                  const std::vector<TrackPtr>& second);
std::vector<TrackPtr> subtractTracks(const std::vector<TrackPtr>& first,
                                     const std::vector<TrackPtr>& second);
void applyGMC(const std::vector<TrackPtr>& tracks, const cv::Mat& warp);

struct DuplicateResult {
    std::vector<TrackPtr> tracked;
    std::vector<TrackPtr> lost;
    std::vector<TrackPtr> removed;
};

DuplicateResult removeDuplicateTracks(const std::vector<TrackPtr>& tracked,
                                      const std::vector<TrackPtr>& lost,
                                      double iou_threshold = 0.85);

}
