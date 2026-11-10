#pragma once

// TrackTrack association building blocks (exposed for tests). Costs are float32, like the Ultralytics
// NumPy implementation, so assignment decisions match it.

#include "tracking/tracktrack/tt_track.hpp"

#include <utility>
#include <vector>

namespace mot::tracktrack {

// Row-major (rows x cols) float32 matrix.
struct CostMatrix {
    int rows{0};
    int cols{0};
    std::vector<float> values;

    CostMatrix() = default;
    CostMatrix(int row_count, int col_count, float fill)
        : rows(row_count), cols(col_count), values(static_cast<std::size_t>(row_count) * col_count, fill) {}
    float& at(int row, int col) { return values[static_cast<std::size_t>(row) * cols + col]; }
    float at(int row, int col) const { return values[static_cast<std::size_t>(row) * cols + col]; }
};

struct IterativeResult {
    std::vector<std::pair<int, int>> matches;  // In match order: by round, then track index.
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_dets;
};

// float32 IoU with Ultralytics' eps (bbox_ioa(iou=True)).
float iouF(const BBox& first, const BBox& second);

// Multi-cue cost: (1 - HIoU*IoU) + conf_weight*|projected score - det score| + angle_weight*angle distance,
// where pairs with IoU <= 0.1 get cost 1. Clipped to [0, 1].
CostMatrix costMatrix(const std::vector<TTTrackPtr>& tracks,
                      const std::vector<TTTrackPtr>& detections,
                      int frame_id,
                      double conf_weight,
                      double angle_weight);

// Mutual-nearest matching; the gate shrinks by reduce_step after each round.
IterativeResult iterativeAssociate(CostMatrix cost, double match_thresh, double reduce_step);

// Track-aware NMS: a detection may start a track when score > new_track_thresh, its IoU with every
// active track is <= tai_thr, and no stronger allowed detection overlaps it by more than tai_thr.
std::vector<bool> trackAwareNms(const std::vector<TTTrackPtr>& tracks,
                                const std::vector<TTTrackPtr>& detections,
                                double tai_thr,
                                double new_track_thresh);

}
