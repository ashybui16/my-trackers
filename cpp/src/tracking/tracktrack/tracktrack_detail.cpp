#include "tracking/tracktrack/tracktrack_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mot::tracktrack {
namespace {

constexpr float kMinAngleIou = 0.10F;  // Angle cost only where IoU supports the pair; others cost 1.
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<int, 4> kCornerX{0, 0, 2, 2};
constexpr std::array<int, 4> kCornerY{1, 3, 1, 3};

using FloatBox = std::array<float, 4>;

FloatBox toFloat(const BBox& box) {
    return {static_cast<float>(box[0]), static_cast<float>(box[1]),
            static_cast<float>(box[2]), static_cast<float>(box[3])};
}

float iouF(const FloatBox& a, const FloatBox& b) {
    const float inter = std::max(std::min(a[2], b[2]) - std::max(a[0], b[0]), 0.0F) *
                        std::max(std::min(a[3], b[3]) - std::max(a[1], b[1]), 0.0F);
    const float area = (b[2] - b[0]) * (b[3] - b[1]) + (a[2] - a[0]) * (a[3] - a[1]) - inter;
    return inter / (area + 1e-7F);
}

// Vertical IoU (HIoU) of two boxes, clipped to [0, 1].
float verticalIouF(const FloatBox& a, const FloatBox& b) {
    const float overlap = std::min(a[3], b[3]) - std::max(a[1], b[1]);
    const float uni = std::max(a[3], b[3]) - std::min(a[1], b[1]);
    return std::clamp(overlap / (uni + 1e-9F), 0.0F, 1.0F);
}

// Mean corner angle between the track's velocity and its (dt-back box -> detection) motion, times det score.
double angleDistance(const TTTrack& track, const TTTrack& detection, int frame_id) {
    const BBox past = track.historyBox(frame_id, TTTrack::kDeltaT);
    const BBox current = detection.bbox();
    const auto& velocity = track.velocity();
    double sum = 0.0;
    for (std::size_t corner = 0; corner < 4; ++corner) {
        const double dx = current[static_cast<std::size_t>(kCornerX[corner])] - past[static_cast<std::size_t>(kCornerX[corner])];
        const double dy = current[static_cast<std::size_t>(kCornerY[corner])] - past[static_cast<std::size_t>(kCornerY[corner])];
        const double norm = std::sqrt(dx * dx + dy * dy) + 1e-5;
        const double dot = static_cast<double>(velocity[corner][0]) * (dx / norm) +
                           static_cast<double>(velocity[corner][1]) * (dy / norm);
        sum += std::abs(std::acos(std::clamp(dot, -1.0, 1.0)));
    }
    return sum / 4.0 / kPi * static_cast<float>(detection.score());
}

}

float iouF(const BBox& first, const BBox& second) {
    return iouF(toFloat(first), toFloat(second));
}

CostMatrix costMatrix(const std::vector<TTTrackPtr>& tracks,
                      const std::vector<TTTrackPtr>& detections,
                      int frame_id,
                      double conf_weight,
                      double angle_weight) {
    const int rows = static_cast<int>(tracks.size());
    const int cols = static_cast<int>(detections.size());
    CostMatrix cost(rows, cols, 1.0F);
    if (rows == 0 || cols == 0) return cost;

    std::vector<FloatBox> det_boxes;
    std::vector<float> det_scores;
    det_boxes.reserve(detections.size());
    det_scores.reserve(detections.size());
    for (const auto& detection : detections) {
        det_boxes.push_back(toFloat(detection->bbox()));
        det_scores.push_back(static_cast<float>(detection->score()));
    }
    const float conf_w = static_cast<float>(conf_weight);
    for (int row = 0; row < rows; ++row) {
        const TTTrack& track = *tracks[static_cast<std::size_t>(row)];
        const FloatBox track_box = toFloat(track.bbox());
        const float score = static_cast<float>(track.score());
        // First-order score extrapolation.
        const float projected = score + (score - static_cast<float>(track.prevScore()));
        for (int col = 0; col < cols; ++col) {
            const float iou_sim = iouF(track_box, det_boxes[static_cast<std::size_t>(col)]);
            if (!(iou_sim > kMinAngleIou)) {
                continue;  // Stays 1.
            }
            float value = 1.0F - verticalIouF(track_box, det_boxes[static_cast<std::size_t>(col)]) * iou_sim;
            value += conf_w * std::abs(projected - det_scores[static_cast<std::size_t>(col)]);
            value = static_cast<float>(static_cast<double>(value) +
                                       angle_weight * angleDistance(track, *detections[static_cast<std::size_t>(col)], frame_id));
            cost.at(row, col) = std::clamp(value, 0.0F, 1.0F);
        }
    }
    return cost;
}

IterativeResult iterativeAssociate(CostMatrix cost, double match_thresh, double reduce_step) {
    // Same result as recomputing np.argmin over the whole matrix each round (with matched rows and
    // columns set to inf), but each round only scans rows/columns that can still match:
    //  - matched rows/columns are inf, so they are never a finite argmin of an active line;
    //  - a line whose minimum is >= the next gate can never match later (the gate only shrinks and
    //    remaining costs never decrease), and it never blocks a match: if it were the nearest
    //    neighbour of some line, every cost on that line would also be >= the gate.
    IterativeResult result;
    const int rows = cost.rows;
    const int cols = cost.cols;
    constexpr float inf = std::numeric_limits<float>::infinity();
    std::vector<int> active_rows(static_cast<std::size_t>(rows));
    std::vector<int> active_cols(static_cast<std::size_t>(cols));
    std::iota(active_rows.begin(), active_rows.end(), 0);
    std::iota(active_cols.begin(), active_cols.end(), 0);
    std::vector<float> row_min(static_cast<std::size_t>(rows));
    std::vector<int> row_arg(static_cast<std::size_t>(rows));
    std::vector<float> col_min(static_cast<std::size_t>(cols));
    std::vector<int> col_arg(static_cast<std::size_t>(cols));
    std::vector<bool> track_matched(static_cast<std::size_t>(rows), false);
    std::vector<bool> det_matched(static_cast<std::size_t>(cols), false);
    std::vector<int> next;
    while (!active_rows.empty() && !active_cols.empty()) {
        for (int col : active_cols) {
            col_min[static_cast<std::size_t>(col)] = inf;
            col_arg[static_cast<std::size_t>(col)] = -1;
        }
        // Row and column argmins in one row-major pass; strict '<' in ascending order keeps the first
        // index on ties, as np.argmin.
        for (int row : active_rows) {
            const float* line = &cost.values[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)];
            float best = inf;
            int arg = -1;
            for (int col : active_cols) {
                const float value = line[col];
                if (value < best) {
                    best = value;
                    arg = col;
                }
                if (value < col_min[static_cast<std::size_t>(col)]) {
                    col_min[static_cast<std::size_t>(col)] = value;
                    col_arg[static_cast<std::size_t>(col)] = row;
                }
            }
            row_min[static_cast<std::size_t>(row)] = best;
            row_arg[static_cast<std::size_t>(row)] = arg;
        }
        bool matched_any = false;
        for (int row : active_rows) {
            const int col = row_arg[static_cast<std::size_t>(row)];
            if (col >= 0 && col_arg[static_cast<std::size_t>(col)] == row &&
                static_cast<double>(row_min[static_cast<std::size_t>(row)]) < match_thresh) {
                result.matches.emplace_back(row, col);
                track_matched[static_cast<std::size_t>(row)] = true;
                det_matched[static_cast<std::size_t>(col)] = true;
                matched_any = true;
            }
        }
        if (!matched_any) break;
        match_thresh -= reduce_step;
        next.clear();
        for (int row : active_rows) {
            if (!track_matched[static_cast<std::size_t>(row)] &&
                static_cast<double>(row_min[static_cast<std::size_t>(row)]) < match_thresh) {
                next.push_back(row);
            }
        }
        active_rows.swap(next);
        next.clear();
        for (int col : active_cols) {
            if (!det_matched[static_cast<std::size_t>(col)] &&
                static_cast<double>(col_min[static_cast<std::size_t>(col)]) < match_thresh) {
                next.push_back(col);
            }
        }
        active_cols.swap(next);
    }
    for (int row = 0; row < rows; ++row) {
        if (!track_matched[static_cast<std::size_t>(row)]) result.unmatched_tracks.push_back(row);
    }
    for (int col = 0; col < cols; ++col) {
        if (!det_matched[static_cast<std::size_t>(col)]) result.unmatched_dets.push_back(col);
    }
    return result;
}

std::vector<bool> trackAwareNms(const std::vector<TTTrackPtr>& tracks,
                                const std::vector<TTTrackPtr>& detections,
                                double tai_thr,
                                double new_track_thresh) {
    const std::size_t n = detections.size();
    std::vector<bool> allow(n);
    std::vector<float> scores(n);
    std::vector<FloatBox> boxes(n);
    for (std::size_t i = 0; i < n; ++i) {
        scores[i] = static_cast<float>(detections[i]->score());
        boxes[i] = toFloat(detections[i]->bbox());
        allow[i] = scores[i] > static_cast<float>(new_track_thresh);
    }
    if (tracks.size() + n < 2) return allow;
    const float thr = static_cast<float>(tai_thr);
    if (!tracks.empty()) {
        std::vector<FloatBox> track_boxes;
        track_boxes.reserve(tracks.size());
        for (const auto& track : tracks) track_boxes.push_back(toFloat(track->bbox()));
        for (std::size_t i = 0; i < n; ++i) {
            if (!allow[i]) continue;
            for (const auto& box : track_boxes) {
                if (!(iouF(boxes[i], box) <= thr)) {
                    allow[i] = false;
                    break;
                }
            }
        }
    }
    // Descending score. Ties: higher index first, as NumPy's argsort()[::-1] for small inputs
    // (larger inputs use an unstable sort there, so tied-score order may differ).
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
    std::reverse(order.begin(), order.end());
    for (std::size_t i : order) {
        if (!allow[i]) continue;
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i && allow[j] && iouF(boxes[i], boxes[j]) > thr) allow[j] = false;
        }
    }
    return allow;
}

}
