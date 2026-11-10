#pragma once

#include "mot/core/types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace mot::detail {

struct OutputTransform {
    double scale{1.0};
    int pad_x{0};
    int pad_y{0};
    int original_width{0};
    int original_height{0};
};

struct OutputOptions {
    int class_count{10};
    double conf_threshold{0.10};
    double iou_threshold{0.70};
    int max_det{300};
    OutputTransform transform{};
};

struct RawCandidate {
    std::array<float, 4> box{};
    float score{0.0F};
    int class_id{0};
    int source_index{0};
};
struct OutputWorkspace {
    std::vector<RawCandidate> candidates;
    std::vector<unsigned char> suppressed;
    std::vector<float> areas;
    std::vector<std::size_t> keep;
    std::vector<std::size_t> next_same_class;  // Next sorted candidate of the same class.
    std::vector<std::size_t> last_of_class;
    std::vector<unsigned char> loose_suppressed;
    std::vector<std::size_t> loose_keep;
};

// Looser NMS used to recover boxes the tight NMS dropped, and the IoU at which a recovered box counts
// as a duplicate of a kept one (Ultralytics TrackTrack `_LOOSE_NMS_IOU`, `_LOOSE_NMS_DEDUP_IOU`).
inline constexpr double kLooseNmsIou = 0.95;
inline constexpr double kLooseNmsDedupIou = 0.97;

// When `recovered` is non-null it receives the recovered detections (see TrackTrack).
std::vector<mot::Detection> processOutput(const float* output, int candidate_count,
                                          const OutputOptions& options, OutputWorkspace& workspace,
                                          std::vector<mot::Detection>* recovered = nullptr);

std::vector<mot::Detection> processOutput(const float* output,
                                                  int candidate_count,
                                                  const OutputOptions& options);

}
