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
};
std::vector<mot::Detection> processOutput(const float* output, int candidate_count,
                                                const OutputOptions& options, OutputWorkspace& workspace);

std::vector<mot::Detection> processOutput(const float* output,
                                                  int candidate_count,
                                                  const OutputOptions& options);

}
