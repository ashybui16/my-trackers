#pragma once

#include "mot_tracker/types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace mot_inference::detail {

// Thông tin letterbox và kích thước frame để đưa box đã chọn về ảnh gốc.
struct OutputTransform {
    double scale{1.0};
    int pad_x{0};
    int pad_y{0};
    int original_width{0};
    int original_height{0};
};

// Các tham số lọc nội bộ; không làm thay đổi cấu hình/public API detector.
struct OutputOptions {
    int class_count{10};
    double conf_threshold{0.01};
    double iou_threshold{0.70};
    int max_det{300};
    OutputTransform transform{};
};

// Per-detector scratch storage; the standalone overload below owns local storage.
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
std::vector<mot_tracker::Detection> processOutput(const float* output, int candidate_count,
                                                const OutputOptions& options, OutputWorkspace& workspace);

// Giải mã output raw YOLO [1,4+class_count,N] và lọc box.
std::vector<mot_tracker::Detection> processOutput(const float* output,
                                                  int candidate_count,
                                                  const OutputOptions& options);

}  // namespace mot_inference::detail
