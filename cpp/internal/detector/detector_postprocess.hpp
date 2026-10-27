#pragma once

#include "mot_tracker/types.hpp"

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

// Giải mã output batch-1 [1,4+nc,N], [1,N,4+nc], hoặc [1,N,6] và lọc box.
std::vector<mot_tracker::Detection> processOutput(const float* output,
                                                  int dimension_1,
                                                  int dimension_2,
                                                  const OutputOptions& options);

}  // namespace mot_inference::detail
