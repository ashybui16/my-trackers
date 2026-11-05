#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot {

// Giao diện chung của detector để pipeline không phụ thuộc vào backend (TensorRT, ONNX, ...).
class IDetector {
public:
    virtual ~IDetector() = default;

    // Nhận ảnh BGR CV_8UC3; trả hộp theo tọa độ pixel của ảnh gốc.
    virtual std::vector<Detection> detect(const cv::Mat& bgr_frame) = 0;
    // Tên lớp theo class_id; có thể rỗng nếu không biết.
    virtual const std::vector<std::string>& classNames() const = 0;
};

}  // namespace mot
