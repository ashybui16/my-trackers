#pragma once

#include "mot_tracker/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot_inference {

struct DetectorConfig {
    // Kích thước phải khớp input cố định [1,3,H,W] của engine TensorRT.
    int input_width{640};
    int input_height{640};
    // Ngưỡng điểm tin cậy và IoU dùng để lọc detection/NMS.
    double conf_threshold{0.01};
    double iou_threshold{0.70};
    // Số detection tối đa sau lọc và số lớp mà engine dự đoán.
    int max_det{300};
    int class_count{10};

    // Kiểm tra kích thước, số lượng và các ngưỡng đầu vào.
    void validate() const;
};

// Engine dùng binding FP32 tuyến tính, output cố định [1,4+class_count,N].
class TensorRTDetector {
public:
    TensorRTDetector(const std::string& engine_path, DetectorConfig config = {});
    ~TensorRTDetector();

    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;
    TensorRTDetector(TensorRTDetector&&) noexcept;
    TensorRTDetector& operator=(TensorRTDetector&&) noexcept;   

    // Chỉ nhận ảnh BGR 2D không rỗng kiểu CV_8UC3; trả các hộp sau lọc.
    std::vector<mot_tracker::Detection> infer(const cv::Mat& frame);

    const DetectorConfig& config() const { return config_; }
    const std::string& enginePath() const { return engine_path_; }
    const std::string& inputBindingName() const;
    const std::string& outputBindingName() const;

private:
    // Impl giữ tài nguyên TensorRT/CUDA; con trỏ cho phép ẩn chi tiết triển khai.
    struct Impl;
    // Cấu hình đã kiểm tra và đường dẫn engine đang sử dụng.
    DetectorConfig config_;
    std::string engine_path_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mot_inference
