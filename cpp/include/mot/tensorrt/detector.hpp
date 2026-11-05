#pragma once

#include "mot/detection/config.hpp"
#include "mot/detection/detector.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot {

// Engine dùng binding FP32 tuyến tính, output cố định [1,4+class_count,N].
class TensorRTDetector final : public IDetector {
public:
    TensorRTDetector(const std::string& engine_path, DetectorConfig config = {},
                     std::vector<std::string> class_names = {});
    ~TensorRTDetector() override;

    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;
    TensorRTDetector(TensorRTDetector&&) noexcept;
    TensorRTDetector& operator=(TensorRTDetector&&) noexcept;

    // Chỉ nhận ảnh BGR 2D không rỗng kiểu CV_8UC3; trả các hộp sau lọc.
    std::vector<Detection> detect(const cv::Mat& frame) override;
    const std::vector<std::string>& classNames() const override { return class_names_; }

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
    std::vector<std::string> class_names_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mot
