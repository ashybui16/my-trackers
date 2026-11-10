#pragma once

#include "mot/detection/config.hpp"
#include "mot/detection/detector.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot {

// The engine uses linear FP32 bindings with a fixed output [1,4+class_count,N].
class TensorRTDetector final : public IDetector {
public:
    TensorRTDetector(const std::string& engine_path, DetectorConfig config = {},
                     std::vector<std::string> class_names = {});
    ~TensorRTDetector() override;

    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;
    TensorRTDetector(TensorRTDetector&&) noexcept;
    TensorRTDetector& operator=(TensorRTDetector&&) noexcept;

    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::vector<Detection> detectWithRecovered(const cv::Mat& frame, std::vector<Detection>& recovered) override;
    const std::vector<std::string>& classNames() const override { return class_names_; }

    const DetectorConfig& config() const { return config_; }
    const std::string& enginePath() const { return engine_path_; }
    const std::string& inputBindingName() const;
    const std::string& outputBindingName() const;

private:
    struct Impl;
    DetectorConfig config_;
    std::string engine_path_;
    std::vector<std::string> class_names_;
    std::unique_ptr<Impl> impl_;
};

}
