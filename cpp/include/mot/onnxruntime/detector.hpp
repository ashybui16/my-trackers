#pragma once

#include "mot/detection/config.hpp"
#include "mot/detection/detector.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot {

// CPU ONNX Runtime detector for raw YOLO models (exported with end2end=False).
// The model must have one FP32 input [1,3,H,W] and one fixed FP32 output [1,4+class_count,N].
class OnnxRuntimeDetector final : public IDetector {
public:
    // intra_op_threads = 0 lets ONNX Runtime choose the thread count.
    OnnxRuntimeDetector(const std::string& model_path, DetectorConfig config = {},
                        std::vector<std::string> class_names = {}, int intra_op_threads = 0);
    ~OnnxRuntimeDetector() override;

    OnnxRuntimeDetector(const OnnxRuntimeDetector&) = delete;
    OnnxRuntimeDetector& operator=(const OnnxRuntimeDetector&) = delete;
    OnnxRuntimeDetector(OnnxRuntimeDetector&&) noexcept;
    OnnxRuntimeDetector& operator=(OnnxRuntimeDetector&&) noexcept;

    std::vector<Detection> detect(const cv::Mat& frame) override;
    const std::vector<std::string>& classNames() const override { return class_names_; }

    const DetectorConfig& config() const { return config_; }
    const std::string& modelPath() const { return model_path_; }
    const std::string& inputName() const;
    const std::string& outputName() const;

private:
    struct Impl;
    DetectorConfig config_;
    std::string model_path_;
    std::vector<std::string> class_names_;
    std::unique_ptr<Impl> impl_;
};

}
