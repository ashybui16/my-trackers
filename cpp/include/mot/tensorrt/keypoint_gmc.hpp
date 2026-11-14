#pragma once

#include "mot/tracking/keypoint_gmc.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot {

// TensorRT backend for gmc_method "dpoint" (serialized engines built from the ONNX models,
// with linear FP32 device bindings). Tensor contract:
// dpoint: input [1,1,H,W] FP32 gray in [0,1]; outputs keypoints [1,N,2] (x,y input pixels),
//         descriptors [1,N,D], scores [1,N].
// matcher: inputs kpts1, descs1 (previous frame), kpts2, descs2 (current frame);
//          outputs matched_kpts1 [1,N,2], matched_kpts2 [1,N,2], valid_mask [1,N].
class TensorRTKeypointGmc final : public IKeypointGmcModel {
public:
    TensorRTKeypointGmc(const std::string& dpoint_engine_path, const std::string& matcher_engine_path);
    ~TensorRTKeypointGmc() override;

    TensorRTKeypointGmc(const TensorRTKeypointGmc&) = delete;
    TensorRTKeypointGmc& operator=(const TensorRTKeypointGmc&) = delete;
    TensorRTKeypointGmc(TensorRTKeypointGmc&&) noexcept;
    TensorRTKeypointGmc& operator=(TensorRTKeypointGmc&&) noexcept;

    cv::Size inputSize() const override;
    bool extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                         std::vector<cv::Point2f>& current) override;
    void reset() override;

    int keypointCount() const;
    int descriptorSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
