#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace mot {

// Learned keypoint extractor + matcher for GMC (gmc_method: "dpoint").
// Stateful: each call matches the given frame against the frame of the previous call, so one
// instance must serve exactly one tracker. Backends: OnnxRuntimeKeypointGmc, TensorRTKeypointGmc.
class IKeypointGmcModel {
public:
    virtual ~IKeypointGmcModel() = default;

    // Model input size; GMC resizes the grayscale frame to it.
    virtual cv::Size inputSize() const = 0;

    // `image` is CV_32FC1 grayscale in [0, 1] of inputSize(). Extracts its keypoints, then fills
    // prev/cur with the valid matches (input-size pixel coordinates) against the previous call.
    // Returns false, with empty outputs, when there is no previous frame (first call, after reset()).
    virtual bool extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                                 std::vector<cv::Point2f>& current) = 0;

    // Forgets the previous frame.
    virtual void reset() = 0;
};

}
