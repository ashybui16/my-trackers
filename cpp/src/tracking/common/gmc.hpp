#pragma once

#include "mot/core/types.hpp"
#include "mot/tracking/keypoint_gmc.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace mot {

class GMC {
public:
    // method: "sparseOptFlow", "dpoint" (needs `model`) or "none". `downscale` only applies to
    // sparseOptFlow; dpoint resizes to the model input instead.
    GMC(std::string method = "sparseOptFlow", int downscale = 2,
        std::unique_ptr<IKeypointGmcModel> model = nullptr);

    cv::Mat apply(const cv::Mat& frame);
    void reset();

    const cv::Mat& previousFrame() const { return previous_frame_; }
    const std::vector<cv::Point2f>& previousKeypoints() const { return previous_keypoints_; }
    bool initializedFirstFrame() const { return initialized_first_frame_; }

private:
    std::string method_;
    int downscale_{2};
    cv::Mat previous_frame_;
    std::vector<cv::Point2f> previous_keypoints_;
    bool initialized_first_frame_{false};
    // Ping-pong current/history buffers; never write to a retained history image.
    cv::Mat current_frame_, gray_buffer_, normalized_buffer_;
    std::vector<cv::Point2f> keypoints_, matched_, previous_good_, current_good_;
    std::vector<unsigned char> status_;
    std::vector<float> error_;
    std::unique_ptr<IKeypointGmcModel> model_;
    cv::Mat model_input_, resized_buffer_;

    static cv::Mat identityWarp();
    cv::Mat applySparseOptFlow(const cv::Mat& frame);
    cv::Mat applyDpoint(const cv::Mat& frame);
    // RANSAC partial affine prev -> cur; identity (with a warning) when it cannot be estimated.
    static cv::Mat estimateWarp(const std::vector<cv::Point2f>& previous,
                                const std::vector<cv::Point2f>& current, double translation_scale);
    cv::Mat toGray(const cv::Mat& frame);
    void remember(const cv::Mat& gray, const std::vector<cv::Point2f>& keypoints);
    static void warn(const std::string& message);
};

}
