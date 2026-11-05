#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot {

class GMC {
public:
    GMC(std::string method = "sparseOptFlow", int downscale = 2);

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

    static cv::Mat identityWarp();
    cv::Mat toGray(const cv::Mat& frame);
    void remember(const cv::Mat& gray, const std::vector<cv::Point2f>& keypoints);
    static void warn(const std::string& message);
};

}
