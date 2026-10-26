#pragma once

#include "mot_tracker/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot_tracker {

class GMC {
public:
    // method chọn cách ước lượng chuyển động; downscale giảm kích thước ảnh đầu vào.
    GMC(std::string method = "sparseOptFlow", int downscale = 2);

    // Ước lượng biến đổi affine 2x3 từ frame trước sang frame hiện tại.
    cv::Mat apply(const cv::Mat& frame);
    // Xóa dữ liệu frame trước để bắt đầu một chuỗi mới.
    void reset();

    const cv::Mat& previousFrame() const { return previous_frame_; }
    const std::vector<cv::Point2f>& previousKeypoints() const { return previous_keypoints_; }
    bool initializedFirstFrame() const { return initialized_first_frame_; }

private:
    // Phương pháp ước lượng và hệ số thu nhỏ ảnh.
    std::string method_;
    int downscale_{2};
    // Ảnh xám và các điểm đặc trưng của frame trước dùng cho optical flow.
    cv::Mat previous_frame_;
    std::vector<cv::Point2f> previous_keypoints_;
    bool initialized_first_frame_{false};

    // Trả ma trận đơn vị khi chưa có hoặc không ước lượng được chuyển động.
    static cv::Mat identityWarp();
    // Chuyển ảnh sang xám và thu nhỏ theo downscale_.
    cv::Mat toGray(const cv::Mat& frame) const;
    // Lưu dữ liệu frame hiện tại để đối chiếu ở lần gọi sau.
    void remember(const cv::Mat& gray, const std::vector<cv::Point2f>& keypoints);
    static void warn(const std::string& message);
};

}  // namespace mot_tracker
