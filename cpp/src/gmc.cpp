#include "mot_tracker/gmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace mot_tracker {

GMC::GMC(std::string method, int downscale)
    : method_(std::move(method)), downscale_(downscale) {
    // Chỉ nhận phương pháp hỗ trợ và tỉ lệ thu nhỏ dương.
    if (method_ != "sparseOptFlow" && method_ != "none") {
        throw std::invalid_argument("method must be 'sparseOptFlow' or 'none'");
    }
    if (downscale_ < 1) {
        throw std::invalid_argument("downscale must be a positive integer");
    }
}

cv::Mat GMC::identityWarp() {
    // Ma trận affine không dịch chuyển dùng khi chưa có ước lượng hợp lệ.
    return cv::Mat::eye(2, 3, CV_64F);
}

cv::Mat GMC::toGray(const cv::Mat& frame) const {
    // Đổi sang ảnh xám phù hợp optical flow rồi thu nhỏ để xử lý nhanh hơn.
    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame;
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        throw std::invalid_argument("frame must have one grayscale or three BGR channels");
    }
    if (gray.depth() != CV_8U && gray.depth() != CV_32F) {
        cv::Mat normalized;
        cv::normalize(gray, normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
        gray = std::move(normalized);
    }
    if (downscale_ > 1) {
        const int width = std::max(1, gray.cols / downscale_);
        const int height = std::max(1, gray.rows / downscale_);
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(width, height));
        gray = std::move(resized);
    }
    return gray;
}

void GMC::remember(const cv::Mat& gray, const std::vector<cv::Point2f>& keypoints) {
    // Lưu bản sao frame và các điểm để đối chiếu ở lần apply kế tiếp.
    previous_frame_ = gray.clone();
    previous_keypoints_ = keypoints;
    initialized_first_frame_ = true;
}

cv::Mat GMC::apply(const cv::Mat& frame) {
    // Tìm điểm đặc trưng, theo dõi sang frame mới, ước lượng affine bằng RANSAC.
    if (frame.empty() || frame.dims != 2 || frame.rows < 1 || frame.cols < 1) {
        throw std::invalid_argument("frame must be a non-empty grayscale or BGR image");
    }
    if (frame.channels() != 1 && frame.channels() != 3) {
        throw std::invalid_argument("frame must have one grayscale or three BGR channels");
    }
    if (method_ == "none") {
        return identityWarp();
    }

    cv::Mat gray;
    std::vector<cv::Point2f> keypoints;
    try {
        gray = toGray(frame);
        cv::goodFeaturesToTrack(gray, keypoints, 400, 0.01, 0.0, cv::Mat(), 3, false, 0.04);
        if (!initialized_first_frame_) {
            remember(gray, keypoints);
            return identityWarp();
        }
        if (previous_keypoints_.empty()) {
            remember(gray, keypoints);
            warn("GMC has no previous features; using identity warp");
            return identityWarp();
        }
        if (keypoints.empty()) {
            remember(gray, keypoints);
            warn("GMC found no features; using identity warp");
            return identityWarp();
        }

        // Optical flow ghép điểm cũ với frame hiện tại; bỏ các điểm theo dõi lỗi.
        std::vector<cv::Point2f> matched;
        std::vector<unsigned char> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(previous_frame_, gray, previous_keypoints_, matched, status, error);
        std::vector<cv::Point2f> previous_good;
        std::vector<cv::Point2f> current_good;
        for (std::size_t index = 0; index < status.size() && index < matched.size(); ++index) {
            if (status[index] != 0) {
                previous_good.push_back(previous_keypoints_[index]);
                current_good.push_back(matched[index]);
            }
        }

        // Chỉ dùng affine khi đủ cặp điểm và mọi hệ số đều hữu hạn.
        cv::Mat warp = identityWarp();
        if (previous_good.size() >= 5) {
            cv::Mat inliers;
            cv::Mat estimated = cv::estimateAffinePartial2D(
                previous_good, current_good, inliers, cv::RANSAC);
            if (!estimated.empty() && estimated.rows == 2 && estimated.cols == 3) {
                cv::Mat converted;
                estimated.convertTo(converted, CV_64F);
                bool finite = true;
                for (int row = 0; row < converted.rows; ++row) {
                    for (int column = 0; column < converted.cols; ++column) {
                        finite = finite && std::isfinite(converted.at<double>(row, column));
                    }
                }
                if (finite) {
                    warp = converted;
                    if (downscale_ > 1) {
                        warp.at<double>(0, 2) *= downscale_;
                        warp.at<double>(1, 2) *= downscale_;
                    }
                } else {
                    warn("GMC affine estimation returned non-finite values; using identity warp");
                }
            } else {
                warn("GMC affine estimation failed; using identity warp");
            }
        } else {
            warn("GMC found too few optical-flow matches; using identity warp");
        }
        remember(gray, keypoints);
        return warp;
    } catch (const cv::Exception& exception) {
        previous_frame_ = gray.empty() ? frame.clone() : gray.clone();
        previous_keypoints_ = keypoints;
        initialized_first_frame_ = true;
        warn(std::string("GMC failed (") + exception.what() + "); using identity warp");
        return identityWarp();
    } catch (const std::exception& exception) {
        previous_frame_ = gray.empty() ? frame.clone() : gray.clone();
        previous_keypoints_ = keypoints;
        initialized_first_frame_ = true;
        warn(std::string("GMC failed (") + exception.what() + "); using identity warp");
        return identityWarp();
    }
}

void GMC::reset() {
    // Bỏ lịch sử frame để lần apply sau bắt đầu từ ma trận đơn vị.
    previous_frame_.release();
    previous_keypoints_.clear();
    initialized_first_frame_ = false;
}

void GMC::warn(const std::string& message) {
    // Ghi cảnh báo khi không ước lượng được chuyển động camera.
    std::cerr << "warning: " << message << '\n';
}

}  // namespace mot_tracker
