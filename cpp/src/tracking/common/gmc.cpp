#include "tracking/common/gmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace mot {

GMC::GMC(std::string method, int downscale, std::unique_ptr<IKeypointGmcModel> model)
    : method_(std::move(method)), downscale_(downscale), model_(std::move(model)) {
    if (method_ != "sparseOptFlow" && method_ != "dpoint" && method_ != "none") {
        throw std::invalid_argument("method must be 'sparseOptFlow', 'dpoint' or 'none'");
    }
    if (downscale_ < 1) {
        throw std::invalid_argument("downscale must be a positive integer");
    }
    if (method_ == "dpoint") {
        if (!model_) throw std::invalid_argument("gmc_method 'dpoint' needs a keypoint GMC model");
        const cv::Size size = model_->inputSize();
        if (size.width < 1 || size.height < 1) {
            throw std::invalid_argument("keypoint GMC model reported an empty input size");
        }
    }
}

cv::Mat GMC::identityWarp() {
    return cv::Mat::eye(2, 3, CV_64F);
}

cv::Mat GMC::toGray(const cv::Mat& frame) {
    // A caller may retain a shallow snapshot through previousFrame().
    if (current_frame_.u != nullptr && current_frame_.u->refcount > 1) current_frame_.release();
    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame;
    } else if (frame.channels() == 3) {
        cv::Mat& destination = downscale_ > 1 ? gray_buffer_ : current_frame_;
        cv::cvtColor(frame, destination, cv::COLOR_BGR2GRAY);
        gray = destination;
    } else {
        throw std::invalid_argument("frame must have one grayscale or three BGR channels");
    }
    if (gray.depth() != CV_8U && gray.depth() != CV_32F) {
        cv::normalize(gray, normalized_buffer_, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
        gray = normalized_buffer_;
    }
    if (downscale_ > 1) {
        const int width = std::max(1, gray.cols / downscale_);
        const int height = std::max(1, gray.rows / downscale_);
        cv::resize(gray, current_frame_, cv::Size(width, height));
        gray = current_frame_;
    }
    if (gray.data != current_frame_.data) gray.copyTo(current_frame_);
    return current_frame_;
}

void GMC::remember(const cv::Mat& gray, const std::vector<cv::Point2f>& keypoints) {
    if (gray.data == current_frame_.data) std::swap(previous_frame_, current_frame_);
    else previous_frame_ = gray.clone();
    previous_keypoints_ = keypoints;
    initialized_first_frame_ = true;
}

cv::Mat GMC::apply(const cv::Mat& frame) {
    if (frame.empty() || frame.dims != 2 || frame.rows < 1 || frame.cols < 1) {
        throw std::invalid_argument("frame must be a non-empty grayscale or BGR image");
    }
    if (frame.channels() != 1 && frame.channels() != 3) {
        throw std::invalid_argument("frame must have one grayscale or three BGR channels");
    }
    if (method_ == "none") {
        return identityWarp();
    }
    if (method_ == "dpoint") {
        return applyDpoint(frame);
    }
    return applySparseOptFlow(frame);
}

cv::Mat GMC::estimateWarp(const std::vector<cv::Point2f>& previous,
                          const std::vector<cv::Point2f>& current, double translation_scale) {
    if (previous.size() < 5) {
        warn("GMC found too few matches; using identity warp");
        return identityWarp();
    }
    cv::Mat inliers;
    const cv::Mat estimated = cv::estimateAffinePartial2D(previous, current, inliers, cv::RANSAC);
    if (estimated.empty() || estimated.rows != 2 || estimated.cols != 3) {
        warn("GMC affine estimation failed; using identity warp");
        return identityWarp();
    }
    cv::Mat warp;
    estimated.convertTo(warp, CV_64F);
    for (int row = 0; row < warp.rows; ++row) {
        for (int column = 0; column < warp.cols; ++column) {
            if (!std::isfinite(warp.at<double>(row, column))) {
                warn("GMC affine estimation returned non-finite values; using identity warp");
                return identityWarp();
            }
        }
    }
    warp.at<double>(0, 2) *= translation_scale;
    warp.at<double>(1, 2) *= translation_scale;
    return warp;
}

cv::Mat GMC::applyDpoint(const cv::Mat& frame) {
    const cv::Size size = model_->inputSize();
    auto& previous = previous_good_;
    auto& current = current_good_;
    previous.clear();
    current.clear();
    try {
        cv::Mat gray = frame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray_buffer_, cv::COLOR_BGR2GRAY);
            gray = gray_buffer_;
        }
        cv::Mat resized = gray;
        if (gray.size() != size) {
            cv::resize(gray, resized_buffer_, size, 0.0, 0.0, cv::INTER_AREA);
            resized = resized_buffer_;
        }
        if (resized.depth() == CV_8U) {
            resized.convertTo(model_input_, CV_32F, 1.0 / 255.0);
        } else {
            cv::normalize(resized, model_input_, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
        }
        const bool has_previous = model_->extractAndMatch(model_input_, previous, current);
        initialized_first_frame_ = true;
        if (!has_previous) return identityWarp();
    } catch (const std::exception& exception) {
        // The model may hold a half-updated previous frame; start over from the next frame.
        model_->reset();
        initialized_first_frame_ = false;
        warn(std::string("GMC keypoint model failed (") + exception.what() + "); using identity warp");
        return identityWarp();
    }
    if (previous.size() != current.size()) {
        warn("keypoint GMC model returned unpaired matches; using identity warp");
        return identityWarp();
    }
    // Model pixels -> frame pixels, with pixel centers aligned as in cv::resize.
    const float scale_x = static_cast<float>(frame.cols) / static_cast<float>(size.width);
    const float scale_y = static_cast<float>(frame.rows) / static_cast<float>(size.height);
    const auto to_frame = [&](cv::Point2f& point) {
        point.x = (point.x + 0.5F) * scale_x - 0.5F;
        point.y = (point.y + 0.5F) * scale_y - 0.5F;
    };
    for (auto& point : previous) to_frame(point);
    for (auto& point : current) to_frame(point);
    try {
        return estimateWarp(previous, current, 1.0);
    } catch (const std::exception& exception) {
        warn(std::string("GMC failed (") + exception.what() + "); using identity warp");
        return identityWarp();
    }
}

cv::Mat GMC::applySparseOptFlow(const cv::Mat& frame) {
    cv::Mat gray;
    auto& keypoints = keypoints_;
    keypoints.clear();
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

        auto& matched = matched_;
        auto& status = status_;
        auto& error = error_;
        matched.clear();
        status.clear();
        error.clear();
        cv::calcOpticalFlowPyrLK(previous_frame_, gray, previous_keypoints_, matched, status, error);
        auto& previous_good = previous_good_;
        auto& current_good = current_good_;
        previous_good.clear();
        current_good.clear();
        previous_good.reserve(status.size());
        current_good.reserve(status.size());
        for (std::size_t index = 0; index < status.size() && index < matched.size(); ++index) {
            if (status[index] != 0) {
                previous_good.push_back(previous_keypoints_[index]);
                current_good.push_back(matched[index]);
            }
        }

        if (previous_good.size() < 5) {
            remember(gray, keypoints);
            warn("GMC found too few optical-flow matches; using identity warp");
            return identityWarp();
        }
        const cv::Mat warp = estimateWarp(previous_good, current_good, static_cast<double>(downscale_));
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
    previous_frame_.release();
    current_frame_.release();
    gray_buffer_.release();
    normalized_buffer_.release();
    keypoints_.clear(); matched_.clear(); status_.clear(); error_.clear();
    previous_good_.clear(); current_good_.clear();
    previous_keypoints_.clear();
    model_input_.release();
    resized_buffer_.release();
    if (model_) model_->reset();
    initialized_first_frame_ = false;
}

void GMC::warn(const std::string& message) {
    std::cerr << "warning: " << message << '\n';
}

}
