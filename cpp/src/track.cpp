#include "mot_tracker/track.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mot_tracker {

namespace {

constexpr double kMinimumTrackSize = 1e-6;

bool finiteMatrix(const cv::Mat& matrix) {
    // Quét toàn bộ phần tử để loại NaN và vô cực trước khi dùng trạng thái.
    if (matrix.empty()) {
        return false;
    }
    for (int row = 0; row < matrix.rows; ++row) {
        for (int column = 0; column < matrix.cols; ++column) {
            if (!std::isfinite(matrix.at<double>(row, column))) {
                return false;
            }
        }
    }
    return true;
}

bool validState(const cv::Mat& mean, const cv::Mat& covariance) {
    // Chấp nhận trạng thái Kalman đúng kích thước, hữu hạn và có hộp đủ lớn.
    return mean.rows == 8 && mean.cols == 1 && mean.type() == CV_64F &&
           covariance.rows == 8 && covariance.cols == 8 && covariance.type() == CV_64F &&
           finiteMatrix(mean) && finiteMatrix(covariance) &&
           mean.at<double>(2, 0) > kMinimumTrackSize &&
           mean.at<double>(3, 0) > kMinimumTrackSize;
}

void repairState(cv::Mat& mean, cv::Mat& covariance, const std::array<double, 4>& fallback) {
    // Thay phần trạng thái hỏng bằng hộp hợp lệ gần nhất và vận tốc 0.
    if (mean.rows != 8 || mean.cols != 1 || mean.type() != CV_64F) {
        mean = cv::Mat::zeros(8, 1, CV_64F);
    }
    if (covariance.rows != 8 || covariance.cols != 8 || covariance.type() != CV_64F ||
        !finiteMatrix(covariance)) {
        covariance = cv::Mat::eye(8, 8, CV_64F);
    }
    for (int index = 0; index < 4; ++index) {
        double& value = mean.at<double>(index, 0);
        if (!std::isfinite(value)) {
            value = fallback[static_cast<std::size_t>(index)];
        }
    }
    if (mean.at<double>(2, 0) <= kMinimumTrackSize) {
        mean.at<double>(2, 0) = std::max(fallback[2], kMinimumTrackSize);
    }
    if (mean.at<double>(3, 0) <= kMinimumTrackSize) {
        mean.at<double>(3, 0) = std::max(fallback[3], kMinimumTrackSize);
    }
    for (int index = 4; index < 8; ++index) {
        if (!std::isfinite(mean.at<double>(index, 0))) {
            mean.at<double>(index, 0) = 0.0;
        }
    }
}

std::array<double, 4> stateXywh(const cv::Mat& mean) {
    // Trích bốn thành phần hộp từ vector trạng thái 8 chiều.
    std::array<double, 4> result{};
    for (int index = 0; index < 4; ++index) {
        result[static_cast<std::size_t>(index)] = mean.at<double>(index, 0);
    }
    return result;
}

}  // namespace

Track::Track(const Detection& detection, int detection_index)
    : initial_bbox_(detection.bbox_xyxy),
      last_valid_xywh_(xyxyToXywh(detection.bbox_xyxy)),
      score_(detection.score),
      class_id_(detection.class_id),
      detection_index_(detection_index) {}

BBox Track::bbox() const {
    // Ưu tiên hộp Kalman; nếu hỏng thì trả hộp hợp lệ gần nhất.
    if (mean_.empty()) {
        return initial_bbox_;
    }
    std::array<double, 4> state_box{};
    for (int index = 0; index < 4; ++index) {
        state_box[static_cast<std::size_t>(index)] = mean_.at<double>(index, 0);
    }
    if (!std::isfinite(state_box[0]) || !std::isfinite(state_box[1]) ||
        !std::isfinite(state_box[2]) || !std::isfinite(state_box[3]) ||
        state_box[2] <= kMinimumTrackSize || state_box[3] <= kMinimumTrackSize) {
        return xywhToXyxy(last_valid_xywh_);
    }
    return xywhToXyxy(state_box);
}

std::array<double, 4> Track::xywh() const {
    // Dùng cùng logic dự phòng của bbox() rồi đổi sang dạng tâm.
    return xyxyToXywh(bbox());
}

void Track::activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id) {
    // Cấp ID, khởi tạo trạng thái Kalman và ghi frame bắt đầu.
    if (track_id <= 0) {
        throw std::invalid_argument("track_id must be positive");
    }
    kalman_filter_ = &kalman_filter;
    track_id_ = track_id;
    const auto state = kalman_filter.initiate(xyxyToXywh(initial_bbox_));
    mean_ = state.mean;
    covariance_ = state.covariance;
    last_valid_xywh_ = stateXywh(mean_);
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = frame_id == 1;
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

void Track::predict() {
    // Dự đoán frame kế; nếu trạng thái mới hỏng thì sửa từ hộp cuối cùng.
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        return;
    }
    KalmanState state{mean_.clone(), covariance_.clone()};
    if (state_ != TrackState::Tracked) {
        state.mean.at<double>(6, 0) = 0.0;
        state.mean.at<double>(7, 0) = 0.0;
    }
    const KalmanState predicted = kalman_filter_->predict(state);
    mean_ = predicted.mean;
    covariance_ = predicted.covariance;
    if (!validState(mean_, covariance_)) {
        repairState(mean_, covariance_, last_valid_xywh_);
    }
    last_valid_xywh_ = stateXywh(mean_);
}

void Track::applyWarp(const cv::Mat& warp) {
    // Bù chuyển động camera; khôi phục trạng thái cũ nếu phép biến đổi lỗi.
    if (mean_.empty() || covariance_.empty()) {
        return;
    }
    const cv::Mat previous_mean = mean_.clone();
    const cv::Mat previous_covariance = covariance_.clone();
    try {
        warpXywhState(mean_, covariance_, warp);
    } catch (const std::exception&) {
        mean_ = previous_mean;
        covariance_ = previous_covariance;
        return;
    }
    if (!validState(mean_, covariance_)) {
        mean_ = previous_mean;
        covariance_ = previous_covariance;
        return;
    }
    last_valid_xywh_ = stateXywh(mean_);
}

void Track::update(const Track& detection_track, int frame_id) {
    // Hiệu chỉnh Kalman từ detection mới, rồi cập nhật metadata của track.
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        throw std::runtime_error("track must be activated before update");
    }
    repairState(mean_, covariance_, last_valid_xywh_);
    const std::array<double, 4> measurement = xyxyToXywh(detection_track.bbox());
    const KalmanState updated = kalman_filter_->update(
        KalmanState{mean_.clone(), covariance_.clone()}, measurement);
    mean_ = updated.mean;
    covariance_ = updated.covariance;
    repairState(mean_, covariance_, measurement);
    last_valid_xywh_ = stateXywh(mean_);
    frame_id_ = frame_id;
    ++tracklet_len_;
    state_ = TrackState::Tracked;
    is_activated_ = true;
    score_ = detection_track.score_;
    class_id_ = detection_track.class_id_;
    detection_index_ = detection_track.detection_index_;
}

void Track::reActivate(const Track& detection_track, int frame_id, bool new_id, int track_id) {
    // Cập nhật track bị mất và tùy chọn cấp ID khác khi tìm lại.
    update(detection_track, frame_id);
    if (new_id) {
        if (track_id <= 0) {
            throw std::invalid_argument("track_id is required when assigning a new ID");
        }
        track_id_ = track_id;
    }
    tracklet_len_ = 0;
}

void Track::markLost() {
    // Giữ trạng thái Kalman để còn thử ghép trong các frame tiếp theo.
    state_ = TrackState::Lost;
}

void Track::markRemoved() {
    // Đánh dấu hết thời gian theo dõi; tracker sẽ bỏ track khỏi pool hoạt động.
    state_ = TrackState::Removed;
}

}  // namespace mot_tracker
