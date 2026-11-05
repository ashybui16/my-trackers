#include "tracking/common/track.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

namespace {

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
      score_(detection.score),
      class_id_(detection.class_id),
      detection_index_(detection_index) {}

BBox Track::bbox() const {
    // Giữ kích thước có dấu sau GMC, cùng cách đổi state của Ultralytics.
    return mean_.empty() ? initial_bbox_ : xywhToXyxy(stateXywh(mean_));
}

std::array<double, 4> Track::xywh() const {
    return mean_.empty() ? xyxyToXywh(initial_bbox_) : stateXywh(mean_);
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
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = frame_id == 1;
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

void Track::predict() {
    // Dự đoán frame kế và giữ nguyên kích thước có dấu như Ultralytics.
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        return;
    }
    KalmanState state{mean_, covariance_};
    if (state_ != TrackState::Tracked) {
        state.mean = mean_.clone(); // Only the lost-track mean is modified before predict.
        state.mean.at<double>(6, 0) = 0.0;
        state.mean.at<double>(7, 0) = 0.0;
    }
    const KalmanState predicted = kalman_filter_->predict(state);
    mean_ = predicted.mean;
    covariance_ = predicted.covariance;
}

void Track::applyWarp(const cv::Mat& warp) {
    // Bù chuyển động camera; khôi phục trạng thái cũ nếu phép biến đổi lỗi.
    if (mean_.empty() || covariance_.empty()) {
        return;
    }
    const cv::Mat previous_mean = mean_;
    const cv::Mat previous_covariance = covariance_;
    try {
        warpXywhState(mean_, covariance_, warp);
    } catch (const std::exception&) {
        mean_ = previous_mean;
        covariance_ = previous_covariance;
        return;
    }
    if (!finiteMatrix(mean_) || !finiteMatrix(covariance_)) {
        mean_ = previous_mean;
        covariance_ = previous_covariance;
        return;
    }
    // Giữ nguyên kết quả hữu hạn, kể cả khi phép quay làm w/h âm.
}

void Track::update(const Track& detection_track, int frame_id) {
    // Hiệu chỉnh Kalman từ detection mới, rồi cập nhật metadata của track.
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        throw std::runtime_error("track must be activated before update");
    }
    const std::array<double, 4> measurement = xyxyToXywh(detection_track.bbox());
    const KalmanState updated = kalman_filter_->update(
        KalmanState{mean_, covariance_}, measurement);
    mean_ = updated.mean;
    covariance_ = updated.covariance;
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

}  // namespace mot
