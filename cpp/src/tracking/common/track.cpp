#include "tracking/common/track.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

namespace {

bool finiteMatrix(const cv::Mat& matrix) {
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
    std::array<double, 4> result{};
    for (int index = 0; index < 4; ++index) {
        result[static_cast<std::size_t>(index)] = mean.at<double>(index, 0);
    }
    return result;
}

// Detector scores when available, otherwise score at class_id (one-hot).
std::vector<double> classVectorFrom(const Detection& detection) {
    if (!detection.class_scores.empty()) {
        std::vector<double> scores(detection.class_scores.begin(), detection.class_scores.end());
        if (detection.class_id >= 0 && static_cast<std::size_t>(detection.class_id) >= scores.size()) {
            scores.resize(static_cast<std::size_t>(detection.class_id) + 1, 0.0);
            scores[static_cast<std::size_t>(detection.class_id)] = detection.score;
        }
        return scores;
    }
    if (detection.class_id < 0) {
        return {};
    }
    std::vector<double> one_hot(static_cast<std::size_t>(detection.class_id) + 1, 0.0);
    one_hot.back() = detection.score;
    return one_hot;
}

}

Track::Track(const Detection& detection, int detection_index, double class_ema_alpha)
    : initial_bbox_(detection.bbox_xyxy),
      score_(detection.score),
      class_id_(detection.class_id),
      class_vector_(classVectorFrom(detection)),
      class_ema_alpha_(class_ema_alpha),
      detection_index_(detection_index) {}

void Track::updateClass(const Track& detection_track) {
    const std::vector<double>& observed = detection_track.class_vector_;
    if (class_ema_alpha_ <= 0.0 || observed.empty() || class_vector_.empty()) {
        class_vector_ = observed;
        class_id_ = detection_track.class_id_;
        return;
    }
    if (class_vector_.size() < observed.size()) {
        class_vector_.resize(observed.size(), 0.0);
    }
    for (std::size_t index = 0; index < class_vector_.size(); ++index) {
        const double value = index < observed.size() ? observed[index] : 0.0;
        class_vector_[index] = class_ema_alpha_ * class_vector_[index] + (1.0 - class_ema_alpha_) * value;
    }
    // First maximum wins ties.
    std::size_t best = 0;
    for (std::size_t index = 1; index < class_vector_.size(); ++index) {
        if (class_vector_[index] > class_vector_[best]) {
            best = index;
        }
    }
    class_id_ = static_cast<int>(best);
}

BBox Track::bbox() const {
    return mean_.empty() ? initial_bbox_ : xywhToXyxy(stateXywh(mean_));
}

std::array<double, 4> Track::xywh() const {
    return mean_.empty() ? xyxyToXywh(initial_bbox_) : stateXywh(mean_);
}

void Track::activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id) {
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
}

void Track::update(const Track& detection_track, int frame_id) {
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
    updateClass(detection_track);
    detection_index_ = detection_track.detection_index_;
}

void Track::reActivate(const Track& detection_track, int frame_id, bool new_id, int track_id) {
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
    state_ = TrackState::Lost;
}

void Track::markRemoved() {
    state_ = TrackState::Removed;
}

}
