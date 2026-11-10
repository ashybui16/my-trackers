#include "tracking/tracktrack/tt_track.hpp"

#include "tracking/common/geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {
namespace {

// Corner (x, y) coordinate indices into an xyxy box, for corners LT, LB, RT, RB.
constexpr std::array<int, 4> kCornerX{0, 0, 2, 2};
constexpr std::array<int, 4> kCornerY{1, 3, 1, 3};

bool finiteMatrix(const cv::Mat& matrix) {
    if (matrix.empty()) return false;
    for (int row = 0; row < matrix.rows; ++row) {
        for (int column = 0; column < matrix.cols; ++column) {
            if (!std::isfinite(matrix.at<double>(row, column))) return false;
        }
    }
    return true;
}

}

TTTrack::TTTrack(const Detection& detection, int detection_index, int min_track_len)
    : initial_bbox_(detection.bbox_xyxy),
      score_(detection.score),
      prev_score_(detection.score),
      class_id_(detection.class_id),
      detection_index_(detection_index),
      min_track_len_(min_track_len) {}

BBox TTTrack::bbox() const {
    if (mean_.empty()) return initial_bbox_;
    return xywhToXyxy({mean_.at<double>(0, 0), mean_.at<double>(1, 0),
                       mean_.at<double>(2, 0), mean_.at<double>(3, 0)});
}

void TTTrack::pushHistory(int frame_id, const BBox& box) {
    if (history_size_ < kHistorySize) {
        history_[static_cast<std::size_t>((history_start_ + history_size_) % kHistorySize)] = {frame_id, box};
        ++history_size_;
    } else {
        history_[static_cast<std::size_t>(history_start_)] = {frame_id, box};
        history_start_ = (history_start_ + 1) % kHistorySize;
    }
}

BBox TTTrack::historyBox(int frame_id, int dt) const {
    const int target = frame_id - dt;
    for (int offset = 0; offset < history_size_; ++offset) {
        const auto& entry = history_[static_cast<std::size_t>((history_start_ + offset) % kHistorySize)];
        if (entry.frame_id == target) return entry.box;
    }
    if (history_size_ > 0) {
        return history_[static_cast<std::size_t>((history_start_ + history_size_ - 1) % kHistorySize)].box;
    }
    return bbox();
}

void TTTrack::activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id) {
    if (track_id <= 0) {
        throw std::invalid_argument("track_id must be positive");
    }
    kalman_filter_ = &kalman_filter;
    track_id_ = track_id;
    const auto state = kalman_filter.initiate(xyxyToXywh(initial_bbox_));
    mean_ = state.mean;
    covariance_ = state.covariance;
    pushHistory(frame_id, bbox());
    tracklet_len_ = 0;
    state_ = min_track_len_ <= 1 ? TrackState::Tracked : TrackState::New;
    // Frame-1 tracks are visible immediately but stay New (unconfirmed) until confirmed.
    is_activated_ = frame_id == 1 || state_ == TrackState::Tracked;
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

void TTTrack::predict() {
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        return;
    }
    KalmanState state{mean_, covariance_};
    if (state_ != TrackState::Tracked) {
        state.mean = mean_.clone();
        state.mean.at<double>(6, 0) = 0.0;
        state.mean.at<double>(7, 0) = 0.0;
    }
    const KalmanState predicted = kalman_filter_->predict(state);
    mean_ = predicted.mean;
    covariance_ = predicted.covariance;
}

void TTTrack::applyWarp(const cv::Mat& warp) {
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
    }
}

void TTTrack::kalmanUpdate(const TTTrack& detection_track) {
    if (kalman_filter_ == nullptr || mean_.empty() || covariance_.empty()) {
        throw std::runtime_error("track must be activated before update");
    }
    // NSA-Kalman: measurement noise scaled by the detection confidence.
    const KalmanState updated = kalman_filter_->update(
        KalmanState{mean_, covariance_}, xyxyToXywh(detection_track.bbox()), detection_track.score_);
    mean_ = updated.mean;
    covariance_ = updated.covariance;
}

void TTTrack::update(const TTTrack& detection_track, int frame_id) {
    frame_id_ = frame_id;
    ++tracklet_len_;
    prev_score_ = score_;
    kalmanUpdate(detection_track);
    const BBox current = detection_track.bbox();
    pushHistory(frame_id, current);

    CornerVelocity velocity{};
    for (int dt = 1; dt <= kDeltaT; ++dt) {
        const BBox past = historyBox(frame_id, dt);
        for (std::size_t corner = 0; corner < 4; ++corner) {
            const double dx = current[static_cast<std::size_t>(kCornerX[corner])] - past[static_cast<std::size_t>(kCornerX[corner])];
            const double dy = current[static_cast<std::size_t>(kCornerY[corner])] - past[static_cast<std::size_t>(kCornerY[corner])];
            const double norm = std::sqrt(dx * dx + dy * dy) + 1e-5;
            velocity[corner][0] = static_cast<float>(static_cast<double>(velocity[corner][0]) + dx / norm / dt);
            velocity[corner][1] = static_cast<float>(static_cast<double>(velocity[corner][1]) + dy / norm / dt);
        }
    }
    for (auto& corner : velocity) {
        corner[0] /= static_cast<float>(kDeltaT);
        corner[1] /= static_cast<float>(kDeltaT);
    }
    velocity_ = velocity;

    score_ = detection_track.score_;
    if (state_ == TrackState::Tracked || tracklet_len_ + 1 >= min_track_len_) {
        state_ = TrackState::Tracked;
        is_activated_ = true;
    }
    class_id_ = detection_track.class_id_;
    detection_index_ = detection_track.detection_index_;
}

void TTTrack::reActivate(const TTTrack& detection_track, int frame_id) {
    prev_score_ = score_;
    kalmanUpdate(detection_track);
    pushHistory(frame_id, bbox());
    score_ = detection_track.score_;
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    class_id_ = detection_track.class_id_;
    detection_index_ = detection_track.detection_index_;
}

}
