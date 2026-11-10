#pragma once

#include "mot/core/types.hpp"
#include "tracking/common/kalman_filter.hpp"
#include "tracking/common/track.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <memory>

namespace mot {

// Single-object track for TrackTrack (Ultralytics `TTSTrack`): XYWH Kalman state with NSA updates,
// a short box history, per-corner unit velocity, and score history for confidence projection.
class TTTrack {
public:
    static constexpr int kDeltaT = 3;
    using CornerVelocity = std::array<std::array<float, 2>, 4>;  // (dx, dy) for corners LT, LB, RT, RB; float32 as in Ultralytics.

    TTTrack(const Detection& detection, int detection_index, int min_track_len);

    void activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id);
    void predict();
    void applyWarp(const cv::Mat& warp);
    void update(const TTTrack& detection_track, int frame_id);
    void reActivate(const TTTrack& detection_track, int frame_id);
    void markLost() { state_ = TrackState::Lost; }
    void markRemoved() { state_ = TrackState::Removed; }

    // Box `dt` frames before `frame_id`; else the newest history box; else the current box.
    BBox historyBox(int frame_id, int dt) const;

    BBox bbox() const;
    int trackId() const { return track_id_; }
    double score() const { return score_; }
    double prevScore() const { return prev_score_; }
    int classId() const { return class_id_; }
    int detectionIndex() const { return detection_index_; }
    bool isActivated() const { return is_activated_; }
    TrackState state() const { return state_; }
    int frameId() const { return frame_id_; }
    int startFrame() const { return start_frame_; }
    int endFrame() const { return frame_id_; }
    const CornerVelocity& velocity() const { return velocity_; }

private:
    struct HistoryEntry {
        int frame_id{0};
        BBox box{};
    };
    static constexpr int kHistorySize = kDeltaT + 1;

    void kalmanUpdate(const TTTrack& detection_track);
    void pushHistory(int frame_id, const BBox& box);

    BBox initial_bbox_{};
    KalmanFilterXYWH* kalman_filter_{nullptr};
    cv::Mat mean_;
    cv::Mat covariance_;
    int track_id_{0};
    bool is_activated_{false};
    TrackState state_{TrackState::New};
    double score_{0.0};
    double prev_score_{0.0};
    int class_id_{0};
    int detection_index_{-1};
    int min_track_len_{3};
    int tracklet_len_{0};
    int frame_id_{0};
    int start_frame_{0};
    CornerVelocity velocity_{};
    // Ring buffer, oldest first from history_start_.
    std::array<HistoryEntry, kHistorySize> history_{};
    int history_start_{0};
    int history_size_{0};
};

using TTTrackPtr = std::shared_ptr<TTTrack>;

}
