#pragma once

#include "tracking/common/geometry.hpp"
#include "tracking/common/kalman_filter.hpp"
#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace mot {

enum class TrackState {
    NEW = 0,
    TRACKED = 1,
    LOST = 2,
    REMOVED = 3,
    New = NEW,
    Tracked = TRACKED,
    Lost = LOST,
    Removed = REMOVED,
};

class Track {
public:
    explicit Track(const Detection& detection, int detection_index = -1);

    void activate(KalmanFilterXYWH& kalman_filter, int frame_id, int track_id);
    void predict();
    void applyWarp(const cv::Mat& warp);
    void update(const Track& detection_track, int frame_id);
    void reActivate(const Track& detection_track, int frame_id, bool new_id = false, int track_id = 0);
    void markLost();
    void markRemoved();

    // Box from the current state; w/h may be negative after GMC, as in Ultralytics.
    BBox bbox() const;
    std::array<double, 4> xywh() const;
    const cv::Mat& mean() const { return mean_; }
    const cv::Mat& covariance() const { return covariance_; }

    int trackId() const { return track_id_; }
    double score() const { return score_; }
    int classId() const { return class_id_; }
    int detectionIndex() const { return detection_index_; }
    bool isActivated() const { return is_activated_; }
    TrackState state() const { return state_; }
    int frameId() const { return frame_id_; }
    int startFrame() const { return start_frame_; }
    int endFrame() const { return frame_id_; }
    int age() const { return frame_id_ - start_frame_; }

private:
    BBox initial_bbox_{};
    KalmanFilterXYWH* kalman_filter_{nullptr};
    cv::Mat mean_;
    cv::Mat covariance_;
    int track_id_{0};
    bool is_activated_{false};
    TrackState state_{TrackState::New};
    double score_{0.0};
    int class_id_{0};
    int detection_index_{-1};
    int tracklet_len_{0};
    int frame_id_{0};
    int start_frame_{0};
};

using TrackPtr = std::shared_ptr<Track>;

}
