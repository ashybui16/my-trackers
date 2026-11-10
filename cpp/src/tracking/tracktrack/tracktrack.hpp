#pragma once

#include "mot/tracking/tracker.hpp"
#include "mot/tracking/tracktrack_config.hpp"
#include "tracking/common/gmc.hpp"
#include "tracking/common/kalman_filter.hpp"
#include "tracking/tracktrack/tt_track.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace mot {

// TrackTrack: track-perspective association (HMIoU + confidence + corner-angle cost, iterative
// assignment over high, low and recovered detections) and track-aware initialization.
// Port of Ultralytics' TRACKTRACK without ReID.
class TrackTrack final : public ITracker {
public:
    explicit TrackTrack(TrackTrackConfig config = {});

    std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                    const cv::Mat& frame = cv::Mat()) override;
    std::vector<TrackOutput> updateWithRecovered(const std::vector<Detection>& detections,
                                                 const std::vector<Detection>& recovered,
                                                 const cv::Mat& frame = cv::Mat()) override;
    bool usesRecoveredDetections() const override { return true; }
    void reset() override;
    std::string_view name() const override { return TrackTrackConfig::kType; }

    const TrackTrackConfig& config() const { return config_; }
    int frameId() const { return frame_id_; }
    const std::vector<TTTrackPtr>& trackedTracks() const { return tracked_tracks_; }
    const std::vector<TTTrackPtr>& lostTracks() const { return lost_tracks_; }
    const std::vector<TTTrackPtr>& removedTracks() const { return removed_tracks_; }

private:
    TrackTrackConfig config_;
    GMC gmc_;
    KalmanFilterXYWH kalman_filter_;
    std::vector<TTTrackPtr> tracked_tracks_;
    std::vector<TTTrackPtr> lost_tracks_;
    std::vector<TTTrackPtr> removed_tracks_;
    int frame_id_{0};
    int next_track_id_{1};

    void mergePools(const std::vector<TTTrackPtr>& activated,
                    const std::vector<TTTrackPtr>& refound,
                    const std::vector<TTTrackPtr>& newly_lost,
                    const std::vector<TTTrackPtr>& newly_removed);
};

}
