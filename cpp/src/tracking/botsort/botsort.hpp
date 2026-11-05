#pragma once

#include "mot/tracking/botsort_config.hpp"
#include "mot/tracking/tracker.hpp"
#include "tracking/common/gmc.hpp"
#include "tracking/common/track.hpp"

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace mot {

class BoTSORT final : public ITracker {
public:
    explicit BoTSORT(BoTSORTConfig config = {});

    std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                    const cv::Mat& frame = cv::Mat()) override;
    void reset() override;
    std::string_view name() const override { return BoTSORTConfig::kType; }

    const BoTSORTConfig& config() const { return config_; }
    int frameId() const { return frame_id_; }
    const std::vector<TrackPtr>& trackedTracks() const { return tracked_tracks_; }
    const std::vector<TrackPtr>& lostTracks() const { return lost_tracks_; }
    const std::vector<TrackPtr>& removedTracks() const { return removed_tracks_; }

private:
    BoTSORTConfig config_;
    GMC gmc_;
    std::vector<TrackPtr> tracked_tracks_;
    std::vector<TrackPtr> lost_tracks_;
    std::vector<TrackPtr> removed_tracks_;
    int frame_id_{0};
    int next_track_id_{1};
    KalmanFilterXYWH kalman_filter_;

    std::pair<std::vector<int>, std::vector<int>> firstAssociation(
        const std::vector<TrackPtr>& track_pool,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& refound);
    std::pair<std::vector<int>, std::vector<int>> secondAssociation(
        const std::vector<TrackPtr>& track_pool,
        const std::vector<int>& unmatched_pool_indices,
        const std::vector<int>& unmatched_high_detection_indices,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& refound,
        std::vector<TrackPtr>& newly_lost);
    std::pair<std::vector<int>, std::vector<TrackPtr>> unconfirmedAssociation(
        const std::vector<TrackPtr>& unconfirmed,
        const std::vector<int>& unmatched_detection_indices,
        const std::vector<TrackPtr>& detections,
        std::vector<TrackPtr>& activated,
        std::vector<TrackPtr>& newly_removed);
    void initNewTracks(const std::vector<int>& indices,
                       const std::vector<TrackPtr>& detections,
                       std::vector<TrackPtr>& activated);
    void mergePools(const std::vector<TrackPtr>& activated,
                    const std::vector<TrackPtr>& refound,
                    const std::vector<TrackPtr>& newly_lost,
                    const std::vector<TrackPtr>& newly_removed);
    std::vector<TrackOutput> currentOutputs() const;
};

}
