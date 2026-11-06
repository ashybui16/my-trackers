#include "tracking/botsort/botsort.hpp"

#include "tracking/common/association.hpp"

#include <stdexcept>

namespace mot {

BoTSORT::BoTSORT(BoTSORTConfig config)
    : config_(std::move(config)), gmc_(config_.gmc_method, config_.gmc_downscale) {
    config_.validate();
}

void BoTSORT::reset() {
    tracked_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
    frame_id_ = 0;
    next_track_id_ = 1;
    kalman_filter_ = KalmanFilterXYWH();
    gmc_.reset();
}

std::vector<TrackOutput> BoTSORT::update(const std::vector<Detection>& detections,
                                         const cv::Mat& frame) {
    if (config_.gmc_method == "sparseOptFlow" && frame.empty()) {
        throw std::invalid_argument("frame is required when gmc_method='sparseOptFlow'");
    }
    ++frame_id_;
    std::vector<TrackPtr> high_detections;
    std::vector<TrackPtr> low_detections;
    high_detections.reserve(detections.size());
    low_detections.reserve(detections.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        if (!(detection.score >= config_.track_high_thresh || detection.score > config_.track_low_thresh)) continue;
        auto track = std::make_shared<Track>(detection, static_cast<int>(index), config_.class_ema_alpha);
        if (detection.score >= config_.track_high_thresh) {
            high_detections.push_back(std::move(track));
        } else if (detection.score > config_.track_low_thresh) {
            low_detections.push_back(std::move(track));
        }
    }

    std::vector<TrackPtr> activated;
    std::vector<TrackPtr> refound;
    std::vector<TrackPtr> newly_lost;
    std::vector<TrackPtr> newly_removed;

    std::vector<TrackPtr> unconfirmed;
    std::vector<TrackPtr> confirmed;
    confirmed.reserve(tracked_tracks_.size());
    unconfirmed.reserve(tracked_tracks_.size());
    for (const auto& track : tracked_tracks_) {
        if (track->isActivated()) {
            confirmed.push_back(track);
        } else {
            unconfirmed.push_back(track);
        }
    }
    const std::vector<TrackPtr> track_pool = jointTracks(confirmed, lost_tracks_);
    for (const auto& track : track_pool) {
        track->predict();
    }
    if (config_.gmc_method == "sparseOptFlow") {
        const cv::Mat warp = gmc_.apply(frame);
        applyGMC(track_pool, warp);
        applyGMC(unconfirmed, warp);
    }

    const auto first = firstAssociation(track_pool, high_detections, activated, refound);
    const auto second = secondAssociation(track_pool,
                                          first.first,
                                          first.second,
                                          low_detections,
                                          activated,
                                          refound,
                                          newly_lost);
    const auto unconfirmed_result = unconfirmedAssociation(unconfirmed,
                                                            second.second,
                                                            high_detections,
                                                            activated,
                                                            newly_removed);
    initNewTracks(unconfirmed_result.first, unconfirmed_result.second, activated);

    for (const auto& track : lost_tracks_) {
        if (frame_id_ - track->endFrame() > config_.track_buffer) {
            track->markRemoved();
            newly_removed.push_back(track);
        }
    }
    mergePools(activated, refound, newly_lost, newly_removed);
    return currentOutputs();
}

std::pair<std::vector<int>, std::vector<int>> BoTSORT::firstAssociation(
    const std::vector<TrackPtr>& track_pool,
    const std::vector<TrackPtr>& detections,
    std::vector<TrackPtr>& activated,
    std::vector<TrackPtr>& refound) {
    const AssignmentResult result = associate(track_pool, detections, config_.match_thresh, config_.fuse_score);
    for (const auto& match : result.matches) {
        const auto& track = track_pool[static_cast<std::size_t>(match.first)];
        const auto& detection = detections[static_cast<std::size_t>(match.second)];
        if (track->state() == TrackState::Tracked) {
            track->update(*detection, frame_id_);
            activated.push_back(track);
        } else {
            track->reActivate(*detection, frame_id_);
            refound.push_back(track);
        }
    }
    return {result.unmatched_rows, result.unmatched_cols};
}

std::pair<std::vector<int>, std::vector<int>> BoTSORT::secondAssociation(
    const std::vector<TrackPtr>& track_pool,
    const std::vector<int>& unmatched_pool_indices,
    const std::vector<int>& unmatched_high_detection_indices,
    const std::vector<TrackPtr>& detections,
    std::vector<TrackPtr>& activated,
    std::vector<TrackPtr>& /*refound*/,
    std::vector<TrackPtr>& newly_lost) {
    std::vector<TrackPtr> remaining_active;
    for (int index : unmatched_pool_indices) {
        const auto& track = track_pool[static_cast<std::size_t>(index)];
        if (track->state() == TrackState::Tracked) {
            remaining_active.push_back(track);
        }
    }
    std::vector<int> unmatched_active;
    if (!remaining_active.empty() && !detections.empty()) {
        const AssignmentResult result = associate(remaining_active, detections, 0.5, false);
        unmatched_active = result.unmatched_rows;
        for (const auto& match : result.matches) {
            const auto& track = remaining_active[static_cast<std::size_t>(match.first)];
            track->update(*detections[static_cast<std::size_t>(match.second)], frame_id_);
            activated.push_back(track);
        }
    } else {
        for (std::size_t index = 0; index < remaining_active.size(); ++index) {
            unmatched_active.push_back(static_cast<int>(index));
        }
    }
    for (int index : unmatched_active) {
        const auto& track = remaining_active[static_cast<std::size_t>(index)];
        if (track->state() != TrackState::Lost) {
            track->markLost();
            newly_lost.push_back(track);
        }
    }
    return {unmatched_pool_indices, unmatched_high_detection_indices};
}

std::pair<std::vector<int>, std::vector<TrackPtr>> BoTSORT::unconfirmedAssociation(
    const std::vector<TrackPtr>& unconfirmed,
    const std::vector<int>& unmatched_detection_indices,
    const std::vector<TrackPtr>& detections,
    std::vector<TrackPtr>& activated,
    std::vector<TrackPtr>& newly_removed) {
    std::vector<TrackPtr> remaining;
    remaining.reserve(unmatched_detection_indices.size());
    for (int index : unmatched_detection_indices) {
        remaining.push_back(detections[static_cast<std::size_t>(index)]);
    }
    if (unconfirmed.empty()) {
        std::vector<int> indices;
        indices.reserve(remaining.size());
        for (std::size_t index = 0; index < remaining.size(); ++index) {
            indices.push_back(static_cast<int>(index));
        }
        return {indices, remaining};
    }
    const AssignmentResult result = associate(unconfirmed, remaining, 0.7, config_.fuse_score);
    for (const auto& match : result.matches) {
        const auto& track = unconfirmed[static_cast<std::size_t>(match.first)];
        track->update(*remaining[static_cast<std::size_t>(match.second)], frame_id_);
        activated.push_back(track);
    }
    for (int index : result.unmatched_rows) {
        const auto& track = unconfirmed[static_cast<std::size_t>(index)];
        track->markRemoved();
        newly_removed.push_back(track);
    }
    return {result.unmatched_cols, remaining};
}

void BoTSORT::initNewTracks(const std::vector<int>& indices,
                            const std::vector<TrackPtr>& detections,
                            std::vector<TrackPtr>& activated) {
    for (int index : indices) {
        const auto& track = detections[static_cast<std::size_t>(index)];
        if (track->score() < config_.new_track_thresh) {
            continue;
        }
        track->activate(kalman_filter_, frame_id_, next_track_id_++);
        activated.push_back(track);
    }
}

void BoTSORT::mergePools(const std::vector<TrackPtr>& activated,
                         const std::vector<TrackPtr>& refound,
                         const std::vector<TrackPtr>& newly_lost,
                         const std::vector<TrackPtr>& newly_removed) {
    std::vector<TrackPtr> tracked;
    for (const auto& track : tracked_tracks_) {
        if (track->state() == TrackState::Tracked) {
            tracked.push_back(track);
        }
    }
    tracked = jointTracks(tracked, activated);
    tracked = jointTracks(tracked, refound);
    std::vector<TrackPtr> lost = subtractTracks(lost_tracks_, tracked);
    lost.insert(lost.end(), newly_lost.begin(), newly_lost.end());
    // Prune using the previous frame's history: a just-REMOVED track can still be matched next frame.
    lost = subtractTracks(lost, removed_tracks_);

    DuplicateResult duplicate = removeDuplicateTracks(tracked, lost);
    tracked_tracks_ = std::move(duplicate.tracked);
    lost_tracks_ = std::move(duplicate.lost);

    // Append, then keep only the latest 1000 entries, in the same order as Ultralytics.
    removed_tracks_.insert(removed_tracks_.end(), newly_removed.begin(), newly_removed.end());
    constexpr std::size_t removed_buffer = 1000;
    if (removed_tracks_.size() > removed_buffer) {
        removed_tracks_.erase(removed_tracks_.begin(), removed_tracks_.end() - removed_buffer);
    }
}

std::vector<TrackOutput> BoTSORT::currentOutputs() const {
    std::vector<TrackOutput> outputs;
    outputs.reserve(tracked_tracks_.size());
    for (const auto& track : tracked_tracks_) {
        if (track->state() != TrackState::Tracked || !track->isActivated()) {
            continue;
        }
        outputs.emplace_back(track->trackId(), track->bbox(), track->score(), track->classId());
    }
    return outputs;
}

}
