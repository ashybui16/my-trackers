#include "tracking/tracktrack/tracktrack.hpp"

#include "tracking/common/track_pool.hpp"
#include "tracking/tracktrack/tracktrack_detail.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace mot {

using tracktrack::CostMatrix;
using tracktrack::costMatrix;
using tracktrack::iterativeAssociate;
using tracktrack::trackAwareNms;

TrackTrack::TrackTrack(TrackTrackConfig config, std::unique_ptr<IKeypointGmcModel> gmc_model)
    : config_(std::move(config)), gmc_(config_.gmc_method, 2, std::move(gmc_model)) {
    config_.validate();
}

void TrackTrack::reset() {
    tracked_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
    frame_id_ = 0;
    next_track_id_ = 1;
    kalman_filter_ = KalmanFilterXYWH();
    gmc_.reset();
}

std::vector<TrackOutput> TrackTrack::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    return updateWithRecovered(detections, {}, frame);
}

std::vector<TrackOutput> TrackTrack::updateWithRecovered(const std::vector<Detection>& detections,
                                                         const std::vector<Detection>& recovered,
                                                         const cv::Mat& frame) {
    if (config_.gmc_method != "none" && frame.empty()) {
        throw std::invalid_argument("frame is required when gmc_method='" + config_.gmc_method + "'");
    }
    ++frame_id_;
    std::vector<TTTrackPtr> activated;
    std::vector<TTTrackPtr> refound;
    std::vector<TTTrackPtr> newly_lost;
    std::vector<TTTrackPtr> newly_removed;

    // Detection buckets: high, low, then recovered (loose-NMS) detections above the high threshold.
    std::vector<TTTrackPtr> all_dets;
    all_dets.reserve(detections.size() + recovered.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
        if (detections[index].score >= config_.track_high_thresh) {
            all_dets.push_back(std::make_shared<TTTrack>(detections[index], static_cast<int>(index), config_.min_track_len, config_.class_ema_alpha));
        }
    }
    const std::size_t n_high = all_dets.size();
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const double score = detections[index].score;
        if (score > config_.track_low_thresh && score < config_.track_high_thresh) {
            all_dets.push_back(std::make_shared<TTTrack>(detections[index], static_cast<int>(index), config_.min_track_len, config_.class_ema_alpha));
        }
    }
    const std::size_t n_high_low = all_dets.size();
    for (const auto& detection : recovered) {
        if (detection.score > config_.track_high_thresh) {
            all_dets.push_back(std::make_shared<TTTrack>(detection, -1, config_.min_track_len, config_.class_ema_alpha));
        }
    }

    // Route by state: frame-1 tracks are visible (activated) but still New, so they stay unconfirmed.
    std::vector<TTTrackPtr> unconfirmed;
    std::vector<TTTrackPtr> confirmed;
    for (const auto& track : tracked_tracks_) {
        (track->state() == TrackState::Tracked ? confirmed : unconfirmed).push_back(track);
    }
    const std::vector<TTTrackPtr> pool = jointTracks<TTTrack>(confirmed, lost_tracks_);

    // GMC before prediction (unlike BoT-SORT).
    if (config_.gmc_method != "none") {
        const cv::Mat warp = gmc_.apply(frame);
        applyGMC<TTTrack>(pool, warp);
        applyGMC<TTTrack>(unconfirmed, warp);
    }
    for (const auto& track : pool) track->predict();
    for (const auto& track : unconfirmed) track->predict();

    // Main association: pool vs all detections, with per-bucket penalties.
    CostMatrix cost = costMatrix(pool, all_dets, frame_id_, config_.conf_weight, config_.angle_weight);
    const float penalty_p = static_cast<float>(config_.penalty_p);
    const float penalty_q = static_cast<float>(config_.penalty_q);
    for (int row = 0; row < cost.rows; ++row) {
        for (int col = static_cast<int>(n_high); col < cost.cols; ++col) {
            const float penalty = col < static_cast<int>(n_high_low) ? penalty_p : penalty_q;
            cost.at(row, col) = std::clamp(cost.at(row, col) + penalty, 0.0F, 1.0F);
        }
    }
    const auto main = iterativeAssociate(std::move(cost), config_.match_thresh, config_.reduce_step);
    for (const auto& [row, col] : main.matches) {
        const auto& track = pool[static_cast<std::size_t>(row)];
        const auto& detection = all_dets[static_cast<std::size_t>(col)];
        if (track->state() == TrackState::Tracked) {
            track->update(*detection, frame_id_);
            activated.push_back(track);
        } else {
            track->reActivate(*detection, frame_id_);
            refound.push_back(track);
        }
    }
    for (int row : main.unmatched_tracks) {
        const auto& track = pool[static_cast<std::size_t>(row)];
        if (track->state() != TrackState::Lost) {
            track->markLost();
            newly_lost.push_back(track);
        }
    }

    // Unconfirmed tracks vs leftover high-confidence detections.
    std::vector<TTTrackPtr> leftover;
    for (int col : main.unmatched_dets) {
        if (static_cast<std::size_t>(col) < n_high) leftover.push_back(all_dets[static_cast<std::size_t>(col)]);
    }
    if (!unconfirmed.empty() && !leftover.empty()) {
        const auto second = iterativeAssociate(
            costMatrix(unconfirmed, leftover, frame_id_, config_.conf_weight, config_.angle_weight),
            config_.match_thresh, config_.reduce_step);
        for (const auto& [row, col] : second.matches) {
            const auto& track = unconfirmed[static_cast<std::size_t>(row)];
            track->update(*leftover[static_cast<std::size_t>(col)], frame_id_);
            activated.push_back(track);
        }
        for (int row : second.unmatched_tracks) {
            const auto& track = unconfirmed[static_cast<std::size_t>(row)];
            track->markRemoved();
            newly_removed.push_back(track);
        }
        std::vector<TTTrackPtr> remaining;
        for (int col : second.unmatched_dets) remaining.push_back(leftover[static_cast<std::size_t>(col)]);
        leftover = std::move(remaining);
    } else {
        for (const auto& track : unconfirmed) {
            track->markRemoved();
            newly_removed.push_back(track);
        }
    }

    // Optional relaxed rebind: tracks that were already Lost and are still unmatched.
    if (config_.lost_match_thr > 0.0 && !leftover.empty()) {
        const std::unordered_set<const TTTrack*> just_lost = [&] {
            std::unordered_set<const TTTrack*> set;
            for (const auto& track : newly_lost) set.insert(track.get());
            return set;
        }();
        std::vector<TTTrackPtr> unmatched_lost;
        for (const auto& track : pool) {
            if (track->state() == TrackState::Lost && just_lost.count(track.get()) == 0) {
                unmatched_lost.push_back(track);
            }
        }
        if (!unmatched_lost.empty()) {
            const auto relaxed = iterativeAssociate(
                costMatrix(unmatched_lost, leftover, frame_id_, config_.conf_weight, config_.angle_weight),
                config_.lost_match_thr, config_.reduce_step);
            for (const auto& [row, col] : relaxed.matches) {
                const auto& track = unmatched_lost[static_cast<std::size_t>(row)];
                track->reActivate(*leftover[static_cast<std::size_t>(col)], frame_id_);
                refound.push_back(track);
            }
            std::vector<TTTrackPtr> remaining;
            for (int col : relaxed.unmatched_dets) remaining.push_back(leftover[static_cast<std::size_t>(col)]);
            leftover = std::move(remaining);
        }
    }

    // Track-aware initialization from the leftover detections.
    std::vector<TTTrackPtr> active;
    for (const auto& track : tracked_tracks_) {
        if (track->state() == TrackState::Tracked) active.push_back(track);
    }
    active.insert(active.end(), activated.begin(), activated.end());
    const std::vector<bool> allow = trackAwareNms(active, leftover, config_.tai_thr, config_.new_track_thresh);
    for (std::size_t index = 0; index < leftover.size(); ++index) {
        if (!allow[index]) continue;
        leftover[index]->activate(kalman_filter_, frame_id_, next_track_id_++);
        activated.push_back(leftover[index]);
    }

    for (const auto& track : lost_tracks_) {
        if (frame_id_ - track->endFrame() > config_.track_buffer) {
            track->markRemoved();
            newly_removed.push_back(track);
        }
    }
    mergePools(activated, refound, newly_lost, newly_removed);

    std::vector<TrackOutput> outputs;
    outputs.reserve(tracked_tracks_.size());
    for (const auto& track : tracked_tracks_) {
        if (track->isActivated() && track->frameId() == frame_id_) {
            outputs.emplace_back(track->trackId(), track->bbox(), track->score(), track->classId(),
                                 track->detectionIndex());
        }
    }
    return outputs;
}

void TrackTrack::mergePools(const std::vector<TTTrackPtr>& activated,
                            const std::vector<TTTrackPtr>& refound,
                            const std::vector<TTTrackPtr>& newly_lost,
                            const std::vector<TTTrackPtr>& newly_removed) {
    std::vector<TTTrackPtr> tracked;
    for (const auto& track : tracked_tracks_) {
        if (track->state() == TrackState::Tracked) tracked.push_back(track);
    }
    tracked = jointTracks<TTTrack>(tracked, activated);
    tracked = jointTracks<TTTrack>(tracked, refound);
    std::vector<TTTrackPtr> lost = subtractTracks<TTTrack>(lost_tracks_, tracked);
    lost.insert(lost.end(), newly_lost.begin(), newly_lost.end());
    // Prune with the previous frames' removals, as Ultralytics' merge_track_pools.
    lost = subtractTracks<TTTrack>(lost, removed_tracks_);

    auto duplicate = removeDuplicateTracks<TTTrack>(tracked, lost);
    tracked_tracks_ = std::move(duplicate.tracked);
    lost_tracks_ = std::move(duplicate.lost);

    removed_tracks_.insert(removed_tracks_.end(), newly_removed.begin(), newly_removed.end());
    constexpr std::size_t removed_buffer = 1000;
    if (removed_tracks_.size() > removed_buffer) {
        removed_tracks_.erase(removed_tracks_.begin(), removed_tracks_.end() - removed_buffer);
    }
}

}
