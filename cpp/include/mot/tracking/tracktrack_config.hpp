#pragma once

#include <string>
#include <string_view>

namespace mot {

// TrackTrack ("Focusing on Tracks for Online Multi-Object Tracking", CVPR 2025), Ultralytics-compatible.
// Defaults follow ultralytics/cfg/trackers/tracktrack.yaml. ReID is not supported (motion-only cost).
struct TrackTrackConfig {
    static constexpr std::string_view kType{"tracktrack"};

    double track_high_thresh{0.6};
    double track_low_thresh{0.25};
    double new_track_thresh{0.7};   // Minimum score to start a track via track-aware initialization (TAI).
    int track_buffer{30};
    double match_thresh{0.7};       // Initial cost gate of the iterative assignment.
    double lost_match_thr{0.0};     // Relaxed rebind gate for still-Lost tracks; 0 disables the pass.

    double conf_weight{0.1};        // Weight of the confidence distance.
    double angle_weight{0.05};      // Weight of the corner angle distance.

    double penalty_p{0.2};          // Cost penalty for low-confidence detections.
    double penalty_q{0.4};          // Cost penalty for recovered (loose-NMS) detections.
    double reduce_step{0.05};       // Gate reduction per assignment round.

    double tai_thr{0.55};           // IoU threshold of the track-aware NMS.
    int min_track_len{3};           // Hits before a New track is confirmed; 0 or 1 confirms at once.

    std::string gmc_method{"sparseOptFlow"};
    int gmc_downscale{2};

    void validate() const;
};

}
