#pragma once

#include <string>
#include <string_view>

namespace mot {

struct BoTSORTConfig {
    static constexpr std::string_view kType{"botsort"};

    double track_high_thresh{0.25};
    double track_low_thresh{0.10};
    double new_track_thresh{0.25};
    int track_buffer{30};
    double match_thresh{0.80};
    bool fuse_score{true};
    std::string gmc_method{"sparseOptFlow"};  // "sparseOptFlow", "dpoint" or "none".
    // dpoint keypoint and matcher models (.onnx -> ONNX Runtime, .engine/.trt/.plan -> TensorRT).
    // Apps load them into an IKeypointGmcModel passed to createTracker(); the library ignores the paths.
    std::string gmc_dpoint_model;
    std::string gmc_matcher_model;
    // EMA weight on a track's class history: v = a*v + (1-a)*detection_scores. 0 = latest detection.
    double class_ema_alpha{0.0};

    void validate() const;
};

}
