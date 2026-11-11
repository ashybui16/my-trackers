#include "mot/tracking/tracktrack_config.hpp"

#include "tracking/common/class_ema.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

void TrackTrackConfig::validate() const {
    const auto probability = [](double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
        }
    };
    const auto non_negative = [](double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
        }
    };
    probability(track_high_thresh, "track_high_thresh");
    probability(track_low_thresh, "track_low_thresh");
    probability(new_track_thresh, "new_track_thresh");
    probability(match_thresh, "match_thresh");
    probability(lost_match_thr, "lost_match_thr");
    probability(tai_thr, "tai_thr");
    if (!(track_low_thresh < track_high_thresh)) {
        throw std::invalid_argument("track_low_thresh must be smaller than track_high_thresh");
    }
    non_negative(conf_weight, "conf_weight");
    non_negative(angle_weight, "angle_weight");
    non_negative(penalty_p, "penalty_p");
    non_negative(penalty_q, "penalty_q");
    non_negative(reduce_step, "reduce_step");
    if (track_buffer < 0) {
        throw std::invalid_argument("track_buffer must be non-negative");
    }
    if (min_track_len < 0) {
        throw std::invalid_argument("min_track_len must be non-negative");
    }
    if (gmc_method != "sparseOptFlow" && gmc_method != "none") {
        throw std::invalid_argument("gmc_method must be 'sparseOptFlow' or 'none'");
    }
    validateClassEmaAlpha(class_ema_alpha);
}

}
