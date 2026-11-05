#include "mot/tracking/botsort_config.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

void BoTSORTConfig::validate() const {
    const auto probability = [](double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
        }
    };
    probability(track_high_thresh, "track_high_thresh");
    probability(track_low_thresh, "track_low_thresh");
    probability(new_track_thresh, "new_track_thresh");
    probability(match_thresh, "match_thresh");
    if (!(track_low_thresh < track_high_thresh)) {
        throw std::invalid_argument("track_low_thresh must be smaller than track_high_thresh");
    }
    if (new_track_thresh < track_high_thresh) {
        throw std::invalid_argument("new_track_thresh must be at least track_high_thresh");
    }
    if (track_buffer < 0) {
        throw std::invalid_argument("track_buffer must be non-negative");
    }
    if (gmc_method != "sparseOptFlow" && gmc_method != "none") {
        throw std::invalid_argument("gmc_method must be 'sparseOptFlow' or 'none'");
    }
    if (gmc_downscale < 1) {
        throw std::invalid_argument("gmc_downscale must be at least 1");
    }
}

}
