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
    std::string gmc_method{"sparseOptFlow"};
    int gmc_downscale{2};

    void validate() const;
};

}
