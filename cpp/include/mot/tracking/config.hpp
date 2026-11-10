#pragma once

#include "mot/tracking/botsort_config.hpp"
#include "mot/tracking/tracktrack_config.hpp"

#include <string_view>
#include <variant>

namespace mot {

using TrackerConfig = std::variant<BoTSORTConfig, TrackTrackConfig>;

std::string_view trackerType(const TrackerConfig& config);
TrackerConfig defaultTrackerConfig(std::string_view type);
void validate(const TrackerConfig& config);

}
