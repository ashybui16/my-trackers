#pragma once

#include "mot/tracking/config.hpp"
#include "mot/tracking/tracker.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace mot {

std::unique_ptr<ITracker> createTracker(const TrackerConfig& config);
std::vector<std::string_view> availableTrackers();

}
