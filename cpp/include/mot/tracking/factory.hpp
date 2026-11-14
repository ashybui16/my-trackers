#pragma once

#include "mot/tracking/config.hpp"
#include "mot/tracking/keypoint_gmc.hpp"
#include "mot/tracking/tracker.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace mot {

// gmc_model is required when the config's gmc_method is "dpoint" (see mot/onnxruntime/keypoint_gmc.hpp,
// mot/tensorrt/keypoint_gmc.hpp) and ignored otherwise. The tracker takes ownership.
std::unique_ptr<ITracker> createTracker(const TrackerConfig& config,
                                        std::unique_ptr<IKeypointGmcModel> gmc_model = nullptr);
std::vector<std::string_view> availableTrackers();

}
