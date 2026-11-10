#include "mot/tracking/factory.hpp"

#include "tracking/botsort/botsort.hpp"
#include "tracking/tracktrack/tracktrack.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace mot {
namespace {

std::unique_ptr<ITracker> makeTracker(const BoTSORTConfig& config) {
    return std::make_unique<BoTSORT>(config);
}

std::unique_ptr<ITracker> makeTracker(const TrackTrackConfig& config) {
    return std::make_unique<TrackTrack>(config);
}

template <std::size_t... I>
std::vector<std::string_view> typeNames(std::index_sequence<I...>) {
    return {std::variant_alternative_t<I, TrackerConfig>::kType...};
}

template <std::size_t I = 0>
TrackerConfig defaultFor(std::string_view type) {
    if constexpr (I == std::variant_size_v<TrackerConfig>) {
        std::string known;
        for (const auto name : availableTrackers()) {
            known += (known.empty() ? "" : ", ") + std::string(name);
        }
        throw std::invalid_argument("unknown tracker type '" + std::string(type) +
                                    "' (available: " + known + ")");
    } else {
        using Config = std::variant_alternative_t<I, TrackerConfig>;
        if (type == Config::kType) return Config{};
        return defaultFor<I + 1>(type);
    }
}

}

std::string_view trackerType(const TrackerConfig& config) {
    return std::visit([](const auto& value) {
        return std::decay_t<decltype(value)>::kType;
    }, config);
}

TrackerConfig defaultTrackerConfig(std::string_view type) {
    return defaultFor(type);
}

void validate(const TrackerConfig& config) {
    std::visit([](const auto& value) { value.validate(); }, config);
}

std::vector<std::string_view> availableTrackers() {
    return typeNames(std::make_index_sequence<std::variant_size_v<TrackerConfig>>{});
}

std::unique_ptr<ITracker> createTracker(const TrackerConfig& config) {
    validate(config);
    return std::visit([](const auto& value) { return makeTracker(value); }, config);
}

}
