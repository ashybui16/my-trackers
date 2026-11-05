#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string_view>
#include <vector>

namespace mot {

class ITracker {
public:
    virtual ~ITracker() = default;

    virtual std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                            const cv::Mat& frame = cv::Mat()) = 0;
    virtual void reset() = 0;
    virtual std::string_view name() const = 0;
};

}
