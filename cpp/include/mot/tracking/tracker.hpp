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
    // Like update(), plus `recovered`: detections the detector's NMS dropped that a looser NMS keeps
    // (see IDetector::detectWithRecovered). Trackers that do not use them ignore the extra input.
    virtual std::vector<TrackOutput> updateWithRecovered(const std::vector<Detection>& detections,
                                                         const std::vector<Detection>& /*recovered*/,
                                                         const cv::Mat& frame = cv::Mat()) {
        return update(detections, frame);
    }
    // True when updateWithRecovered() uses the recovered detections, so callers should compute them.
    virtual bool usesRecoveredDetections() const { return false; }
    virtual void reset() = 0;
    virtual std::string_view name() const = 0;
};

}
