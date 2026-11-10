#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot {

class IDetector {
public:
    virtual ~IDetector() = default;

    virtual std::vector<Detection> detect(const cv::Mat& bgr_frame) = 0;
    // Like detect(), and also fills `recovered` with boxes the NMS dropped that a looser NMS keeps
    // (for trackers whose usesRecoveredDetections() is true). The default reports none.
    virtual std::vector<Detection> detectWithRecovered(const cv::Mat& bgr_frame, std::vector<Detection>& recovered) {
        recovered.clear();
        return detect(bgr_frame);
    }
    virtual const std::vector<std::string>& classNames() const = 0;
};

}
