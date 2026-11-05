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
    virtual const std::vector<std::string>& classNames() const = 0;
};

}
