#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot {

class Annotator {
public:
    explicit Annotator(std::vector<std::string> class_names = {});

    void draw(cv::Mat& bgr_frame, const std::vector<TrackOutput>& tracks) const;
    void drawDetections(cv::Mat& bgr_frame, const std::vector<Detection>& detections) const;

    const std::vector<std::string>& classNames() const { return class_names_; }

private:
    std::vector<std::string> class_names_;
};

}
