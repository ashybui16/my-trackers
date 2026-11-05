#pragma once

#include "mot_tracker/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot_inference {

// Draws tracked boxes and class/ID labels on a BGR frame.
class MotAnnotator {
public:
    MotAnnotator(const std::string& engine_path, int class_count);

    void draw(cv::Mat& bgr_frame, const std::vector<mot_tracker::TrackOutput>& tracks) const;
    void drawDetections(cv::Mat& bgr_frame,
                        const std::vector<mot_tracker::Detection>& detections) const;

private:
    std::vector<std::string> class_names_;
};

}  // namespace mot_inference
