#pragma once

#include "mot/detection/detector.hpp"
#include "mot/tracking/tracker.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace mot {

class Pipeline {
public:
    Pipeline(std::unique_ptr<IDetector> detector, std::unique_ptr<ITracker> tracker);

    std::vector<TrackOutput> process(const cv::Mat& frame);
    void reset();
    void setTracker(std::unique_ptr<ITracker> tracker);

    IDetector& detector() { return *detector_; }
    const IDetector& detector() const { return *detector_; }
    ITracker& tracker() { return *tracker_; }
    const ITracker& tracker() const { return *tracker_; }

private:
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<ITracker> tracker_;
};

}
