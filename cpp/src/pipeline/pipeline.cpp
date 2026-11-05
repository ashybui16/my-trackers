#include "mot/pipeline/pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace mot {

Pipeline::Pipeline(std::unique_ptr<IDetector> detector, std::unique_ptr<ITracker> tracker)
    : detector_(std::move(detector)), tracker_(std::move(tracker)) {
    if (!detector_) throw std::invalid_argument("Pipeline requires a detector");
    if (!tracker_) throw std::invalid_argument("Pipeline requires a tracker");
}

std::vector<TrackOutput> Pipeline::process(const cv::Mat& frame) {
    const auto detections = detector_->detect(frame);
    return tracker_->update(detections, frame);
}

void Pipeline::reset() {
    tracker_->reset();
}

void Pipeline::setTracker(std::unique_ptr<ITracker> tracker) {
    if (!tracker) throw std::invalid_argument("Pipeline requires a tracker");
    tracker_ = std::move(tracker);
}

}
