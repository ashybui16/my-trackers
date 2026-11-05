#include "mot_pipeline/pipeline.hpp"

#include <utility>

namespace mot_inference {

MotPipeline::MotPipeline(const std::string& engine_path, PipelineConfig config)
    : detector_(engine_path, config.detector), tracker_(std::move(config.tracker)) {}

std::vector<mot_tracker::TrackOutput> MotPipeline::processFrame(const cv::Mat& frame) {
    // Một frame đi qua detector trước, rồi các hộp được gắn ID qua tracker.
    const auto detections = detector_.infer(frame);
    return tracker_.update(detections, frame);
}

void MotPipeline::reset() {
    // Giữ engine đã nạp và chỉ xóa lịch sử theo dõi.
    tracker_.reset();
}

}  // namespace mot_inference