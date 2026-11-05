#pragma once

#include "mot_detector/detector.hpp"
#include "mot_tracker/tracker.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace mot_inference {

struct PipelineConfig {
    // Tùy chọn cho hai bước phát hiện và theo dõi.
    DetectorConfig detector{};
    mot_tracker::BoTSORTConfig tracker{};
    std::string engine_path{"26leaf1.engine"};

    // Resolve relative engine paths against the YAML file directory.
    static PipelineConfig fromFile(const std::string& path);
};

class MotPipeline {
public:
    // Nạp detector và khởi tạo tracker với cùng cấu hình pipeline.
    MotPipeline(const std::string& engine_path, PipelineConfig config = {});

    // Phát hiện đối tượng rồi cập nhật ID theo dõi cho frame hiện tại.
    std::vector<mot_tracker::TrackOutput> processFrame(const cv::Mat& frame);
    // Bắt đầu chuỗi video mới mà không cần nạp lại engine.
    void reset();

    TensorRTDetector& detector() { return detector_; }
    const TensorRTDetector& detector() const { return detector_; }
    mot_tracker::BoTSORT& tracker() { return tracker_; }
    const mot_tracker::BoTSORT& tracker() const { return tracker_; }

private:
    // Detector sinh hộp; tracker nối các hộp thành quỹ đạo qua nhiều frame.
    TensorRTDetector detector_;
    mot_tracker::BoTSORT tracker_;
};

}  // namespace mot_inference
