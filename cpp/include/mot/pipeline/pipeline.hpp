#pragma once

#include "mot/detection/detector.hpp"
#include "mot/tracking/tracker.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace mot {

// Ghép một detector và một tracker bất kỳ thành pipeline detect → track.
class Pipeline {
public:
    Pipeline(std::unique_ptr<IDetector> detector, std::unique_ptr<ITracker> tracker);

    // Phát hiện đối tượng rồi cập nhật ID theo dõi cho frame hiện tại.
    std::vector<TrackOutput> process(const cv::Mat& frame);
    // Bắt đầu chuỗi video mới mà không cần nạp lại detector.
    void reset();
    // Thay tracker (ví dụ đổi thuật toán) mà vẫn giữ detector đã nạp.
    void setTracker(std::unique_ptr<ITracker> tracker);

    IDetector& detector() { return *detector_; }
    const IDetector& detector() const { return *detector_; }
    ITracker& tracker() { return *tracker_; }
    const ITracker& tracker() const { return *tracker_; }

private:
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<ITracker> tracker_;
};

}  // namespace mot
