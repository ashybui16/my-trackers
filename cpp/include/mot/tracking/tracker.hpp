#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <string_view>
#include <vector>

namespace mot {

// Giao diện chung của mọi tracker: nhận detection của một frame, trả các track đã xác nhận.
class ITracker {
public:
    virtual ~ITracker() = default;

    // frame là ảnh BGR gốc; tracker có thể dùng nó (GMC, ReID) hoặc bỏ qua.
    virtual std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                            const cv::Mat& frame = cv::Mat()) = 0;
    // Xóa toàn bộ trạng thái để bắt đầu một chuỗi video mới.
    virtual void reset() = 0;
    // Tên kiểu tracker, trùng với Config::kType.
    virtual std::string_view name() const = 0;
};

}  // namespace mot
