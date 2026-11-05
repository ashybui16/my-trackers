#pragma once

#include "mot/tracking/config.hpp"
#include "mot/tracking/tracker.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace mot {

// Tạo tracker tương ứng với kiểu cấu hình đang chọn; cấu hình được kiểm tra trước.
std::unique_ptr<ITracker> createTracker(const TrackerConfig& config);
// Tên của mọi tracker có thể tạo, theo thứ tự trong TrackerConfig.
std::vector<std::string_view> availableTrackers();

}  // namespace mot
