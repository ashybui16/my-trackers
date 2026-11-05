#pragma once

#include "mot/tracking/botsort_config.hpp"

#include <string_view>
#include <variant>

namespace mot {

// Cấu hình của mọi tracker được hỗ trợ. Mỗi kiểu có `static constexpr kType`.
// Thêm tracker mới: thêm struct cấu hình của nó vào danh sách này
// (xem src/tracking/README.md).
using TrackerConfig = std::variant<BoTSORTConfig>;

// Tên kiểu của cấu hình đang chọn, ví dụ "botsort".
std::string_view trackerType(const TrackerConfig& config);
// Cấu hình mặc định cho một kiểu tracker; ném std::invalid_argument nếu không biết.
TrackerConfig defaultTrackerConfig(std::string_view type);
// Kiểm tra cấu hình của tracker đang chọn.
void validate(const TrackerConfig& config);

}  // namespace mot
