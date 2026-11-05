#pragma once

#include "mot/detection/config.hpp"
#include "mot/tracking/config.hpp"

#include <string>
#include <vector>

namespace mot {

// Cấu hình đầy đủ cho detect → track, thường đọc từ YAML.
struct PipelineConfig {
    DetectorConfig detector{};
    // Đường dẫn engine; fromFile() chuyển đường dẫn tương đối theo thư mục YAML.
    std::string engine_path;
    // Tên lớp theo class_id (rỗng nếu không cấu hình).
    std::vector<std::string> class_names;
    TrackerConfig tracker{};

    // Đọc YAML dạng:
    //   detector: { engine_path, class_names, input_width, ... }
    //   tracker:  { type: botsort, botsort: { ... } }
    // `tracker` không có `type` và chứa khóa BoT-SORT phẳng vẫn được chấp nhận.
    static PipelineConfig fromFile(const std::string& path);
};

// Đọc `names:` từ YAML kiểu dataset Ultralytics (`0: pedestrian`, ...).
// class_count > 0 giới hạn số lớp; ngược lại kích thước = id lớn nhất + 1.
std::vector<std::string> loadClassNames(const std::string& yaml_path, int class_count = 0);

}  // namespace mot
