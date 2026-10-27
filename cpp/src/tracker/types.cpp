#include "mot_tracker/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mot_tracker {

namespace {

void validateFiniteBBox(const BBox& bbox, const char* name) {
    for (double value : bbox) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

}  // namespace

void validateBBox(const BBox& bbox, const char* name) {
    // Detection đầu vào vẫn phải có tọa độ hữu hạn và kích thước dương.
    validateFiniteBBox(bbox, name);
    if (bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
        throw std::invalid_argument(std::string(name) + " must have positive width and height");
    }
}

void validateScore(double score, const char* name) {
    // Điểm tin cậy phải là xác suất hữu hạn trong [0, 1].
    if (!std::isfinite(score) || score < 0.0 || score > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
    }
}

Detection::Detection(const BBox& bbox, double confidence, int class_value)
    : bbox_xyxy(bbox), score(confidence), class_id(class_value) {
    // Kiểm tra dữ liệu detector ngay khi tạo detection.
    validateBBox(bbox_xyxy);
    validateScore(score);
}

TrackOutput::TrackOutput(int id, const BBox& bbox, double confidence, int class_value)
    : track_id(id), bbox_xyxy(bbox), score(confidence), class_id(class_value) {
    // Cho phép góc đảo từ state GMC như Ultralytics; vẫn kiểm tra số hữu hạn.
    if (track_id <= 0) {
        throw std::invalid_argument("track_id must be positive");
    }
    validateFiniteBBox(bbox_xyxy, "bbox_xyxy");
    validateScore(score);
}

}  // namespace mot_tracker
