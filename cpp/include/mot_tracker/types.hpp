#pragma once

#include <array>

namespace mot_tracker {

using BBox = std::array<double, 4>;  // (x1, y1, x2, y2), đơn vị pixel.

// Báo lỗi nếu tọa độ hộp hoặc điểm tin cậy không hợp lệ.
void validateBBox(const BBox& bbox, const char* name = "bbox_xyxy");
void validateScore(double score, const char* name = "score");

// Một đối tượng do detector tìm thấy trong frame hiện tại.
struct Detection {
    BBox bbox_xyxy{};
    double score{0.0};
    int class_id{0};

    Detection() = default;
    Detection(const BBox& bbox, double confidence, int class_value);
};

// Kết quả theo dõi; bbox hữu hạn có thể đảo góc sau GMC như Ultralytics.
struct TrackOutput {
    int track_id{0};
    BBox bbox_xyxy{};
    double score{0.0};
    int class_id{0};

    TrackOutput() = default;
    TrackOutput(int id, const BBox& bbox, double confidence, int class_value);
};

}  // namespace mot_tracker
