#pragma once

#include <array>

namespace mot {

using BBox = std::array<double, 4>;  // (x1, y1, x2, y2) in pixels.

void validateBBox(const BBox& bbox, const char* name = "bbox_xyxy");
void validateScore(double score, const char* name = "score");

struct Detection {
    BBox bbox_xyxy{};
    double score{0.0};
    int class_id{0};

    Detection() = default;
    Detection(const BBox& bbox, double confidence, int class_value);
};

// Tracking result; the finite bbox may have inverted corners after GMC, as in Ultralytics.
struct TrackOutput {
    int track_id{0};
    BBox bbox_xyxy{};
    double score{0.0};
    int class_id{0};

    TrackOutput() = default;
    TrackOutput(int id, const BBox& bbox, double confidence, int class_value);
};

}
