#include "mot/core/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace mot {

namespace {

void validateFiniteBBox(const BBox& bbox, const char* name) {
    for (double value : bbox) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

}

void validateBBox(const BBox& bbox, const char* name) {
    validateFiniteBBox(bbox, name);
    if (bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
        throw std::invalid_argument(std::string(name) + " must have positive width and height");
    }
}

void validateScore(double score, const char* name) {
    if (!std::isfinite(score) || score < 0.0 || score > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
    }
}

Detection::Detection(const BBox& bbox, double confidence, int class_value)
    : bbox_xyxy(bbox), score(confidence), class_id(class_value) {
    validateBBox(bbox_xyxy);
    validateScore(score);
}

Detection::Detection(const BBox& bbox, double confidence, int class_value, std::vector<float> scores)
    : Detection(bbox, confidence, class_value) {
    for (float value : scores) {
        validateScore(value, "class_scores");
    }
    class_scores = std::move(scores);
}

TrackOutput::TrackOutput(int id, const BBox& bbox, double confidence, int class_value, int detection_value)
    : track_id(id), bbox_xyxy(bbox), score(confidence), class_id(class_value), detection_index(detection_value) {
    if (track_id <= 0) {
        throw std::invalid_argument("track_id must be positive");
    }
    if (detection_index < -1) {
        throw std::invalid_argument("detection_index must be -1 or non-negative");
    }
    validateFiniteBBox(bbox_xyxy, "bbox_xyxy");
    validateScore(score);
}

}
