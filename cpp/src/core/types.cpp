#include "mot/core/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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

TrackOutput::TrackOutput(int id, const BBox& bbox, double confidence, int class_value)
    : track_id(id), bbox_xyxy(bbox), score(confidence), class_id(class_value) {
    if (track_id <= 0) {
        throw std::invalid_argument("track_id must be positive");
    }
    validateFiniteBBox(bbox_xyxy, "bbox_xyxy");
    validateScore(score);
}

}
