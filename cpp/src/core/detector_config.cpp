#include "mot/detection/config.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

void DetectorConfig::validate() const {
    if (input_width <= 0 || input_height <= 0) {
        throw std::invalid_argument("detector input dimensions must be positive");
    }
    if (!std::isfinite(conf_threshold) || conf_threshold < 0.0 || conf_threshold > 1.0) {
        throw std::invalid_argument("conf_threshold must be finite and in [0, 1]");
    }
    if (!std::isfinite(iou_threshold) || iou_threshold < 0.0 || iou_threshold > 1.0) {
        throw std::invalid_argument("iou_threshold must be finite and in [0, 1]");
    }
    if (max_det <= 0 || class_count <= 0) {
        throw std::invalid_argument("max_det and class_count must be positive");
    }
}

}
