#pragma once

#include "mot/detection/config.hpp"
#include "mot/tracking/config.hpp"

#include <string>
#include <vector>

namespace mot {

struct PipelineConfig {
    DetectorConfig detector{};
    // Model path (.engine → TensorRT, .onnx → ONNX Runtime);
    // fromFile() resolves relative paths against the YAML file's directory.
    std::string model_path;
    std::vector<std::string> class_names;
    TrackerConfig tracker{};

    static PipelineConfig fromFile(const std::string& path);
};

std::vector<std::string> loadClassNames(const std::string& yaml_path, int class_count = 0);

}
