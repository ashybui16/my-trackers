#pragma once

// Shared by mot_video and mot_benchmark: picks the keypoint GMC backend from the model extensions.
// Define MOT_HAS_ONNXRUNTIME / MOT_HAS_TENSORRT for the backends linked into the binary.

#include "mot/tracking/config.hpp"
#include "mot/tracking/keypoint_gmc.hpp"
#if defined(MOT_HAS_TENSORRT)
#include "mot/tensorrt/keypoint_gmc.hpp"
#endif
#if defined(MOT_HAS_ONNXRUNTIME)
#include "mot/onnxruntime/keypoint_gmc.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace mot::apps {

inline std::string lowerExtension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

// Keypoint GMC model for gmc_method "dpoint" (nullptr for other methods).
inline std::unique_ptr<mot::IKeypointGmcModel> makeGmcModel(const mot::TrackerConfig& tracker, std::string& backend) {
    return std::visit([&](const auto& config) -> std::unique_ptr<mot::IKeypointGmcModel> {
        if (config.gmc_method != "dpoint") return nullptr;
        const std::string& dpoint = config.gmc_dpoint_model;
        const std::string& matcher = config.gmc_matcher_model;
        if (dpoint.empty() || matcher.empty()) {
            throw std::runtime_error("gmc_method 'dpoint' needs gmc_dpoint_model and gmc_matcher_model "
                                     "in the tracker YAML section");
        }
        for (const auto& path : {dpoint, matcher}) {
            if (!std::filesystem::is_regular_file(path)) {
                throw std::runtime_error("GMC model does not exist: " + path);
            }
        }
        const auto is_engine = [](const std::string& extension) {
            return extension == ".engine" || extension == ".trt" || extension == ".plan";
        };
        const std::string dpoint_extension = lowerExtension(dpoint);
        const std::string matcher_extension = lowerExtension(matcher);
        if (dpoint_extension == ".onnx" && matcher_extension == ".onnx") {
#if defined(MOT_HAS_ONNXRUNTIME)
            backend = "onnxruntime (CPU)";
            return std::make_unique<mot::OnnxRuntimeKeypointGmc>(dpoint, matcher);
#else
            throw std::runtime_error("ONNX GMC models given but this binary was built without ONNX Runtime");
#endif
        }
        if (is_engine(dpoint_extension) && is_engine(matcher_extension)) {
#if defined(MOT_HAS_TENSORRT)
            backend = "tensorrt";
            return std::make_unique<mot::TensorRTKeypointGmc>(dpoint, matcher);
#else
            throw std::runtime_error("TensorRT GMC engines given but this binary was built without TensorRT");
#endif
        }
        throw std::runtime_error("GMC models must both be .onnx or both be .engine/.trt/.plan: '" + dpoint +
                                 "', '" + matcher + "'");
    }, tracker);
}

}
