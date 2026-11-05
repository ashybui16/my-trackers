#include "mot_pipeline/pipeline.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace mot_inference {
namespace {

void checkKeys(const cv::FileNode& node, const std::set<std::string>& allowed,
               const std::string& section) {
    if (!node.isMap()) {
        throw std::invalid_argument(section + " must be a mapping");
    }
    std::set<std::string> seen;
    for (const auto& entry : node) {
        const std::string name = entry.name();
        if (allowed.count(name) == 0 || !seen.insert(name).second) {
            throw std::invalid_argument("unknown or duplicate key: " + section + "." + name);
        }
    }
}

void readValue(const cv::FileNode& section, const char* key, int& value) {
    const auto node = section[key];
    if (node.isNone()) return;
    if (!node.isInt()) throw std::invalid_argument(std::string(key) + " must be an integer");
    value = static_cast<int>(node);
}

void readValue(const cv::FileNode& section, const char* key, double& value) {
    const auto node = section[key];
    if (node.isNone()) return;
    if (!node.isInt() && !node.isReal()) {
        throw std::invalid_argument(std::string(key) + " must be a number");
    }
    value = static_cast<double>(node);
}

void readValue(const cv::FileNode& section, const char* key, std::string& value) {
    const auto node = section[key];
    if (node.isNone()) return;
    if (!node.isString()) throw std::invalid_argument(std::string(key) + " must be a string");
    value = static_cast<std::string>(node);
}

void readValue(const cv::FileNode& section, const char* key, bool& value) {
    const auto node = section[key];
    if (node.isNone()) return;
    if (node.isInt()) {
        const int number = static_cast<int>(node);
        if (number == 0 || number == 1) {
            value = number == 1;
            return;
        }
    } else if (node.isString()) {
        const std::string text = static_cast<std::string>(node);
        if (text == "true" || text == "false") {
            value = text == "true";
            return;
        }
    }
    throw std::invalid_argument(std::string(key) + " must be true, false, 0 or 1");
}

}  // namespace

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

PipelineConfig PipelineConfig::fromFile(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("cannot open file");
        std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) throw std::runtime_error("cannot read file");
        const auto first = yaml.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) throw std::invalid_argument("file is empty");
        // OpenCV requires its YAML version header; accept ordinary YAML on disk.
        if (yaml.compare(first, 5, "%YAML") != 0) {
            yaml.insert(0, yaml.compare(first, 3, "---") == 0
                ? "%YAML:1.0\n" : "%YAML:1.0\n---\n");
        }
        cv::FileStorage storage(yaml, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                     cv::FileStorage::FORMAT_YAML);
        if (!storage.isOpened()) throw std::runtime_error("cannot parse YAML");
        checkKeys(storage.root(), {"detector", "tracker"}, "config");
        if (storage.root(1).size() != 0) {
            throw std::invalid_argument("only one YAML document is supported");
        }
        const auto detector_node = storage["detector"];
        const auto tracker_node = storage["tracker"];
        checkKeys(detector_node, {"engine_path", "input_width", "input_height", "class_count",
                                 "conf_threshold", "iou_threshold", "max_det"}, "detector");
        checkKeys(tracker_node, {"track_high_thresh", "track_low_thresh", "new_track_thresh",
                                "track_buffer", "match_thresh", "fuse_score", "gmc_method",
                                "gmc_downscale"}, "tracker");
        PipelineConfig config;
        readValue(detector_node, "engine_path", config.engine_path);
        readValue(detector_node, "input_width", config.detector.input_width);
        readValue(detector_node, "input_height", config.detector.input_height);
        readValue(detector_node, "class_count", config.detector.class_count);
        readValue(detector_node, "conf_threshold", config.detector.conf_threshold);
        readValue(detector_node, "iou_threshold", config.detector.iou_threshold);
        readValue(detector_node, "max_det", config.detector.max_det);
        readValue(tracker_node, "track_high_thresh", config.tracker.track_high_thresh);
        readValue(tracker_node, "track_low_thresh", config.tracker.track_low_thresh);
        readValue(tracker_node, "new_track_thresh", config.tracker.new_track_thresh);
        readValue(tracker_node, "track_buffer", config.tracker.track_buffer);
        readValue(tracker_node, "match_thresh", config.tracker.match_thresh);
        readValue(tracker_node, "fuse_score", config.tracker.fuse_score);
        readValue(tracker_node, "gmc_method", config.tracker.gmc_method);
        readValue(tracker_node, "gmc_downscale", config.tracker.gmc_downscale);
        config.detector.validate();
        config.tracker.validate();
        if (config.engine_path.empty()) throw std::invalid_argument("engine_path must not be empty");
        const auto directory = std::filesystem::absolute(path).parent_path();
        config.engine_path = (directory / config.engine_path).lexically_normal().string();
        return config;
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot load MOT config '" + path + "': " + error.what());
    }
}

}  // namespace mot_inference
