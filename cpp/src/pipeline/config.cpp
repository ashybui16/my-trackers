#include "mot/pipeline/config.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <stdexcept>
#include <variant>

namespace mot {
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

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::filesystem::path resolve(const std::filesystem::path& directory, const std::string& path) {
    return (directory / path).lexically_normal();
}

// Một overload cho mỗi tracker: đọc mục `tracker.<type>` vào cấu hình tương ứng.
void readTracker(const cv::FileNode& node, const std::string& section, BoTSORTConfig& config) {
    checkKeys(node, {"track_high_thresh", "track_low_thresh", "new_track_thresh",
                     "track_buffer", "match_thresh", "fuse_score", "gmc_method",
                     "gmc_downscale"}, section);
    readValue(node, "track_high_thresh", config.track_high_thresh);
    readValue(node, "track_low_thresh", config.track_low_thresh);
    readValue(node, "new_track_thresh", config.new_track_thresh);
    readValue(node, "track_buffer", config.track_buffer);
    readValue(node, "match_thresh", config.match_thresh);
    readValue(node, "fuse_score", config.fuse_score);
    readValue(node, "gmc_method", config.gmc_method);
    readValue(node, "gmc_downscale", config.gmc_downscale);
}

TrackerConfig readTrackerConfig(const cv::FileNode& node) {
    if (!node.isMap()) throw std::invalid_argument("tracker must be a mapping");
    if (node["type"].isNone()) {
        // Định dạng cũ: khóa BoT-SORT nằm thẳng dưới `tracker`.
        BoTSORTConfig config;
        readTracker(node, "tracker", config);
        return config;
    }
    std::string type;
    readValue(node, "type", type);
    TrackerConfig config = defaultTrackerConfig(type);
    checkKeys(node, {"type", type}, "tracker");
    const auto section = node[type];
    if (!section.isNone()) {
        std::visit([&](auto& value) { readTracker(section, "tracker." + type, value); }, config);
    }
    return config;
}

std::vector<std::string> readClassNames(const cv::FileNode& node,
                                        const std::filesystem::path& directory,
                                        int class_count) {
    if (node.isNone()) return {};
    if (node.isString()) {
        const auto path = resolve(directory, static_cast<std::string>(node));
        if (!std::filesystem::is_regular_file(path)) {
            throw std::invalid_argument("class_names file does not exist: " + path.string());
        }
        return loadClassNames(path.string(), class_count);
    }
    if (!node.isSeq()) {
        throw std::invalid_argument("class_names must be a file path or a list of names");
    }
    std::vector<std::string> names;
    for (const auto& entry : node) {
        if (!entry.isString()) throw std::invalid_argument("class_names entries must be strings");
        names.push_back(static_cast<std::string>(entry));
    }
    if (static_cast<int>(names.size()) > class_count) {
        throw std::invalid_argument("class_names has more entries than class_count");
    }
    names.resize(static_cast<std::size_t>(class_count));
    return names;
}

}  // namespace

std::vector<std::string> loadClassNames(const std::string& yaml_path, int class_count) {
    std::ifstream file(yaml_path);
    if (!file) throw std::runtime_error("cannot read class names: " + yaml_path);

    std::vector<std::string> names(static_cast<std::size_t>(std::max(0, class_count)));
    const std::regex class_entry(R"(^[ \t]+(\d+)[ \t]*:[ \t]*([^#]+))");
    bool in_names = false;
    std::string line;
    while (std::getline(file, line)) {
        if (!in_names) {
            in_names = trim(line) == "names:";
            continue;
        }
        if (!trim(line).empty() && line.front() != ' ' && line.front() != '\t') break;

        std::smatch match;
        if (!std::regex_search(line, match, class_entry)) continue;
        try {
            const auto id = std::stoll(match[1].str());
            std::string name = trim(match[2].str());
            if (name.size() >= 2 &&
                ((name.front() == '"' && name.back() == '"') ||
                 (name.front() == '\'' && name.back() == '\''))) {
                name = name.substr(1, name.size() - 2);
            }
            if (id < 0 || name.empty()) continue;
            if (class_count > 0 && id >= class_count) continue;
            if (static_cast<std::size_t>(id) >= names.size()) {
                names.resize(static_cast<std::size_t>(id) + 1);
            }
            names[static_cast<std::size_t>(id)] = name;
        } catch (const std::out_of_range&) {
            // Bỏ qua chỉ số lớp không hợp lệ.
        }
    }
    return names;
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
        const auto directory = std::filesystem::absolute(path).parent_path();
        const auto detector_node = storage["detector"];
        checkKeys(detector_node, {"engine_path", "class_names", "input_width", "input_height",
                                  "class_count", "conf_threshold", "iou_threshold", "max_det"},
                  "detector");
        PipelineConfig config;
        readValue(detector_node, "engine_path", config.engine_path);
        readValue(detector_node, "input_width", config.detector.input_width);
        readValue(detector_node, "input_height", config.detector.input_height);
        readValue(detector_node, "class_count", config.detector.class_count);
        readValue(detector_node, "conf_threshold", config.detector.conf_threshold);
        readValue(detector_node, "iou_threshold", config.detector.iou_threshold);
        readValue(detector_node, "max_det", config.detector.max_det);
        config.detector.validate();
        config.class_names = readClassNames(detector_node["class_names"], directory,
                                            config.detector.class_count);
        config.tracker = readTrackerConfig(storage["tracker"]);
        validate(config.tracker);
        // engine_path là tùy chọn: ứng dụng dùng detector khác có thể bỏ qua nó.
        if (!config.engine_path.empty()) {
            config.engine_path = resolve(directory, config.engine_path).string();
        }
        return config;
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot load MOT config '" + path + "': " + error.what());
    }
}

}  // namespace mot
