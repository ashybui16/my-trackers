#include "mot_pipeline/pipeline.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "pass the repository config.yaml path");
        const auto shipped = mot_inference::PipelineConfig::fromFile(argv[1]);
        expect(shipped.detector.input_width == 640 && shipped.tracker.fuse_score,
               "shipped YAML values");
        expect(std::filesystem::path(shipped.engine_path).is_absolute(), "resolved shipped engine path");
        struct TemporaryDirectory {
            std::filesystem::path path = std::filesystem::temp_directory_path() /
                ("mot-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            TemporaryDirectory() { std::filesystem::create_directory(path); }
            ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
        } temporary;
        const auto directory = temporary.path;
        const auto path = directory / "config.yaml";
        const auto load = [&](const std::string& text) {
            std::ofstream file(path);
            file << text;
            file.close();
            return mot_inference::PipelineConfig::fromFile(path.string());
        };
        const auto reject = [&](const std::string& text) {
            try { load(text); }
            catch (const std::exception& error) {
                expect(std::string(error.what()).find(path.string()) != std::string::npos,
                       "error includes config path");
                return;
            }
            throw std::runtime_error("accepted invalid config: " + text);
        };
        const auto custom = load(R"(detector:
  engine_path: "custom.engine"
  input_width: 1280
  input_height: 736
  class_count: 2
  conf_threshold: 0.05
  iou_threshold: 0.6
  max_det: 42
tracker:
  track_high_thresh: 0.5
  track_low_thresh: 0.2
  new_track_thresh: 0.6
  track_buffer: 75
  match_thresh: 0.9
  fuse_score: false
  gmc_method: "none"
  gmc_downscale: 4
)");
        expect(custom.engine_path == (directory / "custom.engine").string(), "relative engine");
        expect(custom.detector.input_width == 1280 && custom.detector.input_height == 736 &&
               custom.detector.class_count == 2 && custom.detector.conf_threshold == 0.05 &&
               custom.detector.iou_threshold == 0.6 && custom.detector.max_det == 42,
               "all detector overrides");
        expect(custom.tracker.track_high_thresh == 0.5 && custom.tracker.track_low_thresh == 0.2 &&
               custom.tracker.new_track_thresh == 0.6 && custom.tracker.track_buffer == 75 &&
               custom.tracker.match_thresh == 0.9 && !custom.tracker.fuse_score &&
               custom.tracker.gmc_method == "none" && custom.tracker.gmc_downscale == 4,
               "all tracker overrides");
        expect(load("---\ndetector: {}\ntracker: {}\n").detector.max_det == 300, "optional defaults");
        expect(load("%YAML:1.0\n---\ndetector: {}\ntracker: { fuse_score: 1 }\n").tracker.fuse_score,
               "OpenCV YAML header and numeric boolean");
        const auto absolute_engine = (directory / "absolute.engine").string();
        expect(load("detector: { engine_path: \"" + absolute_engine + "\" }\ntracker: {}\n").engine_path ==
               absolute_engine, "absolute engine path");
        reject("");
        reject("detector: {}\n");
        reject("detector: [1, 2]\ntracker: {}\n");
        reject("detector: { max_det: 2.5 }\ntracker: {}\n");
        reject("detector: { conf_threshold: \"invalid\" }\ntracker: {}\n");
        reject("detector: { conf_threshold: .Nan }\ntracker: {}\n");
        reject("detector: { input_width: 0 }\ntracker: {}\n");
        reject("detector: { conf_threshold: 1.5 }\ntracker: {}\n");
        reject("detector: { engine_path: \"\" }\ntracker: {}\n");
        reject("detector: { max_dett: 20 }\ntracker: {}\n");
        reject("detector: { max_det: 20, max_det: 30 }\ntracker: {}\n");
        reject("detector: {}\ntracker: { fuse_score: maybe }\n");
        reject("detector: {}\ntracker: { track_low_thresh: 0.9 }\n");
        reject("detector: {}\ntracker: { track_buffer: -1 }\n");
        reject("detector: {}\ntracker: { gmc_method: typo }\n");
        reject("detector: {}\ntracker: {}\nextra: 1\n");
        reject("detector: {}\ntracker: {}\n---\ndetector: {}\ntracker: {}\n");
        reject("detector: [\n");
        bool missing_rejected = false;
        try { mot_inference::PipelineConfig::fromFile((directory / "missing.yaml").string()); }
        catch (const std::runtime_error&) { missing_rejected = true; }
        expect(missing_rejected, "missing file was accepted");
        std::cout << "MOT configuration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
