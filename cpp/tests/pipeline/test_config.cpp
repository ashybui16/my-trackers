#include "mot/pipeline/config.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "pass the repository config.yaml path");
        const auto shipped = mot::PipelineConfig::fromFile(argv[1]);
        expect(shipped.detector.input_width == 640 &&
               std::get<mot::BoTSORTConfig>(shipped.tracker).fuse_score, "shipped YAML values");
        expect(shipped.class_names.size() == 10 && shipped.class_names[0] == "pedestrian" &&
               shipped.class_names[9] == "motor" && shipped.detector.class_count == 10,
               "shipped class names define class_count");
        expect(std::filesystem::path(shipped.model_path).is_absolute(), "resolved shipped model path");
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
            return mot::PipelineConfig::fromFile(path.string());
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
  model_path: "custom.onnx"
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
  class_ema_alpha: 0.8
)");
        expect(custom.model_path == (directory / "custom.onnx").string(), "relative model");
        expect(custom.detector.input_width == 1280 && custom.detector.input_height == 736 &&
               custom.detector.class_count == 2 && custom.detector.conf_threshold == 0.05 &&
               custom.detector.iou_threshold == 0.6 && custom.detector.max_det == 42,
               "all detector overrides");
        const auto& bot = std::get<mot::BoTSORTConfig>(custom.tracker);
        expect(bot.track_high_thresh == 0.5 && bot.track_low_thresh == 0.2 &&
               bot.new_track_thresh == 0.6 && bot.track_buffer == 75 &&
               bot.match_thresh == 0.9 && !bot.fuse_score &&
               bot.gmc_method == "none" && bot.gmc_downscale == 4 && bot.class_ema_alpha == 0.8,
               "all legacy flat tracker overrides");
        const auto typed = load(R"(detector: { class_count: 3, class_names: ["a", "b"] }
tracker:
  type: botsort
  botsort: { track_buffer: 12, gmc_method: "none" }
)");
        expect(mot::trackerType(typed.tracker) == "botsort" &&
               std::get<mot::BoTSORTConfig>(typed.tracker).track_buffer == 12,
               "typed tracker section");
        expect(typed.class_names == std::vector<std::string>{"a", "b", ""}, "inline class names");
        expect(load("detector: {}\ntracker: { type: botsort }\n").tracker.index() == 0,
               "typed tracker without parameters");
        {
            std::ofstream names(directory / "names.yaml");
            names << "nc: 2\nnames:\n  0: cat\n  1: 'dog'\n";
        }
        const auto from_file =
            load("detector: { class_count: 2, class_names: names.yaml }\ntracker: {}\n");
        expect(from_file.class_names == std::vector<std::string>{"cat", "dog"}, "class names file");
        const auto inferred = load("detector: { class_names: [\"x\", \"y\", \"z\"] }\ntracker: {}\n");
        expect(inferred.detector.class_count == 3 && inferred.class_names.size() == 3,
               "class_count inferred from inline names");
        const auto commented = load("detector:\n  class_names:\n    - a#b   # 0\n    - c\ntracker: {}\n");
        expect(commented.class_names == std::vector<std::string>{"a#b", "c"},
               "YAML comments stripped from class names");
        const auto inferred_file = load("detector: { class_names: names.yaml }\ntracker: {}\n");
        expect(inferred_file.detector.class_count == 2, "class_count inferred from names file");
        expect(load("detector: {}\ntracker: {}\n").detector.class_count == 10,
               "class_count default without names");
        expect(load("detector: {}\ntracker: {}\n").model_path.empty(), "model path is optional");
        expect(load("---\ndetector: {}\ntracker: {}\n").detector.max_det == 300, "optional defaults");
        expect(std::get<mot::BoTSORTConfig>(
                   load("%YAML:1.0\n---\ndetector: {}\ntracker: { fuse_score: 1 }\n").tracker)
                   .fuse_score,
               "OpenCV YAML header and numeric boolean");
        const auto absolute_engine = (directory / "absolute.engine").string();
        expect(load("detector: { model_path: \"" + absolute_engine + "\" }\ntracker: {}\n").model_path ==
               absolute_engine, "absolute model path");
        expect(load("detector: { engine_path: \"legacy.engine\" }\ntracker: {}\n").model_path ==
               (directory / "legacy.engine").string(), "legacy engine_path alias");
        reject("detector: { model_path: a.onnx, engine_path: b.engine }\ntracker: {}\n");
        reject("");
        reject("detector: {}\n");
        reject("detector: [1, 2]\ntracker: {}\n");
        reject("detector: { max_det: 2.5 }\ntracker: {}\n");
        reject("detector: { conf_threshold: \"invalid\" }\ntracker: {}\n");
        reject("detector: { conf_threshold: .Nan }\ntracker: {}\n");
        reject("detector: { input_width: 0 }\ntracker: {}\n");
        reject("detector: { conf_threshold: 1.5 }\ntracker: {}\n");
        reject("detector: {}\ntracker: { type: nope }\n");
        reject("detector: {}\ntracker: { class_ema_alpha: 1.0 }\n");
        reject("detector: {}\ntracker: { class_ema_alpha: -0.1 }\n");
        reject("detector: {}\ntracker: { type: botsort, other: {} }\n");
        reject("detector: {}\ntracker: { type: botsort, botsort: { typo: 1 } }\n");
        reject("detector: {}\ntracker: { type: botsort, track_buffer: 3 }\n");
        reject("detector: { class_names: missing.yaml }\ntracker: {}\n");
        reject("detector: { class_count: 1, class_names: [\"a\", \"b\"] }\ntracker: {}\n");
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
        try { mot::PipelineConfig::fromFile((directory / "missing.yaml").string()); }
        catch (const std::runtime_error&) { missing_rejected = true; }
        expect(missing_rejected, "missing file was accepted");
        std::cout << "MOT configuration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
