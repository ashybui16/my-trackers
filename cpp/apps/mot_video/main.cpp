#include "mot/pipeline/config.hpp"
#include "mot/pipeline/pipeline.hpp"
#if defined(MOT_HAS_TENSORRT)
#include "mot/tensorrt/detector.hpp"
#endif
#if defined(MOT_HAS_ONNXRUNTIME)
#include "mot/onnxruntime/detector.hpp"
#endif
#include "mot/tracking/factory.hpp"
#include "gmc_model_factory.hpp"
#include "mot/visualization/annotator.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

struct Options {
    std::filesystem::path video;
    std::filesystem::path output_dir{"runs/video_inference_cpp"};
    mot::PipelineConfig config;
    bool overwrite{false};
};

std::string backends() {
    std::string names;
#if defined(MOT_HAS_TENSORRT)
    names += " tensorrt(.engine,.trt,.plan)";
#endif
#if defined(MOT_HAS_ONNXRUNTIME)
    names += " onnxruntime(.onnx)";
#endif
    return names;
}

[[noreturn]] void usageError(const std::string& message) {
    throw std::invalid_argument(message +
                                "\nUsage: mot_video VIDEO [--config YAML] [options]\n"
                                "  --config YAML                 default: configs/config.yaml (or MOT_CONFIG_PATH)\n"
                                "  --model MODEL                 override YAML model_path (.engine or .onnx)\n"
                                "  --engine ENGINE               alias of --model\n"
                                "  --tracker TYPE                override YAML tracker.type\n"
                                "  --class-count N\n"
                                "  --output-dir DIR              output directory\n"
                                "  --input-width N --input-height N\n"
                                "  --conf F --iou F --max-det N\n"
                                "Trackers (botsort, tracktrack):\n"
                                "  --track-high-thresh F --track-low-thresh F --new-track-thresh F\n"
                                "  --track-buffer N --match-thresh F\n"
                                "  --gmc-method sparseOptFlow|dpoint|none\n"
                                "  --gmc-dpoint MODEL --gmc-matcher MODEL   dpoint GMC models (.onnx or .engine)\n"
                                "  --class-ema-alpha F\n"
                                "BoT-SORT only (--tracker botsort):\n"
                                "  --fuse-score | --no-fuse-score\n"
                                "Other TrackTrack options are set in YAML (tracker.tracktrack).\n"
                                "  --overwrite");
}

double parseDouble(const std::string& option, const std::string& text) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            usageError(option + " expects a finite number");
        }
        return value;
    } catch (const std::exception&) {
        usageError(option + " expects a number, got '" + text + "'");
    }
}

int parseInt(const std::string& option, const std::string& text) {
    try {
        std::size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != text.size() || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            usageError(option + " expects an integer");
        }
        return static_cast<int>(value);
    } catch (const std::exception&) {
        usageError(option + " expects an integer, got '" + text + "'");
    }
}

std::string nextValue(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) {
        usageError(option + " requires a value");
    }
    return argv[++index];
}

Options parseOptions(int argc, char** argv) {
    Options options;
    // Read YAML first so explicit CLI overrides are independent of argument order.
    const char* environment_config = std::getenv("MOT_CONFIG_PATH");
    std::string config_path = environment_config && *environment_config
        ? environment_config : "configs/config.yaml";
    std::string tracker_type;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::string trackers;
            for (const auto name : mot::availableTrackers()) trackers += " " + std::string(name);
            std::cout << "Usage: mot_video VIDEO [--config YAML] [options]\n"
                      << "Default config: configs/config.yaml (or MOT_CONFIG_PATH)\n"
                      << "Trackers:" << trackers << " (select with --tracker TYPE)\n"
                      << "Detector backends:" << backends() << " (chosen by model extension)\n"
                      << "CLI overrides: --model (alias --engine), --class-count, --input-width, --input-height,\n"
                      << "  --conf, --iou, --max-det, --track-high-thresh, --track-low-thresh,\n"
                      << "  --new-track-thresh, --track-buffer, --match-thresh, --fuse-score,\n"
                      << "  --no-fuse-score, --gmc-method, --class-ema-alpha\n"
                      << "Output: --output-dir DIR, --overwrite\n";
            std::exit(0);
        }
        if (argument == "--overwrite" || argument == "--fuse-score" || argument == "--no-fuse-score")
            continue;
        if (argument.rfind("--", 0) == 0) {
            const auto value = nextValue(index, argc, argv, argument);
            if (argument == "--config") config_path = value;
            if (argument == "--tracker") tracker_type = value;
        }
    }
    options.config = mot::PipelineConfig::fromFile(config_path);
    // Switch tracker first so tracker-specific flags apply to the right type.
    if (!tracker_type.empty() && tracker_type != mot::trackerType(options.config.tracker)) {
        options.config.tracker = mot::defaultTrackerConfig(tracker_type);
    }
    const auto botsort = [&](const std::string& option) -> mot::BoTSORTConfig& {
        auto* config = std::get_if<mot::BoTSORTConfig>(&options.config.tracker);
        if (!config) usageError(option + " only applies to --tracker botsort");
        return *config;
    };
    // Applies a setting that every tracker config has under the same name.
    const auto anyTracker = [&](const auto& apply) { std::visit(apply, options.config.tracker); };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: mot_video VIDEO [--config YAML] [options]\n";
            std::exit(0);
        }
        if (argument.rfind("--", 0) != 0) {
            if (!options.video.empty()) {
                usageError("only one positional VIDEO is accepted");
            }
            options.video = argument;
            continue;
        }
        if (argument == "--overwrite") {
            options.overwrite = true;
        } else if (argument == "--fuse-score") {
            botsort(argument).fuse_score = true;
        } else if (argument == "--no-fuse-score") {
            botsort(argument).fuse_score = false;
        } else if (argument == "--config" || argument == "--tracker") {
            nextValue(index, argc, argv, argument);
        } else if (argument == "--class-count") {
            options.config.detector.class_count = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--model" || argument == "--engine") {
            options.config.model_path = nextValue(index, argc, argv, argument);
        } else if (argument == "--output-dir") {
            options.output_dir = nextValue(index, argc, argv, argument);
        } else if (argument == "--input-width") {
            options.config.detector.input_width = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--input-height") {
            options.config.detector.input_height = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--conf") {
            options.config.detector.conf_threshold = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--iou") {
            options.config.detector.iou_threshold = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--max-det") {
            options.config.detector.max_det = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-high-thresh") {
            const auto value = parseDouble(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.track_high_thresh = value; });
        } else if (argument == "--track-low-thresh") {
            const auto value = parseDouble(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.track_low_thresh = value; });
        } else if (argument == "--new-track-thresh") {
            const auto value = parseDouble(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.new_track_thresh = value; });
        } else if (argument == "--track-buffer") {
            const auto value = parseInt(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.track_buffer = value; });
        } else if (argument == "--match-thresh") {
            const auto value = parseDouble(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.match_thresh = value; });
        } else if (argument == "--gmc-method") {
            const auto value = nextValue(index, argc, argv, argument);
            anyTracker([&](auto& config) { config.gmc_method = value; });
        } else if (argument == "--gmc-dpoint") {
            const auto value = nextValue(index, argc, argv, argument);
            anyTracker([&](auto& config) { config.gmc_dpoint_model = value; });
        } else if (argument == "--gmc-matcher") {
            const auto value = nextValue(index, argc, argv, argument);
            anyTracker([&](auto& config) { config.gmc_matcher_model = value; });
        } else if (argument == "--class-ema-alpha") {
            const auto value = parseDouble(argument, nextValue(index, argc, argv, argument));
            anyTracker([&](auto& config) { config.class_ema_alpha = value; });
        } else {
            usageError("unknown option '" + argument + "'");
        }
    }
    if (options.video.empty()) {
        usageError("VIDEO is required");
    }
    if (options.config.model_path.empty()) usageError("model path must not be empty");
    options.config.detector.validate();
    mot::validate(options.config.tracker);
    return options;
}

std::filesystem::path temporaryPath(const std::filesystem::path& directory,
                                     const std::string& stem,
                                     const std::string& suffix) {
#if defined(__unix__) || defined(__APPLE__)
    const auto process_id = static_cast<long long>(::getpid());
#else
    const auto process_id = 0LL;
#endif
    return directory / ("." + stem + "." + std::to_string(process_id) + ".tmp" + suffix);
}

std::unique_ptr<mot::IDetector> makeDetector(const mot::PipelineConfig& config,
                                             const std::vector<std::string>& class_names,
                                             std::string& backend) {
    const std::string extension = mot::apps::lowerExtension(config.model_path);
    if (extension == ".onnx") {
#if defined(MOT_HAS_ONNXRUNTIME)
        backend = "onnxruntime (CPU)";
        return std::make_unique<mot::OnnxRuntimeDetector>(config.model_path, config.detector, class_names);
#else
        throw std::runtime_error("ONNX model given but mot_video was built without ONNX Runtime "
                                 "(MOT_WITH_ONNXRUNTIME=OFF): " + config.model_path);
#endif
    }
    if (extension == ".engine" || extension == ".trt" || extension == ".plan") {
#if defined(MOT_HAS_TENSORRT)
        backend = "tensorrt";
        return std::make_unique<mot::TensorRTDetector>(config.model_path, config.detector, class_names);
#else
        throw std::runtime_error("TensorRT engine given but mot_video was built without TensorRT "
                                 "(MOT_WITH_TENSORRT=OFF): " + config.model_path);
#endif
    }
    throw std::runtime_error("cannot pick a detector backend for '" + config.model_path +
                             "'; available:" + backends());
}

int run(const Options& options) {
    if (!std::filesystem::is_regular_file(options.video)) {
        throw std::runtime_error("input video does not exist: " + options.video.string());
    }
    if (!std::filesystem::is_regular_file(options.config.model_path)) {
        throw std::runtime_error("model does not exist: " + options.config.model_path);
    }
    std::filesystem::create_directories(options.output_dir);
    const std::string stem = options.video.stem().string();
    const auto predictions = options.output_dir / (stem + "_predictions.txt");
    const auto annotated = options.output_dir / (stem + "_annotated.mp4");
    if (!options.overwrite && (std::filesystem::exists(predictions) || std::filesystem::exists(annotated))) {
        throw std::runtime_error("output already exists; pass --overwrite to replace it");
    }
    const auto temporary_predictions = temporaryPath(options.output_dir, stem, ".txt");
    const auto temporary_video = temporaryPath(options.output_dir, stem, ".mp4");
    std::error_code cleanup_error;
    std::filesystem::remove(temporary_predictions, cleanup_error);
    std::filesystem::remove(temporary_video, cleanup_error);

    cv::VideoCapture capture(options.video.string());
    if (!capture.isOpened()) {
        throw std::runtime_error("OpenCV cannot open input video: " + options.video.string());
    }
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 30.0;
    }
    const int expected_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int expected_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    auto class_names = options.config.class_names;
    class_names.resize(static_cast<std::size_t>(options.config.detector.class_count));
    std::string backend;
    std::string gmc_backend;
    auto gmc_model = mot::apps::makeGmcModel(options.config.tracker, gmc_backend);
    mot::Pipeline pipeline(makeDetector(options.config, class_names, backend),
                           mot::createTracker(options.config.tracker, std::move(gmc_model)));
    mot::Annotator annotator(class_names);
    std::cout << "Detector: " << backend << '\n';
    std::cout << "Tracker: " << pipeline.tracker().name() << '\n';
    if (!gmc_backend.empty()) std::cout << "GMC: dpoint (" << gmc_backend << ")\n";

    std::ofstream output_text(temporary_predictions);
    if (!output_text) {
        throw std::runtime_error("cannot create temporary predictions file: " + temporary_predictions.string());
    }
    output_text << std::fixed;
    cv::VideoWriter writer;
    bool writer_started = false;
    int output_width = 0;
    int output_height = 0;
    int frame_count = 0;
    std::size_t row_count = 0;
    try {
        cv::Mat frame;
        while (capture.read(frame)) {
            ++frame_count;
            auto tracks = pipeline.process(frame);
            cv::Mat annotated_frame = frame; // Pipeline has finished reading the original frame.
            annotator.draw(annotated_frame, tracks);
            if (!writer_started) {
                if (expected_width > 0 && expected_height > 0 &&
                    (annotated_frame.cols != expected_width || annotated_frame.rows != expected_height)) {
                    throw std::runtime_error("decoded frame size changed before output writer initialization");
                }
                writer.open(temporary_video,
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps,
                            cv::Size(annotated_frame.cols, annotated_frame.rows));
                if (!writer.isOpened()) {
                    throw std::runtime_error("OpenCV cannot create temporary output video: " + temporary_video.string());
                }
                writer_started = true;
                output_width = annotated_frame.cols;
                output_height = annotated_frame.rows;
            } else if (annotated_frame.cols != output_width || annotated_frame.rows != output_height) {
                throw std::runtime_error("decoded frame dimensions changed during video inference");
            }
            writer.write(annotated_frame);
            for (const auto& track : tracks) {
                const double width = track.bbox_xyxy[2] - track.bbox_xyxy[0];
                const double height = track.bbox_xyxy[3] - track.bbox_xyxy[1];
                output_text << frame_count << ',' << track.track_id << ','
                            << std::setprecision(3) << track.bbox_xyxy[0] << ','
                            << track.bbox_xyxy[1] << ',' << width << ',' << height << ','
                            << std::setprecision(6) << track.score << ',' << track.class_id + 1
                            << ",-1,-1\n";
                ++row_count;
            }
        }
        if (frame_count == 0) {
            throw std::runtime_error("no frames were decoded from input video");
        }
        output_text.flush();
        if (!output_text) {
            throw std::runtime_error("writing temporary predictions file failed");
        }
        output_text.close();
        writer.release();
        if (options.overwrite) {
            std::filesystem::remove(predictions);
            std::filesystem::remove(annotated);
        }
        std::filesystem::rename(temporary_predictions, predictions);
        std::filesystem::rename(temporary_video, annotated);
    } catch (...) {
        output_text.close();
        writer.release();
        std::filesystem::remove(temporary_predictions, cleanup_error);
        std::filesystem::remove(temporary_video, cleanup_error);
        throw;
    }
    std::cout << "Processed " << frame_count << " frames and wrote " << row_count << " tracked objects\n"
              << "Predictions: " << predictions << "\n"
              << "Annotated video: " << annotated << '\n';
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 2;
    }
}
