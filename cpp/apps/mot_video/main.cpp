#include "mot/pipeline/config.hpp"
#include "mot/pipeline/pipeline.hpp"
#include "mot/tensorrt/detector.hpp"
#include "mot/tracking/factory.hpp"
#include "mot/visualization/annotator.hpp"

#include <opencv2/videoio.hpp>

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

// Giá trị dòng lệnh cho video, engine, đầu ra và hai bước xử lý.
struct Options {
    std::filesystem::path video;
    std::filesystem::path output_dir{"runs/video_inference_cpp"};
    mot::PipelineConfig config;
    bool overwrite{false};
};

// Báo lỗi tham số kèm cú pháp chạy chương trình.
[[noreturn]] void usageError(const std::string& message) {
    // Ghép lỗi cụ thể với hướng dẫn cú pháp để người dùng sửa tham số.
    throw std::invalid_argument(message +
                                "\nUsage: mot_video VIDEO [--config YAML] [options]\n"
                                "  --config YAML                 default: weights/config.yaml (or MOT_CONFIG_PATH)\n"
                                "  --engine ENGINE               override YAML engine path\n"
                                "  --tracker TYPE                override YAML tracker.type\n"
                                "  --class-count N\n"
                                "  --output-dir DIR              output directory\n"
                                "  --input-width N --input-height N\n"
                                "  --conf F --iou F --max-det N\n"
                                "BoT-SORT (--tracker botsort):\n"
                                "  --track-high-thresh F --track-low-thresh F --new-track-thresh F\n"
                                "  --track-buffer N --match-thresh F\n"
                                "  --fuse-score | --no-fuse-score\n"
                                "  --gmc-method sparseOptFlow|none --gmc-downscale N\n"
                                "  --overwrite");
}

// Đọc số thực và báo tên tùy chọn khi giá trị không hợp lệ.
double parseDouble(const std::string& option, const std::string& text) {
    // Chuyển chuỗi thành số thực và gắn tên tùy chọn vào lỗi phân tích.
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

// Đọc số nguyên từ đối số dòng lệnh.
int parseInt(const std::string& option, const std::string& text) {
    // Chuyển chuỗi thành số nguyên và bác bỏ ký tự thừa.
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
    // Lấy đối số tiếp theo và báo lỗi nếu tùy chọn thiếu giá trị.
    if (index + 1 >= argc) {
        usageError(option + " requires a value");
    }
    return argv[++index];
}

// Ánh xạ đối số dòng lệnh vào cấu hình chạy video.
Options parseOptions(int argc, char** argv) {
    // Duyệt argv, ánh xạ từng cờ vào Options và kiểm tra tham số bắt buộc.
    Options options;
    // Read YAML first so explicit CLI overrides are independent of argument order.
    const char* environment_config = std::getenv("MOT_CONFIG_PATH");
    std::string config_path = environment_config && *environment_config
        ? environment_config : "weights/config.yaml";
    std::string tracker_type;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::string trackers;
            for (const auto name : mot::availableTrackers()) trackers += " " + std::string(name);
            std::cout << "Usage: mot_video VIDEO [--config YAML] [options]\n"
                      << "Default config: weights/config.yaml (or MOT_CONFIG_PATH)\n"
                      << "Trackers:" << trackers << " (select with --tracker TYPE)\n"
                      << "CLI overrides: --engine, --class-count, --input-width, --input-height,\n"
                      << "  --conf, --iou, --max-det, --track-high-thresh, --track-low-thresh,\n"
                      << "  --new-track-thresh, --track-buffer, --match-thresh, --fuse-score,\n"
                      << "  --no-fuse-score, --gmc-method, --gmc-downscale\n"
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
    // Đổi tracker trước để các cờ riêng của tracker áp dụng đúng kiểu.
    if (!tracker_type.empty() && tracker_type != mot::trackerType(options.config.tracker)) {
        options.config.tracker = mot::defaultTrackerConfig(tracker_type);
    }
    const auto botsort = [&](const std::string& option) -> mot::BoTSORTConfig& {
        auto* config = std::get_if<mot::BoTSORTConfig>(&options.config.tracker);
        if (!config) usageError(option + " only applies to --tracker botsort");
        return *config;
    };
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
        } else if (argument == "--engine") {
            options.config.engine_path = nextValue(index, argc, argv, argument);
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
            botsort(argument).track_high_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-low-thresh") {
            botsort(argument).track_low_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--new-track-thresh") {
            botsort(argument).new_track_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-buffer") {
            botsort(argument).track_buffer = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--match-thresh") {
            botsort(argument).match_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--gmc-method") {
            botsort(argument).gmc_method = nextValue(index, argc, argv, argument);
        } else if (argument == "--gmc-downscale") {
            botsort(argument).gmc_downscale = parseInt(argument, nextValue(index, argc, argv, argument));
        } else {
            usageError("unknown option '" + argument + "'");
        }
    }
    if (options.video.empty()) {
        usageError("VIDEO is required");
    }
    if (options.config.engine_path.empty()) usageError("engine path must not be empty");
    options.config.detector.validate();
    mot::validate(options.config.tracker);
    return options;
}

std::filesystem::path temporaryPath(const std::filesystem::path& directory,
                                     const std::string& stem,
                                     const std::string& suffix) {
    // Gắn PID để tệp tạm của các tiến trình song song không trùng tên.
#if defined(__unix__) || defined(__APPLE__)
    const auto process_id = static_cast<long long>(::getpid());
#else
    const auto process_id = 0LL;
#endif
    return directory / ("." + stem + "." + std::to_string(process_id) + ".tmp" + suffix);
}

// Chạy suy luận từng frame và ghi video cùng kết quả theo dõi.
int run(const Options& options) {
    // Kiểm tra đầu vào và chuẩn bị tên file tạm trước khi mở video/engine.
    if (!std::filesystem::is_regular_file(options.video)) {
        throw std::runtime_error("input video does not exist: " + options.video.string());
    }
    if (!std::filesystem::is_regular_file(options.config.engine_path)) {
        throw std::runtime_error("TensorRT engine does not exist: " + options.config.engine_path);
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
    mot::Pipeline pipeline(
        std::make_unique<mot::TensorRTDetector>(options.config.engine_path, options.config.detector,
                                                class_names),
        mot::createTracker(options.config.tracker));
    mot::Annotator annotator(class_names);
    std::cout << "Tracker: " << pipeline.tracker().name() << '\n';

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
        // Mỗi frame: phát hiện/theo dõi, vẽ hộp, ghi video và các dòng kết quả.
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
        // Chỉ chuyển file tạm thành đầu ra chính sau khi xử lý thành công.
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

}  // namespace

int main(int argc, char** argv) {
    // Phân tích tham số, chạy pipeline và in lỗi ra stderr nếu thất bại.
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 2;
    }
}
