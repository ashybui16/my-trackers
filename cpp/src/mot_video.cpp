#include "mot_inference/pipeline.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

// Giá trị dòng lệnh cho video, engine, đầu ra và hai bước xử lý.
struct Options {
    std::filesystem::path video;
    std::filesystem::path engine;
    std::filesystem::path output_dir{"runs/video_inference_cpp"};
    int input_width{640};
    int input_height{640};
    double conf{0.01};
    double iou{0.70};
    int max_det{300};
    double track_high_thresh{0.25};
    double track_low_thresh{0.10};
    double new_track_thresh{0.25};
    int track_buffer{30};
    double match_thresh{0.80};
    bool fuse_score{true};
    std::string gmc_method{"sparseOptFlow"};
    int gmc_downscale{2};
    bool overwrite{false};
};

// Báo lỗi tham số kèm cú pháp chạy chương trình.
[[noreturn]] void usageError(const std::string& message) {
    // Ghép lỗi cụ thể với hướng dẫn cú pháp để người dùng sửa tham số.
    throw std::invalid_argument(message +
                                "\nUsage: mot_video VIDEO --engine ENGINE [options]\n"
                                "  --output-dir DIR              output directory\n"
                                "  --input-width N --input-height N\n"
                                "  --conf F --iou F --max-det N\n"
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
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: mot_video VIDEO --engine ENGINE [options]\n";
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
            options.fuse_score = true;
        } else if (argument == "--no-fuse-score") {
            options.fuse_score = false;
        } else if (argument == "--engine") {
            options.engine = nextValue(index, argc, argv, argument);
        } else if (argument == "--output-dir") {
            options.output_dir = nextValue(index, argc, argv, argument);
        } else if (argument == "--input-width") {
            options.input_width = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--input-height") {
            options.input_height = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--conf") {
            options.conf = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--iou") {
            options.iou = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--max-det") {
            options.max_det = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-high-thresh") {
            options.track_high_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-low-thresh") {
            options.track_low_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--new-track-thresh") {
            options.new_track_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--track-buffer") {
            options.track_buffer = parseInt(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--match-thresh") {
            options.match_thresh = parseDouble(argument, nextValue(index, argc, argv, argument));
        } else if (argument == "--gmc-method") {
            options.gmc_method = nextValue(index, argc, argv, argument);
        } else if (argument == "--gmc-downscale") {
            options.gmc_downscale = parseInt(argument, nextValue(index, argc, argv, argument));
        } else {
            usageError("unknown option '" + argument + "'");
        }
    }
    if (options.video.empty()) {
        usageError("VIDEO is required");
    }
    if (options.engine.empty()) {
        usageError("--engine is required");
    }
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

// Vẽ hộp và ID lên frame xuất ra video.
void drawTracks(cv::Mat& frame, const std::vector<mot_tracker::TrackOutput>& tracks) {
    // Đổi hộp sang pixel nguyên rồi vẽ khung, ID, lớp và điểm tin cậy.
    for (const auto& track : tracks) {
        const int x1 = static_cast<int>(std::lround(track.bbox_xyxy[0]));
        const int y1 = static_cast<int>(std::lround(track.bbox_xyxy[1]));
        const int x2 = static_cast<int>(std::lround(track.bbox_xyxy[2]));
        const int y2 = static_cast<int>(std::lround(track.bbox_xyxy[3]));
        const cv::Scalar color((37 * track.track_id) % 255,
                               (67 * track.track_id) % 255,
                               (101 * track.track_id) % 255);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        std::ostringstream label;
        label << track.track_id << ':' << track.class_id + 1 << ' '
              << std::fixed << std::setprecision(2) << track.score;
        cv::putText(frame, label.str(), cv::Point(x1, std::max(0, y1 - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
}

// Chạy suy luận từng frame và ghi video cùng kết quả theo dõi.
int run(const Options& options) {
    // Kiểm tra đầu vào và chuẩn bị tên file tạm trước khi mở video/engine.
    if (!std::filesystem::is_regular_file(options.video)) {
        throw std::runtime_error("input video does not exist: " + options.video.string());
    }
    if (!std::filesystem::is_regular_file(options.engine)) {
        throw std::runtime_error("TensorRT engine does not exist: " + options.engine.string());
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

    // Chuyển tùy chọn dòng lệnh thành cấu hình detector và tracker.
    mot_inference::PipelineConfig config;
    config.detector.input_width = options.input_width;
    config.detector.input_height = options.input_height;
    config.detector.conf_threshold = options.conf;
    config.detector.iou_threshold = options.iou;
    config.detector.max_det = options.max_det;
    config.tracker.track_high_thresh = options.track_high_thresh;
    config.tracker.track_low_thresh = options.track_low_thresh;
    config.tracker.new_track_thresh = options.new_track_thresh;
    config.tracker.track_buffer = options.track_buffer;
    config.tracker.match_thresh = options.match_thresh;
    config.tracker.fuse_score = options.fuse_score;
    config.tracker.gmc_method = options.gmc_method;
    config.tracker.gmc_downscale = options.gmc_downscale;
    mot_inference::MotPipeline pipeline(options.engine.string(), config);

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
            auto tracks = pipeline.processFrame(frame);
            cv::Mat annotated_frame = frame.clone();
            drawTracks(annotated_frame, tracks);
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
