#include "mot/onnxruntime/detector.hpp"
#include "mot/pipeline/pipeline.hpp"
#include "mot/tracking/factory.hpp"

#include "support/check.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mot::test::expect;

void expectInvalidFrame(mot::OnnxRuntimeDetector& detector, const cv::Mat& frame) {
    try {
        detector.detect(frame);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("detector accepted a frame outside the CV_8UC3 contract");
}

void expectSameDetections(const std::vector<mot::Detection>& first,
                          const std::vector<mot::Detection>& second) {
    expect(first.size() == second.size(), "reusing buffers changed the detection count");
    for (std::size_t index = 0; index < first.size(); ++index) {
        expect(first[index].class_id == second[index].class_id &&
                   std::abs(first[index].score - second[index].score) <= 1e-5,
               "reusing buffers changed detection scores or classes");
        for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
            expect(std::abs(first[index].bbox_xyxy[coordinate] - second[index].bbox_xyxy[coordinate]) <= 1e-5,
                   "reusing buffers changed detection coordinates");
        }
    }
}

cv::Mat testFrame() {
    const char* video_path = std::getenv("MOT_VIDEO_PATH");
    if (video_path == nullptr || std::string(video_path).empty()) {
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(90, 110, 100));
        frame(cv::Rect(100, 200, 60, 30)).setTo(cv::Scalar(30, 30, 200));
        frame(cv::Rect(400, 120, 20, 50)).setTo(cv::Scalar(200, 200, 200));
        return frame;
    }
    cv::VideoCapture capture(video_path);
    cv::Mat frame;
    if (!capture.isOpened() || !capture.read(frame) || frame.empty()) {
        throw std::runtime_error("could not read a frame from MOT_VIDEO_PATH");
    }
    return frame;
}

}

int main(int argc, char** argv) {
    const char* environment_model = std::getenv("MOT_ONNX_MODEL_PATH");
    const std::string path = environment_model && *environment_model ? environment_model
                             : argc > 1                              ? argv[1]
                                                                     : "";
    if (path.empty() || !std::filesystem::is_regular_file(path)) {
        std::cerr << "ONNX model '" << path << "' not found; skipping ONNX Runtime test\n";
        return 77;
    }
    try {
        mot::DetectorConfig config;
        config.conf_threshold = 0.0;
        config.max_det = 20;
        mot::OnnxRuntimeDetector detector(path, config);
        expect(!detector.inputName().empty() && !detector.outputName().empty(), "tensor names");

        for (const auto& bad : {mot::DetectorConfig{640, 640, 0.0, 0.7, 20, 3},
                                mot::DetectorConfig{320, 320, 0.0, 0.7, 20, 10}}) {
            bool rejected = false;
            try {
                mot::OnnxRuntimeDetector mismatched(path, bad);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            expect(rejected, "detector accepted a config that does not match the model shape");
        }

        expectInvalidFrame(detector, cv::Mat{});
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_8UC1));
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_8UC4));
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_32FC3));
        const int sizes[] = {2, 3, 4};
        expectInvalidFrame(detector, cv::Mat(3, sizes, CV_8UC3, cv::Scalar::all(0)));

        const cv::Mat frame = testFrame();
        const auto detections = detector.detect(frame);
        // With conf_threshold = 0 there must always be max_det boxes, all inside the image.
        expect(!detections.empty() && static_cast<int>(detections.size()) <= config.max_det,
               "detection count within (0, max_det]");
        for (const auto& detection : detections) {
            expect(detection.class_id >= 0 && detection.class_id < config.class_count, "class id range");
            expect(detection.score >= 0.0 && detection.score <= 1.0, "score range");
            expect(detection.bbox_xyxy[0] >= 0 && detection.bbox_xyxy[1] >= 0 &&
                       detection.bbox_xyxy[2] <= frame.cols && detection.bbox_xyxy[3] <= frame.rows &&
                       detection.bbox_xyxy[0] <= detection.bbox_xyxy[2] &&
                       detection.bbox_xyxy[1] <= detection.bbox_xyxy[3],
                   "box inside frame");
        }

        // Change aspect ratio, then return to the original frame to catch stale padding data.
        detector.detect(cv::Mat(641, 319, CV_8UC3, cv::Scalar(10, 80, 200)));
        expectSameDetections(detections, detector.detect(frame));

        // A ROI has a different stride from a continuous image but must give the same result.
        cv::Mat padded(frame.rows + 2, frame.cols + 2, CV_8UC3, cv::Scalar::all(0));
        cv::Mat roi = padded(cv::Rect(1, 1, frame.cols, frame.rows));
        frame.copyTo(roi);
        expectSameDetections(detections, detector.detect(roi));

        mot::OnnxRuntimeDetector moved(std::move(detector));
        expectSameDetections(detections, moved.detect(frame));

        mot::Pipeline pipeline(std::make_unique<mot::OnnxRuntimeDetector>(path, config),
                               mot::createTracker(mot::BoTSORTConfig{}));
        const auto original = frame.clone();
        pipeline.process(frame);
        expect(cv::norm(frame, original, cv::NORM_INF) == 0, "pipeline modified caller image");

        std::cout << "ONNX Runtime inference succeeded with " << detections.size()
                  << " detections using tensors '" << moved.inputName() << "' -> '"
                  << moved.outputName() << "'\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "ONNX Runtime inference test failed: " << exception.what() << '\n';
        return 1;
    }
}
