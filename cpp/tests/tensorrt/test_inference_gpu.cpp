#include "mot/tensorrt/detector.hpp"
#include "mot/pipeline/pipeline.hpp"
#include "mot/tracking/factory.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void expectInvalidFrame(mot::TensorRTDetector& detector, const cv::Mat& frame) {
    try {
        detector.detect(frame);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("detector accepted a frame outside the CV_8UC3 contract");
}

void expectSameDetections(const std::vector<mot::Detection>& first,
                          const std::vector<mot::Detection>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("reusing buffers changed the detection count");
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (first[index].class_id != second[index].class_id ||
            std::abs(first[index].score - second[index].score) > 1e-5) {
            throw std::runtime_error("reusing buffers changed detection scores or classes");
        }
        for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
            if (std::abs(first[index].bbox_xyxy[coordinate] - second[index].bbox_xyxy[coordinate]) > 1e-5) {
                throw std::runtime_error("reusing buffers changed detection coordinates");
            }
        }
    }
}

}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "No NVIDIA CUDA device is available; skipping TensorRT test\n";
        return 77;
    }
    const char* path = std::getenv("MOT_ENGINE_PATH");
    if (path == nullptr || std::string(path).empty()) {
        std::cerr << "MOT_ENGINE_PATH is not set; skipping TensorRT test\n";
        return 77;
    }
    try {
        mot::DetectorConfig config;
        config.conf_threshold = 0.0;
        config.max_det = 20;
        mot::TensorRTDetector detector(path, config);
        expectInvalidFrame(detector, cv::Mat{});
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_8UC1));
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_8UC4));
        expectInvalidFrame(detector, cv::Mat::zeros(16, 16, CV_32FC3));
        const int sizes[] = {2, 3, 4};
        expectInvalidFrame(detector, cv::Mat(3, sizes, CV_8UC3, cv::Scalar::all(0)));
        cv::Mat frame;
        const char* video_path = std::getenv("MOT_VIDEO_PATH");
        if (video_path != nullptr && std::string(video_path).size() > 0) {
            cv::VideoCapture capture(video_path);
            if (!capture.isOpened() || !capture.read(frame) || frame.empty()) {
                throw std::runtime_error("could not read a frame from MOT_VIDEO_PATH");
            }
        } else {
            frame = cv::Mat::zeros(480, 640, CV_8UC3);
        }
        const auto detections = detector.detect(frame);
        // Change aspect ratio, then return to the original frame to catch stale padding data.
        detector.detect(cv::Mat(641, 319, CV_8UC3, cv::Scalar(10, 80, 200)));
        expectSameDetections(detections, detector.detect(frame));

        // A ROI has a different stride from a continuous image but must give the same result.
        cv::Mat padded(frame.rows + 2, frame.cols + 2, CV_8UC3, cv::Scalar::all(0));
        cv::Mat roi = padded(cv::Rect(1, 1, frame.cols, frame.rows));
        frame.copyTo(roi);
        expectSameDetections(detections, detector.detect(roi));
        mot::Pipeline pipeline(std::make_unique<mot::TensorRTDetector>(path, config),
                               mot::createTracker(mot::BoTSORTConfig{}));
        const auto original = frame.clone();
        const auto first_tracks = pipeline.process(frame);
        if (cv::norm(frame, original, cv::NORM_INF) != 0)
            throw std::runtime_error("pipeline modified caller image");
        pipeline.reset();
        const auto reset_tracks = pipeline.process(frame);
        if (first_tracks.size() != reset_tracks.size())
            throw std::runtime_error("pipeline reset changed first-frame outputs");
        for (std::size_t i = 0; i < first_tracks.size(); ++i) {
            if (first_tracks[i].track_id != reset_tracks[i].track_id ||
                first_tracks[i].bbox_xyxy != reset_tracks[i].bbox_xyxy)
                throw std::runtime_error("pipeline reset changed first-frame IDs/boxes");
        }
        std::cout << "TensorRT inference succeeded with " << detections.size()
                  << " detections using bindings '" << detector.inputBindingName()
                  << "' -> '" << detector.outputBindingName() << "'\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "TensorRT inference test failed: " << exception.what() << '\n';
        return 1;
    }
}
