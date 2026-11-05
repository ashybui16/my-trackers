#include "mot_annotator/annotation.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("pass the engine path for class-name lookup");
        mot_inference::MotAnnotator annotator(argv[1], 10);
        cv::Mat empty;
        annotator.draw(empty, {});
        cv::Mat frame = cv::Mat::zeros(120, 160, CV_8UC3);
        const auto original = frame.clone();
        annotator.draw(frame, {{7, {10, 20, 100, 90}, 0.9, 3}});
        if (cv::norm(frame, original, cv::NORM_L1) == 0)
            throw std::runtime_error("track annotation missing");
        frame.setTo(0);
        annotator.drawDetections(frame, {{{-20, -10, 80, 100}, 0.8, 99}});
        if (cv::norm(frame, original, cv::NORM_L1) == 0)
            throw std::runtime_error("clipped detection annotation missing");
        frame.setTo(0);
        mot_tracker::TrackOutput invalid;
        invalid.bbox_xyxy = {100, 90, 10, 20};
        annotator.draw(frame, {invalid});
        invalid.bbox_xyxy = {0, 0, std::numeric_limits<double>::quiet_NaN(), 10};
        annotator.draw(frame, {invalid});
        if (cv::norm(frame, original, cv::NORM_L1) != 0)
            throw std::runtime_error("invalid boxes must be skipped");
        cv::Mat tiny = cv::Mat::zeros(1, 1, CV_8UC3);
        annotator.drawDetections(tiny, {{{0, 0, 10, 10}, 0.5, 0}});
        std::cout << "MOT annotation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
