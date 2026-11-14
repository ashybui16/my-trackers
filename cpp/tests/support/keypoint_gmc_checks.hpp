#pragma once

#include "mot/tracking/factory.hpp"
#include "mot/tracking/keypoint_gmc.hpp"

#include "support/check.hpp"
#include "tracking/common/gmc.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mot::test {

using ModelFactory =
    std::function<std::unique_ptr<IKeypointGmcModel>(const std::string& dpoint, const std::string& matcher)>;


inline void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + " +/- " +
                                 std::to_string(tolerance) + ", got " + std::to_string(actual));
    }
}

// Corner-rich texture so the keypoint model has something to detect everywhere.
inline cv::Mat texturedFrame(int width, int height) {
    cv::Mat noise(height, width, CV_8UC1);
    cv::RNG rng(7);
    rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(noise, noise, cv::Size(0, 0), 3.0);
    cv::normalize(noise, noise, 0, 255, cv::NORM_MINMAX);
    cv::Mat frame;
    cv::cvtColor(noise, frame, cv::COLOR_GRAY2BGR);
    for (int index = 0; index < 120; ++index) {
        const cv::Point corner(rng.uniform(0, width), rng.uniform(0, height));
        const cv::Size size(rng.uniform(15, 90), rng.uniform(15, 90));
        const cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::rectangle(frame, cv::Rect(corner, size), color, cv::FILLED);
    }
    return frame;
}

inline cv::Mat warped(const cv::Mat& frame, const cv::Mat& warp) {
    cv::Mat result;
    cv::warpAffine(frame, result, warp, frame.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return result;
}

inline void expectWarp(const cv::Mat& actual, const cv::Mat& expected, double linear_tol, double translation_tol,
                const std::string& label) {
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            expectNear(actual.at<double>(row, column), expected.at<double>(row, column), linear_tol,
                       label + " linear part");
        }
        expectNear(actual.at<double>(row, 2), expected.at<double>(row, 2), translation_tol,
                   label + " translation");
    }
}


// Shared checks for IKeypointGmcModel backends. argv: <dpoint model> <matcher model>; the
// MOT_DPOINT_MODEL_PATH / MOT_MATCHER_MODEL_PATH environment variables override them.
// Returns 77 (skip) when the models are missing.
inline int runKeypointGmcTest(int argc, char** argv, const std::string& label, const ModelFactory& make) {
    const char* dpoint_env = std::getenv("MOT_DPOINT_MODEL_PATH");
    const char* matcher_env = std::getenv("MOT_MATCHER_MODEL_PATH");
    const std::string dpoint = dpoint_env && *dpoint_env ? dpoint_env : (argc > 1 ? argv[1] : "");
    const std::string matcher = matcher_env && *matcher_env ? matcher_env : (argc > 2 ? argv[2] : "");
    if (!std::filesystem::is_regular_file(dpoint) || !std::filesystem::is_regular_file(matcher)) {
        std::cout << "SKIP: " << label << " keypoint GMC models not found ('" << dpoint << "', '" << matcher << "')\n";
        return 77;
    }
    try {
        auto model = make(dpoint, matcher);
        expect(model->inputSize().area() > 0, "model reports its input size");

        // Raw model: the first call has no previous frame; an identical frame matches itself.
        const cv::Size size = model->inputSize();
        cv::Mat image(size, CV_32FC1);
        cv::Mat gray;
        cv::cvtColor(texturedFrame(size.width, size.height), gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(image, CV_32F, 1.0 / 255.0);
        std::vector<cv::Point2f> previous, current;
        expect(!model->extractAndMatch(image, previous, current) && previous.empty(),
               "first call has no previous frame");
        expect(model->extractAndMatch(image, previous, current), "second call matches");
        expect(previous.size() >= 50 && previous.size() == current.size(), "identical frames give many matches");
        for (std::size_t index = 0; index < previous.size(); ++index) {
            expect(cv::norm(previous[index] - current[index]) < 1e-3, "identical frames match in place");
        }
        model->reset();
        expect(!model->extractAndMatch(image, previous, current), "reset forgets the previous frame");
        model->reset();

        // GMC end to end on a non-square frame: recover a known similarity transform in frame pixels.
        mot::GMC gmc("dpoint", 2, std::move(model));
        const cv::Mat frame = texturedFrame(1280, 720);
        const cv::Mat identity = gmc.apply(frame);
        expectWarp(identity, cv::Mat::eye(2, 3, CV_64F), 1e-12, 1e-12, "first frame");

        const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 12, 0, 1, -7);
        const cv::Mat shifted = warped(frame, shift);
        const cv::Mat recovered_shift = gmc.apply(shifted);
        std::cout << "recovered shift:\n" << recovered_shift << '\n';
        expectWarp(recovered_shift, shift, 0.01, 1.5, "translation");

        const cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(640, 360), 2.0, 1.02);
        const cv::Mat recovered_rotation = gmc.apply(warped(shifted, rotation));
        std::cout << "expected rotation:\n" << rotation << "\nrecovered rotation:\n" << recovered_rotation << '\n';
        expectWarp(recovered_rotation, rotation, 0.01, 3.0, "rotation + scale");

        gmc.reset();
        expectWarp(gmc.apply(frame), cv::Mat::eye(2, 3, CV_64F), 1e-12, 1e-12, "first frame after reset");

        // Trackers accept the model through the factory.
        mot::BoTSORTConfig config;
        config.gmc_method = "dpoint";
        auto tracker = mot::createTracker(config, make(dpoint, matcher));
        tracker->update({{{100, 100, 140, 180}, 0.9, 0}}, frame);
        tracker->update({{{112, 93, 152, 173}, 0.9, 0}}, shifted);

        std::cout << label << " keypoint GMC succeeded\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << label << " keypoint GMC test failed: " << exception.what() << '\n';
        return 1;
    }
}

}
