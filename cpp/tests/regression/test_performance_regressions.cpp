#include "detection/yolo/preprocess.hpp"
#include "detection/yolo/postprocess.hpp"
#include "tracking/common/assignment.hpp"
#include "tracking/common/association.hpp"
#include "tracking/common/geometry.hpp"
#include "tracking/common/gmc.hpp"
#include "tracking/common/track.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void expect(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
void assignmentTests() {
    std::mt19937 rng(42);
    for (int rows = 1; rows <= 4; ++rows)
        for (int cols = 1; cols <= 4; ++cols)
            for (int sample = 0; sample < 40; ++sample) {
                std::vector<std::vector<double>> cost(rows, std::vector<double>(cols));
                for (auto &row : cost)
                    for (auto &value : row)
                        value = (rng() % 11) / 10.0;
                const double limit = 0.5;
                double optimum = 0;
                std::function<void(int, unsigned, double)> enumerate = [&](int row, unsigned used,
                                                                           double sum) {
                    if (row == rows) {
                        optimum = std::min(optimum, sum);
                        return;
                    }
                    enumerate(row + 1, used, sum);
                    for (int col = 0; col < cols; ++col)
                        if (!(used & (1U << col)) && cost[row][col] <= limit)
                            enumerate(row + 1, used | (1U << col), sum + cost[row][col] - limit - 1e-12);
                };
                enumerate(0, 0, 0);
                const auto result = mot::linearAssignment(cost, limit);
                double actual = 0;
                std::vector<bool> seen_rows(rows), seen_cols(cols);
                for (auto match : result.matches) {
                    expect(!seen_rows[match.first] && !seen_cols[match.second], "duplicate assignment");
                    expect(cost[match.first][match.second] <= limit, "forbidden assignment");
                    seen_rows[match.first] = true;
                    seen_cols[match.second] = true;
                    actual += cost[match.first][match.second] - limit - 1e-12;
                }
                expect(std::abs(actual - optimum) < 1e-10, "assignment differs from exhaustive optimum");
                for (int row : result.unmatched_rows) {
                    expect(!seen_rows[row], "matched row also unmatched");
                    seen_rows[row] = true;
                }
                for (int col : result.unmatched_cols) {
                    expect(!seen_cols[col], "matched col also unmatched");
                    seen_cols[col] = true;
                }
                expect(std::all_of(seen_rows.begin(), seen_rows.end(), [](bool x) { return x; }),
                       "missing assignment row");
                expect(std::all_of(seen_cols.begin(), seen_cols.end(), [](bool x) { return x; }),
                       "missing assignment column");
            }
    const auto equal = mot::linearAssignment({{0.5, 0.9}, {0.9, 0.5}}, 0.5);
    expect(equal.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "cost equal to threshold");
    const auto forbidden = mot::linearAssignment({{0.9, 0.8}, {0.9, 0.8}}, 0.5);
    expect(forbidden.matches.empty() && forbidden.unmatched_rows.size() == 2 &&
               forbidden.unmatched_cols.size() == 2,
           "all forbidden");
    expect(mot::linearAssignment({}, 0.5).matches.empty(), "empty assignment");
}
void preprocessTests() {
    cv::Mat canvas(64, 64, CV_8UC3), reference(64, 64, CV_8UC3);
    std::vector<float> actual(3 * 64 * 64), expected(actual.size());
    cv::RNG rng(43);
    for (auto size : {cv::Size(64, 64), cv::Size(90, 33), cv::Size(19, 87), cv::Size(1, 1), cv::Size(81, 63),
                      cv::Size(64, 64)}) {
        cv::Mat padded(size.height + 2, size.width + 4, CV_8UC3);
        rng.fill(padded, cv::RNG::UNIFORM, 0, 256);
        const cv::Mat frame = padded(cv::Rect(1, 1, size.width, size.height));
        const auto before = frame.clone();
        const auto info = mot::detail::preprocess(frame, canvas, actual.data());
        const double scale = std::min(64.0 / frame.cols, 64.0 / frame.rows);
        const int width = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
        const int height = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
        const int px = (64 - width) / 2, py = (64 - height) / 2;
        reference.setTo(cv::Scalar(114, 114, 114));
        cv::Mat roi = reference(cv::Rect(px, py, width, height));
        cv::resize(frame, roi, roi.size());
        for (int r = 0; r < 64; ++r)
            for (int c = 0; c < 64; ++c)
                for (int ch = 0; ch < 3; ++ch)
                    expected[ch * 64 * 64 + r * 64 + c] =
                        static_cast<float>(reference.at<cv::Vec3b>(r, c)[2 - ch]) / 255.0F;
        expect(actual == expected, "preprocess tensor/padding/stride changed");
        expect(info.scale == scale && info.pad_x == px && info.pad_y == py, "letterbox changed");
        expect(cv::norm(frame, before, cv::NORM_INF) == 0, "preprocess modified caller image");
    }
}
void workspaceTests() {
    mot::detail::OutputWorkspace scratch;
    mot::detail::OutputOptions options;
    options.class_count = 2;
    options.max_det = 40000;
    options.transform = {1, 0, 0, 100, 100};
    for (int n : {30001, 2, 1000, 1, 30001}) {
        std::vector<float> raw(6 * n, 0.0F);
        for (int i = 0; i < n; ++i) {
            raw[i] = 50;
            raw[n + i] = 50;
            raw[2 * n + i] = 20;
            raw[3 * n + i] = 20;
            raw[4 * n + i] = 0.9F;
        }
        raw[4 * n + n - 1] = 0;
        raw[5 * n + n - 1] = 0.9F;
        const auto result = mot::detail::processOutput(raw.data(), n, options, scratch);
        expect(result.size() == static_cast<std::size_t>(n > 30000 ? 1
                                                         : n == 1  ? 1
                                                                   : 2),
               "NMS workspace or top-K tie order");
        expect(result.front().class_id == (n == 1 ? 1 : 0), "NMS source-index tie changed");
        std::fill(raw.begin(), raw.end(), 0);
        expect(mot::detail::processOutput(raw.data(), n, options, scratch).empty(),
               "stale NMS detections");
    }
}
void stateTests() {
    mot::KalmanFilterXYWH filter;
    mot::Track track({{10, 20, 30, 60}, 0.9, 0});
    track.activate(filter, 1, 1);
    const cv::Mat original_mean = track.mean(), original_cov = track.covariance();
    const auto saved_mean = original_mean.clone(), saved_cov = original_cov.clone();
    track.update(mot::Track({{12, 20, 32, 60}, 0.8, 1}), 2);
    expect(cv::norm(original_mean, saved_mean, cv::NORM_INF) == 0 &&
               cv::norm(original_cov, saved_cov, cv::NORM_INF) == 0,
           "update mutated shared input state");
    track.markLost();
    const auto old_mean = track.mean(), saved_lost = old_mean.clone();
    track.predict();
    expect(cv::norm(old_mean, saved_lost, cv::NORM_INF) == 0, "lost predict mutated old mean");
    const auto stable_mean = track.mean().clone(), stable_cov = track.covariance().clone();
    cv::Mat warp = cv::Mat::eye(2, 3, CV_64F);
    warp.at<double>(0, 0) = std::numeric_limits<double>::max();
    track.applyWarp(warp);
    expect(cv::norm(stable_mean, track.mean(), cv::NORM_INF) == 0 &&
               cv::norm(stable_cov, track.covariance(), cv::NORM_INF) == 0,
           "overflow warp rollback");
    warp.at<double>(0, 0) = std::numeric_limits<double>::quiet_NaN();
    track.applyWarp(warp);
    expect(cv::norm(stable_mean, track.mean(), cv::NORM_INF) == 0, "invalid warp rollback");
    auto first = std::make_shared<mot::Track>(mot::Detection({0, 0, 10, 10}, 0.8, 0));
    auto second = std::make_shared<mot::Track>(mot::Detection({2, 0, 12, 10}, 0.6, 1));
    const auto plain = mot::trackDistance({first}, {second}, false);
    expect(mot::trackDistance({first}, {second}, true) == mot::fuseScore(plain, {0.6}),
           "in-place score fusion");
}
void gmcOwnershipTests() {
    cv::RNG rng(44);
    for (int channels : {1, 3})
        for (int downscale : {1, 2}) {
            mot::GMC gmc("sparseOptFlow", downscale);
            cv::Mat frame(120, 160, CV_MAKETYPE(CV_8U, channels));
            rng.fill(frame, cv::RNG::UNIFORM, 0, 256);
            const auto first = frame.clone();
            gmc.apply(frame);
            const cv::Mat held = gmc.previousFrame();
            const auto held_copy = held.clone();
            frame.setTo(0);
            expect(cv::norm(held, held_copy, cv::NORM_INF) == 0, "GMC history aliases caller");
            for (int i = 0; i < 4; ++i)
                gmc.apply(first);
            expect(cv::norm(held, held_copy, cv::NORM_INF) == 0, "GMC recycled retained history buffer");
            gmc.reset();
            expect(gmc.previousFrame().empty() && !gmc.initializedFirstFrame(), "GMC reset history");
            const auto identity = gmc.apply(first);
            expect(identity.at<double>(0, 2) == 0 && identity.at<double>(1, 2) == 0, "GMC reset first warp");
        }
}
}
int main() {
    try {
        assignmentTests();
        preprocessTests();
        workspaceTests();
        stateTests();
        gmcOwnershipTests();
        std::cout << "performance regression tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
