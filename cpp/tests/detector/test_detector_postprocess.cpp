#include "detector/detector_postprocess.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kClassCount = 10;
constexpr int kFeatureCount = 4 + kClassCount;

struct Tensor {
    std::vector<float> values;
    int candidate_count{0};
};

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

Tensor makeTensor(int candidate_count) {
    return {std::vector<float>(static_cast<std::size_t>(kFeatureCount) * candidate_count, 0.0F),
            candidate_count};
}

void setValue(Tensor& tensor, int candidate, int feature, float value) {
    const std::size_t offset = static_cast<std::size_t>(feature) * tensor.candidate_count + candidate;
    tensor.values[offset] = value;
}

void setCandidate(Tensor& tensor,
                  int candidate,
                  float center_x,
                  float center_y,
                  float width,
                  float height,
                  int class_id,
                  float score) {
    setValue(tensor, candidate, 0, center_x);
    setValue(tensor, candidate, 1, center_y);
    setValue(tensor, candidate, 2, width);
    setValue(tensor, candidate, 3, height);
    setValue(tensor, candidate, 4 + class_id, score);
}

mot_inference::detail::OutputOptions options() {
    mot_inference::detail::OutputOptions result;
    result.class_count = kClassCount;
    result.conf_threshold = 0.01;
    result.iou_threshold = 0.70;
    result.max_det = 300;
    result.transform = {1.0, 0, 0, 100, 100};
    return result;
}

std::vector<mot_tracker::Detection> run(const Tensor& tensor,
                                        const mot_inference::detail::OutputOptions& config) {
    return mot_inference::detail::processOutput(
        tensor.values.data(),
        tensor.candidate_count,
        config);
}

void testVendoredUltralyticsReference() {
    // Golden output from vendored Ultralytics 8.4.157 NMS + scale_boxes, then
    // filtered for positive clipped area to satisfy the Detection contract.
    // input shape (50, 50), original shape (60, 80), ratio_pad=((0.625, 0.625), (0, 6)).
    // The equal class-0 boxes overlap. The last Ultralytics row clips to zero width and is discarded here.
    const std::array<std::array<float, 4>, 2> expected_boxes{{
        {{8.0F, 0.0F, 24.0F, 14.4F}},
        {{24.0F, 8.0F, 40.0F, 20.8F}},
    }};
    const std::array<float, 2> expected_scores{{0.9F, 0.7F}};
    const std::array<int, 2> expected_classes{{0, 1}};
    Tensor tensor = makeTensor(4);
    setCandidate(tensor, 0, 10.0F, 10.0F, 10.0F, 10.0F, 0, 0.9F);
    setCandidate(tensor, 1, 10.0F, 10.0F, 10.0F, 10.0F, 0, 0.8F);
    setCandidate(tensor, 2, 20.0F, 15.0F, 10.0F, 8.0F, 1, 0.7F);
    setCandidate(tensor, 3, 60.0F, 15.0F, 10.0F, 8.0F, 2, 0.6F);

    auto config = options();
    config.conf_threshold = 0.25;
    config.iou_threshold = 0.45;
    config.transform = {0.625, 0, 6, 80, 60};
    const auto detections = run(tensor, config);
    expect(detections.size() == expected_boxes.size(), "vendored NMS reference detection count");
    for (std::size_t index = 0; index < detections.size(); ++index) {
        expect(detections[index].class_id == expected_classes[index], "vendored NMS reference class");
        expectNear(detections[index].score, expected_scores[index], 1e-6, "vendored NMS reference score");
        for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
            expectNear(detections[index].bbox_xyxy[coordinate],
                       expected_boxes[index][coordinate],
                       1e-3,
                       "vendored NMS reference coordinate");
        }
    }
}

void testStrictConfidenceThreshold() {
    Tensor tensor = makeTensor(2);
    setCandidate(tensor, 0, 10.0F, 10.0F, 4.0F, 4.0F, 0, 0.5F);
    setCandidate(tensor, 1, 30.0F, 10.0F, 4.0F, 4.0F, 1, 0.5001F);
    setValue(tensor, 1, 6, 0.7F);
    auto config = options();
    config.conf_threshold = 0.5;
    const auto detections = run(tensor, config);
    expect(detections.size() == 1, "score equal to conf must be rejected");
    expectNear(detections.front().score, 0.7F, 1e-6, "highest class score must be kept");
    expect(detections.front().class_id == 2, "class with the highest score must be selected");
}

void testStrictIouThresholdAndClassAwareNms() {
    Tensor tensor = makeTensor(2);
    setCandidate(tensor, 0, 1.0F, 1.0F, 2.0F, 2.0F, 0, 0.9F);
    setCandidate(tensor, 1, 2.0F, 1.0F, 2.0F, 2.0F, 0, 0.8F);
    auto config = options();
    config.iou_threshold = 1.0F / 3.0F;
    expect(run(tensor, config).size() == 2, "IoU equal to threshold must be kept");
    config.iou_threshold = 0.3;
    expect(run(tensor, config).size() == 1, "IoU above threshold must be suppressed");

    setValue(tensor, 1, 4, 0.0F);
    setValue(tensor, 1, 5, 0.8F);
    const auto different_classes = run(tensor, config);
    expect(different_classes.size() == 2, "overlapping boxes from different classes must both be kept");
}

void testStableScoreTiesAndMaxDet() {
    Tensor tied = makeTensor(2);
    setCandidate(tied, 0, 10.0F, 10.0F, 4.0F, 4.0F, 0, 0.8F);
    setCandidate(tied, 1, 30.0F, 10.0F, 4.0F, 4.0F, 0, 0.8F);
    const auto tied_detections = run(tied, options());
    expect(tied_detections.size() == 2, "disjoint tied boxes must both be kept");
    expectNear(tied_detections[0].bbox_xyxy[0], 8.0, 1e-6, "source index must break score ties");
    expectNear(tied_detections[1].bbox_xyxy[0], 28.0, 1e-6, "stable score tie order");

    Tensor limited = makeTensor(3);
    setCandidate(limited, 0, 10.0F, 10.0F, 4.0F, 4.0F, 0, 0.5F);
    setCandidate(limited, 1, 30.0F, 10.0F, 4.0F, 4.0F, 0, 0.9F);
    setCandidate(limited, 2, 50.0F, 10.0F, 4.0F, 4.0F, 0, 0.7F);
    auto config = options();
    config.max_det = 2;
    const auto detections = run(limited, config);
    expect(detections.size() == 2, "max_det must cap kept boxes");
    expectNear(detections[0].score, 0.9, 1e-6, "max_det result order first");
    expectNear(detections[1].score, 0.7, 1e-6, "max_det result order second");
}

void testThirtyThousandCandidateLimit() {
    constexpr int candidate_count = 30001;
    Tensor tensor = makeTensor(candidate_count);
    for (int candidate = 0; candidate < candidate_count - 1; ++candidate) {
        setCandidate(tensor,
                     candidate,
                     50.0F,
                     50.0F,
                     20.0F,
                     20.0F,
                     0,
                     0.9F - static_cast<float>(candidate) * 1e-6F);
    }
    setCandidate(tensor, candidate_count - 1, 50.0F, 50.0F, 20.0F, 20.0F, 1, 0.1F);
    auto config = options();
    config.conf_threshold = 0.001;
    config.max_det = 40000;
    const auto detections = run(tensor, config);
    expect(detections.size() == 1 && detections.front().class_id == 0,
           "only the top 30000 candidates may enter NMS");
}

void testClipAfterNmsAndNoBackfill() {
    Tensor tensor = makeTensor(5);
    setCandidate(tensor, 0, 10.0F, 10.0F, 40.0F, 40.0F, 0, 0.9F);
    setCandidate(tensor, 1, 64.5F, 20.0F, 31.0F, 20.0F, 1, 0.8F);
    setCandidate(tensor, 2, 70.0F, 20.0F, 20.0F, 20.0F, 2, 0.7F);
    setCandidate(tensor, 3, std::numeric_limits<float>::quiet_NaN(), 20.0F, 10.0F, 10.0F, 3, 0.6F);
    setCandidate(tensor,
                 4,
                 0.0F,
                 20.0F,
                 std::numeric_limits<float>::infinity(),
                 10.0F,
                 4,
                 0.5F);
    auto config = options();
    config.iou_threshold = 0.5;
    config.transform = {2.0, 10, 10, 20, 20};
    const auto detections = run(tensor, config);
    expect(detections.size() == 2, "fully clipped box must be discarded after NMS");
    expectNear(detections[0].bbox_xyxy[0], 0.0, 1e-6, "left clip");
    expectNear(detections[0].bbox_xyxy[2], 10.0, 1e-6, "right coordinate after scale");
    expectNear(detections[1].bbox_xyxy[0], 19.5, 1e-6, "box touching image boundary");
    expectNear(detections[1].bbox_xyxy[2], 20.0, 1e-6, "right image boundary clip");

    Tensor no_backfill = makeTensor(2);
    setCandidate(no_backfill, 0, 25.0F, 5.0F, 5.0F, 5.0F, 0, 0.9F);
    setCandidate(no_backfill, 1, 2.5F, 5.0F, 5.0F, 5.0F, 1, 0.8F);
    auto limited = options();
    limited.max_det = 1;
    limited.transform = {1.0, 0, 0, 10, 10};
    expect(run(no_backfill, limited).empty(), "clipped max_det result must not be backfilled");
}

}  // namespace

int main() {
    try {
        testVendoredUltralyticsReference();
        testStrictConfidenceThreshold();
        testStrictIouThresholdAndClassAwareNms();
        testStableScoreTiesAndMaxDet();
        testThirtyThousandCandidateLimit();
        testClipAfterNmsAndNoBackfill();
        std::cout << "All detector postprocess tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Detector postprocess test failed: " << exception.what() << '\n';
        return 1;
    }
}
