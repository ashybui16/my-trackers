#include "tracking/common/assignment.hpp"
#include "tracking/common/association.hpp"
#include "tracking/common/geometry.hpp"
#include "tracking/common/kalman_filter.hpp"
#include "tracking/botsort/botsort.hpp"
#include "mot/core/types.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

mot::Detection detection(double x = 0.0, double score = 0.9, int class_id = 0) {
    return {{x, 0.0, x + 10.0, 10.0}, score, class_id};
}

void testTypesGeometryAssignment() {
    bool rejected = false;
    try {
        mot::Detection invalid{{0.0, 0.0, 0.0, 1.0}, 0.5, 0};
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "invalid detection was accepted");
    expectNear(mot::iou({0.0, 0.0, 10.0, 10.0}, {0.0, 0.0, 10.0, 10.0}), 1.0, 1e-12, "IoU");
    expectNear(mot::iou({0.0, 0.0, 10.0, 10.0}, {20.0, 20.0, 30.0, 30.0}), 0.0, 1e-12, "disjoint IoU");

    const auto assignment = mot::linearAssignment({{0.1, 0.9, 0.9}, {0.9, 0.1, 0.9}}, 0.5);
    expect(assignment.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "rectangular assignment");
    expect(assignment.unmatched_rows.empty(), "unexpected unmatched row");
    expect(assignment.unmatched_cols == std::vector<int>{2}, "unexpected unmatched column");
    const auto tie = mot::linearAssignment({{0.0, 0.0}, {0.0, 0.0}}, 0.1);
    expect(tie.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "assignment tie behavior");
    const auto limited = mot::linearAssignment({{0.9, 0.2}, {0.2, 0.9}}, 0.5);
    expect(limited.matches == std::vector<std::pair<int, int>>{{0, 1}, {1, 0}}, "cost-limit assignment");
    const auto empty_rows = mot::linearAssignment({{}, {}}, 0.5);
    expect(empty_rows.unmatched_rows == std::vector<int>{0, 1} && empty_rows.unmatched_cols.empty(), "empty assignment");
    const auto fused = mot::fuseScore({{0.2, 0.4}}, {0.5, 0.75});
    expectNear(fused[0][0], 0.6, 1e-12, "score fusion first value");
    expectNear(fused[0][1], 0.55, 1e-12, "score fusion second value");

    mot::BoTSORTConfig config;
    config.gmc_method = "none";
    config.validate();
    bool bad_config = false;
    config.track_low_thresh = config.track_high_thresh;
    try {
        config.validate();
    } catch (const std::invalid_argument&) {
        bad_config = true;
    }
    expect(bad_config, "invalid tracker config was accepted");
}

void testIouMatchesUltralytics() {
    // Golden values from matching.iou_distance/bbox_ioa, vendored Ultralytics 8.4.157.
    // Covers epsilon, float32, inverted boxes and zero union after GMC.
    struct Case {
        mot::BBox first;
        mot::BBox second;
        double overlap;
        double distance;
    };
    const std::array<Case, 6> cases{{
        {{0, 0, 1, 1}, {0, 0, 1, 1}, 0.9999998807907104, 1.1920928955078125e-7},
        {{0, 0, 0.001, 0.001}, {0, 0, 0.001, 0.001}, 0.9090909361839294, 0.09090906381607056},
        {{0, 0, 2, 2}, {1, 0, 3, 2}, 0.3333333432674408, 0.6666666269302368},
        {{50, 25, 30, 35}, {30, 25, 50, 35}, 0.0, 1.0},
        {{0, 0, 0, 0}, {0, 0, 0, 0}, 0.0, 1.0},
        {{10000000.1, 0, 10000001.1, 1}, {10000000.4, 0, 10000001.4, 1},
         0.9999998807907104, 1.1920928955078125e-7},
    }};
    for (const auto& item : cases) {
        expectNear(mot::iou(item.first, item.second), item.overlap, 1e-12, "Ultralytics IoU");
        const auto distance = mot::iouDistance({item.first}, {item.second});
        expectNear(distance[0][0], item.distance, 1e-12, "Ultralytics IoU distance");
    }
    bool rejected = false;
    try {
        mot::iou({0, 0, std::numeric_limits<double>::infinity(), 1}, {0, 0, 1, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "non-finite IoU input was accepted");
}

void testKalman() {
    mot::KalmanFilterXYWH kalman;
    const auto initial = kalman.initiate({10.0, 20.0, 4.0, 8.0});
    expectNear(initial.mean.at<double>(0, 0), 10.0, 1e-12, "Kalman initial center");
    expectNear(initial.covariance.at<double>(0, 0), 0.16, 1e-12, "Kalman initial covariance");
    const auto predicted = kalman.predict(initial);
    expectNear(predicted.mean.at<double>(0, 0), 10.0, 1e-12, "Kalman prediction");
    const auto updated = kalman.update(predicted, {10.5, 20.2, 4.1, 7.9});
    expectNear(updated.covariance.at<double>(0, 1), updated.covariance.at<double>(1, 0), 1e-12, "Kalman covariance symmetry");
    expect(updated.covariance.at<double>(0, 0) > 0.0, "Kalman covariance must remain positive");
}

void testLifecycle() {
    mot::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 0;
    mot::BoTSORT tracker(config);
    expect(tracker.update({detection()})[0].track_id == 1, "first activation");
    expect(tracker.update({}).empty(), "empty frame output");
    expect(tracker.removedTracks().empty(), "track expired too early");
    expect(tracker.update({}).empty(), "second empty frame output");
    expect(!tracker.removedTracks().empty() && tracker.removedTracks().front()->trackId() == 1, "track expiry");
    expect(tracker.lostTracks().size() == 1 &&
               tracker.lostTracks().front()->state() == mot::TrackState::Removed,
           "expired track remains in the lost pool for one frame");
    const auto revived = tracker.update({detection(0.0, 0.9, 2)});
    expect(revived.size() == 1 && revived.front().track_id == 1 && revived.front().class_id == 2,
           "expired track can recover its ID on the next frame");
    tracker.update({});
    tracker.update({});
    tracker.update({});
    expect(tracker.lostTracks().empty(), "removed history clears the lost pool on the next frame");
    expect(tracker.update({detection(0.0, 0.9, 2)}).empty(), "new track should be unconfirmed");
    const auto confirmed = tracker.update({detection(0.0, 0.9, 2)});
    expect(confirmed.size() == 1 && confirmed.front().track_id == 2, "unconfirmed activation");
    tracker.reset();
    expect(tracker.frameId() == 0, "reset frame counter");
    expect(tracker.update({detection()})[0].track_id == 1, "reset ID counter");

    mot::BoTSORT reactivation(config);
    const int first_id = reactivation.update({detection(0.0, 0.9, 1)})[0].track_id;
    reactivation.update({});
    const auto recovered = reactivation.update({detection(0.0, 0.65, 4)});
    expect(recovered.size() == 1 && recovered.front().track_id == first_id, "reactivation ID");
    expectNear(recovered.front().score, 0.65, 1e-12, "reactivation score");
    expect(recovered.front().class_id == 4, "reactivation class");
}

void testGmcRequirement() {
    mot::BoTSORT tracker;
    bool rejected = false;
    try {
        tracker.update({detection()});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "sparse GMC should require a frame");
}

void testSparseGmcTranslation() {
    cv::Mat frame = cv::Mat::zeros(160, 200, CV_8UC1);
    for (int x = 15; x < 190; x += 20) {
        cv::circle(frame, cv::Point(x, 40 + (x % 50)), 4, cv::Scalar(255), -1);
    }
    cv::Mat shifted;
    const cv::Mat translation = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 4.0, 0.0, 1.0, 3.0);
    cv::warpAffine(frame, shifted, translation, frame.size());
    mot::GMC gmc("sparseOptFlow", 1);
    const cv::Mat first = gmc.apply(frame);
    expectNear(first.at<double>(0, 2), 0.0, 1e-12, "GMC first-frame translation");
    const cv::Mat warp = gmc.apply(shifted);
    expectNear(warp.at<double>(0, 2), 4.0, 1.0, "GMC x translation");
    expectNear(warp.at<double>(1, 2), 3.0, 1.0, "GMC y translation");
    gmc.reset();
    expect(!gmc.initializedFirstFrame(), "GMC reset");

    cv::Mat blank = cv::Mat::zeros(32, 32, CV_8UC1);
    mot::GMC no_features("sparseOptFlow", 1);
    no_features.apply(blank);
    const cv::Mat identity = no_features.apply(blank);
    expectNear(identity.at<double>(0, 0), 1.0, 1e-12, "GMC identity after missing features");
    expect(no_features.initializedFirstFrame(), "GMC reference should advance after missing features");
}

void testWarpMatchesUltralytics() {
    // Reference values from Ultralytics multi_gmc (sizes follow abs(R)), 10-degree rotation and (5, 7) shift.
    mot::KalmanFilterXYWH kalman;
    auto state = kalman.initiate({40.0, 30.0, 20.0, 10.0});
    for (int index = 4; index < 8; ++index) {
        state.mean.at<double>(index, 0) = std::array<double, 4>{3.0, 4.0, 2.0, 1.0}[index - 4];
    }
    const double angle = std::acos(-1.0) / 18.0;
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << std::cos(angle), -std::sin(angle), 5.0,
                          std::sin(angle), std::cos(angle), 7.0);
    mot::warpXywhState(state.mean, state.covariance, warp);
    const std::array<double, 8> expected_mean = {
        39.18286479048041, 43.49015969704345, 21.432636836913463, 13.321041083460686,
        2.2598305483689027, 4.460175545049623, 2.1432636836913463, 1.3321041083460687};
    const std::array<double, 8> expected_diagonal = {
        3.9095389311788624, 1.0904610688211374, 3.9095389311788624, 1.0904610688211374,
        1.527163644991743, 0.42596135500825677, 1.527163644991743, 0.42596135500825677};
    for (int index = 0; index < 8; ++index) {
        expectNear(state.mean.at<double>(index, 0), expected_mean[index], 1e-12, "GMC mean parity");
        for (int column = 0; column < 8; ++column) {
            // Off-diagonal term inside each (x, y), (w, h), (vx, vy), (vw, vh) block.
            constexpr std::array<double, 4> block_offdiagonal{
                0.513030214988503, 0.8550503583141716, 0.200402427729884, 0.33400404621647334};
            const double expected = index == column ? expected_diagonal[index]
                : index / 2 == column / 2 ? block_offdiagonal[static_cast<std::size_t>(index / 2)]
                                         : 0.0;
            expectNear(state.covariance.at<double>(index, column), expected, 1e-12, "GMC covariance parity");
        }
    }

    mot::Track track({{30.0, 25.0, 50.0, 35.0}, 0.9, 0});
    track.activate(kalman, 1, 1);
    const cv::Mat large_rotation = (cv::Mat_<double>(2, 3) << std::cos(2.2), -std::sin(2.2), 5.0,
                                    std::sin(2.2), std::cos(2.2), 7.0);
    track.applyWarp(large_rotation);
    // A 2.2 rad rotation: the size becomes the enclosing box, never negative.
    expectNear(track.mean().at<double>(0, 0), -42.79493680480154, 1e-12, "GMC rotated center");
    expectNear(track.mean().at<double>(1, 0), 21.68482263512323, 1e-12, "GMC rotated center y");
    expectNear(track.mean().at<double>(2, 0), 19.854986383302816, 1e-12, "GMC enclosing width");
    expectNear(track.mean().at<double>(3, 0), 22.05493924894526, 1e-12, "GMC enclosing height");
    const mot::BBox expected_box{
        -52.722429996452945, 10.6573530106506, -32.86744361315013, 32.712292259595856};
    const auto box = track.bbox();
    for (std::size_t index = 0; index < box.size(); ++index) {
        expectNear(box[index], expected_box[index], 1e-12, "GMC bbox parity");
    }
    expect(box[0] < box[2] && box[1] < box[3], "GMC must not invert box corners");
    const mot::TrackOutput output(1, box, 0.9, 0);
    expect(output.bbox_xyxy == box, "track output keeps the GMC box");
    bool rejected = false;
    try {
        mot::TrackOutput invalid(1, {0, 0, std::numeric_limits<double>::infinity(), 1}, 0.9, 0);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "non-finite track output was accepted");

    track.predict();
    expectNear(track.mean().at<double>(2, 0), 19.854986383302816, 1e-12,
               "predict keeps the GMC width");
    track.update(mot::Track({{39.5, 29.5, 40.5, 30.5}, 0.8, 2}), 2);
    // Golden values from Ultralytics multi_gmc + KalmanFilterXYWH predict/update (no confidence).
    const std::array<double, 8> updated_mean{
        20.004283535053432, 20.978919039334453, 3.5836856844065323, 3.2988925237897746,
        12.023456144637185, -2.7322990327118863, -3.844835678227458, -4.622796123979132};
    const std::array<double, 8> updated_diagonal{
        0.7545623754118265, 0.9579741599185647, 0.6757772647368383, 0.8699499850531582,
        0.6707765223027614, 0.9611918406110518, 0.6459106244757863, 0.9334099396940575};
    for (int index = 0; index < 8; ++index) {
        expectNear(track.mean().at<double>(index, 0), updated_mean[index], 1e-10,
                   "Kalman update after GMC mean parity");
        expectNear(track.covariance().at<double>(index, index), updated_diagonal[index], 1e-10,
                   "Kalman update after GMC covariance parity");
    }
}

void testLargeRotationGmcLifecycle() {
    mot::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 1;
    mot::BoTSORT tracker(config);
    const mot::Detection observed({30, 25, 50, 35}, 0.9, 0);
    expect(tracker.update({observed}).front().track_id == 1, "large-rotation GMC initial ID");
    // 2.2 rad rotation about the box center: abs(R) turns the 20x10 box into its ~19.9x22.1 enclosing box.
    const double cosine = std::cos(2.2);
    const double sine = std::sin(2.2);
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << cosine, -sine, 40 - 40 * cosine + 30 * sine,
                          sine, cosine, 30 - 40 * sine - 30 * cosine);
    tracker.trackedTracks().front()->applyWarp(warp);
    const auto box = tracker.trackedTracks().front()->bbox();
    expect(box[0] < box[2] && box[1] < box[3], "large-rotation GMC keeps a valid box");
    const auto matched = tracker.update({observed});
    expect(matched.size() == 1 && matched.front().track_id == 1,
           "track keeps its ID after a large-rotation GMC");
}

void testRemovedHistoryLimit() {
    mot::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 0;
    mot::BoTSORT tracker(config);
    for (int index = 0; index < 1010; ++index) {
        tracker.update({detection(20.0 * index)});
        tracker.update({});
        tracker.update({});
        tracker.update({});
        expect(tracker.removedTracks().size() <= 1000, "removed history exceeds 1000 entries");
    }
    expect(tracker.removedTracks().size() == 1000, "removed history should retain 1000 entries");
    expect(tracker.removedTracks().front()->trackId() == 11 &&
               tracker.removedTracks().back()->trackId() == 1010,
           "removed history must retain the newest entries in order");
    expect(tracker.lostTracks().empty(), "removed tracks must leave the lost pool");
    tracker.reset();
    expect(tracker.removedTracks().empty(), "reset must clear removed history");
}

void testDuplicateRemoval() {
    mot::KalmanFilterXYWH kalman;
    auto active = std::make_shared<mot::Track>(detection());
    auto stale = std::make_shared<mot::Track>(detection());
    active->activate(kalman, 1, 1);
    stale->activate(kalman, 1, 2);
    active->update(*std::make_shared<mot::Track>(detection(0.0, 0.9, 0)), 3);
    stale->markLost();
    const auto result = mot::removeDuplicateTracks({active}, {stale});
    expect(result.tracked.size() == 1 && result.lost.empty() && result.removed.size() == 1,
           "duplicate removal should keep the longer-lived track");
}

void testClassEma() {
    mot::KalmanFilterXYWH kalman;
    const auto run = [&](double alpha) {
        auto track = std::make_shared<mot::Track>(detection(0.0, 0.9, 4), -1, alpha);
        track->activate(kalman, 1, 1);
        for (int frame = 2; frame <= 5; ++frame) {
            track->update(mot::Track(detection(0.0, 0.9, 4), -1, alpha), frame);
        }
        track->update(mot::Track(detection(0.0, 0.9, 3), -1, alpha), 6);
        return track;
    };
    expect(run(0.9)->classId() == 4, "EMA class should resist a single flip");
    expect(run(0.0)->classId() == 3, "alpha=0 should follow the latest detection");

    const mot::Detection first{{0.0, 0.0, 10.0, 10.0}, 0.6, 1, {0.2F, 0.6F, 0.1F}};
    const mot::Detection second{{0.0, 0.0, 10.0, 10.0}, 0.8, 0, {0.8F, 0.1F, 0.0F}};
    mot::Track track(first, -1, 0.5);
    track.activate(kalman, 1, 1);
    track.update(mot::Track(second, -1, 0.5), 2);
    const auto& vector = track.classVector();
    expect(vector.size() == 3, "class vector size");
    expectNear(vector[0], 0.5, 1e-6, "class vector[0]");
    expectNear(vector[1], 0.35, 1e-6, "class vector[1]");
    expectNear(vector[2], 0.05, 1e-6, "class vector[2]");
    expect(track.classId() == 0, "argmax of blended class vector");

    mot::BoTSORTConfig config;
    config.gmc_method = "none";
    for (double bad : {1.0, -0.1}) {
        config.class_ema_alpha = bad;
        bool rejected = false;
        try {
            config.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid class_ema_alpha was accepted");
    }

    config.class_ema_alpha = 0.9;
    mot::BoTSORT tracker(config);
    std::vector<mot::TrackOutput> outputs;
    for (int frame = 0; frame < 6; ++frame) {
        outputs = tracker.update({detection(0.0, 0.9, frame < 5 ? 4 : 3)});
    }
    expect(outputs.size() == 1 && outputs.front().class_id == 4, "tracker should report the smoothed class");
}

}

int main() {
    try {
        testTypesGeometryAssignment();
        testIouMatchesUltralytics();
        testKalman();
        testLifecycle();
        testGmcRequirement();
        testSparseGmcTranslation();
        testWarpMatchesUltralytics();
        testLargeRotationGmcLifecycle();
        testRemovedHistoryLimit();
        testDuplicateRemoval();
        testClassEma();
        std::cout << "BoT-SORT tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "BoT-SORT test failure: " << exception.what() << '\n';
        return 1;
    }
}
