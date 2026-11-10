#include "tracking/tracktrack/tracktrack.hpp"
#include "tracking/tracktrack/tracktrack_detail.hpp"

#include "support/check.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mot::test::expect;

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

mot::TTTrackPtr det(double x1, double y1, double x2, double y2, double score, int min_track_len = 3) {
    return std::make_shared<mot::TTTrack>(mot::Detection{{x1, y1, x2, y2}, score, 0}, 0, min_track_len);
}

mot::TrackTrackConfig noGmc() {
    mot::TrackTrackConfig config;
    config.gmc_method = "none";
    return config;
}

void testIterativeAssociate() {
    mot::tracktrack::CostMatrix cost(2, 2, 1.0F);
    cost.at(0, 0) = 0.10F;
    cost.at(0, 1) = 0.60F;
    cost.at(1, 0) = 0.60F;
    cost.at(1, 1) = 0.68F;
    // Round 1 matches (0, 0) under 0.7; the gate then drops to 0.65, so (1, 1) at 0.68 is rejected.
    const auto result = mot::tracktrack::iterativeAssociate(cost, 0.7, 0.05);
    expect(result.matches == std::vector<std::pair<int, int>>{{0, 0}}, "iterative: shrinking gate");
    expect(result.unmatched_tracks == std::vector<int>{1}, "iterative: unmatched track");
    expect(result.unmatched_dets == std::vector<int>{1}, "iterative: unmatched det");

    const auto relaxed = mot::tracktrack::iterativeAssociate(cost, 0.7, 0.0);
    expect(relaxed.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "iterative: second round");

    // Not mutual nearest: track 1's nearest det (0) prefers track 0, so only one match per round.
    mot::tracktrack::CostMatrix contested(2, 1, 0.2F);
    contested.at(0, 0) = 0.1F;
    const auto single = mot::tracktrack::iterativeAssociate(contested, 0.7, 0.05);
    expect(single.matches == std::vector<std::pair<int, int>>{{0, 0}}, "iterative: mutual nearest");
    expect(mot::tracktrack::iterativeAssociate(mot::tracktrack::CostMatrix(0, 3, 1.0F), 0.7, 0.05)
               .unmatched_dets.size() == 3, "iterative: empty rows");
}

void testTrackAwareNms() {
    const std::vector<mot::TTTrackPtr> none;
    const auto a = det(0, 0, 100, 100, 0.90);
    const auto b = det(5, 5, 105, 105, 0.95);    // Overlaps a; stronger.
    const auto c = det(300, 0, 400, 100, 0.65);  // Below new_track_thresh.
    const auto d = det(600, 0, 700, 100, 0.80);
    const auto allow = mot::tracktrack::trackAwareNms(none, {a, b, c, d}, 0.55, 0.7);
    expect(allow == std::vector<bool>{false, true, false, true}, "TAI: suppression by stronger detection");

    auto existing = det(600, 0, 700, 100, 0.9);
    mot::KalmanFilterXYWH filter;
    existing->activate(filter, 1, 1);
    const auto blocked = mot::tracktrack::trackAwareNms({existing}, {d}, 0.55, 0.7);
    expect(blocked == std::vector<bool>{false}, "TAI: suppression by existing track");
    expect(mot::tracktrack::trackAwareNms(none, {d}, 0.55, 0.7) == std::vector<bool>{true}, "TAI: single detection");
}

void testCostMatrix() {
    mot::KalmanFilterXYWH filter;
    auto track = det(0, 0, 100, 100, 0.9);
    track->activate(filter, 1, 1);
    const auto same = det(0, 0, 100, 100, 0.9);
    const auto shifted = det(0, 50, 100, 150, 0.9);  // IoU 1/3, HIoU 1/3.
    const auto far = det(500, 500, 600, 600, 0.9);
    const auto cost = mot::tracktrack::costMatrix({track}, {same, shifted, far}, 1, 0.1, 0.0);
    expectNear(cost.at(0, 0), 0.0, 1e-6, "cost: identical box");
    expectNear(cost.at(0, 1), 1.0 - 1.0 / 9.0, 1e-5, "cost: HMIoU");
    expectNear(cost.at(0, 2), 1.0, 0.0, "cost: unsupported pair");
    // Confidence term: projected score 0.9 vs detection 0.5.
    const auto weak = det(0, 0, 100, 100, 0.5);
    expectNear(mot::tracktrack::costMatrix({track}, {weak}, 1, 0.1, 0.0).at(0, 0), 0.04, 1e-6, "cost: confidence");
}

void testVelocityAndHistory() {
    mot::KalmanFilterXYWH filter;
    auto track = det(0, 0, 10, 10, 0.9);
    track->activate(filter, 1, 1);
    for (int frame = 2; frame <= 5; ++frame) {
        const double x = 2.0 * (frame - 1);
        track->predict();
        track->update(*det(x, 0, x + 10, 10, 0.9), frame);
    }
    for (const auto& corner : track->velocity()) {
        // Unit motion per dt, weighted 1/dt and averaged: (1 + 1/2 + 1/3) / 3.
        expectNear(corner[0], (1.0 + 1.0 / 2.0 + 1.0 / 3.0) / 3.0, 1e-4, "velocity x");
        expectNear(corner[1], 0.0, 1e-6, "velocity y");
    }
    expectNear(track->historyBox(5, 3)[0], 2.0, 1e-12, "history box at frame - dt");
    expectNear(track->historyBox(5, 30)[0], 8.0, 1e-12, "history falls back to newest box");
}

void testLifecycle() {
    // Mirrors Ultralytics test_tracktrack_new_lifecycle.
    auto config = noGmc();
    config.min_track_len = 4;
    const auto box = [](double cx) { return std::vector<mot::Detection>{{{cx - 50, 50, cx + 50, 150}, 0.9, 0}}; };
    {
        mot::TrackTrack tracker(config);
        tracker.update({});  // Avoid frame-1 auto-activation.
        std::vector<std::size_t> sizes;
        for (double cx : {100.0, 135.0, 170.0, 205.0}) sizes.push_back(tracker.update(box(cx)).size());
        expect(sizes == std::vector<std::size_t>{0, 0, 0, 1}, "lifecycle: confirmed after min_track_len hits");
    }
    for (int min_track_len : {0, 1}) {
        config.min_track_len = min_track_len;
        mot::TrackTrack tracker(config);
        tracker.update({});
        expect(tracker.update(box(100)).size() == 1, "lifecycle: min_track_len <= 1 confirms at once");
    }
    config.min_track_len = 4;
    mot::TrackTrack tracker(config);
    for (double cx : {100.0, 135.0, 170.0, 205.0}) {
        expect(tracker.update(box(cx)).size() == 1, "lifecycle: frame-1 track is visible");
        if (tracker.frameId() == 2) {
            expect(tracker.trackedTracks()[0]->state() != mot::TrackState::Tracked, "lifecycle: frame-1 track still New");
        }
    }
    expect(tracker.trackedTracks()[0]->state() == mot::TrackState::Tracked, "lifecycle: frame-1 track confirmed");

    // Unmatched unconfirmed tracks are removed; lost tracks expire after track_buffer.
    mot::TrackTrack expiring([] { auto c = noGmc(); c.track_buffer = 2; c.min_track_len = 1; return c; }());
    expiring.update(box(100));
    expect(expiring.trackedTracks().size() == 1, "buffer: track started");
    expiring.update({});
    expect(expiring.lostTracks().size() == 1, "buffer: track lost");
    expiring.update({});  // Frame 3: 3 - 1 <= track_buffer.
    expect(expiring.lostTracks().size() == 1, "buffer: kept within track_buffer");
    expiring.update({});  // Frame 4: 4 - 1 > track_buffer.
    // As in Ultralytics, the lost pool is pruned with the previous frames' removals, so it leaves one frame later.
    expect(expiring.lostTracks().size() == 1 && expiring.lostTracks()[0]->state() == mot::TrackState::Removed,
           "buffer: marked removed after track_buffer");
    expiring.update({});
    expect(expiring.lostTracks().empty(), "buffer: pruned from the lost pool");
}

void testRecoveredDetections() {
    // A recovered detection keeps a track alive (with penalty_q) when the tight NMS dropped its box.
    auto config = noGmc();
    config.min_track_len = 1;
    mot::TrackTrack tracker(config);
    const mot::Detection steady{{100, 100, 150, 200}, 0.9, 0};
    tracker.update({steady});
    const auto kept = tracker.updateWithRecovered({}, {{{101, 100, 151, 200}, 0.9, 0}});
    expect(kept.size() == 1 && kept[0].track_id == 1, "recovered detection continues the track");
    // Below track_high_thresh, a recovered detection is ignored.
    const auto ignored = tracker.updateWithRecovered({}, {{{102, 100, 152, 200}, 0.5, 0}});
    expect(ignored.empty(), "weak recovered detection is ignored");
    expect(tracker.usesRecoveredDetections(), "TrackTrack requests recovered detections");
}

void testClassEma() {
    auto config = noGmc();
    config.min_track_len = 1;
    const auto run = [&](double alpha) {
        config.class_ema_alpha = alpha;
        mot::TrackTrack tracker(config);
        std::vector<mot::TrackOutput> outputs;
        for (int frame = 0; frame < 6; ++frame) {
            outputs = tracker.update({{{100, 100, 150, 200}, 0.9, frame < 5 ? 4 : 3}});
        }
        expect(outputs.size() == 1 && outputs[0].track_id == 1, "class EMA: one continuous track");
        return outputs[0].class_id;
    };
    expect(run(0.9) == 4, "class EMA: resists a single flip");
    expect(run(0.0) == 3, "class EMA: alpha=0 follows the latest detection");

    const mot::Detection first{{0.0, 0.0, 10.0, 10.0}, 0.6, 1, {0.2F, 0.6F, 0.1F}};
    const mot::Detection second{{0.0, 0.0, 10.0, 10.0}, 0.8, 0, {0.8F, 0.1F, 0.0F}};
    mot::KalmanFilterXYWH kalman;
    mot::TTTrack track(first, -1, 1, 0.5);
    track.activate(kalman, 1, 1);
    track.update(mot::TTTrack(second, -1, 1, 0.5), 2);
    expect(track.classVector().size() == 3, "class EMA: vector size");
    expectNear(track.classVector()[0], 0.5, 1e-6, "class EMA: vector[0]");
    expectNear(track.classVector()[1], 0.35, 1e-6, "class EMA: vector[1]");
    expect(track.classId() == 0, "class EMA: argmax of blended vector");

    for (double bad : {1.0, -0.1}) {
        config.class_ema_alpha = bad;
        bool rejected = false;
        try {
            config.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "class EMA: invalid alpha accepted");
    }
}

std::vector<double> reals(const cv::FileNode& node) {
    std::vector<double> values;
    for (const auto& item : node) values.push_back(static_cast<double>(item));
    return values;
}

std::vector<mot::Detection> detections(const std::vector<double>& flat) {
    std::vector<mot::Detection> result;
    for (std::size_t i = 0; i + 6 <= flat.size(); i += 6) {
        result.emplace_back(mot::BBox{flat[i], flat[i + 1], flat[i + 2], flat[i + 3]}, flat[i + 4],
                            static_cast<int>(flat[i + 5]));
    }
    return result;
}

// Replays sequences recorded from Ultralytics TRACKTRACK (gmc_method=none) and compares every output.
void testUltralyticsParity(const std::string& fixture) {
    cv::FileStorage storage(fixture, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    expect(storage.isOpened(), "cannot open fixture " + fixture);
    int run_index = 0;
    std::size_t compared = 0;
    for (const auto& run : storage["runs"]) {
        const auto cfg = run["config"];
        auto config = noGmc();
        config.track_high_thresh = static_cast<double>(cfg["track_high_thresh"]);
        config.track_low_thresh = static_cast<double>(cfg["track_low_thresh"]);
        config.new_track_thresh = static_cast<double>(cfg["new_track_thresh"]);
        config.track_buffer = static_cast<int>(cfg["track_buffer"]);
        config.match_thresh = static_cast<double>(cfg["match_thresh"]);
        config.lost_match_thr = static_cast<double>(cfg["lost_match_thr"]);
        config.conf_weight = static_cast<double>(cfg["conf_weight"]);
        config.angle_weight = static_cast<double>(cfg["angle_weight"]);
        config.penalty_p = static_cast<double>(cfg["penalty_p"]);
        config.penalty_q = static_cast<double>(cfg["penalty_q"]);
        config.reduce_step = static_cast<double>(cfg["reduce_step"]);
        config.tai_thr = static_cast<double>(cfg["tai_thr"]);
        config.min_track_len = static_cast<int>(cfg["min_track_len"]);
        mot::TrackTrack tracker(config);
        int frame_index = 0;
        for (const auto& frame : run["frames"]) {
            const auto outputs = tracker.updateWithRecovered(detections(reals(frame["dets"])),
                                                             detections(reals(frame["recovered"])));
            const auto expected = reals(frame["expected"]);
            const std::string where = "parity run " + std::to_string(run_index) + " frame " + std::to_string(frame_index);
            expect(outputs.size() * 7 == expected.size(),
                   where + ": expected " + std::to_string(expected.size() / 7) + " tracks, got " +
                   std::to_string(outputs.size()));
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                const double* row = &expected[i * 7];
                expect(outputs[i].track_id == static_cast<int>(row[0]), where + ": track id");
                for (std::size_t k = 0; k < 4; ++k) {
                    // Ultralytics returns float32 results, so expected boxes carry float32 rounding.
                    expectNear(outputs[i].bbox_xyxy[k], row[1 + k], 1e-3, where + ": bbox");
                }
                expectNear(outputs[i].score, row[5], 1e-6, where + ": score");
                expect(outputs[i].class_id == static_cast<int>(row[6]), where + ": class");
                ++compared;
            }
            ++frame_index;
        }
        ++run_index;
    }
    expect(run_index > 0 && compared > 0, "fixture has no data");
    std::cout << "  parity: " << compared << " track outputs over " << run_index << " runs match Ultralytics\n";
}

}

int main(int argc, char** argv) {
    try {
        testIterativeAssociate();
        testTrackAwareNms();
        testCostMatrix();
        testVelocityAndHistory();
        testLifecycle();
        testRecoveredDetections();
        testClassEma();
        if (argc > 1) testUltralyticsParity(argv[1]);
        std::cout << "TrackTrack tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tracktrack test failure: " << error.what() << '\n';
        return 1;
    }
}
