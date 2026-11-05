// Behavior every tracker returned by createTracker() must satisfy.
// New trackers are covered automatically once they are added to TrackerConfig.
#include "mot/tracking/factory.hpp"

#include "support/check.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using mot::test::expect;

std::vector<mot::Detection> frameDetections(int frame) {
    // Hai đối tượng tách rời, di chuyển đều sang phải.
    const double dx = 2.0 * frame;
    return {{{100 + dx, 100, 140 + dx, 180}, 0.95, 0},
            {{400 + dx, 300, 460 + dx, 340}, 0.90, 3}};
}

std::vector<std::vector<mot::TrackOutput>> runSequence(mot::ITracker& tracker, int frames,
                                                       const cv::Mat& image) {
    std::vector<std::vector<mot::TrackOutput>> outputs;
    for (int frame = 0; frame < frames; ++frame) {
        outputs.push_back(tracker.update(frameDetections(frame), image));
    }
    return outputs;
}

void checkTracker(std::string_view type) {
    const std::string name(type);
    const auto config = mot::defaultTrackerConfig(type);
    expect(mot::trackerType(config) == type, name + ": trackerType round-trip");
    auto tracker = mot::createTracker(config);
    expect(tracker != nullptr, name + ": createTracker returned null");
    expect(tracker->name() == type, name + ": name() matches type");

    const cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(40, 80, 120));
    expect(tracker->update({}, image).empty(), name + ": empty input gives no tracks");
    tracker->reset();

    const auto first = runSequence(*tracker, 12, image);
    const auto& last = first.back();
    expect(last.size() == 2, name + ": two steady objects are tracked");

    // ID của mỗi đối tượng phải ổn định qua các frame sau khi đã xác nhận.
    std::map<int, int> id_by_class;
    for (const auto& frame : first) {
        for (const auto& track : frame) {
            const auto [it, inserted] = id_by_class.emplace(track.class_id, track.track_id);
            expect(inserted || it->second == track.track_id, name + ": track IDs are stable");
            expect(track.track_id > 0, name + ": track IDs are positive");
        }
    }
    expect(last[0].track_id != last[1].track_id, name + ": distinct objects get distinct IDs");

    // reset() phải cho kết quả giống hệt một tracker mới.
    tracker->reset();
    const auto again = runSequence(*tracker, 12, image);
    auto fresh = mot::createTracker(config);
    const auto expected = runSequence(*fresh, 12, image);
    expect(again.size() == expected.size(), name + ": reset sequence length");
    for (std::size_t f = 0; f < again.size(); ++f) {
        expect(again[f].size() == expected[f].size(), name + ": reset matches fresh tracker");
        for (std::size_t i = 0; i < again[f].size(); ++i) {
            expect(again[f][i].track_id == expected[f][i].track_id &&
                   again[f][i].bbox_xyxy == expected[f][i].bbox_xyxy,
                   name + ": reset matches fresh tracker");
        }
    }
    std::cout << "  " << name << ": ok\n";
}

}  // namespace

int main() {
    try {
        const auto trackers = mot::availableTrackers();
        expect(!trackers.empty(), "no trackers registered");
        bool unknown_rejected = false;
        try { mot::defaultTrackerConfig("does-not-exist"); }
        catch (const std::invalid_argument&) { unknown_rejected = true; }
        expect(unknown_rejected, "unknown tracker type must be rejected");
        for (const auto type : trackers) checkTracker(type);
        std::cout << "Tracker contract tests passed for " << trackers.size() << " tracker(s)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tracker contract failure: " << error.what() << '\n';
        return 1;
    }
}
