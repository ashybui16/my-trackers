#include "mot/tracking/factory.hpp"
#include "mot/tracking/keypoint_gmc.hpp"

#include "support/check.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace {

using mot::test::expect;

std::vector<mot::Detection> frameDetections(int frame) {
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

    std::map<int, int> id_by_class;
    for (const auto& frame : first) {
        for (const auto& track : frame) {
            const auto [it, inserted] = id_by_class.emplace(track.class_id, track.track_id);
            expect(inserted || it->second == track.track_id, name + ": track IDs are stable");
            expect(track.track_id > 0, name + ": track IDs are positive");
        }
    }
    expect(last[0].track_id != last[1].track_id, name + ": distinct objects get distinct IDs");
    for (std::size_t frame = 0; frame < first.size(); ++frame) {
        const auto detections = frameDetections(static_cast<int>(frame));
        for (const auto& track : first[frame]) {
            const int index = track.detection_index;
            expect(index >= 0 && index < static_cast<int>(detections.size()),
                   name + ": detection_index points into this frame's input");
            expect(detections[static_cast<std::size_t>(index)].class_id == track.class_id,
                   name + ": detection_index maps the track to its own detection");
        }
    }

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

// Fake keypoint model: every frame "moves" by +3 px in x; shared state counts calls.
class ShiftModel final : public mot::IKeypointGmcModel {
public:
    explicit ShiftModel(int* calls) : calls_(calls) {}
    cv::Size inputSize() const override { return {64, 32}; }
    bool extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                         std::vector<cv::Point2f>& current) override {
        expect(image.type() == CV_32FC1 && image.size() == inputSize(), "fake model gets the model input");
        ++*calls_;
        previous.clear();
        current.clear();
        if (!has_previous_) {
            has_previous_ = true;
            return false;
        }
        for (int index = 0; index < 20; ++index) {
            const cv::Point2f point(2.0F + 3.0F * static_cast<float>(index % 5), 2.0F + 6.0F * static_cast<float>(index / 5));
            previous.push_back(point);
            current.push_back(point + cv::Point2f(0.15F, 0.0F));   // 0.15 model px = 3 frame px at 1280 wide
        }
        return true;
    }
    void reset() override { has_previous_ = false; }

private:
    int* calls_;
    bool has_previous_{false};
};

void checkDpointGmc(std::string_view type) {
    const std::string name(type);
    auto config = mot::defaultTrackerConfig(type);
    std::visit([](auto& value) { value.gmc_method = "dpoint"; }, config);
    bool rejected = false;
    try { mot::createTracker(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, name + ": dpoint GMC without a model is rejected");

    int calls = 0;
    auto tracker = mot::createTracker(config, std::make_unique<ShiftModel>(&calls));
    const cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(40, 80, 120));
    rejected = false;
    try { tracker->update({}); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, name + ": dpoint GMC needs a frame");
    const auto outputs = runSequence(*tracker, 12, image);
    expect(calls == 12, name + ": dpoint model runs once per frame");
    expect(outputs.back().size() == 2, name + ": dpoint GMC keeps tracking");
    std::cout << "  " << name << " (dpoint GMC): ok\n";
}

}

int main() {
    try {
        const auto trackers = mot::availableTrackers();
        expect(!trackers.empty(), "no trackers registered");
        bool unknown_rejected = false;
        try { mot::defaultTrackerConfig("does-not-exist"); }
        catch (const std::invalid_argument&) { unknown_rejected = true; }
        expect(unknown_rejected, "unknown tracker type must be rejected");
        for (const auto type : trackers) checkTracker(type);
        for (const auto type : trackers) checkDpointGmc(type);
        std::cout << "Tracker contract tests passed for " << trackers.size() << " tracker(s)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tracker contract failure: " << error.what() << '\n';
        return 1;
    }
}
