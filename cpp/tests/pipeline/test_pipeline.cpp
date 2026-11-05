#include "mot/pipeline/pipeline.hpp"
#include "mot/tracking/factory.hpp"

#include "support/check.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using mot::test::expect;

class FakeDetector final : public mot::IDetector {
public:
    std::vector<mot::Detection> detect(const cv::Mat& frame) override {
        expect(!frame.empty(), "detector received an empty frame");
        ++calls;
        const double dx = 3.0 * calls;
        return {{{50 + dx, 60, 90 + dx, 120}, 0.9, 1}};
    }
    const std::vector<std::string>& classNames() const override { return names; }

    int calls{0};
    std::vector<std::string> names{"a", "b"};
};

class CountingTracker final : public mot::ITracker {
public:
    std::vector<mot::TrackOutput> update(const std::vector<mot::Detection>& detections,
                                         const cv::Mat&) override {
        ++updates;
        std::vector<mot::TrackOutput> out;
        for (const auto& d : detections) out.emplace_back(1, d.bbox_xyxy, d.score, d.class_id);
        return out;
    }
    void reset() override { ++resets; }
    std::string_view name() const override { return "counting"; }

    int updates{0};
    int resets{0};
};

}

int main() {
    try {
        const cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(10, 20, 30));

        auto detector = std::make_unique<FakeDetector>();
        auto tracker = std::make_unique<CountingTracker>();
        auto* detector_view = detector.get();
        auto* tracker_view = tracker.get();
        mot::Pipeline pipeline(std::move(detector), std::move(tracker));
        expect(pipeline.process(frame).size() == 1, "pipeline forwards detections to tracker");
        pipeline.reset();
        expect(detector_view->calls == 1 && tracker_view->updates == 1 && tracker_view->resets == 1,
               "pipeline calls detector, tracker and reset exactly once");
        expect(pipeline.detector().classNames().size() == 2, "detector accessor");

        for (const auto type : mot::availableTrackers()) {
            pipeline.setTracker(mot::createTracker(mot::defaultTrackerConfig(type)));
            expect(pipeline.tracker().name() == type, "setTracker replaced the tracker");
            std::vector<mot::TrackOutput> tracks;
            for (int i = 0; i < 6; ++i) tracks = pipeline.process(frame);
            expect(tracks.size() == 1, std::string(type) + ": one tracked object");
        }

        bool rejected = false;
        try { mot::Pipeline(nullptr, std::make_unique<CountingTracker>()); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "null detector must be rejected");
        std::cout << "Pipeline tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pipeline test failure: " << error.what() << '\n';
        return 1;
    }
}
