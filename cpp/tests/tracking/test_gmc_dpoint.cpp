#include "mot/tracking/keypoint_gmc.hpp"

#include "support/check.hpp"
#include "tracking/common/gmc.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mot::test::expect;

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

void expectIdentity(const cv::Mat& warp, const std::string& message) {
    expect(cv::norm(warp, cv::Mat::eye(2, 3, CV_64F), cv::NORM_INF) == 0.0, message);
}

// Scripted model: `respond` fills the matches for each call after the first.
struct FakeModel final : mot::IKeypointGmcModel {
    struct State {
        int calls{0};
        int resets{0};
        cv::Size last_input;
    };
    using Respond = std::function<void(std::vector<cv::Point2f>&, std::vector<cv::Point2f>&)>;

    FakeModel(State* state, Respond respond) : state_(state), respond_(std::move(respond)) {}
    cv::Size inputSize() const override { return {512, 512}; }
    bool extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                         std::vector<cv::Point2f>& current) override {
        ++state_->calls;
        state_->last_input = image.size();
        expect(image.type() == CV_32FC1, "model input is CV_32FC1");
        double low = 0.0, high = 0.0;
        cv::minMaxLoc(image, &low, &high);
        expect(low >= 0.0 && high <= 1.0, "model input is in [0, 1]");
        previous.clear();
        current.clear();
        if (!has_previous_) {
            has_previous_ = true;
            return false;
        }
        respond_(previous, current);
        return true;
    }
    void reset() override {
        ++state_->resets;
        has_previous_ = false;
    }

private:
    State* state_;
    Respond respond_;
    bool has_previous_{false};
};

// Grid of model-space points moved by `delta` model pixels.
FakeModel::Respond translate(cv::Point2f delta, int count = 25) {
    return [delta, count](std::vector<cv::Point2f>& previous, std::vector<cv::Point2f>& current) {
        for (int index = 0; index < count; ++index) {
            const cv::Point2f point(40.0F + 80.0F * static_cast<float>(index % 5),
                                    40.0F + 80.0F * static_cast<float>(index / 5));
            previous.push_back(point);
            current.push_back(point + delta);
        }
    };
}

}

int main() {
    try {
        bool rejected = false;
        try { mot::GMC gmc("dpoint"); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "dpoint without a model is rejected");
        rejected = false;
        try { mot::GMC gmc("orb"); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "unknown GMC method is rejected");

        // Non-square frame: model pixels scale by 1280/512 in x and 720/512 in y.
        FakeModel::State state;
        mot::GMC gmc("dpoint", 2, std::make_unique<FakeModel>(&state, translate({2.0F, 4.0F})));
        const cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(10, 120, 250));
        expectIdentity(gmc.apply(frame), "first frame is identity");
        expect(gmc.initializedFirstFrame() && state.last_input == cv::Size(512, 512), "frame resized to model input");
        const cv::Mat warp = gmc.apply(frame);
        expectNear(warp.at<double>(0, 0), 1.0, 1e-6, "scale");
        expectNear(warp.at<double>(0, 1), 0.0, 1e-6, "rotation");
        expectNear(warp.at<double>(0, 2), 2.0 * 1280.0 / 512.0, 1e-4, "x translation in frame pixels");
        expectNear(warp.at<double>(1, 2), 4.0 * 720.0 / 512.0, 1e-4, "y translation in frame pixels");
        expect(state.calls == 2, "one model call per frame");

        // Grayscale and float frames are accepted too.
        gmc.apply(cv::Mat(720, 1280, CV_8UC1, cv::Scalar(90)));
        gmc.apply(cv::Mat(720, 1280, CV_32FC1, cv::Scalar(3.5F)));

        gmc.reset();
        expect(state.resets == 1 && !gmc.initializedFirstFrame(), "reset forwards to the model");
        expectIdentity(gmc.apply(frame), "first frame after reset");

        // Too few matches -> identity.
        FakeModel::State few_state;
        mot::GMC few("dpoint", 2, std::make_unique<FakeModel>(&few_state, translate({1.0F, 1.0F}, 4)));
        few.apply(frame);
        expectIdentity(few.apply(frame), "fewer than five matches give identity");

        // Unpaired matches -> identity.
        FakeModel::State unpaired_state;
        mot::GMC unpaired("dpoint", 2, std::make_unique<FakeModel>(&unpaired_state,
            [](std::vector<cv::Point2f>& previous, std::vector<cv::Point2f>& current) {
                translate({1.0F, 1.0F})(previous, current);
                current.pop_back();
            }));
        unpaired.apply(frame);
        expectIdentity(unpaired.apply(frame), "unpaired matches give identity");

        // A failing model resets itself and GMC restarts from the next frame.
        FakeModel::State failing_state;
        bool fail = true;
        mot::GMC failing("dpoint", 2, std::make_unique<FakeModel>(&failing_state,
            [&fail](std::vector<cv::Point2f>& previous, std::vector<cv::Point2f>& current) {
                if (fail) throw std::runtime_error("inference failed");
                translate({1.0F, 0.0F})(previous, current);
            }));
        failing.apply(frame);
        expectIdentity(failing.apply(frame), "model failure gives identity");
        expect(failing_state.resets == 1 && !failing.initializedFirstFrame(), "model failure resets the model");
        fail = false;
        expectIdentity(failing.apply(frame), "frame after failure is a first frame");
        expectNear(failing.apply(frame).at<double>(0, 2), 1280.0 / 512.0, 1e-4, "recovers after failure");

        std::cout << "dpoint GMC tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dpoint GMC test failure: " << error.what() << '\n';
        return 1;
    }
}
