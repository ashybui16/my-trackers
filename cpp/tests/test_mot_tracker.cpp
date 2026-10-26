#include "mot_tracker/assignment.hpp"
#include "mot_tracker/association.hpp"
#include "mot_tracker/geometry.hpp"
#include "mot_tracker/kalman_filter.hpp"
#include "mot_tracker/tracker.hpp"
#include "mot_tracker/types.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    // Dừng bài test tại điều kiện sai đầu tiên.
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    // So số thực bằng sai số cho phép thay vì so trực tiếp.
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

mot_tracker::Detection detection(double x = 0.0, double score = 0.9, int class_id = 0) {
    // Tạo hộp 10x10 để các ca kiểm thử chỉ cần đổi tọa độ x.
    return {{x, 0.0, x + 10.0, 10.0}, score, class_id};
}

void testTypesGeometryAssignment() {
    // Kiểm tra dữ liệu đầu vào, IoU, ghép ma trận và ngưỡng cấu hình.
    bool rejected = false;
    try {
        mot_tracker::Detection invalid{{0.0, 0.0, 0.0, 1.0}, 0.5, 0};
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "invalid detection was accepted");
    expectNear(mot_tracker::iou({0.0, 0.0, 10.0, 10.0}, {0.0, 0.0, 10.0, 10.0}), 1.0, 1e-12, "IoU");
    expectNear(mot_tracker::iou({0.0, 0.0, 10.0, 10.0}, {20.0, 20.0, 30.0, 30.0}), 0.0, 1e-12, "disjoint IoU");

    const auto assignment = mot_tracker::linearAssignment({{0.1, 0.9, 0.9}, {0.9, 0.1, 0.9}}, 0.5);
    expect(assignment.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "rectangular assignment");
    expect(assignment.unmatched_rows.empty(), "unexpected unmatched row");
    expect(assignment.unmatched_cols == std::vector<int>{2}, "unexpected unmatched column");
    const auto tie = mot_tracker::linearAssignment({{0.0, 0.0}, {0.0, 0.0}}, 0.1);
    expect(tie.matches == std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}, "assignment tie behavior");
    const auto limited = mot_tracker::linearAssignment({{0.9, 0.2}, {0.2, 0.9}}, 0.5);
    expect(limited.matches == std::vector<std::pair<int, int>>{{0, 1}, {1, 0}}, "cost-limit assignment");
    const auto empty_rows = mot_tracker::linearAssignment({{}, {}}, 0.5);
    expect(empty_rows.unmatched_rows == std::vector<int>{0, 1} && empty_rows.unmatched_cols.empty(), "empty assignment");
    const auto fused = mot_tracker::fuseScore({{0.2, 0.4}}, {0.5, 0.75});
    expectNear(fused[0][0], 0.6, 1e-12, "score fusion first value");
    expectNear(fused[0][1], 0.55, 1e-12, "score fusion second value");

    mot_tracker::BoTSORTConfig config;
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

void testKalman() {
    // Khởi tạo, dự đoán và hiệu chỉnh phải giữ trạng thái/hiệp phương sai hợp lệ.
    mot_tracker::KalmanFilterXYWH kalman;
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
    // Theo dõi vòng đời: tạo, mất, hết hạn, xác nhận lại và reset ID.
    mot_tracker::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 0;
    mot_tracker::BoTSORT tracker(config);
    expect(tracker.update({detection()})[0].track_id == 1, "first activation");
    expect(tracker.update({}).empty(), "empty frame output");
    expect(tracker.removedTracks().empty(), "track expired too early");
    expect(tracker.update({}).empty(), "second empty frame output");
    expect(!tracker.removedTracks().empty() && tracker.removedTracks().front()->trackId() == 1, "track expiry");
    expect(tracker.update({detection(0.0, 0.9, 2)}).empty(), "new track should be unconfirmed");
    expect(tracker.update({detection(0.0, 0.9, 2)})[0].track_id == 2, "unconfirmed activation");
    tracker.reset();
    expect(tracker.frameId() == 0, "reset frame counter");
    expect(tracker.update({detection()})[0].track_id == 1, "reset ID counter");

    mot_tracker::BoTSORT reactivation(config);
    const int first_id = reactivation.update({detection(0.0, 0.9, 1)})[0].track_id;
    reactivation.update({});
    const auto recovered = reactivation.update({detection(0.0, 0.65, 4)});
    expect(recovered.size() == 1 && recovered.front().track_id == first_id, "reactivation ID");
    expectNear(recovered.front().score, 0.65, 1e-12, "reactivation score");
    expect(recovered.front().class_id == 4, "reactivation class");
}

void testGmcRequirement() {
    // Optical flow cần frame ảnh khi tracker cập nhật.
    mot_tracker::BoTSORT tracker;
    bool rejected = false;
    try {
        tracker.update({detection()});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "sparse GMC should require a frame");
}

void testSparseGmcTranslation() {
    // Tạo ảnh dịch chuyển nhân tạo để đo affine và nhánh thiếu đặc trưng.
    cv::Mat frame = cv::Mat::zeros(160, 200, CV_8UC1);
    for (int x = 15; x < 190; x += 20) {
        cv::circle(frame, cv::Point(x, 40 + (x % 50)), 4, cv::Scalar(255), -1);
    }
    cv::Mat shifted;
    const cv::Mat translation = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 4.0, 0.0, 1.0, 3.0);
    cv::warpAffine(frame, shifted, translation, frame.size());
    mot_tracker::GMC gmc("sparseOptFlow", 1);
    const cv::Mat first = gmc.apply(frame);
    expectNear(first.at<double>(0, 2), 0.0, 1e-12, "GMC first-frame translation");
    const cv::Mat warp = gmc.apply(shifted);
    expectNear(warp.at<double>(0, 2), 4.0, 1.0, "GMC x translation");
    expectNear(warp.at<double>(1, 2), 3.0, 1.0, "GMC y translation");
    gmc.reset();
    expect(!gmc.initializedFirstFrame(), "GMC reset");

    cv::Mat blank = cv::Mat::zeros(32, 32, CV_8UC1);
    mot_tracker::GMC no_features("sparseOptFlow", 1);
    no_features.apply(blank);
    const cv::Mat identity = no_features.apply(blank);
    expectNear(identity.at<double>(0, 0), 1.0, 1e-12, "GMC identity after missing features");
    expect(no_features.initializedFirstFrame(), "GMC reference should advance after missing features");
}

void testWarpKeepsPositiveDimensions() {
    // Xoay trạng thái Kalman nhưng chiều rộng/chiều cao hộp vẫn phải dương.
    mot_tracker::KalmanFilterXYWH kalman;
    const auto state = kalman.initiate({40.0, 30.0, 20.0, 10.0});
    cv::Mat mean = state.mean.clone();
    cv::Mat covariance = state.covariance.clone();
    const double angle = 2.2;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << cosine, -sine, 0.0,
                          sine, cosine, 0.0);
    mot_tracker::warpXywhState(mean, covariance, warp);
    expect(mean.at<double>(2, 0) > 0.0 && mean.at<double>(3, 0) > 0.0,
           "GMC rotation must preserve positive width and height");
    const auto bbox = mot_tracker::xywhToXyxy({mean.at<double>(0, 0), mean.at<double>(1, 0),
                                                mean.at<double>(2, 0), mean.at<double>(3, 0)});
    expect(bbox[2] > bbox[0] && bbox[3] > bbox[1],
           "warped state must produce a valid box");
}

void testDuplicateRemoval() {
    // Hai quỹ đạo trùng nhau phải giữ quỹ đạo tồn tại lâu hơn.
    mot_tracker::KalmanFilterXYWH kalman;
    auto active = std::make_shared<mot_tracker::Track>(detection());
    auto stale = std::make_shared<mot_tracker::Track>(detection());
    active->activate(kalman, 1, 1);
    stale->activate(kalman, 1, 2);
    active->update(*std::make_shared<mot_tracker::Track>(detection(0.0, 0.9, 0)), 3);
    stale->markLost();
    const auto result = mot_tracker::removeDuplicateTracks({active}, {stale});
    expect(result.tracked.size() == 1 && result.lost.empty() && result.removed.size() == 1,
           "duplicate removal should keep the longer-lived track");
}

}  // namespace

int main() {
    // Chạy lần lượt các ca kiểm thử lõi và trả mã lỗi khi có ngoại lệ.
    try {
        testTypesGeometryAssignment();
        testKalman();
        testLifecycle();
        testGmcRequirement();
        testSparseGmcTranslation();
        testWarpKeepsPositiveDimensions();
        testDuplicateRemoval();
        std::cout << "mot_tracker core tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "mot_tracker core test failure: " << exception.what() << '\n';
        return 1;
    }
}
