#include "mot_tracker/assignment.hpp"
#include "mot_tracker/association.hpp"
#include "mot_tracker/geometry.hpp"
#include "mot_tracker/kalman_filter.hpp"
#include "mot_tracker/tracker.hpp"
#include "mot_tracker/types.hpp"

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
    // Dừng bài test tại điều kiện sai đầu tiên.
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    // So số thực bằng sai số cho phép thay vì so trực tiếp.
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
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

void testIouMatchesUltralytics() {
    // Golden từ matching.iou_distance/bbox_ioa, vendored Ultralytics 8.4.157.
    // Bao phủ epsilon, float32, hộp đảo góc và hợp bằng 0 sau GMC.
    struct Case {
        mot_tracker::BBox first;
        mot_tracker::BBox second;
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
        expectNear(mot_tracker::iou(item.first, item.second), item.overlap, 1e-12, "Ultralytics IoU");
        const auto distance = mot_tracker::iouDistance({item.first}, {item.second});
        expectNear(distance[0][0], item.distance, 1e-12, "Ultralytics IoU distance");
    }
    bool rejected = false;
    try {
        mot_tracker::iou({0, 0, std::numeric_limits<double>::infinity(), 1}, {0, 0, 1, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "non-finite IoU input was accepted");
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
    expect(tracker.lostTracks().size() == 1 &&
               tracker.lostTracks().front()->state() == mot_tracker::TrackState::Removed,
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

void testWarpMatchesUltralytics() {
    // Giá trị tham chiếu từ multi_gmc của Ultralytics 8.4.157, quay 10 độ và dịch (5, 7).
    mot_tracker::KalmanFilterXYWH kalman;
    auto state = kalman.initiate({40.0, 30.0, 20.0, 10.0});
    for (int index = 4; index < 8; ++index) {
        state.mean.at<double>(index, 0) = std::array<double, 4>{3.0, 4.0, 2.0, 1.0}[index - 4];
    }
    const double angle = std::acos(-1.0) / 18.0;
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << std::cos(angle), -std::sin(angle), 5.0,
                          std::sin(angle), std::cos(angle), 7.0);
    mot_tracker::warpXywhState(state.mean, state.covariance, warp);
    const std::array<double, 8> expected_mean = {
        39.18286479048041, 43.49015969704345, 17.959673283574855, 13.321041083460686,
        2.2598305483689027, 4.460175545049623, 1.7959673283574857, 1.3321041083460687};
    const std::array<double, 8> expected_diagonal = {
        3.9095389311788624, 1.0904610688211374, 3.9095389311788624, 1.0904610688211374,
        1.527163644991743, 0.42596135500825677, 1.527163644991743, 0.42596135500825677};
    for (int index = 0; index < 8; ++index) {
        expectNear(state.mean.at<double>(index, 0), expected_mean[index], 1e-12, "GMC mean parity");
        for (int column = 0; column < 8; ++column) {
            const double expected = index == column ? expected_diagonal[index]
                : index / 2 == column / 2 ? (index < 4 ? 0.513030214988503 : 0.200402427729884)
                                         : 0.0;
            expectNear(state.covariance.at<double>(index, column), expected, 1e-12, "GMC covariance parity");
        }
    }

    // Phép quay lớn phải giữ kết quả R*w/h thay vì ép dương hoặc hoàn tác warp.
    mot_tracker::Track track({{30.0, 25.0, 50.0, 35.0}, 0.9, 0});
    track.activate(kalman, 1, 1);
    const cv::Mat large_rotation = (cv::Mat_<double>(2, 3) << std::cos(2.2), -std::sin(2.2), 5.0,
                                    std::sin(2.2), std::cos(2.2), 7.0);
    track.applyWarp(large_rotation);
    expectNear(track.mean().at<double>(0, 0), -42.79493680480154, 1e-12, "GMC rotated center");
    expectNear(track.mean().at<double>(2, 0), -19.854986383302816, 1e-12, "GMC signed width");
    expectNear(track.mean().at<double>(3, 0), 10.284916903838344, 1e-12, "GMC rotated height");
    const mot_tracker::BBox expected_box{
        -32.86744361315013, 16.542364183204057, -52.722429996452945, 26.8272810870424};
    const auto box = track.bbox();
    for (std::size_t index = 0; index < box.size(); ++index) {
        expectNear(box[index], expected_box[index], 1e-12, "signed GMC bbox parity");
    }
    expectNear(track.xywh()[2], -19.854986383302816, 1e-12, "signed GMC xywh parity");
    const mot_tracker::TrackOutput output(1, box, 0.9, 0);
    expect(output.bbox_xyxy == box, "track output must preserve inverted GMC corners");
    bool rejected = false;
    try {
        mot_tracker::TrackOutput invalid(1, {0, 0, std::numeric_limits<double>::infinity(), 1}, 0.9, 0);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "non-finite track output was accepted");

    track.predict();
    expectNear(track.mean().at<double>(2, 0), -19.854986383302816, 1e-12,
               "predict must not repair signed GMC width");
    track.update(mot_tracker::Track({{39.5, 29.5, 40.5, 30.5}, 0.8, 2}), 2);
    // Golden từ BOTrack.predict/update, không sửa kích thước trước hoặc sau Kalman.
    const std::array<double, 8> updated_mean{
        18.502678114060394, 27.062568670904014, -3.20462825092617, 1.0464663753364114,
        11.179857593517506, 0.6854816591811217, 3.49599153973328, -2.5820180205521126};
    const std::array<double, 8> updated_diagonal{
        0.7401845468830346, 0.24617419319732825, 0.7401845468830346, 0.24617419319732825,
        0.6662386392927906, 0.8718378512325102, 0.6662386392927906, 0.8718378512325102};
    for (int index = 0; index < 8; ++index) {
        expectNear(track.mean().at<double>(index, 0), updated_mean[index], 1e-10,
                   "Kalman update after signed GMC mean parity");
        expectNear(track.covariance().at<double>(index, index), updated_diagonal[index], 1e-10,
                   "Kalman update after signed GMC covariance parity");
    }
}

void testSignedGmcLifecycle() {
    // Bbox đảo góc không được dùng hộp cũ để cứu ID; kết quả theo Ultralytics.
    mot_tracker::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 1;
    mot_tracker::BoTSORT tracker(config);
    const mot_tracker::Detection observed({30, 25, 50, 35}, 0.9, 0);
    expect(tracker.update({observed}).front().track_id == 1, "signed GMC initial ID");
    const double cosine = std::cos(2.2);
    const double sine = std::sin(2.2);
    const cv::Mat warp = (cv::Mat_<double>(2, 3) << cosine, -sine, 40 - 40 * cosine + 30 * sine,
                          sine, cosine, 30 - 40 * sine - 30 * cosine);
    tracker.trackedTracks().front()->applyWarp(warp);
    expect(tracker.update({observed}).empty(), "inverted track must not match using a fallback bbox");
    expect(tracker.lostTracks().size() == 1 && tracker.lostTracks().front()->trackId() == 1 &&
               tracker.lostTracks().front()->state() == mot_tracker::TrackState::Lost,
           "inverted track must become lost");
    expect(tracker.lostTracks().front()->bbox()[0] > tracker.lostTracks().front()->bbox()[2],
           "lost track must retain signed dimensions");
    const auto recovered = tracker.update({observed});
    expect(recovered.size() == 1 && recovered.front().track_id == 2,
           "new detection should confirm a new ID after signed GMC");
}

void testRemovedHistoryLimit() {
    mot_tracker::BoTSORTConfig config;
    config.gmc_method = "none";
    config.track_buffer = 0;
    mot_tracker::BoTSORT tracker(config);
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
        testIouMatchesUltralytics();
        testKalman();
        testLifecycle();
        testGmcRequirement();
        testSparseGmcTranslation();
        testWarpMatchesUltralytics();
        testSignedGmcLifecycle();
        testRemovedHistoryLimit();
        testDuplicateRemoval();
        std::cout << "mot_tracker core tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "mot_tracker core test failure: " << exception.what() << '\n';
        return 1;
    }
}
