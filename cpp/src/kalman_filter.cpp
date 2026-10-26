#include "mot_tracker/kalman_filter.hpp"

#include <cmath>
#include <stdexcept>

namespace mot_tracker {
namespace {

void validateMeasurement(const std::array<double, 4>& measurement) {
    // Phép đo phải hữu hạn và có kích thước hộp dương.
    for (double value : measurement) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("measurement must contain only finite values");
        }
    }
    if (measurement[2] <= 0.0 || measurement[3] <= 0.0) {
        throw std::invalid_argument("measurement must have positive width and height");
    }
}

cv::Mat measurementMat(const std::array<double, 4>& measurement) {
    // Đổi hộp 4 phần tử sang vector cột OpenCV cho phép toán ma trận.
    cv::Mat result(4, 1, CV_64F);
    for (int index = 0; index < 4; ++index) {
        result.at<double>(index, 0) = measurement[static_cast<std::size_t>(index)];
    }
    return result;
}

}  // namespace

KalmanFilterXYWH::KalmanFilterXYWH() {
    // Vị trí ở frame sau bằng vị trí hiện tại cộng vận tốc; phép đo lấy bốn tọa độ đầu.
    motion_mat_ = cv::Mat::eye(8, 8, CV_64F);
    for (int index = 0; index < 4; ++index) {
        motion_mat_.at<double>(index, index + 4) = 1.0;
    }
    update_mat_ = cv::Mat::zeros(4, 8, CV_64F);
    for (int index = 0; index < 4; ++index) {
        update_mat_.at<double>(index, index) = 1.0;
    }
}

void KalmanFilterXYWH::validateState(const KalmanState& state) {
    // Chặn ma trận sai kích thước hoặc có NaN/vô cực trước khi tính toán.
    if (state.mean.rows != 8 || state.mean.cols != 1 || state.mean.type() != CV_64F ||
        state.covariance.rows != 8 || state.covariance.cols != 8 || state.covariance.type() != CV_64F) {
        throw std::invalid_argument("mean must have shape (8, 1) and covariance shape (8, 8)");
    }
    for (int row = 0; row < state.mean.rows; ++row) {
        if (!std::isfinite(state.mean.at<double>(row, 0))) {
            throw std::invalid_argument("mean and covariance must be finite");
        }
    }
    for (int row = 0; row < state.covariance.rows; ++row) {
        for (int column = 0; column < state.covariance.cols; ++column) {
            if (!std::isfinite(state.covariance.at<double>(row, column))) {
                throw std::invalid_argument("mean and covariance must be finite");
            }
        }
    }
}

KalmanState KalmanFilterXYWH::initiate(const std::array<double, 4>& measurement) const {
    // Đặt hộp đo làm trạng thái ban đầu, vận tốc bằng 0 và độ bất định theo kích thước hộp.
    validateMeasurement(measurement);
    const cv::Mat value = measurementMat(measurement);
    KalmanState state;
    state.mean = cv::Mat::zeros(8, 1, CV_64F);
    value.copyTo(state.mean.rowRange(0, 4));
    const double width = measurement[2];
    const double height = measurement[3];
    const std::array<double, 8> standard_deviation = {
        2.0 * std_weight_position_ * width,
        2.0 * std_weight_position_ * height,
        2.0 * std_weight_position_ * width,
        2.0 * std_weight_position_ * height,
        10.0 * std_weight_velocity_ * width,
        10.0 * std_weight_velocity_ * height,
        10.0 * std_weight_velocity_ * width,
        10.0 * std_weight_velocity_ * height,
    };
    state.covariance = cv::Mat::zeros(8, 8, CV_64F);
    for (int index = 0; index < 8; ++index) {
        state.covariance.at<double>(index, index) =
            standard_deviation[static_cast<std::size_t>(index)] *
            standard_deviation[static_cast<std::size_t>(index)];
    }
    return state;
}

KalmanState KalmanFilterXYWH::predict(const KalmanState& state) const {
    // Tăng độ bất định theo kích thước, rồi truyền trạng thái qua mô hình chuyển động.
    validateState(state);
    cv::Mat standard_position(4, 1, CV_64F);
    cv::Mat standard_velocity(4, 1, CV_64F);
    const double width = state.mean.at<double>(2, 0);
    const double height = state.mean.at<double>(3, 0);
    const std::array<double, 4> sizes = {width, height, width, height};
    for (int index = 0; index < 4; ++index) {
        standard_position.at<double>(index, 0) = std_weight_position_ * sizes[static_cast<std::size_t>(index)];
        standard_velocity.at<double>(index, 0) = std_weight_velocity_ * sizes[static_cast<std::size_t>(index)];
    }
    cv::Mat motion_cov = cv::Mat::zeros(8, 8, CV_64F);
    for (int index = 0; index < 4; ++index) {
        motion_cov.at<double>(index, index) =
            standard_position.at<double>(index, 0) * standard_position.at<double>(index, 0);
        motion_cov.at<double>(index + 4, index + 4) =
            standard_velocity.at<double>(index, 0) * standard_velocity.at<double>(index, 0);
    }
    return {motion_mat_ * state.mean,
            motion_mat_ * state.covariance * motion_mat_.t() + motion_cov};
}

std::vector<KalmanState> KalmanFilterXYWH::multiPredict(const std::vector<KalmanState>& states) const {
    // Lặp phép dự đoán độc lập và giữ nguyên thứ tự đầu vào.
    std::vector<KalmanState> result;
    result.reserve(states.size());
    for (const auto& state : states) {
        result.push_back(predict(state));
    }
    return result;
}

KalmanState KalmanFilterXYWH::project(const KalmanState& state,
                                      std::optional<double> confidence) const {
    // Chiếu trạng thái sang hộp đo và cộng nhiễu đo, có xét điểm tin cậy nếu có.
    validateState(state);
    cv::Mat projected_mean = update_mat_ * state.mean;
    const double width = state.mean.at<double>(2, 0);
    const double height = state.mean.at<double>(3, 0);
    const std::array<double, 4> sizes = {width, height, width, height};
    double confidence_scale = 1.0;
    if (confidence.has_value()) {
        if (!std::isfinite(*confidence)) {
            throw std::invalid_argument("confidence must be finite");
        }
        confidence_scale = std::max(1.0 - *confidence, 0.05);
    }
    cv::Mat innovation = cv::Mat::zeros(4, 4, CV_64F);
    for (int index = 0; index < 4; ++index) {
        const double standard = std_weight_position_ * sizes[static_cast<std::size_t>(index)];
        innovation.at<double>(index, index) = standard * standard * confidence_scale;
    }
    return {projected_mean, update_mat_ * state.covariance * update_mat_.t() + innovation};
}

KalmanState KalmanFilterXYWH::update(const KalmanState& state,
                                     const std::array<double, 4>& measurement,
                                     std::optional<double> confidence) const {
    // Tính Kalman gain từ dự đoán, rồi sửa trung bình và hiệp phương sai bằng sai số đo.
    validateMeasurement(measurement);
    validateState(state);
    const KalmanState projected = project(state, confidence);
    cv::Mat gain_transposed;
    if (!cv::solve(projected.covariance,
                   state.covariance.colRange(0, 4).t(),
                   gain_transposed,
                   cv::DECOMP_LU)) {
        throw std::runtime_error("Kalman update could not solve innovation covariance");
    }
    const cv::Mat gain = gain_transposed.t();
    const cv::Mat innovation = measurementMat(measurement) - projected.mean;
    cv::Mat updated_covariance = state.covariance - gain * projected.covariance * gain.t();
    updated_covariance = (updated_covariance + updated_covariance.t()) * 0.5;
    return {state.mean + gain * innovation, updated_covariance};
}

std::vector<double> KalmanFilterXYWH::gatingDistance(
    const KalmanState& state,
    const std::vector<std::array<double, 4>>& measurements,
    bool only_position,
    const std::string& metric) const {
    // So từng hộp đo với trạng thái chiếu bằng khoảng cách Euclid hoặc Mahalanobis.
    if (metric != "maha" && metric != "gaussian") {
        throw std::invalid_argument("metric must be 'maha' or 'gaussian'");
    }
    const KalmanState projected = project(state);
    const int dimensions = only_position ? 2 : 4;
    cv::Mat projected_mean = projected.mean.rowRange(0, dimensions);
    cv::Mat covariance = projected.covariance(cv::Range(0, dimensions), cv::Range(0, dimensions));
    std::vector<double> result;
    result.reserve(measurements.size());
    for (const auto& measurement : measurements) {
        validateMeasurement(measurement);
        cv::Mat difference = measurementMat(measurement).rowRange(0, dimensions) - projected_mean;
        if (metric == "gaussian") {
            result.push_back(difference.dot(difference));
            continue;
        }
        cv::Mat solved;
        if (!cv::solve(covariance, difference, solved, cv::DECOMP_CHOLESKY)) {
            throw std::runtime_error("Kalman gating could not solve covariance");
        }
        result.push_back(difference.dot(solved));
    }
    return result;
}

}  // namespace mot_tracker
