#include "mot_tracker/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mot_tracker {

std::array<double, 4> xyxyToXywh(const BBox& box) {
    // Kiểm tra hộp góc rồi tính tâm và kích thước.
    validateBBox(box);
    return {(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[2] - box[0], box[3] - box[1]};
}

BBox xywhToXyxy(const std::array<double, 4>& box) {
    // Kiểm tra kích thước rồi trải hộp tâm thành hai góc.
    for (double value : box) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("xywh must contain only finite values");
        }
    }
    if (box[2] <= 0.0 || box[3] <= 0.0) {
        throw std::invalid_argument("xywh must have positive width and height");
    }
    return {box[0] - box[2] / 2.0, box[1] - box[3] / 2.0,
            box[0] + box[2] / 2.0, box[1] + box[3] / 2.0};
}

double iou(const BBox& first, const BBox& second) {
    // Lấy diện tích giao chia diện tích hợp; hộp không giao có IoU bằng 0.
    validateBBox(first);
    validateBBox(second);
    const double left = std::max(first[0], second[0]);
    const double top = std::max(first[1], second[1]);
    const double right = std::min(first[2], second[2]);
    const double bottom = std::min(first[3], second[3]);
    const double intersection = std::max(0.0, right - left) * std::max(0.0, bottom - top);
    const double first_area = (first[2] - first[0]) * (first[3] - first[1]);
    const double second_area = (second[2] - second[0]) * (second[3] - second[1]);
    const double union_area = first_area + second_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

std::vector<std::vector<double>> iouDistance(const std::vector<BBox>& first,
                                             const std::vector<BBox>& second) {
    // Mỗi ô [row][column] là 1 - IoU của một cặp hộp.
    std::vector<std::vector<double>> result(first.size(), std::vector<double>(second.size(), 0.0));
    for (std::size_t row = 0; row < first.size(); ++row) {
        for (std::size_t column = 0; column < second.size(); ++column) {
            result[row][column] = 1.0 - iou(first[row], second[column]);
        }
    }
    return result;
}

std::vector<std::vector<double>> fuseScore(const std::vector<std::vector<double>>& cost,
                                           const std::vector<double>& scores) {
    // Kiểm tra ma trận rồi nhân độ tương đồng của từng cột với điểm detection.
    if (cost.empty()) {
        if (!scores.empty()) {
            throw std::invalid_argument("scores must contain one value per detection column");
        }
        return {};
    }
    const std::size_t columns = cost.front().size();
    for (const auto& row : cost) {
        if (row.size() != columns) {
            throw std::invalid_argument("cost matrix must be rectangular");
        }
    }
    if (scores.size() != columns) {
        throw std::invalid_argument("scores must contain one value per detection column");
    }
    for (double score : scores) {
        validateScore(score);
    }
    std::vector<std::vector<double>> result = cost;
    for (auto& row : result) {
        for (std::size_t column = 0; column < columns; ++column) {
            if (!std::isfinite(row[column])) {
                throw std::invalid_argument("cost matrix must contain only finite values");
            }
            row[column] = 1.0 - (1.0 - row[column]) * scores[column];
        }
    }
    return result;
}

void warpXywhState(cv::Mat& mean, cv::Mat& covariance, const cv::Mat& warp) {
    // Kiểm tra trạng thái, biến đổi tâm/kích thước/vận tốc và truyền hiệp phương sai.
    if (mean.rows != 8 || mean.cols != 1 || mean.type() != CV_64F ||
        covariance.rows != 8 || covariance.cols != 8 || covariance.type() != CV_64F ||
        warp.rows != 2 || warp.cols != 3 || warp.type() != CV_64F) {
        throw std::invalid_argument("invalid Kalman/GMC shapes");
    }
    cv::Mat transform = cv::Mat::zeros(8, 8, CV_64F);
    const double a00 = warp.at<double>(0, 0);
    const double a01 = warp.at<double>(0, 1);
    const double a10 = warp.at<double>(1, 0);
    const double a11 = warp.at<double>(1, 1);
    for (int row = 0; row < mean.rows; ++row) {
        if (!std::isfinite(mean.at<double>(row, 0))) {
            throw std::invalid_argument("Kalman mean must contain only finite values");
        }
    }
    for (int row = 0; row < covariance.rows; ++row) {
        for (int column = 0; column < covariance.cols; ++column) {
            if (!std::isfinite(covariance.at<double>(row, column))) {
                throw std::invalid_argument("Kalman covariance must contain only finite values");
            }
        }
    }
    for (double value : {a00, a01, a10, a11, warp.at<double>(0, 2), warp.at<double>(1, 2)}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("GMC warp must contain only finite values");
        }
    }
    if (mean.at<double>(2, 0) <= 0.0 || mean.at<double>(3, 0) <= 0.0) {
        throw std::invalid_argument("cannot warp a Kalman state with non-positive size");
    }

    // Tâm là một điểm, còn w/h là độ rộng của hộp song song trục.
    // Lấy trị tuyệt đối của phép quay để kích thước sau biến đổi luôn dương.
    const double transformed_width = std::abs(a00) * mean.at<double>(2, 0) +
                                     std::abs(a01) * mean.at<double>(3, 0);
    const double transformed_height = std::abs(a10) * mean.at<double>(2, 0) +
                                      std::abs(a11) * mean.at<double>(3, 0);
    const double scale_x = std::hypot(a00, a10);
    const double scale_y = std::hypot(a01, a11);
    const double size_scale = (scale_x + scale_y) * 0.5;
    if (!std::isfinite(transformed_width) || !std::isfinite(transformed_height) ||
        transformed_width <= 0.0 || transformed_height <= 0.0 || !std::isfinite(size_scale) ||
        size_scale <= std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("GMC warp produced a non-positive Kalman size");
    }

    for (int block = 0; block < 4; ++block) {
        transform.at<double>(2 * block, 2 * block) = a00;
        transform.at<double>(2 * block, 2 * block + 1) = a01;
        transform.at<double>(2 * block + 1, 2 * block) = a10;
        transform.at<double>(2 * block + 1, 2 * block + 1) = a11;
    }
    // Vận tốc kích thước chỉ cần hệ số tỉ lệ dương để lần dự đoán sau không âm.
    transform.at<double>(2, 2) = std::abs(a00);
    transform.at<double>(2, 3) = std::abs(a01);
    transform.at<double>(3, 2) = std::abs(a10);
    transform.at<double>(3, 3) = std::abs(a11);
    transform.at<double>(6, 6) = size_scale;
    transform.at<double>(6, 7) = 0.0;
    transform.at<double>(7, 6) = 0.0;
    transform.at<double>(7, 7) = size_scale;
    cv::Mat transformed_mean = transform * mean;
    transformed_mean.at<double>(0, 0) += warp.at<double>(0, 2);
    transformed_mean.at<double>(1, 0) += warp.at<double>(1, 2);
    transformed_mean.at<double>(2, 0) = transformed_width;
    transformed_mean.at<double>(3, 0) = transformed_height;
    cv::Mat transformed_covariance = transform * covariance * transform.t();
    mean = std::move(transformed_mean);
    covariance = std::move(transformed_covariance);
}

}  // namespace mot_tracker
