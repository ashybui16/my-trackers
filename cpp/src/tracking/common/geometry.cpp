#include "tracking/common/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mot {

std::array<double, 4> xyxyToXywh(const BBox& box) {
    validateBBox(box);
    return {(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[2] - box[0], box[3] - box[1]};
}

BBox xywhToXyxy(const std::array<double, 4>& box) {
    for (double value : box) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("xywh must contain only finite values");
        }
    }
    return {box[0] - box[2] / 2.0, box[1] - box[3] / 2.0,
            box[0] + box[2] / 2.0, box[1] + box[3] / 2.0};
}

double iou(const BBox& first, const BBox& second) {
    // Follows Ultralytics matching.iou_distance -> bbox_ioa: float32,
    // clamps only negative intersections, keeps signed areas and adds epsilon to the union.
    std::array<float, 4> a{};
    std::array<float, 4> b{};
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (!std::isfinite(first[index]) || !std::isfinite(second[index])) {
            throw std::invalid_argument("IoU boxes must contain only finite values");
        }
        a[index] = static_cast<float>(first[index]);
        b[index] = static_cast<float>(second[index]);
    }
    const float left = std::max(a[0], b[0]);
    const float top = std::max(a[1], b[1]);
    const float right = std::min(a[2], b[2]);
    const float bottom = std::min(a[3], b[3]);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float first_area = (a[2] - a[0]) * (a[3] - a[1]);
    const float second_area = (b[2] - b[0]) * (b[3] - b[1]);
    const float union_area = second_area + first_area - intersection;
    constexpr float epsilon = 1e-7F;
    return intersection / (union_area + epsilon);
}

std::vector<std::vector<double>> iouDistance(const std::vector<BBox>& first,
                                             const std::vector<BBox>& second) {
    std::vector<std::vector<double>> result(first.size(), std::vector<double>(second.size(), 0.0));
    for (std::size_t row = 0; row < first.size(); ++row) {
        for (std::size_t column = 0; column < second.size(); ++column) {
            result[row][column] = 1.0F - static_cast<float>(iou(first[row], second[column]));
        }
    }
    return result;
}

std::vector<std::vector<double>> fuseScore(const std::vector<std::vector<double>>& cost,
                                           const std::vector<double>& scores) {
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
    // As Ultralytics multi_gmc: centers and center velocities follow the signed R; sizes and size
    // velocities follow abs(R) (the box enclosing the rotated box), so sizes never flip sign.
    for (int block = 0; block < 4; ++block) {
        const bool size_block = block == 1 || block == 3;
        const auto entry = [size_block](double value) { return size_block ? std::abs(value) : value; };
        transform.at<double>(2 * block, 2 * block) = entry(a00);
        transform.at<double>(2 * block, 2 * block + 1) = entry(a01);
        transform.at<double>(2 * block + 1, 2 * block) = entry(a10);
        transform.at<double>(2 * block + 1, 2 * block + 1) = entry(a11);
    }
    cv::Mat transformed_mean = transform * mean;
    transformed_mean.at<double>(0, 0) += warp.at<double>(0, 2);
    transformed_mean.at<double>(1, 0) += warp.at<double>(1, 2);
    cv::Mat transformed_covariance = transform * covariance * transform.t();
    mean = std::move(transformed_mean);
    covariance = std::move(transformed_covariance);
}

}
