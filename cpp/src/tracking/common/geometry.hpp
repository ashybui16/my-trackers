#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace mot {

std::array<double, 4> xyxyToXywh(const BBox& box);
BBox xywhToXyxy(const std::array<double, 4>& box);
double iou(const BBox& first, const BBox& second);
std::vector<std::vector<double>> iouDistance(const std::vector<BBox>& first,
                                             const std::vector<BBox>& second);
std::vector<std::vector<double>> fuseScore(const std::vector<std::vector<double>>& cost,
                                           const std::vector<double>& scores);

void warpXywhState(cv::Mat& mean, cv::Mat& covariance, const cv::Mat& warp);

}
