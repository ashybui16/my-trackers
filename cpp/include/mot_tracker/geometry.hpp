#pragma once

#include "mot_tracker/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace mot_tracker {

// Đổi giữa hộp góc (x1, y1, x2, y2) và hộp tâm (cx, cy, w, h).
std::array<double, 4> xyxyToXywh(const BBox& box);
BBox xywhToXyxy(const std::array<double, 4>& box);
// Tính tỉ lệ giao trên hợp của hai hộp; giá trị thuộc [0, 1].
double iou(const BBox& first, const BBox& second);
// Tạo ma trận chi phí 1 - IoU cho hai nhóm hộp.
std::vector<std::vector<double>> iouDistance(const std::vector<BBox>& first,
                                             const std::vector<BBox>& second);
// Điều chỉnh chi phí ghép bằng điểm tin cậy ứng với từng cột detection.
std::vector<std::vector<double>> fuseScore(const std::vector<std::vector<double>>& cost,
                                           const std::vector<double>& scores);

// Biến đổi trung bình và hiệp phương sai của trạng thái XYWH theo ma trận affine 2x3.
void warpXywhState(cv::Mat& mean, cv::Mat& covariance, const cv::Mat& warp);

}  // namespace mot_tracker
