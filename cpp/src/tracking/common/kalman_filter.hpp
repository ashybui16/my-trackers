#pragma once

#include "mot/core/types.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace mot {

struct KalmanState {
    cv::Mat mean;       // Vector 8x1: (cx, cy, w, h, vx, vy, vw, vh), CV_64F.
    cv::Mat covariance; // Ma trận độ bất định 8x8, CV_64F.
};

class KalmanFilterXYWH {
public:
    KalmanFilterXYWH();

    // Khởi tạo trạng thái từ hộp đo (cx, cy, w, h).
    KalmanState initiate(const std::array<double, 4>& measurement) const;
    // Dự đoán trạng thái ở frame kế tiếp theo mô hình vận tốc không đổi.
    KalmanState predict(const KalmanState& state) const;
    // Dự đoán độc lập cho nhiều trạng thái.
    std::vector<KalmanState> multiPredict(const std::vector<KalmanState>& states) const;
    // Hiệu chỉnh dự đoán bằng hộp đo mới; confidence điều chỉnh nhiễu đo.
    KalmanState update(const KalmanState& state,
                       const std::array<double, 4>& measurement,
                       std::optional<double> confidence = std::nullopt) const;
    // Chiếu trạng thái 8 chiều xuống không gian hộp đo 4 chiều.
    KalmanState project(const KalmanState& state,
                        std::optional<double> confidence = std::nullopt) const;
    // Tính khoảng cách tới từng hộp đo để lọc ứng viên ghép.
    std::vector<double> gatingDistance(const KalmanState& state,
                                       const std::vector<std::array<double, 4>>& measurements,
                                       bool only_position = false,
                                       const std::string& metric = "maha") const;

private:
    // Ma trận chuyển trạng thái và ma trận lấy bốn thành phần vị trí/kích thước.
    cv::Mat motion_mat_;
    cv::Mat update_mat_;
    // Hệ số nhiễu vị trí và vận tốc, tỉ lệ với kích thước đối tượng.
    double std_weight_position_{1.0 / 20.0};
    double std_weight_velocity_{1.0 / 160.0};

    static void validateState(const KalmanState& state);
};

}  // namespace mot
