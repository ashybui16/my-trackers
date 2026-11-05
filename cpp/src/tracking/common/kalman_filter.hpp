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
    cv::Mat covariance;
};

class KalmanFilterXYWH {
public:
    KalmanFilterXYWH();

    KalmanState initiate(const std::array<double, 4>& measurement) const;
    KalmanState predict(const KalmanState& state) const;
    std::vector<KalmanState> multiPredict(const std::vector<KalmanState>& states) const;
    KalmanState update(const KalmanState& state,
                       const std::array<double, 4>& measurement,
                       std::optional<double> confidence = std::nullopt) const;
    KalmanState project(const KalmanState& state,
                        std::optional<double> confidence = std::nullopt) const;
    std::vector<double> gatingDistance(const KalmanState& state,
                                       const std::vector<std::array<double, 4>>& measurements,
                                       bool only_position = false,
                                       const std::string& metric = "maha") const;

private:
    cv::Mat motion_mat_;
    cv::Mat update_mat_;
    double std_weight_position_{1.0 / 20.0};
    double std_weight_velocity_{1.0 / 160.0};

    static void validateState(const KalmanState& state);
};

}
