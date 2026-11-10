#include "tracking/common/class_ema.hpp"

#include <cmath>
#include <stdexcept>

namespace mot {

namespace {

// Detector scores when available, otherwise score at class_id (one-hot).
std::vector<double> classVectorFrom(const Detection& detection) {
    if (!detection.class_scores.empty()) {
        std::vector<double> scores(detection.class_scores.begin(), detection.class_scores.end());
        if (detection.class_id >= 0 && static_cast<std::size_t>(detection.class_id) >= scores.size()) {
            scores.resize(static_cast<std::size_t>(detection.class_id) + 1, 0.0);
            scores[static_cast<std::size_t>(detection.class_id)] = detection.score;
        }
        return scores;
    }
    if (detection.class_id < 0) {
        return {};
    }
    std::vector<double> one_hot(static_cast<std::size_t>(detection.class_id) + 1, 0.0);
    one_hot.back() = detection.score;
    return one_hot;
}

}

ClassEma::ClassEma(const Detection& detection, double alpha)
    : vector_(classVectorFrom(detection)), alpha_(alpha), class_id_(detection.class_id) {}

void ClassEma::update(const ClassEma& observation) {
    const std::vector<double>& observed = observation.vector_;
    if (alpha_ <= 0.0 || observed.empty() || vector_.empty()) {
        vector_ = observed;
        class_id_ = observation.class_id_;
        return;
    }
    if (vector_.size() < observed.size()) {
        vector_.resize(observed.size(), 0.0);
    }
    for (std::size_t index = 0; index < vector_.size(); ++index) {
        const double value = index < observed.size() ? observed[index] : 0.0;
        vector_[index] = alpha_ * vector_[index] + (1.0 - alpha_) * value;
    }
    // First maximum wins ties.
    std::size_t best = 0;
    for (std::size_t index = 1; index < vector_.size(); ++index) {
        if (vector_[index] > vector_[best]) {
            best = index;
        }
    }
    class_id_ = static_cast<int>(best);
}

void validateClassEmaAlpha(double alpha) {
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("class_ema_alpha must be finite and in [0, 1)");
    }
}

}
