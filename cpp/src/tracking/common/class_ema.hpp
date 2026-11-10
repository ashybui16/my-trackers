#pragma once

#include "mot/core/types.hpp"

#include <vector>

namespace mot {

// A track's class estimate, shared by every tracker. The class vector holds the detector's
// per-class scores (or `score` at `class_id` when they are missing) and is smoothed with
// v = alpha * v + (1 - alpha) * observed; the class is its argmax. alpha = 0 follows the latest detection.
class ClassEma {
public:
    ClassEma() = default;
    ClassEma(const Detection& detection, double alpha);

    // Blends in the class state of a detection (itself a freshly constructed ClassEma).
    void update(const ClassEma& observation);

    int classId() const { return class_id_; }
    const std::vector<double>& vector() const { return vector_; }

private:
    std::vector<double> vector_;
    double alpha_{0.0};
    int class_id_{0};
};

// Throws std::invalid_argument unless alpha is finite and in [0, 1).
void validateClassEmaAlpha(double alpha);

}
