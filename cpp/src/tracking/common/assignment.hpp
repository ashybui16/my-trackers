#pragma once

#include <utility>
#include <vector>

namespace mot {

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_rows;
    std::vector<int> unmatched_cols;
};

AssignmentResult linearAssignment(const std::vector<std::vector<double>>& cost,
                                  double cost_limit);

}
