#pragma once

#include <utility>
#include <vector>

namespace mot {

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;  // Cặp chỉ số (hàng, cột) được ghép.
    std::vector<int> unmatched_rows;            // Hàng chưa ghép.
    std::vector<int> unmatched_cols;            // Cột chưa ghép.
};

// Tìm phép ghép có tổng chi phí thấp; bỏ các cặp vượt cost_limit.
AssignmentResult linearAssignment(const std::vector<std::vector<double>>& cost,
                                  double cost_limit);

}  // namespace mot
