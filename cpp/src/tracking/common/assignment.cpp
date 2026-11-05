#include "tracking/common/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mot {
namespace {

std::vector<int> hungarian(const std::vector<double>& input, int size) {
    // Tăng dần phép ghép tối ưu bằng thế u/v và đường tăng trong ma trận vuông.
    std::vector<double> u(static_cast<std::size_t>(size) + 1, 0.0);
    std::vector<double> v(static_cast<std::size_t>(size) + 1, 0.0);
    std::vector<int> p(static_cast<std::size_t>(size) + 1, 0);
    std::vector<int> way(static_cast<std::size_t>(size) + 1, 0);

    std::vector<double> min_value(static_cast<std::size_t>(size) + 1);
    std::vector<unsigned char> used(static_cast<std::size_t>(size) + 1);
    for (int row = 1; row <= size; ++row) {
        p[0] = row;
        int column0 = 0;
        std::fill(min_value.begin(), min_value.end(), std::numeric_limits<double>::infinity());
        std::fill(used.begin(), used.end(), 0);
        do {
            used[static_cast<std::size_t>(column0)] = true;
            const int row0 = p[static_cast<std::size_t>(column0)];
            double delta = std::numeric_limits<double>::infinity();
            int column1 = 0;
            for (int column = 1; column <= size; ++column) {
                if (used[static_cast<std::size_t>(column)]) {
                    continue;
                }
                const double current = input[static_cast<std::size_t>(row0 - 1) * size + column - 1]
                                       - u[static_cast<std::size_t>(row0)]
                                       - v[static_cast<std::size_t>(column)];
                if (current < min_value[static_cast<std::size_t>(column)] - 1e-15) {
                    min_value[static_cast<std::size_t>(column)] = current;
                    way[static_cast<std::size_t>(column)] = column0;
                }
                if (min_value[static_cast<std::size_t>(column)] < delta - 1e-15) {
                    delta = min_value[static_cast<std::size_t>(column)];
                    column1 = column;
                }
            }
            for (int column = 0; column <= size; ++column) {
                if (used[static_cast<std::size_t>(column)]) {
                    u[static_cast<std::size_t>(p[static_cast<std::size_t>(column)])] += delta;
                    v[static_cast<std::size_t>(column)] -= delta;
                } else {
                    min_value[static_cast<std::size_t>(column)] -= delta;
                }
            }
            column0 = column1;
        } while (p[static_cast<std::size_t>(column0)] != 0);

        do {
            const int column1 = way[static_cast<std::size_t>(column0)];
            p[static_cast<std::size_t>(column0)] = p[static_cast<std::size_t>(column1)];
            column0 = column1;
        } while (column0 != 0);
    }

    std::vector<int> row_to_column(static_cast<std::size_t>(size), -1);
    for (int column = 1; column <= size; ++column) {
        if (p[static_cast<std::size_t>(column)] != 0) {
            row_to_column[static_cast<std::size_t>(p[static_cast<std::size_t>(column)] - 1)] = column - 1;
        }
    }
    return row_to_column;
}

}  // namespace

AssignmentResult linearAssignment(const std::vector<std::vector<double>>& cost,
                                  double cost_limit) {
    // Kiểm tra ma trận chi phí trước khi thêm hàng/cột giả cho lựa chọn không ghép.
    if (!std::isfinite(cost_limit) || cost_limit < 0.0) {
        throw std::invalid_argument("cost_limit must be finite and non-negative");
    }
    const int rows = static_cast<int>(cost.size());
    const int columns = rows == 0 ? 0 : static_cast<int>(cost.front().size());
    for (const auto& row : cost) {
        if (static_cast<int>(row.size()) != columns) {
            throw std::invalid_argument("cost matrix must be rectangular");
        }
        for (double value : row) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("cost matrix must contain only finite values");
            }
        }
    }
    AssignmentResult result;
    if (rows == 0 || columns == 0) {
        for (int row = 0; row < rows; ++row) {
            result.unmatched_rows.push_back(row);
        }
        for (int column = 0; column < columns; ++column) {
            result.unmatched_cols.push_back(column);
        }
        return result;
    }

    // Hàng/cột giả cho phép bỏ ghép; cặp thật chỉ có lợi khi chi phí không vượt ngưỡng.
    const int size = rows + columns;
    constexpr double forbidden = 1.0e12;
    std::vector<double> padded(static_cast<std::size_t>(size) * size, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const double value = cost[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)];
            padded[static_cast<std::size_t>(row) * size + column] =
                value <= cost_limit ? value - cost_limit - 1e-12 : forbidden;
        }
    }

    // Giải bài toán vuông rồi đổi về chỉ số hàng/cột thật.
    const std::vector<int> assignment = hungarian(padded, size);
    std::vector<bool> matched_columns(static_cast<std::size_t>(columns), false);
    for (int row = 0; row < rows; ++row) {
        const int column = assignment[static_cast<std::size_t>(row)];
        if (column >= 0 && column < columns &&
            cost[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)] <= cost_limit) {
            result.matches.emplace_back(row, column);
            matched_columns[static_cast<std::size_t>(column)] = true;
        } else {
            result.unmatched_rows.push_back(row);
        }
    }
    for (int column = 0; column < columns; ++column) {
        if (!matched_columns[static_cast<std::size_t>(column)]) {
            result.unmatched_cols.push_back(column);
        }
    }
    return result;
}

}  // namespace mot
