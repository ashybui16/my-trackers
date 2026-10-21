"""LAPJV assignment with an explicit cost limit."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


def _validate_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(cost_matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if not np.all(np.isfinite(value)):
        raise ValueError("cost_matrix must contain only finite values")
    return value


def _lapjv(cost: np.ndarray, cost_limit: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import lap

        _, row_to_col, col_to_row = lap.lapjv(cost, extend_cost=True, cost_limit=cost_limit)
        return np.asarray(row_to_col, dtype=np.int64), np.asarray(col_to_row, dtype=np.int64)
    except ImportError:
        try:
            import lapjv  # type: ignore[import-not-found]

            result = lapjv.lapjv(cost)
            if len(result) == 3:
                row_to_col, col_to_row, _ = result
            else:
                row_to_col, col_to_row = result
            row_to_col = np.asarray(row_to_col, dtype=np.int64)
            col_to_row = np.asarray(col_to_row, dtype=np.int64)
            valid = (row_to_col >= 0) & (row_to_col < cost.shape[1])
            over_limit = np.zeros(len(row_to_col), dtype=bool)
            rows = np.flatnonzero(valid)
            over_limit[rows] = cost[rows, row_to_col[rows]] > cost_limit
            row_to_col[~valid | over_limit] = -1
            return row_to_col, col_to_row
        except ImportError:
            return _fallback_assignment(cost, cost_limit)


def _fallback_assignment(cost: np.ndarray, cost_limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic fallback used only when the LAP package is absent."""

    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return np.full(rows, -1, dtype=np.int64), np.full(cols, -1, dtype=np.int64)
    if rows > 8 or cols > 8:
        # A deterministic greedy fallback keeps imports usable in a minimal
        # environment; production installs use LAPJV from the declared dependency.
        candidates = sorted((float(cost[i, j]), i, j) for i in range(rows) for j in range(cols))
        row_to_col = np.full(rows, -1, dtype=np.int64)
        col_to_row = np.full(cols, -1, dtype=np.int64)
        for value, i, j in candidates:
            if value <= cost_limit and row_to_col[i] < 0 and col_to_row[j] < 0:
                row_to_col[i] = j
                col_to_row[j] = i
        return row_to_col, col_to_row

    best: tuple[float, int, tuple[tuple[int, int], ...]] | None = None
    # LAPJV's cost-limit mode is equivalent to allowing unmatched rows/columns
    # while selecting the least-cost valid matching. Enumerating the tiny test
    # fallback also gives stable tie behavior.
    for cardinality in range(min(rows, cols), -1, -1):
        for row_subset in combinations(range(rows), cardinality):
            for col_subset in combinations(range(cols), cardinality):
                for columns in permutations(col_subset):
                    pairs = tuple(zip(row_subset, columns))
                    values = tuple(float(cost[i, j]) for i, j in pairs)
                    if any(value > cost_limit for value in values):
                        continue
                    # A finite LAPJV cost limit acts like an unmatched-row
                    # penalty.  This lets a cheap single match beat several
                    # expensive-but-allowed matches instead of blindly
                    # maximizing cardinality.
                    objective = sum(values) - cardinality * cost_limit
                    candidate = (objective, cardinality, pairs)
                    if best is None or candidate[0] < best[0] - 1e-12 or (
                        abs(candidate[0] - best[0]) <= 1e-12
                        and (
                            candidate[1] > best[1]
                            or candidate[1] == best[1] and candidate[2] < best[2]
                        )
                    ):
                        best = candidate
    row_to_col = np.full(rows, -1, dtype=np.int64)
    col_to_row = np.full(cols, -1, dtype=np.int64)
    if best is not None:
        for i, j in best[2]:
            row_to_col[i] = j
            col_to_row[j] = i
    return row_to_col, col_to_row


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_limit: float | None = None,
    *,
    thresh: float | None = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Solve a rectangular assignment and discard pairs over ``cost_limit``."""

    cost = _validate_cost_matrix(cost_matrix)
    if cost_limit is None:
        cost_limit = thresh
    elif thresh is not None:
        raise ValueError("pass only one of cost_limit or thresh")
    if cost_limit is None:
        raise TypeError("cost_limit is required")
    if not np.isfinite(cost_limit):
        raise ValueError("cost_limit must be finite")
    if cost_limit < 0:
        raise ValueError("cost_limit must be non-negative")
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return np.empty((0, 2), dtype=np.int64), tuple(range(rows)), tuple(range(cols))
    row_to_col, col_to_row = _lapjv(cost, float(cost_limit))
    matches = np.asarray(
        [[row, int(column)] for row, column in enumerate(row_to_col) if column >= 0 and cost[row, column] <= cost_limit],
        dtype=np.int64,
    ).reshape(-1, 2)
    matched_rows = {int(row) for row in matches[:, 0]} if len(matches) else set()
    matched_cols = {int(column) for column in matches[:, 1]} if len(matches) else set()
    unmatched_rows = tuple(row for row in range(rows) if row not in matched_rows)
    unmatched_cols = tuple(column for column in range(cols) if column not in matched_cols)
    return matches, unmatched_rows, unmatched_cols


def assign(cost_matrix: np.ndarray, cost_limit: float) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    return linear_assignment(cost_matrix, cost_limit)


def lapjv(cost_matrix: np.ndarray, cost_limit: float | None = None, *, thresh: float | None = None):
    """Compatibility alias for callers that name the solver directly."""

    return linear_assignment(cost_matrix, cost_limit, thresh=thresh)
