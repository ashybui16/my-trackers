"""LAPJV assignment with an explicit cost limit."""

from __future__ import annotations

import lap
import numpy as np


def _validate_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(cost_matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if not np.all(np.isfinite(value)):
        raise ValueError("cost_matrix must contain only finite values")
    return value


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_limit: float | None = None,
    *,
    thresh: float | None = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Solve with ``lap.lapjv`` and its built-in unmatched cost limit."""

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
    _, row_to_col, col_to_row = lap.lapjv(cost, extend_cost=True, cost_limit=float(cost_limit))
    matches = np.asarray(
        [[row, int(column)] for row, column in enumerate(row_to_col) if column >= 0],
        dtype=np.int64,
    ).reshape(-1, 2)
    unmatched_rows = tuple(int(row) for row in np.flatnonzero(row_to_col < 0))
    unmatched_cols = tuple(int(column) for column in np.flatnonzero(col_to_row < 0))
    return matches, unmatched_rows, unmatched_cols


def assign(cost_matrix: np.ndarray, cost_limit: float) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    return linear_assignment(cost_matrix, cost_limit)


def lapjv(cost_matrix: np.ndarray, cost_limit: float | None = None, *, thresh: float | None = None):
    """Compatibility alias for callers that name the solver directly."""

    return linear_assignment(cost_matrix, cost_limit, thresh=thresh)
