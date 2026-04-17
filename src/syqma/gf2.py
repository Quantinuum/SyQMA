"""Dense GF(2) linear algebra helpers used by SyQMA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GF2Array = NDArray[np.uint8]


def _as_binary_matrix(matrix: ArrayLike) -> GF2Array:
    """Return ``matrix`` as a contiguous dense ``uint8`` GF(2) array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    out = np.asarray(matrix, dtype=np.uint8)
    if out.ndim != 2:
        raise ValueError("GF(2) matrix must be two-dimensional.")
    return np.ascontiguousarray(out & np.uint8(1))


def _as_binary_vector(vector: ArrayLike, expected_size: int) -> GF2Array:
    """Return ``vector`` as a contiguous one-dimensional ``uint8`` GF(2) array."""
    out = np.asarray(vector, dtype=np.uint8).reshape(-1)
    if out.size != expected_size:
        raise ValueError(
            f"GF(2) right-hand side has length {out.size}, expected {expected_size}."
        )
    return np.ascontiguousarray(out & np.uint8(1))


def _swap_vector_entries(vector: GF2Array, left: int, right: int) -> None:
    if left == right:
        return
    tmp = vector[left].copy()
    vector[left] = vector[right]
    vector[right] = tmp


def _particular_solution(matrix: GF2Array, rhs_in: GF2Array) -> GF2Array:
    row_count, col_count = matrix.shape
    work = matrix.copy()
    rhs = rhs_in.copy()
    pivot_cols: list[int] = []
    rank = 0

    for col in range(col_count):
        pivot_offsets = np.flatnonzero(work[rank:, col])
        if pivot_offsets.size == 0:
            continue

        pivot_row = rank + int(pivot_offsets[0])
        if pivot_row != rank:
            work[[rank, pivot_row]] = work[[pivot_row, rank]]
            _swap_vector_entries(rhs, rank, pivot_row)

        eliminate_offsets = np.flatnonzero(work[rank + 1 :, col])
        if eliminate_offsets.size:
            rows = rank + 1 + eliminate_offsets
            work[rows] ^= work[rank]
            rhs[rows] ^= rhs[rank]

        pivot_cols.append(col)
        rank += 1
        if rank == row_count:
            break

    if rank < row_count:
        zero_rows = ~np.any(work[rank:], axis=1)
        if np.any(zero_rows & (rhs[rank:] == 1)):
            return np.empty(0, dtype=np.uint8)

    for pivot_index in range(rank - 1, -1, -1):
        col = pivot_cols[pivot_index]
        rows = np.flatnonzero(work[:pivot_index, col])
        if rows.size:
            work[rows] ^= work[pivot_index]
            rhs[rows] ^= rhs[pivot_index]

    solution = np.zeros(col_count, dtype=np.uint8)
    if pivot_cols:
        solution[np.asarray(pivot_cols, dtype=np.int64)] = rhs[:rank]
    return solution


def _ldpc_dense_kernel(matrix: GF2Array) -> GF2Array:
    row_count, col_count = matrix.shape
    max_rank = min(row_count, col_count)
    elimination_rows: list[GF2Array] = []
    swap_rows: list[int] = []
    rank = 0

    # This mirrors ldpc.mod2.kernel(..., method="dense") so K ordering stays byte-identical.
    for col_idx in range(row_count):
        rr_col = matrix[col_idx].copy()
        for pivot_idx, swap_idx in enumerate(swap_rows):
            _swap_vector_entries(rr_col, pivot_idx, swap_idx)
            if rr_col[pivot_idx]:
                rr_col ^= elimination_rows[pivot_idx]

        pivot_offsets = np.flatnonzero(rr_col[rank:])
        if pivot_offsets.size == 0:
            continue

        swap_idx = rank + int(pivot_offsets[0])
        _swap_vector_entries(rr_col, rank, swap_idx)
        elimination_row = np.zeros(col_count, dtype=np.uint8)
        elimination_row[rank + 1 :] = rr_col[rank + 1 :]
        elimination_rows.append(elimination_row)
        swap_rows.append(swap_idx)

        rank += 1
        if rank == max_rank:
            break

    kernel = np.zeros((col_count - rank, col_count), dtype=np.uint8)
    for col_idx in range(col_count):
        rr_col = np.zeros(col_count, dtype=np.uint8)
        rr_col[col_idx] = 1

        for pivot_idx, swap_idx in enumerate(swap_rows):
            _swap_vector_entries(rr_col, pivot_idx, swap_idx)
            if rr_col[pivot_idx]:
                rr_col ^= elimination_rows[pivot_idx]

        kernel[:, col_idx] = rr_col[rank:]

    return kernel


def solve_gf2_system(matrix: ArrayLike, rhs: ArrayLike) -> tuple[GF2Array, GF2Array]:
    """Return one solution and a row basis for the kernel over GF(2).

    If ``matrix @ x == rhs`` is inconsistent over GF(2), both returned arrays
    are empty ``uint8`` arrays.  Otherwise ``x`` has shape ``(n,)`` and the
    kernel basis has shape ``(nullity, n)``.
    """
    binary_matrix = _as_binary_matrix(matrix)
    binary_rhs = _as_binary_vector(rhs, binary_matrix.shape[0])

    solution = _particular_solution(binary_matrix, binary_rhs)
    if solution.size == 0:
        return solution, np.empty(0, dtype=np.uint8)
    return solution, _ldpc_dense_kernel(binary_matrix)


def gf2_particular_solution(matrix: ArrayLike, rhs: ArrayLike) -> GF2Array:
    """Return one solution to ``matrix @ x == rhs`` over GF(2)."""
    binary_matrix = _as_binary_matrix(matrix)
    binary_rhs = _as_binary_vector(rhs, binary_matrix.shape[0])
    return _particular_solution(binary_matrix, binary_rhs)


def gf2_kernel(matrix: ArrayLike) -> GF2Array:
    """Return a dense row basis for the kernel of ``matrix`` over GF(2)."""
    return _ldpc_dense_kernel(_as_binary_matrix(matrix))
