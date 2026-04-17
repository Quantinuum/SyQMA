"""Small linear-algebra helpers used by SyQMA."""

import time
from collections.abc import Generator

import numpy as np
import numba as nb

from .gf2 import solve_gf2_system


@nb.njit
def kernel_enumerator(x_particular, K) -> Generator:
    """Yield all affine GF(2) solutions by walking the kernel in Gray-code order."""
    d = K.shape[0]
    n = x_particular.shape[0]
    num_solutions = 1 << d
    current_x = x_particular.copy()
    yield current_x.copy()
    prev_gray = 0
    for sol in range(1, num_solutions):
        cur_gray = sol ^ (sol >> 1)
        diff = cur_gray ^ prev_gray
        bit_index = 0
        while diff:
            if diff & 1:
                break
            diff >>= 1
            bit_index += 1
        for j in range(n):
            current_x[j] ^= K[bit_index, j]

        yield current_x.copy()
        prev_gray = cur_gray


def num2vec(x: int, w: int) -> list[int]:
    """Convert a number to a vector of bits.

    Args:
        x (int): The number to convert.
        w (int): The width of the vector.

    Returns:
        list[int]: The vector of bits.

    """
    return [(x >> (w - 1 - i)) & 1 for i in range(w)]


def solve_gf2_original(
    A: list[list[int]] | np.ndarray,
    b: list[int] | np.ndarray,
) -> Generator[list[int], None, None]:
    """Yield all solutions to ``A @ x == b`` over GF(2).

    Adapted from https://github.com/nneonneo/pwn-stuff/blob/main/math/gf2.py.

    A: MxN array of bits representing N linear equations over GF(2)
    b: M-element vector of bits

    yields N-element vectors x containing valid solutions to Ax=b
    """
    if isinstance(A, np.ndarray):
        A = A.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()

    # Construct augmented matrix M
    M = [Ar + [bi] for Ar, bi in zip(A, b, strict=False)]
    nr, nc = len(A), len(A[0])

    # Pack the bits of M into a column of bigints
    M = [sum((int(v) << (nc - i)) for i, v in enumerate(row)) for row in M]

    leads = [-1] * nr
    c = 0
    # gaussian elimination
    for i in range(nc):
        mask = 1 << (nc - i)
        for j in range(c, nr):
            if M[j] & mask:
                z = M[j]
                M[c], M[j] = M[j], M[c]
                for k in range(c + 1, nr):
                    if M[k] & mask:
                        M[k] ^= z
                leads[c] = i
                c += 1
                break
        else:
            continue  # zeros in this col
        if c >= nr:
            break

    if 1 in M:
        print("We should get no solutions.")

    if 1 in M:
        # 0000...001 => impossible
        # return "No solutions."
        return

    M = [num2vec(row, nc + 1) for row in M]

    free_vars = sorted(set(range(nc)) - set(leads))
    num_free_vars = len(free_vars)

    # To iterate over solutions in lexicographical order, use bit shifts.
    for i in range(1 << num_free_vars):
        x = [-1] * nc
        free_assign = [(i >> j) & 1 for j in range(num_free_vars)]
        for idx, var in enumerate(free_vars):
            x[var] = free_assign[idx]
        # Back-substitution to determine pivot variables.
        for i in reversed(range(nr)):
            if leads[i] == -1:
                continue
            k = leads[i]
            Mx = sum(M[i][j] * x[j] for j in range(k + 1, nc))
            x[k] = (Mx % 2) ^ M[i][nc]
        yield x


def solve_gf2_only(
    A: list[list[int]] | np.ndarray,
    b: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return one GF(2) solution and the kernel basis."""
    x_particular, K = solve_gf2_system(A, b)

    if x_particular.size == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

    return x_particular, K
