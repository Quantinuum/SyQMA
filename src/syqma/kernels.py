"""Low-level kernels for symbolic solution enumeration."""

from collections.abc import Generator

import numba as nb
import numpy as np


@nb.njit
def process_o_operator(solution, c_operators, o_operators) -> np.ndarray:
    """Return the residual O operator for a selected solution."""
    n, m = c_operators.shape
    c_op = np.zeros(m, dtype=np.uint8)
    o_op = np.zeros(m, dtype=np.uint8)
    for i in range(n):
        if solution[i]:
            for j in range(m):
                c_op[j] ^= c_operators[i, j]
                o_op[j] ^= o_operators[i, j]
    s_op = c_op & o_op
    o_op ^= s_op

    return o_op


@nb.njit
def process_cso_operators(
    solution, c_operators, o_operators
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the C, S, and O operator components for a selected solution."""
    n, m = c_operators.shape
    c_op = np.zeros(m, dtype=np.uint8)
    o_op = np.zeros(m, dtype=np.uint8)
    for i in range(n):
        if solution[i]:
            for j in range(m):
                c_op[j] ^= c_operators[i, j]
                o_op[j] ^= o_operators[i, j]
    s_op = c_op & o_op
    c_op ^= s_op
    o_op ^= s_op

    return c_op, s_op, o_op


def dfs_int(i, c_int, o_int, y, out, delta_c_int, delta_o_int, salv_int) -> None:
    """Append valid packed DFS solutions to ``out``."""
    bad = o_int & (~c_int)  # bits where o=1 & c=0

    if i == len(delta_c_int):
        if bad == 0:
            # unpack y if you still need the numpy copy:
            out.append((c_int, o_int, y.copy()))
        return

    # prune if any bad bit cannot be flipped by future deltas
    if bad & (~salv_int[i]):
        return

    # branch y[i] = 0
    y[i] = 0
    dfs_int(i + 1, c_int, o_int, y, out, delta_c_int, delta_o_int, salv_int)

    # branch y[i] = 1
    y[i] = 1
    dfs_int(
        i + 1,
        c_int ^ delta_c_int[i],
        o_int ^ delta_o_int[i],
        y,
        out,
        delta_c_int,
        delta_o_int,
        salv_int,
    )

    y[i] = 0  # restore


def dfs_int_generator(
    i: int, c_int, o_int, y_int: int, delta_c_int, delta_o_int, salv_int
) -> Generator:
    """Yield packed DFS solutions using the classic recursive enumeration."""
    bad = o_int & (~c_int)  # bits where o=1 & c=0

    if i == len(delta_c_int):
        if bad == 0:
            yield (c_int, o_int, y_int)
        return

    # prune if any bad bit cannot be flipped by future deltas
    if bad & (~salv_int[i]):
        return

    # branch y[i] = 0
    yield from dfs_int_generator(
        i + 1, c_int, o_int, y_int, delta_c_int, delta_o_int, salv_int
    )

    # branch y[i] = 1
    yield from dfs_int_generator(
        i + 1,
        c_int ^ delta_c_int[i],
        o_int ^ delta_o_int[i],
        y_int | (1 << i),
        delta_c_int,
        delta_o_int,
        salv_int,
    )


@nb.njit
def fill_dfs_batch(
    delta_c_int,
    delta_o_int,
    salv_int,
    stack_i,
    stack_took_one,
    state,  # [c, o, y_int, curr_i]
    batch_c,
    batch_o,
    batch_y,
) -> int:
    """Fill one batch of packed DFS solutions.

    Updates stack arrays and state in-place.
    Returns the number of solutions added to the batch.
    """
    n_total = len(delta_c_int)
    batch_size = len(batch_c)
    count = 0

    c = np.uint64(state[0])
    o = np.uint64(state[1])
    y_int = np.uint64(state[2])
    curr_i = np.int32(state[3])
    stack_ptr = np.int32(state[4])

    while True:
        bad = o & (~c)

        # Leaf node
        if curr_i == n_total:
            if bad == 0:
                batch_c[count] = c
                batch_o[count] = o
                batch_y[count] = y_int
                count += 1

            # Backtrack to the next possible branch
            found_next = False
            while stack_ptr > 0:
                stack_ptr -= 1
                i = stack_i[stack_ptr]
                took_one = stack_took_one[stack_ptr]

                if not took_one:
                    # switch from 0-branch to 1-branch
                    c ^= delta_c_int[i]
                    o ^= delta_o_int[i]
                    y_int |= np.uint64(1) << i
                    stack_took_one[stack_ptr] = True
                    stack_ptr += 1
                    curr_i = i + 1
                    found_next = True
                    break
                else:
                    # already did 1-branch, undo and backtrack further
                    c ^= delta_c_int[i]
                    o ^= delta_o_int[i]
                    y_int &= ~(np.uint64(1) << i)

            if not found_next:
                state[4] = np.int64(-1)  # Finished entire tree
                return count

            if count == batch_size:
                # Save state and return current batch
                state[0] = np.int64(c)
                state[1] = np.int64(o)
                state[2] = np.int64(y_int)
                state[3] = np.int64(curr_i)
                state[4] = np.int64(stack_ptr)
                return count
            continue

        # Pruning
        if bad & (~salv_int[curr_i]):
            # Backtrack immediately
            found_next = False
            while stack_ptr > 0:
                stack_ptr -= 1
                i = stack_i[stack_ptr]
                took_one = stack_took_one[stack_ptr]

                if not took_one:
                    c ^= delta_c_int[i]
                    o ^= delta_o_int[i]
                    y_int |= np.uint64(1) << i
                    stack_took_one[stack_ptr] = True
                    stack_ptr += 1
                    curr_i = i + 1
                    found_next = True
                    break
                else:
                    c ^= delta_c_int[i]
                    o ^= delta_o_int[i]
                    y_int &= ~(np.uint64(1) << i)

            if not found_next:
                state[4] = np.int64(-1)
                return count
            continue

        # Descend 0-branch
        stack_i[stack_ptr] = curr_i
        stack_took_one[stack_ptr] = False
        stack_ptr += 1
        curr_i += 1


def dfs_batch_generator(
    c_int, o_int, y_int_initial, delta_c_int, delta_o_int, salv_int, batch_size=8192
) -> Generator:
    """Yield batches of packed DFS solutions from the Numba batch filler."""
    n = len(delta_c_int)
    # State: [c, o, y_int, curr_i, stack_ptr]
    state = np.array([c_int, o_int, y_int_initial, 0, 0], dtype=np.int64)

    stack_i = np.zeros(n, dtype=np.int32)
    stack_took_one = np.zeros(n, dtype=np.bool_)

    # Pre-allocate batch arrays
    batch_c = np.zeros(batch_size, dtype=np.uint64)
    batch_o = np.zeros(batch_size, dtype=np.uint64)
    batch_y = np.zeros(batch_size, dtype=np.uint64)

    while state[4] != -1:
        count = fill_dfs_batch(
            np.array(delta_c_int, dtype=np.uint64),
            np.array(delta_o_int, dtype=np.uint64),
            np.array(salv_int, dtype=np.uint64),
            stack_i,
            stack_took_one,
            state,
            batch_c,
            batch_o,
            batch_y,
        )
        if count > 0:
            # Mask to actual count if partial batch
            if count < batch_size:
                yield (
                    batch_c[:count].copy(),
                    batch_o[:count].copy(),
                    batch_y[:count].copy(),
                )
            else:
                yield (batch_c.copy(), batch_o.copy(), batch_y.copy())
        else:
            break
