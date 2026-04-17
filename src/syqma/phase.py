"""Phase kernels for products of Pauli and symbolic trigonometric operators."""

import numba as nb
import numpy as np


@nb.njit
def _overall_phase_from_ops_mod4(ops_1: np.ndarray, ops_2: np.ndarray) -> np.complex128:
    """Compute the overall complex phase in Numba.

    Implements per-row exponent f_k = (2*dot1_k + dot2_k + dot3_k - dot4_k) mod 4, where
      dot1_k = sum(P_prev_X[k] & Z[k])
      dot2_k = sum(P_prev_Z[k] & P_prev_X[k])
      dot3_k = sum(Z[k] & X[k])
      dot4_k = sum(P_next_X[k] & P_next_Z[k])
    with P_prev_* the XOR-prefix before row k, and P_next_* the XOR-prefix including row k.
    Returns (-1j)^sum_k f_k as complex128.
    """
    m = ops_1.shape[0]
    if m == 0:
        return np.complex128(1.0 + 0.0j)
    n = ops_1.shape[1]

    prefix_x = np.zeros(n, dtype=np.uint8)
    prefix_z = np.zeros(n, dtype=np.uint8)
    F = 0  # total exponent mod 4

    for k in range(m):
        # dot2_k = sum(prefix_z & prefix_x)
        s2 = 0
        for q in range(n):
            s2 += prefix_z[q] & prefix_x[q]

        # dot1_k = sum(prefix_x & z_k)
        s1 = 0
        for q in range(n):
            s1 += prefix_x[q] & ops_2[k, q]

        # dot3_k = sum(z_k & x_k)
        s3 = 0
        for q in range(n):
            s3 += ops_2[k, q] & ops_1[k, q]

        # dot4_k = sum((prefix_x ^ x_k) & (prefix_z ^ z_k)) = sum(P_next_X & P_next_Z)
        s4 = 0
        for q in range(n):
            pxn = prefix_x[q] ^ ops_1[k, q]
            pzn = prefix_z[q] ^ ops_2[k, q]
            s4 += pxn & pzn

        fk = (2 * s1 + s2 + s3 - s4) & 3
        F = (F + fk) & 3

        # update prefixes
        for q in range(n):
            prefix_x[q] ^= ops_1[k, q]
            prefix_z[q] ^= ops_2[k, q]

    if F == 0:
        return np.complex128(1.0 + 0.0j)
    elif F == 1:
        return np.complex128(0.0 - 1.0j)
    elif F == 2:
        return np.complex128(-1.0 + 0.0j)
    else:  # F == 3
        return np.complex128(0.0 + 1.0j)


@nb.njit
def _compute_batch_non_linear_phases(
    solutions, span_x, span_z, c_ops, o_ops
) -> np.ndarray:
    """Compute non-linear phase corrections for a batch of solutions.

    Avoids allocations and Python loops by processing the batch in Numba.
    """
    batch_size, m = solutions.shape
    n_q = span_x.shape[1]
    n_m = c_ops.shape[1]

    phases = np.zeros(batch_size, dtype=np.uint8)

    for i in range(batch_size):
        # Initialize phase accumulators and prefix buffers for this solution
        F_p = 0
        prefix_x_p = np.zeros(n_q, dtype=np.uint8)
        prefix_z_p = np.zeros(n_q, dtype=np.uint8)

        F_c = 0
        prefix_x_c = np.zeros(n_m, dtype=np.uint8)
        prefix_z_c = np.zeros(n_m, dtype=np.uint8)

        for k in range(m):
            if solutions[i, k]:
                # --- Update Pauli Phase (span_x, span_z) ---
                s1_p = 0
                s2_p = 0
                s3_p = 0
                s4_p = 0
                for q in range(n_q):
                    px = prefix_x_p[q]
                    pz = prefix_z_p[q]
                    sx = span_x[k, q]
                    sz = span_z[k, q]
                    s2_p += pz & px
                    s1_p += px & sz
                    s3_p += sz & sx
                    s4_p += (px ^ sx) & (pz ^ sz)
                    prefix_x_p[q] = px ^ sx
                    prefix_z_p[q] = pz ^ sz
                F_p = (F_p + (2 * s1_p + s2_p + s3_p - s4_p)) & 3

                # --- Update C/O Phase (c_ops, o_ops) ---
                s1_c = 0
                s2_c = 0
                s3_c = 0
                s4_c = 0
                for q in range(n_m):
                    px = prefix_x_c[q]
                    pz = prefix_z_c[q]
                    cx = c_ops[k, q]
                    ox = o_ops[k, q]
                    s2_c += pz & px
                    s1_c += px & ox
                    s3_c += ox & cx
                    s4_c += (px ^ cx) & (pz ^ ox)
                    prefix_x_c[q] = px ^ cx
                    prefix_z_c[q] = pz ^ ox
                F_c = (F_c + (2 * s1_c + s2_c + s3_c - s4_c)) & 3

        # Calculate overall phase: Real(i^-Fp * i^-Fc)
        # Returns 1 (flip) if (Fp + Fc) % 4 == 2, else 0
        phases[i] = 1 if ((F_p + F_c) & 3) == 2 else 0

    return phases
