"""Pauli string and symplectic binary representation helpers."""

from itertools import product
from typing import Any

import numpy as np


def pauli_string_to_symplectic(pauli_string: str) -> np.ndarray:
    """Convert a Pauli string to the XZ symplectic binary representation."""
    n = len(pauli_string)
    np_pauli = np.zeros(2 * n, dtype=np.uint8)
    for i in range(n):
        if pauli_string[i] == "X":
            np_pauli[i] = 1
        elif pauli_string[i] == "Z":
            np_pauli[i + n] = 1
        elif pauli_string[i] == "Y":
            np_pauli[i] = 1
            np_pauli[i + n] = 1
        else:
            raise ValueError(f"Invalid Pauli string: {pauli_string}")

    return np_pauli


def commutation(pauli_1: str | Any, pauli_2: str | Any) -> bool:
    """Return whether two Paulis anticommute."""
    if isinstance(pauli_1, str):
        np_pauli_1 = pauli_string_to_symplectic(pauli_1)
    else:
        np_pauli_1 = np.array(pauli_1)
    if isinstance(pauli_2, str):
        np_pauli_2 = pauli_string_to_symplectic(pauli_2)
    else:
        np_pauli_2 = np.array(pauli_2)

    if np_pauli_1.shape != np_pauli_2.shape:
        raise ValueError("Pauli strings must be of the same length")

    n = np_pauli_1.shape[0] // 2

    x1 = np_pauli_1[:n]
    z1 = np_pauli_1[n:]
    x2 = np_pauli_2[:n]
    z2 = np_pauli_2[n:]

    result = (np.dot(x1, z2) + np.dot(z1, x2)) % 2 != 0

    return result


def generate_paulis() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict
]:
    """Generate lookup tables for one- and two-qubit Pauli products."""
    # The symplectic convention is X bits followed by Z bits.
    # 1q: I, Z, X, Y
    # 2q: II, IZ, ZI, ZZ, IX, IY, ZX, ZY, XI, XZ, YI, YZ, XX, XY, YX, YY
    #      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    all_paulis_1q = list(product([0, 1], repeat=2))
    all_paulis_2q = list(product([0, 1], repeat=4))

    all_pauli_indices_1q = np.zeros((2, 2), dtype=int)
    all_pauli_indices_1q_reverse = np.zeros((4, 2), dtype=int)
    for i in range(4):
        all_pauli_indices_1q[all_paulis_1q[i]] = i
        all_pauli_indices_1q_reverse[i] = all_paulis_1q[i]
    all_pauli_indices_2q = np.zeros((2, 2, 2, 2), dtype=int)
    all_pauli_indices_2q_reverse = np.zeros((16, 4), dtype=int)
    for i in range(16):
        all_pauli_indices_2q[all_paulis_2q[i]] = i
        all_pauli_indices_2q_reverse[i] = all_paulis_2q[i]

    pauli_multiplication_dict_1q = {}
    for i in range(4):
        for j in range(4):
            pauli_multiplication_dict_1q[(i, j)] = all_pauli_indices_1q[
                tuple(
                    all_pauli_indices_1q_reverse[i] ^ all_pauli_indices_1q_reverse[j].T
                )
            ]

    pauli_multiplication_dict_2q = {}
    for i in range(16):
        for j in range(16):
            pauli_multiplication_dict_2q[(i, j)] = all_pauli_indices_2q[
                tuple(
                    all_pauli_indices_2q_reverse[i] ^ all_pauli_indices_2q_reverse[j].T
                )
            ]

    return (
        all_pauli_indices_1q,
        all_pauli_indices_1q_reverse,
        all_pauli_indices_2q,
        all_pauli_indices_2q_reverse,
        pauli_multiplication_dict_1q,
        pauli_multiplication_dict_2q,
    )


def get_commutations() -> tuple[np.ndarray, np.ndarray]:
    """Return commutation lookup tables for one- and two-qubit Pauli products."""
    all_commutations_1q = np.array(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
    )

    all_commutations_2q = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        ]
    )

    return all_commutations_1q, all_commutations_2q
