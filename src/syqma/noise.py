"""Noise-channel conversion helpers."""

import numpy as np
from scipy.optimize import fsolve

all_commutations_1q = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])

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


def generate_linear_system(x: np.ndarray, *x_0: float) -> list[float]:
    """Generate a system of linear equations to be solved.

    Args:
        x (float): Solution array.
        *x_0 (float): Target channel coefficients.

    Returns:
        List[float]: List of equations.

    """
    equations = []
    for i in range(len(x)):
        term_1 = 1
        term_2 = 1
        for j in range(len(x)):
            term_1 *= (1 - x[j]) if i == j else x[j]
            term_2 *= x[j] if i == j else (1 - x[j])
        equations.append(term_1 + term_2 - x_0[i])

    return equations


def calculate_q_S_numerical(p_0: np.ndarray) -> np.ndarray:
    """Calculate the q_S parameter for a Pauli noise channel using a numerical method.

    Args:
        p_0 (np.ndarray): Array of Pauli error channel probabilities.

    Returns:
        np.ndarray: Array of decomposed individual Pauli error channels probabilities.

    """
    return fsolve(generate_linear_system, p_0, args=tuple(p_0))


def calculate_q_S_analytical(p_0: np.ndarray) -> np.ndarray:
    """Calculate the q_S parameter using the analytical formula.

    Formulas are from Etienne and Henryk's paper
    (https://arxiv.org/abs/2410.08639).

    Args:
        p_0 (np.ndarray): Array of Pauli error channel probabilities.

    Returns:
        np.ndarray: Array of decomposed individual Pauli error channels probabilities.

    """
    if p_0.size == 4:
        all_commutations = all_commutations_1q
    elif p_0.size == 16:
        all_commutations = all_commutations_2q
    else:
        raise ValueError(
            "A valid Pauli noise channel has 4 ** n parameters. Only 1q and 2q channels are supported."
        )

    sum_a_S_prime_prime = np.sum(p_0 * all_commutations, axis=1)
    prod_a_S_prime = np.prod(
        np.where(1 - all_commutations, 1 - 2 * sum_a_S_prime_prime, 1), axis=1
    )
    prod_c_S_prime = np.prod(
        np.where(all_commutations, 1 - 2 * sum_a_S_prime_prime, 1), axis=1
    )
    q = 1 / 2 - 1 / 2 * (prod_a_S_prime / prod_c_S_prime) ** (2 / p_0.size)

    return q
