import numpy as np
import pytest

from syqma.noise import (
    calculate_q_S_analytical,
    calculate_q_S_numerical,
)

np.random.seed(0)


def test_wrong_size():
    p_0 = np.random.rand(5)
    with pytest.raises(
        ValueError,
        match=r"A valid Pauli noise channel has 4 \*\* n parameters. Only 1q and 2q channels are supported.",
    ):
        calculate_q_S_analytical(p_0)


def test_analytical_equals_numerical_small():
    x_0 = [0.01, 0.02, 0.03]
    p_0 = np.array([1 - sum(x_0), *x_0])

    q_a = calculate_q_S_analytical(p_0)
    q_n = calculate_q_S_numerical(x_0)

    assert np.allclose(q_a[1:], q_n)


def test_analytical_equals_numerical_small_negative():
    x_0 = [0.01, 0.02, 0.00]
    p_0 = np.array([1 - sum(x_0), *x_0])

    q_a = calculate_q_S_analytical(p_0)
    q_n = calculate_q_S_numerical(x_0)

    assert np.allclose(q_a[1:], q_n)


def test_analytical_equals_numerical_random():
    n = 1

    x_0 = np.random.rand(4**n - 1) / 10 / (4**n - 1)
    p_0 = np.array([1 - sum(x_0), *x_0])

    q_a = calculate_q_S_analytical(p_0)
    q_n = calculate_q_S_numerical(x_0)

    assert np.allclose(q_a[1:], q_n)
