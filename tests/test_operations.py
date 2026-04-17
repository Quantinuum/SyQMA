from copy import deepcopy
from syqma import Simulator
import numpy as np

sim = Simulator(n=4)

sim.h(0)
sim.pauli_error_1(0)
sim.cnot(0, 1)
sim.pauli_error_2((0, 1))
sim.cnot(1, 2)
sim.pauli_error_2((1, 2))
sim.cnot(2, 3)
sim.pauli_error_2((2, 3))
sim.rz(0, 0)
sim.pauli_error_1(0)


def test_symplectic_span():
    sim_copy = deepcopy(sim)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

def test_pauli_span():
    sim_copy = deepcopy(sim)

    assert sim_copy.pauli_span() == (
        """XXXX\nZZ..\n.ZZ.\n..ZZ\nZ..."""
    )

def test_draw_circuit():
    sim_copy = deepcopy(sim)

    sim_copy.y(3)
    sim_copy.measure(1)
    sim_copy.measure(2)
    sim_copy.mpp([0, 2, 3], "XZY")
    sim_copy.measure(1)
    sim_copy.measure(2)

    assert sim_copy.draw_circuit() == (
        """0: ---H-E---@-E---RZ(θ_0)-E-------X-------\n            | |                   |       \n1: ---------X-E---@-E---M---------|---M---\n                  | |             |       \n2: ---------------X-E---@-E---M---Z---M---\n                        | |       |       \n3: ---------------------X-E---Y---Y-------"""
    )

def test_x():
    sim_copy = deepcopy(sim)

    sim_copy.x(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 1, 0, 0, 1])
    )

def test_y():
    sim_copy = deepcopy(sim)

    sim_copy.y(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([1, 1, 0, 0, 1])
    )

def test_z():
    sim_copy = deepcopy(sim)

    sim_copy.z(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([1, 0, 0, 0, 0])
    )

def test_h():
    sim_copy = deepcopy(sim)

    sim_copy.h(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0])
    )

def test_s():
    sim_copy = deepcopy(sim)

    sim_copy.s(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0])
    )

def test_sdag():
    sim_copy = deepcopy(sim)

    sim_copy.sdag(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([1, 0, 0, 0, 0])
    )

def test_cnot():
    sim_copy = deepcopy(sim)

    sim_copy.cnot(0, 1)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0])
    )

def test_rz():
    sim_copy = deepcopy(sim)

    sim_copy.rz(0, 0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0, 0])
    )

    assert np.allclose(
        sim_copy.c_operators, 
        np.array([
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.o_operators, 
        np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
    )

    assert np.allclose(
        sim_copy.theta_indices, 
        np.array([0, 0])
    )

    assert sim_copy.n_stabs == 6
    assert sim_copy.n_magic == 2

def test_ry():
    sim_copy = deepcopy(sim)

    sim_copy.ry(0, 0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0, 1])
    )

    assert np.allclose(
        sim_copy.c_operators, 
        np.array([
            [1, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.o_operators, 
        np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
    )

    assert np.allclose(
        sim_copy.theta_indices, 
        np.array([0, 0])
    )

    assert sim_copy.n_stabs == 6
    assert sim_copy.n_magic == 2

def test_ch():
    sim_copy = deepcopy(sim)

    sim_copy.ch(0, 1, 0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0]
       ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0, 1, 1])
    )
    assert np.allclose(
        sim_copy.c_operators, 
        np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.o_operators, 
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )

    assert np.allclose(
        sim_copy.theta_indices, 
        np.array([0, 1, 0])
    )

    assert sim_copy.n_stabs == 7
    assert sim_copy.n_magic == 3

def test_measure():
    sim_copy = deepcopy(sim)

    sim_copy.measure(0)

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0, 0])
    )

    assert np.allclose(
        sim_copy.symbolic_phases, 
        np.array([
            [0],
            [0],
            [0],
            [0],
            [0],
            [1]
        ])
    )

    assert sim_copy.n_m == 1
    assert sim_copy.measurement_qubits == [[0]]

def test_mpp():
    sim_copy = deepcopy(sim)

    sim_copy.mpp([0, 2, 3], "XZY")

    assert np.allclose(
        sim_copy.symplectic_span(), 
        np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 1]
        ])
    )

    assert np.allclose(
        sim_copy.phases, 
        np.array([0, 0, 0, 0, 0, 0])
    )

    assert np.allclose(
        sim_copy.symbolic_phases, 
        np.array([
            [0],
            [0],
            [0],
            [0],
            [0],
            [1]
        ])
    )

    assert sim_copy.n_m == 1
    assert all([np.allclose(a, b) for a, b in zip(
        sim_copy.measurement_qubits,
        [np.array([0, 2, 3])]
        )])

def test_pauli_error_1():
    sim_copy = deepcopy(sim)

    sim_copy.pauli_error_1(0)

    assert all([np.allclose(a, b) for a, b in zip(
        sim_copy.error_indices_list,
        [np.array([2, 0, 0, 0]), np.array([12,  3,  0,  0]), np.array([12,  2,  3,  0]), np.array([12,  0,  2,  3]), np.array([2, 1, 0, 0, 1]), np.array([2, 1, 0, 0, 1])]
        )])

    assert sim_copy.list_error_channels == [1, 2, 2, 2, 1, 1]

def test_pauli_error_2():
    sim_copy = deepcopy(sim)

    sim_copy.pauli_error_2((0, 1))

    assert all([np.allclose(a, b) for a, b in zip(
        sim_copy.error_indices_list,
        [np.array([2, 0, 0, 0]), np.array([12,  3,  0,  0]), np.array([12,  2,  3,  0]), np.array([12,  0,  2,  3]), np.array([2, 1, 0, 0, 1]), np.array([12,  3,  1,  0,  2])]
        )])

    assert sim_copy.list_error_channels == [1, 2, 2, 2, 1, 2]

def test_get_error_coefficients():
    sim_copy = deepcopy(sim)

    # depolarising 1q: p(I)=1-3p, p(X)=p(Z)=p(Y)=p  =>  [1-3p, p, p, p]
    # depolarising 2q: p(II)=1-15p, rest = p  =>  [1-15p, p, p, ..., p]
    p1 = 0.01
    p2 = 0.005
    errors_1q = np.array([1 - 3 * p1, p1, p1, p1])
    errors_2q = np.array([1 - 15 * p2] + [p2] * 15)

    coeffs = sim_copy.get_error_coefficients(errors_1q, errors_2q)

    # should have one row per error channel
    assert coeffs.shape[0] == len(sim_copy.list_error_channels)
    assert coeffs.shape[1] == 16

    # for 1q channels (list_error_channels[i] == 1), only first 4 columns matter
    # FWHT of depolarising [1-3p, p, p, p] => [1, 1-4p, 1-4p, 1-4p]
    expected_1q_fwht_0 = 1.0
    expected_1q_fwht_nonzero = 1 - 4 * p1
    for i, chan in enumerate(sim_copy.list_error_channels):
        if chan == 1:
            assert np.isclose(coeffs[i, 0], expected_1q_fwht_0)
            # after symplectic permutation, indices 1,2,3 should all equal 1-4p
            assert np.isclose(coeffs[i, 1], expected_1q_fwht_nonzero)
            assert np.isclose(coeffs[i, 2], expected_1q_fwht_nonzero)
            assert np.isclose(coeffs[i, 3], expected_1q_fwht_nonzero)

    # for 2q channels (list_error_channels[i] == 2), FWHT of uniform depolarising
    # [1-15p, p, ..., p] => [1, 1-16p, 1-16p, ..., 1-16p]
    expected_2q_fwht_0 = 1.0
    expected_2q_fwht_nonzero = 1 - 16 * p2
    for i, chan in enumerate(sim_copy.list_error_channels):
        if chan == 2:
            assert np.isclose(coeffs[i, 0], expected_2q_fwht_0)
            for j in range(1, 16):
                assert np.isclose(coeffs[i, j], expected_2q_fwht_nonzero)

def test_get_error_coefficients_noiseless():
    """With no error probability, all coefficients should be identity-like."""
    sim_copy = deepcopy(sim)

    errors_1q = np.array([1.0, 0.0, 0.0, 0.0])
    errors_2q = np.array([1.0] + [0.0] * 15)

    coeffs = sim_copy.get_error_coefficients(errors_1q, errors_2q)

    # FWHT of [1, 0, 0, 0] = [1, 1, 1, 1]
    for i, chan in enumerate(sim_copy.list_error_channels):
        if chan == 1:
            for j in range(4):
                assert np.isclose(coeffs[i, j], 1.0)
        elif chan == 2:
            for j in range(16):
                assert np.isclose(coeffs[i, j], 1.0)

def test_get_error_coefficients_caching():
    """Calling get_error_coefficients twice with same rates should use cache."""
    sim_copy = deepcopy(sim)

    errors_1q = np.array([0.97, 0.01, 0.01, 0.01])
    errors_2q = np.array([0.925] + [0.005] * 15)

    coeffs1 = sim_copy.get_error_coefficients(errors_1q, errors_2q)
    coeffs2 = sim_copy.get_error_coefficients(errors_1q, errors_2q)

    assert np.allclose(coeffs1, coeffs2)
