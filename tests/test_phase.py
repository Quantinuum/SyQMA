import numpy as np
from syqma.phase import (
    _overall_phase_from_ops_mod4,
    _compute_batch_non_linear_phases,
)


class TestOverallPhaseFromOpsMod4:
    def test_empty(self):
        ops_1 = np.zeros((0, 2), dtype=np.uint8)
        ops_2 = np.zeros((0, 2), dtype=np.uint8)
        result = _overall_phase_from_ops_mod4(ops_1, ops_2)
        assert result == 1.0 + 0.0j

    def test_single_identity(self):
        ops_1 = np.zeros((1, 2), dtype=np.uint8)
        ops_2 = np.zeros((1, 2), dtype=np.uint8)
        result = _overall_phase_from_ops_mod4(ops_1, ops_2)
        assert result == 1.0 + 0.0j

    def test_single_pauli(self):
        # Single X operator: f_0 = (2*0 + 0 + 0 - 0) = 0
        ops_1 = np.array([[1, 0]], dtype=np.uint8)
        ops_2 = np.array([[0, 0]], dtype=np.uint8)
        result = _overall_phase_from_ops_mod4(ops_1, ops_2)
        assert result == 1.0 + 0.0j

    def test_returns_valid_power_of_minus_i(self):
        """Result should always be one of {1, -i, -1, i}."""
        valid = {(1, 0), (0, -1), (-1, 0), (0, 1)}
        rng = np.random.default_rng(42)
        for _ in range(20):
            m = rng.integers(1, 6)
            n = rng.integers(1, 4)
            ops_1 = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
            ops_2 = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
            result = _overall_phase_from_ops_mod4(ops_1, ops_2)
            assert (round(result.real), round(result.imag)) in valid

    def test_x_then_z_single_qubit(self):
        # Product X * Z = -iY
        # X: ops_1 = [1], ops_2 = [0]
        # Z: ops_1 = [0], ops_2 = [1]
        ops_1 = np.array([[1], [0]], dtype=np.uint8)
        ops_2 = np.array([[0], [1]], dtype=np.uint8)
        result = _overall_phase_from_ops_mod4(ops_1, ops_2)
        # f_0 (X): prefix=(0,0), s1=0, s2=0, s3=0, s4=1*0=0 => f_0=0
        # f_1 (Z): prefix=(1,0), s1=1*1=1, s2=0*1=0, s3=1*0=0, s4=(1^0)&(0^1)=1 => f_1 = (2+0+0-1)%4 = 1
        # F = 1 => (-1j)^1 = -1j
        assert np.isclose(result, -1j)


class TestComputeBatchNonLinearPhases:
    def test_single_solution_all_zeros(self):
        """Solution that selects no rows should give phase 0 (no flip)."""
        m = 3
        n_q = 2
        n_m = 1
        solutions = np.zeros((1, m), dtype=np.uint8)
        span_x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        span_z = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.uint8)
        c_ops = np.array([[1], [0], [1]], dtype=np.uint8)
        o_ops = np.array([[0], [1], [0]], dtype=np.uint8)

        phases = _compute_batch_non_linear_phases(solutions, span_x, span_z, c_ops, o_ops)
        assert phases[0] == 0  # no flip

    def test_batch_output_shape(self):
        batch = 5
        m = 4
        solutions = np.zeros((batch, m), dtype=np.uint8)
        span_x = np.zeros((m, 3), dtype=np.uint8)
        span_z = np.zeros((m, 3), dtype=np.uint8)
        c_ops = np.zeros((m, 2), dtype=np.uint8)
        o_ops = np.zeros((m, 2), dtype=np.uint8)

        phases = _compute_batch_non_linear_phases(solutions, span_x, span_z, c_ops, o_ops)
        assert phases.shape == (batch,)
        assert phases.dtype == np.uint8

    def test_phase_is_binary(self):
        """Phase values should only be 0 or 1."""
        rng = np.random.default_rng(77)
        m = 5
        batch = 10
        solutions = rng.integers(0, 2, size=(batch, m), dtype=np.uint8)
        span_x = rng.integers(0, 2, size=(m, 3), dtype=np.uint8)
        span_z = rng.integers(0, 2, size=(m, 3), dtype=np.uint8)
        c_ops = rng.integers(0, 2, size=(m, 2), dtype=np.uint8)
        o_ops = rng.integers(0, 2, size=(m, 2), dtype=np.uint8)

        phases = _compute_batch_non_linear_phases(solutions, span_x, span_z, c_ops, o_ops)
        assert np.all((phases == 0) | (phases == 1))

    def test_no_magic_gates(self):
        """With empty c/o operators (0 columns), phase should come only from XZ."""
        m = 3
        solutions = np.array([[1, 1, 0]], dtype=np.uint8)
        span_x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        span_z = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.uint8)
        c_ops = np.zeros((m, 0), dtype=np.uint8)
        o_ops = np.zeros((m, 0), dtype=np.uint8)

        phases = _compute_batch_non_linear_phases(solutions, span_x, span_z, c_ops, o_ops)
        # With all Z=0, all s1=s3=s4=s2=0, so F_p=0 => phase=0
        assert phases[0] == 0
