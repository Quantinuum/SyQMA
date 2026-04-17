import numpy as np
import pytest
from syqma.pauli import pauli_string_to_symplectic, commutation, generate_paulis, get_commutations


class TestPauliStringToSymplectic:
    def test_single_x(self):
        result = pauli_string_to_symplectic("X")
        assert np.array_equal(result, np.array([1, 0], dtype=np.uint8))

    def test_single_z(self):
        result = pauli_string_to_symplectic("Z")
        assert np.array_equal(result, np.array([0, 1], dtype=np.uint8))

    def test_single_y(self):
        result = pauli_string_to_symplectic("Y")
        assert np.array_equal(result, np.array([1, 1], dtype=np.uint8))

    def test_two_qubit_xz(self):
        # XZ => x part [1,0], z part [0,1] => [1, 0, 0, 1]
        result = pauli_string_to_symplectic("XZ")
        assert np.array_equal(result, np.array([1, 0, 0, 1], dtype=np.uint8))

    def test_two_qubit_yy(self):
        # YY => x part [1,1], z part [1,1] => [1, 1, 1, 1]
        result = pauli_string_to_symplectic("YY")
        assert np.array_equal(result, np.array([1, 1, 1, 1], dtype=np.uint8))

    def test_three_qubit(self):
        # XYZ => x part [1,1,0], z part [0,1,1] => [1, 1, 0, 0, 1, 1]
        result = pauli_string_to_symplectic("XYZ")
        assert np.array_equal(result, np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8))

    def test_invalid_character(self):
        with pytest.raises(ValueError, match="Invalid Pauli string"):
            pauli_string_to_symplectic("A")

    def test_identity_raises(self):
        # I is not handled, should raise
        with pytest.raises(ValueError, match="Invalid Pauli string"):
            pauli_string_to_symplectic("I")


class TestCommutation:
    def test_x_z_anticommute(self):
        assert commutation("X", "Z") == True

    def test_x_x_commute(self):
        assert commutation("X", "X") == False

    def test_z_z_commute(self):
        assert commutation("Z", "Z") == False

    def test_x_y_anticommute(self):
        assert commutation("X", "Y") == True

    def test_y_z_anticommute(self):
        assert commutation("Y", "Z") == True

    def test_y_y_commute(self):
        assert commutation("Y", "Y") == False

    def test_symmetry(self):
        # commutation(a, b) should equal commutation(b, a)
        for a in ["X", "Y", "Z"]:
            for b in ["X", "Y", "Z"]:
                assert commutation(a, b) == commutation(b, a)

    def test_two_qubit_commuting(self):
        # XX and ZZ commute (both anticommute on each qubit, but product commutes)
        assert commutation("XX", "ZZ") == False

    def test_two_qubit_anticommuting(self):
        # XZ and ZX anticommute
        assert commutation("XZ", "ZX") == False  # actually commutes: (X,Z) anticommute + (Z,X) anticommute = even = commute

    def test_array_input(self):
        # Test with array input instead of strings
        pauli_x = np.array([1, 0], dtype=np.uint8)
        pauli_z = np.array([0, 1], dtype=np.uint8)
        assert commutation(pauli_x, pauli_z) == True

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            commutation("X", "XX")


class TestGeneratePaulis:
    def test_1q_index_shape(self):
        idx_1q, rev_1q, idx_2q, rev_2q, mult_1q, mult_2q = generate_paulis()
        assert idx_1q.shape == (2, 2)
        assert rev_1q.shape == (4, 2)

    def test_2q_index_shape(self):
        idx_1q, rev_1q, idx_2q, rev_2q, mult_1q, mult_2q = generate_paulis()
        assert idx_2q.shape == (2, 2, 2, 2)
        assert rev_2q.shape == (16, 4)

    def test_1q_roundtrip(self):
        """Forward and reverse indices should be inverses."""
        idx_1q, rev_1q, *_ = generate_paulis()
        for i in range(4):
            xz = tuple(rev_1q[i])
            assert idx_1q[xz] == i

    def test_2q_roundtrip(self):
        idx_1q, rev_1q, idx_2q, rev_2q, *_ = generate_paulis()
        for i in range(16):
            xz = tuple(rev_2q[i])
            assert idx_2q[xz] == i

    def test_1q_multiplication_closure(self):
        """Product of any two 1q Paulis should still be a valid 1q Pauli index."""
        *_, mult_1q, mult_2q = generate_paulis()
        for (i, j), k in mult_1q.items():
            assert 0 <= k < 4

    def test_2q_multiplication_closure(self):
        *_, mult_1q, mult_2q = generate_paulis()
        for (i, j), k in mult_2q.items():
            assert 0 <= k < 16

    def test_1q_identity_multiplication(self):
        """Multiplying by identity (index 0) should be the identity operation."""
        *_, mult_1q, mult_2q = generate_paulis()
        for i in range(4):
            assert mult_1q[(0, i)] == i
            assert mult_1q[(i, 0)] == i


class TestGetCommutations:
    def test_1q_shape(self):
        comm_1q, comm_2q = get_commutations()
        assert comm_1q.shape == (4, 4)

    def test_2q_shape(self):
        comm_1q, comm_2q = get_commutations()
        assert comm_2q.shape == (16, 16)

    def test_1q_symmetric(self):
        comm_1q, _ = get_commutations()
        assert np.array_equal(comm_1q, comm_1q.T)

    def test_2q_symmetric(self):
        _, comm_2q = get_commutations()
        assert np.array_equal(comm_2q, comm_2q.T)

    def test_1q_identity_commutes_with_all(self):
        comm_1q, _ = get_commutations()
        # index 0 is identity (I), should commute with everything
        assert np.all(comm_1q[0] == 1)
        assert np.all(comm_1q[:, 0] == 1)

    def test_1q_matches_commutation_function(self):
        """Verify hardcoded table matches the commutation() function."""
        comm_1q, _ = get_commutations()
        # Pauli ordering: I(00), Z(01), X(10), Y(11)
        pauli_labels = ["X", "Z", "Y"]  # skip I since commutation() doesn't handle I
        _, rev_1q, *_ = generate_paulis()

        # For non-identity pairs, verify consistency
        for i in range(1, 4):
            for j in range(1, 4):
                vec_i = rev_1q[i]
                vec_j = rev_1q[j]
                # commutation returns True if anticommuting
                anticommutes = commutation(vec_i, vec_j)
                # comm_1q: 1 = commute, 0 = anticommute
                if anticommutes:
                    assert comm_1q[i, j] == 0
                else:
                    assert comm_1q[i, j] == 1
