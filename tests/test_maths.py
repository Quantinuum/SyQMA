import numpy as np
from syqma.gf2 import gf2_kernel, gf2_particular_solution, solve_gf2_system
from syqma.maths import solve_gf2_only, solve_gf2_original, num2vec, kernel_enumerator


def _collect_solutions(gen):
    """Collect all solutions from a generator into a list of numpy arrays."""
    return [np.array(x, dtype=np.uint8) for x in gen]


def _solution_set(x_part, K):
    if x_part.size == 0:
        return set()
    return {tuple(x) for x in kernel_enumerator(x_part, K)}


def _assert_valid_solution_family(A, b, x_part, K):
    A = np.asarray(A, dtype=np.uint8) & 1
    b = np.asarray(b, dtype=np.uint8) & 1
    assert x_part.dtype == np.uint8
    assert K.dtype == np.uint8
    assert np.array_equal((A @ x_part) % 2, b)
    if K.size:
        assert np.array_equal(
            (A @ K.T) % 2,
            np.zeros((A.shape[0], K.shape[0]), dtype=np.uint8),
        )


class TestNum2Vec:
    def test_zero(self):
        assert num2vec(0, 4) == [0, 0, 0, 0]

    def test_all_ones(self):
        assert num2vec(0b1111, 4) == [1, 1, 1, 1]

    def test_msb_first(self):
        # 0b1010 = 10, width 4 => [1, 0, 1, 0]
        assert num2vec(0b1010, 4) == [1, 0, 1, 0]

    def test_width_1(self):
        assert num2vec(0, 1) == [0]
        assert num2vec(1, 1) == [1]

    def test_wider_than_needed(self):
        # 0b11 = 3, width 8 => [0, 0, 0, 0, 0, 0, 1, 1]
        assert num2vec(3, 8) == [0, 0, 0, 0, 0, 0, 1, 1]


class TestSolveGF2Only:
    def test_identity_system(self):
        """Ax = b with A = I should give x = b, kernel empty."""
        A = np.eye(3, dtype=np.uint8)
        b = np.array([1, 0, 1], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        assert np.array_equal((A @ x_part) % 2, b)
        assert K.shape[0] == 0  # no kernel

    def test_no_solution(self):
        """Inconsistent system should return empty arrays."""
        A = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        b = np.array([0, 1], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        assert x_part.size == 0
        assert K.size == 0

    def test_underdetermined_system(self):
        """System with kernel dimension > 0."""
        # A = [[1, 1, 0], [0, 1, 1]], b = [1, 0]
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        assert np.array_equal((A @ x_part) % 2, b)
        assert K.shape[0] >= 1  # at least one kernel vector
        # kernel vectors should satisfy A @ k = 0
        for i in range(K.shape[0]):
            assert np.array_equal((A @ K[i]) % 2, np.zeros(A.shape[0], dtype=np.uint8))

    def test_full_rank_square(self):
        """Full rank square system has unique solution, empty kernel."""
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        b = np.array([0, 1], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        assert np.array_equal((A @ x_part) % 2, b)
        assert K.shape[0] == 0

    def test_larger_system(self):
        """4x6 system with nontrivial kernel."""
        A = np.array([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ], dtype=np.uint8)
        b = np.array([1, 1, 0, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        if x_part.size > 0:
            assert np.array_equal((A @ x_part) % 2, b)
            for i in range(K.shape[0]):
                assert np.array_equal((A @ K[i]) % 2, np.zeros(4, dtype=np.uint8))

    def test_zero_row_consistent_system(self):
        A = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        _assert_valid_solution_family(A, b, x_part, K)
        assert K.shape == (2, 3)

    def test_zero_row_inconsistent_system(self):
        A = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)
        b = np.array([1, 1], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        assert x_part.size == 0
        assert K.size == 0

    def test_duplicate_rows(self):
        A = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        _assert_valid_solution_family(A, b, x_part, K)
        assert K.shape == (1, 3)

    def test_tall_rectangular_system(self):
        A = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        _assert_valid_solution_family(A, b, x_part, K)

    def test_wide_rectangular_system(self):
        A = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        _assert_valid_solution_family(A, b, x_part, K)
        assert K.shape == (3, 5)

    def test_non_binary_inputs_are_reduced_mod2(self):
        A = np.array([[3, 2, 1], [4, 5, 7]], dtype=np.uint8)
        b = np.array([3, 4], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        _assert_valid_solution_family(A, b, x_part, K)


class TestKernelEnumerator:
    def test_trivial_kernel(self):
        """Empty kernel (d=0) should yield just the particular solution."""
        x_part = np.array([1, 0, 1], dtype=np.uint8)
        K = np.zeros((0, 3), dtype=np.uint8)
        sols = _collect_solutions(kernel_enumerator(x_part, K))

        assert len(sols) == 1
        assert np.array_equal(sols[0], x_part)

    def test_single_kernel_vector(self):
        """Kernel of dimension 1 should yield 2 solutions."""
        x_part = np.array([1, 0, 0], dtype=np.uint8)
        K = np.array([[0, 1, 1]], dtype=np.uint8)
        sols = _collect_solutions(kernel_enumerator(x_part, K))

        assert len(sols) == 2
        # both should differ from x_part by K[0] or not at all
        sol_set = {tuple(s) for s in sols}
        assert tuple(x_part) in sol_set
        assert tuple((x_part ^ K[0]) % 2) in sol_set

    def test_two_kernel_vectors(self):
        """Kernel of dimension 2 should yield 4 solutions."""
        x_part = np.array([0, 0, 0, 0], dtype=np.uint8)
        K = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        sols = _collect_solutions(kernel_enumerator(x_part, K))

        assert len(sols) == 4
        # all solutions should be distinct
        sol_tuples = [tuple(s) for s in sols]
        assert len(set(sol_tuples)) == 4

    def test_gray_code_property(self):
        """Consecutive solutions should differ by exactly one kernel vector XOR."""
        x_part = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
        K = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ], dtype=np.uint8)
        sols = _collect_solutions(kernel_enumerator(x_part, K))

        assert len(sols) == 8
        # consecutive solutions should differ by exactly one kernel row
        K_rows = {tuple(K[i]) for i in range(K.shape[0])}
        for i in range(1, len(sols)):
            diff = (sols[i] ^ sols[i - 1])
            assert tuple(diff) in K_rows

    def test_all_solutions_satisfy_system(self):
        """Enumerated solutions should all satisfy the original linear system."""
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_only(A, b)

        if x_part.size > 0:
            sols = _collect_solutions(kernel_enumerator(x_part, K))
            for sol in sols:
                assert np.array_equal((A @ sol) % 2, b)


class TestSolveGF2Original:
    def test_identity(self):
        A = np.eye(3, dtype=np.uint8)
        b = np.array([1, 0, 1], dtype=np.uint8)
        sols = list(solve_gf2_original(A, b))

        assert len(sols) == 1
        assert np.array_equal(np.array(sols[0], dtype=np.uint8), b)

    def test_no_solution(self):
        A = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        b = np.array([0, 1], dtype=np.uint8)
        sols = list(solve_gf2_original(A, b))

        assert len(sols) == 0

    def test_underdetermined(self):
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        sols = list(solve_gf2_original(A, b))

        assert len(sols) >= 2
        for sol in sols:
            x = np.array(sol, dtype=np.uint8)
            assert np.array_equal((A @ x) % 2, b)

    def test_matches_solve_gf2_only(self):
        """Cross-validate: both solvers should produce the same solution set."""
        A = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 1, 0], dtype=np.uint8)

        # solve_gf2_original
        sols_original = [tuple(s) for s in solve_gf2_original(A, b)]

        # solve_gf2_only + kernel_enumerator
        x_part, K = solve_gf2_only(A, b)
        if x_part.size > 0:
            sols_kernel = [tuple(s) for s in kernel_enumerator(x_part, K)]
        else:
            sols_kernel = []

        assert set(sols_original) == set(sols_kernel)

    def test_random_small_systems_match_solution_set(self):
        rng = np.random.default_rng(123)

        for rows in range(1, 6):
            for cols in range(1, 7):
                for _ in range(12):
                    A = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
                    b = rng.integers(0, 2, size=rows, dtype=np.uint8)

                    sols_original = {tuple(s) for s in solve_gf2_original(A, b)}
                    x_part, K = solve_gf2_only(A, b)
                    assert _solution_set(x_part, K) == sols_original


class TestGF2Helpers:
    def test_solve_gf2_system(self):
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        b = np.array([1, 0], dtype=np.uint8)
        x_part, K = solve_gf2_system(A, b)

        _assert_valid_solution_family(A, b, x_part, K)

    def test_gf2_particular_solution(self):
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        b = np.array([0, 1], dtype=np.uint8)
        x_part = gf2_particular_solution(A, b)

        assert np.array_equal((A @ x_part) % 2, b)

    def test_gf2_kernel(self):
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        K = gf2_kernel(A)

        assert K.shape == (1, 3)
        assert np.array_equal((A @ K.T) % 2, np.zeros((2, 1), dtype=np.uint8))
