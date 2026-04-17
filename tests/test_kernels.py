import numpy as np
from syqma.kernels import dfs_int, dfs_int_generator, dfs_batch_generator, process_o_operator, process_cso_operators


class TestProcessOOperator:
    def test_basic(self):
        c_ops = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        o_ops = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        solution = np.array([1, 1], dtype=np.uint8)

        o_op = process_o_operator(solution, c_ops, o_ops)

        assert np.array_equal(o_op, np.array([0, 0, 0], dtype=np.uint8))

    def test_no_active_rows(self):
        c_ops = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        o_ops = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        solution = np.array([0, 0], dtype=np.uint8)

        o_op = process_o_operator(solution, c_ops, o_ops)
        assert np.array_equal(o_op, np.array([0, 0], dtype=np.uint8))

    def test_single_row(self):
        c_ops = np.array([[1, 0, 1]], dtype=np.uint8)
        o_ops = np.array([[0, 1, 1]], dtype=np.uint8)
        solution = np.array([1], dtype=np.uint8)

        o_op = process_o_operator(solution, c_ops, o_ops)
        assert np.array_equal(o_op, np.array([0, 1, 0], dtype=np.uint8))


class TestProcessCSOOperators:
    def test_mutual_exclusivity(self):
        """After decomposition, c & o should be zero (S is extracted)."""
        c_ops = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        o_ops = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        solution = np.array([1, 1], dtype=np.uint8)

        c_op, s_op, o_op = process_cso_operators(solution, c_ops, o_ops)

        # c and o should not overlap after S extraction
        assert np.array_equal(c_op & o_op, np.zeros_like(c_op))
        # s should be the intersection
        assert np.array_equal(c_op & s_op, np.zeros_like(c_op))
        assert np.array_equal(o_op & s_op, np.zeros_like(o_op))

    def test_reconstructs_original(self):
        """c XOR s and o XOR s should reconstruct the original accumulated values."""
        c_ops = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
        o_ops = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
        solution = np.array([1, 1], dtype=np.uint8)

        c_op, s_op, o_op = process_cso_operators(solution, c_ops, o_ops)

        # The raw XOR of selected rows
        raw_c = c_ops[0] ^ c_ops[1]
        raw_o = o_ops[0] ^ o_ops[1]

        # c_op ^ s_op should equal raw_c, o_op ^ s_op should equal raw_o
        assert np.array_equal(c_op ^ s_op, raw_c)
        assert np.array_equal(o_op ^ s_op, raw_o)


class TestDFS:
    def _make_simple_problem(self):
        """Create a small problem where we know the valid solutions.

        We set up delta_c and delta_o such that starting from (c=0, o=0),
        flipping bits in y can lead to valid solutions (where o & ~c == 0).
        """
        # 3 bits, 2 kernel vectors
        # delta_c[0] = 0b101 = 5, delta_o[0] = 0b001 = 1  => flipping y[0]: c^=5, o^=1
        # delta_c[1] = 0b010 = 2, delta_o[1] = 0b010 = 2  => flipping y[1]: c^=2, o^=2
        delta_c_int = [5, 2]
        delta_o_int = [1, 2]
        # salvageables: OR of all future deltas for c bits
        # salv[0] = delta_c[0] | delta_c[1] = 7
        # salv[1] = delta_c[1] = 2
        salv_int = [7, 2]
        return delta_c_int, delta_o_int, salv_int

    def test_dfs_int_finds_solutions(self):
        delta_c, delta_o, salv = self._make_simple_problem()
        out = []
        y = np.zeros(2, dtype=np.uint8)
        dfs_int(0, 0, 0, y, out, delta_c, delta_o, salv)

        # y=[0,0]: c=0, o=0 => bad=0 => valid
        # y=[1,0]: c=5, o=1 => bad = 1 & ~5 = 0 => valid
        # y=[0,1]: c=2, o=2 => bad = 2 & ~2 = 0 => valid
        # y=[1,1]: c=7, o=3 => bad = 3 & ~7 = 0 => valid
        assert len(out) == 4

    def test_dfs_batch_generator_matches_recursive(self):
        delta_c, delta_o, salv = self._make_simple_problem()

        out_rec = []
        y = np.zeros(2, dtype=np.uint8)
        dfs_int(0, 0, 0, y, out_rec, delta_c, delta_o, salv)

        gen = dfs_batch_generator(0, 0, 0, delta_c, delta_o, salv, batch_size=2)
        batch_results = []
        for batch_c, batch_o, batch_y in gen:
            for i in range(len(batch_c)):
                batch_results.append((batch_c[i], batch_o[i], batch_y[i]))

        rec_set = {(c, o, tuple(yy)) for c, o, yy in out_rec}
        batch_set = set()
        for c, o, y_packed in batch_results:
            y_unpacked = tuple((y_packed >> i) & 1 for i in range(2))
            batch_set.add((c, o, y_unpacked))
            
        assert rec_set == batch_set

    def test_dfs_int_generator_matches_recursive(self):
        delta_c, delta_o, salv = self._make_simple_problem()

        out_rec = []
        y = np.zeros(2, dtype=np.uint8)
        dfs_int(0, 0, 0, y, out_rec, delta_c, delta_o, salv)

        y_int = 0
        out_gen = list(dfs_int_generator(0, 0, 0, y_int, delta_c, delta_o, salv))

        rec_set = {(c, o, tuple(yy)) for c, o, yy in out_rec}
        # Expand out_gen y_int back to tuple(yy)
        gen_set = set()
        for c, o, y_packed in out_gen:
            y_unpacked = tuple((y_packed >> i) & 1 for i in range(2))
            gen_set.add((c, o, y_unpacked))
            
        assert rec_set == gen_set

    def test_pruning(self):
        """When o has bits that c can never cover, those branches get pruned."""
        # delta_c[0] = 0b01 = 1, delta_o[0] = 0b10 = 2
        # Starting from c=0, o=0:
        #   y[0]=0: c=0, o=0 => valid
        #   y[0]=1: c=1, o=2 => bad = 2 & ~1 = 2 != 0 => pruned at leaf
        delta_c_int = [1]
        delta_o_int = [2]
        salv_int = [1]

        out = []
        y = np.zeros(1, dtype=np.uint8)
        dfs_int(0, 0, 0, y, out, delta_c_int, delta_o_int, salv_int)

        # only y=[0] is valid
        assert len(out) == 1
        assert tuple(out[0][2]) == (0,)

    def test_all_solutions_valid(self):
        """Every returned solution should satisfy o & ~c == 0."""
        delta_c_int = [0b1010, 0b0101, 0b1111]
        delta_o_int = [0b0010, 0b0100, 0b0001]
        salv_int = [
            delta_c_int[0] | delta_c_int[1] | delta_c_int[2],
            delta_c_int[1] | delta_c_int[2],
            delta_c_int[2],
        ]

        out = []
        y = np.zeros(3, dtype=np.uint8)
        dfs_int(0, 0, 0, y, out, delta_c_int, delta_o_int, salv_int)

        for c, o, yy in out:
            assert (o & (~c)) == 0
