import numpy as np
from syqma.result import MeasurementResult, SimulationResult


def _make_simple_measurement_result(n_terms=4, n_errors=2, n_magic=0, n_measurements=1):
    """Create a MeasurementResult with known simple data."""
    phases = np.zeros(n_terms, dtype=np.uint8)
    phases[1] = 1  # second term has negative phase

    symbolic_phases = np.zeros((n_terms, n_measurements), dtype=np.uint8)
    symbolic_phases[2, 0] = 1  # third term depends on measurement 0

    error_terms = np.zeros((n_terms, n_errors), dtype=np.uint8)
    error_terms[0, 0] = 1  # first term has error on channel 0

    c_operators = np.zeros((n_terms, max(n_magic, 1)), dtype=np.uint8)
    s_operators = np.zeros((n_terms, max(n_magic, 1)), dtype=np.uint8)
    o_operators = np.zeros((n_terms, max(n_magic, 1)), dtype=np.uint8)
    weights = np.ones(n_terms, dtype=np.float64)
    theta_indices = np.zeros(max(n_magic, 1), dtype=np.uint8)

    list_error_channels = [1, 2]  # one 1q, one 2q channel
    error_indices_list = [
        np.array([2, 0], dtype=int),
        np.array([12, 3], dtype=int),
    ]

    return MeasurementResult(
        phases, symbolic_phases, error_terms,
        c_operators, s_operators, o_operators,
        weights, theta_indices,
        list_error_channels, error_indices_list,
    )


class TestMeasurementResult:
    def test_init_prod_phases(self):
        mr = _make_simple_measurement_result()
        # prod_phases = 1 - 2 * phases => [1, -1, 1, 1]
        expected = np.array([1, -1, 1, 1], dtype=np.int8)
        assert np.array_equal(mr.prod_phases, expected)

    def test_init_n_error_channels(self):
        mr = _make_simple_measurement_result()
        assert mr.n_error_channels == 4  # len(error_terms)

    def test_set_measurement_results(self):
        mr = _make_simple_measurement_result()
        mr.set_measurement_results(np.array([1]))

        # symbolic_phases @ [1] => [0, 0, 1, 0]
        # prod_symbolic_phases = 1 - 2*(... % 2) => [1, 1, -1, 1]
        expected = np.array([1, 1, -1, 1])
        assert np.array_equal(mr.prod_symbolic_phases, expected)

    def test_set_measurement_results_zero(self):
        mr = _make_simple_measurement_result()
        mr.set_measurement_results(np.array([0]))

        # All symbolic phases evaluated to 0 => all prod = 1
        expected = np.array([1, 1, 1, 1])
        assert np.array_equal(mr.prod_symbolic_phases, expected)


class TestSimulationResult:
    def test_init(self):
        obs = _make_simple_measurement_result()
        trace = _make_simple_measurement_result()
        sr = SimulationResult(observable_result=obs, trace_result=trace)

        assert sr.observable_result is obs
        assert sr.trace_result is trace

    def test_set_measurement_results_propagates(self):
        obs = _make_simple_measurement_result()
        trace = _make_simple_measurement_result()
        sr = SimulationResult(observable_result=obs, trace_result=trace)

        sr.set_measurement_results(np.array([1]))

        assert obs.prod_symbolic_phases is not None
        assert trace.prod_symbolic_phases is not None
        assert np.array_equal(obs.prod_symbolic_phases, trace.prod_symbolic_phases)

    def test_save_load_roundtrip(self, tmp_path):
        obs = _make_simple_measurement_result()
        trace = _make_simple_measurement_result()
        sr = SimulationResult(observable_result=obs, trace_result=trace)

        filepath = str(tmp_path / "test_result")
        sr.save_to_file(filepath)

        loaded = SimulationResult.load_from_file(filepath + ".zstd")

        assert np.array_equal(loaded.observable_result.phases, obs.phases)
        assert np.array_equal(loaded.trace_result.phases, trace.phases)
        assert np.array_equal(loaded.observable_result.prod_phases, obs.prod_phases)
        assert np.array_equal(
            loaded.observable_result.symbolic_phases,
            obs.symbolic_phases,
        )
