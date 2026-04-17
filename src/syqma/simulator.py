"""Public simulator facade for measuring and evaluating symbolic circuits."""

import numpy as np
from rich.console import Console
from rich.traceback import install

from .evaluate_result import (
    _evaluate_expectation_value,
    _evaluate_expectation_value_taylor,
    _evaluate_expectation_values_from_scratch,
    _get_all_marginal_probabilities,
    _get_marginal_probabilities,
)
from .operations import Span
from .pauli import get_commutations
from .measure_result import _measure_all
from .result import SimulationResult

# Enables pretty tracebacks
install(width=1000, code_width=200, word_wrap=True)
console = Console(log_time=False, log_path=False)

ErrorRates = float | list[float] | np.ndarray


class Simulator(Span):
    """Simulate symbolic quantum circuits."""

    def __init__(
        self,
        n: int = None,
        gates: list = None,
        timing: bool = False,
    ) -> None:
        """Initialise the simulator.

        Args:
            n (int): The number of qubits in the circuit.
            gates (list): The list of gates to be applied to the circuit.
            timing (bool): Whether to time the execution of the circuit.

        """
        if n is None and gates is None:
            self.gates = []
            self.finished_circuit = False
        else:
            self.gates = gates

            super().__init__(n)

            self.all_solution_phases = []
            self.all_solution_symbolic_phases = []
            self.all_solution_c_operators = []
            self.all_solution_s_operators = []
            self.all_solution_o_operators = []
            self.all_solution_error_terms = []
            self.all_solution_weights = []
            self.exp_vals_baseline = None

            self.all_commutations_1q, self.all_commutations_2q = get_commutations()

            self.timing: bool = timing
            self.finished_circuit = True

            self.outputs_probs = None

    def measure_all(
        self,
        measured_qubits: np.ndarray | list[int],
        printing: bool = True,
        trace: bool = False,
    ) -> SimulationResult:
        """Measure the requested qubits and return symbolic observable data."""
        return _measure_all(self, measured_qubits, printing, trace)

    def evaluate_expectation_value(
        self,
        simulation_result: SimulationResult,
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        measurement_results: list[int] = None,
        thetas: list[float] = None,
        normalise_by_trace: bool = True,
        weight_cutoff: int = None,
        theta_indices: list[int] = None,
        printing: bool = False,
    ) -> float | np.ndarray:
        """Evaluate an expectation value from a precomputed measurement result."""
        return _evaluate_expectation_value(
            self,
            simulation_result,
            errors_1q,
            errors_2q,
            measurement_results,
            thetas,
            normalise_by_trace,
            weight_cutoff,
            theta_indices,
            printing,
        )

    def evaluate_expectation_value_taylor(
        self,
        simulation_result: SimulationResult,
        measurement_results: list[int] = None,
        thetas: list[float] = None,
        normalise_by_trace: bool = True,
        weight_cutoff: int = None,
        theta_indices: list[int] = None,
        p: float = 1e-3,
        printing: bool = False,
    ) -> None:
        """Print Taylor-series diagnostics for an expectation value."""
        return _evaluate_expectation_value_taylor(
            self,
            simulation_result,
            measurement_results,
            thetas,
            normalise_by_trace,
            weight_cutoff,
            theta_indices,
            p,
            printing,
        )

    def evaluate_expectation_values_from_scratch(
        self,
        measured_qubits: np.ndarray | list[int],
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        all_measurement_results: list[list[int]] = None,
        all_thetas: list[list[float]] = None,
        postselected_indices: list[int] = None,
        weight_cutoff: int = None,
        batch_size: int = 1,
        theta_indices: list[int] = None,
        normalise_by_trace: bool = True,
        printing: bool = True,
        n_jobs: int = -1,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Measure and evaluate expectation values for many inputs in one call."""
        return _evaluate_expectation_values_from_scratch(
            self,
            measured_qubits,
            errors_1q,
            errors_2q,
            all_measurement_results,
            all_thetas,
            postselected_indices,
            weight_cutoff,
            batch_size,
            theta_indices,
            normalise_by_trace,
            printing,
            n_jobs,
        )

    def get_marginal_probabilities(
        self,
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        measurement_results: list[int] = None,
        thetas: list[float] = None,
    ) -> float:
        """Return the probability of a partial measurement-result assignment."""
        return _get_marginal_probabilities(
            self, errors_1q, errors_2q, measurement_results, thetas
        )

    def get_all_marginal_probabilities(
        self,
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        thetas: list[float] = None,
        postselected_indices: list[int] = [],
        printing: bool = False,
    ) -> dict[tuple[int, ...], float]:
        """Return probabilities for all outcomes after fixed postselection."""
        return _get_all_marginal_probabilities(
            self, errors_1q, errors_2q, thetas, postselected_indices, printing
        )
